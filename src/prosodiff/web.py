"""Local-only browser interface for Prosodiff."""

from __future__ import annotations

import atexit
import hmac
import secrets
import shutil
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO

import soundfile as sf
from flask import (
    Flask,
    abort,
    current_app,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import RequestEntityTooLarge

from prosodiff.analysis import compare_takes
from prosodiff.errors import ProsodiffError
from prosodiff.models import Comparison
from prosodiff.render import render_comparison
from prosodiff.report import write_json_report


_MAX_FILES = 4
_MIN_FILES = 2
_MAX_FILE_BYTES = 24 * 1024 * 1024
_MAX_LABEL_LENGTH = 64
_MAX_TEXT_LENGTH = 300
_MAX_RESULTS = 10
_READ_CHUNK_BYTES = 1024 * 1024
_ANALYSIS_LOCK = threading.Lock()


@dataclass(frozen=True)
class ResultRecord:
    """One completed comparison retained for the life of the local server."""

    run_id: str
    directory: Path
    comparison: Comparison

    @property
    def image_path(self) -> Path:
        return self.directory / "prosodiff-card.png"

    @property
    def json_path(self) -> Path:
        return self.directory / "prosodiff-card.json"


class ResultStore:
    """Own short-lived result files without retaining uploaded recordings."""

    def __init__(self, root: Path | None = None) -> None:
        self._temporary = (
            TemporaryDirectory(prefix="prosodiff-ui-") if root is None else None
        )
        self.root = (
            Path(self._temporary.name) if self._temporary is not None else root.resolve()
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self._records: OrderedDict[str, ResultRecord] = OrderedDict()
        self._lock = threading.Lock()
        self._closed = False

    def reserve(self) -> tuple[str, Path]:
        """Create an unguessable, request-private working directory."""

        with self._lock:
            if self._closed:
                raise RuntimeError("The result store is closed.")
            while True:
                run_id = secrets.token_urlsafe(18)
                directory = self.root / run_id
                try:
                    directory.mkdir()
                except FileExistsError:
                    continue
                return run_id, directory

    def register(self, record: ResultRecord) -> None:
        with self._lock:
            self._records[record.run_id] = record
            while len(self._records) > _MAX_RESULTS:
                _, expired = self._records.popitem(last=False)
                shutil.rmtree(expired.directory, ignore_errors=True)

    def get(self, run_id: str) -> ResultRecord | None:
        with self._lock:
            return self._records.get(run_id)

    def discard(self, run_id: str, directory: Path) -> None:
        with self._lock:
            self._records.pop(run_id, None)
        shutil.rmtree(directory, ignore_errors=True)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._records.clear()
        if self._temporary is not None:
            self._temporary.cleanup()


def _result_store() -> ResultStore:
    return current_app.extensions["prosodiff_results"]


def _default_label(filename: str, index: int) -> str:
    stem = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
    return stem.title()[:_MAX_LABEL_LENGTH] or f"Take {index}"


def _contains_control_characters(value: str) -> bool:
    return any(ord(character) < 32 and character not in "\t\n\r" for character in value)


def _clean_labels(files: list[FileStorage], raw_labels: list[str]) -> list[str]:
    if raw_labels and len(raw_labels) != len(files):
        raise ValueError("Choose the recordings again so every take has one label.")
    labels = [
        (raw_labels[index - 1].strip() if raw_labels else "")
        or _default_label(file.filename or "", index)
        for index, file in enumerate(files, start=1)
    ]
    for index, label in enumerate(labels, start=1):
        if len(label) > _MAX_LABEL_LENGTH:
            raise ValueError(f"Take {index} label must be {_MAX_LABEL_LENGTH} characters or fewer.")
        if _contains_control_characters(label):
            raise ValueError(f"Take {index} label contains unsupported control characters.")
    if len({label.casefold() for label in labels}) != len(labels):
        raise ValueError("Take labels must be unique.")
    return labels


def _copy_limited(source: BinaryIO, destination: Path) -> None:
    total = 0
    with destination.open("wb") as output:
        while chunk := source.read(_READ_CHUNK_BYTES):
            total += len(chunk)
            if total > _MAX_FILE_BYTES:
                raise ValueError("Each WAV must be 24 MB or smaller.")
            output.write(chunk)


def _save_wavs(files: list[FileStorage], directory: Path) -> list[Path]:
    paths: list[Path] = []
    for index, upload in enumerate(files, start=1):
        filename = upload.filename or ""
        if not filename:
            raise ValueError(f"Take {index} has no selected file.")
        if Path(filename).suffix.casefold() != ".wav":
            raise ValueError(f"Take {index} must be a WAV file.")
        destination = directory / f"take_{index}.wav"
        _copy_limited(upload.stream, destination)
        try:
            info = sf.info(destination)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Take {index} is not a readable WAV file.") from exc
        if info.format not in {"WAV", "WAVEX"}:
            raise ValueError(f"Take {index} is not a WAV container.")
        paths.append(destination)
    return paths


def _delete_uploads(paths: list[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _deduplicated_warnings(comparison: Comparison) -> list[str]:
    return list(dict.fromkeys(comparison.warnings))


def _index_response(error: str | None = None, *, status: int = 200):
    return (
        render_template(
            "index.html",
            csrf_token=current_app.config["PROSODIFF_CSRF_TOKEN"],
            error=error,
        ),
        status,
    )


def create_app(
    *,
    result_root: Path | None = None,
    csrf_token: str | None = None,
) -> Flask:
    """Create the loopback UI application."""

    app = Flask(__name__)
    app.config.update(
        MAX_CONTENT_LENGTH=64 * 1024 * 1024,
        MAX_FORM_MEMORY_SIZE=256 * 1024,
        MAX_FORM_PARTS=24,
        TRUSTED_HOSTS=["localhost", "127.0.0.1", "[::1]"],
        PROSODIFF_CSRF_TOKEN=csrf_token or secrets.token_urlsafe(32),
    )
    store = ResultStore(result_root)
    app.extensions["prosodiff_results"] = store
    atexit.register(store.close)

    @app.after_request
    def secure_local_response(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self'; media-src 'self' blob:; "
            "style-src 'self'; script-src 'self'; object-src 'none'; base-uri 'none'; "
            "form-action 'self'; frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(self), geolocation=(), payment=()"
        )
        return response

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(_error):
        return _index_response(
            "The upload is too large. Use up to four WAV files, 24 MB each and 64 MB total.",
            status=413,
        )

    @app.get("/")
    def index():
        return _index_response()

    @app.post("/compare")
    def compare():
        submitted_token = request.form.get("csrf_token", "")
        expected_token = current_app.config["PROSODIFF_CSRF_TOKEN"]
        if not hmac.compare_digest(submitted_token, expected_token):
            return _index_response("The form expired. Reload this page and try again.", status=403)
        origin = request.headers.get("Origin")
        # ``Referrer-Policy: no-referrer`` can make Chromium serialize a normal
        # same-origin form POST as ``Origin: null``. The unguessable form token
        # remains the primary CSRF boundary in that case.
        if origin not in {None, "null"} and origin.rstrip("/") != request.host_url.rstrip("/"):
            return _index_response("The request origin was not accepted.", status=403)

        files = [upload for upload in request.files.getlist("wavs") if upload.filename]
        if not _MIN_FILES <= len(files) <= _MAX_FILES:
            return _index_response("Select between two and four WAV recordings.", status=400)
        text = request.form.get("text", "").strip()
        if len(text) > _MAX_TEXT_LENGTH:
            return _index_response(
                f"The matched sentence must be {_MAX_TEXT_LENGTH} characters or fewer.",
                status=400,
            )
        if _contains_control_characters(text):
            return _index_response(
                "The matched sentence contains unsupported control characters.",
                status=400,
            )
        if request.form.get("protocol_ack") != "yes":
            return _index_response(
                "Confirm the matched recording protocol before generating a comparison.",
                status=400,
            )
        try:
            labels = _clean_labels(files, request.form.getlist("labels"))
        except ValueError as exc:
            return _index_response(str(exc), status=400)

        run_id, directory = _result_store().reserve()
        wav_paths: list[Path] = []
        try:
            wav_paths = _save_wavs(files, directory)
            with _ANALYSIS_LOCK:
                comparison = compare_takes(
                    wav_paths,
                    labels=labels,
                    text=text or None,
                    synthetic_demo=False,
                )
                image_path = render_comparison(comparison, directory / "prosodiff-card.png")
                json_path = write_json_report(comparison, directory / "prosodiff-card.json")
            if not image_path.is_file() or not json_path.is_file():
                raise RuntimeError("Prosodiff did not produce both result files.")
            record = ResultRecord(run_id=run_id, directory=directory, comparison=comparison)
            _result_store().register(record)
        except ValueError as exc:
            _result_store().discard(run_id, directory)
            return _index_response(str(exc), status=400)
        except ProsodiffError as exc:
            _result_store().discard(run_id, directory)
            return _index_response(str(exc), status=422)
        except Exception:
            current_app.logger.exception("Prosodiff comparison failed for run %s", run_id)
            _result_store().discard(run_id, directory)
            return _index_response(
                "The comparison could not be generated. Your files remained local "
                "and were deleted. Try again or use the CLI for diagnostics.",
                status=500,
            )
        finally:
            _delete_uploads(wav_paths)
        return redirect(url_for("result", run_id=run_id), code=303)

    @app.get("/results/<run_id>")
    def result(run_id: str):
        record = _result_store().get(run_id)
        if record is None:
            abort(404)
        labels = {take.id: take.label for take in record.comparison.takes}
        return render_template(
            "result.html",
            record=record,
            labels=labels,
            warnings=_deduplicated_warnings(record.comparison),
        )

    @app.get("/results/<run_id>/card.png")
    def result_image(run_id: str):
        record = _result_store().get(run_id)
        if record is None:
            abort(404)
        return send_file(record.image_path, mimetype="image/png", conditional=True)

    @app.get("/results/<run_id>/download/card.png")
    def download_image(run_id: str):
        record = _result_store().get(run_id)
        if record is None:
            abort(404)
        return send_file(
            record.image_path,
            mimetype="image/png",
            as_attachment=True,
            download_name="prosodiff-card.png",
        )

    @app.get("/results/<run_id>/download/card.json")
    def download_json(run_id: str):
        record = _result_store().get(run_id)
        if record is None:
            abort(404)
        return send_file(
            record.json_path,
            mimetype="application/json",
            as_attachment=True,
            download_name="prosodiff-card.json",
        )

    return app
