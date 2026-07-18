"""Local-only browser interface for Prosodiff."""

from __future__ import annotations

import atexit
import hmac
import json
import math
import re
import secrets
import shutil
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, BinaryIO

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
from prosodiff.models import CaptureMetadata, Comparison
from prosodiff.render import render_comparison
from prosodiff.report import write_json_report


_MAX_FILES = 4
_MIN_FILES = 2
_MAX_FILE_BYTES = 24 * 1024 * 1024
_MAX_LABEL_LENGTH = 64
_MAX_TEXT_LENGTH = 300
_MAX_RESULTS = 10
_MAX_CAPTURE_METADATA_BYTES = 8 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_ANALYSIS_LOCK = threading.Lock()
_WARNING_PATTERN = re.compile(r"^\[([A-Z0-9_]+)\]\s*(.*)$")
_CLIENT_QA_CODES = {
    "LOW_LEVEL",
    "CLIPPING",
    "LEVEL_MISMATCH",
    "DEVICE_CHANGED",
}
_WARNING_TITLES = {
    "PITCH_LOW_COVERAGE": "Pitch coverage",
    "PITCH_REFERENCE_INCOMPLETE": "Pitch comparison",
    "PITCH_BOUNDARY_HITS": "Pitch search range",
    "PITCH_JUMPS_SUSPECT": "Pitch tracking",
    "PAUSE_THRESHOLD_UNSTABLE": "Pause sensitivity",
    "LOW_ENERGY_SEPARATION": "Speech/pause separation",
    "PAIR_LEVEL_MISMATCH": "Recording-level mismatch",
    "PAIR_DURATION_MISMATCH": "Duration mismatch",
    "LOW_RECORDED_LEVEL": "Low recorded level",
    "CLIPPING_DETECTED": "Clipping",
    "EXCESSIVE_EDGE_SILENCE": "Edge silence",
    "UTTERANCE_TOO_SHORT": "Short utterance",
    "STEREO_DOWNMIX": "Stereo downmix",
    "CAPTURE_PROCESSING_ACTIVE": "Browser processing active",
    "CAPTURE_SETTINGS_UNKNOWN": "Browser settings unavailable",
    "CAPTURE_DEVICE_CHANGED": "Input device changed",
    "CAPTURE_CONSTRAINTS_FALLBACK": "Capture constraints fallback",
}


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


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite number {value}")


def _mapping(value: object, context: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object.")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{context} contains unsupported fields.")
    return value


def _optional_bool(value: object, context: str) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise ValueError(f"{context} must be true, false, or null.")


def _optional_number(
    value: object,
    context: str,
    *,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a number or null.")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{context} is outside the supported range.")
    return number


def _optional_int(
    value: object,
    context: str,
    *,
    minimum: int,
    maximum: int,
) -> int | None:
    number = _optional_number(
        value,
        context,
        minimum=float(minimum),
        maximum=float(maximum),
    )
    if number is None:
        return None
    if not number.is_integer():
        raise ValueError(f"{context} must be a whole number.")
    return int(number)


def _parse_capture_metadata(
    raw: str,
    *,
    expected_count: int,
) -> list[dict[str, Any]] | None:
    """Validate untrusted, privacy-preserving metadata from live capture."""

    if not raw:
        return None
    if len(raw.encode("utf-8")) > _MAX_CAPTURE_METADATA_BYTES:
        raise ValueError("Capture metadata is too large. Record the takes again.")
    try:
        payload = json.loads(raw, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Capture metadata is invalid. Record the takes again.") from exc
    payload = _mapping(payload, "Capture metadata", {"schema_version", "takes"})
    if payload.get("schema_version") != "0.1.0":
        raise ValueError("Capture metadata uses an unsupported schema version.")
    takes = payload.get("takes")
    if not isinstance(takes, list) or len(takes) != expected_count:
        raise ValueError("Capture metadata must match the selected recordings.")

    normalized: list[dict[str, Any]] = []
    for index, raw_take in enumerate(takes, start=1):
        context = f"Capture metadata for Take {index}"
        take = _mapping(
            raw_take,
            context,
            {"source", "wav", "track", "constraints_fallback", "client_qa"},
        )
        if take.get("source") != "live":
            raise ValueError(f"{context} has an unsupported source.")
        fallback = take.get("constraints_fallback", False)
        if not isinstance(fallback, bool):
            raise ValueError(f"{context} fallback flag must be true or false.")
        wav = _mapping(
            take.get("wav"),
            f"{context} WAV",
            {"sample_rate_hz", "channels", "duration_s"},
        )
        track = _mapping(
            take.get("track"),
            f"{context} track settings",
            {
                "sample_rate_hz",
                "channel_count",
                "sample_size_bits",
                "latency_s",
                "echo_cancellation",
                "noise_suppression",
                "auto_gain_control",
                "same_device_as_reference",
            },
        )
        qa = _mapping(
            take.get("client_qa"),
            f"{context} level check",
            {
                "active_rms_estimate_dbfs",
                "peak_dbfs",
                "clipped_sample_fraction",
                "codes",
            },
        )
        codes = qa.get("codes")
        if not isinstance(codes, list) or len(codes) > 8:
            raise ValueError(f"{context} level-check codes are invalid.")
        if any(not isinstance(code, str) or code not in _CLIENT_QA_CODES for code in codes):
            raise ValueError(f"{context} level-check codes are invalid.")

        normalized.append(
            {
                "wav_sample_rate_hz": _optional_int(
                    wav.get("sample_rate_hz"),
                    f"{context} WAV sample rate",
                    minimum=8_000,
                    maximum=192_000,
                ),
                "wav_channels": _optional_int(
                    wav.get("channels"),
                    f"{context} WAV channels",
                    minimum=1,
                    maximum=32,
                ),
                "wav_duration_s": _optional_number(
                    wav.get("duration_s"),
                    f"{context} WAV duration",
                    minimum=0.0,
                    maximum=30.1,
                ),
                "track_sample_rate_hz": _optional_int(
                    track.get("sample_rate_hz"),
                    f"{context} track sample rate",
                    minimum=8_000,
                    maximum=192_000,
                ),
                "track_channel_count": _optional_int(
                    track.get("channel_count"),
                    f"{context} track channel count",
                    minimum=1,
                    maximum=32,
                ),
                "track_sample_size_bits": _optional_int(
                    track.get("sample_size_bits"),
                    f"{context} track sample size",
                    minimum=8,
                    maximum=64,
                ),
                "track_latency_s": _optional_number(
                    track.get("latency_s"),
                    f"{context} track latency",
                    minimum=0.0,
                    maximum=10.0,
                ),
                "echo_cancellation": _optional_bool(
                    track.get("echo_cancellation"),
                    f"{context} echo cancellation",
                ),
                "noise_suppression": _optional_bool(
                    track.get("noise_suppression"),
                    f"{context} noise suppression",
                ),
                "auto_gain_control": _optional_bool(
                    track.get("auto_gain_control"),
                    f"{context} auto gain control",
                ),
                "same_device_as_reference": _optional_bool(
                    track.get("same_device_as_reference"),
                    f"{context} device consistency",
                ),
                "constraints_fallback": fallback,
                "client_active_rms_dbfs": _optional_number(
                    qa.get("active_rms_estimate_dbfs"),
                    f"{context} active RMS",
                    minimum=-160.0,
                    maximum=10.0,
                ),
                "client_peak_dbfs": _optional_number(
                    qa.get("peak_dbfs"),
                    f"{context} peak",
                    minimum=-160.0,
                    maximum=3.0,
                ),
                "client_clipped_fraction": _optional_number(
                    qa.get("clipped_sample_fraction"),
                    f"{context} clipped fraction",
                    minimum=0.0,
                    maximum=1.0,
                ),
                "client_qa_codes": tuple(codes),
            }
        )
    return normalized


def _capture_records(
    paths: list[Path],
    parsed: list[dict[str, Any]] | None,
) -> list[CaptureMetadata]:
    """Combine client provenance with server-observed WAV properties."""

    records: list[CaptureMetadata] = []
    for index, path in enumerate(paths):
        info = sf.info(path)
        client = parsed[index] if parsed is not None else None
        records.append(
            CaptureMetadata(
                source="live" if client is not None else "upload",
                client_reported=client is not None,
                encoded_sample_rate_hz=int(info.samplerate),
                encoded_channels=int(info.channels),
                encoded_duration_s=float(info.duration),
                track_sample_rate_hz=(
                    client["track_sample_rate_hz"] if client is not None else None
                ),
                track_channel_count=(
                    client["track_channel_count"] if client is not None else None
                ),
                track_sample_size_bits=(
                    client["track_sample_size_bits"] if client is not None else None
                ),
                track_latency_s=(
                    client["track_latency_s"] if client is not None else None
                ),
                echo_cancellation=(
                    client["echo_cancellation"] if client is not None else None
                ),
                noise_suppression=(
                    client["noise_suppression"] if client is not None else None
                ),
                auto_gain_control=(
                    client["auto_gain_control"] if client is not None else None
                ),
                same_device_as_reference=(
                    client["same_device_as_reference"] if client is not None else None
                ),
                constraints_fallback=(
                    bool(client["constraints_fallback"])
                    if client is not None
                    else False
                ),
                client_active_rms_dbfs=(
                    client["client_active_rms_dbfs"] if client is not None else None
                ),
                client_peak_dbfs=(
                    client["client_peak_dbfs"] if client is not None else None
                ),
                client_clipped_fraction=(
                    client["client_clipped_fraction"] if client is not None else None
                ),
                client_qa_codes=(
                    client["client_qa_codes"] if client is not None else ()
                ),
            )
        )
    return records


def _warning_groups(comparison: Comparison) -> list[dict[str, Any]]:
    """Group repeated audit notices into readable result-page checks."""

    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for warning in dict.fromkeys(comparison.warnings):
        match = _WARNING_PATTERN.match(warning)
        code, message = (match.group(1), match.group(2)) if match else ("GENERAL", warning)
        if code in {"RECORDING_PROTOCOL", "SAMPLE_RATE_RESAMPLED"}:
            continue
        group = grouped.setdefault(
            code,
            {
                "code": code,
                "title": _WARNING_TITLES.get(code, code.replace("_", " ").title()),
                "messages": [],
            },
        )
        if message not in group["messages"]:
            group["messages"].append(message)
    for group in grouped.values():
        group["count"] = len(group["messages"])
    return list(grouped.values())


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
            parsed_capture = _parse_capture_metadata(
                request.form.get("capture_metadata", ""),
                expected_count=len(files),
            )
        except ValueError as exc:
            return _index_response(str(exc), status=400)

        run_id, directory = _result_store().reserve()
        wav_paths: list[Path] = []
        try:
            wav_paths = _save_wavs(files, directory)
            capture_records = _capture_records(wav_paths, parsed_capture)
            with _ANALYSIS_LOCK:
                comparison = compare_takes(
                    wav_paths,
                    labels=labels,
                    text=text or None,
                    synthetic_demo=False,
                    capture_metadata=capture_records,
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
            warning_groups=_warning_groups(record.comparison),
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
