from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from prosodiff.web import create_app


def _wav_upload(filename: str = "take.wav") -> tuple[BytesIO, str]:
    buffer = BytesIO()
    sf.write(buffer, np.zeros(4_000), 16_000, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer, filename


def _comparison() -> SimpleNamespace:
    first_metrics = SimpleNamespace(
        utterance_duration_s=1.5,
        f0_median_hz=180.0,
        f0_iqr_st=2.0,
        pause_fraction=0.10,
    )
    second_metrics = SimpleNamespace(
        utterance_duration_s=1.8,
        f0_median_hz=200.0,
        f0_iqr_st=3.2,
        pause_fraction=0.15,
    )
    takes = [
        SimpleNamespace(id="take_1", label="First", metrics=first_metrics),
        SimpleNamespace(id="take_2", label="Second", metrics=second_metrics),
    ]
    pairs = [
        SimpleNamespace(
            a="take_1",
            b="take_2",
            duration_percent=20.0,
            f0_median_st=1.82,
            pause_fraction_percentage_points=5.0,
        )
    ]
    return SimpleNamespace(
        takes=takes,
        pairs=pairs,
        text="Matched sentence",
        warnings=["[TEST_NOTICE] Synthetic route fixture."],
    )


def _mock_pipeline(monkeypatch) -> None:
    comparison = _comparison()

    def fake_render(_comparison, path: Path) -> Path:
        path.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
        return path

    def fake_report(_comparison, path: Path) -> Path:
        path.write_text('{"fixture": true}\n', encoding="utf-8")
        return path

    monkeypatch.setattr("prosodiff.web.compare_takes", lambda *args, **kwargs: comparison)
    monkeypatch.setattr("prosodiff.web.render_comparison", fake_render)
    monkeypatch.setattr("prosodiff.web.write_json_report", fake_report)


def _app(tmp_path: Path):
    app = create_app(result_root=tmp_path / "results", csrf_token="test-token")
    app.config.update(TESTING=True)
    return app


def test_index_is_accessible_and_hardened(tmp_path: Path) -> None:
    app = _app(tmp_path)
    response = app.test_client().get("/")

    assert response.status_code == 200
    assert b'<html lang="en">' in response.data
    assert b'name="csrf_token" value="test-token"' in response.data
    assert b'id="comparison-form"' in response.data
    assert response.headers["Cache-Control"] == "no-store"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert response.headers["X-Frame-Options"] == "DENY"


def test_compare_requires_valid_csrf_and_file_count(tmp_path: Path) -> None:
    app = _app(tmp_path)
    client = app.test_client()

    missing_token = client.post("/compare", data={"protocol_ack": "yes"})
    assert missing_token.status_code == 403

    one_file = client.post(
        "/compare",
        data={
            "csrf_token": "test-token",
            "protocol_ack": "yes",
            "wavs": [_wav_upload()],
        },
        content_type="multipart/form-data",
    )
    assert one_file.status_code == 400
    assert b"between two and four" in one_file.data


def test_compare_rejects_non_wav_and_untrusted_host(tmp_path: Path) -> None:
    app = _app(tmp_path)
    client = app.test_client()
    response = client.post(
        "/compare",
        data={
            "csrf_token": "test-token",
            "protocol_ack": "yes",
            "wavs": [_wav_upload("one.wav"), (BytesIO(b"not audio"), "two.mp3")],
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert b"must be a WAV file" in response.data

    untrusted = client.get("/", headers={"Host": "attacker.example"})
    assert untrusted.status_code == 400

    foreign_origin = client.post(
        "/compare",
        data={"csrf_token": "test-token"},
        headers={"Origin": "https://attacker.example"},
    )
    assert foreign_origin.status_code == 403


def test_success_keeps_only_outputs_and_supports_downloads(tmp_path: Path, monkeypatch) -> None:
    _mock_pipeline(monkeypatch)
    app = _app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/compare",
        data={
            "csrf_token": "test-token",
            "protocol_ack": "yes",
            "text": "Matched sentence",
            "labels": ["First", "Second"],
            "wavs": [_wav_upload("../../one.wav"), _wav_upload("C:\\two.wav")],
        },
        content_type="multipart/form-data",
        headers={"Origin": "null"},
    )
    assert response.status_code == 303
    assert response.headers["Location"].startswith("/results/")

    result = client.get(response.headers["Location"])
    assert result.status_code == 200
    assert b"Comparison ready" in result.data
    assert b"Synthetic route fixture" in result.data

    run_id = response.headers["Location"].rsplit("/", 1)[-1]
    image = client.get(f"/results/{run_id}/download/card.png")
    report = client.get(f"/results/{run_id}/download/card.json")
    assert image.status_code == 200
    assert image.data.startswith(b"\x89PNG")
    assert report.json == {"fixture": True}
    assert not list((tmp_path / "results").rglob("*.wav"))
    assert {path.name for path in (tmp_path / "results" / run_id).iterdir()} == {
        "prosodiff-card.png",
        "prosodiff-card.json",
    }


def test_result_escapes_labels(tmp_path: Path, monkeypatch) -> None:
    comparison = _comparison()
    comparison.takes[0].label = "<script>alert(1)</script>"
    monkeypatch.setattr("prosodiff.web.compare_takes", lambda *args, **kwargs: comparison)

    def fake_render(_comparison, path: Path) -> Path:
        path.write_bytes(b"png")
        return path

    def fake_report(_comparison, path: Path) -> Path:
        path.write_text("{}", encoding="utf-8")
        return path

    monkeypatch.setattr("prosodiff.web.render_comparison", fake_render)
    monkeypatch.setattr("prosodiff.web.write_json_report", fake_report)
    app = _app(tmp_path)
    client = app.test_client()
    response = client.post(
        "/compare",
        data={
            "csrf_token": "test-token",
            "protocol_ack": "yes",
            "labels": ["<script>alert(1)</script>", "Second"],
            "wavs": [_wav_upload("one.wav"), _wav_upload("two.wav")],
        },
        content_type="multipart/form-data",
    )
    result = client.get(response.headers["Location"])
    assert b"&lt;script&gt;alert(1)&lt;/script&gt;" in result.data
    assert b"<script>alert(1)</script>" not in result.data


def test_unknown_result_is_not_found(tmp_path: Path) -> None:
    app = _app(tmp_path)
    assert app.test_client().get("/results/not-a-run").status_code == 404


def test_request_size_limit(tmp_path: Path) -> None:
    app = _app(tmp_path)
    app.config["MAX_CONTENT_LENGTH"] = 100
    response = app.test_client().post(
        "/compare",
        data={
            "csrf_token": "test-token",
            "protocol_ack": "yes",
            "wavs": [(BytesIO(b"x" * 200), "one.wav"), _wav_upload("two.wav")],
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 413
    assert b"upload is too large" in response.data
