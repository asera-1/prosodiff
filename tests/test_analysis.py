from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from prosodiff import __version__
from prosodiff.analysis import _valid_pitch_mask, analyse_take, compare_takes
from prosodiff.errors import AudioInputError
from prosodiff.models import AnalysisSettings, Comparison


def _pair(comparison: Comparison, a: str, b: str):
    return next(pair for pair in comparison.pairs if pair.a == a and pair.b == b)


def test_enumerates_all_unordered_pairs(four_take_comparison: Comparison) -> None:
    assert len(four_take_comparison.pairs) == 6
    assert [(pair.a, pair.b) for pair in four_take_comparison.pairs] == [
        ("take_1", "take_2"),
        ("take_1", "take_3"),
        ("take_1", "take_4"),
        ("take_2", "take_3"),
        ("take_2", "take_4"),
        ("take_3", "take_4"),
    ]


def test_recovers_known_pitch_shift(four_take_comparison: Comparison) -> None:
    pair = _pair(four_take_comparison, "take_1", "take_2")
    assert pair.f0_median_st == pytest.approx(3.0, abs=0.2)
    assert pair.duration_s == pytest.approx(0.0, abs=0.05)


def test_recovers_known_recorded_level_shift(four_take_comparison: Comparison) -> None:
    pair = _pair(four_take_comparison, "take_1", "take_3")
    assert pair.active_rms_median_db == pytest.approx(6.02, abs=0.12)
    assert pair.f0_median_st == pytest.approx(0.0, abs=0.1)


def test_recovers_duration_and_internal_pause(four_take_comparison: Comparison) -> None:
    pair = _pair(four_take_comparison, "take_1", "take_4")
    paused_take = four_take_comparison.takes[3]
    assert pair.duration_s == pytest.approx(0.40, abs=0.08)
    assert paused_take.metrics.pause_count == 1
    assert paused_take.metrics.pause_total_s == pytest.approx(0.15, abs=0.04)
    assert pair.pause_fraction_percentage_points > 4.0


def test_decoded_voicing_selection_does_not_delete_low_probability_f0() -> None:
    f0 = np.asarray([100.0, 101.0, np.nan, 102.0])
    flags = np.asarray([True, True, True, True])
    probabilities = np.asarray([0.90, 0.40, 0.99, 0.80])
    eligible = np.asarray([True, True, True, True])
    mask = _valid_pitch_mask(
        f0,
        flags,
        probabilities,
        eligible,
        threshold=0.80,
    )
    assert mask.tolist() == [True, True, False, True]


def test_low_probability_decoded_frames_still_produce_pitch_summary(
    synthetic_wavs: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def low_probability_pyin(audio: np.ndarray, **kwargs):
        frame_count = 1 + (
            audio.size - kwargs["frame_length"]
        ) // kwargs["hop_length"]
        return (
            np.full(frame_count, 180.0),
            np.ones(frame_count, dtype=bool),
            np.full(frame_count, 0.40),
        )

    monkeypatch.setattr("prosodiff.analysis.librosa.pyin", low_probability_pyin)
    take = analyse_take(
        synthetic_wavs[0],
        take_id="take_1",
        label="Natural delivery",
        settings=AnalysisSettings(),
    )

    assert take.metrics.f0_median_hz == pytest.approx(180.0)
    assert take.metrics.pitch_valid_frames >= 20
    assert take.metrics.pitch_high_probability_frames == 0
    assert take.metrics.high_probability_fraction == 0.0
    assert not any("PITCH_LOW_COVERAGE" in warning for warning in take.warnings)


def test_json_schema_is_standard_and_auditable(
    four_take_comparison: Comparison,
) -> None:
    payload = four_take_comparison.to_dict(version=__version__)
    encoded = json.dumps(payload, allow_nan=False)
    assert "NaN" not in encoded
    assert payload["schema"] == "prosodiff.explicit-delivery-attribute-delta"
    assert payload["schema_version"] == "0.2.0"
    assert payload["analysis"]["pitch"]["voiced_probability_role"] == "diagnostic_only"
    assert len(payload["takes"]) == 4
    assert len(payload["pairs"]) == 6
    assert payload["pairs"][0]["direction"] == "b_minus_a"
    assert payload["takes"][0]["path"] == "take_1.wav"
    for take in payload["takes"]:
        pitch = take["pitch"]
        contours = take["contours"]
        assert pitch["selection_method"] == "pyin_viterbi_voiced_flag"
        assert pitch["valid_frames"] == sum(
            value is not None for value in contours["f0_hz"]
        )
        pitch_lengths = {
            len(contours[key])
            for key in (
                "pitch_time_s",
                "pitch_time_norm",
                "f0_hz",
                "pyin_candidate_f0_hz",
                "pyin_voiced_flag",
                "pitch_eligible_mask",
                "voiced_probability",
            )
        }
        assert pitch_lengths == {len(contours["pitch_time_s"])}
        assert all(isinstance(value, bool) for value in contours["pyin_voiced_flag"])
        assert all(isinstance(value, bool) for value in contours["pitch_eligible_mask"])
        assert min(take["contours"]["pitch_time_norm"]) >= 0.0
        assert max(take["contours"]["pitch_time_norm"]) <= 1.0
        assert min(take["contours"]["energy_time_norm"]) >= 0.0
        assert max(take["contours"]["energy_time_norm"]) <= 1.0


def test_duplicate_input_is_rejected(synthetic_wavs: list[Path]) -> None:
    with pytest.raises(AudioInputError, match="different WAV"):
        compare_takes([synthetic_wavs[0], synthetic_wavs[0]])


def test_silence_is_rejected_with_helpful_code(tmp_path: Path) -> None:
    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros(22_050), 22_050, subtype="PCM_16")
    with pytest.raises(AudioInputError, match="NO_ACTIVE_SPEECH"):
        compare_takes([silent, silent.with_name("missing.wav")])
