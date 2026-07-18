from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from prosodiff.analysis import compare_takes
from prosodiff.demo import harmonic_signal
from prosodiff.models import Comparison


@pytest.fixture(scope="session")
def synthetic_wavs(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    directory = tmp_path_factory.mktemp("prosodiff-wavs")
    sample_rate_hz = 22_050
    edge = np.zeros(int(0.22 * sample_rate_hz), dtype=float)

    base = harmonic_signal(
        sample_rate_hz=sample_rate_hz,
        duration_s=2.0,
        base_f0_hz=180.0,
        pitch_span_st=0.0,
        amplitude=0.16,
        syllable_rate_hz=2.2,
        emphasis_at=0.6,
    )
    shifted = harmonic_signal(
        sample_rate_hz=sample_rate_hz,
        duration_s=2.0,
        base_f0_hz=180.0 * 2.0 ** (3.0 / 12.0),
        pitch_span_st=0.0,
        amplitude=0.16,
        syllable_rate_hz=2.2,
        emphasis_at=0.6,
    )
    louder = base * 10.0 ** (6.0 / 20.0)
    longer_paused = harmonic_signal(
        sample_rate_hz=sample_rate_hz,
        duration_s=2.4,
        base_f0_hz=180.0,
        pitch_span_st=0.0,
        amplitude=0.16,
        syllable_rate_hz=2.2,
        emphasis_at=0.6,
        pause=(1.0, 1.15),
    )

    signals = [base, shifted, louder, longer_paused]
    paths: list[Path] = []
    for index, signal in enumerate(signals, start=1):
        path = directory / f"take_{index}.wav"
        sf.write(
            path, np.concatenate([edge, signal, edge]), sample_rate_hz, subtype="PCM_16"
        )
        paths.append(path)
    return paths


@pytest.fixture(scope="session")
def four_take_comparison(synthetic_wavs: list[Path]) -> Comparison:
    return compare_takes(
        synthetic_wavs,
        labels=["Base", "Pitch +3 st", "Level +6 dB", "Longer + pause"],
        text="Synthetic matched-text fixture",
        synthetic_demo=True,
    )
