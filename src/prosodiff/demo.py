"""Deterministic synthetic signals for the no-audio demo and tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class DemoTake:
    """Parameters for one deliberately stylized synthetic delivery."""

    label: str
    duration_s: float
    base_f0_hz: float
    pitch_span_st: float
    amplitude: float
    syllable_rate_hz: float
    emphasis_at: float
    pause: tuple[float, float] | None = None


DEMO_TAKES = (
    DemoTake("Calm", 3.55, 155.0, 3.4, 0.30, 2.4, 0.70, (1.67, 1.86)),
    DemoTake("Encouraging", 3.30, 174.0, 6.2, 0.36, 2.8, 0.62, (1.48, 1.65)),
    DemoTake("Deliberate", 4.15, 148.0, 4.5, 0.31, 2.0, 0.55, (1.92, 2.31)),
    DemoTake("Urgent", 2.85, 188.0, 7.2, 0.40, 3.3, 0.77, None),
)


def harmonic_signal(
    *,
    sample_rate_hz: int,
    duration_s: float,
    base_f0_hz: float,
    pitch_span_st: float,
    amplitude: float,
    syllable_rate_hz: float,
    emphasis_at: float,
    pause: tuple[float, float] | None = None,
) -> np.ndarray:
    """Create a deterministic speech-like harmonic signal.

    The signal validates the analysis and rendering pipeline; it is not a
    perceptual model of speech.
    """

    sample_count = int(round(duration_s * sample_rate_hz))
    time = np.arange(sample_count, dtype=float) / sample_rate_hz
    progress = time / duration_s

    contour_st = (
        0.55 * pitch_span_st * np.sin(2.0 * np.pi * (0.70 * progress + 0.03))
        + 0.45 * pitch_span_st * np.sin(2.0 * np.pi * (1.55 * progress - 0.17))
        + 0.70 * pitch_span_st * np.exp(-0.5 * ((progress - emphasis_at) / 0.095) ** 2)
    )
    f0_hz = base_f0_hz * 2.0 ** (contour_st / 12.0)
    phase = 2.0 * np.pi * np.cumsum(f0_hz) / sample_rate_hz
    carrier = (
        np.sin(phase)
        + 0.48 * np.sin(2.0 * phase + 0.2)
        + 0.24 * np.sin(3.0 * phase - 0.1)
        + 0.10 * np.sin(4.0 * phase + 0.4)
    ) / 1.82

    syllables = (
        0.48 + 0.52 * (0.5 + 0.5 * np.sin(2.0 * np.pi * syllable_rate_hz * time)) ** 1.6
    )
    emphasis = 0.78 + 0.40 * np.exp(-0.5 * ((progress - emphasis_at) / 0.10) ** 2)
    envelope = amplitude * syllables * emphasis

    fade_samples = max(1, int(round(0.025 * sample_rate_hz)))
    fade = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade_samples))
    envelope[:fade_samples] *= fade
    envelope[-fade_samples:] *= fade[::-1]
    if pause is not None:
        pause_start = int(round(pause[0] * sample_rate_hz))
        pause_end = int(round(pause[1] * sample_rate_hz))
        envelope[pause_start:pause_end] = 0.0

    return np.clip(carrier * envelope, -0.95, 0.95).astype(np.float64)


def write_demo_wavs(
    directory: Path, *, sample_rate_hz: int = 22_050
) -> tuple[list[Path], list[str]]:
    """Write four temporary WAVs and return their paths and labels."""

    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    labels: list[str] = []
    for index, take in enumerate(DEMO_TAKES, start=1):
        core = harmonic_signal(
            sample_rate_hz=sample_rate_hz,
            duration_s=take.duration_s,
            base_f0_hz=take.base_f0_hz,
            pitch_span_st=take.pitch_span_st,
            amplitude=take.amplitude,
            syllable_rate_hz=take.syllable_rate_hz,
            emphasis_at=take.emphasis_at,
            pause=take.pause,
        )
        edge = np.zeros(int(round(0.24 * sample_rate_hz)), dtype=float)
        audio = np.concatenate([edge, core, edge])
        path = directory / f"take_{index}_{take.label.lower()}.wav"
        sf.write(path, audio, sample_rate_hz, subtype="PCM_16")
        paths.append(path)
        labels.append(take.label)
    return paths, labels
