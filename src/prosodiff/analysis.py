"""Scientifically conservative acoustic analysis for matched deliveries."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Iterator

import librosa
import numpy as np
import soundfile as sf

from prosodiff.errors import AudioInputError
from prosodiff.models import (
    AnalysisSettings,
    CaptureMetadata,
    Comparison,
    DeliveryDelta,
    TakeAnalysis,
    TakeMetrics,
)


_EPSILON = 1e-10
_MINIMUM_INPUT_DURATION_S = 0.35
_MINIMUM_UTTERANCE_DURATION_S = 0.75
_MAXIMUM_RECOMMENDED_DURATION_S = 30.0


@dataclass(frozen=True)
class _PauseDetection:
    start_s: float
    end_s: float
    pause_mask: np.ndarray
    pause_durations_s: tuple[float, ...]

    @property
    def pause_total_s(self) -> float:
        return float(sum(self.pause_durations_s))

    @property
    def pause_count(self) -> int:
        return len(self.pause_durations_s)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def pause_fraction(self) -> float:
        return self.pause_total_s / self.duration_s if self.duration_s > 0 else 0.0


def _warning(code: str, message: str) -> str:
    return f"[{code}] {message}"


def _dbfs(values: np.ndarray) -> np.ndarray:
    """Convert full-scale amplitude values to dBFS without infinities."""

    return 20.0 * np.log10(np.maximum(values, _EPSILON))


def _runs(mask: np.ndarray) -> Iterator[tuple[int, int]]:
    """Yield half-open runs of true values."""

    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    yield from zip(starts, stops, strict=True)


def _remove_short_true(mask: np.ndarray, minimum_frames: int) -> np.ndarray:
    result = mask.copy()
    for start, stop in _runs(mask):
        if stop - start < minimum_frames:
            result[start:stop] = False
    return result


def _energy_run_duration_s(
    start: int,
    stop: int,
    settings: AnalysisSettings,
) -> float:
    """Duration covered by overlapping energy frames in one run."""

    if stop <= start:
        return 0.0
    samples = (stop - start - 1) * settings.hop_length + settings.energy_frame_length
    return samples / settings.sample_rate_hz


def _fill_short_false(
    mask: np.ndarray,
    minimum_duration_s: float,
    settings: AnalysisSettings,
) -> np.ndarray:
    result = mask.copy()
    # Threshold crossings are quantized to the hop grid. Allow one hop of
    # total boundary uncertainty so a nominal 150 ms gap is not rejected only
    # because its first/last crossing landed between adjacent frames.
    boundary_tolerance_s = settings.hop_length / settings.sample_rate_hz
    for start, stop in _runs(~mask):
        if (
            _energy_run_duration_s(start, stop, settings) + boundary_tolerance_s
            < minimum_duration_s
        ):
            result[start:stop] = True
    return result


def _minimum_frames(milliseconds: float, settings: AnalysisSettings) -> int:
    return max(
        1,
        int(
            np.ceil(
                milliseconds / 1_000.0 * settings.sample_rate_hz / settings.hop_length
            )
        ),
    )


def _detect_pauses(
    rms_dbfs: np.ndarray,
    *,
    threshold_dbfs: float,
    file_duration_s: float,
    settings: AnalysisSettings,
) -> _PauseDetection:
    """Detect an utterance span and internal energy-defined pauses."""

    active = rms_dbfs >= threshold_dbfs
    active = _remove_short_true(
        active,
        _minimum_frames(settings.maximum_noise_burst_ms, settings),
    )
    active = _fill_short_false(
        active,
        settings.minimum_pause_ms / 1_000.0,
        settings,
    )
    active_indices = np.flatnonzero(active)
    if active_indices.size == 0:
        raise AudioInputError(
            _warning("NO_ACTIVE_SPEECH", "No active speech-like region was detected.")
        )

    frame_starts_s = (
        np.arange(rms_dbfs.size, dtype=float)
        * settings.hop_length
        / settings.sample_rate_hz
    )
    frame_ends_s = np.minimum(
        frame_starts_s + settings.energy_frame_length / settings.sample_rate_hz,
        file_duration_s,
    )
    edge_pad_s = settings.edge_pad_ms / 1_000.0
    start_s = max(0.0, float(frame_starts_s[active_indices[0]]) - edge_pad_s)
    end_s = min(file_duration_s, float(frame_ends_s[active_indices[-1]]) + edge_pad_s)

    inside_span = (frame_ends_s > start_s) & (frame_starts_s < end_s)
    pause_candidates = (~active) & inside_span
    pause_mask = np.zeros_like(active, dtype=bool)
    pause_durations: list[float] = []
    hop_s = settings.hop_length / settings.sample_rate_hz
    for start, stop in _runs(pause_candidates):
        if (
            _energy_run_duration_s(start, stop, settings) + hop_s + 1e-9
            < settings.minimum_pause_ms / 1_000.0
        ):
            continue
        interval_start = max(start_s, float(frame_starts_s[start]) - hop_s / 2.0)
        interval_end = min(end_s, float(frame_ends_s[stop - 1]) + hop_s / 2.0)
        if interval_start <= start_s or interval_end >= end_s:
            continue
        duration = interval_end - interval_start
        if duration + 1e-9 < settings.minimum_pause_ms / 1_000.0:
            continue
        pause_mask[start:stop] = True
        pause_durations.append(duration)

    return _PauseDetection(
        start_s=start_s,
        end_s=end_s,
        pause_mask=pause_mask,
        pause_durations_s=tuple(pause_durations),
    )


def _inside_pause(
    times_s: np.ndarray, energy_times_s: np.ndarray, pause_mask: np.ndarray
) -> np.ndarray:
    """Map an energy-frame pause mask onto arbitrary frame-center times."""

    if energy_times_s.size == 0 or not np.any(pause_mask):
        return np.zeros(times_s.shape, dtype=bool)
    nearest = np.searchsorted(energy_times_s, times_s, side="left")
    nearest = np.clip(nearest, 0, energy_times_s.size - 1)
    previous = np.clip(nearest - 1, 0, energy_times_s.size - 1)
    choose_previous = np.abs(times_s - energy_times_s[previous]) < np.abs(
        times_s - energy_times_s[nearest]
    )
    nearest = np.where(choose_previous, previous, nearest)
    return pause_mask[nearest]


def _has_clipped_run(audio: np.ndarray, minimum_run: int = 3) -> bool:
    clipped = np.abs(audio) >= 0.999
    return any(stop - start >= minimum_run for start, stop in _runs(clipped))


def _pitch_iqr_semitones(f0_hz: np.ndarray) -> float:
    semitones = 12.0 * np.log2(f0_hz)
    return float(np.percentile(semitones, 75) - np.percentile(semitones, 25))


def _valid_pitch_mask(
    f0_hz: np.ndarray,
    voiced_flag: np.ndarray,
    voiced_probability: np.ndarray,
    eligible: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    """Select finite F0 frames decoded as voiced, independent of probability."""

    # Retain the legacy arguments so downstream callers do not break.  pYIN's
    # Viterbi-decoded voiced flag is the frame-selection decision; probability
    # and its threshold are diagnostics only.
    del voiced_probability, threshold
    return eligible & voiced_flag & np.isfinite(f0_hz)


def _high_probability_pitch_mask(
    selected_pitch: np.ndarray,
    voiced_probability: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    """Identify selected frames meeting the probability diagnostic threshold."""

    return selected_pitch & np.isfinite(voiced_probability) & (
        voiced_probability >= threshold
    )


def analyse_take(
    path: Path,
    *,
    take_id: str,
    label: str,
    settings: AnalysisSettings,
    capture_metadata: CaptureMetadata | None = None,
) -> TakeAnalysis:
    """Analyse one WAV without normalizing, denoising, or repairing pitch."""

    path = path.expanduser().resolve()
    if not path.exists():
        raise AudioInputError(f"Audio file does not exist: {path}")
    if path.suffix.lower() != ".wav":
        raise AudioInputError(
            f"Prosodiff v0.2 accepts WAV files only, received: {path.name}"
        )

    try:
        info = sf.info(path)
        samples, original_sample_rate = sf.read(
            path,
            dtype="float64",
            always_2d=True,
        )
    except (RuntimeError, ValueError) as exc:
        raise AudioInputError(f"Could not read WAV file {path.name}: {exc}") from exc

    if samples.size == 0 or info.frames == 0:
        raise AudioInputError(f"Audio file is empty: {path.name}")
    if not np.all(np.isfinite(samples)):
        raise AudioInputError(f"Audio contains non-finite samples: {path.name}")

    file_duration_s = float(info.frames / original_sample_rate)
    if file_duration_s < _MINIMUM_INPUT_DURATION_S:
        raise AudioInputError(
            _warning(
                "UTTERANCE_TOO_SHORT",
                f"{path.name} is {file_duration_s:.2f} s; at least "
                f"{_MINIMUM_INPUT_DURATION_S:.2f} s is required.",
            )
        )
    if file_duration_s > _MAXIMUM_RECOMMENDED_DURATION_S:
        raise AudioInputError(
            f"{path.name} is {file_duration_s:.1f} s; v0.2 is limited to short utterances under "
            f"{_MAXIMUM_RECOMMENDED_DURATION_S:.0f} s."
        )

    original_channels = int(samples.shape[1])
    audio = np.mean(samples, axis=1)
    warnings: list[str] = []
    if capture_metadata is not None and capture_metadata.source == "live":
        processing = (
            capture_metadata.echo_cancellation,
            capture_metadata.noise_suppression,
            capture_metadata.auto_gain_control,
        )
        if any(value is True for value in processing):
            warnings.append(
                _warning(
                    "CAPTURE_PROCESSING_ACTIVE",
                    f"{label}: the browser reports active microphone processing; energy may be confounded.",
                )
            )
        if any(value is None for value in processing):
            warnings.append(
                _warning(
                    "CAPTURE_SETTINGS_UNKNOWN",
                    f"{label}: the browser did not report every microphone-processing setting.",
                )
            )
        if capture_metadata.same_device_as_reference is False:
            warnings.append(
                _warning(
                    "CAPTURE_DEVICE_CHANGED",
                    f"{label}: the browser reports a different input device from the reference take.",
                )
            )
        if capture_metadata.constraints_fallback:
            warnings.append(
                _warning(
                    "CAPTURE_CONSTRAINTS_FALLBACK",
                    f"{label}: the browser retried capture without exact processing constraints.",
                )
            )
    if original_channels > 1:
        warnings.append(
            _warning(
                "STEREO_DOWNMIX",
                f"{label}: {original_channels} channels were averaged to mono.",
            )
        )

    peak_abs = float(np.max(np.abs(audio)))
    clipping_fraction = float(np.mean(np.abs(audio) >= 0.999))
    if clipping_fraction >= 0.001 or _has_clipped_run(audio):
        warnings.append(
            _warning(
                "CLIPPING_DETECTED",
                f"{label}: {clipping_fraction:.2%} of samples approach full scale.",
            )
        )
    if peak_abs < 1e-6:
        raise AudioInputError(
            _warning("NO_ACTIVE_SPEECH", f"{path.name} is effectively silent.")
        )

    # Remove DC only. Do not peak-normalize, denoise, or pre-emphasize.
    audio = audio - float(np.mean(audio))
    if original_sample_rate != settings.sample_rate_hz:
        audio = librosa.resample(
            audio,
            orig_sr=original_sample_rate,
            target_sr=settings.sample_rate_hz,
            res_type="polyphase",
        )

    rms = librosa.feature.rms(
        y=audio,
        frame_length=settings.energy_frame_length,
        hop_length=settings.hop_length,
        center=False,
    )[0]
    rms_dbfs = _dbfs(rms).astype(np.float64)
    energy_times_s = (
        np.arange(rms_dbfs.size, dtype=float) * settings.hop_length
        + settings.energy_frame_length / 2.0
    ) / settings.sample_rate_hz

    q10, q90 = np.percentile(rms_dbfs, [10, 90])
    threshold_dbfs = float(min(q90 - 12.0, max(q10 + 6.0, q90 - 35.0)))
    if q90 - q10 < 12.0:
        warnings.append(
            _warning(
                "LOW_ENERGY_SEPARATION",
                f"{label}: energy separation is only {q90 - q10:.1f} dB; pause estimates are unstable.",
            )
        )

    pauses = _detect_pauses(
        rms_dbfs,
        threshold_dbfs=threshold_dbfs,
        file_duration_s=file_duration_s,
        settings=settings,
    )
    pauses_minus = _detect_pauses(
        rms_dbfs,
        threshold_dbfs=threshold_dbfs - settings.pause_sensitivity_db,
        file_duration_s=file_duration_s,
        settings=settings,
    )
    pauses_plus = _detect_pauses(
        rms_dbfs,
        threshold_dbfs=threshold_dbfs + settings.pause_sensitivity_db,
        file_duration_s=file_duration_s,
        settings=settings,
    )
    pause_unstable = (
        abs(pauses.pause_fraction - pauses_minus.pause_fraction) > 0.05
        or abs(pauses.pause_fraction - pauses_plus.pause_fraction) > 0.05
        or abs(pauses.pause_count - pauses_minus.pause_count) > 1
        or abs(pauses.pause_count - pauses_plus.pause_count) > 1
    )
    if pause_unstable:
        warnings.append(
            _warning(
                "PAUSE_THRESHOLD_UNSTABLE",
                f"{label}: pause results change materially under a ±"
                f"{settings.pause_sensitivity_db:g} dB threshold check.",
            )
        )

    utterance_duration_s = pauses.duration_s
    if utterance_duration_s < _MINIMUM_UTTERANCE_DURATION_S:
        warnings.append(
            _warning(
                "UTTERANCE_TOO_SHORT",
                f"{label}: detected utterance span is {utterance_duration_s:.2f} s.",
            )
        )
    edge_silence = pauses.start_s + (file_duration_s - pauses.end_s)
    if edge_silence > 0.20 * file_duration_s or edge_silence > 1.0:
        warnings.append(
            _warning(
                "EXCESSIVE_EDGE_SILENCE",
                f"{label}: {edge_silence:.2f} s lies outside the detected utterance span.",
            )
        )

    pitch_times_s: np.ndarray
    try:
        candidate_f0_hz, voiced_flag, voiced_probability = librosa.pyin(
            audio,
            fmin=settings.fmin_hz,
            fmax=min(settings.fmax_hz, settings.sample_rate_hz / 2.0 - 1.0),
            sr=settings.sample_rate_hz,
            frame_length=settings.pitch_frame_length,
            hop_length=settings.hop_length,
            resolution=0.1,
            center=False,
            fill_na=None,
        )
    except Exception as exc:
        raise AudioInputError(f"Pitch analysis failed for {path.name}: {exc}") from exc

    pitch_times_s = (
        np.arange(candidate_f0_hz.size, dtype=float) * settings.hop_length
        + settings.pitch_frame_length / 2.0
    ) / settings.sample_rate_hz
    eligible_pitch = (
        (pitch_times_s >= pauses.start_s)
        & (pitch_times_s <= pauses.end_s)
        & (~_inside_pause(pitch_times_s, energy_times_s, pauses.pause_mask))
    )
    valid_pitch = _valid_pitch_mask(
        candidate_f0_hz,
        voiced_flag,
        voiced_probability,
        eligible_pitch,
        threshold=settings.voiced_probability_threshold,
    )
    high_probability_pitch = _high_probability_pitch_mask(
        valid_pitch,
        voiced_probability,
        threshold=settings.voiced_probability_threshold,
    )
    candidate_f0_hz = candidate_f0_hz.astype(np.float64)
    voiced_flag = voiced_flag.astype(bool)
    f0_hz = np.where(valid_pitch, candidate_f0_hz, np.nan).astype(np.float64)
    voiced_probability = voiced_probability.astype(np.float64)
    eligible_count = int(np.sum(eligible_pitch))
    valid_count = int(np.sum(valid_pitch))
    high_probability_count = int(np.sum(high_probability_pitch))
    f0_coverage = valid_count / eligible_count if eligible_count else 0.0
    high_probability_fraction = (
        high_probability_count / valid_count if valid_count else 0.0
    )
    reliable_pitch = (
        valid_count >= settings.minimum_pitch_frames
        and f0_coverage >= settings.minimum_pitch_coverage
    )
    valid_f0 = f0_hz[np.isfinite(f0_hz)]
    selected_probabilities = voiced_probability[
        valid_pitch & np.isfinite(voiced_probability)
    ]
    voiced_probability_median = (
        float(np.median(selected_probabilities))
        if selected_probabilities.size
        else None
    )
    voiced_probability_p10 = (
        float(np.percentile(selected_probabilities, 10))
        if selected_probabilities.size
        else None
    )
    voiced_probability_p90 = (
        float(np.percentile(selected_probabilities, 90))
        if selected_probabilities.size
        else None
    )
    if reliable_pitch:
        f0_median_hz: float | None = float(np.median(valid_f0))
        f0_iqr_st: float | None = _pitch_iqr_semitones(valid_f0)
    else:
        f0_median_hz = None
        f0_iqr_st = None
        warnings.append(
            _warning(
                "PITCH_LOW_COVERAGE",
                f"{label}: {valid_count} frames were decoded as voiced outside pauses "
                f"({f0_coverage:.0%} coverage); pitch summaries are unavailable.",
            )
        )

    if valid_count:
        boundary_distance_st = 0.5
        lower = settings.fmin_hz * 2 ** (boundary_distance_st / 12.0)
        upper = settings.fmax_hz / 2 ** (boundary_distance_st / 12.0)
        boundary_fraction = float(np.mean((valid_f0 <= lower) | (valid_f0 >= upper)))
        if boundary_fraction > 0.05:
            warnings.append(
                _warning(
                    "PITCH_BOUNDARY_HITS",
                    f"{label}: {boundary_fraction:.0%} of valid F0 frames touch a search boundary.",
                )
            )
        valid_indices = np.flatnonzero(valid_pitch)
        if valid_indices.size > 1:
            contiguous = np.diff(valid_indices) == 1
            jumps = np.abs(12.0 * np.diff(np.log2(valid_f0))) > 7.0
            jump_denominator = int(np.sum(contiguous))
            jump_fraction = (
                float(np.sum(jumps & contiguous) / jump_denominator)
                if jump_denominator
                else 0.0
            )
            if jump_fraction > 0.05:
                warnings.append(
                    _warning(
                        "PITCH_JUMPS_SUSPECT",
                        f"{label}: {jump_fraction:.0%} of adjacent valid frames jump by over 7 st.",
                    )
                )

    energy_inside = (energy_times_s >= pauses.start_s) & (
        energy_times_s <= pauses.end_s
    )
    active_energy = energy_inside & (~pauses.pause_mask)
    active_values = rms_dbfs[active_energy]
    if active_values.size == 0:
        raise AudioInputError(
            _warning(
                "NO_ACTIVE_SPEECH", f"No active energy frames remain in {path.name}."
            )
        )
    active_rms_median_dbfs = float(np.median(active_values))
    active_rms_iqr_db = float(
        np.percentile(active_values, 75) - np.percentile(active_values, 25)
    )
    active_rms_dynamic_10_90_db = float(
        np.percentile(active_values, 90) - np.percentile(active_values, 10)
    )
    if active_rms_median_dbfs < -35.0:
        warnings.append(
            _warning(
                "LOW_RECORDED_LEVEL",
                f"{label}: active recorded level is {active_rms_median_dbfs:.1f} dBFS.",
            )
        )

    start_sample = max(0, int(round(pauses.start_s * settings.sample_rate_hz)))
    end_sample = min(audio.size, int(round(pauses.end_s * settings.sample_rate_hz)))
    utterance_audio = audio[start_sample:end_sample]
    utterance_rms_dbfs = float(
        _dbfs(np.asarray([np.sqrt(np.mean(np.square(utterance_audio)))]))[0]
    )

    pause_durations = np.asarray(pauses.pause_durations_s, dtype=float)
    pause_total_s = pauses.pause_total_s
    pause_median_s = float(np.median(pause_durations)) if pause_durations.size else 0.0
    pause_max_s = float(np.max(pause_durations)) if pause_durations.size else 0.0

    pitch_time_norm = (pitch_times_s - pauses.start_s) / utterance_duration_s
    energy_time_norm = (energy_times_s - pauses.start_s) / utterance_duration_s
    pitch_inside = (pitch_time_norm >= 0.0) & (pitch_time_norm <= 1.0)
    energy_norm_inside = (energy_time_norm >= 0.0) & (energy_time_norm <= 1.0)
    pitch_time_relative = (pitch_times_s - pauses.start_s)[pitch_inside]
    pitch_time_norm = pitch_time_norm[pitch_inside]
    f0_hz = f0_hz[pitch_inside]
    candidate_f0_hz = candidate_f0_hz[pitch_inside]
    voiced_flag = voiced_flag[pitch_inside]
    eligible_pitch = eligible_pitch[pitch_inside]
    voiced_probability = voiced_probability[pitch_inside]
    energy_time_relative = (energy_times_s - pauses.start_s)[energy_norm_inside]
    energy_time_norm = energy_time_norm[energy_norm_inside]
    rms_dbfs = rms_dbfs[energy_norm_inside]
    pause_mask = pauses.pause_mask[energy_norm_inside]

    metrics = TakeMetrics(
        file_duration_s=file_duration_s,
        peak_abs=peak_abs,
        clipping_fraction=clipping_fraction,
        utterance_start_s=pauses.start_s,
        utterance_end_s=pauses.end_s,
        utterance_duration_s=utterance_duration_s,
        leading_silence_s=pauses.start_s,
        trailing_silence_s=file_duration_s - pauses.end_s,
        nonpause_time_s=max(0.0, utterance_duration_s - pause_total_s),
        pause_count=pauses.pause_count,
        pause_total_s=pause_total_s,
        pause_fraction=pauses.pause_fraction,
        pause_median_s=pause_median_s,
        pause_max_s=pause_max_s,
        pause_threshold_dbfs=threshold_dbfs,
        pause_fraction_threshold_minus_3db=pauses_minus.pause_fraction,
        pause_fraction_threshold_plus_3db=pauses_plus.pause_fraction,
        pitch_eligible_frames=eligible_count,
        pitch_valid_frames=valid_count,
        pitch_high_probability_frames=high_probability_count,
        f0_coverage=f0_coverage,
        high_probability_fraction=high_probability_fraction,
        voiced_probability_median=voiced_probability_median,
        voiced_probability_p10=voiced_probability_p10,
        voiced_probability_p90=voiced_probability_p90,
        f0_median_hz=f0_median_hz,
        f0_iqr_st=f0_iqr_st,
        active_rms_median_dbfs=active_rms_median_dbfs,
        active_rms_iqr_db=active_rms_iqr_db,
        active_rms_dynamic_10_90_db=active_rms_dynamic_10_90_db,
        utterance_rms_dbfs=utterance_rms_dbfs,
    )
    return TakeAnalysis(
        id=take_id,
        label=label,
        source=path,
        original_sample_rate_hz=int(original_sample_rate),
        original_channels=original_channels,
        metrics=metrics,
        pitch_time_s=pitch_time_relative.astype(np.float64),
        pitch_time_norm=pitch_time_norm.astype(np.float64),
        f0_hz=f0_hz,
        f0_candidate_hz=candidate_f0_hz,
        f0_voiced_flag=voiced_flag,
        pitch_eligible_mask=eligible_pitch,
        f0_probability=voiced_probability,
        energy_time_s=energy_time_relative.astype(np.float64),
        energy_time_norm=energy_time_norm.astype(np.float64),
        rms_dbfs=rms_dbfs,
        pause_mask=pause_mask,
        capture_metadata=capture_metadata,
        warnings=warnings,
    )


def _semitone_delta(
    target_hz: float | None, reference_hz: float | None
) -> float | None:
    if target_hz is None or reference_hz is None:
        return None
    return float(12.0 * np.log2(target_hz / reference_hz))


def _pause_stable(take: TakeAnalysis) -> bool:
    return not any("[PAUSE_THRESHOLD_UNSTABLE]" in warning for warning in take.warnings)


def _pair_delta(a: TakeAnalysis, b: TakeAnalysis) -> DeliveryDelta:
    ma = a.metrics
    mb = b.metrics
    warnings: list[str] = []
    duration_ratio = mb.utterance_duration_s / ma.utterance_duration_s
    if max(duration_ratio, 1.0 / duration_ratio) > 1.35:
        warnings.append(
            _warning(
                "PAIR_DURATION_MISMATCH",
                f"{a.label} and {b.label} differ in duration by more than 35%; text or trimming may differ.",
            )
        )
    level_difference = mb.active_rms_median_dbfs - ma.active_rms_median_dbfs
    if abs(level_difference) > 6.0:
        warnings.append(
            _warning(
                "PAIR_LEVEL_MISMATCH",
                f"Recorded levels differ by {abs(level_difference):.1f} dB; gain, distance, or AGC may confound energy.",
            )
        )
    pitch_delta = _semitone_delta(mb.f0_median_hz, ma.f0_median_hz)
    pitch_span_delta = (
        mb.f0_iqr_st - ma.f0_iqr_st
        if mb.f0_iqr_st is not None and ma.f0_iqr_st is not None
        else None
    )
    return DeliveryDelta(
        a=a.id,
        b=b.id,
        duration_s=mb.utterance_duration_s - ma.utterance_duration_s,
        duration_percent=(duration_ratio - 1.0) * 100.0,
        duration_ratio=duration_ratio,
        pause_total_s=mb.pause_total_s - ma.pause_total_s,
        pause_fraction_percentage_points=(mb.pause_fraction - ma.pause_fraction)
        * 100.0,
        f0_median_st=pitch_delta,
        f0_iqr_st=pitch_span_delta,
        active_rms_median_db=level_difference,
        active_rms_dynamic_10_90_db=(
            mb.active_rms_dynamic_10_90_db - ma.active_rms_dynamic_10_90_db
        ),
        minimum_pitch_coverage=min(ma.f0_coverage, mb.f0_coverage),
        pitch_delta_available=pitch_delta is not None,
        pause_detection_stable=_pause_stable(a) and _pause_stable(b),
        warnings=tuple(warnings),
    )


def compare_takes(
    paths: Iterable[Path],
    *,
    labels: Iterable[str] | None = None,
    text: str | None = None,
    settings: AnalysisSettings | None = None,
    synthetic_demo: bool = False,
    capture_metadata: Iterable[CaptureMetadata | None] | None = None,
) -> Comparison:
    """Analyse two to four recordings and compare every unordered pair."""

    path_list = list(paths)
    if not 2 <= len(path_list) <= 4:
        raise AudioInputError(
            f"Expected 2 to 4 matched-text WAV files, received {len(path_list)}."
        )
    resolved_paths = [path.expanduser().resolve() for path in path_list]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise AudioInputError("Each input path must refer to a different WAV file.")

    label_list = list(labels) if labels is not None else []
    if label_list and len(label_list) != len(path_list):
        raise AudioInputError(
            f"Received {len(label_list)} labels for {len(path_list)} recordings. "
            "Repeat --label once per input."
        )
    if not label_list:
        label_list = [
            path.stem.replace("_", " ").replace("-", " ").title() for path in path_list
        ]
    if len(set(label_list)) != len(label_list):
        raise AudioInputError("Take labels must be unique.")

    resolved_settings = settings or AnalysisSettings()
    capture_list = list(capture_metadata) if capture_metadata is not None else []
    if capture_list and len(capture_list) != len(path_list):
        raise AudioInputError(
            f"Received capture metadata for {len(capture_list)} of {len(path_list)} recordings."
        )
    if not capture_list:
        capture_list = [None] * len(path_list)
    takes = [
        analyse_take(
            path,
            take_id=f"take_{index}",
            label=label,
            settings=resolved_settings,
            capture_metadata=capture,
        )
        for index, (path, label, capture) in enumerate(
            zip(path_list, label_list, capture_list, strict=True), start=1
        )
    ]

    reliable_medians_hz = [
        take.metrics.f0_median_hz
        for take in takes
        if take.metrics.f0_median_hz is not None
    ]
    pooled_f0_hz = (
        float(np.exp(np.median(np.log(reliable_medians_hz))))
        if reliable_medians_hz
        else None
    )
    pooled_rms_dbfs = float(
        np.median([take.metrics.active_rms_median_dbfs for take in takes])
    )
    pairs = [_pair_delta(a, b) for a, b in combinations(takes, 2)]

    warnings = [warning for take in takes for warning in take.warnings]
    warnings.extend(warning for pair in pairs for warning in pair.warnings)
    if len(reliable_medians_hz) < 2:
        warnings.append(
            _warning(
                "PITCH_REFERENCE_INCOMPLETE",
                "Fewer than two takes have reliable pitch; pitch comparisons are incomplete.",
            )
        )
    warnings.append(
        _warning(
            "RECORDING_PROTOCOL",
            "Recorded-level deltas require back-to-back takes with identical device, gain, distance, room, and disabled auto-gain/voice enhancement.",
        )
    )

    return Comparison(
        settings=resolved_settings,
        takes=takes,
        pairs=pairs,
        pooled_f0_hz=pooled_f0_hz,
        pooled_rms_dbfs=pooled_rms_dbfs,
        text=text,
        warnings=warnings,
        synthetic_demo=synthetic_demo,
    )
