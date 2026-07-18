"""Typed data models for Prosodiff analyses and JSON output."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None and np.isfinite(value) else None


def _float_array(values: FloatArray) -> list[float | None]:
    """Serialize finite values while mapping NaN/inf to JSON ``null``."""

    return [_rounded(float(value)) for value in values]


@dataclass(frozen=True)
class CaptureMetadata:
    """Privacy-preserving browser capture provenance reported by the local UI."""

    source: str
    client_reported: bool = False
    encoded_sample_rate_hz: int | None = None
    encoded_channels: int | None = None
    encoded_duration_s: float | None = None
    track_sample_rate_hz: int | None = None
    track_channel_count: int | None = None
    track_sample_size_bits: int | None = None
    track_latency_s: float | None = None
    echo_cancellation: bool | None = None
    noise_suppression: bool | None = None
    auto_gain_control: bool | None = None
    same_device_as_reference: bool | None = None
    constraints_fallback: bool = False
    client_active_rms_dbfs: float | None = None
    client_peak_dbfs: float | None = None
    client_clipped_fraction: float | None = None
    client_qa_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return capture provenance without browser device identifiers."""

        return {
            "source": self.source,
            "client_reported": self.client_reported,
            "encoded_wav": {
                "sample_rate_hz": self.encoded_sample_rate_hz,
                "channels": self.encoded_channels,
                "duration_s": _rounded(self.encoded_duration_s),
            },
            "track_settings": {
                "sample_rate_hz": self.track_sample_rate_hz,
                "channel_count": self.track_channel_count,
                "sample_size_bits": self.track_sample_size_bits,
                "latency_s": _rounded(self.track_latency_s),
                "echo_cancellation": self.echo_cancellation,
                "noise_suppression": self.noise_suppression,
                "auto_gain_control": self.auto_gain_control,
                "same_device_as_reference": self.same_device_as_reference,
            },
            "constraints_fallback": self.constraints_fallback,
            "client_qa": {
                "active_rms_estimate_dbfs": _rounded(self.client_active_rms_dbfs),
                "peak_dbfs": _rounded(self.client_peak_dbfs),
                "clipped_sample_fraction": _rounded(self.client_clipped_fraction),
                "codes": list(self.client_qa_codes),
            },
        }


@dataclass(frozen=True)
class AnalysisSettings:
    """Fixed numerical settings recorded with every result."""

    sample_rate_hz: int = 22_050
    pitch_frame_length: int = 2_048
    energy_frame_length: int = 1_024
    hop_length: int = 256
    fmin_hz: float = 50.0
    fmax_hz: float = 600.0
    voiced_probability_threshold: float = 0.80
    minimum_pitch_frames: int = 20
    minimum_pitch_coverage: float = 0.25
    minimum_pause_ms: float = 150.0
    maximum_noise_burst_ms: float = 50.0
    edge_pad_ms: float = 50.0
    pause_sensitivity_db: float = 3.0

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-compatible representation."""

        return asdict(self)


@dataclass(frozen=True)
class TakeMetrics:
    """Take-level descriptive acoustic attributes."""

    file_duration_s: float
    peak_abs: float
    clipping_fraction: float
    utterance_start_s: float
    utterance_end_s: float
    utterance_duration_s: float
    leading_silence_s: float
    trailing_silence_s: float
    nonpause_time_s: float
    pause_count: int
    pause_total_s: float
    pause_fraction: float
    pause_median_s: float
    pause_max_s: float
    pause_threshold_dbfs: float
    pause_fraction_threshold_minus_3db: float
    pause_fraction_threshold_plus_3db: float
    pitch_eligible_frames: int
    pitch_valid_frames: int
    pitch_high_probability_frames: int
    f0_coverage: float
    high_probability_fraction: float
    voiced_probability_median: float | None
    voiced_probability_p10: float | None
    voiced_probability_p90: float | None
    f0_median_hz: float | None
    f0_iqr_st: float | None
    active_rms_median_dbfs: float
    active_rms_iqr_db: float
    active_rms_dynamic_10_90_db: float
    utterance_rms_dbfs: float


@dataclass
class TakeAnalysis:
    """Metrics plus frame-level arrays used to draw and audit the report."""

    id: str
    label: str
    source: Path
    original_sample_rate_hz: int
    original_channels: int
    metrics: TakeMetrics
    pitch_time_s: FloatArray
    pitch_time_norm: FloatArray
    f0_hz: FloatArray
    f0_candidate_hz: FloatArray
    f0_voiced_flag: BoolArray
    pitch_eligible_mask: BoolArray
    f0_probability: FloatArray
    energy_time_s: FloatArray
    energy_time_norm: FloatArray
    rms_dbfs: FloatArray
    pause_mask: BoolArray
    capture_metadata: CaptureMetadata | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(
        self,
        *,
        pooled_f0_hz: float | None,
        pooled_rms_dbfs: float,
    ) -> dict[str, Any]:
        """Return the durable take record including auditable contours."""

        metrics = self.metrics
        if pooled_f0_hz is None or metrics.f0_median_hz is None:
            median_st_rel = None
            f0_st_rel = np.full_like(self.f0_hz, np.nan)
        else:
            median_st_rel = 12.0 * np.log2(metrics.f0_median_hz / pooled_f0_hz)
            f0_st_rel = 12.0 * np.log2(self.f0_hz / pooled_f0_hz)

        finite_st = f0_st_rel[np.isfinite(f0_st_rel)]
        p10_st = float(np.percentile(finite_st, 10)) if finite_st.size else None
        p90_st = float(np.percentile(finite_st, 90)) if finite_st.size else None
        rms_relative_db = self.rms_dbfs - pooled_rms_dbfs

        return {
            "id": self.id,
            "label": self.label,
            "path": self.source.name,
            "input": {
                "original_sr_hz": self.original_sample_rate_hz,
                "channels": self.original_channels,
                "file_duration_s": _rounded(metrics.file_duration_s),
                "peak_abs": _rounded(metrics.peak_abs),
                "clipped_sample_fraction": _rounded(metrics.clipping_fraction),
                "capture": (
                    self.capture_metadata.to_dict()
                    if self.capture_metadata is not None
                    else None
                ),
            },
            "utterance": {
                "start_s": _rounded(metrics.utterance_start_s),
                "end_s": _rounded(metrics.utterance_end_s),
                "duration_s": _rounded(metrics.utterance_duration_s),
                "leading_silence_s": _rounded(metrics.leading_silence_s),
                "trailing_silence_s": _rounded(metrics.trailing_silence_s),
                "nonpause_time_s": _rounded(metrics.nonpause_time_s),
                "pause_count": metrics.pause_count,
                "pause_total_s": _rounded(metrics.pause_total_s),
                "pause_fraction": _rounded(metrics.pause_fraction),
                "pause_median_s": _rounded(metrics.pause_median_s),
                "pause_max_s": _rounded(metrics.pause_max_s),
                "pause_threshold_dbfs": _rounded(metrics.pause_threshold_dbfs),
                "pause_fraction_at_threshold_minus_3db": _rounded(
                    metrics.pause_fraction_threshold_minus_3db
                ),
                "pause_fraction_at_threshold_plus_3db": _rounded(
                    metrics.pause_fraction_threshold_plus_3db
                ),
            },
            "pitch": {
                "selection_method": "pyin_viterbi_voiced_flag",
                "eligible_frames": metrics.pitch_eligible_frames,
                "valid_frames": metrics.pitch_valid_frames,
                "decoded_voiced_frames": metrics.pitch_valid_frames,
                "coverage": _rounded(metrics.f0_coverage),
                "decoded_voiced_coverage": _rounded(metrics.f0_coverage),
                "high_probability_frames": metrics.pitch_high_probability_frames,
                "high_probability_fraction": _rounded(
                    metrics.high_probability_fraction
                ),
                "voiced_probability_median": _rounded(
                    metrics.voiced_probability_median
                ),
                "voiced_probability_p10": _rounded(metrics.voiced_probability_p10),
                "voiced_probability_p90": _rounded(metrics.voiced_probability_p90),
                "summary_available": metrics.f0_median_hz is not None,
                "f0_median_hz": _rounded(metrics.f0_median_hz),
                "f0_median_st_rel": _rounded(median_st_rel),
                "f0_iqr_st": _rounded(metrics.f0_iqr_st),
                "f0_p10_st_rel": _rounded(p10_st),
                "f0_p90_st_rel": _rounded(p90_st),
            },
            "energy": {
                "active_rms_median_dbfs": _rounded(metrics.active_rms_median_dbfs),
                "active_rms_iqr_db": _rounded(metrics.active_rms_iqr_db),
                "active_rms_dynamic_10_90_db": _rounded(
                    metrics.active_rms_dynamic_10_90_db
                ),
                "utterance_rms_dbfs": _rounded(metrics.utterance_rms_dbfs),
            },
            "contours": {
                "pitch_time_s": _float_array(self.pitch_time_s),
                "pitch_time_norm": _float_array(self.pitch_time_norm),
                "f0_hz": _float_array(self.f0_hz),
                "pyin_candidate_f0_hz": _float_array(self.f0_candidate_hz),
                "pyin_voiced_flag": [bool(value) for value in self.f0_voiced_flag],
                "pitch_eligible_mask": [
                    bool(value) for value in self.pitch_eligible_mask
                ],
                "f0_st_rel": _float_array(f0_st_rel),
                "voiced_probability": _float_array(self.f0_probability),
                "energy_time_s": _float_array(self.energy_time_s),
                "energy_time_norm": _float_array(self.energy_time_norm),
                "rms_dbfs": _float_array(self.rms_dbfs),
                "rms_relative_db": _float_array(rms_relative_db),
                "pause_mask": [bool(value) for value in self.pause_mask],
            },
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class DeliveryDelta:
    """Explicit attribute deltas for one unordered pair, always B minus A."""

    a: str
    b: str
    duration_s: float
    duration_percent: float
    duration_ratio: float
    pause_total_s: float
    pause_fraction_percentage_points: float
    f0_median_st: float | None
    f0_iqr_st: float | None
    active_rms_median_db: float
    active_rms_dynamic_10_90_db: float
    minimum_pitch_coverage: float
    pitch_delta_available: bool
    pause_detection_stable: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a nested, JSON-compatible B-minus-A record."""

        return {
            "a": self.a,
            "b": self.b,
            "direction": "b_minus_a",
            "delta": {
                "duration_s": _rounded(self.duration_s),
                "duration_percent": _rounded(self.duration_percent),
                "pause_total_s": _rounded(self.pause_total_s),
                "pause_fraction_percentage_points": _rounded(
                    self.pause_fraction_percentage_points
                ),
                "f0_median_st": _rounded(self.f0_median_st),
                "f0_iqr_st": _rounded(self.f0_iqr_st),
                "active_rms_median_db": _rounded(self.active_rms_median_db),
                "active_rms_dynamic_10_90_db": _rounded(
                    self.active_rms_dynamic_10_90_db
                ),
            },
            "ratio": {"duration": _rounded(self.duration_ratio)},
            "reliability": {
                "minimum_pitch_coverage": _rounded(self.minimum_pitch_coverage),
                "pitch_delta_available": self.pitch_delta_available,
                "pause_detection_stable": self.pause_detection_stable,
            },
            "warnings": list(self.warnings),
        }


@dataclass
class Comparison:
    """Complete Prosodiff result used by renderers and serializers."""

    settings: AnalysisSettings
    takes: list[TakeAnalysis]
    pairs: list[DeliveryDelta]
    pooled_f0_hz: float | None
    pooled_rms_dbfs: float
    text: str | None = None
    warnings: list[str] = field(default_factory=list)
    synthetic_demo: bool = False

    def to_dict(self, *, version: str) -> dict[str, Any]:
        """Return the versioned explicit delivery-attribute delta schema."""

        return {
            "schema": "prosodiff.explicit-delivery-attribute-delta",
            "schema_version": "0.2.0",
            "tool_version": version,
            "synthetic_demo": self.synthetic_demo,
            "text": self.text,
            "assumptions": [
                "same speaker",
                "same text",
                "same session, room, device, gain, and microphone distance",
            ],
            "analysis": {
                "sample_rate_hz": self.settings.sample_rate_hz,
                "pitch": {
                    "frame_length": self.settings.pitch_frame_length,
                    "hop_length": self.settings.hop_length,
                    "fmin_hz": self.settings.fmin_hz,
                    "fmax_hz": self.settings.fmax_hz,
                    "frame_selection": "eligible_and_pyin_viterbi_voiced",
                    "voiced_probability_role": "diagnostic_only",
                    "voiced_probability_diagnostic_threshold": (
                        self.settings.voiced_probability_threshold
                    ),
                    "minimum_frames_for_summary": (
                        self.settings.minimum_pitch_frames
                    ),
                    "minimum_coverage_for_summary": (
                        self.settings.minimum_pitch_coverage
                    ),
                    "shared_reference_hz": _rounded(self.pooled_f0_hz),
                },
                "energy": {
                    "frame_length": self.settings.energy_frame_length,
                    "hop_length": self.settings.hop_length,
                    "pooled_active_median_dbfs": _rounded(self.pooled_rms_dbfs),
                },
                "pause": {
                    "minimum_pause_ms": self.settings.minimum_pause_ms,
                    "maximum_noise_burst_ms": (self.settings.maximum_noise_burst_ms),
                    "sensitivity_db": self.settings.pause_sensitivity_db,
                },
            },
            "takes": [
                take.to_dict(
                    pooled_f0_hz=self.pooled_f0_hz,
                    pooled_rms_dbfs=self.pooled_rms_dbfs,
                )
                for take in self.takes
            ],
            "pairs": [pair.to_dict() for pair in self.pairs],
            "warnings": list(self.warnings),
            "interpretation": (
                "Descriptive acoustic attributes only; not emotion labels, "
                "perceptual judgments, or a quality score."
            ),
        }
