"""Static, publication-grade Matplotlib rendering."""

from __future__ import annotations

from pathlib import Path
from textwrap import shorten
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from prosodiff.errors import ProsodiffError
from prosodiff.models import Comparison, TakeAnalysis


_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
_LINESTYLES = ("-", (0, (6, 2)), "-.", ":")
_MARKERS = ("o", "s", "D", "^")
_TEXT = "#202124"
_MUTED = "#667085"
_GRID = "#D8DEE8"
_BACKGROUND = "#FCFCFD"
_PANEL = "#F4F7FA"


def _style_axis(axis: Axes, *, grid_axis: str = "both") -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(_GRID)
    axis.spines[["left", "bottom"]].set_linewidth(0.6)
    axis.tick_params(colors=_MUTED, labelsize=6.8, length=2.5, width=0.5)
    axis.grid(True, axis=grid_axis, color=_GRID, linewidth=0.45, alpha=0.75)
    axis.set_axisbelow(True)


def _finite_range(
    series: Sequence[np.ndarray], *, minimum_span: float
) -> tuple[float, float]:
    finite_parts = [
        item[np.isfinite(item)] for item in series if np.any(np.isfinite(item))
    ]
    if not finite_parts:
        return -1.0, 1.0
    values = np.concatenate(finite_parts)
    low, high = np.percentile(values, [2, 98])
    center = (low + high) / 2.0
    span = max(float(high - low), minimum_span)
    return center - span * 0.62, center + span * 0.62


def _pitch_contour(take: TakeAnalysis, pooled_f0_hz: float | None) -> np.ndarray:
    if pooled_f0_hz is None:
        return np.full_like(take.f0_hz, np.nan)
    return 12.0 * np.log2(take.f0_hz / pooled_f0_hz)


def _pause_intervals(
    take: TakeAnalysis, comparison: Comparison
) -> list[tuple[float, float]]:
    mask = take.pause_mask
    if not np.any(mask):
        return []
    times = take.energy_time_s
    hop_s = comparison.settings.hop_length / comparison.settings.sample_rate_hz
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [
        (
            max(0.0, float(times[start] - hop_s / 2.0)),
            float(times[stop - 1] + hop_s / 2.0),
        )
        for start, stop in zip(starts, stops, strict=True)
    ]


def _metric_specs(
    comparison: Comparison,
) -> list[tuple[str, list[float | None], Callable[[float], str]]]:
    takes = comparison.takes
    pooled_f0 = comparison.pooled_f0_hz
    median_pitch = [
        (
            12.0 * np.log2(take.metrics.f0_median_hz / pooled_f0)
            if take.metrics.f0_median_hz is not None and pooled_f0 is not None
            else None
        )
        for take in takes
    ]
    return [
        (
            "Duration · s",
            [take.metrics.utterance_duration_s for take in takes],
            lambda value: f"{value:.2f}",
        ),
        ("Median pitch · st", median_pitch, lambda value: f"{value:+.1f}"),
        (
            "Pitch span · IQR st",
            [take.metrics.f0_iqr_st for take in takes],
            lambda value: f"{value:.1f}",
        ),
        (
            "Recorded level† · dBFS",
            [take.metrics.active_rms_median_dbfs for take in takes],
            lambda value: f"{value:.1f}",
        ),
        (
            "Energy dynamics · P90–P10 dB",
            [take.metrics.active_rms_dynamic_10_90_db for take in takes],
            lambda value: f"{value:.1f}",
        ),
        (
            "Pause share · %",
            [100.0 * take.metrics.pause_fraction for take in takes],
            lambda value: f"{value:.1f}",
        ),
    ]


def _draw_metric(
    axis: Axes,
    comparison: Comparison,
    title: str,
    values: list[float | None],
    formatter: Callable[[float], str],
) -> None:
    finite = np.asarray([value for value in values if value is not None], dtype=float)
    axis.set_facecolor(_BACKGROUND)
    axis.set_title(
        title, loc="left", fontsize=7.4, fontweight="bold", color=_TEXT, pad=4
    )
    axis.set_yticks([])
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(_GRID)
    axis.spines["bottom"].set_linewidth(0.55)
    axis.tick_params(axis="x", colors=_MUTED, labelsize=6.0, length=2.2, width=0.5)
    axis.grid(True, axis="x", color=_GRID, linewidth=0.4, alpha=0.7)
    axis.set_axisbelow(True)

    if finite.size:
        low = float(np.min(finite))
        high = float(np.max(finite))
        span = max(high - low, max(abs(low), abs(high), 1.0) * 0.15)
        axis.set_xlim(low - 0.22 * span, high + 0.42 * span)
        axis.axvline(
            values[0] if values[0] is not None else low,
            color=_MUTED,
            linewidth=0.7,
            alpha=0.35,
        )
    else:
        axis.set_xlim(0.0, 1.0)

    y_positions = np.linspace(0.77, 0.23, len(values))
    for index, (value, y) in enumerate(zip(values, y_positions, strict=True)):
        if value is None:
            axis.text(
                0.02,
                y,
                f"T{index + 1} N/A",
                transform=axis.transAxes,
                fontsize=6.2,
                color=_MUTED,
                va="center",
            )
            continue
        axis.scatter(
            [value],
            [y],
            s=26,
            color=_COLORS[index],
            marker=_MARKERS[index],
            edgecolor=_BACKGROUND,
            linewidth=0.7,
            zorder=3,
        )
        axis.annotate(
            f"T{index + 1} {formatter(value)}",
            (value, y),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=5.9,
            color=_TEXT,
        )
    axis.set_ylim(0.05, 0.95)


def _observed_summary(comparison: Comparison) -> str:
    reference = comparison.takes[0]
    candidates: list[tuple[float, str]] = []
    for take in comparison.takes[1:]:
        metrics = take.metrics
        ref = reference.metrics
        duration_pct = 100.0 * (
            metrics.utterance_duration_s / ref.utterance_duration_s - 1.0
        )
        candidates.append(
            (
                abs(duration_pct) / 10.0,
                f"{take.label} measured {abs(duration_pct):.0f}% "
                f"{'longer' if duration_pct >= 0 else 'shorter'} than {reference.label}",
            )
        )
        if metrics.f0_median_hz is not None and ref.f0_median_hz is not None:
            pitch_delta = 12.0 * np.log2(metrics.f0_median_hz / ref.f0_median_hz)
            candidates.append(
                (
                    abs(pitch_delta),
                    f"{take.label} median pitch measured {abs(pitch_delta):.1f} st "
                    f"{'higher' if pitch_delta >= 0 else 'lower'}",
                )
            )
        if metrics.f0_iqr_st is not None and ref.f0_iqr_st is not None:
            span_delta = metrics.f0_iqr_st - ref.f0_iqr_st
            candidates.append(
                (
                    abs(span_delta),
                    f"{take.label} pitch span measured {abs(span_delta):.1f} st "
                    f"{'wider' if span_delta >= 0 else 'narrower'}",
                )
            )
        pause_delta = 100.0 * (metrics.pause_fraction - ref.pause_fraction)
        candidates.append(
            (
                abs(pause_delta) / 5.0,
                f"{take.label} pause share measured {abs(pause_delta):.1f} points "
                f"{'higher' if pause_delta >= 0 else 'lower'}",
            )
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    for score, statement in candidates:
        if score < 0.5 or statement in selected:
            continue
        selected.append(statement)
        if len(selected) == 2:
            break
    if not selected:
        return "Observed: no large take-level differences under the displayed measures."
    return "Observed: " + "; ".join(selected) + "."


def _figure(comparison: Comparison) -> Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": _TEXT,
            "axes.labelcolor": _TEXT,
            "axes.titlecolor": _TEXT,
            "figure.facecolor": _BACKGROUND,
            "savefig.facecolor": _BACKGROUND,
        }
    )
    figure = plt.figure(figsize=(7.2, 9.0), dpi=300, facecolor=_BACKGROUND)
    grid = figure.add_gridspec(
        100,
        12,
        left=0.095,
        right=0.975,
        top=0.97,
        bottom=0.055,
        hspace=0.0,
        wspace=0.55,
    )

    figure.text(
        0.075,
        0.964,
        "PROSODIFF · EXPLICIT DELIVERY-ATTRIBUTE DELTAS",
        fontsize=7.3,
        fontweight="bold",
        color=_MUTED,
    )
    figure.text(
        0.075,
        0.928,
        "Same words. Different delivery.",
        fontsize=22,
        fontweight="bold",
        color=_TEXT,
    )
    if comparison.text:
        sentence = f"“{shorten(comparison.text, width=82, placeholder='…')}”"
    else:
        sentence = "Matched-text assumption supplied by the user"
    figure.text(0.075, 0.897, sentence, fontsize=10.2, color=_TEXT)
    demo_note = "synthetic demonstration · " if comparison.synthetic_demo else ""
    figure.text(
        0.075,
        0.870,
        f"{demo_note}{len(comparison.takes)} takes · raw time · first take is the delta reference",
        fontsize=7.2,
        color=_MUTED,
    )

    legend_y = 0.840
    x_positions = np.linspace(0.075, 0.79, len(comparison.takes))
    marker_glyphs = ("●", "■", "◆", "▲")
    for index, (take, x) in enumerate(zip(comparison.takes, x_positions, strict=True)):
        display_label = shorten(take.label, width=18, placeholder="…")
        figure.text(
            x,
            legend_y,
            f"{marker_glyphs[index]}  T{index + 1} · {display_label}",
            fontsize=7.3,
            color=_COLORS[index],
            fontweight="bold",
        )
        figure.text(
            x,
            legend_y - 0.016,
            f"{take.metrics.utterance_duration_s:.2f} s",
            fontsize=6.5,
            color=_MUTED,
        )

    max_duration = max(take.metrics.utterance_duration_s for take in comparison.takes)
    pitch_axis = figure.add_subplot(grid[21:39, :])
    pitch_series = [
        _pitch_contour(take, comparison.pooled_f0_hz) for take in comparison.takes
    ]
    for index, (take, values) in enumerate(
        zip(comparison.takes, pitch_series, strict=True)
    ):
        # Probability controls emphasis only. Every finite Viterbi-decoded
        # voiced estimate remains in the analysis and the pale contour.
        pitch_axis.plot(
            take.pitch_time_s,
            values,
            color=_COLORS[index],
            linewidth=0.95,
            linestyle=_LINESTYLES[index],
            alpha=0.28,
            label=f"T{index + 1} · {take.label}",
        )
        emphasized = np.where(
            take.f0_probability
            >= comparison.settings.voiced_probability_threshold,
            values,
            np.nan,
        )
        pitch_axis.plot(
            take.pitch_time_s,
            emphasized,
            color=_COLORS[index],
            linewidth=1.3,
            linestyle=_LINESTYLES[index],
        )
    pitch_axis.axhline(0.0, color=_MUTED, linewidth=0.65, alpha=0.6)
    pitch_axis.set_xlim(0.0, max_duration)
    pitch_axis.set_ylim(*_finite_range(pitch_series, minimum_span=5.0))
    pitch_axis.set_ylabel("Pitch\n(st rel. pooled median)", fontsize=7.3)
    pitch_axis.set_xticklabels([])
    pitch_axis.set_title(
        "A · Raw-time delivery contours",
        loc="left",
        fontsize=9.2,
        fontweight="bold",
        pad=8,
    )
    pitch_axis.text(
        1.0,
        1.055,
        f"pale = decoded voiced below p={comparison.settings.voiced_probability_threshold:.2f} · gaps = unvoiced or pause-excluded",
        transform=pitch_axis.transAxes,
        ha="right",
        fontsize=6.2,
        color=_MUTED,
    )
    _style_axis(pitch_axis)
    if comparison.pooled_f0_hz is None:
        pitch_axis.text(
            0.5,
            0.5,
            "Pitch unavailable: too few decoded voiced frames",
            transform=pitch_axis.transAxes,
            ha="center",
            va="center",
            fontsize=7.0,
            color=_MUTED,
        )

    energy_axis = figure.add_subplot(grid[42:56, :], sharex=pitch_axis)
    energy_series = [
        np.where(
            (take.rms_dbfs >= take.metrics.pause_threshold_dbfs) & (~take.pause_mask),
            take.rms_dbfs - comparison.pooled_rms_dbfs,
            np.nan,
        )
        for take in comparison.takes
    ]
    for index, (take, values) in enumerate(
        zip(comparison.takes, energy_series, strict=True)
    ):
        energy_axis.plot(
            take.energy_time_s,
            values,
            color=_COLORS[index],
            linewidth=1.05,
            linestyle=_LINESTYLES[index],
        )
    energy_axis.axhline(0.0, color=_MUTED, linewidth=0.65, alpha=0.6)
    energy_axis.set_ylim(*_finite_range(energy_series, minimum_span=14.0))
    energy_axis.set_ylabel("RMS contour†\n(dB rel. pooled median)", fontsize=7.3)
    energy_axis.set_xticklabels([])
    energy_axis.text(
        1.0,
        1.04,
        "† microphone / gain / AGC sensitive",
        transform=energy_axis.transAxes,
        ha="right",
        fontsize=6.2,
        color=_MUTED,
    )
    _style_axis(energy_axis)

    pause_axis = figure.add_subplot(grid[59:67, :], sharex=pitch_axis)
    for index, take in enumerate(comparison.takes):
        y = len(comparison.takes) - index - 1
        pause_axis.broken_barh(
            [(0.0, take.metrics.utterance_duration_s)],
            (y - 0.28, 0.56),
            facecolors=_COLORS[index],
            alpha=0.24,
            linewidth=0,
        )
        for start, stop in _pause_intervals(take, comparison):
            pause_axis.broken_barh(
                [(start, stop - start)],
                (y - 0.28, 0.56),
                facecolors=_BACKGROUND,
                edgecolors=_GRID,
                linewidth=0.5,
            )
    pause_axis.set_yticks(range(len(comparison.takes)))
    pause_axis.set_yticklabels(
        [f"T{index}" for index in range(len(comparison.takes), 0, -1)],
        fontsize=6.5,
    )
    pause_axis.set_xlabel(
        "Raw time from detected utterance onset (s) · not phonetic alignment",
        fontsize=6.6,
        labelpad=1,
    )
    pause_axis.set_title(
        "Energy-defined pause ribbons",
        loc="left",
        fontsize=7.5,
        fontweight="bold",
        pad=3,
    )
    pause_axis.spines[["top", "right", "left"]].set_visible(False)
    pause_axis.spines["bottom"].set_color(_GRID)
    pause_axis.tick_params(colors=_MUTED, labelsize=6.4, length=2.2, width=0.5)
    pause_axis.grid(True, axis="x", color=_GRID, linewidth=0.4, alpha=0.65)

    figure.text(
        0.075,
        0.315,
        "B · Descriptive measures — not a quality score",
        fontsize=9.2,
        fontweight="bold",
        color=_TEXT,
    )
    metric_grid = grid[74:90, :].subgridspec(2, 3, hspace=0.78, wspace=0.46)
    for metric_index, (title, values, formatter) in enumerate(
        _metric_specs(comparison)
    ):
        axis = figure.add_subplot(metric_grid[metric_index // 3, metric_index % 3])
        _draw_metric(axis, comparison, title, values, formatter)

    summary = _observed_summary(comparison)
    figure.text(
        0.075,
        0.086,
        summary,
        fontsize=7.4,
        color=_TEXT,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": _PANEL,
            "edgecolor": _GRID,
            "linewidth": 0.6,
        },
    )
    figure.text(
        0.075,
        0.058,
        f"F0: pYIN Viterbi voicing; pale < p {comparison.settings.voiced_probability_threshold:.2f} · pitch span: IQR · pauses: nominal ≥150 ms (11.6 ms hop) · no time warping.",
        fontsize=5.8,
        color=_MUTED,
    )
    figure.text(
        0.075,
        0.041,
        "Descriptive only. Same text/speaker/session cannot be verified; mic distance, gain, noise, auto-processing, and tracker errors can affect results.",
        fontsize=5.8,
        color=_MUTED,
    )
    return figure


def render_comparison(comparison: Comparison, output: Path) -> Path:
    """Render an exact 4:5 PNG, SVG, or PDF without GUI state."""

    destination = output.expanduser().resolve()
    if destination.suffix.lower() not in {".png", ".svg", ".pdf"}:
        raise ProsodiffError("Output must end in .png, .svg, or .pdf.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = _figure(comparison)
    save_kwargs: dict[str, object] = {"facecolor": figure.get_facecolor()}
    if destination.suffix.lower() == ".png":
        save_kwargs["dpi"] = 300
    figure.savefig(destination, **save_kwargs)
    plt.close(figure)
    return destination
