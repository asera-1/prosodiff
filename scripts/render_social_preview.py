from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


WIDTH_PX = 1280
HEIGHT_PX = 640
DPI = 100
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
LINESTYLES = ("-", "--", "-.", ":")
MARKERS = ("●", "■", "◆", "▲")


def _load_report(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        report: dict[str, Any] = json.load(handle)
    if not report.get("takes"):
        raise ValueError(f"No takes found in {path}.")
    return report


def render_social_preview(report_path: Path, output_path: Path) -> Path:
    """Render the 1280 x 640 GitHub social-preview image."""

    report = _load_report(report_path)
    takes = report["takes"][:4]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": "#202124",
            "axes.labelcolor": "#667085",
            "xtick.color": "#667085",
            "ytick.color": "#667085",
        }
    )

    figure = plt.figure(
        figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI),
        dpi=DPI,
        facecolor="#F7F9FC",
    )

    background = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    background.set_axis_off()

    accent_width = 0.89 / len(COLORS)
    for index, color in enumerate(COLORS):
        background.add_patch(
            FancyBboxPatch(
                (0.055 + index * accent_width, 0.914),
                accent_width - 0.006,
                0.008,
                boxstyle="round,pad=0,rounding_size=0.004",
                transform=figure.transFigure,
                linewidth=0,
                facecolor=color,
            )
        )

    background.add_patch(
        FancyBboxPatch(
            (0.555, 0.105),
            0.395,
            0.765,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            transform=figure.transFigure,
            linewidth=1.2,
            edgecolor="#D8DEE8",
            facecolor="#FFFFFF",
        )
    )

    figure.text(
        0.06,
        0.855,
        "PROSODIFF · EXPLICIT DELIVERY-ATTRIBUTE DELTAS",
        fontsize=14,
        fontweight="bold",
        color="#667085",
    )
    figure.text(
        0.06,
        0.735,
        "Same words.\nDifferent delivery.",
        fontsize=43,
        fontweight="bold",
        color="#202124",
        linespacing=0.98,
        va="top",
    )
    figure.text(
        0.06,
        0.405,
        "Audit-friendly prosody deltas\nfor matched speech recordings.",
        fontsize=19,
        color="#344054",
        linespacing=1.35,
        va="top",
    )

    background.add_patch(
        FancyBboxPatch(
            (0.06, 0.18),
            0.295,
            0.085,
            boxstyle="round,pad=0.006,rounding_size=0.014",
            transform=figure.transFigure,
            linewidth=0,
            facecolor="#202124",
        )
    )
    figure.text(
        0.078,
        0.222,
        "uvx prosodiff demo",
        fontsize=16,
        fontfamily="DejaVu Sans Mono",
        color="#FFFFFF",
        va="center",
    )
    figure.text(
        0.06,
        0.10,
        "Open source  ·  local-first  ·  PNG + JSON  ·  v0.2.0",
        fontsize=13,
        color="#667085",
    )

    figure.text(
        0.59,
        0.81,
        "Four takes · one sentence",
        fontsize=17,
        fontweight="bold",
        color="#202124",
    )
    sentence = str(report.get("text") or "Matched delivery comparison")
    figure.text(
        0.59,
        0.765,
        f"“{sentence}”",
        fontsize=11.5,
        color="#667085",
    )

    legend_positions = ((0.59, 0.705), (0.75, 0.705), (0.59, 0.66), (0.75, 0.66))
    for index, take in enumerate(takes):
        x_position, y_position = legend_positions[index]
        figure.text(
            x_position,
            y_position,
            MARKERS[index],
            fontsize=13,
            color=COLORS[index],
            va="center",
        )
        figure.text(
            x_position + 0.02,
            y_position,
            str(take["label"]),
            fontsize=10.5,
            color="#344054",
            va="center",
        )

    axis = figure.add_axes([0.59, 0.275, 0.32, 0.33], facecolor="#FFFFFF")
    maximum_time = 0.0
    pitch_values: list[np.ndarray] = []
    for index, take in enumerate(takes):
        contours = take["contours"]
        times = np.asarray(contours["pitch_time_s"], dtype=float)
        pitch = np.asarray(contours["f0_st_rel"], dtype=float)
        eligible = np.asarray(contours["pitch_eligible_mask"], dtype=bool)
        voiced = np.asarray(contours["pyin_voiced_flag"], dtype=bool)
        mask = eligible & voiced & np.isfinite(pitch)
        visible_pitch = np.where(mask, pitch, np.nan)
        pitch_values.append(visible_pitch[np.isfinite(visible_pitch)])
        maximum_time = max(maximum_time, float(times[-1]))
        axis.plot(
            times,
            visible_pitch,
            color=COLORS[index],
            linestyle=LINESTYLES[index],
            linewidth=2.15,
            solid_capstyle="round",
        )

    finite_pitch = np.concatenate([values for values in pitch_values if values.size])
    lower = float(np.floor(np.nanpercentile(finite_pitch, 2.0) - 0.5))
    upper = float(np.ceil(np.nanpercentile(finite_pitch, 98.0) + 0.5))
    axis.set_xlim(0.0, maximum_time + 0.08)
    axis.set_ylim(lower, upper)
    axis.axhline(0.0, color="#98A2B3", linewidth=1.0, zorder=0)
    axis.grid(axis="y", color="#E4E7EC", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#D0D5DD")
    axis.tick_params(labelsize=8.5, length=3)
    axis.set_xlabel("Raw time (s)", fontsize=9, labelpad=4)
    axis.set_ylabel("Pitch · st rel. median", fontsize=9, labelpad=4)

    figure.text(
        0.59,
        0.17,
        "Descriptive measures — not a quality score",
        fontsize=11.5,
        fontweight="bold",
        color="#344054",
    )
    figure.text(
        0.59,
        0.135,
        "Duration · pitch · energy · pauses",
        fontsize=10.5,
        color="#667085",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=DPI,
        facecolor=figure.get_facecolor(),
        edgecolor="none",
        metadata={"Software": "Prosodiff"},
    )
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/prosodiff-card.json"),
        help="Prosodiff JSON report used for the plotted contours.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/prosodiff-social-preview.png"),
        help="Destination PNG; rendered at exactly 1280 x 640 pixels.",
    )
    arguments = parser.parse_args()
    rendered = render_social_preview(arguments.report, arguments.output)
    print(f"Created {rendered} ({WIDTH_PX} x {HEIGHT_PX})")


if __name__ == "__main__":
    main()
