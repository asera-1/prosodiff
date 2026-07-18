from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from prosodiff.models import Comparison
from prosodiff.render import _figure, render_comparison
from prosodiff.report import write_json_report


def test_linkedin_card_has_exact_four_by_five_canvas(
    four_take_comparison: Comparison,
    tmp_path: Path,
) -> None:
    output = render_comparison(four_take_comparison, tmp_path / "card.png")
    image = mpimg.imread(output)
    assert image.shape[:2] == (2700, 2160)
    assert output.stat().st_size > 100_000


def test_json_report_is_written(
    four_take_comparison: Comparison,
    tmp_path: Path,
) -> None:
    output = write_json_report(four_take_comparison, tmp_path / "delta.json")
    assert output.exists()
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_pitch_figure_describes_probability_as_diagnostic(
    four_take_comparison: Comparison,
) -> None:
    figure = _figure(four_take_comparison)
    try:
        text = " ".join(
            item.get_text()
            for item in [
                *figure.texts,
                *(child for axis in figure.axes for child in axis.texts),
            ]
        )
    finally:
        plt.close(figure)

    assert "pYIN Viterbi voicing" in text
    assert "below pYIN confidence threshold" not in text
    assert "p ≥ .80" not in text
