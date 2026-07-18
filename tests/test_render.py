from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg

from prosodiff.models import Comparison
from prosodiff.render import render_comparison
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
