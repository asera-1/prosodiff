"""Serialization helpers for Prosodiff reports."""

from __future__ import annotations

import json
from pathlib import Path

from prosodiff import __version__
from prosodiff.models import Comparison


def write_json_report(comparison: Comparison, path: Path) -> Path:
    """Write the versioned explicit delivery-attribute delta schema."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = comparison.to_dict(version=__version__)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination
