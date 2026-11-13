"""Shared pytest configuration.

The project is intentionally not installed as a distribution yet, so the
repository root and the tools directory are added to sys.path explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for path in (ROOT, ROOT / "tools"):
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)
