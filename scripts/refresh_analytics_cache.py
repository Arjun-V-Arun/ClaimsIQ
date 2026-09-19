"""Refresh materialised low-latency analytics tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.cache import refresh_analytics_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "claims_real.db")
    args = parser.parse_args()
    result = refresh_analytics_cache(args.db)
    print(f"Refreshed analytics cache: {result}")


if __name__ == "__main__":
    main()
