#!/usr/bin/env python3
"""Run the frozen Bashi real-environment proxy validation without Bellhop."""

from __future__ import annotations

import argparse
from pathlib import Path

from ocean_acoustic_surrogate.field_validation import run_bashi_reuse_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path("configs/field_validation/bashi_reuse_v0.1.yaml"),
    )
    args = parser.parse_args()
    run_bashi_reuse_validation(args.config)


if __name__ == "__main__":
    main()
