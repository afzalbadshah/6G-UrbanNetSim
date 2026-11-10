"""Command-line entry point for completed multi-seed campaign figures.

How to extend: add figure logic to
``visualization.supportive.training_aggregate_plots`` and keep this file limited to
user-facing command arguments.
"""

from __future__ import annotations

import argparse

from visualization.plot_config import plot_phase_dir
from visualization.supportive.training_aggregate_plots import generate_training_figures


def main(argv: list[str] | None = None) -> list[str]:
    """Generate aggregate campaign plots and return their paths for callers/tests."""
    parser = argparse.ArgumentParser(
        description="Plot seed-level and all-model behavior after a campaign."
    )
    parser.add_argument(
        "phase_dir",
        nargs="?",
        default=None,
        help=(
            "Optional phase folder containing seed_* directories. By default "
            "the exploration phase of the highest configured run is used."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model subset; default discovers every completed model",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Moving-average window in simulation rounds (default: 10)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution (default: 180)",
    )
    args = parser.parse_args(argv)

    phase_dir = args.phase_dir or plot_phase_dir("exploration")
    paths = generate_training_figures(
        phase_dir,
        models=args.models,
        moving_average_window=args.window,
        dpi=args.dpi,
    )
    for path in paths:
        print(f"[PLOT] {path}")
    return [str(path) for path in paths]


if __name__ == "__main__":
    main()
