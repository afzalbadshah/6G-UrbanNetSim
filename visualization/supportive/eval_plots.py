"""Compatibility entry point for the focused IEEE panel generator.

The plotting implementation lives in :mod:`visualization.ieee_paper_plots`.
This module keeps older commands and imports working without maintaining a
second plotting code path.
"""

from pathlib import Path

from visualization.ieee_paper_plots import OUTPUT_DIR, generate_ieee_panels, main
from visualization.plot_config import plot_phase_dir


def _default_evaluation_phase() -> Path:
    return plot_phase_dir("evaluation")


def generate_zone_compact_figures(
    phase_dir: str | Path,
    output_dir: str | Path | None = None,
) -> list[Path]:
    return generate_ieee_panels(phase_dir, output_dir or OUTPUT_DIR)


def generate_zone_figures(
    phase_dir: str | Path,
    output_dir: str | Path | None = None,
) -> list[Path]:
    return generate_ieee_panels(phase_dir, output_dir or OUTPUT_DIR)


if __name__ == "__main__":
    main()
