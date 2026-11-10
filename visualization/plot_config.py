"""
Shared configuration for all visualization plotters.
Colors follow Option 1 (Performance-Ranked) scheme.
Used by: energy_plotter.py, delay_plotter.py, cost_plotter.py, etc.
"""

import glob
import os
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config as cfg
from core.run_history import select_phase

# ============================================================================
# RESULT-SOURCE SELECTION (shared by every plotter)
# ----------------------------------------------------------------------------
# Leave a phase as None to use its highest numbered compatible run. To pin a
# result set, write either a run folder name below OUTPUT_DIR or an absolute run
# directory. Training and evaluation are independent because they are normally
# produced by different simulator launches.
#
# Examples:
#   "exploration": "run_005_fr3_city_coverage_6g_training"
#   "evaluation": r"D:\6G-UrbanNetSim\output\run_006_fr3_city_coverage_6g_evaluation"
#
# An explicit path passed to a plotting function/command still takes priority.
# ============================================================================
VISUALIZATION_RUNS = {
    "exploration": None,
    "evaluation": "run_008_zoned_satellite_6g_evaluation",
}


# ============================================================================
# SEED-AWARE DATA LOADING (shared by every plotter)
# ----------------------------------------------------------------------------
# A multi-seed campaign writes one results.csv per (seed, model):
#     <phase>/seed_1/offloading/<model>/results.csv
#     <phase>/seed_2/offloading/<model>/results.csv   ...
# Every plotter selects only the columns it needs; this loader concatenates all
# seeds for a model and tags each row with a 1-based ``seed`` column, so a
# plotter's existing per-round aggregation/band now spans seeds. By default the
# highest numbered compatible run under OUTPUT_DIR is selected. The phase entry
# in VISUALIZATION_RUNS pins an older run; SMARTCITY_PLOT_PHASE remains the
# highest-priority direct path override. The legacy output/<phase> layout is
# used before run_001 exists.
# ============================================================================
_EXPLICIT_RESULTS_PHASE = os.environ.get("SMARTCITY_PLOT_PHASE")


def plot_phase_dir(phase_name="evaluation", explicit_phase=None):
    """Resolve a plotting phase from an override or the configured run history."""

    chosen = explicit_phase if explicit_phase is not None else _EXPLICIT_RESULTS_PHASE
    if chosen:
        path = Path(chosen)
        return path if path.is_absolute() else (_ROOT / path)
    phase_name = str(phase_name).strip().lower()
    if phase_name not in VISUALIZATION_RUNS:
        raise ValueError(
            f"unknown visualization phase {phase_name!r}; "
            f"expected one of {sorted(VISUALIZATION_RUNS)}"
        )
    output_root = Path(str(getattr(cfg, "OUTPUT_DIR", "output") or "output"))
    if not output_root.is_absolute():
        output_root = _ROOT / output_root
    return select_phase(
        output_root,
        phase_name,
        VISUALIZATION_RUNS[phase_name],
    )


RESULTS_PHASE = str(plot_phase_dir("evaluation"))


def _phase_dir(phase=None):
    return plot_phase_dir("evaluation", explicit_phase=phase)


def seed_dirs(phase=None):
    """Ordered list of seed_* directories under a phase (empty if single-run)."""
    base = _phase_dir(phase)

    def _idx(path):
        tail = os.path.basename(path).split("seed_", 1)[-1]
        return int(tail) if tail.isdigit() else 0

    return sorted(glob.glob(str(base / "seed_*")), key=_idx)


def model_result_paths(model, phase=None):
    """Every seed's results.csv path for one model (single-run path if no seeds).

    This is the ONE place the on-disk layout is encoded; plotters never build
    ``output/.../offloading/<model>/results.csv`` themselves anymore.
    """
    names = ("results.csv", f"{model}_offloading_results.csv")
    dirs = seed_dirs(phase)
    roots = [Path(d) / "offloading" / model for d in dirs] if dirs else [_phase_dir(phase) / "offloading" / model]
    paths = []
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.exists():
                paths.append(candidate)
                break
    return paths


def load_model_seeds(model, columns=None, phase=None):
    """Concatenate every seed's results.csv for one model, tagged with ``seed``.

    ``columns`` selects a subset (fast on wide CSVs); ``None`` reads all columns.
    Returns a DataFrame (with a 1-based ``seed`` column) or ``None`` if no file
    is found.
    """
    wanted = set(columns) if columns is not None else None
    frames = []
    for i, path in enumerate(model_result_paths(model, phase), start=1):
        usecols = (lambda c: c in wanted) if wanted is not None else None
        frame = pd.read_csv(path, sep=";", usecols=usecols, low_memory=False)
        frame["seed"] = i
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else None

# ============================================================================
# COLOR SCHEME: Option 1 - Performance-Ranked (Red → Green → Blue → Orange → Purple → Black)
# ============================================================================
# Each model has a color reflecting its relative performance tier
# Red (warmest) = Best, Black (neutral) = Baseline

COLORS_MODELS = {
    'dqn': '#D62728',           # Red - Best performer
    'double_dqn': '#2CA02C',    # Green - Very good
    'ppo': '#1F77B4',           # Blue - Good
    'a2c': '#FF7F0E',           # Orange - Moderate
    'q_learning': '#9467BD',    # Purple - Weaker
    'random': '#000000'         # Black - Baseline
}

# Alternative names/aliases for models.
# Comment out an entry here to exclude that model from ALL plotters --
# every script derives its active model list from ACTIVE_MODELS below,
# so this is the single place to enable/disable a model.
MODEL_ALIASES = {
    'dqn': 'DQN',
    'double_dqn': 'Double DQN',
    'ppo': 'PPO',
    'a2c': 'A2C',
    'q_learning': 'Q-Learning',
    'random': 'Random'
}

# Models to include in plots, in display order. Derived from MODEL_ALIASES
# so commenting a model out above excludes it everywhere automatically.
ACTIVE_MODELS = list(MODEL_ALIASES.keys())

# ============================================================================
# PLOT STYLING
# ============================================================================

# Figure size for IEEE journal subfigures (square format)
# Suitable for 2x2 or 3x3 grid layouts
FIGURE_SIZE_SQUARE = (7.0, 5.2)      # Optimized: 1.35:1 aspect ratio for crowd heatmaps with 5 rows × 4-6 columns
FIGURE_SIZE_WIDE = (12, 6)           # Wide format (optional, for single plots)
FIGURE_DPI = 150                     # Resolution for high-quality printing

# Font sizes (for IEEE paper readability)
FONTSIZE_TITLE = 12
FONTSIZE_AXIS_LABEL = 11
FONTSIZE_TICK_LABEL = 9              # Increased from 8 to 9 for better readability in 3x4 grids
FONTSIZE_LEGEND = 9

# Grid and axis styling
GRID_ALPHA = 0.15
GRID_LINEWIDTH = 0.5
GRID_LINESTYLE = '-'

# Shading (confidence intervals)
SHADING_ALPHA = 0.38              # 35-40% opacity for bands

# Box plot styling
BOXPLOT_WIDTH = 0.6               # Width of boxes
BOXPLOT_LINE_WIDTH = 2.0          # Line width for box edges and median
BOXPLOT_WHISKER_WIDTH = 1.5       # Line width for whiskers
BOXPLOT_PATCH_ALPHA = 0.7         # Fill opacity for box patches
BOXPLOT_OUTLIER_MARKERSIZE = 4    # Size of outlier points

# Legend styling
LEGEND_NCOL = 2                   # Number of columns in legend
LEGEND_FRAMEALPHA = 0.95          # Legend background transparency

# ============================================================================
# SERVICE CLASSES (standard across all simulations)
# ============================================================================
SERVICE_CLASSES = ["URLLC", "mMTC", "eMBB"]

# ============================================================================
# DEVICE TYPES (common across simulations)
# ============================================================================
DEVICE_TYPES_COMMON = [
    'drone',
    'camera',
    'connected_car',
    'emergency_vehicle',
    'traffic_controller',
    'smartphone',
    'iot_sensor',
    'wearable'
]

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================
OUTPUT_DIR = "visualization/energy_plots"
SAVE_FORMAT = "png"
BBOX_INCHES = 'tight'

# White background for all plots (IEEE standard)
PLOT_BACKGROUND_COLOR = 'white'
FIGURE_BACKGROUND_COLOR = 'white'
