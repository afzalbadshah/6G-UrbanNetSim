import config as cfg
from scheduler.common.common_action import get_action_dim

# MainApplication.py owns run-mode choices and overwrites these at startup.
COMPARISON_ENABLED = True
RUN_MODELS = ["random", "q_learning", "dqn", "double_dqn", "ppo", "a2c"]
RANDOM_SEED = 42
OUTPUT_DIR = "output/exploration"


def seed_for(repetition=0, base_seed=None):
    """Effective seed for one repetition of a comparison run.

    repetition == 0 (the default) returns the fixed base seed, so a single run
    reproduces the historical DQN/Double-DQN/Random/Rule-Based results exactly.
    repetition r > 0 returns base_seed + r, giving independent-but-reproducible
    realizations for multi-seed validation (Phase 5). Within a repetition every
    policy is given the same seed, so all policies see the same environment
    realization and remain directly comparable.
    """
    base = int(RANDOM_SEED if base_seed is None else base_seed)
    return base + int(repetition)
# Canonical offloading output is ``results.csv``.  The old
# ``<model>_offloading_results.csv`` file duplicated the same full row data and
# roughly doubled run size, so it is disabled by default.  Enable this only for
# legacy scripts that cannot read ``results.csv``.
WRITE_LEGACY_OFFLOADING_COPY = False
OFFLOADING_FILE_SUFFIX = "offloading_results.csv"

# Training histories are now written as one tidy ``training_metrics.csv`` per
# offloading/mobility model folder.  Enable this only when old plotting scripts
# need the previous split reward/epsilon/action/loss CSV files.
WRITE_LEGACY_TRAINING_LOGS = False

# Canonical mobility detail output is ``mobility/<model>/results.csv``.  The
# previous handover_results.csv + decision_quality.csv pair repeated many
# signal columns; keep the split files only for legacy notebooks.
WRITE_LEGACY_MOBILITY_DETAIL_CSVS = False

# Paper/evaluation runs should be compact: the canonical offloading and
# mobility ``results.csv`` files carry the detailed records.  These secondary
# files are useful while debugging, but are disabled by default to avoid clutter.
WRITE_SERVER_RESULTS_CSV = False
WRITE_MOBILITY_SUMMARY_CSV = False
WRITE_EVALUATION_TRAINING_METRICS = False
WRITE_DIAGNOSTICS_JSON = False
STATE_DIM = int(getattr(cfg, "STATE_DIM", 44))
ACTION_DIM = int(getattr(cfg, "ACTION_DIM", get_action_dim()))


def __getattr__(name):
    return getattr(cfg, name)
