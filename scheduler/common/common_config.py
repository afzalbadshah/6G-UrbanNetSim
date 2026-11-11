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
OFFLOADING_FILE_SUFFIX = "offloading_results.csv"
STATE_DIM = int(getattr(cfg, "STATE_DIM", 44))
ACTION_DIM = int(getattr(cfg, "ACTION_DIM", get_action_dim()))


def __getattr__(name):
    return getattr(cfg, name)
