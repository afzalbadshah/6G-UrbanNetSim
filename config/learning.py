"""RL model / learning hyperparameters.

The knobs that shape how the agents learn: network dimensions, learning rate,
replay, epsilon exploration schedule, and the PPO/A2C/DQN specifics. To change
how a model trains, edit the value here (one place).
"""

from config.workload import TRAIN_WORKLOAD

# --- Network / replay ----------------------------------------------------
# 22 task features + 6 tiers x 10 per-tier features = 82. Changing ACTION_DIM
# or the per-tier feature count in common_state.py requires updating STATE_DIM,
# or features are silently truncated.
STATE_DIM = 82
ACTION_DIM = 6
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.95
MEMORY_SIZE = 10000
BATCH_SIZE = 32
# Store every transition, but only run a weight update once per this many calls
# (Nature-DQN update_frequency=4 convention).
LEARN_EVERY_N_DECISIONS = 4
# Double DQN only: gradient steps between hard target-network syncs.
TARGET_NETWORK_SYNC_EVERY = 250

# --- Exploration (epsilon-greedy) ---------------------------------------
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
EPSILON_DECAY_GRANULARITY = "round"
# Import-time default only -- apply_workload_profile() recomputes this from the
# active training schedule at run start (see config/__init__.py).
EPSILON_DECAY_STEPS = int(TRAIN_WORKLOAD.get("rounds", 200) or 200)

# --- Policy-gradient specifics ------------------------------------------
PPO_CLIP = 0.2
PPO_EPOCHS = 4
A2C_VALUE_WEIGHT = 0.5
ENTROPY_WEIGHT = 0.01
# L2 pull on raw policy logits: prevents late-training PPO policy collapse.
# 1e-3 learns stably to ~96% optimal (1e-5 diverged; 1e-2 was marginally tighter).
PPO_LOGIT_L2 = 1e-3
