"""Run / experiment control -- the knobs you change most, all in one place.

This is the file you edit to launch a run: which models, how many rounds and
devices, train vs evaluate, GUI on/off, and whether jobs run in parallel.
``applications/MainApplication.py`` reads these values and defines none of its
own -- change a run here, not in the launcher.
"""

# --- What to run ---------------------------------------------------------
RANDOM_SEED = 42                 # master seed (single authority for every run)
RUN_TRAINING = False              # fresh FR3/6G training run
RUN_MODELS = ["random", "q_learning", "dqn", "double_dqn", "ppo", "a2c"] #["random", "q_learning", "dqn", "double_dqn", "ppo", "a2c"] 
COMPARISON_ENABLED = True        # run all RUN_MODELS (False = only MAIN_SCHEDULER)
MAIN_SCHEDULER = "dqn"           # used only when COMPARISON_ENABLED is False

# --- Run size ------------------------------------------------------------
# Pick a density profile (defined in config/workload.py WORKLOAD_PROFILES):
#   "sparse_urban" | "urban" | "dense_urban"  -> the profile sets the device ramp
#   "custom"                                   -> use ROUNDS / MAX_DEVICES below
WORKLOAD_PROFILE = "custom"
ROUNDS = 200                     # used only when WORKLOAD_PROFILE == "custom"
MAX_DEVICES = 200                # used only when WORKLOAD_PROFILE == "custom"
WORKLOAD_OVERRIDES = {}          # extra workload overrides, e.g. {"increment": 5}

# --- Output & GUI --------------------------------------------------------
# Stable history root. Each application launch creates the next recognizable
# folder inside it, for example:
#   output/run_001_fr3_city_coverage_6g_training/...
#   output/run_002_fr3_city_coverage_6g_evaluation/seed_1/...
#   output/run_002_fr3_city_coverage_6g_evaluation/seed_2/...
OUTPUT_DIR = "output"
TRAINING_GUI = True
EVALUATION_GUI = True
ACTIVE_DEVICES_ONLY = True       # update mobility only for round-active devices

# --- Multi-seed campaign -------------------------------------------------
MULTISEED = True                 # False = single-seed run (uses RANDOM_SEED)
TRAINING_SEED_COUNT = 3          # training seeds, each written as seed_<n> when count > 1
MULTISEED_COUNT = 3              # evaluation seeds written under one run as seed_1..seed_3
MULTISEED_PARALLEL = False        # run (seed, model) jobs in parallel (safe on local disk)
MULTISEED_MAX_WORKERS = None     # None -> min(cpu_count, number of jobs)
EVAL_PLAN_NAME = "default"       # EvaluationPlan providing methodology metadata
