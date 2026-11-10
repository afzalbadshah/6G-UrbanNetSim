"""Configuration package -- the single ``cfg`` namespace.

Every setting lives in a domain file (run, servers, devices_config, workload,
reward, mobility, radio, learning, scheduling, simulation, energy, topology,
evaluation, units). This package re-exports them all so application code can do
``import config as cfg`` and read ``cfg.<ANYTHING>`` from one place, while each
value is still DEFINED in exactly one domain file. There is no monolithic
config module -- edit the concerned domain file.

The workload orchestration functions live here (not in a domain file) because
they span domains: they rebuild the active WORKLOAD and refresh the learning
epsilon schedule together.
"""

# Re-export every domain file's public settings into the cfg namespace.
from config.run import *            # noqa: F401,F403  run/experiment control
from config.servers import *        # noqa: F401,F403  compute tiers (capacity+price)
from config.devices_config import * # noqa: F401,F403  end-device profiles
from config.workload import *       # noqa: F401,F403  traffic + SLA + density profiles
from config.reward import *         # noqa: F401,F403  per-class objective weights
from config.mobility import *       # noqa: F401,F403  city/movement/handover
from config.radio import *          # noqa: F401,F403  channel + bands + links
from config.learning import *       # noqa: F401,F403  RL hyperparameters
from config.scheduling import *     # noqa: F401,F403  scheduler rules
from config.simulation import *     # noqa: F401,F403  run toggles + timing scales

# Names needed by the orchestration functions below.
from config.run import WORKLOAD_PROFILE, ROUNDS, MAX_DEVICES, WORKLOAD_OVERRIDES
from config.workload import (
    WORKLOAD, TRAIN_WORKLOAD, EVAL_WORKLOAD, density_overrides,
)


def total_rounds(workload=None):
    """Return the number of rounds for a workload (its device ramp length)."""
    workload = workload or WORKLOAD
    configured_rounds = workload.get("rounds")
    if configured_rounds is not None:
        return max(1, int(configured_rounds or 1))
    start = int(workload.get("start_devices", 1) or 1)
    maximum = int(workload.get("max_devices", start) or start)
    increment = max(1, int(workload.get("increment", 1) or 1))
    return max(1, ((maximum - start) // increment) + 1)


def decisions_per_model(workload=None):
    """Return the total number of device-decisions across the whole run."""
    workload = workload or WORKLOAD
    start = int(workload.get("start_devices", 1) or 1)
    maximum = int(workload.get("max_devices", start) or start)
    increment = max(1, int(workload.get("increment", 1) or 1))
    total = 0
    for round_index in range(total_rounds(workload)):
        total += min(maximum, start + round_index * increment)
    return total


def apply_workload_profile(train_mode=False, overrides=None):
    """Activate the workload for this run and refresh the epsilon schedule.

    Device ramp comes from config/run.py: the selected density profile
    (WORKLOAD_PROFILE) sets it, unless it is "custom" (then ROUNDS / MAX_DEVICES
    are used). WORKLOAD_OVERRIDES is applied on top; an explicit ``overrides``
    argument wins over everything. The active WORKLOAD is mutated in place so
    every module that imported the object sees the change.
    """
    profile = TRAIN_WORKLOAD if bool(train_mode) else EVAL_WORKLOAD
    if overrides is not None:
        run_overrides = dict(overrides)
    else:
        run_overrides = dict(density_overrides(WORKLOAD_PROFILE))
        run_overrides.update(WORKLOAD_OVERRIDES)
        if str(WORKLOAD_PROFILE or "custom") == "custom":
            if ROUNDS is not None:
                run_overrides.setdefault("rounds", ROUNDS)
            if MAX_DEVICES is not None:
                run_overrides.setdefault("max_devices", MAX_DEVICES)
    merged = {**profile, **run_overrides}
    WORKLOAD.clear()
    WORKLOAD.update(merged)
    # Epsilon decays over the TRAINING schedule regardless of run mode. Rebind
    # on this package namespace so cfg.EPSILON_DECAY_STEPS (and common_config's
    # delegation) sees the refreshed value.
    train_workload = merged if bool(train_mode) else dict(TRAIN_WORKLOAD)
    globals()["EPSILON_DECAY_STEPS"] = total_rounds(train_workload)
    return WORKLOAD
