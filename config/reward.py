"""Single declarative reward specification for learned offloading policies.

This module defines *what* each service class optimizes. It deliberately does
not name preferred tiers, define tier ladders, or impose utilization cut-offs.
The pure evaluator in :mod:`metrics.reward` combines these weights with the
canonical delay, energy, cost, and resource observations produced by the
simulator. Schedulers only call that evaluator; they do not redefine reward
logic.

``deadline_ms`` remains a compatibility default for the three service classes
and is only the hard failure boundary; the task/scenario SLA is authoritative
at runtime.  ``delay_target_ms`` is the end-to-end delay at which the delay
preference score is neutral (0): faster earns a positive score, slower a
negative one, bounded in [-1, 1].  It is the single authority for delay
preference and is deliberately NOT the SLA ``radio_latency_target_ms``
(config.workload): that is a radio-only technology target which end-to-end
delay can never meet, so scoring end-to-end delay against it saturates every
action at the score floor.  Objective weights sum to 1.0 per class, so each
weight is that term's share of the reward scale.

How to extend: add or tune objective weights here, then add the corresponding
tier-neutral measured component to ``metrics/reward.py``. Never encode a tier
name, fallback ladder, or utilization threshold in a learned-policy profile.
"""


# Scalarization constants live beside the class matrix so there is one reward
# specification and no scheduler-owned reward tuning.
REWARD_SCALE = 10.0
INVALID_ACTION_PENALTY = 4.0
COST_NORMALIZER = 10.0
DEVICE_ENERGY_BATTERY_FRACTION = 0.00001
MIN_DEVICE_ENERGY_NORMALIZER_J = 0.05

# Class-neutral learning cost of selecting a tier whose projected utilization
# exceeds the operating threshold defined once in config.scheduling.  The
# canonical formula lives in metrics.reward; this is only its scalar weight.
# Objective weights sum to 1.0 per class, so this is a share of the reward
# scale: 0.2 costs 2 reward points at 100 percent utilization.
OVERUTILIZATION_PENALTY_WEIGHT = 0.2

PRIORITY_PROFILES = {
    1: {
        "name": "URLLC",
        "deadline_ms": 300,
        "delay_target_ms": 2.0,
        "objective_weights": {
            "delay": 0.75,
            "device_battery": 0.05,
            "server_battery": 0.05,
            "compute": 0.05,
            "cost": 0.0,
            "mobility": 0.10,
            "shared_capacity_externality": 0.0,
        },
    },
    2: {
        "name": "mMTC",
        "deadline_ms": 600,
        "delay_target_ms": 100.0,
        # Reward scaling only: these are representative per-task magnitudes
        # already produced by the canonical cost/energy matrix.  They do not
        # calculate or alter physical cost or energy.  The former global
        # scales (10 USD and 0.05 J) made the measured mMTC differences
        # (~1e-7 USD and a few mJ) numerically invisible to learning.
        "normalizers": {
            # Offloaded mMTC tasks cost ~0.7e-6 USD in the FR3/6G campaign; a
            # 1e-7 scale clipped every one of them at the penalty ceiling.
            "cost_usd": 0.000001,
            # Device energy of an offloaded mMTC task is ~2e-5 J in the FR3/6G
            # campaign (p99 ~4e-5 J for far devices); onboard execution is
            # ~2e-2 J.  The penalty saturates at 3 x this scale, so it must sit
            # above the offload tail (6e-5 J here) or far-away offloads become
            # indistinguishable from onboard execution (a 1e-5 J scale did
            # exactly that and sent 4 percent of tasks onboard), yet low enough
            # to separate the offload tiers (a 5e-3 J scale resolved them to
            # ~1e-3 reward points).  Re-size when the radio profile changes the
            # offload energy.
            "device_energy_j": 0.00002,
        },
        "objective_weights": {
            "delay": 0.05,
            "device_battery": 0.75,
            "server_battery": 0.10,
            "compute": 0.10,
            # The USD cost model is still unit-unvalidated (board Task 16), so
            # mMTC is battery-led: with cost weighted, Cloud's cheaper compute
            # outweighed MEC's ~8 percent lower device energy.
            "cost": 0.0,
            "mobility": 0.0,
            "shared_capacity_externality": 0.0,
        },
    },
    3: {
        "name": "eMBB",
        "deadline_ms": 1000,
        "delay_target_ms": 100.0,
        "objective_weights": {
            "delay": 0.15,
            "device_battery": 0.0,
            "server_battery": 0.10,
            "compute": 0.60,
            "cost": 0.10,
            "mobility": 0.0,
            # Smooth, tier-neutral opportunity cost of consuming scarce shared
            # capacity. It lets eMBB learn when moving outward is worthwhile.
            "shared_capacity_externality": 0.05,
        },
    },
}
