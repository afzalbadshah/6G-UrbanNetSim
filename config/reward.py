"""Single declarative reward specification for learned offloading policies.

This module defines *what* each service class optimizes. It deliberately does
not name preferred tiers, define tier ladders, or impose utilization cut-offs.
The pure evaluator in :mod:`metrics.reward` combines these weights with the
canonical delay, energy, cost, and resource observations produced by the
simulator. Schedulers only call that evaluator; they do not redefine reward
logic.

``deadline_ms`` remains a compatibility default for the three service classes
and is only the hard failure boundary; the task/scenario SLA is authoritative
at runtime. ``delay_target_ms`` is a preference scale, not a service deadline:
the delay score is 1 / (1 + delay / target). Successful outcomes remain
positive without a terrestrial-coverage penalty. Zones supply availability
observations; they never prescribe actions or relax application deadlines.
Each class has one primary objective: delay, device task energy, or scalable
compute resources. Physical failure and overload are common penalties.

How to extend: add or tune objective weights here, then add the corresponding
tier-neutral measured component to ``metrics/reward.py``. Never encode a tier
name, fallback ladder, or utilization threshold in a learned-policy profile.
"""


# Scalarization constants live beside the class matrix so there is one reward
# specification and no scheduler-owned reward tuning.
REWARD_SCHEMA_VERSION = "unmasked-service-objectives-v1"
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
            "delay": 1.0,
            "device_battery": 0.0,
            "server_battery": 0.0,
            "compute": 0.0,
            "cost": 0.0,
            "mobility": 0.0,
            "shared_capacity_externality": 0.0,
        },
    },
    2: {
        "name": "mMTC",
        "deadline_ms": 600,
        "delay_target_ms": 100.0,
        # Reference scales do not change physical energy or cost.
        "normalizers": {
            "cost_usd": 0.000001,
            # Log scaling avoids a hard ceiling for higher satellite/onboard
            # task energies. This is a scale, not a preferred-tier threshold.
            "device_energy_j": 0.00002,
        },
        "objective_weights": {
            "delay": 0.0,
            "device_battery": 1.0,
            "server_battery": 0.0,
            "compute": 0.0,
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
            "delay": 0.0,
            "device_battery": 0.0,
            "server_battery": 0.0,
            "compute": 1.0,
            "cost": 0.0,
            "mobility": 0.0,
            "shared_capacity_externality": 0.0,
        },
    },
}
