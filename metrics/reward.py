"""Pure conversion of measured outcomes into the scalar learning reward.

The reward evaluator is intentionally independent of schedulers and tier
identities.  It consumes a complete observation, reads the single class-weight
matrix from :mod:`config.reward`, and returns an auditable component breakdown.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, log1p

import numpy as np

from config.reward import (
    COST_NORMALIZER,
    DEVICE_ENERGY_BATTERY_FRACTION,
    INVALID_ACTION_PENALTY,
    MIN_DEVICE_ENERGY_NORMALIZER_J,
    OVERUTILIZATION_PENALTY_WEIGHT,
    PRIORITY_PROFILES,
    REWARD_SCALE,
)
from config.scheduling import COMPUTE_SPILLOVER_THRESHOLD_PERCENT


def _finite(name: str, value: float) -> float:
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _unit(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def marginal_shared_capacity_pressure(
    current_utilization_percent: float,
    projected_utilization_percent: float,
) -> float:
    """Return a smooth marginal shared-capacity cost with no fixed threshold.

    Squaring utilization makes the same incremental load progressively more
    expensive near saturation while remaining continuous across every value,
    including the former 50 percent boundary.
    """

    before = _unit(_finite("current_utilization_percent", current_utilization_percent) / 100.0)
    after = _unit(_finite("projected_utilization_percent", projected_utilization_percent) / 100.0)
    return _unit((after * after) - (before * before))


def overutilization_pressure(projected_utilization_percent: float) -> float:
    """Return a smooth penalty only above the configured spillover threshold.

    The threshold has one owner in :mod:`config.scheduling`.  Pressure is zero
    at and below that operating point, then rises quadratically to one at full
    utilization.  A quadratic rise avoids a discontinuous reward cliff while
    still strongly discouraging decisions close to saturation.
    """

    projected = _finite(
        "selected_projected_utilization_percent",
        projected_utilization_percent,
    )
    threshold = _finite(
        "COMPUTE_SPILLOVER_THRESHOLD_PERCENT",
        COMPUTE_SPILLOVER_THRESHOLD_PERCENT,
    )
    if not 0.0 <= threshold < 100.0:
        raise ValueError("compute spillover threshold must be in [0, 100)")
    excess_ratio = _unit((projected - threshold) / (100.0 - threshold))
    return excess_ratio * excess_ratio


@dataclass(frozen=True, slots=True)
class RewardObservation:
    priority: int
    deadline_ms: float
    complete_latency_ms: float
    execution_failed: bool
    deadline_missed: bool
    device_battery_capacity_j: float
    device_battery_remaining_j: float
    device_battery_energy_j: float
    total_energy_j: float
    cost: float
    current_utilization_percent: float
    projected_utilization_percent: float
    resource_fit_ratio: float
    tier_capacity_ratio: float
    projected_server_battery_ratio: float
    mobility_violation: bool
    shared_capacity: bool
    invalid_action: bool = False
    communication_energy_j: float = 0.0
    device_propulsion_energy_j: float = 0.0
    server_hover_energy_j: float = 0.0
    # Preference scale for the positive, monotonic delay score. Actual task
    # deadlines remain a separate service-success boundary in every zone.
    delay_target_ms: float | None = None
    # Utilization of the action chosen by the policy.  It can differ from the
    # executed tier when the runtime protects a saturated tier by spilling the
    # task outward.  None means no spillover-aware caller supplied it.
    selected_projected_utilization_percent: float | None = None
    # Legacy input retained for compatibility; it no longer exempts failures.
    deadline_penalty_exempt: bool = False

    def __post_init__(self) -> None:
        if int(self.priority) not in PRIORITY_PROFILES:
            raise ValueError(f"no reward profile for priority {self.priority!r}")
        if _finite("deadline_ms", self.deadline_ms) <= 0.0:
            raise ValueError("deadline_ms must be greater than zero")
        for name in (
            "complete_latency_ms",
            "device_battery_capacity_j",
            "device_battery_remaining_j",
            "device_battery_energy_j",
            "total_energy_j",
            "cost",
        ):
            if _finite(name, getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        _finite("current_utilization_percent", self.current_utilization_percent)
        _finite("projected_utilization_percent", self.projected_utilization_percent)
        _finite("resource_fit_ratio", self.resource_fit_ratio)
        _finite("tier_capacity_ratio", self.tier_capacity_ratio)
        _finite("projected_server_battery_ratio", self.projected_server_battery_ratio)
        if self.selected_projected_utilization_percent is not None:
            _finite(
                "selected_projected_utilization_percent",
                self.selected_projected_utilization_percent,
            )
        if self.delay_target_ms is not None and _finite(
            "delay_target_ms",
            self.delay_target_ms,
        ) <= 0.0:
            raise ValueError("delay_target_ms must be greater than zero")


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    reward: float
    raw_reward: float
    objective: float
    deadline_pen: float
    deadline_success: float
    delay_score: float
    util_reward: float
    cost_pen: float
    energy_pen: float
    device_energy_pen: float
    device_battery_score: float
    server_battery_score: float
    compute_score: float
    shared_capacity_externality_pen: float
    overutilization_pen: float
    overutilization_deduction_points: float
    invalid_pen: float
    execution_failed: bool
    mobility_pen: float
    delay_weight: float
    device_battery_weight: float
    server_battery_weight: float
    compute_weight: float
    cost_weight: float
    mobility_weight: float
    shared_capacity_externality_weight: float
    overutilization_weight: float
    delay_ms: float
    deadline_ms: float
    delay_target_ms: float
    cost: float
    energy: float
    device_energy: float
    cost_normalizer: float
    device_energy_normalizer_j: float
    current_util: float
    future_util: float
    selected_future_util: float
    device_battery_ratio: float
    projected_server_battery_ratio: float
    communication_energy_j: float
    device_propulsion_energy_j: float
    server_hover_energy_j: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_reward(observation: RewardObservation) -> RewardBreakdown:
    """Evaluate one realized offloading outcome from the canonical matrix."""

    profile = PRIORITY_PROFILES[int(observation.priority)]
    weights = profile["objective_weights"]
    normalizers = profile.get("normalizers", {})
    deadline_ms = float(observation.deadline_ms)
    delay_target_ms = float(
        observation.delay_target_ms
        if observation.delay_target_ms is not None
        else profile.get("delay_target_ms", deadline_ms)
    )
    if delay_target_ms <= 0.0:
        raise ValueError("delay_target_ms must be greater than zero")
    delay_ms = float(observation.complete_latency_ms)
    missed = bool(
        observation.execution_failed
        or observation.deadline_missed
        or delay_ms > deadline_ms
    )

    deadline_pen = float(np.clip(delay_ms / deadline_ms, 0.0, 4.0))
    delay_ratio = delay_ms / delay_target_ms
    delay_score = 1.0 / (1.0 + delay_ratio)
    current_util = float(np.clip(observation.current_utilization_percent, 0.0, 100.0))
    future_util = float(np.clip(observation.projected_utilization_percent, 0.0, 100.0))
    util_reward = _unit(1.0 - (future_util / 100.0))
    cost_normalizer = float(normalizers.get("cost_usd", COST_NORMALIZER))
    if cost_normalizer <= 0.0:
        raise ValueError("cost normalizer must be greater than zero")
    cost_pen = float(np.clip(observation.cost / cost_normalizer, 0.0, 3.0))
    energy_pen = float(np.clip(observation.total_energy_j / 20.0, 0.0, 3.0))

    battery_capacity = float(observation.device_battery_capacity_j)
    battery_ratio = (
        _unit(observation.device_battery_remaining_j / battery_capacity)
        if battery_capacity > 0.0
        else 1.0
    )
    default_battery_norm = max(
        MIN_DEVICE_ENERGY_NORMALIZER_J,
        battery_capacity * DEVICE_ENERGY_BATTERY_FRACTION,
    )
    battery_norm = float(
        normalizers.get("device_energy_j", default_battery_norm)
    )
    if battery_norm <= 0.0:
        raise ValueError("device-energy normalizer must be greater than zero")
    device_energy_pen = float(
        np.clip(observation.device_battery_energy_j / battery_norm, 0.0, 3.0)
    )
    # Optimize incremental task energy, not the device's battery history.
    # Log scaling retains discrimination across radio and compute magnitudes.
    device_battery_score = 1.0 / (
        1.0 + log1p(observation.device_battery_energy_j / battery_norm)
    )
    server_battery_score = float(
        np.clip(observation.projected_server_battery_ratio, -1.0, 1.0)
    )
    resource_fit = _unit(observation.resource_fit_ratio)
    capacity_ratio = _unit(observation.tier_capacity_ratio)
    compute_score = _unit(0.5 * resource_fit + 0.5 * capacity_ratio)
    shared_capacity_pen = (
        marginal_shared_capacity_pressure(current_util, future_util)
        if observation.shared_capacity
        else 0.0
    )
    selected_future_util = float(
        future_util
        if observation.selected_projected_utilization_percent is None
        else observation.selected_projected_utilization_percent
    )
    overutilization_pen = overutilization_pressure(selected_future_util)
    mobility_pen = 1.0 if observation.mobility_violation else 0.0
    invalid_pen = INVALID_ACTION_PENALTY if observation.invalid_action else 0.0

    delay_weight = float(weights["delay"])
    device_battery_weight = float(weights["device_battery"])
    server_battery_weight = float(weights["server_battery"])
    compute_weight = float(weights["compute"])
    cost_weight = float(weights["cost"])
    mobility_weight = float(weights["mobility"])
    externality_weight = float(weights["shared_capacity_externality"])
    overutilization_weight = float(OVERUTILIZATION_PENALTY_WEIGHT)
    weight_sum = sum(
        abs(value)
        for value in (
            delay_weight,
            device_battery_weight,
            server_battery_weight,
            compute_weight,
            cost_weight,
            mobility_weight,
            externality_weight,
        )
    )
    objective = (
        delay_weight * delay_score
        + device_battery_weight * device_battery_score
        + server_battery_weight * server_battery_score
        + compute_weight * compute_score
        - cost_weight * cost_pen
        - mobility_weight * mobility_pen
        - externality_weight * shared_capacity_pen
    ) / max(weight_sum, 1e-9)
    # Preserve the established reward scale exactly at/below the threshold;
    # only excess utilization changes the objective.
    overutilization_deduction_points = (
        REWARD_SCALE * overutilization_weight * overutilization_pen
    )
    objective -= overutilization_deduction_points / REWARD_SCALE
    objective = float(np.clip(objective, -1.0, 1.0))

    raw_reward = REWARD_SCALE * objective - invalid_pen
    reward = float(np.clip(raw_reward, -REWARD_SCALE, REWARD_SCALE))
    if missed:
        raw_reward = min(float(raw_reward), -REWARD_SCALE)
        reward = -REWARD_SCALE

    return RewardBreakdown(
        reward=reward,
        raw_reward=float(raw_reward),
        objective=objective,
        deadline_pen=deadline_pen,
        deadline_success=0.0 if missed else 1.0,
        delay_score=delay_score,
        util_reward=util_reward,
        cost_pen=cost_pen,
        energy_pen=energy_pen,
        device_energy_pen=device_energy_pen,
        device_battery_score=device_battery_score,
        server_battery_score=server_battery_score,
        compute_score=compute_score,
        shared_capacity_externality_pen=shared_capacity_pen,
        overutilization_pen=overutilization_pen,
        overutilization_deduction_points=overutilization_deduction_points,
        invalid_pen=float(invalid_pen),
        execution_failed=bool(observation.execution_failed),
        mobility_pen=mobility_pen,
        delay_weight=delay_weight,
        device_battery_weight=device_battery_weight,
        server_battery_weight=server_battery_weight,
        compute_weight=compute_weight,
        cost_weight=cost_weight,
        mobility_weight=mobility_weight,
        shared_capacity_externality_weight=externality_weight,
        overutilization_weight=overutilization_weight,
        delay_ms=delay_ms,
        deadline_ms=deadline_ms,
        delay_target_ms=delay_target_ms,
        cost=float(observation.cost),
        energy=float(observation.total_energy_j),
        device_energy=float(observation.device_battery_energy_j),
        cost_normalizer=cost_normalizer,
        device_energy_normalizer_j=battery_norm,
        current_util=current_util,
        future_util=future_util,
        selected_future_util=selected_future_util,
        device_battery_ratio=battery_ratio,
        projected_server_battery_ratio=float(observation.projected_server_battery_ratio),
        communication_energy_j=float(observation.communication_energy_j),
        device_propulsion_energy_j=float(observation.device_propulsion_energy_j),
        server_hover_energy_j=float(observation.server_hover_energy_j),
    )
