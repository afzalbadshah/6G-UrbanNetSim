"""Calculate application and packet-hop KPIs with explicit denominators.

How to extend: add an observation field and define its population before adding
an aggregate. Keep useful application bytes separate from on-wire/retry bytes and
keep failed/late policies explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from typing import Iterable

from network.contracts import CompletionRule, DeliveryStatus
from network.engine import RouteTransmissionResult


def _nonnegative(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _positive(name: str, value: float) -> float:
    result = _nonnegative(name, value)
    if result == 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def bounded_ratio(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator``, or ``0.0`` for a zero denominator.

    The single formula for every "successes over attempts" or "useful over
    total" ratio in the simulator (delivery ratio, wire efficiency, and
    similar KPIs), so no other module reimplements this division.
    """

    num = _nonnegative("numerator", numerator)
    denom = _nonnegative("denominator", denominator)
    return num / denom if denom else 0.0


def offered_load_ratio(
    offered_bits: float,
    bit_rate_bps: float,
    observation_window_s: float,
) -> float:
    """Return offered bits divided by service available in a time window."""

    bits = _nonnegative("offered_bits", offered_bits)
    rate = _positive("bit_rate_bps", bit_rate_bps)
    window = _positive("observation_window_s", observation_window_s)
    return bits / (rate * window)


def percentile_type7(values: Iterable[float], probability: float) -> float | None:
    """Hyndman-Fan type-7 percentile, stated explicitly for reproducibility."""

    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not isfinite(probability)
        or not 0.0 <= probability <= 1.0
    ):
        raise ValueError("probability must be within [0, 1]")
    ordered = sorted(_nonnegative("sample", value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * float(probability)
    lower = floor(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


@dataclass(frozen=True, slots=True)
class ApplicationObservation:
    """Application-level completion, deadline, byte, and packet-hop observation."""

    object_id: str
    created_at_s: float
    payload_bytes: int
    deadline_at_s: float | None
    completed_at_s: float | None
    packet_hop_offered_count: int
    packet_hop_delivered_count: int
    packet_hop_dropped_count: int
    packet_hop_expired_count: int
    on_wire_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str) or not self.object_id.strip():
            raise ValueError("object_id must be a non-empty string")
        object.__setattr__(self, "created_at_s", _nonnegative("created_at_s", self.created_at_s))
        if isinstance(self.payload_bytes, bool) or not isinstance(self.payload_bytes, int) or self.payload_bytes < 0:
            raise ValueError("payload_bytes must be a non-negative integer")
        if self.deadline_at_s is not None:
            deadline = _nonnegative("deadline_at_s", self.deadline_at_s)
            if deadline < self.created_at_s:
                raise ValueError("deadline_at_s cannot precede created_at_s")
            object.__setattr__(self, "deadline_at_s", deadline)
        if self.completed_at_s is not None:
            completed = _nonnegative("completed_at_s", self.completed_at_s)
            if completed < self.created_at_s:
                raise ValueError("completed_at_s cannot precede created_at_s")
            object.__setattr__(self, "completed_at_s", completed)
        for name in (
            "packet_hop_offered_count",
            "packet_hop_delivered_count",
            "packet_hop_dropped_count",
            "packet_hop_expired_count",
            "on_wire_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        final_count = (
            self.packet_hop_delivered_count
            + self.packet_hop_dropped_count
            + self.packet_hop_expired_count
        )
        if final_count != self.packet_hop_offered_count:
            raise ValueError("packet-hop final statuses must equal the offered denominator")

    @property
    def completed(self) -> bool:
        """Return whether the completion rule was satisfied."""

        return self.completed_at_s is not None

    @property
    def on_time(self) -> bool:
        """Return whether the application completed by its deadline."""

        return (
            self.completed_at_s is not None
            and self.deadline_at_s is not None
            and self.completed_at_s <= self.deadline_at_s
        )

    @property
    def useful_application_bytes(self) -> int:
        """Return payload bytes once for a completed application, otherwise zero."""

        return self.payload_bytes if self.completed else 0

    @property
    def completion_delay_s(self) -> float | None:
        """Return completion delay in seconds, or ``None`` when incomplete."""

        if self.completed_at_s is None:
            return None
        return self.completed_at_s - self.created_at_s


@dataclass(frozen=True, slots=True)
class DelayDistribution:
    """Application completion-delay summary in seconds."""

    population: str
    includes_late_completions: bool
    sample_count: int
    p50_s: float | None
    p95_s: float | None
    p99_s: float | None


@dataclass(frozen=True, slots=True)
class KpiSnapshot:
    """Aggregate KPIs with separate application and packet-hop populations."""

    measurement_window_s: float
    offered_application_count: int
    completed_application_count: int
    deadline_bearing_application_count: int
    on_time_application_count: int
    packet_hop_offered_count: int
    packet_hop_delivered_count: int
    useful_application_bytes: int
    on_wire_bytes: int
    application_completion_ratio: float
    deadline_success_ratio: float
    packet_hop_delivery_ratio: float
    goodput_bps: float
    wire_efficiency_ratio: float
    completion_delay: DelayDistribution


def observe_route(result: RouteTransmissionResult) -> ApplicationObservation:
    """Collapse routed packet-hop results without counting payload once per hop."""

    if not isinstance(result, RouteTransmissionResult):
        raise ValueError("result must be a RouteTransmissionResult")
    pdu_set = result.packetization.pdu_set
    if pdu_set.completion_rule is not CompletionRule.ALL:
        raise ValueError("only CompletionRule.ALL is implemented for routed application KPIs")
    outcomes = [
        transmission.outcome
        for packet_route in result.packet_routes
        for transmission in packet_route.hop_transmissions
    ]
    return ApplicationObservation(
        object_id=pdu_set.traffic_object_id,
        created_at_s=pdu_set.created_at_s,
        payload_bytes=pdu_set.total_payload_bytes,
        deadline_at_s=pdu_set.deadline_at_s,
        completed_at_s=result.completed_at_s,
        packet_hop_offered_count=len(outcomes),
        packet_hop_delivered_count=sum(
            item.status is DeliveryStatus.DELIVERED for item in outcomes
        ),
        packet_hop_dropped_count=sum(
            item.status is DeliveryStatus.DROPPED for item in outcomes
        ),
        packet_hop_expired_count=sum(
            item.status is DeliveryStatus.EXPIRED for item in outcomes
        ),
        on_wire_bytes=sum(item.on_wire_bytes for item in outcomes),
    )


def aggregate_kpis(
    observations: Iterable[ApplicationObservation],
    *,
    measurement_window_s: float,
    include_late_completions_in_delay: bool = True,
) -> KpiSnapshot:
    """Aggregate the fixed §15 denominators over a declared measurement window."""

    window = _nonnegative("measurement_window_s", measurement_window_s)
    if window == 0.0:
        raise ValueError("measurement_window_s must be greater than zero")
    items = tuple(observations)
    if any(not isinstance(item, ApplicationObservation) for item in items):
        raise ValueError("observations must contain ApplicationObservation values")
    offered = len(items)
    completed = sum(item.completed for item in items)
    deadline_bearing = sum(item.deadline_at_s is not None for item in items)
    on_time = sum(item.on_time for item in items)
    hop_offered = sum(item.packet_hop_offered_count for item in items)
    hop_delivered = sum(item.packet_hop_delivered_count for item in items)
    useful_bytes = sum(item.useful_application_bytes for item in items)
    wire_bytes = sum(item.on_wire_bytes for item in items)
    delay_samples = [
        item.completion_delay_s
        for item in items
        if item.completed
        and (
            include_late_completions_in_delay
            or item.deadline_at_s is None
            or item.on_time
        )
    ]
    return KpiSnapshot(
        measurement_window_s=window,
        offered_application_count=offered,
        completed_application_count=completed,
        deadline_bearing_application_count=deadline_bearing,
        on_time_application_count=on_time,
        packet_hop_offered_count=hop_offered,
        packet_hop_delivered_count=hop_delivered,
        useful_application_bytes=useful_bytes,
        on_wire_bytes=wire_bytes,
        application_completion_ratio=bounded_ratio(completed, offered),
        deadline_success_ratio=bounded_ratio(on_time, deadline_bearing),
        packet_hop_delivery_ratio=bounded_ratio(hop_delivered, hop_offered),
        goodput_bps=(useful_bytes * 8.0) / window,
        wire_efficiency_ratio=bounded_ratio(useful_bytes, wire_bytes),
        completion_delay=DelayDistribution(
            population="completed TrafficObjects",
            includes_late_completions=include_late_completions_in_delay,
            sample_count=len(delay_samples),
            p50_s=percentile_type7(delay_samples, 0.50),
            p95_s=percentile_type7(delay_samples, 0.95),
            p99_s=percentile_type7(delay_samples, 0.99),
        ),
    )
