"""Calculate every simulator delay from observed physical events.

============================================================================
READ THIS FIRST -- how a task's delay is built
============================================================================
Every delay number the simulator reports is built from exactly three kinds
of interval, kept separate so a reader can always see where the
milliseconds came from::

    total_delay_s = access_delay_s + network_delay_s + processing_delay_s

    access_delay_s      time lost switching radio cells or waiting for
                         coverage (handover, coverage outage)
    network_delay_s     time to move the bytes across the network: queueing
                         + transmission + propagation + retry/outage waits,
                         tracked separately for the wireless and wired parts
                         of the route
    processing_delay_s  CPU time actually spent executing the task

The wireless achievable rate used by ``transmission_delay_s`` is calculated
once by :mod:`metrics.channel` from bandwidth, received power, thermal
noise, interference and SINR -- this module never invents a second radio
capacity formula. Shared-link queue reservations, packetization, retries
and outages are observed from the packet engine and are not approximated
here.

Layout of this file
--------------------
SECTION 1  Core physical delay formulas   -- the 3 atomic equations above
SECTION 2  Mobility/scheduling formulas   -- same "distance/speed" and
                                              "load" shapes, needed outside
                                              the packet engine
SECTION 3  Result structures              -- what a delay measurement IS,
                                              from one packet-hop up to one
                                              whole task
SECTION 4  Aggregation                    -- turn a raw engine trace INTO
                                              one of the structures above;
                                              the only place that classifies
                                              timestamps into named delays
SECTION 5  Supporting functions           -- input validation only, not
                                              physics; every function above
                                              calls these to fail loudly on
                                              bad input

All stored values use seconds. Millisecond properties on
``TaskDelayBreakdown`` exist only for the legacy scheduler/reporting
boundary -- never use them for further arithmetic, only for display.

How to extend: add a new delay only when the packet/compute engine records a
physically distinct interval, then add that interval to ``TaskDelayBreakdown``
(SECTION 3) and account for it in ``calculate_task_delay`` (SECTION 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from network.engine import (
        PacketRouteTransmission,
        PacketTransmissionResult,
        RouteTransmissionResult,
    )
    from network.topology import ResolvedRoute


# Delay components are reconstructed from differences between absolute
# floating-point timestamps.  Re-grouping those intervals (per hop, medium,
# and task) can change the last few bits even when the physical intervals are
# identical.  One nanosecond is well below the simulator's reported timing
# resolution, while still rejecting any meaningful accounting discrepancy.
_DELAY_REL_TOL = 1e-9
_DELAY_ABS_TOL_S = 1e-9


# ============================================================================
# SECTION 1 -- CORE PHYSICAL DELAY FORMULAS
#
#   Three atomic equations. Every other delay number in the simulator is a
#   sum of values that ultimately came from one of these three.
# ============================================================================

def transmission_delay_s(*, on_wire_bytes: int, achievable_rate_bps: float) -> float:
    """Serialization delay: time to push the bytes onto the link.

    ``on_wire_bytes * 8 / achievable_rate_bps``.

    Real-world equivalent: how long a radio/NIC takes to clock a packet's
    bits out at the link's achieved bit-rate, before the signal can even
    start travelling. More bytes or a slower rate both increase this delay.
    """

    if (
        isinstance(on_wire_bytes, bool)
        or not isinstance(on_wire_bytes, int)
        or on_wire_bytes < 0
    ):
        raise ValueError("on_wire_bytes must be a non-negative integer")
    rate_bps = _positive("achievable_rate_bps", achievable_rate_bps)
    return (on_wire_bytes * 8.0) / rate_bps


def propagation_delay_s(*, distance_m: float, propagation_speed_m_s: float) -> float:
    """Propagation delay: time for the signal to physically travel.

    ``distance_m / propagation_speed_m_s``.

    Real-world equivalent: distance divided by the signal's speed in the
    medium (~3e8 m/s for radio, ~2e8 m/s for fibre). Independent of how much
    data is sent -- one bit or one gigabyte take the same time to cross the
    distance once transmission has finished.
    """

    distance = _nonnegative("distance_m", distance_m)
    speed = _positive("propagation_speed_m_s", propagation_speed_m_s)
    return distance / speed


def processing_delay_s(*, compute_work_cycles: float, compute_rate_cycles_per_s: float) -> float:
    """Processing delay: CPU time to execute the task's work.

    ``compute_work_cycles / compute_rate_cycles_per_s``.

    Real-world equivalent: wall-clock compute time on the server/device --
    the task's work in CPU cycles divided by the cycles/second allocated to
    it. This is the single physical processing-delay formula for the whole
    simulator; on-device vs. server processing time differs only because the
    two sides plug in a different cycle rate, never a different formula.
    """

    work = _nonnegative("compute_work_cycles", compute_work_cycles)
    rate = _positive("compute_rate_cycles_per_s", compute_rate_cycles_per_s)
    return work / rate


# ============================================================================
# SECTION 2 -- MOBILITY / SCHEDULING SUPPORT FORMULAS
#
#   Small delay-shaped formulas needed by modules outside the packet engine
#   (handover dwell-time prediction, admission/queueing control). Kept here,
#   not duplicated locally, so no other module reinvents this math.
# ============================================================================

def remaining_coverage_time_s(
    *, remaining_distance_m: float, relative_speed_m_s: float
) -> float:
    """Time until a moving link leaves coverage: remaining distance / speed.

    Real-world equivalent: the same distance/speed shape as
    ``propagation_delay_s`` above, but applied to a device (or a mobile base
    station) closing on the edge of a coverage radius, to predict how long
    it can still be served before a handover becomes necessary.

    ``remaining_distance_m`` may be negative (already out of range), which
    returns 0.0. ``relative_speed_m_s`` is the closing speed between the two
    endpoints and must be positive.
    """

    if not isfinite(remaining_distance_m):
        raise ValueError("remaining_distance_m must be finite")
    speed = _positive("relative_speed_m_s", relative_speed_m_s)
    return max(0.0, float(remaining_distance_m)) / speed


def congestion_multiplier(*, utilization_ratio: float, headroom_floor: float) -> float:
    """Bounded M/M/1-style waiting multiplier: ``1 / max(1 - u, floor)``.

    Real-world equivalent: the closer a shared resource (a CPU, a link) is
    to fully busy, the longer an arriving unit of work waits behind the
    existing backlog -- the classic queueing-theory blow-up as utilization
    approaches 100%. ``headroom_floor`` caps that blow-up at a finite worst
    case instead of dividing by zero at ``u = 1``.

    ``utilization_ratio`` is the fraction of capacity already committed
    before the arriving unit of work, in ``[0, 1]``.
    """

    utilization = _nonnegative("utilization_ratio", utilization_ratio)
    if utilization > 1.0:
        raise ValueError("utilization_ratio must be within [0, 1]")
    floor = _positive("headroom_floor", headroom_floor)
    if floor > 1.0:
        raise ValueError("headroom_floor must be within (0, 1]")
    return 1.0 / max(1.0 - utilization, floor)


# ============================================================================
# SECTION 3 -- RESULT STRUCTURES
#
#   What a delay measurement looks like, from smallest to largest scope:
#     3a. PacketDelayBreakdown  one packet crossing one hop
#     3b. PacketRouteDelay      one packet crossing every hop of its route
#     3c. TaskDelayBreakdown    one whole task: access + network + processing
# ============================================================================

# ---- 3a. one packet, one hop ----------------------------------------------

@dataclass(frozen=True, slots=True)
class PacketDelayBreakdown:
    """Mutually exclusive components of one packet's one-hop elapsed time.

    The first serviced attempt contributes route propagation; later serviced
    attempts contribute retry propagation instead of changing the 1.1 objective.
    """

    queue_delay_s: float
    transmission_delay_s: float
    propagation_delay_s: float
    retry_backoff_s: float
    outage_observation_s: float
    terminal_wait_s: float
    total_delay_s: float
    retry_propagation_s: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "queue_delay_s",
            "transmission_delay_s",
            "propagation_delay_s",
            "retry_backoff_s",
            "outage_observation_s",
            "terminal_wait_s",
            "total_delay_s",
            "retry_propagation_s",
        ):
            object.__setattr__(self, name, _nonnegative(name, getattr(self, name)))
        if not isclose(
            self.component_sum_s,
            self.total_delay_s,
            rel_tol=_DELAY_REL_TOL,
            abs_tol=_DELAY_ABS_TOL_S,
        ):
            raise ValueError("delay components must sum to total_delay_s")

    @property
    def component_sum_s(self) -> float:
        """Return the sum of mutually exclusive delay components in seconds."""

        return (
            self.queue_delay_s
            + self.transmission_delay_s
            + self.propagation_delay_s
            + self.retry_propagation_s
            + self.retry_backoff_s
            + self.outage_observation_s
            + self.terminal_wait_s
        )


# ---- 3b. one packet, every hop of its route -------------------------------

@dataclass(frozen=True, slots=True)
class PacketRouteDelay:
    """Ordered per-hop delay breakdown for one packet route."""

    packet_id: str
    hop_delays: tuple[PacketDelayBreakdown, ...]
    elapsed_delay_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.packet_id, str) or not self.packet_id.strip():
            raise ValueError("packet_id must be a non-empty string")
        object.__setattr__(
            self,
            "elapsed_delay_s",
            _nonnegative("elapsed_delay_s", self.elapsed_delay_s),
        )
        if not isclose(
            sum(item.total_delay_s for item in self.hop_delays),
            self.elapsed_delay_s,
            rel_tol=_DELAY_REL_TOL,
            abs_tol=_DELAY_ABS_TOL_S,
        ):
            raise ValueError("hop delays must sum to the packet's routed elapsed delay")


# ---- 3c. one whole task: access + network + processing --------------------

@dataclass(frozen=True, slots=True)
class TaskDelayBreakdown:
    """Authoritative realized delay result, with explicit legacy views.

    Every stored field is mutually exclusive. Aggregate and millisecond values
    are properties, so downstream code cannot accidentally rebuild the total.
    ``complete_latency_s`` is the canonical elapsed latency and includes access,
    network, and processing time. ``legacy_objective_delay_s`` preserves
    the old 1.1 scheduler proxy only for historical comparison.
    """

    # -- 3c-i. stored fields: the ONLY numbers this object is built from ----
    handover_delay_s: float
    coverage_outage_delay_s: float
    wireless_queue_delay_s: float
    wired_queue_delay_s: float
    wireless_transmission_delay_s: float
    wired_transmission_delay_s: float
    wireless_propagation_delay_s: float
    wired_propagation_delay_s: float
    retry_backoff_delay_s: float
    channel_outage_delay_s: float
    terminal_wait_delay_s: float
    processing_delay_s: float
    network_delay_s: float
    total_delay_s: float
    retry_propagation_delay_s: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "handover_delay_s",
            "coverage_outage_delay_s",
            "wireless_queue_delay_s",
            "wired_queue_delay_s",
            "wireless_transmission_delay_s",
            "wired_transmission_delay_s",
            "wireless_propagation_delay_s",
            "wired_propagation_delay_s",
            "retry_backoff_delay_s",
            "channel_outage_delay_s",
            "terminal_wait_delay_s",
            "processing_delay_s",
            "network_delay_s",
            "total_delay_s",
            "retry_propagation_delay_s",
        ):
            object.__setattr__(self, name, _nonnegative(name, getattr(self, name)))

        if not isclose(
            self.calculated_network_delay_s,
            self.network_delay_s,
            rel_tol=_DELAY_REL_TOL,
            abs_tol=_DELAY_ABS_TOL_S,
        ):
            difference_s = self.calculated_network_delay_s - self.network_delay_s
            raise ValueError(
                "network components must sum to network_delay_s: "
                f"components={self.calculated_network_delay_s:.17g}, "
                f"network={self.network_delay_s:.17g}, "
                f"difference={difference_s:.3e} s"
            )
        if not isclose(
            self.calculated_total_delay_s,
            self.total_delay_s,
            rel_tol=_DELAY_REL_TOL,
            abs_tol=_DELAY_ABS_TOL_S,
        ):
            raise ValueError(
                "access, network, and processing delays "
                "must sum to total_delay_s"
            )

    # -- 3c-ii. seconds-domain aggregates: grouped views of the fields above -

    @property
    def access_delay_s(self) -> float:
        """Return handover plus non-overlapping coverage-outage time."""

        return self.handover_delay_s + self.coverage_outage_delay_s

    @property
    def queue_delay_s(self) -> float:
        """Return wireless plus wired shared-resource queue waiting."""

        return self.wireless_queue_delay_s + self.wired_queue_delay_s

    @property
    def transmission_delay_s(self) -> float:
        """Return wireless plus wired packet transmission time."""

        return self.wireless_transmission_delay_s + self.wired_transmission_delay_s

    @property
    def propagation_delay_s(self) -> float:
        """Return wireless plus wired one-way propagation time."""

        return self.wireless_propagation_delay_s + self.wired_propagation_delay_s

    @property
    def legacy_objective_delay_s(self) -> float:
        """Return the former 1.1 propagation + handover proxy."""

        return sum_delay_components_s(
            self.propagation_delay_s,
            self.handover_delay_s,
        )

    @property
    def legacy_complete_latency_s(self) -> float:
        """Return the former 1.1 proxy plus transmission and processing."""

        return sum_delay_components_s(
            self.legacy_objective_delay_s,
            self.transmission_delay_s,
            self.processing_delay_s,
        )

    @property
    def packet_observation_overhead_s(self) -> float:
        """Return queue, retry, outage, and access intervals beyond 1.1."""

        return sum_delay_components_s(
            self.coverage_outage_delay_s,
            self.queue_delay_s,
            self.retry_backoff_delay_s,
            self.retry_propagation_delay_s,
            self.channel_outage_delay_s,
            self.terminal_wait_delay_s,
        )

    @property
    def observed_complete_latency_s(self) -> float:
        """Return the canonical complete elapsed latency."""

        return self.legacy_complete_latency_s + self.packet_observation_overhead_s

    @property
    def complete_latency_s(self) -> float:
        """Return the canonical complete elapsed latency."""

        return self.observed_complete_latency_s

    @property
    def calculated_network_delay_s(self) -> float:
        """Return the explicit sum of all mutually exclusive network intervals."""

        return sum_delay_components_s(
            self.queue_delay_s,
            self.transmission_delay_s,
            self.propagation_delay_s,
            self.retry_backoff_delay_s,
            self.retry_propagation_delay_s,
            self.channel_outage_delay_s,
            self.terminal_wait_delay_s,
        )

    @property
    def calculated_total_delay_s(self) -> float:
        """Return ``access + network + processing`` in seconds."""

        return sum_delay_components_s(
            self.access_delay_s,
            self.network_delay_s,
            self.processing_delay_s,
        )

    # -- 3c-iii. millisecond views: legacy CSV/reporting names only. --------
    #    Never used for further math inside this codebase -- convert once,
    #    here, at the presentation boundary.

    @property
    def handover_ms(self) -> float:
        """Return the observed handover interruption in milliseconds."""

        return self.handover_delay_s * 1000.0

    @property
    def coverage_outage_ms(self) -> float:
        """Return non-overlapping access-outage time in milliseconds."""

        return self.coverage_outage_delay_s * 1000.0

    @property
    def pre_access_wait_ms(self) -> float:
        """Return total handover and access-outage waiting in milliseconds."""

        return self.access_delay_s * 1000.0

    @property
    def wireless_queue_ms(self) -> float:
        """Return wireless-resource queue waiting in milliseconds."""

        return self.wireless_queue_delay_s * 1000.0

    @property
    def wired_queue_ms(self) -> float:
        """Return wired-resource queue waiting in milliseconds."""

        return self.wired_queue_delay_s * 1000.0

    @property
    def queue_ms(self) -> float:
        """Return total shared-resource queue waiting in milliseconds."""

        return self.queue_delay_s * 1000.0

    @property
    def wireless_transmission_ms(self) -> float:
        """Return wireless packet transmission time in milliseconds."""

        return self.wireless_transmission_delay_s * 1000.0

    @property
    def wired_transmission_ms(self) -> float:
        """Return wired packet transmission time in milliseconds."""

        return self.wired_transmission_delay_s * 1000.0

    @property
    def transmission_ms(self) -> float:
        """Return total transmission time under the legacy CSV name."""

        return self.transmission_delay_s * 1000.0

    @property
    def wireless_propagation_ms(self) -> float:
        """Return wireless propagation time in milliseconds."""

        return self.wireless_propagation_delay_s * 1000.0

    @property
    def wired_propagation_ms(self) -> float:
        """Return wired propagation time in milliseconds."""

        return self.wired_propagation_delay_s * 1000.0

    @property
    def propagation_ms(self) -> float:
        """Return total route propagation time in milliseconds."""

        return self.propagation_delay_s * 1000.0

    @property
    def retry_backoff_ms(self) -> float:
        """Return time between failed and repeated attempts in milliseconds."""

        return self.retry_backoff_delay_s * 1000.0

    @property
    def retry_propagation_ms(self) -> float:
        """Return propagation of extra transmitted attempts in milliseconds."""

        return self.retry_propagation_delay_s * 1000.0

    @property
    def outage_observation_ms(self) -> float:
        """Return time spent observing channel outages in milliseconds."""

        return self.channel_outage_delay_s * 1000.0

    @property
    def terminal_wait_ms(self) -> float:
        """Return terminal deadline waiting without service in milliseconds."""

        return self.terminal_wait_delay_s * 1000.0

    @property
    def network_elapsed_ms(self) -> float:
        """Return complete packet-network delay in milliseconds."""

        return self.network_delay_s * 1000.0

    @property
    def processing_ms(self) -> float:
        """Return compute processing delay in milliseconds."""

        return self.processing_delay_s * 1000.0

    @property
    def legacy_objective_delay_ms(self) -> float:
        """Return the former 1.1 delay proxy in milliseconds."""

        return self.legacy_objective_delay_s * 1000.0

    @property
    def legacy_complete_latency_ms(self) -> float:
        """Return the former 1.1 processing-inclusive latency."""

        return self.legacy_complete_latency_s * 1000.0

    @property
    def complete_latency_ms(self) -> float:
        """Return the canonical complete elapsed latency in milliseconds."""

        return self.complete_latency_s * 1000.0

    @property
    def packet_observation_overhead_ms(self) -> float:
        """Return queue/retry/outage observation overhead in milliseconds."""

        return self.packet_observation_overhead_s * 1000.0

    @property
    def observed_complete_latency_ms(self) -> float:
        """Return the complete observed latency in milliseconds."""

        return self.observed_complete_latency_s * 1000.0

    @property
    def total_ms(self) -> float:
        """Compatibility alias for complete observed latency in milliseconds."""

        return self.total_delay_s * 1000.0


# ============================================================================
# SECTION 4 -- AGGREGATION
#
#   Read a raw engine trace (attempt timestamps, reservations, outcomes) and
#   classify it into the named components declared in SECTION 3. This is the
#   ONLY place in the simulator that performs that classification -- nowhere
#   else recomputes it.
# ============================================================================

def packet_delay_breakdown(result: PacketTransmissionResult) -> PacketDelayBreakdown:
    """Decompose an engine result without adding tier-latency proxies.

    ``terminal_wait_s`` is the auditable residual for a deadline expiry before a
    service attempt (including the valid zero-attempt case). All other components
    are derived directly from attempt and reservation timestamps.
    """

    from network.engine import PacketTransmissionResult

    if not isinstance(result, PacketTransmissionResult):
        raise ValueError("result must be a PacketTransmissionResult")
    attempts = result.attempts
    queue_delay_s = sum(
        attempt.channel_evaluated_at_s - attempt.requested_at_s
        for attempt in attempts
    )
    transmission_delay_s = sum(
        attempt.reservation.transmission_delay_s
        for attempt in attempts
        if attempt.reservation is not None
    )
    serviced_attempts = tuple(
        attempt for attempt in attempts if attempt.reservation is not None
    )
    propagation_delay_s = (
        serviced_attempts[0].reservation.propagation_delay_s
        if serviced_attempts else 0.0
    )
    retry_propagation_s = sum(
        attempt.reservation.propagation_delay_s
        for attempt in serviced_attempts[1:]
    )
    outage_observation_s = sum(
        attempt.ended_at_s - attempt.channel_evaluated_at_s
        for attempt in attempts
        if attempt.reservation is None
    )
    retry_backoff_s = sum(
        following.requested_at_s - previous.ended_at_s
        for previous, following in zip(attempts, attempts[1:])
    )
    total_delay_s = result.outcome.ended_at_s - result.outcome.started_at_s
    accounted_s = (
        queue_delay_s
        + transmission_delay_s
        + propagation_delay_s
        + retry_propagation_s
        + retry_backoff_s
        + outage_observation_s
    )
    residual_s = total_delay_s - accounted_s
    if residual_s < -1e-10:
        raise ValueError("attempt timestamps exceed the final outcome duration")
    terminal_wait_s = max(0.0, residual_s)
    return PacketDelayBreakdown(
        queue_delay_s=queue_delay_s,
        transmission_delay_s=transmission_delay_s,
        propagation_delay_s=propagation_delay_s,
        retry_propagation_s=retry_propagation_s,
        retry_backoff_s=retry_backoff_s,
        outage_observation_s=outage_observation_s,
        terminal_wait_s=terminal_wait_s,
        total_delay_s=total_delay_s,
    )


def packet_route_delay(result: PacketRouteTransmission) -> PacketRouteDelay:
    """Return per-hop delay components for one packet's routed transmission."""

    from network.engine import PacketRouteTransmission

    if not isinstance(result, PacketRouteTransmission):
        raise ValueError("result must be a PacketRouteTransmission")
    if not result.hop_transmissions:
        return PacketRouteDelay(
            packet_id=result.packet_id,
            hop_delays=(),
            elapsed_delay_s=0.0,
        )
    hop_delays = tuple(packet_delay_breakdown(item) for item in result.hop_transmissions)
    elapsed = (
        result.hop_transmissions[-1].outcome.ended_at_s
        - result.hop_transmissions[0].outcome.started_at_s
    )
    return PacketRouteDelay(
        packet_id=result.packet_id,
        hop_delays=hop_delays,
        elapsed_delay_s=elapsed,
    )


def route_completion_delay_s(result: RouteTransmissionResult) -> float | None:
    """Application completion delay for the current ALL-packets rule."""

    from network.engine import RouteTransmissionResult

    if not isinstance(result, RouteTransmissionResult):
        raise ValueError("result must be a RouteTransmissionResult")
    if result.completed_at_s is None:
        return None
    return result.completed_at_s - result.packetization.pdu_set.created_at_s


def calculate_task_delay(
    *,
    critical_packet_delay: PacketRouteDelay | None = None,
    route: ResolvedRoute | None = None,
    network_elapsed_s: float = 0.0,
    handover_delay_s: float = 0.0,
    handover_interruption_s: float = 0.0,
    coverage_outage_duration_s: float = 0.0,
    access_observation_limit_s: float | None = None,
    processing_delay_s: float = 0.0,
) -> TaskDelayBreakdown:
    """Calculate and return the single task-delay value used by the simulator.

    The radio rate already contains bandwidth, noise, interference and SINR.
    This function classifies the observed packet intervals by wireless/wired
    medium and returns one result; callers must not recalculate its totals.
    """

    from network.topology import Medium

    network_elapsed = _nonnegative("network_elapsed_s", network_elapsed_s)
    processing = _nonnegative("processing_delay_s", processing_delay_s)

    # Handover fields describe the same interruption under two legacy names.
    # Use the longer observation once, then count only outage time beyond it.
    handover = max(
        _nonnegative("handover_delay_s", handover_delay_s),
        _nonnegative("handover_interruption_s", handover_interruption_s),
    )
    coverage_outage = _nonnegative(
        "coverage_outage_duration_s", coverage_outage_duration_s
    )
    observed_access_delay = max(handover, coverage_outage)
    if access_observation_limit_s is not None:
        observed_access_delay = min(
            observed_access_delay,
            _nonnegative("access_observation_limit_s", access_observation_limit_s),
        )
    counted_handover = min(handover, observed_access_delay)
    counted_coverage_outage = observed_access_delay - counted_handover

    wireless_queue = 0.0
    wired_queue = 0.0
    wireless_transmission = 0.0
    wired_transmission = 0.0
    wireless_propagation = 0.0
    wired_propagation = 0.0
    retry_backoff = 0.0
    retry_propagation = 0.0
    channel_outage = 0.0
    terminal_wait = 0.0

    if critical_packet_delay is None:
        if route is not None or network_elapsed != 0.0:
            raise ValueError(
                "a route and network delay require a critical_packet_delay"
            )
    else:
        if not isinstance(critical_packet_delay, PacketRouteDelay):
            raise ValueError("critical_packet_delay must be a PacketRouteDelay")
        if route is None:
            raise ValueError("packet delay classification requires a route")
        if len(critical_packet_delay.hop_delays) > len(route.hops):
            raise ValueError("packet delay contains more hops than the route")

        for resolved_hop, hop_delay in zip(
            route.hops, critical_packet_delay.hop_delays
        ):
            if resolved_hop.link.medium is Medium.RADIO:
                wireless_queue += hop_delay.queue_delay_s
                wireless_transmission += hop_delay.transmission_delay_s
                wireless_propagation += hop_delay.propagation_delay_s
            else:
                wired_queue += hop_delay.queue_delay_s
                wired_transmission += hop_delay.transmission_delay_s
                wired_propagation += hop_delay.propagation_delay_s
            retry_backoff += hop_delay.retry_backoff_s
            retry_propagation += hop_delay.retry_propagation_s
            channel_outage += hop_delay.outage_observation_s
            terminal_wait += hop_delay.terminal_wait_s

    return TaskDelayBreakdown(
        handover_delay_s=counted_handover,
        coverage_outage_delay_s=counted_coverage_outage,
        wireless_queue_delay_s=wireless_queue,
        wired_queue_delay_s=wired_queue,
        wireless_transmission_delay_s=wireless_transmission,
        wired_transmission_delay_s=wired_transmission,
        wireless_propagation_delay_s=wireless_propagation,
        wired_propagation_delay_s=wired_propagation,
        retry_backoff_delay_s=retry_backoff,
        retry_propagation_delay_s=retry_propagation,
        channel_outage_delay_s=channel_outage,
        terminal_wait_delay_s=terminal_wait,
        processing_delay_s=processing,
        network_delay_s=network_elapsed,
        total_delay_s=sum_delay_components_s(
            counted_handover,
            counted_coverage_outage,
            network_elapsed,
            processing,
        ),
    )


# ============================================================================
# SECTION 5 -- SUPPORTING FUNCTIONS
#
#   Input validation only. None of these compute a physical quantity; every
#   formula above calls them just to fail loudly on bad input instead of
#   silently producing a wrong delay.
# ============================================================================

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


def sum_delay_components_s(*delay_components_s: float) -> float:
    """Return the sum of validated, mutually exclusive delay components."""

    return sum(
        _nonnegative(f"delay_components_s[{index}]", value)
        for index, value in enumerate(delay_components_s)
    )
