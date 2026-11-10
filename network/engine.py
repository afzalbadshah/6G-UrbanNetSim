"""Exact packet-delivery integration for the packet and flow engine.

This module composes packetization, channel providers, per-resource FIFO link
timing, reproducible PER realization, retries/deadlines, explicit multi-hop
forwarding, and the final transmission ledger. It intentionally does not claim
that FIFO transmission is a radio scheduler. Energy becomes complete only for
the engine's routed active-packet scope when every registered physical resource
has an injected energy model; cost remains incomplete until a tariff model is
injected.

How to extend: implement a packetizer, channel provider, retry policy, or energy
model behind the existing protocols and inject it into ``NetworkEngine``. Keep
technology-specific equations outside this orchestration module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Mapping, Protocol

from network.accounting import TransmissionLedger, TransmissionLedgerSnapshot
from network.contracts import (
    ChannelModel,
    ChannelRequest,
    ChannelResult,
    DeliveryStatus,
    Hop,
    PacketizationResult,
    Packetizer,
    PacketUnit,
    TrafficObject,
    TransmissionOutcome,
)
from network.links import LinkReservation, SharedLink
from network.routing import ResolvedRoute
from network.reliability import (
    BernoulliSample,
    CounterBasedSampler,
    PacketErrorSampler,
    RetryPolicy,
)


def _require_nonnegative(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


class TransmissionFailureReason(str, Enum):
    """Engineering reason a packet attempt did not reach its destination."""

    CHANNEL_OUTAGE = "channel_outage"
    PACKET_ERROR = "packet_error"
    DEADLINE_BEFORE_ATTEMPT = "deadline_before_attempt"
    DEADLINE_DURING_ATTEMPT = "deadline_during_attempt"
    DEADLINE_AT_ARRIVAL = "deadline_at_arrival"


@dataclass(frozen=True, slots=True)
class MeasurementCompleteness:
    """Declares which result dimensions are complete for an engine scope."""

    scope_id: str
    bytes_complete: bool
    delay_complete: bool
    energy_complete: bool
    cost_complete: bool
    energy_incomplete_reason: str | None = None
    cost_incomplete_reason: str | None = None
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scope_id, str) or not self.scope_id.strip():
            raise ValueError("scope_id must be a non-empty string")
        for name in ("bytes_complete", "delay_complete", "energy_complete", "cost_complete"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        if self.energy_complete and self.energy_incomplete_reason is not None:
            raise ValueError("complete energy measurement cannot have an incomplete reason")
        if not self.energy_complete and not self.energy_incomplete_reason:
            raise ValueError("incomplete energy measurement requires a reason")
        if self.cost_complete and self.cost_incomplete_reason is not None:
            raise ValueError("complete cost measurement cannot have an incomplete reason")
        if not self.cost_complete and not self.cost_incomplete_reason:
            raise ValueError("incomplete cost measurement requires a reason")
        if any(not isinstance(item, str) or not item.strip() for item in self.limitations):
            raise ValueError("limitations must contain only non-empty strings")

    @classmethod
    def r1_packet_delivery(cls) -> "MeasurementCompleteness":
        """Describe the original exact one-hop delivery scope."""

        return cls(
            scope_id="r1_exact_one_hop_packet_delivery",
            bytes_complete=True,
            delay_complete=True,
            energy_complete=False,
            cost_complete=False,
            energy_incomplete_reason="R3 physical radio/compute power profiles are not integrated",
            cost_incomplete_reason="R3 versioned tariff profiles are not integrated",
            limitations=(
                "FIFO transmission is not a radio-access contention model",
                "finite buffers and queue admission are not integrated",
                "ACK/NACK control traffic is not packetized",
                "multi-hop routing and application reassembly are not integrated",
            ),
        )

    @classmethod
    def r2_routed_packet_delivery(cls) -> "MeasurementCompleteness":
        """Describe routed delivery before physical energy integration."""

        return cls(
            scope_id="r2_exact_routed_packet_delivery",
            bytes_complete=True,
            delay_complete=True,
            energy_complete=False,
            cost_complete=False,
            energy_incomplete_reason="R3 physical radio/compute power formulas are not integrated",
            cost_incomplete_reason="R3 versioned tariff formulas are not integrated",
            limitations=(
                "FIFO transmission is not a radio-access contention model",
                "finite buffers and queue admission are not integrated",
                "ACK/NACK control traffic is not packetized",
                "in-route network-layer fragmentation is not integrated",
                "application reassembly processing and non-ALL completion rules are not integrated",
            ),
        )

    @classmethod
    def r3_routed_packet_energy(cls) -> "MeasurementCompleteness":
        """Describe routed delivery with complete active-packet energy."""

        return cls(
            scope_id="r3_exact_routed_packet_energy",
            bytes_complete=True,
            delay_complete=True,
            energy_complete=True,
            cost_complete=False,
            cost_incomplete_reason="R3 versioned tariff formulas are not integrated",
            limitations=(
                "FIFO transmission is not a radio-access contention model",
                "finite buffers and queue admission are not integrated",
                "ACK/NACK control traffic is not packetized",
                "in-route network-layer fragmentation is not integrated",
                "application reassembly processing and non-ALL completion rules are not integrated",
                "idle/sleep energy outside packet service requires the run-level state ledger",
                "radio state-transition energy requires the run-level session ledger",
            ),
        )


@dataclass(frozen=True, slots=True)
class TransmissionAttempt:
    """One channel evaluation and optional shared-link reservation."""

    attempt_number: int
    requested_at_s: float
    channel_evaluated_at_s: float
    ended_at_s: float
    tx_power_w: float
    channel_result: ChannelResult
    reservation: LinkReservation | None
    packet_error_sample: BernoulliSample | None
    succeeded: bool
    failure_reason: TransmissionFailureReason | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.attempt_number, bool)
            or not isinstance(self.attempt_number, int)
            or self.attempt_number < 1
        ):
            raise ValueError("attempt_number must be a positive integer")
        requested = _require_nonnegative("requested_at_s", self.requested_at_s)
        evaluated = _require_nonnegative("channel_evaluated_at_s", self.channel_evaluated_at_s)
        ended = _require_nonnegative("ended_at_s", self.ended_at_s)
        _require_nonnegative("tx_power_w", self.tx_power_w)
        if evaluated < requested:
            raise ValueError("channel evaluation cannot precede the attempt request")
        if ended < evaluated:
            raise ValueError("attempt cannot end before channel evaluation")
        if not isinstance(self.succeeded, bool):
            raise ValueError("succeeded must be a bool")
        if self.succeeded != (self.failure_reason is None):
            raise ValueError("failure_reason must be absent exactly when an attempt succeeds")
        if self.channel_result.outage:
            if self.reservation is not None or self.packet_error_sample is not None:
                raise ValueError("outage attempts cannot reserve service or realize PER")
            if self.failure_reason is not TransmissionFailureReason.CHANNEL_OUTAGE:
                raise ValueError("outage attempt must use channel_outage failure reason")
        elif self.reservation is None or self.packet_error_sample is None:
            raise ValueError("non-outage attempts require a reservation and PER sample")


@dataclass(frozen=True, slots=True)
class TransmissionEnergy:
    """Per-attempt transmitter, receiver, and transport energy in joules."""

    tx_energy_j: float = 0.0
    rx_energy_j: float = 0.0
    transport_energy_j: float = 0.0

    def __post_init__(self) -> None:
        for name in ("tx_energy_j", "rx_energy_j", "transport_energy_j"):
            object.__setattr__(self, name, _require_nonnegative(name, getattr(self, name)))

    @property
    def total_energy_j(self) -> float:
        """Return all network energy components in joules."""

        return self.tx_energy_j + self.rx_energy_j + self.transport_energy_j


class HopEnergyModel(Protocol):
    """Strategy that converts observed packet attempts into joules."""

    def validate_tx_power_w(self, tx_power_w: float) -> None:
        """Reject invalid radiated power before queue or ledger state mutates."""

    def calculate(
        self,
        *,
        attempts: tuple[TransmissionAttempt, ...],
        on_wire_bytes_per_attempt: int,
    ) -> TransmissionEnergy:
        """Return energy for the physical attempts of one packet-hop outcome."""


@dataclass(frozen=True, slots=True)
class PacketTransmissionResult:
    """Final hop outcome plus every attempt made for one packet."""

    outcome: TransmissionOutcome
    attempts: tuple[TransmissionAttempt, ...]

    def __post_init__(self) -> None:
        if self.outcome.attempts != len(self.attempts):
            raise ValueError("outcome attempt count must match the attempt trace")
        expected = tuple(range(1, len(self.attempts) + 1))
        if tuple(item.attempt_number for item in self.attempts) != expected:
            raise ValueError("attempt trace numbers must be contiguous from one")


@dataclass(frozen=True, slots=True)
class ObjectTransmissionResult:
    """One-hop results for every packet in an application object."""

    packetization: PacketizationResult
    packet_transmissions: tuple[PacketTransmissionResult, ...]

    def __post_init__(self) -> None:
        if len(self.packet_transmissions) != len(self.packetization.packets):
            raise ValueError("every packet must have exactly one final transmission result")
        expected_ids = tuple(packet.packet_id for packet in self.packetization.packets)
        actual_ids = tuple(result.outcome.unit_id for result in self.packet_transmissions)
        if actual_ids != expected_ids:
            raise ValueError("packet transmission results must preserve packet order and identity")

    @property
    def all_delivered(self) -> bool:
        """Return whether every packet reached the hop destination."""

        return all(
            result.outcome.status is DeliveryStatus.DELIVERED
            for result in self.packet_transmissions
        )

    @property
    def completed_at_s(self) -> float | None:
        """Return final delivery time in seconds, or ``None`` on failure."""

        if not self.all_delivered:
            return None
        return max(result.outcome.ended_at_s for result in self.packet_transmissions)


@dataclass(frozen=True, slots=True)
class HopTransmissionParameters:
    """Runtime radio/link inputs that are not physical-topology properties."""

    tx_power_w: float
    allocated_bandwidth_hz: float | None = None
    interference_dbm: float | None = None
    line_of_sight: bool | None = None
    elevation_angle_deg: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tx_power_w", _require_nonnegative("tx_power_w", self.tx_power_w))
        if self.allocated_bandwidth_hz is not None:
            allocated = _require_nonnegative(
                "allocated_bandwidth_hz", self.allocated_bandwidth_hz
            )
            if allocated == 0.0:
                raise ValueError("allocated_bandwidth_hz must be positive when provided")
            object.__setattr__(self, "allocated_bandwidth_hz", allocated)
        if self.interference_dbm is not None:
            if (
                isinstance(self.interference_dbm, bool)
                or not isinstance(self.interference_dbm, (int, float))
                or not isfinite(self.interference_dbm)
            ):
                raise ValueError("interference_dbm must be finite when provided")
            object.__setattr__(self, "interference_dbm", float(self.interference_dbm))
        if self.line_of_sight is not None and not isinstance(self.line_of_sight, bool):
            raise ValueError("line_of_sight must be a bool or None")
        if self.elevation_angle_deg is not None:
            elevation = _require_nonnegative(
                "elevation_angle_deg", self.elevation_angle_deg
            )
            if elevation > 90.0:
                raise ValueError("elevation_angle_deg must be within [0, 90]")
            object.__setattr__(self, "elevation_angle_deg", elevation)


@dataclass(frozen=True, slots=True)
class _PacketTransmissionSetup:
    """Validated, reusable inputs for one packet-hop state machine."""

    packet: PacketUnit
    hop: Hop
    link: SharedLink
    channel_model: ChannelModel
    on_wire_bytes: int
    started_at_s: float
    distance_m: float
    tx_power_w: float
    deadline_at_s: float | None


@dataclass(frozen=True, slots=True)
class PacketRouteTransmission:
    """Ordered hop results for one packet along a declared route."""

    packet_id: str
    hop_transmissions: tuple[PacketTransmissionResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.packet_id, str) or not self.packet_id.strip():
            raise ValueError("packet_id must be a non-empty string")
        if any(item.outcome.unit_id != self.packet_id for item in self.hop_transmissions):
            raise ValueError("every hop result must belong to packet_id")

    @property
    def delivered_to_route_end(self) -> bool:
        """Return whether this packet reached the route destination."""

        return bool(self.hop_transmissions) and all(
            item.outcome.status is DeliveryStatus.DELIVERED
            for item in self.hop_transmissions
        )


@dataclass(frozen=True, slots=True)
class RouteTransmissionResult:
    """Routed transmission results for every packet in one object."""

    packetization: PacketizationResult
    route: ResolvedRoute
    packet_routes: tuple[PacketRouteTransmission, ...]

    def __post_init__(self) -> None:
        expected_ids = tuple(packet.packet_id for packet in self.packetization.packets)
        actual_ids = tuple(item.packet_id for item in self.packet_routes)
        if actual_ids != expected_ids:
            raise ValueError("route results must preserve packet order and identity")
        expected_hops = tuple(item.hop.hop_id for item in self.route.hops)
        for item in self.packet_routes:
            actual_hops = tuple(
                transmission.outcome.hop_id
                for transmission in item.hop_transmissions
            )
            if actual_hops != expected_hops[: len(actual_hops)]:
                raise ValueError("packet hop results must form a contiguous route prefix")
            if len(actual_hops) > len(expected_hops):
                raise ValueError("packet route contains more hops than the resolved route")
            if len(actual_hops) < len(expected_hops) and (
                not actual_hops
                or item.hop_transmissions[-1].outcome.status is DeliveryStatus.DELIVERED
            ):
                raise ValueError("only a failed packet may stop before the route endpoint")

    @property
    def all_delivered(self) -> bool:
        """Return whether every packet completed every declared route hop."""

        hop_count = len(self.route.hops)
        return all(
            len(item.hop_transmissions) == hop_count
            and item.delivered_to_route_end
            for item in self.packet_routes
        )

    @property
    def completed_at_s(self) -> float | None:
        """Return route completion time in seconds, or ``None`` on failure."""

        if not self.all_delivered:
            return None
        return max(
            item.hop_transmissions[-1].outcome.ended_at_s
            for item in self.packet_routes
        )


@dataclass(frozen=True, slots=True)
class NetworkEngineSnapshot:
    """Immutable ledger plus declared measurement-completeness scope."""

    transmission: TransmissionLedgerSnapshot
    completeness: MeasurementCompleteness


class NetworkEngine:
    """Exact routed packet engine over explicitly registered shared resources.

    It owns packet forwarding, FIFO queue/service time, retry/deadline behavior,
    and final byte/active-energy outcomes. Finite-buffer admission, radio grant
    scheduling, ACK/NACK traffic, reassembly processing, cost, and analytical
    batch mode remain separate increments.
    """

    def __init__(
        self,
        *,
        packetizer: Packetizer,
        channel_model: ChannelModel | Mapping[str, ChannelModel],
        shared_links: Mapping[str, SharedLink],
        retry_policy: RetryPolicy | None = None,
        experiment_seed: int | str = 0,
        sampler: PacketErrorSampler | None = None,
        ledger: TransmissionLedger | None = None,
        energy_models: Mapping[str, HopEnergyModel] | None = None,
    ) -> None:
        if not shared_links:
            raise ValueError("shared_links must contain at least one registered profile")
        self._packetizer = packetizer
        if isinstance(channel_model, Mapping):
            if not channel_model:
                raise ValueError("channel model registry must not be empty")
            self._channel_models = dict(channel_model)
            self._default_channel_model = None
        else:
            self._channel_models = {}
            self._default_channel_model = channel_model
        self._shared_links = dict(shared_links)
        self._retry_policy = retry_policy or RetryPolicy()
        self._sampler = sampler or CounterBasedSampler(experiment_seed)
        self._ledger = ledger if ledger is not None else TransmissionLedger()
        self._energy_models = dict(energy_models or {})

        for registration_id, model in self._energy_models.items():
            if not isinstance(registration_id, str) or not registration_id.strip():
                raise ValueError("energy model registration IDs must be non-empty strings")
            if not callable(getattr(model, "validate_tx_power_w", None)) or not callable(
                getattr(model, "calculate", None)
            ):
                raise ValueError(
                    "energy models must implement validate_tx_power_w() and calculate()"
                )

        resource_owner: dict[str, SharedLink] = {}
        for registration_id, link in self._shared_links.items():
            if not isinstance(registration_id, str) or not registration_id.strip():
                raise ValueError("shared link registration IDs must be non-empty strings")
            if not isinstance(link, SharedLink):
                raise ValueError("shared link registry values must be SharedLink objects")
            existing = resource_owner.get(link.profile.shared_resource_id)
            if existing is not None and existing is not link:
                raise ValueError(
                    "profiles with the same shared_resource_id must use one SharedLink instance"
                )
            resource_owner[link.profile.shared_resource_id] = link

        if energy_models is not None:
            missing_energy = sorted(
                registration_id
                for registration_id, link in self._shared_links.items()
                if registration_id not in self._energy_models
                and link.profile.profile_id not in self._energy_models
            )
            if missing_energy:
                raise ValueError(
                    f"energy model registry does not cover shared links {missing_energy}"
                )
            self._completeness = MeasurementCompleteness.r3_routed_packet_energy()
        else:
            self._completeness = MeasurementCompleteness.r2_routed_packet_delivery()

    @property
    def ledger(self) -> TransmissionLedger:
        """Expose the run's streaming transmission ledger."""

        return self._ledger

    def snapshot(self) -> NetworkEngineSnapshot:
        """Return current counters and measurement-completeness metadata."""

        return NetworkEngineSnapshot(
            transmission=self._ledger.snapshot(),
            completeness=self._completeness,
        )

    def _link_for(self, hop: Hop) -> SharedLink:
        # Physical-hop registration takes precedence so separate towers/links
        # can reuse one technology profile without acquiring one city-wide FIFO.
        # Profile-key fallback preserves the simple one-hop composition API.
        try:
            return self._shared_links.get(hop.hop_id) or self._shared_links[hop.link_profile_id]
        except KeyError as exc:
            raise ValueError(
                f"no SharedLink registered for hop {hop.hop_id!r} or profile "
                f"{hop.link_profile_id!r}"
            ) from exc

    def _channel_for(self, hop: Hop) -> ChannelModel:
        if self._default_channel_model is not None:
            return self._default_channel_model
        try:
            return self._channel_models[hop.link_profile_id]
        except KeyError as exc:
            raise ValueError(
                f"no ChannelModel registered for profile {hop.link_profile_id!r}"
            ) from exc

    def _energy_for(self, hop: Hop) -> HopEnergyModel | None:
        return self._energy_models.get(hop.hop_id) or self._energy_models.get(
            hop.link_profile_id
        )

    @staticmethod
    def _hop_on_wire_bytes(packet: PacketUnit, link: SharedLink) -> int:
        if packet.network_packet_bytes > link.profile.network_mtu_bytes:
            raise ValueError(
                f"packet {packet.packet_id!r} network size "
                f"{packet.network_packet_bytes} exceeds downstream MTU "
                f"{link.profile.network_mtu_bytes} for profile {link.profile.profile_id!r}; "
                "in-route fragmentation is not implemented"
            )
        return packet.network_packet_bytes + link.profile.header_stack.link_overhead_bytes

    def _record_final(
        self,
        *,
        packet: PacketUnit,
        hop: Hop,
        status: DeliveryStatus,
        started_at_s: float,
        ended_at_s: float,
        attempts: list[TransmissionAttempt],
        failure_reason: TransmissionFailureReason | None,
        on_wire_bytes_per_attempt: int,
    ) -> PacketTransmissionResult:
        energy_model = self._energy_for(hop)
        if energy_model is None:
            energy = TransmissionEnergy()
        else:
            energy = energy_model.calculate(
                attempts=tuple(attempts),
                on_wire_bytes_per_attempt=on_wire_bytes_per_attempt,
            )
            if not isinstance(energy, TransmissionEnergy):
                raise ValueError("energy model must return TransmissionEnergy")
        outcome = TransmissionOutcome(
            unit_id=packet.packet_id,
            hop_id=hop.hop_id,
            status=status,
            started_at_s=started_at_s,
            ended_at_s=ended_at_s,
            attempts=len(attempts),
            payload_bytes=packet.payload_bytes,
            # A channel outage is an observation, not a transmitted frame.
            on_wire_bytes=on_wire_bytes_per_attempt * sum(
                attempt.reservation is not None for attempt in attempts
            ),
            useful_payload_bytes=(
                packet.payload_bytes if status is DeliveryStatus.DELIVERED else 0
            ),
            tx_energy_j=energy.tx_energy_j,
            rx_energy_j=energy.rx_energy_j,
            transport_energy_j=energy.transport_energy_j,
            monetary_cost=0.0,
            failure_reason=None if failure_reason is None else failure_reason.value,
        )
        self._ledger.record(outcome)
        return PacketTransmissionResult(outcome=outcome, attempts=tuple(attempts))

    def _prepare_packet_transmission(
        self,
        packet: PacketUnit,
        hop: Hop,
        *,
        requested_at_s: float | None,
        distance_m: float,
        tx_power_w: float,
        deadline_at_s: float | None,
        allocated_bandwidth_hz: float | None,
    ) -> _PacketTransmissionSetup:
        """Resolve services and validate one packet-hop request before mutation."""

        link = self._link_for(hop)
        channel_model = self._channel_for(hop)
        requested_time_s = _require_nonnegative(
            "requested_at_s",
            packet.created_at_s if requested_at_s is None else requested_at_s,
        )
        started_at_s = max(packet.created_at_s, requested_time_s)
        validated_distance_m = _require_nonnegative("distance_m", distance_m)
        validated_tx_power_w = _require_nonnegative("tx_power_w", tx_power_w)
        energy_model = self._energy_for(hop)
        if energy_model is not None:
            energy_model.validate_tx_power_w(validated_tx_power_w)

        if deadline_at_s is None:
            validated_deadline_at_s = None
        else:
            validated_deadline_at_s = _require_nonnegative(
                "deadline_at_s", deadline_at_s
            )
            if validated_deadline_at_s < packet.created_at_s:
                raise ValueError("deadline_at_s cannot precede packet creation")
        if allocated_bandwidth_hz is not None and allocated_bandwidth_hz <= 0:
            raise ValueError(
                "allocated_bandwidth_hz must be positive when provided"
            )

        return _PacketTransmissionSetup(
            packet=packet,
            hop=hop,
            link=link,
            channel_model=channel_model,
            on_wire_bytes=self._hop_on_wire_bytes(packet, link),
            started_at_s=started_at_s,
            distance_m=validated_distance_m,
            tx_power_w=validated_tx_power_w,
            deadline_at_s=validated_deadline_at_s,
        )

    def _finish_packet(
        self,
        setup: _PacketTransmissionSetup,
        *,
        status: DeliveryStatus,
        ended_at_s: float,
        attempts: list[TransmissionAttempt],
        failure_reason: TransmissionFailureReason | None,
    ) -> PacketTransmissionResult:
        """Record and return the single final outcome for this packet-hop."""

        return self._record_final(
            packet=setup.packet,
            hop=setup.hop,
            status=status,
            started_at_s=setup.started_at_s,
            ended_at_s=ended_at_s,
            attempts=attempts,
            failure_reason=failure_reason,
            on_wire_bytes_per_attempt=setup.on_wire_bytes,
        )

    @staticmethod
    def _evaluate_channel(
        setup: _PacketTransmissionSetup,
        *,
        evaluated_at_s: float,
        allocated_bandwidth_hz: float | None,
        interference_dbm: float | None,
        line_of_sight: bool | None,
        elevation_angle_deg: float | None,
    ) -> ChannelResult:
        """Evaluate the configured provider for the current packet attempt."""

        return setup.channel_model.evaluate(
            ChannelRequest(
                link_profile_id=setup.hop.link_profile_id,
                hop_id=setup.hop.hop_id,
                requested_at_s=evaluated_at_s,
                distance_m=setup.distance_m,
                on_wire_bytes=setup.on_wire_bytes,
                direction=setup.hop.direction,
                tx_power_w=setup.tx_power_w,
                allocated_bandwidth_hz=allocated_bandwidth_hz,
                interference_dbm=interference_dbm,
                line_of_sight=line_of_sight,
                elevation_angle_deg=elevation_angle_deg,
            )
        )

    def _outage_attempt(
        self,
        setup: _PacketTransmissionSetup,
        channel: ChannelResult,
        *,
        attempt_number: int,
        requested_at_s: float,
        evaluated_at_s: float,
    ) -> TransmissionAttempt:
        """Create an attempt that observed channel outage and reserved no link."""

        return TransmissionAttempt(
            attempt_number=attempt_number,
            requested_at_s=requested_at_s,
            channel_evaluated_at_s=evaluated_at_s,
            ended_at_s=evaluated_at_s + self._retry_policy.outage_observation_s,
            tx_power_w=setup.tx_power_w,
            channel_result=channel,
            reservation=None,
            packet_error_sample=None,
            succeeded=False,
            failure_reason=TransmissionFailureReason.CHANNEL_OUTAGE,
        )

    def _service_attempt(
        self,
        setup: _PacketTransmissionSetup,
        channel: ChannelResult,
        *,
        attempt_number: int,
        requested_at_s: float,
        evaluated_at_s: float,
    ) -> TransmissionAttempt:
        """Reserve FIFO service, realize packet error, and classify the attempt."""

        reservation = setup.link.reserve(
            requested_at_s=requested_at_s,
            on_wire_bytes=setup.on_wire_bytes,
            achievable_rate_bps=channel.achievable_rate_bps,
            propagation_delay_s=channel.propagation_delay_s,
        )
        packet_error_sample = self._sampler.sample(
            channel.packet_error_probability,
            purpose="per",
            unit_id=setup.packet.packet_id,
            hop_id=setup.hop.hop_id,
            attempt_number=attempt_number,
        )
        arrived_after_deadline = (
            setup.deadline_at_s is not None
            and reservation.arrival_at_s > setup.deadline_at_s
        )
        if arrived_after_deadline:
            failure_reason = TransmissionFailureReason.DEADLINE_AT_ARRIVAL
        elif packet_error_sample.occurred:
            failure_reason = TransmissionFailureReason.PACKET_ERROR
        else:
            failure_reason = None
        return TransmissionAttempt(
            attempt_number=attempt_number,
            requested_at_s=requested_at_s,
            channel_evaluated_at_s=evaluated_at_s,
            ended_at_s=reservation.arrival_at_s,
            tx_power_w=setup.tx_power_w,
            channel_result=channel,
            reservation=reservation,
            packet_error_sample=packet_error_sample,
            succeeded=failure_reason is None,
            failure_reason=failure_reason,
        )

    def transmit_packet(
        self,
        packet: PacketUnit,
        hop: Hop,
        *,
        requested_at_s: float | None = None,
        distance_m: float,
        tx_power_w: float,
        deadline_at_s: float | None = None,
        allocated_bandwidth_hz: float | None = None,
        interference_dbm: float | None = None,
        line_of_sight: bool | None = None,
        elevation_angle_deg: float | None = None,
    ) -> PacketTransmissionResult:
        """Transmit one packet across one hop with retries and a deadline.

        Absolute times are seconds, power is watts, bandwidth is hertz, and
        interference is dBm. The final outcome is recorded exactly once.
        """

        setup = self._prepare_packet_transmission(
            packet,
            hop,
            requested_at_s=requested_at_s,
            distance_m=distance_m,
            tx_power_w=tx_power_w,
            deadline_at_s=deadline_at_s,
            allocated_bandwidth_hz=allocated_bandwidth_hz,
        )
        attempts: list[TransmissionAttempt] = []
        next_requested_at_s = setup.started_at_s

        while True:
            channel_evaluated_at_s = max(
                next_requested_at_s, setup.link.next_available_at_s
            )
            if (
                setup.deadline_at_s is not None
                and channel_evaluated_at_s >= setup.deadline_at_s
            ):
                return self._finish_packet(
                    setup,
                    status=DeliveryStatus.EXPIRED,
                    ended_at_s=max(setup.started_at_s, setup.deadline_at_s),
                    attempts=attempts,
                    failure_reason=TransmissionFailureReason.DEADLINE_BEFORE_ATTEMPT,
            )

            attempt_number = len(attempts) + 1
            channel = self._evaluate_channel(
                setup,
                evaluated_at_s=channel_evaluated_at_s,
                allocated_bandwidth_hz=allocated_bandwidth_hz,
                interference_dbm=interference_dbm,
                line_of_sight=line_of_sight,
                elevation_angle_deg=elevation_angle_deg,
            )

            if channel.outage:
                outage_attempt = self._outage_attempt(
                    setup,
                    channel,
                    attempt_number=attempt_number,
                    requested_at_s=next_requested_at_s,
                    evaluated_at_s=channel_evaluated_at_s,
                )
                attempts.append(outage_attempt)
                if (
                    setup.deadline_at_s is not None
                    and outage_attempt.ended_at_s > setup.deadline_at_s
                ):
                    return self._finish_packet(
                        setup,
                        status=DeliveryStatus.EXPIRED,
                        ended_at_s=outage_attempt.ended_at_s,
                        attempts=attempts,
                        failure_reason=TransmissionFailureReason.DEADLINE_DURING_ATTEMPT,
                    )
                retry_at_s = self._retry_policy.next_attempt_at_s(
                    outage_attempt.ended_at_s,
                    completed_attempts=len(attempts),
                )
                if retry_at_s is None:
                    return self._finish_packet(
                        setup,
                        status=DeliveryStatus.DROPPED,
                        ended_at_s=outage_attempt.ended_at_s,
                        attempts=attempts,
                        failure_reason=TransmissionFailureReason.CHANNEL_OUTAGE,
                    )
                next_requested_at_s = retry_at_s
                continue

            service_attempt = self._service_attempt(
                setup,
                channel,
                attempt_number=attempt_number,
                requested_at_s=next_requested_at_s,
                evaluated_at_s=channel_evaluated_at_s,
            )
            attempts.append(service_attempt)

            if (
                service_attempt.failure_reason
                is TransmissionFailureReason.DEADLINE_AT_ARRIVAL
            ):
                return self._finish_packet(
                    setup,
                    status=DeliveryStatus.EXPIRED,
                    ended_at_s=service_attempt.ended_at_s,
                    attempts=attempts,
                    failure_reason=TransmissionFailureReason.DEADLINE_AT_ARRIVAL,
                )
            if service_attempt.succeeded:
                return self._finish_packet(
                    setup,
                    status=DeliveryStatus.DELIVERED,
                    ended_at_s=service_attempt.ended_at_s,
                    attempts=attempts,
                    failure_reason=None,
                )

            retry_at_s = self._retry_policy.next_attempt_at_s(
                service_attempt.ended_at_s,
                completed_attempts=len(attempts),
            )
            if retry_at_s is None:
                return self._finish_packet(
                    setup,
                    status=DeliveryStatus.DROPPED,
                    ended_at_s=service_attempt.ended_at_s,
                    attempts=attempts,
                    failure_reason=TransmissionFailureReason.PACKET_ERROR,
                )
            next_requested_at_s = retry_at_s

    def transmit_object_one_hop(
        self,
        traffic_object: TrafficObject,
        hop: Hop,
        *,
        distance_m: float,
        tx_power_w: float,
        allocated_bandwidth_hz: float | None = None,
        interference_dbm: float | None = None,
        line_of_sight: bool | None = None,
        elevation_angle_deg: float | None = None,
    ) -> ObjectTransmissionResult:
        """Packetize one object and transmit every packet over one hop."""

        if traffic_object.direction is not hop.direction:
            raise ValueError("traffic object direction must match hop direction")
        link = self._link_for(hop)
        packetization = self._packetizer.packetize(traffic_object, link.profile)
        packet_transmissions = tuple(
            self.transmit_packet(
                packet,
                hop,
                requested_at_s=traffic_object.created_at_s,
                distance_m=distance_m,
                tx_power_w=tx_power_w,
                deadline_at_s=traffic_object.deadline_at_s,
                allocated_bandwidth_hz=allocated_bandwidth_hz,
                interference_dbm=interference_dbm,
                line_of_sight=line_of_sight,
                elevation_angle_deg=elevation_angle_deg,
            )
            for packet in packetization.packets
        )
        return ObjectTransmissionResult(
            packetization=packetization,
            packet_transmissions=packet_transmissions,
        )

    def transmit_object_route(
        self,
        traffic_object: TrafficObject,
        route: ResolvedRoute,
        *,
        hop_parameters: Mapping[str, HopTransmissionParameters],
    ) -> RouteTransmissionResult:
        """Transmit one object over every explicit hop of a resolved route.

        Packets are forwarded only after successful arrival on the previous hop.
        Link framing is recalculated per hop; network-layer fragmentation inside a
        route is rejected until an explicit fragmentation/reassembly model exists.
        """

        if traffic_object.source_id != route.source_node_id:
            raise ValueError("traffic object source must match route source")
        if traffic_object.destination_id != route.destination_node_id:
            raise ValueError("traffic object destination must match route destination")
        if traffic_object.direction is not route.hops[0].hop.direction:
            raise ValueError("traffic object direction must match the route's first hop")

        expected_hop_ids = {item.hop.hop_id for item in route.hops}
        supplied_hop_ids = set(hop_parameters)
        if supplied_hop_ids != expected_hop_ids:
            missing = sorted(expected_hop_ids - supplied_hop_ids)
            extra = sorted(supplied_hop_ids - expected_hop_ids)
            raise ValueError(
                f"hop_parameters must cover the route exactly; missing={missing}, extra={extra}"
            )
        if any(not isinstance(value, HopTransmissionParameters) for value in hop_parameters.values()):
            raise ValueError("hop_parameters values must be HopTransmissionParameters")

        first_link = self._link_for(route.hops[0].hop)
        packetization = self._packetizer.packetize(traffic_object, first_link.profile)
        per_packet: list[list[PacketTransmissionResult]] = [
            [] for _ in packetization.packets
        ]
        active = [True for _ in packetization.packets]

        for resolved_hop in route.hops:
            parameters = hop_parameters[resolved_hop.hop.hop_id]
            for index, packet in enumerate(packetization.packets):
                if not active[index]:
                    continue
                prior = per_packet[index]
                requested_at_s = (
                    traffic_object.created_at_s
                    if not prior
                    else prior[-1].outcome.ended_at_s
                )
                transmission = self.transmit_packet(
                    packet,
                    resolved_hop.hop,
                    requested_at_s=requested_at_s,
                    distance_m=resolved_hop.distance_m,
                    tx_power_w=parameters.tx_power_w,
                    deadline_at_s=traffic_object.deadline_at_s,
                    allocated_bandwidth_hz=parameters.allocated_bandwidth_hz,
                    interference_dbm=parameters.interference_dbm,
                    line_of_sight=parameters.line_of_sight,
                    elevation_angle_deg=parameters.elevation_angle_deg,
                )
                prior.append(transmission)
                active[index] = transmission.outcome.status is DeliveryStatus.DELIVERED

        return RouteTransmissionResult(
            packetization=packetization,
            route=route,
            packet_routes=tuple(
                PacketRouteTransmission(
                    packet_id=packet.packet_id,
                    hop_transmissions=tuple(per_packet[index]),
                )
                for index, packet in enumerate(packetization.packets)
            ),
        )
