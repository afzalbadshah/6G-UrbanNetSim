"""Typed contracts for the SmartCityEdgeSim packet and flow engine.

The objects in this module contain values and validation only. They deliberately
do not import mutable simulator configuration or implement radio/channel science.
All canonical internal units are SI: seconds, bytes, bit/s, hertz, metres,
watts, and joules.

How to extend: add an optional, unit-suffixed field to the smallest relevant
value object and update its validation/tests. Behavior belongs in strategies or
the engine, not in these immutable contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Protocol, runtime_checkable


class Plane(str, Enum):
    """Traffic plane used to separate user, control, and management work."""

    USER = "user"
    CONTROL = "control"
    MANAGEMENT = "management"
    SENSING = "sensing"


class Direction(str, Enum):
    """Direction of packet movement relative to the access network."""

    UPLINK = "uplink"
    DOWNLINK = "downlink"
    SIDELINK = "sidelink"
    ACCESS = "access"
    FRONTHAUL = "fronthaul"
    BACKHAUL = "backhaul"


class CompletionRule(str, Enum):
    """Application rule that decides when a packet set is useful."""

    ALL = "all"
    THRESHOLD = "threshold"
    PROGRESSIVE = "progressive"
    LATEST_UPDATE_WINS = "latest_update_wins"


class DeliveryStatus(str, Enum):
    """Final delivered, dropped, or deadline-expired outcome."""

    CREATED = "created"
    QUEUED = "queued"
    TRANSMITTED = "transmitted"
    DELIVERED = "delivered"
    EXPIRED = "expired"
    DROPPED = "dropped"


def _require_nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_nonnegative_number(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative number")


def _require_finite_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _require_probability(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be within [0, 1]")


def _require_enum(name: str, value: object, enum_type: type[Enum]) -> None:
    if not isinstance(value, enum_type):
        raise ValueError(f"{name} must be a {enum_type.__name__} value")


@dataclass(frozen=True, slots=True)
class HeaderStack:
    """Per-packet byte overhead.

    Network, transport, security and telemetry overhead are inside the network
    MTU. Link header/trailer overhead is outside that MTU but is counted on wire.
    """

    network_header_bytes: int = 40
    transport_header_bytes: int = 8
    security_overhead_bytes: int = 0
    telemetry_overhead_bytes: int = 0
    link_header_bytes: int = 0
    link_trailer_bytes: int = 0

    def __post_init__(self) -> None:
        for name in (
            "network_header_bytes",
            "transport_header_bytes",
            "security_overhead_bytes",
            "telemetry_overhead_bytes",
            "link_header_bytes",
            "link_trailer_bytes",
        ):
            _require_nonnegative_int(name, getattr(self, name))

    @property
    def inside_mtu_bytes(self) -> int:
        """Return headers counted inside the configured network MTU."""

        return (
            self.network_header_bytes
            + self.transport_header_bytes
            + self.security_overhead_bytes
            + self.telemetry_overhead_bytes
        )

    @property
    def link_overhead_bytes(self) -> int:
        """Return link header and trailer bytes outside the network MTU."""

        return self.link_header_bytes + self.link_trailer_bytes


@dataclass(frozen=True, slots=True)
class TrafficObject:
    """One application request before it is segmented into packets."""

    object_id: str
    flow_id: str
    source_id: str
    destination_id: str
    created_at_s: float
    payload_bytes: int
    deadline_at_s: float | None = None
    plane: Plane = Plane.USER
    traffic_kind: str = "data"
    flow_role: str = "request"
    direction: Direction = Direction.UPLINK
    service_class: str = "best_effort"
    slice_id: str | None = None
    importance: float = 1.0
    source_timestamp_s: float | None = None
    utility: float | None = None
    measurement_id: str | None = None
    service_chain_id: str | None = None
    stage_id: str | None = None
    dependency_ids: tuple[str, ...] = field(default_factory=tuple)
    completion_rule: CompletionRule = CompletionRule.ALL
    completion_threshold: float = 1.0
    security_profile_id: str | None = None
    telemetry_profile_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("object_id", "flow_id", "source_id", "destination_id", "traffic_kind", "flow_role", "service_class"):
            _require_nonempty(name, getattr(self, name))
        _require_nonnegative_number("created_at_s", self.created_at_s)
        _require_nonnegative_int("payload_bytes", self.payload_bytes)
        _require_enum("plane", self.plane, Plane)
        _require_enum("direction", self.direction, Direction)
        _require_enum("completion_rule", self.completion_rule, CompletionRule)
        _require_probability("importance", self.importance)
        _require_probability("completion_threshold", self.completion_threshold)
        if self.completion_threshold == 0.0:
            raise ValueError("completion_threshold must be greater than zero")
        if self.deadline_at_s is not None and self.deadline_at_s < self.created_at_s:
            raise ValueError("deadline_at_s cannot precede created_at_s")
        if self.source_timestamp_s is not None:
            _require_nonnegative_number("source_timestamp_s", self.source_timestamp_s)
        if self.utility is not None:
            _require_nonnegative_number("utility", self.utility)
        for dependency_id in self.dependency_ids:
            _require_nonempty("dependency_id", dependency_id)


@dataclass(frozen=True, slots=True)
class PDUSet:
    """Packet-set metadata whose completion has application meaning."""

    pdu_set_id: str
    traffic_object_id: str
    created_at_s: float
    total_payload_bytes: int
    packet_count: int
    deadline_at_s: float | None = None
    completion_rule: CompletionRule = CompletionRule.ALL
    completion_threshold: float = 1.0
    plane: Plane = Plane.USER
    traffic_kind: str = "data"
    flow_role: str = "request"
    measurement_id: str | None = None
    dependency_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _require_nonempty("pdu_set_id", self.pdu_set_id)
        _require_nonempty("traffic_object_id", self.traffic_object_id)
        _require_nonnegative_number("created_at_s", self.created_at_s)
        _require_nonnegative_int("total_payload_bytes", self.total_payload_bytes)
        _require_nonempty("traffic_kind", self.traffic_kind)
        _require_nonempty("flow_role", self.flow_role)
        _require_enum("plane", self.plane, Plane)
        _require_enum("completion_rule", self.completion_rule, CompletionRule)
        if isinstance(self.packet_count, bool) or not isinstance(self.packet_count, int) or self.packet_count < 1:
            raise ValueError("packet_count must be a positive integer")
        _require_probability("completion_threshold", self.completion_threshold)
        if self.completion_threshold == 0.0:
            raise ValueError("completion_threshold must be greater than zero")
        if self.deadline_at_s is not None and self.deadline_at_s < self.created_at_s:
            raise ValueError("deadline_at_s cannot precede created_at_s")


@dataclass(frozen=True, slots=True)
class PacketUnit:
    """One exact packet with useful payload and declared protocol overhead."""

    packet_id: str
    pdu_set_id: str
    sequence_number: int
    payload_bytes: int
    created_at_s: float
    network_header_bytes: int
    transport_header_bytes: int
    security_overhead_bytes: int = 0
    telemetry_overhead_bytes: int = 0
    link_header_bytes: int = 0
    link_trailer_bytes: int = 0
    is_last: bool = False
    replica_group_id: str | None = None
    path_id: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty("packet_id", self.packet_id)
        _require_nonempty("pdu_set_id", self.pdu_set_id)
        _require_nonnegative_int("sequence_number", self.sequence_number)
        _require_nonnegative_number("created_at_s", self.created_at_s)
        for name in (
            "payload_bytes",
            "network_header_bytes",
            "transport_header_bytes",
            "security_overhead_bytes",
            "telemetry_overhead_bytes",
            "link_header_bytes",
            "link_trailer_bytes",
        ):
            _require_nonnegative_int(name, getattr(self, name))

    @property
    def network_packet_bytes(self) -> int:
        """Return bytes inside the network MTU."""

        return (
            self.payload_bytes
            + self.network_header_bytes
            + self.transport_header_bytes
            + self.security_overhead_bytes
            + self.telemetry_overhead_bytes
        )

    @property
    def on_wire_bytes(self) -> int:
        """Return network packet bytes plus link-layer overhead."""

        return self.network_packet_bytes + self.link_header_bytes + self.link_trailer_bytes


@dataclass(frozen=True, slots=True)
class PacketBatch:
    """Aggregate packet representation for a future compact execution mode."""

    batch_id: str
    pdu_set_id: str
    packet_count: int
    payload_bytes: int
    on_wire_bytes: int
    created_at_s: float

    def __post_init__(self) -> None:
        _require_nonempty("batch_id", self.batch_id)
        _require_nonempty("pdu_set_id", self.pdu_set_id)
        if isinstance(self.packet_count, bool) or not isinstance(self.packet_count, int) or self.packet_count < 1:
            raise ValueError("packet_count must be a positive integer")
        _require_nonnegative_int("payload_bytes", self.payload_bytes)
        _require_nonnegative_int("on_wire_bytes", self.on_wire_bytes)
        _require_nonnegative_number("created_at_s", self.created_at_s)
        if self.on_wire_bytes < self.payload_bytes:
            raise ValueError("on_wire_bytes cannot be smaller than payload_bytes")


@dataclass(frozen=True, slots=True)
class PacketizationResult:
    """Exact packet sequence and packet-set metadata for one traffic object."""

    pdu_set: PDUSet
    packets: tuple[PacketUnit, ...]

    def __post_init__(self) -> None:
        if len(self.packets) != self.pdu_set.packet_count:
            raise ValueError("packet tuple length must match PDU-set packet_count")
        if sum(packet.payload_bytes for packet in self.packets) != self.pdu_set.total_payload_bytes:
            raise ValueError("packet payload total must match PDU-set payload total")
        expected_sequence = tuple(range(len(self.packets)))
        if tuple(packet.sequence_number for packet in self.packets) != expected_sequence:
            raise ValueError("packet sequence numbers must be contiguous from zero")
        if any(packet.pdu_set_id != self.pdu_set.pdu_set_id for packet in self.packets):
            raise ValueError("every packet must reference the result PDU set")
        if not self.packets[-1].is_last or any(packet.is_last for packet in self.packets[:-1]):
            raise ValueError("only the final packet may be marked is_last")

    @property
    def payload_bytes(self) -> int:
        """Return conserved useful payload bytes across all packets."""

        return sum(packet.payload_bytes for packet in self.packets)

    @property
    def on_wire_bytes(self) -> int:
        """Return total packet bytes including all declared overhead."""

        return sum(packet.on_wire_bytes for packet in self.packets)

    def to_batch(self, batch_id: str | None = None) -> PacketBatch:
        """Summarize exact packets without losing invariant byte totals."""

        return PacketBatch(
            batch_id=batch_id or f"{self.pdu_set.pdu_set_id}:batch",
            pdu_set_id=self.pdu_set.pdu_set_id,
            packet_count=len(self.packets),
            payload_bytes=self.payload_bytes,
            on_wire_bytes=self.on_wire_bytes,
            created_at_s=self.pdu_set.created_at_s,
        )


@dataclass(frozen=True, slots=True)
class Hop:
    """Directed logical step between two route nodes."""

    hop_id: str
    source_node_id: str
    destination_node_id: str
    link_profile_id: str
    direction: Direction

    def __post_init__(self) -> None:
        for name in ("hop_id", "source_node_id", "destination_node_id", "link_profile_id"):
            _require_nonempty(name, getattr(self, name))
        _require_enum("direction", self.direction, Direction)


@dataclass(frozen=True, slots=True)
class LinkProfile:
    """Immutable packet/frame/service limits for one shared resource."""

    profile_id: str
    technology: str
    generation: str
    band: str
    network_mtu_bytes: int
    nominal_bit_rate_bps: float
    propagation_speed_m_s: float
    shared_resource_id: str
    header_stack: HeaderStack = field(default_factory=HeaderStack)
    channel_model_id: str = "fixed"
    queue_capacity_bytes: int = 0

    def __post_init__(self) -> None:
        for name in ("profile_id", "technology", "generation", "band", "shared_resource_id", "channel_model_id"):
            _require_nonempty(name, getattr(self, name))
        if isinstance(self.network_mtu_bytes, bool) or not isinstance(self.network_mtu_bytes, int):
            raise ValueError("network_mtu_bytes must be an integer")
        if self.network_mtu_bytes <= self.header_stack.inside_mtu_bytes:
            raise ValueError("network_mtu_bytes must exceed inside-MTU overhead")
        _require_nonnegative_number("nominal_bit_rate_bps", self.nominal_bit_rate_bps)
        if self.nominal_bit_rate_bps == 0:
            raise ValueError("nominal_bit_rate_bps must be greater than zero")
        _require_nonnegative_number("propagation_speed_m_s", self.propagation_speed_m_s)
        if self.propagation_speed_m_s == 0:
            raise ValueError("propagation_speed_m_s must be greater than zero")
        _require_nonnegative_int("queue_capacity_bytes", self.queue_capacity_bytes)

    @property
    def max_payload_bytes(self) -> int:
        """Return maximum useful payload bytes per packet."""

        return self.network_mtu_bytes - self.header_stack.inside_mtu_bytes


@dataclass(frozen=True, slots=True)
class ChannelRequest:
    """Immutable physical/link query for one packet attempt."""

    link_profile_id: str
    hop_id: str
    requested_at_s: float
    distance_m: float
    on_wire_bytes: int
    direction: Direction
    tx_power_w: float
    allocated_bandwidth_hz: float | None = None
    interference_dbm: float | None = None
    line_of_sight: bool | None = None
    elevation_angle_deg: float | None = None

    def __post_init__(self) -> None:
        _require_nonempty("link_profile_id", self.link_profile_id)
        _require_nonempty("hop_id", self.hop_id)
        _require_nonnegative_number("requested_at_s", self.requested_at_s)
        _require_nonnegative_number("distance_m", self.distance_m)
        _require_nonnegative_int("on_wire_bytes", self.on_wire_bytes)
        _require_enum("direction", self.direction, Direction)
        _require_nonnegative_number("tx_power_w", self.tx_power_w)
        if self.allocated_bandwidth_hz is not None:
            _require_nonnegative_number("allocated_bandwidth_hz", self.allocated_bandwidth_hz)
            if self.allocated_bandwidth_hz == 0:
                raise ValueError("allocated_bandwidth_hz must be greater than zero when provided")
        if self.interference_dbm is not None:
            _require_finite_number("interference_dbm", self.interference_dbm)
        if self.elevation_angle_deg is not None:
            _require_finite_number("elevation_angle_deg", self.elevation_angle_deg)
            if not 0.0 <= self.elevation_angle_deg <= 90.0:
                raise ValueError("elevation_angle_deg must be within [0, 90]")


@dataclass(frozen=True, slots=True)
class ChannelResult:
    """One authoritative channel observation for a packet attempt.

    Radio providers populate the optional diagnostic fields so delay, energy,
    reporting, and research analysis consume the same frequency, bandwidth,
    power, noise, interference, and SNR/SINR observation. Fixed-rate wired
    providers leave those radio-only fields unset.
    """

    model_id: str
    sinr_db: float
    achievable_rate_bps: float
    outage: bool
    packet_error_probability: float
    propagation_delay_s: float
    rx_power_dbm: float | None = None
    band_name: str | None = None
    generation: str | None = None
    carrier_frequency_hz: float | None = None
    allocated_bandwidth_hz: float | None = None
    tx_power_dbm: float | None = None
    path_loss_db: float | None = None
    noise_power_dbm: float | None = None
    interference_power_dbm: float | None = None
    snr_db: float | None = None
    spatial_streams: int | None = None

    def __post_init__(self) -> None:
        _require_nonempty("model_id", self.model_id)
        _require_finite_number("sinr_db", self.sinr_db)
        _require_nonnegative_number("achievable_rate_bps", self.achievable_rate_bps)
        _require_probability("packet_error_probability", self.packet_error_probability)
        _require_nonnegative_number("propagation_delay_s", self.propagation_delay_s)
        if not isinstance(self.outage, bool):
            raise ValueError("outage must be a bool")
        if self.rx_power_dbm is not None:
            _require_finite_number("rx_power_dbm", self.rx_power_dbm)
        for name in ("band_name", "generation"):
            value = getattr(self, name)
            if value is not None:
                _require_nonempty(name, value)
        for name in (
            "carrier_frequency_hz",
            "allocated_bandwidth_hz",
            "path_loss_db",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_nonnegative_number(name, value)
                if name != "path_loss_db" and value == 0.0:
                    raise ValueError(f"{name} must be positive when provided")
        for name in (
            "tx_power_dbm",
            "noise_power_dbm",
            "interference_power_dbm",
            "snr_db",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_finite_number(name, value)
        if self.spatial_streams is not None:
            if (
                isinstance(self.spatial_streams, bool)
                or not isinstance(self.spatial_streams, int)
                or self.spatial_streams <= 0
            ):
                raise ValueError("spatial_streams must be a positive integer when provided")
        if not self.outage and self.achievable_rate_bps == 0:
            raise ValueError("a non-outage channel result must have a positive achievable rate")


@dataclass(frozen=True, slots=True)
class TransmissionOutcome:
    """Final accounting record for one packet on one hop."""

    unit_id: str
    hop_id: str
    status: DeliveryStatus
    started_at_s: float
    ended_at_s: float
    attempts: int
    payload_bytes: int
    on_wire_bytes: int
    useful_payload_bytes: int
    tx_energy_j: float = 0.0
    rx_energy_j: float = 0.0
    transport_energy_j: float = 0.0
    monetary_cost: float = 0.0
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty("unit_id", self.unit_id)
        _require_nonempty("hop_id", self.hop_id)
        _require_enum("status", self.status, DeliveryStatus)
        _require_nonnegative_number("started_at_s", self.started_at_s)
        _require_nonnegative_number("ended_at_s", self.ended_at_s)
        if self.ended_at_s < self.started_at_s:
            raise ValueError("ended_at_s cannot precede started_at_s")
        if isinstance(self.attempts, bool) or not isinstance(self.attempts, int) or self.attempts < 0:
            raise ValueError("attempts must be a non-negative integer")
        for name in ("payload_bytes", "on_wire_bytes", "useful_payload_bytes"):
            _require_nonnegative_int(name, getattr(self, name))
        if self.attempts == 0 and self.status is not DeliveryStatus.EXPIRED:
            raise ValueError("only an expired outcome may have zero attempts")
        if self.attempts == 0 and self.on_wire_bytes != 0:
            raise ValueError("a zero-attempt expiry cannot contain on-wire bytes")
        # An outage may be observed and retried without reserving the link or
        # transmitting a data frame. Such failed outcomes legitimately have
        # attempts > 0 and zero on-wire bytes.
        if self.on_wire_bytes > 0 and self.on_wire_bytes < self.payload_bytes:
            raise ValueError("transmitted on_wire_bytes cannot be smaller than payload_bytes")
        if self.status is DeliveryStatus.DELIVERED and self.on_wire_bytes < self.payload_bytes:
            raise ValueError("a delivered outcome must contain transmitted payload bytes")
        if self.useful_payload_bytes > self.payload_bytes:
            raise ValueError("useful_payload_bytes cannot exceed payload_bytes")
        if self.status is not DeliveryStatus.DELIVERED and self.useful_payload_bytes != 0:
            raise ValueError("only delivered outcomes may contain useful_payload_bytes")
        for name in ("tx_energy_j", "rx_energy_j", "transport_energy_j", "monetary_cost"):
            _require_nonnegative_number(name, getattr(self, name))
        if self.attempts == 0 and any(
            value != 0.0
            for value in (
                self.tx_energy_j,
                self.rx_energy_j,
                self.transport_energy_j,
                self.monetary_cost,
            )
        ):
            raise ValueError("a zero-attempt expiry cannot contain energy or cost")


@runtime_checkable
class ChannelModel(Protocol):
    """Strategy interface for radio or wired link service calculations."""

    def evaluate(self, request: ChannelRequest) -> ChannelResult:
        """Evaluate one immutable request and return one immutable result."""

        """Return immutable channel/link service information for one request."""


@runtime_checkable
class Packetizer(Protocol):
    """Strategy interface that segments application traffic into packets."""

    def packetize(self, traffic_object: TrafficObject, link_profile: LinkProfile) -> PacketizationResult:
        """Return exact packets that conserve the traffic object's payload."""

        """Convert one application transfer into a validated PDU set and packets."""
