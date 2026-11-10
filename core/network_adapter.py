"""Translate a legacy compute task into one physical network execution.

Workflow: convert decimal kB/ms to bytes/s, create a :class:`TrafficObject`, run
it over an already-resolved route, then summarize delivery, delay, bytes, and
energy in :class:`TaskNetworkExecution`. The adapter never mutates the task.

How to extend: add legacy-unit conversions to :class:`LegacyUnitBoundary`; add
packet or channel behavior in ``network/`` rather than in this compatibility
module. See ``docs/PACKET_EXECUTION_GUIDE.md`` for the complete call chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from math import isfinite
from typing import Mapping, Protocol

from metrics.delay import PacketRouteDelay, packet_route_delay
from metrics.kpi import ApplicationObservation, observe_route
from network import (
    DeliveryStatus,
    HopTransmissionParameters,
    NetworkEngine,
    Plane,
    ResolvedRoute,
    RouteTransmissionResult,
    TrafficObject,
)


class LegacyNetworkTask(Protocol):
    """Read-only legacy fields needed to create a network request."""

    id: str
    data_size_kb: float
    input_payload_bytes: int
    deadline_ms: float | None
    service_class: str
    application_type: str


@dataclass(frozen=True, slots=True)
class LegacyUnitBoundary:
    """Explicit legacy-unit interpretation used at the migration boundary.

    ``bytes_per_legacy_kb`` is injected so an experiment cannot silently switch
    between decimal kB (1000 bytes) and binary KiB (1024 bytes).  SmartCityEdgeSim
    1.1/legacy rate formulas used decimal SI rates, so R5 composition supplies
    1000 and records that choice in the run configuration.
    """

    bytes_per_legacy_kb: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.bytes_per_legacy_kb, bool)
            or not isinstance(self.bytes_per_legacy_kb, int)
            or self.bytes_per_legacy_kb <= 0
        ):
            raise ValueError("bytes_per_legacy_kb must be a positive integer")

    def kilobytes_to_bytes(self, value_kb: float) -> int:
        """Convert legacy decimal kilobytes to an exact integer byte count.

        Raises ``ValueError`` when the value is negative, non-finite, or cannot
        be represented as whole bytes under ``bytes_per_legacy_kb``.
        """

        if (
            isinstance(value_kb, bool)
            or not isinstance(value_kb, (int, float, Decimal))
        ):
            raise ValueError("legacy kilobytes must be a finite non-negative number")
        try:
            value = Decimal(str(value_kb))
        except InvalidOperation as exc:
            raise ValueError(
                "legacy kilobytes must be a finite non-negative number"
            ) from exc
        if not value.is_finite() or value < 0:
            raise ValueError("legacy kilobytes must be a finite non-negative number")
        byte_value = value * self.bytes_per_legacy_kb
        integral = byte_value.to_integral_value()
        if byte_value != integral:
            raise ValueError(
                f"{value_kb!r} legacy kB is not an integral byte count under "
                f"{self.bytes_per_legacy_kb} bytes/kB"
            )
        return int(integral)

    @staticmethod
    def milliseconds_to_seconds(value_ms: float, *, name: str) -> float:
        """Convert a finite non-negative duration from milliseconds to seconds."""

        if (
            isinstance(value_ms, bool)
            or not isinstance(value_ms, (int, float))
            or not isfinite(value_ms)
            or value_ms < 0.0
        ):
            raise ValueError(f"{name} must be a finite non-negative number")
        return float(value_ms) / 1000.0


@dataclass(frozen=True, slots=True)
class TaskNetworkExecution:
    """Immutable delivery result for one application request.

    Times are seconds, data volumes are bytes, and energy components are joules.
    The object links the original traffic intent, packet/hop trace, and
    application-level observation so downstream code has one source of truth.
    """

    traffic_object: TrafficObject
    transmission: RouteTransmissionResult
    observation: ApplicationObservation
    status: DeliveryStatus
    finished_at_s: float
    network_elapsed_s: float
    critical_packet_delay: PacketRouteDelay
    tx_energy_j: float
    rx_energy_j: float
    transport_energy_j: float

    def __post_init__(self) -> None:
        if self.traffic_object.object_id != self.observation.object_id:
            raise ValueError("traffic object and observation identities must match")
        if self.finished_at_s < self.traffic_object.created_at_s:
            raise ValueError("network execution cannot finish before object creation")
        expected_elapsed = self.finished_at_s - self.traffic_object.created_at_s
        if abs(self.network_elapsed_s - expected_elapsed) > 1e-12:
            raise ValueError("network_elapsed_s must match the execution timeline")
        if self.critical_packet_delay.packet_id not in {
            item.packet_id for item in self.transmission.packet_routes
        }:
            raise ValueError("critical packet must belong to the route result")
        for name in ("tx_energy_j", "rx_energy_j", "transport_energy_j"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.status is DeliveryStatus.DELIVERED:
            if not self.transmission.all_delivered or not self.observation.completed:
                raise ValueError("delivered application status requires full route delivery")
        elif self.status not in (DeliveryStatus.DROPPED, DeliveryStatus.EXPIRED):
            raise ValueError("application status must be delivered, dropped, or expired")

    @property
    def packet_count(self) -> int:
        """Return the number of packets created for the application payload."""

        return self.transmission.packetization.pdu_set.packet_count

    @property
    def payload_bytes(self) -> int:
        """Return useful application bytes, counted once per request."""

        return self.observation.payload_bytes

    @property
    def on_wire_bytes(self) -> int:
        """Return all transmitted bytes, including overhead and retries."""

        return self.observation.on_wire_bytes

    @property
    def packet_hop_count(self) -> int:
        """Return attempted packet-hop services across the complete route."""

        return self.observation.packet_hop_offered_count

    @property
    def total_network_energy_j(self) -> float:
        """Return transmitter, receiver, and transport energy in joules."""

        return self.tx_energy_j + self.rx_energy_j + self.transport_energy_j


class NetworkTaskAdapter:
    """Run legacy task data through the canonical packet engine.

    This class owns only the legacy-to-canonical boundary. Route selection is
    performed by ``DynamicAccessRouteBuilder`` and physical packet behavior is
    performed by ``NetworkEngine``.
    """

    def __init__(
        self,
        *,
        engine: NetworkEngine,
        units: LegacyUnitBoundary,
    ) -> None:
        if not isinstance(engine, NetworkEngine):
            raise ValueError("engine must be a NetworkEngine")
        if not isinstance(units, LegacyUnitBoundary):
            raise ValueError("units must be a LegacyUnitBoundary")
        self._engine = engine
        self._units = units

    def _create_traffic_object(
        self,
        task: LegacyNetworkTask,
        route: ResolvedRoute,
        *,
        current_time_ms: float,
    ) -> TrafficObject:
        """Create canonical traffic intent from one legacy task."""

        task_id = str(getattr(task, "id", "") or "").strip()
        if not task_id:
            raise ValueError("task.id must be a non-empty string")
        if not isinstance(route, ResolvedRoute):
            raise ValueError("route must be a ResolvedRoute")

        created_at_s = self._units.milliseconds_to_seconds(
            current_time_ms,
            name="current_time_ms",
        )
        deadline_ms = getattr(task, "deadline_ms", None)
        deadline_at_s = None
        if deadline_ms is not None:
            deadline_at_s = created_at_s + self._units.milliseconds_to_seconds(
                deadline_ms,
                name="task.deadline_ms",
            )

        explicit_payload_bytes = getattr(task, "input_payload_bytes", None)
        if explicit_payload_bytes is None:
            payload_bytes = self._units.kilobytes_to_bytes(task.data_size_kb)
        else:
            if (
                isinstance(explicit_payload_bytes, bool)
                or not isinstance(explicit_payload_bytes, int)
                or explicit_payload_bytes < 0
            ):
                raise ValueError("task.input_payload_bytes must be non-negative bytes")
            payload_bytes = explicit_payload_bytes

        return TrafficObject(
            object_id=f"task:{task_id}:request",
            flow_id=f"task:{task_id}",
            source_id=route.source_node_id,
            destination_id=route.destination_node_id,
            created_at_s=created_at_s,
            payload_bytes=payload_bytes,
            deadline_at_s=deadline_at_s,
            plane=Plane.USER,
            traffic_kind=str(
                getattr(task, "application_type", "") or "generic"
            ),
            flow_role="request",
            direction=route.hops[0].hop.direction,
            service_class=str(
                getattr(task, "service_class", "") or "best_effort"
            ),
            measurement_id=task_id,
            service_chain_id=task_id,
        )

    @staticmethod
    def _summarize_transmission(transmission: RouteTransmissionResult):
        """Find terminal time, critical packet delay, outcomes, and status."""

        terminal_transmissions = tuple(
            packet_route.hop_transmissions[-1]
            for packet_route in transmission.packet_routes
        )
        finished_at_s = max(
            item.outcome.ended_at_s for item in terminal_transmissions
        )
        critical_index = max(
            range(len(terminal_transmissions)),
            key=lambda index: terminal_transmissions[index].outcome.ended_at_s,
        )
        critical_packet_delay = packet_route_delay(
            transmission.packet_routes[critical_index]
        )
        outcomes = tuple(
            hop_transmission.outcome
            for packet_route in transmission.packet_routes
            for hop_transmission in packet_route.hop_transmissions
        )
        if transmission.all_delivered:
            status = DeliveryStatus.DELIVERED
        elif any(outcome.status is DeliveryStatus.EXPIRED for outcome in outcomes):
            status = DeliveryStatus.EXPIRED
        else:
            status = DeliveryStatus.DROPPED
        return finished_at_s, critical_packet_delay, outcomes, status

    def execute(
        self,
        task: LegacyNetworkTask,
        route: ResolvedRoute,
        *,
        current_time_ms: float,
        hop_parameters: Mapping[str, HopTransmissionParameters],
    ) -> TaskNetworkExecution:
        """Execute one task over ``route`` and return its observed result.

        ``current_time_ms`` is the legacy absolute clock. ``hop_parameters``
        contains canonical per-hop RF/link inputs. Raises ``ValueError`` for an
        invalid task identity, route, unit conversion, or incomplete parameters.
        """

        traffic_object = self._create_traffic_object(
            task,
            route,
            current_time_ms=current_time_ms,
        )
        transmission = self._engine.transmit_object_route(
            traffic_object,
            route,
            hop_parameters=hop_parameters,
        )
        observation = observe_route(transmission)
        (
            finished_at_s,
            critical_packet_delay,
            outcomes,
            status,
        ) = self._summarize_transmission(transmission)

        return TaskNetworkExecution(
            traffic_object=traffic_object,
            transmission=transmission,
            observation=observation,
            status=status,
            finished_at_s=finished_at_s,
            network_elapsed_s=finished_at_s - traffic_object.created_at_s,
            critical_packet_delay=critical_packet_delay,
            tx_energy_j=sum(item.tx_energy_j for item in outcomes),
            rx_energy_j=sum(item.rx_energy_j for item in outcomes),
            transport_energy_j=sum(item.transport_energy_j for item in outcomes),
        )
