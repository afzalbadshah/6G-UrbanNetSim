"""Execute one chosen task and produce its authoritative physical observation.

The scheduler may estimate several actions, but :class:`TaskExecutionRuntime`
runs only the selected one. Its stages are: access wait, route construction,
packet transmission, compute execution, then delay/energy/cost accounting. The
immutable result is consumed by reward, learning, batteries, status, and reports.

How to extend: change packet/channel behavior in ``network/``, physical formulas
in ``metrics/``, and tunable values in ``config/``. This orchestrator should only
connect those components. See ``docs/PACKET_EXECUTION_GUIDE.md`` first.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from types import MappingProxyType, SimpleNamespace
from typing import Mapping

from config import SERVER_CONFIG
from config.energy import POWER_PROFILES
from config.units import LEGACY_V1_RESOURCE_CONVERSION
from core.network_adapter import (
    LegacyUnitBoundary,
    NetworkTaskAdapter,
    TaskNetworkExecution,
)
from core.network_runtime import (
    DynamicAccessRouteBuilder,
    RuntimeHopParameterBuilder,
    build_channel_model_registry,
    build_network_engine,
    build_shared_link_registry,
)
from metrics.cost import OperatorCost, linear_cost
from config.emissions import CARBON_INTENSITY_G_CO2_PER_JOULE
from config.servers import ELECTRICITY_PRICE_PER_JOULE, tier_prices
from config.zones import SAT_NODE_IDS, SATELLITE_CONFIG
from metrics.delay import TaskDelayBreakdown, calculate_task_delay
from metrics.emissions import operational_emissions_g
from metrics.energy import (
    TaskEnergyBreakdown,
    calculate_task_energy,
    compute_energy,
    energy_from_power_duration_j,
    incremental_compute_energy,
)
from network import ChannelResult, DeliveryStatus, Medium, NetworkEngine, PhysicalTopology, Position3D


def _nonnegative(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _ratio(name: str, value: float) -> float:
    result = _nonnegative(name, value)
    if result > 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return result


# Backward-compatible import name. The calculation and data now live together
# in metrics.delay; this module only orchestrates the selected task execution.
RealizedDelay = TaskDelayBreakdown
RealizedEnergy = TaskEnergyBreakdown


@dataclass(frozen=True, slots=True)
class RealizedTaskExecution:
    """Single immutable result for one chosen task execution.

    This is the hand-off between physical execution and the scheduler lifecycle.
    It carries timing, delivery, packet/byte counts, energy, cost completeness,
    and exact per-server battery debits.
    """

    task_id: str
    network_status: DeliveryStatus | None
    application_completed: bool
    deadline_missed: bool
    started_at_ms: float
    finished_at_ms: float
    channel_results: tuple[ChannelResult, ...]
    delay: TaskDelayBreakdown
    energy: TaskEnergyBreakdown
    operator_cost: OperatorCost
    payload_bytes: int
    on_wire_bytes: int
    packet_count: int
    packet_hop_count: int
    server_battery_debits_j: Mapping[str, float]
    network_execution: TaskNetworkExecution | None = None
    failure_reason: str | None = None
    server_hover_energy_j: float = 0.0
    # Operational CO2e (grams) for this task's grid-drawn energy -- same
    # scope as operator_cost.energy_cost (see config/emissions.py /
    # metrics/emissions.py). Embodied/manufacturing CO2e is deployment-level,
    # not per-task, and is reported separately.
    operational_co2e_g: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        if not isinstance(self.channel_results, tuple) or any(
            not isinstance(result, ChannelResult) for result in self.channel_results
        ):
            raise ValueError("channel_results must contain ChannelResult observations")
        if not isinstance(self.delay, TaskDelayBreakdown):
            raise ValueError("delay must be a TaskDelayBreakdown")
        if not isinstance(self.energy, TaskEnergyBreakdown):
            raise ValueError("energy must be a TaskEnergyBreakdown")
        if not isinstance(self.operator_cost, OperatorCost):
            raise ValueError("operator_cost must be an OperatorCost")
        start = _nonnegative("started_at_ms", self.started_at_ms)
        finish = _nonnegative("finished_at_ms", self.finished_at_ms)
        if finish < start:
            raise ValueError("finished_at_ms cannot precede started_at_ms")
        if not isclose(finish - start, self.delay.total_ms, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError("execution timeline must match total delay")
        for name in ("payload_bytes", "on_wire_bytes", "packet_count", "packet_hop_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        debits = dict(self.server_battery_debits_j)
        if any(not isinstance(key, str) or not key.strip() for key in debits):
            raise ValueError("server battery debit IDs must be non-empty strings")
        for server_id, value in debits.items():
            debits[server_id] = _nonnegative(
                f"server_battery_debits_j[{server_id!r}]", value
            )
        object.__setattr__(self, "server_battery_debits_j", MappingProxyType(debits))
        if self.application_completed and self.network_status not in (
            None,
            DeliveryStatus.DELIVERED,
        ):
            raise ValueError("a completed application cannot have failed network delivery")
        if not self.application_completed and not self.failure_reason:
            raise ValueError("an incomplete application requires a failure reason")

    @property
    def operator_cost_total(self) -> float | None:
        """Return complete operator cost, or ``None`` when it is incomplete."""

        if not self.operator_cost.complete:
            return None
        return self.operator_cost.total


@dataclass(frozen=True, slots=True)
class _RemoteEnergyAccounting:
    """Internal bundle passed from energy accounting to result construction."""

    realized_energy: RealizedEnergy
    server_battery_debits_j: Mapping[str, float]
    access_wire_bytes: int
    backbone_wire_bytes: int


_COMPUTE_PROFILE_BY_TIER = {
    "LOCAL_DEVICE": "onboard_compute",
    "MEC": "mec_compute",
    "METRO": "metro_compute",
    "CLOUD": "cloud_compute",
    "UAV_BS": "uav_bs_compute",
}

class TaskExecutionRuntime:
    """Orchestrate one selected local or remote task from start to observation.

    One instance is reused for a simulation run so devices share the same packet
    queues and ledger. The class does not choose the server or mutate scheduler
    policy state.
    """

    def __init__(
        self,
        topology: PhysicalTopology,
        *,
        experiment_seed: int | str = 0,
        engine: NetworkEngine | None = None,
        units: LegacyUnitBoundary | None = None,
        power_profiles: Mapping[str, Mapping[str, object]] = POWER_PROFILES,
    ) -> None:
        if not isinstance(topology, PhysicalTopology):
            raise ValueError("topology must be a PhysicalTopology")
        self._topology = topology
        self._units = units or LegacyUnitBoundary(bytes_per_legacy_kb=1000)
        self._engine = engine or build_network_engine(
            topology, experiment_seed=experiment_seed
        )
        self._adapter = NetworkTaskAdapter(engine=self._engine, units=self._units)
        # Same registry shape build_network_engine's own channel_model=
        # composition produces (build_channel_model_registry over a shared-link
        # registry for this topology): ChannelModel instances are pure/stateless,
        # so this independent build is safe to use purely as a read-only outage
        # probe, whether or not `engine` above was supplied by the caller.
        self._routes = DynamicAccessRouteBuilder(
            topology,
            channel_models=build_channel_model_registry(
                topology, build_shared_link_registry(topology)
            ),
        )
        self._parameters = RuntimeHopParameterBuilder(
            topology, power_profiles=power_profiles
        )
        self._power_profiles = {
            key: dict(value) for key, value in power_profiles.items()
        }

    @property
    def engine(self) -> NetworkEngine:
        """Expose the run's shared packet engine for diagnostics and snapshots."""

        return self._engine

    def _task_payload_bytes(self, task: object) -> int:
        """Return configured input bytes, falling back to the legacy KB boundary."""

        explicit_bytes = getattr(task, "input_payload_bytes", None)
        if explicit_bytes is not None:
            if isinstance(explicit_bytes, bool) or not isinstance(explicit_bytes, int):
                raise ValueError("task.input_payload_bytes must be an integer")
            if explicit_bytes < 0:
                raise ValueError("task.input_payload_bytes must be non-negative")
            return explicit_bytes
        return self._units.kilobytes_to_bytes(task.data_size_kb)

    @staticmethod
    def _reference_cpu_seconds(task: object, *, compute_completed: bool) -> float:
        """Return delivered compute as reference-vCPU seconds (0 if unfinished).

        This is the compute quantity the operator compute price is charged on.
        """

        if not compute_completed:
            return 0.0
        work = getattr(task, "resource_demand", None)
        if work is None:
            work = LEGACY_V1_RESOURCE_CONVERSION.work_from_legacy(
                cpu_demand=float(getattr(task, "cpu_demand", 0.0) or 0.0),
                memory_demand=float(getattr(task, "memory_demand", 0.0) or 0.0),
                storage_demand=float(getattr(task, "storage_demand", 0.0) or 0.0),
            )
        return LEGACY_V1_RESOURCE_CONVERSION.cpu_seconds_for(work)

    @staticmethod
    def _tier(server: object) -> str:
        name = str(getattr(server, "name", "") or "")
        if name.startswith("LocalDevice"):
            return "LOCAL_DEVICE"
        tier = str(getattr(server, "tier", "") or "").upper()
        if tier in _COMPUTE_PROFILE_BY_TIER:
            return tier
        for candidate in ("UAV_BS", "MEC", "METRO", "CLOUD"):
            if name.upper().startswith(candidate):
                return candidate
        raise ValueError(f"cannot map server {name!r} to a compute power profile")

    @staticmethod
    def _channel_results(
        network: TaskNetworkExecution,
    ) -> tuple:
        """Return the exact per-attempt channel observations used for delivery."""

        return tuple(
            attempt.channel_result
            for packet_route in network.transmission.packet_routes
            for hop_result in packet_route.hop_transmissions
            for attempt in hop_result.attempts
        )

    def _local_execution(
        self,
        task: object,
        *,
        current_time_ms: float,
        processing_duration_ms: float,
        utilization_after_ratio: float,
    ) -> RealizedTaskExecution:
        """Execute local compute with no route, packets, or network energy."""

        duration_ms = _nonnegative("processing_duration_ms", processing_duration_ms)
        device_power_w = float(getattr(task, "local_power_w", 0.0) or 0.0)
        if device_power_w > 0.0:
            # Device profiles declare their active MCU/SoC power directly.  The
            # one physical formula remains power x duration in metrics.energy.
            compute_it_energy_j = energy_from_power_duration_j(
                power_w=device_power_w,
                duration_s=duration_ms / 1000.0,
            )
            facility_overhead_energy_j = 0.0
        else:
            # Compatibility path for callers that do not carry a device profile.
            energy = compute_energy(
                self._power_profiles["onboard_compute"],
                duration_s=duration_ms / 1000.0,
                utilization_ratio=_ratio(
                    "utilization_after_ratio", utilization_after_ratio
                ),
            )
            compute_it_energy_j = energy.it_energy_j
            facility_overhead_energy_j = energy.facility_overhead_energy_j
        realized_energy = calculate_task_energy(
            device_tx_energy_j=0.0,
            device_rx_energy_j=0.0,
            device_wait_energy_j=0.0,
            infrastructure_network_energy_j=0.0,
            compute_it_energy_j=compute_it_energy_j,
            facility_overhead_energy_j=facility_overhead_energy_j,
            local_execution=True,
        )
        delay = calculate_task_delay(
            processing_delay_s=duration_ms / 1000.0,
        )
        deadline = getattr(task, "deadline_ms", None)
        missed = deadline is not None and delay.total_ms > float(deadline)
        return RealizedTaskExecution(
            task_id=str(getattr(task, "id", "")),
            network_status=None,
            application_completed=True,
            deadline_missed=missed,
            started_at_ms=current_time_ms,
            finished_at_ms=current_time_ms + delay.total_ms,
            channel_results=(),
            delay=delay,
            energy=realized_energy,
            operator_cost=OperatorCost(0.0, 0.0, 0.0),
            payload_bytes=self._task_payload_bytes(task),
            on_wire_bytes=0,
            packet_count=0,
            packet_hop_count=0,
            server_battery_debits_j={},
        )

    def _network_energy_and_debits(
        self,
        execution: TaskNetworkExecution,
        *,
        task: object,
        processing_server_id: str,
        compute_energy_j: float,
    ) -> tuple[float, float, float, dict[str, float], int, int]:
        device_tx_j = 0.0
        device_rx_j = 0.0
        infrastructure_j = 0.0
        debits: dict[str, float] = {}
        access_wire_bytes = 0
        backbone_wire_bytes = 0
        route = execution.transmission.route
        for packet_route in execution.transmission.packet_routes:
            for index, hop_result in enumerate(packet_route.hop_transmissions):
                outcome = hop_result.outcome
                resolved_hop = route.hops[index]
                if index == 0:
                    device_tx_power_w = float(
                        getattr(task, "tx_power_w", 0.0) or 0.0
                    )
                    if device_tx_power_w > 0.0:
                        access_active_s = sum(
                            attempt.reservation.transmission_delay_s
                            for attempt in hop_result.attempts
                            if attempt.reservation is not None
                        )
                        device_tx_j += energy_from_power_duration_j(
                            power_w=device_tx_power_w,
                            duration_s=access_active_s,
                        )
                    else:
                        device_tx_j += outcome.tx_energy_j
                    device_rx_j += 0.0
                    infrastructure_j += outcome.rx_energy_j + outcome.transport_energy_j
                    access_wire_bytes += outcome.on_wire_bytes
                    receiver_id = resolved_hop.hop.destination_node_id
                    debits[receiver_id] = debits.get(receiver_id, 0.0) + outcome.rx_energy_j
                else:
                    infrastructure_j += (
                        outcome.tx_energy_j
                        + outcome.rx_energy_j
                        + outcome.transport_energy_j
                    )
                    backbone_wire_bytes += outcome.on_wire_bytes
                    if resolved_hop.link.medium is Medium.RADIO:
                        source_id = resolved_hop.hop.source_node_id
                        destination_id = resolved_hop.hop.destination_node_id
                        debits[source_id] = debits.get(source_id, 0.0) + outcome.tx_energy_j
                        debits[destination_id] = debits.get(destination_id, 0.0) + outcome.rx_energy_j
        debits[processing_server_id] = (
            debits.get(processing_server_id, 0.0) + compute_energy_j
        )
        return (
            device_tx_j,
            device_rx_j,
            infrastructure_j,
            debits,
            access_wire_bytes,
            backbone_wire_bytes,
        )

    def _transmit_remote_task(
        self,
        task: object,
        *,
        destination_node_id: str,
        source_position: Position3D,
        access_node_id: str,
        position_overrides: Mapping[str, Position3D] | None,
        network_start_time_ms: float,
    ) -> TaskNetworkExecution:
        """Build the route and transfer the task without SLA-based expiry.

        ``task.deadline_ms`` is an application SLA, not a transport kill
        switch.  The complete observed execution is compared with that SLA
        after delivery and processing, so a late task retains its full delay,
        energy, byte, and cost record.
        """

        route = self._routes.build(
            source_node_id=(
                f"device:{getattr(task, 'device_id', getattr(task, 'entity_id', task.id))}"
            ),
            source_position=source_position,
            access_node_id=access_node_id,
            destination_node_id=destination_node_id,
            position_overrides=position_overrides,
            access_interference_dbm=getattr(task, "interference_dbm", None),
        )
        hop_parameters = self._parameters.build(
            route,
            source_position=source_position,
            position_overrides=position_overrides,
            access_interference_dbm=getattr(task, "interference_dbm", None),
        )
        network_task = SimpleNamespace(
            id=str(getattr(task, "id", "")),
            data_size_kb=task.data_size_kb,
            input_payload_bytes=self._task_payload_bytes(task),
            deadline_ms=None,
            service_class=str(getattr(task, "service_class", "") or "best_effort"),
            application_type=str(
                getattr(task, "application_type", "") or "generic"
            ),
        )
        return self._adapter.execute(
            network_task,
            route,
            current_time_ms=network_start_time_ms,
            hop_parameters=hop_parameters,
        )

    def _compute_energy_after_delivery(
        self,
        *,
        network_delivered: bool,
        processing_duration_ms: float,
        compute_tier: str,
        utilization_before_ratio: float,
        utilization_after_ratio: float,
    ) -> tuple[float, float, float]:
        """Return actual processing ms, IT joules, and facility joules."""

        if not network_delivered:
            return 0.0, 0.0, 0.0
        compute = incremental_compute_energy(
            self._power_profiles[_COMPUTE_PROFILE_BY_TIER[compute_tier]],
            duration_s=processing_duration_ms / 1000.0,
            utilization_before_ratio=utilization_before_ratio,
            utilization_after_ratio=utilization_after_ratio,
        )
        return (
            processing_duration_ms,
            compute.incremental_it_energy_j,
            compute.facility_overhead_energy_j,
        )

    def _account_remote_energy(
        self,
        network: TaskNetworkExecution,
        *,
        task: object,
        processing_server_id: str,
        pre_access_wait_ms: float,
        processing_duration_ms: float,
        compute_it_energy_j: float,
        facility_overhead_energy_j: float,
    ) -> _RemoteEnergyAccounting:
        """Combine observed packet energy, device waiting, and compute energy."""

        compute_total_energy_j = compute_it_energy_j + facility_overhead_energy_j
        (
            device_tx_energy_j,
            device_rx_energy_j,
            infrastructure_network_energy_j,
            server_battery_debits_j,
            access_wire_bytes,
            backbone_wire_bytes,
        ) = self._network_energy_and_debits(
            network,
            task=task,
            processing_server_id=processing_server_id,
            compute_energy_j=compute_total_energy_j,
        )
        access_active_s = sum(
            attempt.reservation.transmission_delay_s
            for packet_route in network.transmission.packet_routes
            for attempt in packet_route.hop_transmissions[0].attempts
            if attempt.reservation is not None
        )
        total_duration_ms = (
            pre_access_wait_ms
            + network.network_elapsed_s * 1000.0
            + processing_duration_ms
        )
        device_wait_s = max(0.0, total_duration_ms / 1000.0 - access_active_s)
        device_idle_power_w = float(getattr(task, "idle_power_w", 0.0) or 0.0)
        if device_idle_power_w <= 0.0:
            device_idle_power_w = float(
                self._power_profiles["device_radio_default"]["idle_power_w"]
            )
        device_wait_energy_j = energy_from_power_duration_j(
            power_w=device_idle_power_w,
            duration_s=device_wait_s,
        )
        return _RemoteEnergyAccounting(
            realized_energy=calculate_task_energy(
                device_tx_energy_j=device_tx_energy_j,
                device_rx_energy_j=device_rx_energy_j,
                device_wait_energy_j=device_wait_energy_j,
                infrastructure_network_energy_j=infrastructure_network_energy_j,
                compute_it_energy_j=compute_it_energy_j,
                facility_overhead_energy_j=facility_overhead_energy_j,
                local_execution=False,
            ),
            server_battery_debits_j=server_battery_debits_j,
            access_wire_bytes=access_wire_bytes,
            backbone_wire_bytes=backbone_wire_bytes,
        )

    @staticmethod
    def _network_failure_reason(network: TaskNetworkExecution) -> str | None:
        """Return the last observed non-delivery reason, if delivery failed."""

        if network.status is DeliveryStatus.DELIVERED:
            return None
        terminal_failure = max(
            (
                hop_result.outcome
                for packet_route in network.transmission.packet_routes
                for hop_result in packet_route.hop_transmissions
                if hop_result.outcome.status is not DeliveryStatus.DELIVERED
            ),
            key=lambda outcome: outcome.ended_at_s,
            default=None,
        )
        return (
            terminal_failure.failure_reason
            if terminal_failure is not None
            else network.status.value
        )

    def execute(
        self,
        task: object,
        server: object,
        *,
        current_time_ms: float,
        processing_duration_ms: float,
        utilization_before_ratio: float,
        utilization_after_ratio: float,
        source_position: Position3D,
        access_node_id: str | None,
        position_overrides: Mapping[str, Position3D] | None = None,
    ) -> RealizedTaskExecution:
        """Execute the chosen task and return an immutable realized result.

        Times use legacy milliseconds at this boundary; positions use metres and
        utilization values use ratios in ``[0, 1]``. The method raises
        ``ValueError`` for invalid units, topology access, profiles, or server
        types. Local execution bypasses the network explicitly.
        """

        execution_start_ms = _nonnegative("current_time_ms", current_time_ms)
        requested_processing_ms = _nonnegative(
            "processing_duration_ms", processing_duration_ms
        )
        utilization_before = _ratio(
            "utilization_before_ratio", utilization_before_ratio
        )
        utilization_after = _ratio(
            "utilization_after_ratio", utilization_after_ratio
        )
        if utilization_after < utilization_before:
            raise ValueError("utilization_after_ratio cannot be below utilization_before_ratio")
        compute_tier = self._tier(server)

        # Local execution deliberately skips route construction and packet
        # accounting. This makes "no network was used" explicit in the result.
        if compute_tier == "LOCAL_DEVICE":
            return self._local_execution(
                task,
                current_time_ms=execution_start_ms,
                processing_duration_ms=requested_processing_ms,
                utilization_after_ratio=utilization_after,
            )

        if not isinstance(access_node_id, str) or not access_node_id.strip():
            raise ValueError("remote execution requires a non-empty access_node_id")
        if not isinstance(source_position, Position3D):
            raise ValueError("source_position must be a Position3D")

        payload_bytes = self._task_payload_bytes(task)
        handover_delay_s = _nonnegative(
            "handover_delay_ms",
            float(getattr(task, "handover_delay_ms", 0.0) or 0.0),
        ) / 1000.0
        handover_interruption_s = _nonnegative(
            "handover_interruption_ms",
            float(getattr(task, "handover_interruption_ms", 0.0) or 0.0),
        ) / 1000.0
        coverage_outage_duration_s = 0.0
        if bool(getattr(task, "coverage_outage_flag", False)):
            coverage_outage_duration_s = _nonnegative(
                "coverage_outage_duration_ms",
                float(getattr(task, "coverage_outage_duration_ms", 0.0) or 0.0),
            ) / 1000.0

        access_delay = calculate_task_delay(
            handover_delay_s=handover_delay_s,
            handover_interruption_s=handover_interruption_s,
            coverage_outage_duration_s=coverage_outage_duration_s,
        )
        pre_access_wait_ms = access_delay.pre_access_wait_ms
        task_deadline = getattr(task, "deadline_ms", None)
        deadline_ms = None if task_deadline is None else _nonnegative(
            "task.deadline_ms", float(task_deadline)
        )

        destination_node_id = str(getattr(server, "name", "") or "")
        network_execution = self._transmit_remote_task(
            task,
            destination_node_id=destination_node_id,
            source_position=source_position,
            access_node_id=access_node_id,
            position_overrides=position_overrides,
            network_start_time_ms=execution_start_ms + pre_access_wait_ms,
        )
        (
            actual_processing_ms,
            compute_it_energy_j,
            facility_overhead_energy_j,
        ) = self._compute_energy_after_delivery(
            network_delivered=(
                network_execution.status is DeliveryStatus.DELIVERED
            ),
            processing_duration_ms=requested_processing_ms,
            compute_tier=compute_tier,
            utilization_before_ratio=utilization_before,
            utilization_after_ratio=utilization_after,
        )
        energy_accounting = self._account_remote_energy(
            network_execution,
            task=task,
            processing_server_id=destination_node_id,
            pre_access_wait_ms=pre_access_wait_ms,
            processing_duration_ms=actual_processing_ms,
            compute_it_energy_j=compute_it_energy_j,
            facility_overhead_energy_j=facility_overhead_energy_j,
        )
        delay = calculate_task_delay(
            critical_packet_delay=network_execution.critical_packet_delay,
            route=network_execution.transmission.route,
            network_elapsed_s=network_execution.network_elapsed_s,
            handover_delay_s=handover_delay_s,
            handover_interruption_s=handover_interruption_s,
            coverage_outage_duration_s=coverage_outage_duration_s,
            processing_delay_s=actual_processing_ms / 1000.0,
        )
        grid_energy_j = (
            energy_accounting.realized_energy.infrastructure_network_energy_j
            + compute_it_energy_j
            + facility_overhead_energy_j
        )
        compute_completed = network_execution.status is DeliveryStatus.DELIVERED
        # UAV cells physically hover while they host the task, so the operator
        # spends hover energy for the compute duration (single hover formula).
        server_hover_energy_j = 0.0
        if compute_tier == "UAV_BS" and compute_completed:
            hover_power_w = float(
                SERVER_CONFIG.get("UAV_BS", {}).get("hover_power_w", 0.0) or 0.0
            )
            server_hover_energy_j = energy_from_power_duration_j(
                power_w=hover_power_w,
                duration_s=actual_processing_ms / 1000.0,
            )
        # Operator cost, priced straight from config/servers.py (one source of
        # truth). Each tier block declares its compute and transport price; the
        # electricity price is the shared ELECTRICITY_PRICE_PER_JOULE. The
        # originating device (LOCAL_DEVICE / "Onboard") has zero prices, so
        # onboard execution is free to the operator.
        compute_price_per_cpu_s, transport_price_per_byte = tier_prices(compute_tier)
        if access_node_id in SAT_NODE_IDS:
            # LEO satellite capacity is billed per byte, far above fibre.
            transport_price_per_byte = float(
                SATELLITE_CONFIG.get("transport_price_per_byte", transport_price_per_byte)
            )
        operator_cpu_seconds = (
            self._reference_cpu_seconds(task, compute_completed=compute_completed)
            if compute_tier != "LOCAL_DEVICE"
            else 0.0
        )
        operator_cost = OperatorCost(
            compute_cost=linear_cost(
                measured_quantity=operator_cpu_seconds, price_per_unit=compute_price_per_cpu_s
            ),
            transport_cost=linear_cost(
                measured_quantity=float(network_execution.on_wire_bytes),
                price_per_unit=transport_price_per_byte,
            ),
            energy_cost=linear_cost(
                measured_quantity=grid_energy_j, price_per_unit=ELECTRICITY_PRICE_PER_JOULE
            ),
        )
        operational_co2e_g = operational_emissions_g(
            energy_j=grid_energy_j, carbon_intensity_g_per_j=CARBON_INTENSITY_G_CO2_PER_JOULE
        )
        application_completed = (
            network_execution.status is DeliveryStatus.DELIVERED
        )
        deadline_missed = deadline_ms is not None and delay.total_ms > deadline_ms
        return RealizedTaskExecution(
            task_id=str(getattr(task, "id", "")),
            network_status=network_execution.status,
            application_completed=application_completed,
            # A delivery failure and an SLA miss are independent observations.
            # Keeping them separate lets the CSV distinguish an early channel
            # failure from a task that really exceeded its application budget.
            deadline_missed=deadline_missed,
            started_at_ms=execution_start_ms,
            finished_at_ms=execution_start_ms + delay.total_ms,
            channel_results=self._channel_results(network_execution),
            delay=delay,
            energy=energy_accounting.realized_energy,
            operator_cost=operator_cost,
            payload_bytes=payload_bytes,
            on_wire_bytes=network_execution.on_wire_bytes,
            packet_count=network_execution.packet_count,
            packet_hop_count=network_execution.packet_hop_count,
            server_battery_debits_j=energy_accounting.server_battery_debits_j,
            network_execution=network_execution,
            failure_reason=self._network_failure_reason(network_execution),
            server_hover_energy_j=server_hover_energy_j,
            operational_co2e_g=operational_co2e_g,
        )


__all__ = (
    "RealizedDelay",
    "RealizedEnergy",
    "RealizedTaskExecution",
    "TaskExecutionRuntime",
)
