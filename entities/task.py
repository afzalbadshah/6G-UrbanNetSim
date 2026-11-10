"""Mutable application task state with explicit physical resource demands.

Author: Dr. Afzal Badshah
Department of Software Engineering, University of Sargodha
Website: https://afzalbadshah.com/index.php/afzal-badshah/
LinkedIn: https://www.linkedin.com/in/afzal-badshah-phd-305b03164/
Discussion Forum: resp.afzalbadshah.com

Notes:
- Legacy demand fields remain available for saved policies and external callers.
- Mobility fields are OPTIONAL and default to None/0.0 so legacy flows keep working.

How to extend: add new immutable resource quantities to ``ComputeWork`` and keep
packet/network measurements separate from compute demand.
"""

import uuid
from typing import Optional
from config import WORKLOAD, SERVER_CONFIG
from config.units import LEGACY_V1_RESOURCE_CONVERSION
from entities.resources import ComputeWork


class Task:
    """
    A single workload item produced by an IoT device.
    Holds resource demands, size, priority, status, timing, and basic metrics.

    Mobility-aware fields (optional):
      - round:             simulation round index (int)
      - pos_x, pos_y:      device position at creation time (meters)
      - distance_to_bs:    distance to serving PEaaS/BS (meters)
      - signal_dbm:        simple signal strength proxy (dBm)
      - handover_delay_ms: extra delay if serving BS changed for this task (ms)
    """

    def __init__(self,
                 cpu_demand: float,
                 storage_demand: float,
                 memory_demand: float,
                 bandwidth: float,
                 latency: float,
                 priority: int = 1,
                 *,
                 # Optional mobility/context (safe defaults for legacy code)
                 round_index: Optional[int] = None,
                 pos_x: Optional[float] = None,
                 pos_y: Optional[float] = None,
                 distance_to_bs: Optional[float] = None,
                 signal_dbm: Optional[float] = None,
                 handover_delay_ms: float = 0.0,
                 device_id: Optional[int] = None,
                 device_type: str = "generic",
                 service_class: str = "",
                 device_profile: Optional[dict] = None,
                 mobility_profile: str = "unknown",
                 local_compute_capable: bool = False,
                 local_cpu_capacity: float = 0.0,
                 local_memory_capacity: float = 0.0,
                 local_power_w: float = 0.0,
                 battery_capacity_j: float = 0.0,
                 battery_remaining_j: float = 0.0,
                 tx_power_w: float = 0.0,
                 rx_power_w: float = 0.0,
                 idle_power_w: float = 0.0,
                 propulsion_power_w: float = 0.0,
                 movement_speed_m_s: float = 0.0,
                 handover_sensitivity: float = 0.0,
                 dwell_weight: float = 1.0,
                 deadline_ms: Optional[float] = None,
                 sla_profile_id: str = "",
                 sla_generation: str = "",
                 radio_latency_target_ms: Optional[float] = None,
                 application_deadline_ms: Optional[float] = None,
                 sla_maturity: str = "",
                 sla_source_id: str = "",
                 application_type: str = "generic",
                 compute_work_cycles: Optional[float] = None,
                 memory_working_set_bytes: Optional[int] = None,
                 storage_footprint_bytes: Optional[int] = None,
                 input_payload_bytes: Optional[int] = None):
        # Identity
        self.id = str(uuid.uuid4())

        # These names are compatibility aliases used by older policies. Their
        # unit conversion is explicit below; physical formulas never consume
        # the bare values directly.
        self.cpu_demand = cpu_demand
        self.storage_demand = storage_demand
        self.memory_demand = memory_demand

        converted_work = LEGACY_V1_RESOURCE_CONVERSION.work_from_legacy(
            cpu_demand=cpu_demand,
            memory_demand=memory_demand,
            storage_demand=storage_demand,
        )
        self.resource_demand = ComputeWork(
            compute_work_cycles=(
                converted_work.compute_work_cycles
                if compute_work_cycles is None
                else compute_work_cycles
            ),
            memory_working_set_bytes=(
                converted_work.memory_working_set_bytes
                if memory_working_set_bytes is None
                else memory_working_set_bytes
            ),
            storage_footprint_bytes=(
                converted_work.storage_footprint_bytes
                if storage_footprint_bytes is None
                else storage_footprint_bytes
            ),
        )
        self.resource_allocation = LEGACY_V1_RESOURCE_CONVERSION.allocation_for(
            self.resource_demand
        )
        self.compute_work_cycles = self.resource_demand.compute_work_cycles
        self.compute_rate_demand_cycles_per_s = (
            self.resource_allocation.allocated_compute_rate_cycles_per_s
        )
        self.memory_working_set_bytes = self.resource_demand.memory_working_set_bytes
        self.storage_footprint_bytes = self.resource_demand.storage_footprint_bytes
        self.resource_conversion_profile_id = (
            LEGACY_V1_RESOURCE_CONVERSION.profile_id
        )

        # Network characteristics at creation (used by schedulers/metrics)
        self.bandwidth = bandwidth
        self.latency = latency

        # Priority / flag
        self.priority = priority
        self.flag = priority  # kept for backward compatibility
        self.device_id = device_id
        self.device_type = str(device_type or "generic")
        self.service_class = str(service_class or "")
        self.device_profile = dict(device_profile or {})
        self.mobility_profile = str(mobility_profile or "unknown")
        # On-device compute is per-device-type (each profile's own
        # local_compute.cpu/memory/power_w, see devices_config.py) -- but
        # ONLY for devices that actually have onboard computing
        # (local_compute_capable). Callers normally pass the values in
        # (workload/generator.py, mobility/mobile_entity.py); if a capable
        # caller doesn't, fall back to the universal Onboard spec
        # (config.SERVER_CONFIG["Onboard"]). Incapable devices (MCU-class
        # sensors) always carry zeros so no local execution space is ever
        # created for them.
        self.local_compute_capable = bool(local_compute_capable)
        if self.local_compute_capable:
            _onboard = SERVER_CONFIG.get("Onboard", {}) or {}
            self.local_cpu_capacity = float(local_cpu_capacity or _onboard.get("cpu", 0.0) or 0.0)
            self.local_memory_capacity = float(local_memory_capacity or _onboard.get("memory", 0.0) or 0.0)
            self.local_power_w = float(local_power_w or _onboard.get("power_w", 0.0) or 0.0)
            local_capacity = LEGACY_V1_RESOURCE_CONVERSION.capacity_from_legacy(
                cpu_capacity=self.local_cpu_capacity,
                memory_capacity=self.local_memory_capacity,
                storage_capacity=max(float(storage_demand or 0.0), 1.0),
            )
            self.local_compute_rate_cycles_per_s = (
                local_capacity.compute_rate_cycles_per_s
            )
            self.local_memory_capacity_bytes = local_capacity.memory_capacity_bytes
        else:
            self.local_cpu_capacity = 0.0
            self.local_memory_capacity = 0.0
            self.local_power_w = 0.0
            self.local_compute_rate_cycles_per_s = 0.0
            self.local_memory_capacity_bytes = 0
        self.battery_capacity_j = float(battery_capacity_j or 0.0)
        self.battery_remaining_j = float(battery_remaining_j or 0.0)
        self.tx_power_w = float(tx_power_w or 0.0)
        self.rx_power_w = float(rx_power_w or 0.0)
        self.idle_power_w = float(idle_power_w or 0.0)
        self.propulsion_power_w = float(propulsion_power_w or 0.0)
        self.movement_speed_m_s = float(movement_speed_m_s or 0.0)
        self.handover_sensitivity = float(handover_sensitivity or 0.0)
        self.dwell_weight = float(dwell_weight or 1.0)
        resolved_deadline_ms = (
            deadline_ms
            if application_deadline_ms is None
            else application_deadline_ms
        )
        self.application_deadline_ms = (
            None if resolved_deadline_ms is None else float(resolved_deadline_ms)
        )
        # Compatibility alias consumed by existing reward, masks and reports.
        self.deadline_ms = self.application_deadline_ms
        self.sla_profile_id = str(sla_profile_id or "")
        self.sla_generation = str(sla_generation or "")
        self.radio_latency_target_ms = (
            None
            if radio_latency_target_ms is None
            else float(radio_latency_target_ms)
        )
        self.sla_maturity = str(sla_maturity or "")
        self.sla_source_id = str(sla_source_id or "")
        self.application_type = str(application_type or self.device_type)

        # Assignment fields
        self.server_assigned = None      # kept for compatibility with older code
        self.assigned_server = None      # current field used by set_server()

        # Data size (central knob from WORKLOAD)
        self.data_size_kb = WORKLOAD["data_per_device_kb"]
        default_payload_bytes = round(float(self.data_size_kb) * 1000.0)
        self.input_payload_bytes = (
            default_payload_bytes
            if input_payload_bytes is None
            else int(input_payload_bytes)
        )
        if self.input_payload_bytes < 0:
            raise ValueError("input_payload_bytes must be non-negative")

        # Execution timeline
        self.execution_start_time = None
        self.execution_end_time = None
        self.status = "created"

        # Basic per-task metrics (filled by simulator/scheduler)
        self.delay = 0.0
        self.cost = 0.0
        self.communication_cost = 0.0
        self.processing_cost = 0.0
        self.operation_energy_cost = 0.0
        self.energy = 0.0
        self.device_tx_energy_j = 0.0
        self.device_rx_energy_j = 0.0
        self.device_local_compute_energy_j = 0.0
        self.device_wait_energy_j = 0.0
        self.device_propulsion_energy_j = 0.0
        self.device_paid_energy_j = 0.0
        self.infra_compute_energy_j = 0.0
        self.backhaul_energy_j = 0.0
        self.total_energy_j = 0.0
        self.local_compute_used = False
        self.handover_risk = 0.0
        self.mobility_violation = 0

        # Realized-execution fields.  These remain unset until the chosen action
        # runs; scheduler estimates must never be presented under these names.
        self.measurement_basis = "unobserved"
        self.network_status = ""
        self.network_failure_reason = ""
        self.payload_bytes = 0
        self.on_wire_bytes = 0
        self.packet_count = 0
        self.packet_hop_count = 0
        self.pre_access_wait_ms = 0.0
        self.network_queue_delay_ms = 0.0
        self.network_transmission_delay_ms = 0.0
        self.network_propagation_delay_ms = 0.0
        self.network_retry_backoff_ms = 0.0
        self.network_outage_observation_ms = 0.0
        self.network_terminal_wait_ms = 0.0
        self.network_delay_ms = 0.0
        self.processing_delay_ms = 0.0
        self.completion_delay_ms = 0.0
        self.legacy_objective_delay_ms = 0.0
        self.deadline_missed = False
        self.cost_accounting_perspective = ""
        self.cost_complete = False
        self.cost_incomplete_reason = "unobserved"

        # ---- Mobility / smart-city context (optional) ----
        self.round = round_index                  # used by reporter/scheduler logs
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.distance_to_bs = distance_to_bs      # meters; used by mobility-aware propagation
        self.signal_dbm = signal_dbm              # optional for plotting/HO logic
        self.handover_delay_ms = float(handover_delay_ms or 0.0)
        self.predicted_dwell_time_ms = None
        self.remaining_dwell_time_ms = None
        self.true_dwell_time_ms = None
        self.distance_to_uav_bs = None  # meters; device -> nearest UAV_BS (direct aerial link)

        # ---- RF / handover decision-quality context (filled by the
        # HandoverScheduler via core/simulator.py each round; see
        # mobility/handover_scheduler.py) ----
        self.oracle_bs = ""
        self.oracle_rssi_dbm = ""
        self.rssi_regret_db = ""
        self.snr_db = ""
        self.sinr_db = ""
        self.noise_dbm = ""
        self.interference_dbm = ""
        self.handover_executed = 0
        self.handover_failure_flag = 0
        self.handover_interruption_ms = 0.0
        self.handover_trigger_reason = ""
        self.pingpong_flag = 0
        self.bs_residence_time_ms = ""
        self.coverage_outage_flag = 0
        self.coverage_outage_duration_ms = ""
        self.prev_attached_bs = ""
        self.handover_prev_rssi_dbm = ""
        self.handover_target_bs = ""
        self.handover_target_rssi_dbm = ""
        self.handover_target_rssi_gain_db = ""
        self.handover_best_available_bs = ""
        self.handover_best_available_rssi_dbm = ""
        self.handover_q_margin = ""
        self.handover_chosen_cpu_util = ""
        self.handover_chosen_mem_util = ""
        self.handover_oracle_cpu_util = ""
        self.handover_oracle_mem_util = ""

    # ------------------------------------------------------------------
    # Convenience setters (non-breaking)
    # ------------------------------------------------------------------

    def set_round(self, round_index: Optional[int]):
        """Attach/override round index (used by logs/plots)."""
        self.round = int(round_index) if round_index is not None else None

    def set_mobility(self,
                     x: Optional[float] = None,
                     y: Optional[float] = None,
                     distance_to_bs: Optional[float] = None,
                     signal_dbm: Optional[float] = None,
                     handover_ms: Optional[float] = None,
                     speed_m_s: Optional[float] = None,
                     predicted_dwell_time_ms: Optional[float] = None,
                     remaining_dwell_time_ms: Optional[float] = None,
                     handover_risk: Optional[float] = None):
        """
        Attach/override mobility fields. Any None is ignored to avoid clobbering
        existing values if only one field is updated.
        """
        if x is not None: self.pos_x = float(x)
        if y is not None: self.pos_y = float(y)
        if distance_to_bs is not None: self.distance_to_bs = float(distance_to_bs)
        if signal_dbm is not None: self.signal_dbm = float(signal_dbm)
        if handover_ms is not None: self.handover_delay_ms = float(handover_ms)
        if speed_m_s is not None: self.movement_speed_m_s = float(speed_m_s)
        if predicted_dwell_time_ms is not None: self.predicted_dwell_time_ms = float(predicted_dwell_time_ms)
        if remaining_dwell_time_ms is not None: self.remaining_dwell_time_ms = float(remaining_dwell_time_ms)
        if handover_risk is not None: self.handover_risk = float(handover_risk)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def set_server(self, server_name: str):
        """
        Mark this task as scheduled on a server.
        NOTE: We keep legacy `server_assigned` untouched on purpose to preserve
        is_assigned() behavior for older code paths that depend on it.
        """
        self.assigned_server = server_name
        self.status = "scheduled"

    def is_assigned(self) -> bool:
        """
        Returns True if legacy field `server_assigned` is set.
        NOTE: Preserved behavior; do not change to check `assigned_server`.
        """
        return self.server_assigned is not None

    def complete(self, start_time, end_time):
        """
        Mark the task as completed and record execution window.
        """
        self.execution_start_time = start_time
        self.execution_end_time = end_time
        self.status = "completed"

    def fail(self):
        """
        Mark the task as failed.
        """
        self.status = "failed"

    def execution_delay(self):
        """
        Duration between start and end times (0 if not both are set).
        """
        if self.execution_start_time is not None and self.execution_end_time is not None:
            return self.execution_end_time - self.execution_start_time
        return 0

    # ------------------------------------------------------------------
    # Transmission / logging
    # ------------------------------------------------------------------

    def to_dict(self):
        """
        Export task fields for logging/reporting.
        Mobility fields are included if present; blanks are fine when mobility is off.
        """
        return {
            "id": self.id,
            "round": self.round,
            "cpu_demand": self.cpu_demand,
            "storage_demand": self.storage_demand,
            "memory_demand": self.memory_demand,  # included for completeness
            "bandwidth": self.bandwidth,
            "compute_work_cycles": self.compute_work_cycles,
            "compute_rate_demand_cycles_per_s": self.compute_rate_demand_cycles_per_s,
            "memory_working_set_bytes": self.memory_working_set_bytes,
            "storage_footprint_bytes": self.storage_footprint_bytes,
            "input_payload_bytes": self.input_payload_bytes,
            "resource_conversion_profile_id": self.resource_conversion_profile_id,
            "latency": self.latency,
            "priority": self.priority,
            "device_id": self.device_id,
            "device_type": self.device_type,
            "service_class": self.service_class,
            "mobility_profile": self.mobility_profile,
            "local_compute_capable": self.local_compute_capable,
            "local_cpu_capacity": self.local_cpu_capacity,
            "local_memory_capacity": self.local_memory_capacity,
            "local_compute_rate_cycles_per_s": self.local_compute_rate_cycles_per_s,
            "local_memory_capacity_bytes": self.local_memory_capacity_bytes,
            "local_power_w": self.local_power_w,
            "battery_capacity_j": self.battery_capacity_j,
            "battery_remaining_j": self.battery_remaining_j,
            "movement_speed_m_s": self.movement_speed_m_s,
            "handover_sensitivity": self.handover_sensitivity,
            "dwell_weight": self.dwell_weight,
            "deadline_ms": self.deadline_ms,
            "sla_profile_id": self.sla_profile_id,
            "sla_generation": self.sla_generation,
            "radio_latency_target_ms": self.radio_latency_target_ms,
            "application_deadline_ms": self.application_deadline_ms,
            "sla_maturity": self.sla_maturity,
            "sla_source_id": self.sla_source_id,
            "application_type": self.application_type,
            "data_size_kb": self.data_size_kb,
            "server": self.assigned_server,
            "status": self.status,
            "start_time": self.execution_start_time,
            "end_time": self.execution_end_time,
            "delay": self.delay,
            "cost": self.cost,
            "communication_cost": self.communication_cost,
            "processing_cost": self.processing_cost,
            "operation_energy_cost": self.operation_energy_cost,
            "energy": self.energy,
            "device_tx_energy_j": self.device_tx_energy_j,
            "device_rx_energy_j": self.device_rx_energy_j,
            "device_local_compute_energy_j": self.device_local_compute_energy_j,
            "device_wait_energy_j": self.device_wait_energy_j,
            "device_propulsion_energy_j": self.device_propulsion_energy_j,
            "device_paid_energy_j": self.device_paid_energy_j,
            "infra_compute_energy_j": self.infra_compute_energy_j,
            "backhaul_energy_j": self.backhaul_energy_j,
            "total_energy_j": self.total_energy_j,
            "measurement_basis": self.measurement_basis,
            "network_status": self.network_status,
            "network_failure_reason": self.network_failure_reason,
            "payload_bytes": self.payload_bytes,
            "on_wire_bytes": self.on_wire_bytes,
            "packet_count": self.packet_count,
            "packet_hop_count": self.packet_hop_count,
            "pre_access_wait_ms": self.pre_access_wait_ms,
            "network_queue_delay_ms": self.network_queue_delay_ms,
            "network_transmission_delay_ms": self.network_transmission_delay_ms,
            "network_propagation_delay_ms": self.network_propagation_delay_ms,
            "network_retry_backoff_ms": self.network_retry_backoff_ms,
            "network_outage_observation_ms": self.network_outage_observation_ms,
            "network_terminal_wait_ms": self.network_terminal_wait_ms,
            "network_delay_ms": self.network_delay_ms,
            "processing_delay_ms": self.processing_delay_ms,
            "completion_delay_ms": self.completion_delay_ms,
            "legacy_objective_delay_ms": self.legacy_objective_delay_ms,
            "deadline_missed": self.deadline_missed,
            "cost_accounting_perspective": self.cost_accounting_perspective,
            "cost_complete": self.cost_complete,
            "cost_incomplete_reason": self.cost_incomplete_reason,
            # Mobility extras
            "pos_x": self.pos_x,
            "pos_y": self.pos_y,
            "distance_to_bs": self.distance_to_bs,
            "signal_dbm": self.signal_dbm,
            "handover_ms": self.handover_delay_ms,
            "predicted_dwell_time_ms": self.predicted_dwell_time_ms,
            "remaining_dwell_time_ms": self.remaining_dwell_time_ms,
            "handover_risk": self.handover_risk,
        }

    def __str__(self):
        """
        Human-readable summary (short ID).
        """
        short_id = self.id[:6]
        return (
            f"Task({short_id}) | CPU: {self.cpu_demand} | Storage: {self.storage_demand} | "
            f"Memory: {self.memory_demand} | Data: {self.data_size_kb}KB | "
            f"Priority: {self.priority} | Status: {self.status}"
        )
