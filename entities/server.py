"""Stateful compute server with physical resource capacities and allocations.

Author: Dr. Afzal Badshah
Department of Software Engineering, University of Sargodha
Website: https://afzalbadshah.com/index.php/afzal-badshah/
LinkedIn: https://www.linkedin.com/in/afzal-badshah-phd-305b03164/
Discussion Forum: resp.afzalbadshah.com

Notes:
- Legacy capacity fields remain available for saved policies and callers.

How to extend: add resource dimensions to ``entities.resources`` first, then
update admission, release, and utilization together in this class.
"""

import math
from config import WORKLOAD, SERVER_CONFIG, MOBILITY_CONFIG
from mobility.random_waypoint import RandomWaypoint
from config import CITY_AREA
from config.units import LEGACY_V1_RESOURCE_CONVERSION
from entities.resources import (
    ComputeCapacity,
    ComputeWork,
    ResourceAllocation,
    bounded_utilization_ratio,
    offered_load_ratio,
)


class Server:
    """
    Represents a computing server (Edge, Regional, or Cloud) in the simulation.
    Each server has:
    - Name and index
    - Resource capacities (CPU, memory, storage, bandwidth)
    - Network cost and latency
    - Physical position (x, y) and optional mobility
    """

    def __init__(self, name, index, cpu_capacity, storage_capacity,
                 memory_capacity, bandwidth, latency,
                 tier=None, resource_capacity=None):
        # Basic details
        self.name = name
        self.index = index
        self.cpu_capacity = cpu_capacity
        self.storage_capacity = storage_capacity
        self.memory_capacity = memory_capacity
        self.bandwidth = bandwidth
        self.latency = latency
        self.resource_capacity = (
            LEGACY_V1_RESOURCE_CONVERSION.capacity_from_legacy(
                cpu_capacity=cpu_capacity,
                memory_capacity=memory_capacity,
                storage_capacity=storage_capacity,
            )
            if resource_capacity is None
            else resource_capacity
        )
        if not isinstance(self.resource_capacity, ComputeCapacity):
            raise ValueError("resource_capacity must be a ComputeCapacity")
        self.compute_rate_cycles_per_s = (
            self.resource_capacity.compute_rate_cycles_per_s
        )
        self.memory_capacity_bytes = self.resource_capacity.memory_capacity_bytes
        self.storage_capacity_bytes = self.resource_capacity.storage_capacity_bytes
        self.resource_conversion_profile_id = (
            LEGACY_V1_RESOURCE_CONVERSION.profile_id
        )

        # Explicit tier (preferred) -- falls back to name-splitting for
        # backward compatibility. Multi-token tiers like "UAV_BS" break a
        # naive name.split('_')[0], so callers should pass tier= explicitly.
        self.tier = str(tier) if tier is not None else name.split('_')[0]

        # Position setup (from SERVER_CONFIG, if it defines static positions).
        # Tiers placed dynamically at runtime (MEC, UAV_BS -- see
        # core/simulator.py._initialize_servers()) don't have a "positions"
        # entry at all; default to the origin and let the caller set x,y
        # right after construction.
        positions = SERVER_CONFIG.get(self.tier, {}).get("positions")
        if positions and self.index < len(positions):
            self.x, self.y = positions[self.index]
        else:
            self.x, self.y = 0.0, 0.0
        self.location = self.x  # kept for backward compatibility

        # Mobility (optional; uniform "apply_to_servers" toggle for all tiers).
        # Mobile tiers like UAV_BS instead attach their own mobility model
        # explicitly in core/simulator.py and are exempt from this flag.
        if MOBILITY_CONFIG.get("enabled") and MOBILITY_CONFIG.get("apply_to_servers"):
            area = MOBILITY_CONFIG["area"]
            speed_range = MOBILITY_CONFIG["speed_range"]
            pause_time = MOBILITY_CONFIG["pause_time"]

            self.mobility = RandomWaypoint(area, speed_range, pause_time)

            # Start at configured static position
            try:
                self.mobility.pos = (self.x, self.y)
            except AttributeError:
                pass

            # Update to first simulated position
            self.x, self.y = self.mobility.next_position(0.0)
            self.location = self.x
        else:
            self.mobility = None
        self.force_mobile = False

        tier_spec = SERVER_CONFIG.get(self.tier, {}) or {}
        self.battery_backed = bool(tier_spec.get("battery_backed", False))
        self.battery_capacity_j = float(tier_spec.get("battery_capacity_j", 0.0) or 0.0)
        self.battery_remaining_j = float(tier_spec.get("battery_initial_j", self.battery_capacity_j) or 0.0)
        # Continuous flight/hover power draw (W), independent of workload --
        # only UAV_BS sets this > 0 (config/config.py SERVER_CONFIG); ground
        # tiers (MEC/Metro/Cloud) stay 0.0, i.e. no propulsion cost.
        self.hover_power_w = float(tier_spec.get("hover_power_w", 0.0) or 0.0)
        # Return-to-base recharge cycle (see mobility/manager.py) -- grounded
        # while recharging (no hover cost, not selectable), then reset to
        # full charge and back in rotation. Ground tiers never set this.
        self.recharging = False
        self.recharge_rounds_left = 0

        # Current available resources
        self.available_cpu = cpu_capacity
        self.available_storage = storage_capacity
        self.available_memory = memory_capacity
        self.available_compute_rate_cycles_per_s = self.compute_rate_cycles_per_s
        self.available_memory_bytes = self.memory_capacity_bytes
        self.available_storage_bytes = self.storage_capacity_bytes

        # Task tracking
        self.running_tasks = {}  # {task_id: (cpu, storage, memory, release_time)}
        self.running_resource_allocations = {}
        self.total_data_transferred_kb = 0.0
        self.total_data_transferred_bytes = 0

    # ------------------------------------------------------------------
    # Resource allocation methods
    # ------------------------------------------------------------------

    def can_allocate(self, cpu_demand, storage_demand, memory_demand=0):
        """
        Check a legacy request after explicit conversion to physical units.
        """
        work = LEGACY_V1_RESOURCE_CONVERSION.work_from_legacy(
            cpu_demand=cpu_demand,
            storage_demand=storage_demand,
            memory_demand=memory_demand,
        )
        return self.can_allocate_resources(work)

    def can_allocate_resources(self, work: ComputeWork) -> bool:
        """Return whether the task's physical reservation fits available capacity."""

        allocation = LEGACY_V1_RESOURCE_CONVERSION.allocation_for(work)
        return (
            self.available_compute_rate_cycles_per_s
            >= allocation.allocated_compute_rate_cycles_per_s
            and self.available_storage_bytes >= allocation.storage_allocated_bytes
            and self.available_memory_bytes >= allocation.memory_allocated_bytes
        )

    def _restore_available_resources_from_ledger(self):
        """Rebuild every available-resource value from active allocations.

        ``running_tasks`` and ``running_resource_allocations`` are the
        authoritative ledgers.  Recomputing from capacity minus their totals
        avoids the floating-point drift created by repeatedly subtracting and
        later adding the same fractional reservation.
        """

        legacy_cpu_used = sum(float(values[0]) for values in self.running_tasks.values())
        legacy_storage_used = sum(float(values[1]) for values in self.running_tasks.values())
        legacy_memory_used = sum(float(values[2]) for values in self.running_tasks.values())
        compute_used = sum(
            float(allocation.allocated_compute_rate_cycles_per_s)
            for allocation in self.running_resource_allocations.values()
        )
        storage_used = sum(
            int(allocation.storage_allocated_bytes)
            for allocation in self.running_resource_allocations.values()
        )
        memory_used = sum(
            int(allocation.memory_allocated_bytes)
            for allocation in self.running_resource_allocations.values()
        )

        self.available_cpu = min(
            float(self.cpu_capacity),
            max(0.0, float(self.cpu_capacity) - legacy_cpu_used),
        )
        self.available_storage = min(
            float(self.storage_capacity),
            max(0.0, float(self.storage_capacity) - legacy_storage_used),
        )
        self.available_memory = min(
            float(self.memory_capacity),
            max(0.0, float(self.memory_capacity) - legacy_memory_used),
        )
        self.available_compute_rate_cycles_per_s = min(
            float(self.compute_rate_cycles_per_s),
            max(0.0, float(self.compute_rate_cycles_per_s) - compute_used),
        )
        self.available_storage_bytes = min(
            int(self.storage_capacity_bytes),
            max(0, int(self.storage_capacity_bytes) - storage_used),
        )
        self.available_memory_bytes = min(
            int(self.memory_capacity_bytes),
            max(0, int(self.memory_capacity_bytes) - memory_used),
        )

    def allocate(
        self,
        task_id,
        cpu_demand,
        storage_demand,
        memory_demand,
        release_time,
        data_size_kb=None,
        resource_demand=None,
    ):
        """
        Allocate resources to a new task if possible.
        Returns True if allocation succeeds, otherwise False.
        """
        work = (
            LEGACY_V1_RESOURCE_CONVERSION.work_from_legacy(
                cpu_demand=cpu_demand,
                storage_demand=storage_demand,
                memory_demand=memory_demand,
            )
            if resource_demand is None
            else resource_demand
        )
        if not isinstance(work, ComputeWork):
            raise ValueError("resource_demand must be a ComputeWork")
        if task_id in self.running_tasks or task_id in self.running_resource_allocations:
            return False
        allocation = LEGACY_V1_RESOURCE_CONVERSION.allocation_for(work)
        if self.can_allocate_resources(work):
            # Track first, then derive availability once from the ledgers.
            self.running_tasks[task_id] = (cpu_demand, storage_demand, memory_demand, release_time)
            self.running_resource_allocations[task_id] = allocation
            self._restore_available_resources_from_ledger()

            # Update transferred data and cost
            # The old implementation charged the global fallback workload for
            # every task, even while per-round/per-class task sizes varied.
            # Callers can now provide the actual admitted task volume; omission
            # preserves the legacy API for external schedulers.
            actual_data_size_kb = (
                WORKLOAD["data_per_device_kb"]
                if data_size_kb is None
                else max(0.0, float(data_size_kb))
            )
            self.total_data_transferred_kb += actual_data_size_kb
            self.total_data_transferred_bytes += round(actual_data_size_kb * 1000.0)
            return True
        return False

    def release_completed_tasks(self, current_time):
        """
        Free resources from tasks whose release time has passed.
        """
        completed = [tid for tid, (_, _, _, rt) in self.running_tasks.items() if current_time >= rt]
        for tid in completed:
            self.running_tasks.pop(tid)
            self.running_resource_allocations.pop(tid)
        if completed:
            self._restore_available_resources_from_ledger()

    def battery_ratio(self):
        if not self.battery_backed or self.battery_capacity_j <= 0.0:
            return 1.0
        return max(0.0, min(1.0, self.battery_remaining_j / self.battery_capacity_j))

    def can_consume_battery(self, energy_j, min_ratio=0.0):
        if not self.battery_backed or self.battery_capacity_j <= 0.0:
            return True
        projected = self.battery_remaining_j - max(0.0, float(energy_j or 0.0))
        return projected >= self.battery_capacity_j * max(0.0, float(min_ratio or 0.0))

    def consume_battery(self, energy_j):
        if not self.battery_backed or self.battery_capacity_j <= 0.0:
            return self.battery_remaining_j
        self.battery_remaining_j = max(0.0, self.battery_remaining_j - max(0.0, float(energy_j or 0.0)))
        return self.battery_remaining_j

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def utilization(self):
        """
        Return compute, storage, and memory utilization percentages.
        """
        cpu_util = 100.0 * bounded_utilization_ratio(
            self.compute_rate_cycles_per_s - self.available_compute_rate_cycles_per_s,
            self.compute_rate_cycles_per_s,
        )
        storage_util = 100.0 * bounded_utilization_ratio(
            self.storage_capacity_bytes - self.available_storage_bytes,
            self.storage_capacity_bytes,
        )
        memory_util = 100.0 * bounded_utilization_ratio(
            self.memory_capacity_bytes - self.available_memory_bytes,
            self.memory_capacity_bytes,
        )
        return round(cpu_util, 2), round(storage_util, 2), round(memory_util, 2)
    def congestion(self):
        """
        Return bounded compute pressure for legacy state-vector compatibility.

        Network congestion belongs to the packet/link queues. This method no
        longer divides cumulative data by a rate without a time window.
        """
        cpu_util, storage_util, memory_util = self.utilization()
        return round(max(cpu_util, memory_util, storage_util), 2)

    def network_offered_load_ratio(
        self,
        *,
        bit_rate_bps,
        observation_window_s,
        offered_bytes=None,
    ):
        """Return windowed network offered load for compatibility diagnostics."""

        byte_count = (
            self.total_data_transferred_bytes
            if offered_bytes is None
            else int(offered_bytes)
        )
        return offered_load_ratio(
            byte_count * 8,
            bit_rate_bps,
            observation_window_s,
        )

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __str__(self):
        cpu_u, storage_u, mem_u = self.utilization()
        return (
            f"{self.name} Server | CPU: {cpu_u}% | Memory: {mem_u}% | Storage: {storage_u}% | "
            f"Tasks: {len(self.running_tasks)} | "
            f"Congestion: {self.congestion()}%"
        )


def place_servers_in_city(servers):
    m = [s for s in servers if s.name.startswith("Edge_") or s.name.startswith("PEaaS_")]
    u = [s for s in m if not hasattr(s, "x") or not hasattr(s, "y")]
    if not u:
        return

    xmin, xmax = CITY_AREA["xmin"], CITY_AREA["xmax"]
    ymin, ymax = CITY_AREA["ymin"], CITY_AREA["ymax"]

    n = len(u)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    xs = [xmin + (i + 0.5) * (xmax - xmin) / cols for i in range(cols)]
    ys = [ymin + (j + 0.5) * (ymax - ymin) / rows for j in range(rows)]

    k = 0
    for j in range(rows):
        for i in range(cols):
            if k >= n:
                break
            u[k].x, u[k].y = xs[i], ys[j]
            k += 1
