"""
RegionalMECSimPy — Educational Release

Author: Dr. Afzal Badshah
Department of Software Engineering, University of Sargodha
Website: https://afzalbadshah.com/index.php/afzal-badshah/
LinkedIn: https://www.linkedin.com/in/afzal-badshah-phd-305b03164/
Discussion Forum: resp.afzalbadshah.com

Notes:
- Functionality preserved; comments kept simple for student readability.
- Generates tasks for each simulation round based on WORKLOAD settings.

How to extend: change workload profiles in ``config/config.py``; keep all new
physical quantities explicit when constructing ``Task``.
"""

import random
from entities.task import Task
from config import WORKLOAD, SERVER_CONFIG, total_rounds, PRIORITY_WORKLOADS
from config.units import LEGACY_V1_RESOURCE_CONVERSION
from config.workload import get_sla, normalize_access_generation
try:
    from config import DEVICE_TYPE_PROFILES, PRIORITY_PROFILES
except Exception:
    DEVICE_TYPE_PROFILES = {}
    PRIORITY_PROFILES = {}


class WorkloadGenerator:
    """
    Produces synthetic workload (tasks) to simulate IoT devices generating data.

    Students:
    - To change how tasks are generated, modify generate_tasks().
    - You can change demand values (CPU, storage, etc.) to simulate different
      application types or device capabilities.
    """

    def __init__(
        self,
        workload=None,
        priority_workloads=None,
        access_generation="5G",
    ):
        # Load workload parameters from the dict handed down by the Simulator,
        # falling back to the active config.WORKLOAD profile.
        workload = workload if workload is not None else WORKLOAD
        self.start_devices = workload["start_devices"]
        self.max_devices = workload["max_devices"]
        self.increment = workload["increment"]
        # Authoritative round count for THIS run (same total_rounds() call
        # core/simulator.py uses to size its round loop) -- the device ramp
        # is spread across this whole span rather than a fixed per-round
        # `increment`, so a long run (e.g. rounds=1000) with a small
        # max_devices/increment ratio (e.g. 300/1) doesn't hit max_devices
        # at round 300 and then sit flat, with no further crowd growth, for
        # the remaining 700 rounds.
        self.total_rounds = total_rounds(workload)
        self.data_per_device_kb = workload["data_per_device_kb"]
        # Data-per-device ramp: defaults keep it flat at data_per_device_kb
        # for any workload dict that predates these keys.
        self.start_data_per_device_kb = float(
            workload.get("start_data_per_device_kb", self.data_per_device_kb)
        )
        self.max_data_per_device_kb = float(
            workload.get("max_data_per_device_kb", self.data_per_device_kb)
        )
        self.data_increment_kb = float(workload.get("data_increment_kb", 0) or 0)
        self.cpu_per_kb = float(workload.get("cpu_per_kb", 80.0) or 80.0)
        self.memory_per_kb = float(workload.get("memory_per_kb", 0.2) or 0.2)

        # Priority-specific workload profiles (URLLC, mMTC, eMBB)
        # Allows different data sizes per priority class for realistic modeling
        self.priority_workloads = priority_workloads if priority_workloads is not None else PRIORITY_WORKLOADS
        self.access_generation = normalize_access_generation(access_generation)

    def set_access_generation(self, access_generation):
        """Change the SLA generation before task creation."""
        self.access_generation = normalize_access_generation(access_generation)

    def _devices_for_round(self, round_no):
        """
        Device count for this round, ramped across the run's ENTIRE
        configured length (self.total_rounds) instead of a fixed per-round
        `increment`. Linearly interpolates from start_devices (round 1) to
        max_devices (the final round), so the crowd keeps growing for the
        whole run instead of reaching max_devices early and then plateauing
        with no further growth for the rest of the rounds.
        """
        if self.total_rounds <= 1 or self.max_devices <= self.start_devices:
            return self.max_devices
        fraction = (round_no - 1) / float(self.total_rounds - 1)
        fraction = max(0.0, min(1.0, fraction))
        devices = self.start_devices + fraction * (self.max_devices - self.start_devices)
        return int(round(devices))

    def _choose_device_profile(self):
        profiles = DEVICE_TYPE_PROFILES or {"generic": {"weight": 1.0}}
        names = list(profiles.keys())
        weights = [float(profiles[name].get("weight", 1.0) or 0.0) for name in names]
        if sum(weights) <= 0.0:
            weights = [1.0] * len(names)
        name = random.choices(names, weights=weights, k=1)[0]
        return name, profiles.get(name, {})

    @staticmethod
    def _priority_for_service_class(profile):
        service_class = str(profile.get("service_class", "") or "").lower()
        for priority, priority_profile in (PRIORITY_PROFILES or {}).items():
            if str(priority_profile.get("name", "") or "").lower() == service_class:
                return int(priority)
        return 2

    def _deadline_for(self, priority, profile):
        service_class = profile.get("service_class") or (
            PRIORITY_PROFILES.get(int(priority), {}) or {}
        ).get("name")
        return float(
            get_sla(service_class, self.access_generation)[
                "application_deadline_ms"
            ]
        )

    def _data_per_device_kb_for_priority(self, priority, round_no):
        """
        Get current data size for a specific priority class and round.
        Uses priority-specific workload profiles (URLLC, mMTC, eMBB) if available,
        otherwise falls back to default workload.
        """
        priority = int(priority)
        if priority not in self.priority_workloads:
            # Fallback to default if priority not configured
            return self.data_per_device_kb

        priority_cfg = self.priority_workloads.get(priority, {})
        start = float(priority_cfg.get("start_data_per_device_kb", priority_cfg.get("data_per_device_kb", 1.0)))
        max_kb = float(priority_cfg.get("max_data_per_device_kb", priority_cfg.get("data_per_device_kb", 400.0)))
        increment = float(priority_cfg.get("data_increment_kb", 0.0) or 0.0)

        # Ramp data size from start to max across all rounds
        current = start + ((round_no - 1) * increment)
        return min(current, max_kb)

    def _max_data_kb_for_priority(self, priority):
        """Configured per-priority data-size ceiling (KB), used to cap the task
        AFTER the device data_multiplier so a large multiplier cannot push a task
        past its class cap."""
        priority = int(priority)
        if priority not in self.priority_workloads:
            return float(self.max_data_per_device_kb)
        cfg = self.priority_workloads.get(priority, {})
        return float(cfg.get("max_data_per_device_kb", cfg.get("data_per_device_kb", self.max_data_per_device_kb)))

    def _task_from_profile(self, device_id, device_type, profile, round_no=1):
        priority = self._priority_for_service_class(profile)
        service_class = profile.get("service_class") or (
            PRIORITY_PROFILES.get(int(priority), {}) or {}
        ).get("name", "")
        sla = get_sla(service_class, self.access_generation)
        workload = profile.get("workload", {}) or {}
        local = profile.get("local_compute", {}) or {}
        onboard = SERVER_CONFIG.get("Onboard", {}) or {}
        capable = bool(profile.get("local_compute_capable", local.get("capable", False)))
        battery = profile.get("battery", {}) or {}
        radio = profile.get("radio_energy", {}) or {}
        mobility = profile.get("mobility", {}) or {}
        data_multiplier = float(profile.get("data_multiplier", workload.get("data_multiplier", 1.0)) or 1.0)
        cpu_multiplier = float(profile.get("cpu_multiplier", workload.get("cpu_multiplier", 1.0)) or 1.0)
        memory_per_kb = float(workload.get("memory_per_kb", profile.get("memory_per_kb", self.memory_per_kb)) or self.memory_per_kb)

        # Use priority-specific data size for this round. Enforce the per-priority
        # cap as a TRUE ceiling by applying it AFTER the device data_multiplier --
        # previously the cap was applied before the multiplier, so a large
        # multiplier let URLLC exceed its 80 KB cap (up to 160) and eMBB exceed
        # 400 KB (up to 720).
        priority_cfg = self.priority_workloads.get(int(priority), {}) or {}
        priority_data_kb = self._data_per_device_kb_for_priority(priority, round_no)
        max_data_kb = self._max_data_kb_for_priority(priority)
        effective_data_multiplier = (
            data_multiplier
            if bool(priority_cfg.get("apply_device_data_multiplier", True))
            else 1.0
        )
        data_size_kb = min(
            priority_data_kb * effective_data_multiplier,
            max_data_kb,
        )
        if data_size_kb <= 0.0:
            raise ValueError("generated task data size must be greater than zero")

        # A class may declare physical work directly.  This is used by the one
        # canonical mMTC task so device multipliers cannot collapse its intended
        # 40-million-cycle workload back to the one-unit legacy floor.
        configured_cycles = priority_cfg.get("compute_work_cycles")
        if configured_cycles is None:
            cpu_demand = max(1.0, data_size_kb * self.cpu_per_kb * cpu_multiplier)
            compute_work_cycles = (
                cpu_demand
                * LEGACY_V1_RESOURCE_CONVERSION.compute_work_cycles_per_legacy_unit
            )
        else:
            compute_work_cycles = float(configured_cycles)
            if compute_work_cycles <= 0.0:
                raise ValueError("compute_work_cycles must be greater than zero")
            cpu_demand = (
                compute_work_cycles
                / LEGACY_V1_RESOURCE_CONVERSION.compute_work_cycles_per_legacy_unit
            )

        storage_demand = data_size_kb
        configured_memory_bytes = priority_cfg.get("memory_working_set_bytes")
        if configured_memory_bytes is None:
            memory_demand = max(1.0, data_size_kb * max(0.05, memory_per_kb))
            memory_working_set_bytes = round(
                memory_demand
                * LEGACY_V1_RESOURCE_CONVERSION.memory_bytes_per_legacy_unit
            )
        else:
            memory_working_set_bytes = int(configured_memory_bytes)
            if memory_working_set_bytes <= 0:
                raise ValueError("memory_working_set_bytes must be greater than zero")
            memory_demand = (
                memory_working_set_bytes
                / LEGACY_V1_RESOURCE_CONVERSION.memory_bytes_per_legacy_unit
            )
        input_payload_bytes = round(data_size_kb * 1000.0)
        task = Task(
            cpu_demand=cpu_demand,
            storage_demand=storage_demand,
            memory_demand=memory_demand,
            bandwidth=data_size_kb,
            latency=None,
            priority=priority,
            device_id=device_id,
            device_type=device_type,
            service_class=service_class,
            device_profile=profile,
            mobility_profile=profile.get("mobility_profile", mobility.get("profile", "unknown")),
            # On-device capacity/power is per-device-type (profile's own
            # local_compute.cpu/memory/power_w, see devices_config.py),
            # falling back to the universal SERVER_CONFIG["Onboard"] default
            # only if a capable profile omits its own spec. Devices WITHOUT
            # onboard computing (capable=False, e.g. MCU-class sensors)
            # carry zero local capacity so no local space can exist for them
            # anywhere downstream.
            local_compute_capable=capable,
            local_cpu_capacity=float(local.get("cpu", onboard.get("cpu", 0.0)) or 0.0) if capable else 0.0,
            local_memory_capacity=float(local.get("memory", onboard.get("memory", 0.0)) or 0.0) if capable else 0.0,
            local_power_w=float(local.get("power_w", onboard.get("power_w", 0.0)) or 0.0) if capable else 0.0,
            battery_capacity_j=float(battery.get("capacity_j", 0.0) or 0.0),
            battery_remaining_j=float(battery.get("initial_j", battery.get("capacity_j", 0.0)) or 0.0),
            tx_power_w=float(radio.get("tx_power_w", 0.0) or 0.0),
            rx_power_w=float(radio.get("rx_power_w", 0.0) or 0.0),
            idle_power_w=float(radio.get("idle_power_w", 0.0) or 0.0),
            propulsion_power_w=float(profile.get("propulsion_power_w", 0.0) or 0.0),
            movement_speed_m_s=0.0,
            handover_sensitivity=float(mobility.get("handover_sensitivity", 0.0) or 0.0),
            dwell_weight=float(mobility.get("dwell_weight", 1.0) or 1.0),
            deadline_ms=sla["application_deadline_ms"],
            sla_profile_id=sla["profile_id"],
            sla_generation=self.access_generation,
            radio_latency_target_ms=sla["radio_latency_target_ms"],
            application_deadline_ms=sla["application_deadline_ms"],
            sla_maturity=sla["maturity"],
            sla_source_id=sla["source_id"],
            application_type=profile.get("application_type", device_type),
            compute_work_cycles=compute_work_cycles,
            memory_working_set_bytes=memory_working_set_bytes,
            # Generated task storage is its application payload. The legacy
            # numeric storage_demand remains for saved-policy compatibility.
            storage_footprint_bytes=input_payload_bytes,
            input_payload_bytes=input_payload_bytes,
        )
        task.data_size_kb = data_size_kb
        return task

    def generate_tasks(self, round_no, entities=None):
        """
        Generate tasks for the current round.

        Device count ramps from start_devices to max_devices across the
        run's ENTIRE configured length (see _devices_for_round()) -- not a
        fixed 'increment' per round, which would reach max_devices early on
        a long run and then plateau with no further growth.
        Per-device data size grows by 'data_increment_kb' each round until
        max_data_per_device_kb, so CPU/memory/bandwidth demand (all derived
        from data_size_kb in _task_from_profile) ramps up alongside the
        device count.
        Each task gets identical CPU, storage, and memory demands
        (students can randomize or differentiate if needed).
        """
        tasks = []

        # Determine number of devices for this round -- ramped across the
        # run's full length (see _devices_for_round()), not a fixed
        # per-round `increment`.
        current_devices = self._devices_for_round(round_no)

        # Determine per-device data size for this round
        current_data_kb = self.start_data_per_device_kb + ((round_no - 1) * self.data_increment_kb)
        if current_data_kb > self.max_data_per_device_kb:
            current_data_kb = self.max_data_per_device_kb
        self.data_per_device_kb = current_data_kb

        active_entities = list(entities or [])[:current_devices]
        for device_id in range(current_devices):
            if device_id < len(active_entities):
                ent = active_entities[device_id]
                device_type = getattr(ent, "device_type", "generic")
                profile = getattr(ent, "device_profile", {}) or DEVICE_TYPE_PROFILES.get(device_type, {})
                task = self._task_from_profile(device_id, device_type, profile, round_no)
                task.battery_remaining_j = float(getattr(ent, "battery_remaining_j", task.battery_remaining_j) or 0.0)
                task.movement_speed_m_s = float(getattr(ent, "movement_speed_m_s", task.movement_speed_m_s) or 0.0)
            else:
                device_type, profile = self._choose_device_profile()
                task = self._task_from_profile(device_id, device_type, profile, round_no)
            tasks.append(task)

        return tasks

    def reset(self):
        """
        Retained for compatibility (no internal state to reset here).
        """
        pass
