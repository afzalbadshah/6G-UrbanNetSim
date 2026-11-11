"""Build scheduler observations (the RL state) from OBSERVABLE quantities only.

This module deliberately contains NO physics formulas. Delay, energy and cost are
calculated once, in the matrix (``metrics/`` + ``network/`` via
``core/task_execution``), and the realized results are written back into the
decision context after the chosen action executes. The reward reads those
realized values. Here we only:

  * read observable signals off the task/server objects (utilization, free
    capacity, battery, distance, demand);
  * decide which actions are STRUCTURALLY possible (server exists, device has
    onboard compute, a declared route reaches the tier);
  * assemble the fixed-width feature vector for the policies.

How to extend: add an OBSERVABLE feature in one place and keep the fixed state
schema expected by saved policies. Never add a delay/energy/cost formula here --
put it in its metrics module and let the matrix compute it.
"""

import math

import numpy as np

import config as cfg
from scheduler.common.common_action import (
    ACTION_LABELS,
    ACTION_LOCAL,
)
from scheduler.common.common_servers import (
    allocatable_action_server,
    local_has_capacity,
    projected_utilization,
    server_utilization,
)


class CommonStateBuilder:
    def __init__(self, state_dim=None):
        self.state_dim = int(state_dim or getattr(cfg, "STATE_DIM", 32))
        self.action_dim = len(ACTION_LABELS)
        self.train_mode = False
        # {source_node_id: set(reachable destination_node_id)} from the deployed
        # topology's declared routes. When set, compute actions are gated to
        # tiers the packet engine can actually route to from the device's radio
        # attachment. None -> no route gating (legacy / no access layer).
        self.reachability = None

    def set_train_mode(self, is_training):
        self.train_mode = bool(is_training)

    def set_reachability(self, reachability):
        """Install the topology reachability map {source: set(destinations)}."""
        if reachability is None:
            self.reachability = None
            return
        self.reachability = {
            str(src): {str(d) for d in dests}
            for src, dests in dict(reachability).items()
        }

    # ---- small observable helpers (no physics) ----
    @staticmethod
    def _safe_norm(value, denom):
        norm = float(value or 0.0) / max(float(denom or 1.0), 1e-9)
        if norm < 0.0:
            return 0.0
        if norm > 1.0:
            return 1.0
        return norm

    @staticmethod
    def _battery_ratio(obj):
        capacity = float(getattr(obj, "battery_capacity_j", 0.0) or 0.0)
        if capacity <= 0.0:
            return 1.0
        remaining = float(getattr(obj, "battery_remaining_j", 0.0) or 0.0)
        return max(0.0, min(1.0, remaining / capacity))

    @classmethod
    def _server_battery_ratio(cls, server):
        return cls._battery_ratio(server) if server is not None else 1.0

    @staticmethod
    def _tier_capacity_cps(task, server, action):
        """Absolute compute capacity (cycles/s) of a tier -- read, not computed.

        LOCAL is the device's own (small) rate; a server tier is its configured
        compute rate. This is the number that makes a small device read as
        scarce for a large eMBB task even when idle.
        """
        if int(action) == ACTION_LOCAL:
            return float(getattr(task, "local_compute_rate_cycles_per_s", 0.0) or 0.0)
        if server is None:
            return 0.0
        return float(
            getattr(server, "compute_rate_cycles_per_s", 0.0)
            or getattr(server, "cpu_capacity", 0.0)
            or 0.0
        )

    @staticmethod
    def _task_demand_cps(task):
        return float(
            getattr(task, "compute_rate_demand_cycles_per_s", 0.0)
            or getattr(task, "compute_work_cycles", 0.0)
            or 0.0
        )

    def build(self, task, servers, mobility=None):
        priority = int(getattr(task, "priority", 2) or 2)
        deadline_ms = float(
            getattr(task, "application_deadline_ms", 0.0)
            or getattr(task, "deadline_ms", 0.0)
            or 300.0
        )
        attached_access_id = str(getattr(task, "wireless_ap", "") or "")
        route_required = self.reachability is not None
        reachable = (
            self.reachability.get(attached_access_id, set())
            if route_required and attached_access_id
            else set()
        )
        action_servers = {
            action: allocatable_action_server(
                action,
                servers,
                task,
                reachable_destinations=reachable,
                require_route=route_required,
            )
            for action in range(self.action_dim)
        }

        demand_cps = self._task_demand_cps(task)
        capacity_mask = []
        current_util = {}
        future_util = {}
        congestion = {}
        server_battery_ratio = {}
        tier_capacity = {}
        resource_fit = {}
        tier_enabled = {}

        for action in range(self.action_dim):
            server = action_servers.get(action)
            tier_enabled[action] = int(
                (
                    bool(getattr(cfg, "LOCAL_DEVICE_ENABLED", True))
                    and bool(getattr(task, "local_compute_capable", False))
                )
                if action == ACTION_LOCAL
                else server is not None
            )
            server_battery_ratio[action] = self._server_battery_ratio(server)
            congestion[action] = float(server_utilization(server).get("congestion", 0.0))

            if action == ACTION_LOCAL:
                feasible = (
                    bool(getattr(cfg, "LOCAL_DEVICE_ENABLED", True))
                    and local_has_capacity(task)
                )
                rate = self._tier_capacity_cps(task, server, action)
                current_util[action] = float(
                    getattr(task, "local_compute_utilization_percent", 0.0) or 0.0
                )
                future_util[action] = (
                    current_util[action] + (100.0 * demand_cps / rate)
                    if rate > 0.0
                    else 100.0
                )
            else:
                feasible = server is not None
                utilization = server_utilization(server)
                current_util[action] = max(
                    utilization["cpu"], utilization["memory"], utilization["storage"]
                )
                future_util[action] = projected_utilization(server, task)["peak"] if server is not None else 100.0

            # Headroom view: how much of THIS tier's capacity is free right now,
            # independent of this one task's own (usually tiny) demand -- a busy
            # small tier reads as scarce even for a task whose own admission
            # would trivially fit, and an idle datacenter reads as abundant.
            # (The previous per-task-admission formula was ~always 1.0 at this
            # workload scale, since any single task's demand is small relative
            # to any tier's capacity -- it measured "can I be admitted", not
            # "how much headroom exists", so it carried no gradient.)
            cap = self._tier_capacity_cps(task, server, action)
            tier_capacity[action] = cap
            resource_fit[action] = min(1.0, max(0.0, 1.0 - float(current_util[action]) / 100.0))
            capacity_mask.append(int(feasible))

        # The learning mask is exactly the physical mask. Zero remains zero:
        # no class ladder, utilization cut-off, or fabricated fallback action.
        learning_mask = [int(x) for x in capacity_mask]

        # ---- feature vector: 22 task/context features + per-action block ----
        max_data = float(getattr(cfg, "WORKLOAD", {}).get("data_per_device_kb", 10000.0) or 10000.0) * 2.5
        mobility_metrics = mobility.get_metrics() if mobility is not None and hasattr(mobility, "get_metrics") else {}
        device_battery_ratio = self._battery_ratio(task)
        low = float(getattr(cfg, "BATTERY_LOW_RATIO", 0.30) or 0.30)
        diag = max(1.0, math.hypot(cfg.CITY_AREA["xmax"], cfg.CITY_AREA["ymax"]))
        features = [
            self._safe_norm(getattr(task, "data_size_kb", 0.0), max_data),
            self._safe_norm(getattr(task, "cpu_demand", 0.0), max_data),
            self._safe_norm(getattr(task, "memory_demand", 0.0), max_data),
            self._safe_norm(getattr(task, "storage_demand", 0.0), max_data),
            self._safe_norm(priority, 3.0),
            self._safe_norm(deadline_ms, 1000.0),
            self._safe_norm(getattr(task, "distance_to_bs", 0.0) or 0.0, diag),
            self._safe_norm(abs(float(getattr(task, "signal_dbm", -80.0) or -80.0)), 120.0),
            1.0 if getattr(task, "local_compute_capable", False) else 0.0,
            self._safe_norm(getattr(task, "handover_delay_ms", 0.0), 500.0),
            self._safe_norm(getattr(task, "movement_speed_m_s", 0.0), 30.0),
            self._safe_norm(getattr(task, "predicted_dwell_time_ms", 0.0) or 0.0, 60_000.0),
            self._safe_norm(getattr(task, "handover_risk", 0.0), 1.0),
            device_battery_ratio,
            self._safe_norm(mobility_metrics.get("ho_count", 0), max(1, getattr(cfg, "WORKLOAD", {}).get("max_devices", 1))),
            self._safe_norm(mobility_metrics.get("ho_delay", 0.0), 1000.0),
            self._safe_norm(mobility_metrics.get("avg_pos", 0.0), diag),
            self._safe_norm(getattr(task, "pos_x", 0.0), max(1.0, cfg.CITY_AREA["xmax"])),
            1.0 if device_battery_ratio < low else 0.0,
            1.0 if priority == 1 else 0.0,
            1.0 if priority == 2 else 0.0,
            1.0 if priority == 3 else 0.0,
        ]
        # Capacity adequacy is relative to THIS task's demand, not to the
        # largest tier in the deployment.  Once both MEC and Cloud can fully
        # satisfy a task, neither receives an artificial score merely because
        # its configured datacenter is larger.  Current load/headroom remains
        # a separate observable through resource_fit and future_util.
        tier_capacity_ratio = {
            action: (
                0.0
                if capacity <= 0.0
                else 1.0
                if demand_cps <= 0.0
                else min(1.0, max(0.0, capacity / demand_cps))
            )
            for action, capacity in tier_capacity.items()
        }
        for action in range(self.action_dim):
            features.extend([
                int(capacity_mask[action]),                                     # structural feasible
                self._safe_norm(future_util[action], 100.0),                    # projected utilization
                self._safe_norm(congestion[action], 100.0),                     # congestion
                max(0.0, 1.0 - self._safe_norm(future_util[action], 100.0)),    # free headroom
                float(resource_fit[action]),                                    # free capacity vs demand
                float(tier_capacity_ratio[action]),                            # capacity / task demand
                self._safe_norm(server_battery_ratio[action], 1.0),            # server battery
                1.0 if server_battery_ratio[action] < low else 0.0,            # server battery low
                int(tier_enabled[action]),                                      # tier present
                1.0 if action == ACTION_LOCAL else 0.0,                         # is-local marker
            ])

        if len(features) < self.state_dim:
            features.extend([0.0] * (self.state_dim - len(features)))
        state = np.asarray(features[: self.state_dim], dtype=np.float32)

        context = {
            "action_servers": action_servers,
            "capacity_mask": capacity_mask,
            "physical_mask": list(capacity_mask),
            "learning_mask": list(learning_mask),
            "mask": list(learning_mask),
            "train_mode": bool(self.train_mode),
            "deadline_ms": deadline_ms,
            # Physics dicts start EMPTY: the matrix (core/task_execution) fills
            # them with realized delay/energy/cost for the chosen action, and the
            # reward reads those realized values. common_state computes none of it.
            "total_delay": {},
            "delay_components": {},
            "cost": {},
            "cost_breakdown": {},
            "energy": {},
            "energy_breakdown": {},
            # Observable per-tier signals (read, not computed):
            "current_util": current_util,
            "future_util": future_util,
            "resource_fit": resource_fit,
            "tier_capacity_cps": tier_capacity,
            "capacity_adequacy": tier_capacity_ratio,
            # Backward-compatible key; its meaning is now the documented
            # demand-relative adequacy above, not capacity relative to Cloud.
            "tier_capacity_ratio": tier_capacity_ratio,
            "server_battery_ratio": server_battery_ratio,
            "projected_server_battery_ratio": dict(server_battery_ratio),
            "device_battery_ratio": device_battery_ratio,
            "tier_enabled": tier_enabled,
            "attached_access_id": attached_access_id,
            "mobility": mobility_metrics,
        }
        return state, context
