"""
RegionalMECSimPy — Educational Release

Notes:
- Single-class manager that keeps metrics in a dict.
- Critical fixes:
  (1) write back ent.x/ent.y after mobility updates.
  (2) unpack (server, ho_ms) from pick_best_bs() and use ho_ms directly.
"""

import math
import csv
import os
from copy import deepcopy
from config import MOBILITY_CONFIG, UAV_BS_MIN_BATTERY_RATIO, UAV_BS_RECHARGE_ROUNDS
from metrics.energy import energy_from_power_duration_j
from mobility.handover_scheduler import HandoverScheduler, candidate_access_points


def _fallback_metric_defaults():
    return {
        "handover_count_step": 0,
        "handover_delay_step_ms": 0.0,
        "avg_pos_dispersion": 0.0,
        "total_handovers": 0,
        "handover_attempts": 0,
        "total_handover_delay_ms": 0.0,
        "rss_samples": [],
        "total_outage_time_ms": 0.0,
        "dropped_tasks": 0,
        "total_tasks": 0,
        "throughputs": [],
    }


class MobilityManager:
    def __init__(self, servers, devices=None, deployment_profile_id=None):
        self.servers = servers
        self.entities = devices if devices is not None else []
        self.time_step_ms = MOBILITY_CONFIG.get("time_step_ms", 100)
        self.out_dir = None
        self.step_index = 0
        self.mobility_log = []
        self.train_mode = False

        cfg_defaults = MOBILITY_CONFIG.get("metrics_defaults")
        self.metrics = deepcopy(cfg_defaults) if isinstance(cfg_defaults, dict) else _fallback_metric_defaults()

        for ent in self.entities:
            if not hasattr(ent, "last_handover_ms"):
                ent.last_handover_ms = 0.0

        # RL-based handover scheduler: its only job is deciding/executing
        # which wireless point each device is attached to (see
        # mobility/handover_scheduler.py).  A deployed BS layer is preferred
        # over direct MEC radio attachment.
        self.handover_scheduler = HandoverScheduler(
            self.servers, deployment_profile_id=deployment_profile_id
        )

    # ---- helpers ----
    def _avg_position_dispersion(self, entities=None) -> float:
        active_entities = list(self.entities if entities is None else entities)
        if not active_entities:
            return 0.0
        xs, ys = [], []
        for ent in active_entities:
            # use ent.x/ent.y (kept in sync with ent.position)
            xs.append(float(getattr(ent, "x", 0.0)))
            ys.append(float(getattr(ent, "y", 0.0)))
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        return sum(math.hypot(x - cx, y - cy) for x, y in zip(xs, ys)) / len(xs)

    def _attachable_servers(self):
        # One canonical candidate inventory is shared with the handover agent;
        # device-specific ground-first eligibility is applied inside that agent
        # after it measures each candidate's received signal.
        return candidate_access_points(self.servers)


    # ---- main API ----
    def update_all(self, active_count=None):
        self.step_index += 1
        dt_ms = self.time_step_ms
        step_ho_count = 0
        step_ho_delay_ms = 0.0
        if active_count is None:
            active_entities = self.entities
        else:
            active_entities = self.entities[:max(0, min(int(active_count), len(self.entities)))]

        # Servers (usually static; UAV_BS is always mobile regardless of the
        # generic apply_to_servers toggle -- see entities/server.py/force_mobile).
        apply_to_all = MOBILITY_CONFIG.get("apply_to_servers", False)
        dt_s = dt_ms / 1000.0
        for srv in self.servers:
            # Continuous hover/propulsion energy -- independent of whether
            # the server processed any task this round. Only UAV_BS sets
            # hover_power_w > 0 (config.SERVER_CONFIG); ground tiers pay
            # nothing here.
            if getattr(srv, "hover_power_w", 0.0) > 0.0 and hasattr(srv, "consume_battery"):
                if getattr(srv, "recharging", False):
                    # Grounded at base: no hover cost, not selectable (mask
                    # stays closed since battery_remaining_j is untouched
                    # and still below the min-ratio floor). Once the
                    # swap/recharge cycle elapses, reset to full charge and
                    # rejoin rotation.
                    srv.recharge_rounds_left -= 1
                    if srv.recharge_rounds_left <= 0:
                        srv.battery_remaining_j = srv.battery_capacity_j
                        srv.recharging = False
                else:
                    srv.consume_battery(
                        energy_from_power_duration_j(power_w=srv.hover_power_w, duration_s=dt_s)
                    )
                    min_ratio = float(UAV_BS_MIN_BATTERY_RATIO or 0.25)
                    if srv.battery_capacity_j > 0.0 and (srv.battery_remaining_j / srv.battery_capacity_j) < min_ratio:
                        srv.recharging = True
                        srv.recharge_rounds_left = int(UAV_BS_RECHARGE_ROUNDS or 20)
            if getattr(srv, "recharging", False):
                continue  # grounded at base -- not patrolling this round
            if not (hasattr(srv, "mobility") and srv.mobility):
                continue
            if not (apply_to_all or getattr(srv, "force_mobile", False)):
                continue
            nx, ny = srv.mobility.next_position(dt_ms)
            srv.x, srv.y = float(nx), float(ny)
            srv.location = srv.x  # legacy 1D

        # Devices
        for ent in active_entities:
            # Continuous flight/propulsion energy -- independent of whether
            # this device transmits/computes this round. Only "uav"
            # mobility-kind profiles (the URLLC drone) set
            # propulsion_power_w > 0; everything else pays nothing here.
            if getattr(ent, "propulsion_power_w", 0.0) > 0.0 and getattr(ent, "battery_capacity_j", 0.0) > 0.0:
                ent.battery_remaining_j = max(
                    0.0,
                    ent.battery_remaining_j
                    - energy_from_power_duration_j(power_w=ent.propulsion_power_w, duration_s=dt_s),
                )
            if hasattr(ent, "mobility") and ent.mobility:
                nx, ny = ent.mobility.next_position(dt_ms)
                ent.position = [float(nx), float(ny)]
                ent.x, ent.y = float(nx), float(ny)
            else:
                ent.move(dt_ms)  # legacy 1D fallback updates ent.x

            # Best BS & handover accounting.
            # Always consult the RL handover scheduler so RF/decision-quality
            # signals (RSSI, SNR/SINR, oracle regret, ...) get refreshed for
            # every device every round -- including fixed (0 m/s) devices,
            # whose neighboring-cell load still changes round to round even
            # though their own RSSI doesn't. Quality gates inside the
            # scheduler already prevent a stationary device from being
            # spuriously handed over once it's attached to its best cell.
            attachable_list = self._attachable_servers()
            self.handover_scheduler.select_server(ent, self.servers, self)
            ho_ms = float(getattr(ent, "last_handover_ms", 0.0) or 0.0)
            if ho_ms > 0.0:
                step_ho_count += 1
                step_ho_delay_ms += ho_ms
            if self.train_mode:
                reward = self.handover_scheduler.calculate_reward(ent)
                self.handover_scheduler.update(reward, ent, round_id=self.step_index)

            # MEC/UAV_BS-only enforcement (in case anything upstream set
            # Metro/Cloud on attached_server) -- this must accept the same
            # tiers the handover scheduler itself may choose, or it would
            # silently revert every valid UAV_BS attachment back to MEC.
            attached = ent.attached_server
            attached_tier = str(getattr(attached, "tier", "") or "").upper()
            attached_name = str(getattr(attached, "name", "") or "")
            attached_ok = attached_tier in ("MEC", "UAV_BS") or attached_name.startswith("MEC_") or attached_name.startswith("UAV_BS_")
            if attached is None or not attached_ok:
                fallback_bs, _ = ent.pick_best_bs(attachable_list)
                if fallback_bs is not None:
                    ent.attached_server = fallback_bs


        # Per-step snapshots
        self.metrics["handover_count_step"] = int(step_ho_count)
        self.metrics["handover_delay_step_ms"] = float(step_ho_delay_ms)
        self.metrics["avg_pos_dispersion"] = float(self._avg_position_dispersion(active_entities))

        # Running totals
        self.metrics["total_handovers"] += int(step_ho_count)
        self.metrics["handover_attempts"] += len(active_entities)
        self.metrics["total_handover_delay_ms"] += float(step_ho_delay_ms)

        self.mobility_log.append({
            "round": self.step_index,
            "entities": len(active_entities),
            "handover_count": int(step_ho_count),
            "handover_delay_ms": float(step_ho_delay_ms),
            "avg_pos_dispersion": float(self.metrics.get("avg_pos_dispersion", 0.0)),
            "total_handovers": int(self.metrics.get("total_handovers", 0)),
            "handover_attempts": int(self.metrics.get("handover_attempts", 0)),
            "total_handover_delay_ms": float(self.metrics.get("total_handover_delay_ms", 0.0)),
        })

        return step_ho_count, step_ho_delay_ms


    def get_metrics(self):
        return {
            "avg_pos": round(self.metrics.get("avg_pos_dispersion", 0.0), 2),
            "ho_count": self.metrics.get("handover_count_step", 0),
            "ho_delay": self.metrics.get("handover_delay_step_ms", 0.0),
        }

    def set_train_mode(self, mode: bool):
        self.train_mode = bool(mode)
        if hasattr(self.handover_scheduler, "set_train_mode"):
            self.handover_scheduler.set_train_mode(self.train_mode)

    def save_logs(self, output_dir=None):
        out_dir = str(output_dir or self.out_dir or "output")
        os.makedirs(out_dir, exist_ok=True)
        if hasattr(self.handover_scheduler, "save_logs"):
            self.handover_scheduler.save_logs(output_dir=out_dir)
        path = os.path.join(out_dir, "mobility_results.csv")
        fields = [
            "round",
            "entities",
            "handover_count",
            "handover_delay_ms",
            "avg_pos_dispersion",
            "total_handovers",
            "handover_attempts",
            "total_handover_delay_ms",
        ]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.mobility_log)
