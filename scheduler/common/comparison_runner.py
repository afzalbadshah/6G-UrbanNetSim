"""Coordinate policy decisions, physical execution, learning, and reporting.

The runner selects an action, executes it through the injected runtime, and only
then records rewards and results from realized measurements.

How to extend: add policy algorithms through ``model_factory`` and keep physical
resource or network behavior in their domain modules.
"""

import csv
import math
import os
import tempfile
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace

from config.radio import get_link_profile
from config.servers import ELECTRICITY_PRICE_PER_JOULE
from config.reward import REWARD_SCHEMA_VERSION
from metrics.kpi import bounded_ratio, offered_load_ratio
from metrics.reward import RewardObservation, evaluate_reward
from scheduler.base_scheduler import BaseScheduler
from scheduler.common import common_config as default_config
from scheduler.common.common_action import ACTION_CLOUD, ACTION_LOCAL, ACTION_MEC, ACTION_MEC_COLLAB, ACTION_METRO, ACTION_UAV_BS, ACTION_SAT, action_to_label, get_action_dim, label_to_action
from scheduler.common.common_mask import CommonActionMask
from scheduler.common.common_state import CommonStateBuilder
from scheduler.common.common_transition import build_transition
from scheduler.common.common_servers import service_time_ms, server_utilization
from scheduler.common.model_factory import create_policy
from scheduler.common.policy_utils import set_global_seed
from scheduler.scheduler_helper.training_visualization_helper import TrainingVisualizationHelper
from core.reporter import TrainingReporter


class ComparisonRunner(BaseScheduler):
    """DriveNetSim-style multi-model scheduler adapted to SmartCityEdgeSim."""

    CSV_DELIMITER = ";"

    FIELDNAMES = [
        # A. Simulation meta / SmartCity aliases
        "round",
        "vehicle",
        "vehicle_id",
        "task_id",
        "device_id",
        "device_type",
        "service_class",
        "device_profile",
        "application_type",
        "mobility_profile",
        "movement_speed_m_s",
        "handover_sensitivity",
        "dwell_weight",
        "workload_mode",
        "workload_regime",
        "active_devices_this_round",
        "priority",
        "priority_class",
        "class_name",
        "deadline_ms",
        "sla_profile_id",
        "sla_generation",
        "radio_latency_target_ms",
        "application_deadline_ms",
        "sla_maturity",
        "sla_source_id",
        "deadline_delay_ms",
        "measurement_basis",
        "network_status",
        "network_failure_reason",
        "payload_bytes",
        "on_wire_bytes",
        "packet_count",
        "packet_hop_count",
        # Canonical Task-51 resource inputs (SI units) so processing time is
        # reproducible from the row: work in CPU cycles, allocated rate in
        # cycles/s, memory/storage working sets and input payload in bytes, plus
        # the versioned conversion-profile id and the local compute capacity.
        "compute_work_cycles",
        "allocated_compute_rate_cycles_per_s",
        "memory_working_set_bytes",
        "storage_footprint_bytes",
        "input_payload_bytes",
        "resource_conversion_profile_id",
        "local_compute_capacity_cycles_per_s",
        "data_size_kb",
        "total_workload_kb",
        # B. Connectivity and mobility
        "connected_bs",
        "connected_bs_idx",
        "chosen_bs_idx",
        "oracle_bs_idx",
        "wireless_chosen_bs",
        "processing_server",
        "processing_tier",
        "wireless_ap",
        "pos_x",
        "pos_y",
        "server_x",
        "server_y",
        "server_altitude_m",
        "vehicle_position_m",
        "distance_to_bs",
        "distance_to_bs_m",
        "signal_dbm",
        "RSSI_dBm",
        "chosen_rssi_dbm",
        "oracle_rssi_dbm",
        "rssi_regret_db",
        "handover_prev_rssi_dbm",
        "handover_target_bs",
        "handover_target_rssi_dbm",
        "handover_target_rssi_gain_db",
        "handover_best_available_bs",
        "handover_best_available_rssi_dbm",
        "SNR_dB",
        "SINR_dB",
        "Noise_dBm",
        "Interference_dBm",
        "vehicle_speed_mps",
        "speed_norm",
        "predicted_dwell_time_ms",
        "remaining_dwell_time_ms",
        "handover_flag",
        "handover_executed",
        "handover_delay_ms",
        "handover_interruption_ms",
        "handover_failure",
        "handover_success_flag",
        "handover_failure_flag",
        "handover_reason",
        "handover_trigger_reason",
        "pingpong_handover_flag",
        "bs_residence_time_ms",
        "handover_stability_time_ms",
        "coverage_outage_duration_ms",
        "coverage_outage_flag",
        "served_via_anchor",
        "completion_before_exit",
        "mobility_induced_violation",
        "mobility_handover_count",
        "mobility_handover_delay_ms",
        "mobility_avg_pos",
        # C. Scheduler / decision
        "selected_action",
        "selected_tier",
        "rl_action",
        "oracle_action",
        "oracle_tier",
        "oracle_reward",
        "match_oracle",
        "regret",
        "delay_regret_ms",
        "cost_regret",
        "energy_regret",
        "reward_regret",
        "allocated_tier",
        "execution_tier",
        "cost_tier",
        "offloading_tier",
        "allocated_server",
        "selected_server_id",
        "selected_server_type",
        "offload_route",
        "mask",
        "physical_mask",
        "learning_mask",
        "valid_count",
        "physical_valid_count",
        "learning_valid_count",
        "final_valid_count",
        "action_entropy",
        "action_final",
        "rl_tier_name",
        "rl_reward",
        "predicted_rl_reward",
        "raw_reward",
        "reward_deadline_penalty_norm",
        "reward_util_reward",
        "reward_overutilization_penalty",
        "reward_overutilization_weight",
        "reward_overutilization_deduction_points",
        "reward_schema_version",
        "no_service_available",
        "reward_invalid_action_penalty",
        "reward_deadline_weight",
        "reward_cost_weight",
        "reward_energy_weight",
        "reward_device_battery_weight",
        "reward_server_battery_weight",
        "reward_compute_weight",
        "reward_mobility_weight",
        "only_cloud_feasible",
        "collab_feasible",
        "chosen_layer",
        # D. Collaboration / overflow
        "overflow_used",
        "overflow_route",
        "overflow_fragment_count",
        "overflow_allocated_kb",
        "overflow_leftover_kb",
        "overflow_fully_allocated",
        "overflow_fragment_tiers",
        "overflow_fragment_kb",
        "overflow_fragment_cpu_util",
        "overflow_fragment_mem_util",
        "collab_checked",
        "collab_found",
        "collab_used",
        "collab_dst_id",
        "collab_t_budget",
        "collab_t_comm",
        "collab_t_exec",
        "collab_t_total",
        "collab_reason",
        "collab_reject_diff_bs",
        "collab_reject_no_bs",
        # E. Delay / QoS / queue
        "delay_ms",
        "objective_delay_ms",
        "legacy_objective_delay_ms",
        "legacy_complete_latency_ms",
        "trans_delay_ms",
        "tx_delay_wireless_ms",
        "tx_delay_wired_ms",
        "propagation_wireless_ms",
        "propagation_wired_ms",
        "propagation_total_ms",
        "prop_onboard_ms",
        "queue_length",
        "queue_length_total",
        "queue_length_urllc",
        "queue_length_mmtc",
        "queue_length_embb",
        "queued_workload_ms",
        "estimated_waiting_time_ms",
        "queue_delay_ms",
        "waiting_time_ms",
        "pre_access_wait_ms",
        "network_retry_backoff_ms",
        "network_retry_propagation_ms",
        "network_outage_observation_ms",
        "network_terminal_wait_ms",
        "network_delay_ms",
        "starvation_flag",
        "processing_delay_ms",
        "handover_delay_component_ms",
        "e2e_delay_ms",
        "e2e_delay_incl_processing_ms",
        "e2e_delay_incl_tx_ms",
        "e2e_delay_no_processing_ms",
        "predicted_completion_time_no_proc_ms",
        "latency_ms",
        "complete_latency_ms",
        "observed_complete_latency_ms",
        # F. Utilization / cost / energy / load
        "cpu_util_percent",
        "mem_util_percent",
        "sto_util_percent",
        "selected_tier_cpu_util_percent",
        "selected_tier_mem_util_percent",
        "selected_tier_congestion_percent",
        "selected_projected_utilization_percent",
        "chosen_cpu_util",
        "chosen_mem_util",
        "oracle_cpu_util",
        "oracle_mem_util",
        "Connected",
        "reward",
        "epsilon",
        "cost",
        "cost_accounting_perspective",
        "cost_complete",
        "cost_incomplete_reason",
        "tx_cost",
        "proc_cost",
        "processing_cost",
        "communication_cost",
        "cooling_cost",
        "operation_energy_cost",
        "propulsion_cost",
        "device_wait_cost",
        "operational_co2e_g",
        "energy",
        "tx_energy_proxy",
        "proc_energy_j",
        "total_energy_j",
        "vehicle_battery_energy_j",
        "device_tx_energy_j",
        "device_rx_energy_j",
        "device_local_compute_energy_j",
        "device_wait_energy_j",
        "device_propulsion_energy_j",
        "device_paid_energy_j",
        "infra_compute_energy_j",
        "backhaul_energy_j",
        "server_hover_energy_j",
        "device_battery_energy_j",
        "server_battery_energy_j",
        "communication_energy_j",
        "infrastructure_network_energy_j",
        "facility_overhead_energy_j",
        # IT-only compute energy (facility overhead excluded), so the disjoint
        # components sum to total exactly once. NOTE: infra_compute_energy_j is the
        # TOTAL (compute_it_energy_j + facility_overhead_energy_j), a legacy
        # aggregate alias — do not add it to the two disjoint parts.
        "compute_it_energy_j",
        "objective_energy_j",
        "battery_draw_j",
        "battery_capacity_j",
        "battery_remaining_j",
        "device_battery_before_j",
        "device_battery_after_j",
        "device_battery_ratio",
        "server_battery_before_j",
        "server_battery_after_j",
        "server_battery_ratio",
        "server_battery_projected_ratio",
        "local_cpu_capacity",
        "local_memory_capacity",
        "local_power_w",
        "local_compute_used",
        "handover_risk",
        "mobility_violation",
        "offered_load_percent",
        "network_congestion_percent",
        "task_arrival_rate",
        "arrival_rate_rps",
        "rho",
        "load_factor",
        "throughput_mbps",
        # G. Reward diagnostics / future utilization
        "future_util",
        "u_ve",
        "u_local",
        "u_mec",
        "u_collab",
        "u_met",
        "u_cloud",
        "u_uav",
        "reward_deadline_miss",
        "reward_mobility_violation",
        "reward_resource_overload",
        "reward_collab_needed",
        "reward_energy_norm",
        "reward_cost_norm",
        "reward_congestion_norm",
        "reward_throughput_norm",
        "reward_capacity_norm",
        "reward_compute_capacity_norm",
        "reward_shared_capacity_externality",
        "reward_shared_capacity_externality_weight",
        "current_util",
        "reward_cloud_overuse",
        "reward_deadline_success",
        "reward_device_energy_penalty",
        "reward_server_battery_penalty",
        "reward_device_battery_score",
        "reward_server_battery_score",
        "reward_cost_penalty",
        "reward_capacity_adequacy",
        "reward_cost_normalizer",
        "reward_device_energy_normalizer_j",
        "battery_aware_action_flag",
        "resource_balance_index",
        "utilization_stability_index",
        "decision_stability_index",
        "policy_consistency",
        "action_change_flag",
        "action_change_rate",
        "priority_fairness_index",
        "energy_efficiency",
        "cost_efficiency",
        "useful_bits_per_j",
        "unit_data_cost_usd_per_byte",
        "wire_efficiency",
        "resource_utilization_efficiency",
        "ve_future_cpu",
        "ve_future_mem",
        "local_future_cpu",
        "local_future_mem",
        "mec_future_cpu",
        "mec_future_mem",
        "collab_future_cpu",
        "collab_future_mem",
        "metro_future_cpu",
        "metro_future_mem",
        "cloud_future_cpu",
        "cloud_future_mem",
        "uav_future_cpu",
        "uav_future_mem",
        "ve_peak_future",
        "local_peak_future",
        "mec_peak_future",
        "collab_peak_future",
        "metro_peak_future",
        "cloud_peak_future",
        "uav_peak_future",
        # H. SmartCity-specific retained fields
        "local_compute_capable",
        "mec_available",
        "metro_available",
        "cloud_available",
        "uav_available",
        "bs_ok",
        "me_ok",
        "success",
        "dropped",
        "deadline_missed",
        "status",
        # Satellite-IoT scenario columns.
        "zone_id",
        "sat_available",
        "sat_elevation_deg",
        "sat_slant_range_m",
        "sat_access_node",
        "sat_selected",
        "reward_forced_single_action",
        "reward_deadline_penalty_exempt",
    ]

    HANDOVER_FIELDNAMES = [
        "round",
        "vehicle",
        "vehicle_id",
        "connected_bs_idx",
        "chosen_bs_idx",
        "oracle_bs_idx",
        "wireless_chosen_bs",
        "processing_server",
        "processing_tier",
        "RSSI_dBm",
        "chosen_rssi_dbm",
        "oracle_rssi_dbm",
        "rssi_regret_db",
        "handover_prev_rssi_dbm",
        "handover_target_bs",
        "handover_target_rssi_dbm",
        "handover_target_rssi_gain_db",
        "handover_best_available_bs",
        "handover_best_available_rssi_dbm",
        "distance_to_bs_m",
        "SNR_dB",
        "SINR_dB",
        "Noise_dBm",
        "Interference_dBm",
        "vehicle_speed_mps",
        "speed_norm",
        "vehicle_position_m",
        "predicted_dwell_time_ms",
        "remaining_dwell_time_ms",
        "handover_flag",
        "handover_executed",
        "handover_delay_ms",
        "handover_interruption_ms",
        "handover_failure",
        "handover_success_flag",
        "handover_failure_flag",
        "handover_reason",
        "handover_trigger_reason",
        "pingpong_handover_flag",
        "bs_residence_time_ms",
        "handover_stability_time_ms",
        "coverage_outage_duration_ms",
        "coverage_outage_flag",
        "served_via_anchor",
        "completion_before_exit",
        "mobility_induced_violation",
        "reward",
        "priority",
        "class_name",
        "deadline_ms",
        "deadline_delay_ms",
        "predicted_completion_time_no_proc_ms",
        "trans_delay_ms",
        "propagation_wireless_ms",
        "propagation_wired_ms",
        "propagation_total_ms",
        "e2e_delay_ms",
        "e2e_delay_incl_tx_ms",
        "e2e_delay_no_processing_ms",
        "offloading_tier",
        "selected_server_type",
    ]

    DECISION_QUALITY_FIELDNAMES = [
        "decision_idx",
        "time_ms",
        "vehicle_id",
        "chosen_bs",
        "oracle_bs",
        "chosen_rssi_dbm",
        "oracle_rssi_dbm",
        "regret_sig_norm",
        "speed_norm",
        "distance_m",
        "handover_flag",
        "pingpong_flag",
        "prev_bs",
        "target_bs",
        "target_rssi_dbm",
        "target_rssi_gain_db",
        "best_available_bs",
        "best_available_rssi_dbm",
        "chosen_cpu_util",
        "chosen_mem_util",
        "oracle_cpu_util",
        "oracle_mem_util",
        "load_regret",
        "q_margin",
    ]

    MOBILITY_FIELDNAMES = HANDOVER_FIELDNAMES + [
        "decision_idx",
        "handover_prev_bs",
        "handover_chosen_cpu_util",
        "handover_chosen_mem_util",
        "handover_oracle_cpu_util",
        "handover_oracle_mem_util",
        "load_regret",
        "handover_q_margin",
    ]

    def __init__(self, model_name, config=default_config, output_dir=None, seed=None):
        self.model_name = str(model_name or "random").strip().lower()
        self.config = config
        # Phase 4: honor an explicit per-run seed. Previously this always
        # re-seeded to config.RANDOM_SEED, silently overriding any seed the
        # caller had set for a multi-seed repetition. Passing seed=None keeps
        # the historical behavior (base seed 42 -> identical reproduction).
        self._seed = int(getattr(config, "RANDOM_SEED", 42) if seed is None else seed)
        set_global_seed(self._seed)
        self.state_dim = int(getattr(config, "STATE_DIM", 32))
        self.action_dim = int(getattr(config, "ACTION_DIM", get_action_dim()))
        self.state_builder = CommonStateBuilder(self.state_dim)
        self.mask_builder = CommonActionMask(self.action_dim)
        self.policy = create_policy(self.model_name, self.state_dim, self.action_dim, config)
        self.train_mode = False
        self.comparison_mode = True
        self.servers = None
        self.round_id = 0
        self.reward_log = []
        self.epsilon_log = []
        self.action_dist_log = []
        self.loss_log = []
        self.active_devices_this_round = ""
        # Rolling per-class/per-device windows for the CSV's "learning
        # stability" diagnostics (action_entropy, action_change_rate,
        # priority_fairness_index) -- computed causally at decision time
        # from realized choices, not from internal Q-values/probabilities,
        # so the same logic applies uniformly across DQN/PPO/A2C/Q-learning.
        self._recent_actions_by_class = {1: deque(maxlen=50), 2: deque(maxlen=50), 3: deque(maxlen=50)}
        self._recent_change_by_class = {1: deque(maxlen=50), 2: deque(maxlen=50), 3: deque(maxlen=50)}
        self._recent_success_by_class = {1: deque(maxlen=50), 2: deque(maxlen=50), 3: deque(maxlen=50)}
        self._last_action_by_device = {}
        self.output_root = Path(output_dir or "output")
        self.phase = "exploration"
        self.output_dir = self.output_root / self.model_name
        self.mobility_dir = self.output_dir
        self.offloading_filename = f"{self.model_name}_{getattr(config, 'OFFLOADING_FILE_SUFFIX', 'offloading_results.csv')}"
        self.write_legacy_offloading_copy = bool(getattr(config, "WRITE_LEGACY_OFFLOADING_COPY", False))
        self.write_legacy_mobility_detail_csvs = bool(getattr(config, "WRITE_LEGACY_MOBILITY_DETAIL_CSVS", False))
        self._csv_path = None
        self._csv_ready = False
        self._csv_fh = None
        self._csv_writer = None
        self._main_fh = None
        self._main_writer = None
        self._mobility_fh = None
        self._mobility_writer = None
        self._handover_fh = None
        self._handover_writer = None
        self._decision_quality_fh = None
        self._decision_quality_writer = None
        self._decision_idx = 0
        self._csv_rows_since_flush = 0
        self._csv_flush_interval = max(1, int(getattr(config, "SMARTCITY_CSV_FLUSH_INTERVAL", 1000) or 1000))
        self.visualizer = TrainingVisualizationHelper(self.model_name, output_root=self.output_dir)
        self.policy.set_visualizer(self.visualizer)
        self._diag_decisions = 0
        self._execution_runtime = None
        # Active onboard reservations keyed by originating device. The prior
        # rewrite read local utilization but never maintained it, which made
        # real Local -> MEC saturation spillover impossible.
        self._local_resource_allocations = {}

    def set_output_dirs(self, offloading_dir, mobility_dir=None):
        self._close_csv()
        self.output_dir = Path(offloading_dir)
        self.mobility_dir = Path(mobility_dir) if mobility_dir is not None else self.output_dir
        self.visualizer = TrainingVisualizationHelper(self.model_name, output_root=self.output_dir)
        self.policy.set_visualizer(self.visualizer)
        self._csv_ready = False

    def set_train_mode(self, is_training):
        self.train_mode = bool(is_training)
        self.phase = "exploration" if self.train_mode else "evaluation"
        if self.output_dir is None:
            self.output_dir = self.output_root / self.model_name
        self.state_builder.set_train_mode(self.train_mode)
        self.policy.set_train_mode(self.train_mode)
        self._csv_ready = False

    def _ensure_csv(self):
        if self._csv_ready:
            return

        primary_output_dir = Path(self.output_dir)
        primary_mobility_dir = Path(self.mobility_dir)
        fallback_root = Path(tempfile.gettempdir()) / "SmartCityEdgeSim" / "evaluation_fallback" / self.model_name
        attempts = [
            (primary_output_dir, primary_mobility_dir, False),
            (fallback_root / "offloading", fallback_root / "mobility", True),
        ]
        last_exc = None

        for output_dir, mobility_dir, is_fallback in attempts:
            try:
                self._close_csv()
                output_dir.mkdir(parents=True, exist_ok=True)
                mobility_dir.mkdir(parents=True, exist_ok=True)
                self._csv_path = output_dir / "results.csv"
                self._main_fh = open(self._csv_path, "w", newline="", encoding="utf-8")
                self._main_writer = csv.DictWriter(self._main_fh, fieldnames=self.FIELDNAMES, delimiter=self.CSV_DELIMITER)
                self._main_writer.writeheader()
                if self.write_legacy_offloading_copy:
                    self._csv_fh = open(output_dir / self.offloading_filename, "w", newline="", encoding="utf-8")
                    self._csv_writer = csv.DictWriter(self._csv_fh, fieldnames=self.FIELDNAMES, delimiter=self.CSV_DELIMITER)
                    self._csv_writer.writeheader()
                # RSSI/oracle-regret/ping-pong/handover decision-quality data is
                # mobility content, not offloading content -- goes under mobility_dir
                # (see core/simulator.py's offloading/<model> vs mobility/<model>
                # split) alongside the handover scheduler's own CSVs/plots.
                self._mobility_fh = open(mobility_dir / "results.csv", "w", newline="", encoding="utf-8")
                self._mobility_writer = csv.DictWriter(
                    self._mobility_fh,
                    fieldnames=self.MOBILITY_FIELDNAMES,
                    delimiter=self.CSV_DELIMITER,
                )
                self._mobility_writer.writeheader()
                if self.write_legacy_mobility_detail_csvs:
                    self._handover_fh = open(mobility_dir / "handover_results.csv", "w", newline="", encoding="utf-8")
                    self._handover_writer = csv.DictWriter(self._handover_fh, fieldnames=self.HANDOVER_FIELDNAMES, delimiter=self.CSV_DELIMITER)
                    self._handover_writer.writeheader()
                    self._decision_quality_fh = open(mobility_dir / "decision_quality.csv", "w", newline="", encoding="utf-8")
                    self._decision_quality_writer = csv.DictWriter(
                        self._decision_quality_fh,
                        fieldnames=self.DECISION_QUALITY_FIELDNAMES,
                        delimiter=self.CSV_DELIMITER,
                    )
                    self._decision_quality_writer.writeheader()
                self.output_dir = output_dir
                self.mobility_dir = mobility_dir
                self._csv_rows_since_flush = 0
                self._csv_ready = True
                if is_fallback:
                    print(
                        f"[OUT] {self.model_name} output fallback -> {output_dir} "
                        f"(configured output failed: {last_exc})",
                        flush=True,
                    )
                return
            except OSError as exc:
                last_exc = exc
                self._close_csv()
                continue

        raise last_exc if last_exc is not None else OSError("Unable to open scheduler CSV outputs")

    def _append_row(self, row):
        self._ensure_csv()
        full_row = {key: row.get(key, "") for key in self.FIELDNAMES}
        self._main_writer.writerow(full_row)
        if self._csv_writer is not None:
            self._csv_writer.writerow(full_row)
        self._mobility_writer.writerow(self._mobility_row(row))
        if self._handover_writer is not None:
            self._handover_writer.writerow({key: row.get(key, "") for key in self.HANDOVER_FIELDNAMES})
        if self._decision_quality_writer is not None:
            self._decision_quality_writer.writerow(self._decision_quality_row(row))
        self._decision_idx += 1
        self._csv_rows_since_flush += 1
        if self._csv_rows_since_flush >= self._csv_flush_interval:
            self._main_fh.flush()
            if self._csv_fh is not None:
                self._csv_fh.flush()
            self._mobility_fh.flush()
            if self._handover_fh is not None:
                self._handover_fh.flush()
            if self._decision_quality_fh is not None:
                self._decision_quality_fh.flush()
            self._csv_rows_since_flush = 0

    def _mobility_row(self, row):
        mobility_row = {key: row.get(key, "") for key in self.MOBILITY_FIELDNAMES}
        mobility_row.update({
            "decision_idx": self._decision_idx,
            "handover_prev_bs": row.get("handover_prev_bs", ""),
            "handover_chosen_cpu_util": row.get("handover_chosen_cpu_util", ""),
            "handover_chosen_mem_util": row.get("handover_chosen_mem_util", ""),
            "handover_oracle_cpu_util": row.get("handover_oracle_cpu_util", ""),
            "handover_oracle_mem_util": row.get("handover_oracle_mem_util", ""),
            "load_regret": row.get("regret", ""),
            "handover_q_margin": row.get("handover_q_margin", ""),
        })
        return mobility_row

    def _decision_quality_row(self, row):
        return {
            "decision_idx": self._decision_idx,
            "time_ms": row.get("round", ""),
            "vehicle_id": row.get("vehicle_id", row.get("vehicle", "")),
            "chosen_bs": row.get("connected_bs_idx", ""),
            "oracle_bs": row.get("oracle_bs_idx", ""),
            "chosen_rssi_dbm": row.get("chosen_rssi_dbm", ""),
            "oracle_rssi_dbm": row.get("oracle_rssi_dbm", ""),
            "regret_sig_norm": row.get("rssi_regret_db", ""),
            "speed_norm": row.get("speed_norm", ""),
            "distance_m": row.get("distance_to_bs_m", row.get("distance_to_bs", "")),
            "handover_flag": row.get("handover_flag", ""),
            "pingpong_flag": row.get("pingpong_handover_flag", ""),
            "prev_bs": row.get("handover_prev_bs", ""),
            "target_bs": row.get("handover_target_bs", ""),
            "target_rssi_dbm": row.get("handover_target_rssi_dbm", ""),
            "target_rssi_gain_db": row.get("handover_target_rssi_gain_db", ""),
            "best_available_bs": row.get("handover_best_available_bs", ""),
            "best_available_rssi_dbm": row.get("handover_best_available_rssi_dbm", ""),
            "chosen_cpu_util": row.get("handover_chosen_cpu_util", ""),
            "chosen_mem_util": row.get("handover_chosen_mem_util", ""),
            "oracle_cpu_util": row.get("handover_oracle_cpu_util", ""),
            "oracle_mem_util": row.get("handover_oracle_mem_util", ""),
            "load_regret": row.get("regret", ""),
            "q_margin": row.get("handover_q_margin", ""),
        }

    def _close_csv(self):
        if (
            self._main_fh is None
            and self._csv_fh is None
            and self._mobility_fh is None
            and self._handover_fh is None
            and self._decision_quality_fh is None
        ):
            return
        try:
            if self._main_fh is not None:
                self._main_fh.flush()
                self._main_fh.close()
            if self._csv_fh is not None:
                self._csv_fh.flush()
                self._csv_fh.close()
            if self._mobility_fh is not None:
                self._mobility_fh.flush()
                self._mobility_fh.close()
            if self._handover_fh is not None:
                self._handover_fh.flush()
                self._handover_fh.close()
            if self._decision_quality_fh is not None:
                self._decision_quality_fh.flush()
                self._decision_quality_fh.close()
        finally:
            self._main_fh = None
            self._main_writer = None
            self._csv_fh = None
            self._csv_writer = None
            self._mobility_fh = None
            self._mobility_writer = None
            self._handover_fh = None
            self._handover_writer = None
            self._decision_quality_fh = None
            self._decision_quality_writer = None
            self._csv_ready = False
            self._csv_rows_since_flush = 0

    @staticmethod
    def _mask_to_csv(mask):
        return ",".join(str(int(x)) for x in (mask or []))

    def _sync_policy_round(self, round_id):
        if hasattr(self.policy, "on_training_round"):
            self.policy.on_training_round(round_id)

    @staticmethod
    def _apply_realized_execution(task, action, context, execution):
        if str(getattr(execution, "task_id", "")) != str(getattr(task, "id", "")):
            raise ValueError("realized execution task identity does not match decision")
        delay = execution.delay
        energy = execution.energy
        is_local = execution.network_status is None
        compute_energy_j = energy.compute_it_energy_j + energy.facility_overhead_energy_j
        device_battery_energy_j = energy.device_battery_energy_j
        infrastructure_compute_energy_j = 0.0 if is_local else compute_energy_j
        server_battery_energy_j = sum(execution.server_battery_debits_j.values())
        communication_energy_j = (
            energy.device_tx_energy_j
            + energy.device_rx_energy_j
            + energy.infrastructure_network_energy_j
        )
        energy_breakdown = {
            "objective_energy_j": energy.objective_energy_j,
            "device_tx_energy_j": energy.device_tx_energy_j,
            "device_rx_energy_j": energy.device_rx_energy_j,
            "device_local_compute_energy_j": compute_energy_j if is_local else 0.0,
            "device_wait_energy_j": energy.device_wait_energy_j,
            "device_propulsion_energy_j": 0.0,
            "device_battery_energy_j": device_battery_energy_j,
            "device_paid_energy_j": device_battery_energy_j,
            "infra_compute_energy_j": infrastructure_compute_energy_j,
            "backhaul_energy_j": energy.infrastructure_network_energy_j,
            "infrastructure_network_energy_j": energy.infrastructure_network_energy_j,
            "facility_overhead_energy_j": energy.facility_overhead_energy_j,
            "server_hover_energy_j": float(
                getattr(execution, "server_hover_energy_j", 0.0) or 0.0
            ),
            "server_battery_energy_j": server_battery_energy_j,
            "communication_energy_j": communication_energy_j,
            "total_energy_j": energy.total_energy_j,
        }
        operator_total = execution.operator_cost_total
        if operator_total is None:
            raise ValueError("realized operator cost must be complete")
        # Operator cost is priced directly from config/servers.py (see
        # OperatorCost). It has three non-overlapping headings that sum to the
        # total: compute (per-tier vCPU price), transport (per-tier byte price),
        # and energy (electricity). For reporting, energy is further split into
        # the operating part vs the cooling/PUE-overhead part, using the same
        # electricity price -- so onboard, which has pue=1.0 and zero prices,
        # shows zero on every heading.
        oc = execution.operator_cost
        transport_cost = oc.transport_cost
        proc_cost = oc.compute_cost
        infra_overhead_energy_j = 0.0 if is_local else energy.facility_overhead_energy_j
        cooling_cost = infra_overhead_energy_j * ELECTRICITY_PRICE_PER_JOULE
        operation_energy_cost = oc.energy_cost - cooling_cost
        cost_breakdown = {
            "tx_cost": transport_cost,
            "proc_cost": proc_cost,
            "processing_cost": proc_cost,
            "communication_cost": transport_cost,
            "cooling_cost": cooling_cost,
            "operation_energy_cost": operation_energy_cost,
            "propulsion_cost": 0.0,
            "device_wait_cost": 0.0,
            "total_cost": operator_total,
            "accounting_perspective": "operator_cost",
            "complete": True,
        }
        delay_components = {
            "tx_wireless_ms": delay.wireless_transmission_ms,
            "tx_wired_ms": delay.wired_transmission_ms,
            "prop_onboard_ms": 0.0,
            "prop_wireless_ms": delay.wireless_propagation_ms,
            "prop_wired_ms": delay.wired_propagation_ms,
            "processing_ms": delay.processing_ms,
            "handover_delay_ms": delay.pre_access_wait_ms,
            "queue_delay_ms": delay.queue_ms,
            "retry_backoff_ms": delay.retry_backoff_ms,
            "retry_propagation_ms": delay.retry_propagation_ms,
            "outage_observation_ms": delay.outage_observation_ms,
            "terminal_wait_ms": delay.terminal_wait_ms,
            "network_elapsed_ms": delay.network_elapsed_ms,
            # Only the owner-agreed formula exists: observed_complete_latency =
            # complete_latency + coverage-outage wait + queue + retry backoff +
            # retry propagation + channel-outage observation + terminal wait.
            # There is no separate "objective"/"tier constant" delay concept --
            # this column reports the same complete_latency_ms value.
            "objective_delay_ms": delay.complete_latency_ms,
            "legacy_objective_delay_ms": delay.legacy_objective_delay_ms,
            "legacy_complete_latency_ms": delay.legacy_complete_latency_ms,
            "total_ms": delay.observed_complete_latency_ms,
            "total_ms_incl_processing": delay.observed_complete_latency_ms,
            "total_ms_incl_tx": delay.observed_complete_latency_ms,
            "complete_latency_ms": delay.complete_latency_ms,
            "observed_complete_latency_ms": delay.observed_complete_latency_ms,
        }
        context.setdefault("objective_delay", {})[action] = delay.complete_latency_ms
        context.setdefault("legacy_objective_delay", {})[
            action
        ] = delay.legacy_objective_delay_ms
        # Scheduler learning reads the one owner-agreed delay formula --
        # observed_complete_latency (see the comment on delay_components
        # above) -- there is no separate reward-only objective delay.
        context.setdefault("total_delay", {})[action] = delay.complete_latency_ms
        context.setdefault("observed_complete_delay", {})[action] = delay.observed_complete_latency_ms
        context.setdefault("delay_components", {})[action] = delay_components
        context.setdefault("energy", {})[action] = energy.objective_energy_j
        context.setdefault("energy_breakdown", {})[action] = energy_breakdown
        context.setdefault("cost", {})[action] = operator_total
        context.setdefault("cost_breakdown", {})[action] = cost_breakdown
        # Operational CO2e (grams): same grid-energy scope as
        # cost_breakdown["operation_energy_cost"] + cost_breakdown["cooling_cost"]
        # combined (i.e. oc.energy_cost), just priced in carbon instead of USD.
        # See config/emissions.py / metrics/emissions.py.
        context.setdefault("operational_co2e_g", {})[action] = float(
            getattr(execution, "operational_co2e_g", 0.0) or 0.0
        )
        # Realized end-to-end goodput on the access/backhaul path. Zero for local
        # execution (no radio hop). Because network time already reflects the
        # SINR-driven rate and retries, throughput naturally responds to
        # interference and channel quality.
        network_delay_s = delay.network_delay_s
        throughput_mbps = (
            (execution.on_wire_bytes * 8.0) / (network_delay_s * 1.0e6)
            if (not is_local and network_delay_s > 0.0 and execution.on_wire_bytes)
            else 0.0
        )
        context.setdefault("throughput_mbps", {})[action] = throughput_mbps
        context["measurement_basis"] = "realized_execution"
        context["execution_failed"] = not execution.application_completed

        task.measurement_basis = "realized_execution"
        task.network_status = (
            "local_bypass" if execution.network_status is None else execution.network_status.value
        )
        task.network_failure_reason = execution.failure_reason or ""
        task.payload_bytes = execution.payload_bytes
        task.on_wire_bytes = execution.on_wire_bytes
        task.packet_count = execution.packet_count
        task.packet_hop_count = execution.packet_hop_count
        task.pre_access_wait_ms = delay.pre_access_wait_ms
        task.network_queue_delay_ms = delay.queue_ms
        task.network_transmission_delay_ms = delay.transmission_ms
        task.network_propagation_delay_ms = delay.propagation_ms
        task.network_retry_backoff_ms = delay.retry_backoff_ms
        task.network_retry_propagation_ms = delay.retry_propagation_ms
        task.network_outage_observation_ms = delay.outage_observation_ms
        task.network_terminal_wait_ms = delay.terminal_wait_ms
        task.network_delay_ms = delay.network_elapsed_ms
        task.processing_delay_ms = delay.processing_ms
        task.completion_delay_ms = delay.total_ms
        task.deadline_missed = bool(execution.deadline_missed)
        task.objective_delay_ms = delay.complete_latency_ms
        task.legacy_objective_delay_ms = delay.legacy_objective_delay_ms
        task.delay = delay.observed_complete_latency_ms
        task.objective_energy_j = energy.objective_energy_j
        task.cost = operator_total
        task.communication_cost = transport_cost
        task.processing_cost = proc_cost
        task.operation_energy_cost = operation_energy_cost
        task.cost_accounting_perspective = "operator_cost"
        task.cost_complete = True
        task.cost_incomplete_reason = ""
        task._server_battery_debits_j = dict(execution.server_battery_debits_j)
        task._realized_execution = execution

    @staticmethod
    def _offered_load_and_congestion_percent(network_execution):
        """Derive CSV-only offered-load/congestion percentages from real hops.

        This is pure reporting: it only reads timing already recorded on the
        ``LinkReservation`` for each packet-hop attempt this task actually
        made (``network/links.py``) plus each hop's static declared capacity
        (``config/radio.get_link_profile``). It computes nothing that feeds
        back into ``context["mask"]``/``context["state"]``/reward, and it
        never estimates or fabricates a value for a hop that never reserved
        service (e.g. every attempt ended in channel outage before a
        reservation formed) -- those rows report blank via the ``(None,
        None)`` return, never a fabricated ``0``.

        For every real reservation on the route this task's packets crossed,
        two ratios are computed from that one reservation's own observed
        timestamps and its link's declared bit-rate:

        - congestion_ratio = queue_delay_s / (queue_delay_s +
          transmission_delay_s): the observed fraction of this packet-hop's
          time on the shared FIFO resource spent waiting behind already
          -queued traffic rather than being served. 0 when the resource was
          immediately available; approaches 1 as backlog dominates.
        - offered_ratio = (on_wire_bytes * 8) / (nominal_bit_rate_bps *
          transmission_delay_s): the bits actually clocked onto that hop's
          wire, divided by what its declared/nominal bit-rate could carry in
          that same real interval -- i.e. the achieved rate as a fraction of
          the link's rated capacity for this transmission.

        Both percentages reported for one row describe the single hop with
        the highest observed congestion_ratio (the bottleneck this task's
        packets actually experienced), so the pair always describes one
        physical link rather than mixing two different hops.
        """

        if network_execution is None:
            return None, None
        route = network_execution.transmission.route
        worst_congestion_ratio = None
        worst_offered_ratio = 0.0
        for packet_route in network_execution.transmission.packet_routes:
            for index, hop_result in enumerate(packet_route.hop_transmissions):
                resolved_hop = route.hops[index]
                for attempt in hop_result.attempts:
                    reservation = attempt.reservation
                    if reservation is None:
                        # Channel-outage-only attempts never reserved the
                        # shared resource; there is no queue/service
                        # timing to observe for them.
                        continue
                    queue_delay_s = reservation.queue_delay_s
                    service_delay_s = reservation.transmission_delay_s
                    congestion_ratio = bounded_ratio(
                        queue_delay_s, queue_delay_s + service_delay_s
                    )
                    if service_delay_s > 0.0:
                        try:
                            nominal_bit_rate_bps = float(
                                get_link_profile(resolved_hop.link.link_profile_id)[
                                    "nominal_bit_rate_bps"
                                ]
                            )
                        except KeyError:
                            offered_ratio = 0.0
                        else:
                            offered_ratio = offered_load_ratio(
                                reservation.on_wire_bytes * 8.0,
                                nominal_bit_rate_bps,
                                service_delay_s,
                            )
                    else:
                        # Zero on-wire bytes for this reservation (no data
                        # actually clocked onto the wire): no offered-load
                        # signal to report for it, never a fabricated ratio.
                        offered_ratio = 0.0
                    if (
                        worst_congestion_ratio is None
                        or congestion_ratio > worst_congestion_ratio
                    ):
                        worst_congestion_ratio = congestion_ratio
                        worst_offered_ratio = offered_ratio
        if worst_congestion_ratio is None:
            return None, None
        return (
            round(worst_offered_ratio * 100.0, 4),
            round(worst_congestion_ratio * 100.0, 4),
        )

    @staticmethod
    def _evaluate_reward(
        task,
        action,
        context,
        *,
        selected_action=None,
        force_failure=False,
    ):
        """Build one canonical reward observation from measured runner state."""

        deadline_ms = float(
            context.get("deadline_ms", getattr(task, "deadline_ms", 300.0))
            or 300.0
        )
        if action is None:
            objective_delay_ms = deadline_ms
            observed_delay_ms = deadline_ms
            energy_breakdown = {}
            cost = 0.0
            current_util = 0.0
            future_util = 0.0
            resource_fit = 0.0
            capacity_ratio = 0.0
            projected_server_battery_ratio = 1.0
        else:
            objectives = context.get("objective_delay", {}) or {}
            observed = context.get("observed_complete_delay", {}) or {}
            objective_delay_ms = float(
                objectives.get(action, deadline_ms)
            )
            observed_delay_ms = float(
                observed.get(action, objective_delay_ms)
            )
            energy_breakdown = (
                (context.get("energy_breakdown", {}) or {}).get(action, {}) or {}
            )
            cost = float((context.get("cost", {}) or {}).get(action, 0.0) or 0.0)
            current_util = float(
                (context.get("current_util", {}) or {}).get(action, 0.0) or 0.0
            )
            future_util = float(
                (context.get("future_util", {}) or {}).get(action, current_util)
                or current_util
            )
            resource_fit = float(
                (context.get("resource_fit", {}) or {}).get(action, 0.0) or 0.0
            )
            capacity_ratio = float(
                (context.get("tier_capacity_ratio", {}) or {}).get(action, 0.0)
                or 0.0
            )
            projected_server_battery_ratio = float(
                (context.get("projected_server_battery_ratio", {}) or {}).get(action, 1.0)
                or 0.0
            )

        execution_failed = bool(force_failure or context.get("execution_failed", False))
        deadline_missed = bool(
            execution_failed
            or getattr(task, "deadline_missed", False)
            or observed_delay_ms > deadline_ms
        )
        dwell_ms = float(getattr(task, "remaining_dwell_time_ms", 0.0) or 0.0)
        mobility_violation = bool(
            action in (ACTION_MEC, ACTION_MEC_COLLAB, ACTION_UAV_BS)
            and dwell_ms > 0.0
            and observed_delay_ms > dwell_ms
        )
        reward_action = action if selected_action is None else selected_action
        selected_future_util = float(
            (context.get("future_util", {}) or {}).get(
                reward_action,
                future_util,
            )
            or 0.0
        )
        # Missing services use a utilization sentinel in the state. They must
        # not be charged as overloaded resources when no resource executed.
        if action is None:
            selected_future_util = 0.0
        # None selects the reward profile's end-to-end delay target.  The
        # task's radio_latency_target_ms is a radio-only SLA descriptor and
        # must not scale an end-to-end delay score.
        delay_target_ms = float(context.get("delay_target_ms", 0.0) or 0.0) or None
        observation = RewardObservation(
            priority=int(getattr(task, "priority", 2) or 2),
            deadline_ms=deadline_ms,
            # The learning delay score uses the approved objective.  Physical
            # feasibility is supplied separately through deadline_missed and
            # mobility_violation, both evaluated against observed completion.
            complete_latency_ms=observed_delay_ms,
            delay_target_ms=delay_target_ms,
            execution_failed=execution_failed,
            invalid_action=bool(context.get("invalid_action", False)),
            deadline_missed=deadline_missed,
            device_battery_capacity_j=float(
                getattr(task, "battery_capacity_j", 0.0) or 0.0
            ),
            device_battery_remaining_j=float(
                getattr(task, "battery_remaining_j", 0.0) or 0.0
            ),
            device_battery_energy_j=float(
                energy_breakdown.get("device_battery_energy_j", 0.0) or 0.0
            ),
            total_energy_j=float(
                energy_breakdown.get("total_energy_j", 0.0) or 0.0
            ),
            cost=cost,
            current_utilization_percent=current_util,
            projected_utilization_percent=future_util,
            resource_fit_ratio=resource_fit,
            tier_capacity_ratio=capacity_ratio,
            projected_server_battery_ratio=projected_server_battery_ratio,
            mobility_violation=mobility_violation,
            shared_capacity=bool(action is not None and action != ACTION_LOCAL),
            communication_energy_j=float(
                energy_breakdown.get("communication_energy_j", 0.0) or 0.0
            ),
            device_propulsion_energy_j=float(
                energy_breakdown.get("device_propulsion_energy_j", 0.0) or 0.0
            ),
            server_hover_energy_j=float(
                energy_breakdown.get("server_hover_energy_j", 0.0) or 0.0
            ),
            selected_projected_utilization_percent=selected_future_util,
            deadline_penalty_exempt=False,
        )
        result = evaluate_reward(observation).as_dict()
        result["forced_single_action"] = 0
        result["deadline_penalty_exempt"] = 0
        return result

    def _release_local_allocations(self, current_time):
        """Release completed onboard reservations for every device."""

        for device_id in list(self._local_resource_allocations):
            active = self._local_resource_allocations[device_id]
            self._local_resource_allocations[device_id] = {
                task_id: reservation
                for task_id, reservation in active.items()
                if float(reservation[2]) > float(current_time)
            }
            if not self._local_resource_allocations[device_id]:
                del self._local_resource_allocations[device_id]

    def _apply_local_resource_state(self, task):
        """Expose this device's active onboard load to state and admission."""

        device_id = str(getattr(task, "device_id", getattr(task, "entity_id", "")))
        reservations = self._local_resource_allocations.get(device_id, {})
        compute_used = sum(float(item[0]) for item in reservations.values())
        memory_used = sum(int(item[1]) for item in reservations.values())
        compute_capacity = float(
            getattr(task, "local_compute_rate_cycles_per_s", 0.0) or 0.0
        )
        memory_capacity = int(
            getattr(task, "local_memory_capacity_bytes", 0) or 0
        )
        task.local_available_compute_rate_cycles_per_s = max(
            0.0,
            compute_capacity - compute_used,
        )
        task.local_available_memory_bytes = max(0, memory_capacity - memory_used)
        compute_util = (
            100.0 * compute_used / compute_capacity if compute_capacity > 0.0 else 100.0
        )
        memory_util = (
            100.0 * memory_used / memory_capacity if memory_capacity > 0 else 100.0
        )
        task.local_compute_utilization_percent = min(
            100.0,
            max(compute_util, memory_util),
        )

    def _allocate(self, task, server, current_time, release_time=None):
        if server is None:
            return False
        if str(getattr(server, "name", "")).startswith("LocalDevice"):
            device_id = str(
                getattr(task, "device_id", getattr(task, "entity_id", ""))
            )
            allocation = getattr(task, "resource_allocation", None)
            compute_rate = float(
                getattr(allocation, "allocated_compute_rate_cycles_per_s", 0.0)
                or getattr(task, "compute_rate_demand_cycles_per_s", 0.0)
                or 0.0
            )
            memory_bytes = int(
                getattr(allocation, "memory_allocated_bytes", 0)
                or getattr(task, "memory_working_set_bytes", 0)
                or 0
            )
            finish_time = (
                current_time + service_time_ms(server, task)
                if release_time is None
                else release_time
            )
            self._local_resource_allocations.setdefault(device_id, {})[
                str(getattr(task, "id", id(task)))
            ] = (compute_rate, memory_bytes, float(finish_time))
            task.set_server(server.name)
            task.server_assigned = server.name
            return True
        ok = server.allocate(
            task.id,
            task.cpu_demand,
            task.storage_demand,
            task.memory_demand,
            current_time + service_time_ms(server, task)
            if release_time is None
            else release_time,
            data_size_kb=getattr(task, "data_size_kb", None),
            resource_demand=getattr(task, "resource_demand", None),
        )
        if ok:
            task.set_server(server.name)
            task.server_assigned = server.name
        return bool(ok)

    def _select_allocatable_action(self, selected_action, context, task):
        """Attempt exactly the selected action, without a replacement tier."""
        selected_action = int(selected_action)
        action_servers = context.get("action_servers", {}) or {}
        physical = context.get("physical_mask", context.get("capacity_mask", []))
        route = [selected_action]
        if not (0 <= selected_action < len(physical)) or not physical[selected_action]:
            return None, None, route
        server = (
            self._local_server(task) if selected_action == ACTION_LOCAL
            else action_servers.get(selected_action)
        )
        return (selected_action, server, route) if server is not None else (None, None, route)

    def _learn_attempt(self, state, action, reward, mask, task):
        """Every attempted decision, including rejection, gets one transition."""
        if not self.train_mode:
            return None
        transition = build_transition(
            state=state, action=action, reward=reward, done=True,
            next_state=None, mask=mask, next_mask=list(mask), task=task,
            selected_tier=action_to_label(action),
        )
        loss = self.policy.learn(transition)
        self.visualizer.log_step(
            round_id=self.round_id, reward=reward, action=action,
            epsilon=getattr(self.policy, "epsilon", None), loss=loss,
            priority=getattr(task, "priority", 2),
        )
        return loss

    @staticmethod
    def _local_server(task):
        compute_capacity = max(
            1.0,
            float(getattr(task, "local_compute_rate_cycles_per_s", 1.0) or 1.0),
        )
        memory_capacity = max(
            1,
            int(getattr(task, "local_memory_capacity_bytes", 1) or 1),
        )
        return SimpleNamespace(
            name=f"LocalDevice_{getattr(task, 'device_id', 'x')}",
            latency=0.0,
            cpu_capacity=max(1.0, float(getattr(task, "local_cpu_capacity", 1.0) or 1.0)),
            memory_capacity=max(1.0, float(getattr(task, "local_memory_capacity", 1.0) or 1.0)),
            storage_capacity=max(1.0, float(getattr(task, "storage_demand", 1.0) or 1.0)),
            available_cpu=max(0.0, float(getattr(task, "local_cpu_capacity", 1.0) or 1.0) - float(getattr(task, "cpu_demand", 0.0) or 0.0)),
            available_memory=max(0.0, float(getattr(task, "local_memory_capacity", 1.0) or 1.0) - float(getattr(task, "memory_demand", 0.0) or 0.0)),
            available_storage=0.0,
            compute_rate_cycles_per_s=compute_capacity,
            memory_capacity_bytes=memory_capacity,
            storage_capacity_bytes=max(
                1,
                int(getattr(task, "storage_footprint_bytes", 1) or 1),
            ),
            available_compute_rate_cycles_per_s=max(
                0.0,
                float(
                    getattr(
                        task,
                        "local_available_compute_rate_cycles_per_s",
                        compute_capacity,
                    )
                    or 0.0
                ),
            ),
            available_memory_bytes=max(
                0,
                int(
                    getattr(
                        task,
                        "local_available_memory_bytes",
                        memory_capacity,
                    )
                    or 0
                ),
            ),
            available_storage_bytes=0,
            cost=0.0,
            tx_cost=0.0,
        )

    @staticmethod
    def _ordered_tasks(tasks):
        indexed = list(enumerate(tasks or []))
        return [task for _, task in sorted(indexed, key=lambda item: (int(getattr(item[1], "priority", 2) or 2), item[0]))]

    def schedule(self, tasks, servers, current_time):
        self.servers = servers
        self._release_local_allocations(current_time)
        self.round_id += 1
        self._sync_policy_round(self.round_id)
        assignments = []
        action_counts = Counter()
        priority_action_counts = {1: Counter(), 2: Counter(), 3: Counter()}
        priority_reward_sums = {1: 0.0, 2: 0.0, 3: 0.0}
        priority_reward_counts = {1: 0, 2: 0, 3: 0}
        reward_sum = 0.0
        reward_count = 0
        last_loss = None

        for task in self._ordered_tasks(tasks):
            self._apply_local_resource_state(task)
            task.device_battery_before_j = float(getattr(task, "battery_remaining_j", 0.0) or 0.0)
            state, context = self.state_builder.build(task, servers, getattr(self, "mobility", None))
            mask = self.mask_builder.build(context)
            context["mask"] = list(mask)
            context["forced_single_action"] = False
            context["no_service_available"] = not any(context.get("physical_mask", []))
            selected_action = int(self.policy.select_action(state, mask))
            action, server, route = self._select_allocatable_action(selected_action, context, task)
            context["selected_action"] = selected_action
            context["executed_action"] = action
            context["spillover_invoked"] = bool(
                action is not None and action != selected_action
            )
            if action is None or server is None:
                task.fail()
                task.network_failure_reason = "selected_action_not_physically_feasible"
                if context["no_service_available"]:
                    task.network_failure_reason = "no_service_available"
                context["invalid_action"] = not context["no_service_available"]
                context["execution_failed"] = True
                reward_info = self._evaluate_reward(task, None, context, selected_action=selected_action, force_failure=True)
                reward = reward_info["reward"]
                loss = self._learn_attempt(state, selected_action, reward, mask, task)
                if loss is not None:
                    last_loss = loss
                priority = int(getattr(task, "priority", 2) or 2)
                label = action_to_label(selected_action)
                action_counts[label] += 1
                if priority in priority_action_counts:
                    priority_action_counts[priority][label] += 1
                    priority_reward_sums[priority] += reward
                    priority_reward_counts[priority] += 1
                eps = getattr(self.policy, "epsilon", None)
                if eps is not None:
                    self.epsilon_log.append({"round": self.round_id, "epsilon": float(eps)})
                self.reward_log.append({"round": self.round_id, "reward": reward, "action": selected_action, "priority": priority})
                reward_sum += reward
                reward_count += 1
                self._append_row(self._row(task, selected_action, None, None, route, context, mask, reward, "failed", reward_info=reward_info))
                continue
            execution = None
            allocation_succeeded = False
            if self._execution_runtime is not None:
                execution = self._execution_runtime(
                    task=task,
                    server=server,
                    current_time=current_time,
                    action=action,
                    context=context,
                )
                self._apply_realized_execution(task, action, context, execution)
                task.server_battery_before_j = float(
                    getattr(server, "battery_remaining_j", 0.0) or 0.0
                )
                if execution.application_completed:
                    allocation_succeeded = self._allocate(
                        task,
                        server,
                        current_time,
                        release_time=execution.finished_at_ms,
                    )
                    if allocation_succeeded:
                        task.complete(
                            start_time=execution.started_at_ms,
                            end_time=execution.finished_at_ms,
                        )
                    else:
                        task.fail()
                        task.network_failure_reason = "compute_admission_failed_after_transfer"
                        context["execution_failed"] = True
                else:
                    task.fail()
            else:
                allocation_succeeded = self._allocate(task, server, current_time)
                if not allocation_succeeded:
                    task.fail()
                    task.network_failure_reason = "compute_admission_failed"
                    context["execution_failed"] = True

            reward_info = self._evaluate_reward(
                task,
                action,
                context,
                selected_action=selected_action,
                force_failure=execution is None,
            )
            reward = reward_info["reward"]
            energy_breakdown = (context.get("energy_breakdown", {}) or {}).get(action, {}) or {}
            task.device_tx_energy_j = float(energy_breakdown.get("device_tx_energy_j", 0.0) or 0.0)
            task.device_rx_energy_j = float(energy_breakdown.get("device_rx_energy_j", 0.0) or 0.0)
            task.device_local_compute_energy_j = float(energy_breakdown.get("device_local_compute_energy_j", 0.0) or 0.0)
            task.device_wait_energy_j = float(energy_breakdown.get("device_wait_energy_j", 0.0) or 0.0)
            task.device_propulsion_energy_j = float(energy_breakdown.get("device_propulsion_energy_j", 0.0) or 0.0)
            task.device_paid_energy_j = float(energy_breakdown.get("device_battery_energy_j", 0.0) or 0.0)
            task.infra_compute_energy_j = float(energy_breakdown.get("infra_compute_energy_j", 0.0) or 0.0)
            task.backhaul_energy_j = float(energy_breakdown.get("backhaul_energy_j", 0.0) or 0.0)
            task.server_hover_energy_j = float(energy_breakdown.get("server_hover_energy_j", 0.0) or 0.0)
            task.device_battery_energy_j = float(energy_breakdown.get("device_battery_energy_j", 0.0) or 0.0)
            task.server_battery_energy_j = float(energy_breakdown.get("server_battery_energy_j", 0.0) or 0.0)
            task.communication_energy_j = float(energy_breakdown.get("communication_energy_j", 0.0) or 0.0)
            task.total_energy_j = float(energy_breakdown.get("total_energy_j", reward_info.get("energy", 0.0)) or 0.0)
            task.energy = task.total_energy_j
            task.local_compute_used = bool(action == ACTION_LOCAL)
            if execution is None:
                task.server_battery_before_j = float(getattr(server, "battery_remaining_j", 0.0) or 0.0) if server is not None else 0.0
            # This is the task-specific battery debit. Continuous propulsion
            # is charged once per round by MobilityManager and only reported
            # in the task energy breakdown, so it must not be subtracted here.
            battery_draw = float(getattr(task, "device_paid_energy_j", 0.0) or 0.0)
            task.battery_draw_j = battery_draw
            task.battery_remaining_j = max(0.0, float(getattr(task, "battery_remaining_j", 0.0) or 0.0) - battery_draw)
            task.device_battery_after_j = task.battery_remaining_j
            if execution is not None:
                servers_by_name = {
                    str(getattr(item, "name", "")): item for item in (self.servers or [])
                }
                for server_id, energy_j in execution.server_battery_debits_j.items():
                    battery_server = servers_by_name.get(server_id)
                    if battery_server is not None and hasattr(battery_server, "consume_battery"):
                        battery_server.consume_battery(energy_j)
            elif server is not None and hasattr(server, "consume_battery"):
                # Compute + backhaul communication energy -- propulsion/
                # hover energy is debited separately, once per round
                # regardless of workload (see mobility/manager.py).
                server.consume_battery(task.infra_compute_energy_j + task.backhaul_energy_j)
            task.server_battery_after_j = float(getattr(server, "battery_remaining_j", task.server_battery_before_j) or 0.0) if server is not None else 0.0
            if getattr(task, "entity_id", None) is not None and getattr(self, "mobility", None) is not None:
                try:
                    self.mobility.entities[int(task.entity_id)].battery_remaining_j = task.battery_remaining_j
                except Exception:
                    pass
            loss = self._learn_attempt(state, selected_action, reward, mask, task)
            if loss is not None:
                last_loss = loss
            eps = getattr(self.policy, "epsilon", None)
            self.reward_log.append({"round": self.round_id, "reward": reward, "action": selected_action, "priority": getattr(task, "priority", 2)})
            if eps is not None:
                self.epsilon_log.append({"round": self.round_id, "epsilon": float(eps)})
            action_label = action_to_label(action)
            priority = int(getattr(task, "priority", 2) or 2)
            action_counts[action_label] += 1
            if priority in priority_action_counts:
                priority_action_counts[priority][action_label] += 1
                priority_reward_sums[priority] += reward
                priority_reward_counts[priority] += 1
            reward_sum += reward
            reward_count += 1
            row_status = (
                "completed"
                if execution is not None and task.status == "completed"
                else "failed"
                if execution is not None or not allocation_succeeded
                else "assigned"
            )
            self._append_row(self._row(task, selected_action, action, server, route, context, mask, reward, row_status, reward_info=reward_info))
            if allocation_succeeded:
                assignments.append((task, server))

        action_labels = [action_to_label(i) for i in range(self.action_dim)]
        total = max(1, sum(action_counts.values()))
        self.action_dist_log.append({
            "round": self.round_id,
            "priority": "ALL",
            "priority_class": "ALL",
            **{label: action_counts.get(label, 0) / total for label in action_labels},
            "avg_reward": reward_sum / max(1, reward_count),
        })
        for priority, priority_label in ((1, "URLLC"), (2, "mMTC"), (3, "eMBB")):
            counts = priority_action_counts[priority]
            priority_total = sum(counts.values())
            if priority_total <= 0:
                continue
            self.action_dist_log.append({
                "round": self.round_id,
                "priority": priority,
                "priority_class": priority_label,
                **{label: counts.get(label, 0) / priority_total for label in action_labels},
                "avg_reward": priority_reward_sums[priority] / max(1, priority_reward_counts[priority]),
            })
        if last_loss is not None:
            self.loss_log.append({"round": self.round_id, "loss": last_loss})
        return assignments

    def _row(self, task, selected_action, action, server, route, context, mask, reward, status, reward_info=None):
        reward_info = reward_info or {}
        self._diag_decisions += 1
        allocated = action_to_label(action) if action is not None else ""
        selected_tier = (
            action_to_label(selected_action) if selected_action is not None else ""
        )
        spillover_used = bool(
            selected_action is not None
            and action is not None
            and int(selected_action) != int(action)
        )
        delay = context.get("total_delay", {}).get(action, "") if action is not None else ""
        cost = context.get("cost", {}).get(action, "") if action is not None else ""
        cost_breakdown = (context.get("cost_breakdown", {}) or {}).get(action, {}) if action is not None else {}
        energy = context.get("energy", {}).get(action, "") if action is not None else ""
        energy_breakdown = (context.get("energy_breakdown", {}) or {}).get(action, {}) if action is not None else {}
        operational_co2e_g = context.get("operational_co2e_g", {}).get(action, "") if action is not None else ""
        future_util = context.get("future_util", {}).get(action, "") if action is not None else ""
        deadline_ms = float(context.get("deadline_ms", getattr(task, "deadline_ms", 0.0)) or 0.0)
        realized_basis = context.get("measurement_basis") == "realized_execution"
        access_channel = None
        offered_load_percent = None
        network_congestion_percent = None
        if realized_basis:
            execution = getattr(task, "_realized_execution", None)
            network_execution = getattr(execution, "network_execution", None)
            if network_execution is not None:
                # Report the latest observed access attempt, not an estimated
                # serving-BS RSSI or a later wired/backhaul hop.
                for packet_route in reversed(network_execution.transmission.packet_routes):
                    if not packet_route.hop_transmissions:
                        continue
                    for attempt in reversed(packet_route.hop_transmissions[0].attempts):
                        if attempt.channel_result.rx_power_dbm is not None:
                            access_channel = attempt.channel_result
                            break
                    if access_channel is not None:
                        break
                # Real, observed offered-load/congestion from this task's own
                # shared-link reservations -- see
                # _offered_load_and_congestion_percent for the formulas. Blank
                # ("" below) when no hop ever reserved service, never 0.
                offered_load_percent, network_congestion_percent = (
                    self._offered_load_and_congestion_percent(network_execution)
                )
        reported_signal_dbm = (
            access_channel.rx_power_dbm if access_channel is not None else
            ("" if realized_basis else getattr(task, "signal_dbm", ""))
        )
        reported_snr_db = (
            access_channel.snr_db if access_channel is not None else
            ("" if realized_basis else getattr(task, "snr_db", ""))
        )
        reported_sinr_db = (
            access_channel.sinr_db if access_channel is not None else
            ("" if realized_basis else getattr(task, "sinr_db", ""))
        )
        reported_noise_dbm = (
            access_channel.noise_power_dbm if access_channel is not None else
            ("" if realized_basis else getattr(task, "noise_dbm", ""))
        )
        reported_interference_dbm = (
            access_channel.interference_power_dbm if access_channel is not None else
            ("" if realized_basis else getattr(task, "interference_dbm", ""))
        )
        deadline_missed = (
            int(bool(getattr(task, "deadline_missed", False)))
            if realized_basis
            else int(bool(delay != "" and deadline_ms > 0.0 and float(delay) > deadline_ms))
        )
        util = server_utilization(server)
        selected_server = (
            (context.get("action_servers", {}) or {}).get(selected_action)
            if selected_action is not None and selected_action != ACTION_LOCAL
            else None
        )
        selected_util = (
            server_utilization(selected_server)
            if selected_server is not None
            else {"cpu": 0.0, "memory": 0.0, "storage": 0.0, "congestion": 0.0}
        )
        mobility = context.get("mobility", {}) or {}
        device_id = getattr(task, "device_id", getattr(task, "entity_id", ""))
        priority = int(getattr(task, "priority", 2) or 2)
        priority_class = {1: "URLLC", 2: "mMTC", 3: "eMBB"}.get(priority, str(priority))
        mask_csv = self._mask_to_csv(mask)
        physical_mask = context.get("physical_mask", context.get("capacity_mask", mask)) or []
        learning_mask = context.get("learning_mask", mask) or []
        physical_mask_csv = self._mask_to_csv(physical_mask)
        learning_mask_csv = self._mask_to_csv(learning_mask)
        future = context.get("future_util", {}) or {}
        server_id = getattr(server, "name", "")
        # Real per-segment breakdown from CommonState._action_delay_components
        # (see scheduler/common/common_state.py) -- computed once alongside
        # total_delay so these never drift from the aggregate the way the
        # old min/max handover-delay split (and the old hardcoded-0 wired/
        # onboard fields) used to.
        delay_parts = (context.get("delay_components", {}) or {}).get(action, {}) or {} if action is not None else {}
        tx_wireless_ms = float(delay_parts.get("tx_wireless_ms", 0.0) or 0.0)
        tx_wired_ms = float(delay_parts.get("tx_wired_ms", 0.0) or 0.0)
        prop_onboard_ms = float(delay_parts.get("prop_onboard_ms", 0.0) or 0.0)
        prop_wireless_ms = float(delay_parts.get("prop_wireless_ms", 0.0) or 0.0)
        prop_wired_ms = float(delay_parts.get("prop_wired_ms", 0.0) or 0.0)
        processing_component_ms = float(delay_parts.get("processing_ms", 0.0) or 0.0)
        # Collective (wireless+wired combined) transmission/propagation totals,
        # alongside the already-separate individual segments above.
        transmission_total_ms = tx_wireless_ms + tx_wired_ms
        propagation_total_ms = prop_wireless_ms + prop_wired_ms
        # Both completion views are copied directly from TaskDelayBreakdown.
        # No reporting-time formula is allowed here.
        objective_delay_ms = float(delay_parts.get("objective_delay_ms", 0.0) or 0.0) if action is not None else ""
        legacy_objective_delay_ms = float(delay_parts.get("legacy_objective_delay_ms", 0.0) or 0.0) if action is not None else ""
        legacy_complete_latency_ms = float(delay_parts.get("legacy_complete_latency_ms", 0.0) or 0.0) if action is not None else ""
        e2e_excl_tx_ms = float(delay_parts.get("total_ms_incl_processing", 0.0) or 0.0) if action is not None else ""
        e2e_incl_tx_ms = float(delay_parts.get("total_ms_incl_tx", 0.0) or 0.0) if action is not None else ""
        complete_latency_ms = float(delay_parts.get("complete_latency_ms", e2e_incl_tx_ms) or 0.0) if action is not None else ""
        observed_complete_latency_ms = float(delay_parts.get("observed_complete_latency_ms", e2e_incl_tx_ms) or 0.0) if action is not None else ""
        # Both operands are "" on the no-action (allocation failed) path, so the
        # subtraction must coerce first -- exactly like every other numeric
        # column above. Without this the failure path raises
        # "TypeError: unsupported operand type(s) for -: 'str' and 'float'".
        e2e_no_processing_ms = max(
            0.0,
            float(e2e_incl_tx_ms or 0.0) - processing_component_ms
            if realized_basis and action is not None
            else float(observed_complete_latency_ms or 0.0) - processing_component_ms,
        )

        # ---- Mobility / handover decision-quality (from HandoverScheduler
        # via core/simulator.py; see mobility/handover_scheduler.py) ----
        handover_executed = int(getattr(task, "handover_executed", 0) or 0)
        handover_failure_flag = int(getattr(task, "handover_failure_flag", 0) or 0)
        handover_success_flag = int(handover_executed == 1 and handover_failure_flag == 0)
        remaining_dwell_ms = getattr(task, "remaining_dwell_time_ms", "")
        mobility_wireless_action = action in (ACTION_MEC, ACTION_MEC_COLLAB, ACTION_UAV_BS)
        mobility_violation = int(bool(
            remaining_dwell_ms != "" and remaining_dwell_ms is not None
            and observed_complete_latency_ms != ""
            and float(observed_complete_latency_ms or 0.0) > float(remaining_dwell_ms or 0.0)
            and mobility_wireless_action
        ))
        completion_before_exit = int(not mobility_violation) if mobility_wireless_action else 1

        # ---- Serving-BS spatial context: ground tiers are static, UAV_BS
        # patrols so its (x, y, altitude) genuinely varies per round -- lets
        # later analysis correlate coverage/handover quality with UAV
        # position instead of only the device's own position.
        server_x = getattr(server, "x", "") if server is not None else ""
        server_y = getattr(server, "y", "") if server is not None else ""
        if server is not None:
            server_z = getattr(server, "z", None)
            if server_z is None:
                server_z = getattr(getattr(server, "mobility", None), "z", None)
            server_altitude_m = float(server_z) if server_z is not None else 0.0
        else:
            server_altitude_m = ""

        # ---- Rolling learning-stability diagnostics: computed causally from
        # realized choices/outcomes (last 50 per class) so the same logic
        # applies uniformly whether the policy is value-based (DQN,
        # Q-learning) or exposes its own action probabilities (PPO, A2C).
        class_action_hist = self._recent_actions_by_class.setdefault(priority, deque(maxlen=50))
        if action is not None:
            class_action_hist.append(int(action))
        if class_action_hist:
            counts = Counter(class_action_hist)
            total_n = len(class_action_hist)
            action_entropy = -sum((c / total_n) * math.log(c / total_n) for c in counts.values())
        else:
            action_entropy = 0.0

        prev_action = self._last_action_by_device.get(device_id)
        action_change_flag = int(prev_action is not None and action is not None and int(action) != int(prev_action))
        if action is not None:
            self._last_action_by_device[device_id] = int(action)
        class_change_hist = self._recent_change_by_class.setdefault(priority, deque(maxlen=50))
        class_change_hist.append(action_change_flag)
        action_change_rate = sum(class_change_hist) / len(class_change_hist) if class_change_hist else 0.0

        # Jain's fairness index over each class's rolling deadline-success
        # rate -- are URLLC/mMTC/eMBB being served proportionately fairly,
        # not just individually well. Bounded [0,1] and comparable across
        # classes regardless of their differing reward scales.
        class_success_hist = self._recent_success_by_class.setdefault(priority, deque(maxlen=50))
        class_success_hist.append(0 if deadline_missed else 1)
        class_success_means = [
            sum(hist) / len(hist) for hist in self._recent_success_by_class.values() if hist
        ]
        sum_sq = sum(m * m for m in class_success_means)
        if len(class_success_means) >= 2 and sum_sq > 0:
            priority_fairness_index = (sum(class_success_means) ** 2) / (len(class_success_means) * sum_sq)
        else:
            priority_fairness_index = 1.0

        # ---- Efficiency ratios: goodput per unit energy/cost. ----
        data_kb = float(getattr(task, "data_size_kb", 0.0) or 0.0)
        total_energy_j_val = float(energy_breakdown.get("total_energy_j", energy) or 0.0) if energy != "" else 0.0
        useful_payload_bytes = (
            int(getattr(task, "payload_bytes", 0) or 0)
            if realized_basis and bool(getattr(task, "status", "") == "completed")
            else 0
        )
        energy_efficiency = (
            (useful_payload_bytes * 8.0 / total_energy_j_val)
            if realized_basis and total_energy_j_val > 0.0
            else (data_kb / total_energy_j_val)
            if not realized_basis and total_energy_j_val > 1e-9
            else ""
        )
        cost_val = float(cost or 0.0) if cost != "" else 0.0
        cost_efficiency = (
            (useful_payload_bytes / cost_val)
            if realized_basis and cost_val > 0.0 and useful_payload_bytes > 0
            else (data_kb / cost_val)
            if not realized_basis and cost_val > 1e-9
            else ""
        )
        unit_data_cost = (
            cost_val / useful_payload_bytes
            if realized_basis and useful_payload_bytes > 0
            else ""
        )
        on_wire_bytes = int(getattr(task, "on_wire_bytes", 0) or 0)
        wire_efficiency = (
            useful_payload_bytes / on_wire_bytes if realized_basis and on_wire_bytes > 0 else ""
        )

        row = {
            "round": self.round_id,
            "vehicle": device_id,
            "vehicle_id": device_id,
            "task_id": getattr(task, "id", ""),
            "device_id": device_id,
            "device_type": getattr(task, "device_type", "generic"),
            "service_class": getattr(task, "service_class", ""),
            "device_profile": getattr(task, "device_type", "generic"),
            "application_type": getattr(task, "application_type", getattr(task, "device_type", "generic")),
            "mobility_profile": getattr(task, "mobility_profile", "unknown"),
            "movement_speed_m_s": getattr(task, "movement_speed_m_s", ""),
            "handover_sensitivity": getattr(task, "handover_sensitivity", ""),
            "dwell_weight": getattr(task, "dwell_weight", ""),
            "workload_mode": "smartcity",
            "workload_regime": getattr(task, "device_type", "generic"),
            "active_devices_this_round": self.active_devices_this_round,
            "priority": priority,
            "priority_class": priority_class,
            "class_name": priority_class,
            "deadline_ms": deadline_ms,
            "sla_profile_id": getattr(task, "sla_profile_id", ""),
            "sla_generation": getattr(task, "sla_generation", ""),
            "radio_latency_target_ms": getattr(
                task, "radio_latency_target_ms", ""
            ),
            "application_deadline_ms": getattr(
                task, "application_deadline_ms", deadline_ms
            ),
            "sla_maturity": getattr(task, "sla_maturity", ""),
            "sla_source_id": getattr(task, "sla_source_id", ""),
            "deadline_delay_ms": delay,
            "measurement_basis": context.get("measurement_basis", "predicted_decision"),
            "network_status": getattr(task, "network_status", "") if realized_basis else "unobserved",
            "network_failure_reason": getattr(task, "network_failure_reason", "") if realized_basis else "",
            "payload_bytes": getattr(task, "payload_bytes", "") if realized_basis else "",
            "on_wire_bytes": getattr(task, "on_wire_bytes", "") if realized_basis else "",
            "packet_count": getattr(task, "packet_count", "") if realized_basis else "",
            "packet_hop_count": getattr(task, "packet_hop_count", "") if realized_basis else "",
            # Canonical Task-51 resource inputs (always available on the task, so
            # exported for both predicted and realized rows).
            "compute_work_cycles": getattr(task, "compute_work_cycles", ""),
            "allocated_compute_rate_cycles_per_s": getattr(task, "allocated_compute_rate_cycles_per_s", ""),
            "memory_working_set_bytes": getattr(task, "memory_working_set_bytes", ""),
            "storage_footprint_bytes": getattr(task, "storage_footprint_bytes", ""),
            "input_payload_bytes": getattr(task, "input_payload_bytes", ""),
            "resource_conversion_profile_id": getattr(task, "resource_conversion_profile_id", ""),
            "local_compute_capacity_cycles_per_s": getattr(task, "local_compute_capacity_cycles_per_s", ""),
            "data_size_kb": getattr(task, "data_size_kb", ""),
            "total_workload_kb": getattr(task, "data_size_kb", ""),
            "connected_bs": getattr(task, "wireless_ap", ""),
            "connected_bs_idx": getattr(task, "wireless_ap", ""),
            "chosen_bs_idx": server_id,
            "oracle_bs_idx": getattr(task, "oracle_bs", ""),
            "wireless_chosen_bs": getattr(task, "wireless_ap", ""),
            "processing_server": server_id,
            "processing_tier": allocated,
            "wireless_ap": getattr(task, "wireless_ap", ""),
            "pos_x": getattr(task, "pos_x", ""),
            "pos_y": getattr(task, "pos_y", ""),
            "server_x": server_x,
            "server_y": server_y,
            "server_altitude_m": server_altitude_m,
            "vehicle_position_m": getattr(task, "pos_x", ""),
            "distance_to_bs": getattr(task, "distance_to_bs", ""),
            "distance_to_bs_m": getattr(task, "distance_to_bs", ""),
            "signal_dbm": reported_signal_dbm,
            "RSSI_dBm": reported_signal_dbm,
            "chosen_rssi_dbm": reported_signal_dbm,
            "oracle_rssi_dbm": getattr(task, "oracle_rssi_dbm", ""),
            "rssi_regret_db": getattr(task, "rssi_regret_db", ""),
            "handover_prev_rssi_dbm": getattr(task, "handover_prev_rssi_dbm", ""),
            "handover_target_bs": getattr(task, "handover_target_bs", ""),
            "handover_target_rssi_dbm": getattr(task, "handover_target_rssi_dbm", ""),
            "handover_target_rssi_gain_db": getattr(task, "handover_target_rssi_gain_db", ""),
            "handover_best_available_bs": getattr(task, "handover_best_available_bs", ""),
            "handover_best_available_rssi_dbm": getattr(task, "handover_best_available_rssi_dbm", ""),
            "SNR_dB": reported_snr_db,
            "SINR_dB": reported_sinr_db,
            "Noise_dBm": reported_noise_dbm,
            "Interference_dBm": reported_interference_dbm,
            "vehicle_speed_mps": getattr(task, "movement_speed_m_s", ""),
            "speed_norm": min(1.0, float(getattr(task, "movement_speed_m_s", 0.0) or 0.0) / 30.0),
            "predicted_dwell_time_ms": getattr(task, "predicted_dwell_time_ms", ""),
            "remaining_dwell_time_ms": getattr(task, "remaining_dwell_time_ms", ""),
            "handover_flag": int(float(getattr(task, "handover_delay_ms", 0.0) or 0.0) > 0.0),
            "handover_executed": handover_executed,
            "handover_delay_ms": getattr(task, "handover_delay_ms", ""),
            "handover_interruption_ms": getattr(task, "handover_interruption_ms", ""),
            "handover_failure": handover_failure_flag,
            "handover_success_flag": handover_success_flag,
            "handover_failure_flag": handover_failure_flag,
            "handover_reason": getattr(task, "handover_trigger_reason", ""),
            "handover_trigger_reason": getattr(task, "handover_trigger_reason", ""),
            "pingpong_handover_flag": getattr(task, "pingpong_flag", 0),
            "bs_residence_time_ms": getattr(task, "bs_residence_time_ms", ""),
            "handover_stability_time_ms": getattr(task, "remaining_dwell_time_ms", ""),
            "coverage_outage_duration_ms": getattr(task, "coverage_outage_duration_ms", ""),
            "coverage_outage_flag": getattr(task, "coverage_outage_flag", 0),
            "served_via_anchor": getattr(task, "served_via_anchor", 0),
            "completion_before_exit": completion_before_exit,
            "mobility_induced_violation": mobility_violation,
            "handover_prev_bs": getattr(task, "prev_attached_bs", ""),
            "handover_q_margin": getattr(task, "handover_q_margin", ""),
            "handover_chosen_cpu_util": getattr(task, "handover_chosen_cpu_util", ""),
            "handover_chosen_mem_util": getattr(task, "handover_chosen_mem_util", ""),
            "handover_oracle_cpu_util": getattr(task, "handover_oracle_cpu_util", ""),
            "handover_oracle_mem_util": getattr(task, "handover_oracle_mem_util", ""),
            "mobility_handover_count": mobility.get("ho_count", ""),
            "mobility_handover_delay_ms": mobility.get("ho_delay", ""),
            "mobility_avg_pos": mobility.get("avg_pos", ""),
            "selected_action": int(selected_action) if selected_action is not None else "",
            "selected_tier": selected_tier,
            "rl_action": int(selected_action) if selected_action is not None else "",
            "action_final": int(action) if action is not None else "",
            "rl_tier_name": selected_tier,
            "oracle_action": "",
            "oracle_tier": "",
            "oracle_reward": "",
            "match_oracle": "",
            "regret": "",
            "delay_regret_ms": "",
            "cost_regret": "",
            "energy_regret": "",
            "reward_regret": "",
            "allocated_tier": allocated,
            "execution_tier": allocated,
            "cost_tier": allocated,
            "offloading_tier": allocated,
            "allocated_server": getattr(server, "name", ""),
            "selected_server_id": server_id,
            "selected_server_type": allocated,
            "offload_route": "->".join(action_to_label(a) for a in route),
            "mask": mask_csv,
            "physical_mask": physical_mask_csv,
            "learning_mask": learning_mask_csv,
            "valid_count": int(sum(mask or [])),
            "physical_valid_count": int(sum(physical_mask or [])),
            "learning_valid_count": int(sum(learning_mask or [])),
            "final_valid_count": int(sum(mask or [])),
            "action_entropy": round(float(action_entropy), 6),
            "rl_reward": round(float(reward), 6),
            "predicted_rl_reward": "",
            "raw_reward": round(float(reward_info.get("raw_reward", reward)), 6),
            "reward_deadline_penalty_norm": reward_info.get("deadline_pen", ""),
            "reward_util_reward": reward_info.get("util_reward", ""),
            "reward_overutilization_penalty": reward_info.get("overutilization_pen", ""),
            "reward_overutilization_weight": reward_info.get("overutilization_weight", ""),
            "reward_overutilization_deduction_points": reward_info.get("overutilization_deduction_points", ""),
            "reward_schema_version": REWARD_SCHEMA_VERSION,
            "no_service_available": int(context.get("no_service_available", False)),
            "reward_invalid_action_penalty": reward_info.get("invalid_pen", ""),
            # Keep the three legacy column names populated for downstream
            # notebooks, and also report every canonical objective explicitly.
            "reward_deadline_weight": reward_info.get("delay_weight", ""),
            "reward_cost_weight": reward_info.get("cost_weight", ""),
            "reward_energy_weight": reward_info.get("device_battery_weight", ""),
            "reward_device_battery_weight": reward_info.get("device_battery_weight", ""),
            "reward_server_battery_weight": reward_info.get("server_battery_weight", ""),
            "reward_compute_weight": reward_info.get("compute_weight", ""),
            "reward_mobility_weight": reward_info.get("mobility_weight", ""),
            "only_cloud_feasible": int(
                bool(physical_mask) and int(sum(physical_mask)) == 1 and len(physical_mask) > ACTION_CLOUD and int(physical_mask[ACTION_CLOUD]) == 1
            ),
            "collab_feasible": int(len(physical_mask) > ACTION_MEC_COLLAB and int(physical_mask[ACTION_MEC_COLLAB]) == 1),
            "chosen_layer": int(action) + 1 if action is not None else "",
            "overflow_used": int(spillover_used),
            "overflow_route": (
                "->".join(action_to_label(item) for item in route)
                if spillover_used
                else ""
            ),
            "overflow_fragment_count": 0,
            "overflow_allocated_kb": 0,
            "overflow_leftover_kb": 0,
            "overflow_fully_allocated": int(spillover_used and action is not None),
            "overflow_fragment_tiers": "",
            "overflow_fragment_kb": "",
            "overflow_fragment_cpu_util": "",
            "overflow_fragment_mem_util": "",
            "collab_checked": int(ACTION_MEC_COLLAB in route),
            "collab_found": int(physical_mask[ACTION_MEC_COLLAB] == 1) if len(physical_mask) > ACTION_MEC_COLLAB else 0,
            "collab_used": int(action == ACTION_MEC_COLLAB),
            "collab_dst_id": server_id if action == ACTION_MEC_COLLAB else "",
            "collab_t_budget": deadline_ms if action == ACTION_MEC_COLLAB else "",
            "collab_t_comm": delay if action == ACTION_MEC_COLLAB else "",
            "collab_t_exec": "",
            "collab_t_total": delay if action == ACTION_MEC_COLLAB else "",
            "collab_reason": "selected" if action == ACTION_MEC_COLLAB else "",
            "collab_reject_diff_bs": 0,
            "collab_reject_no_bs": 0,
            "delay_ms": observed_complete_latency_ms,
            "objective_delay_ms": objective_delay_ms,
            "legacy_objective_delay_ms": legacy_objective_delay_ms,
            "legacy_complete_latency_ms": legacy_complete_latency_ms,
            "trans_delay_ms": transmission_total_ms,
            "tx_delay_wireless_ms": tx_wireless_ms,
            "tx_delay_wired_ms": tx_wired_ms,
            "propagation_wireless_ms": prop_wireless_ms,
            "propagation_wired_ms": prop_wired_ms,
            "propagation_total_ms": propagation_total_ms,
            "prop_onboard_ms": prop_onboard_ms,
            "queue_length": "",
            "queue_length_total": "",
            "queue_length_urllc": "",
            "queue_length_mmtc": "",
            "queue_length_embb": "",
            "queued_workload_ms": "",
            "estimated_waiting_time_ms": "",
            "queue_delay_ms": delay_parts.get("queue_delay_ms", ""),
            "waiting_time_ms": delay_parts.get("queue_delay_ms", ""),
            "pre_access_wait_ms": delay_parts.get("handover_delay_ms", ""),
            "network_retry_backoff_ms": delay_parts.get("retry_backoff_ms", ""),
            "network_retry_propagation_ms": delay_parts.get("retry_propagation_ms", ""),
            "network_outage_observation_ms": delay_parts.get("outage_observation_ms", ""),
            "network_terminal_wait_ms": delay_parts.get("terminal_wait_ms", ""),
            "network_delay_ms": delay_parts.get("network_elapsed_ms", ""),
            # Reused from delay_parts (computed once in
            # CommonState._action_delay_components at decision time) instead
            # of re-calling service_time_ms()/processing_time_ms() here --
            # those read live, mutating server utilization, so a second call
            # at reporting time (after other tasks this round have already
            # been admitted) would silently diverge from the congestion
            # snapshot the scheduler's actual decision/reward was based on.
            "starvation_flag": 0,
            "processing_delay_ms": round(float(delay_parts.get("processing_ms", 0.0) or 0.0), 6),
            "handover_delay_component_ms": round(float(delay_parts.get("handover_delay_ms", 0.0) or 0.0), 6),
            # Canonical E2E fields all report the same observed completion
            # interval. Individual components remain available above.
            "e2e_delay_ms": observed_complete_latency_ms,
            "e2e_delay_incl_processing_ms": observed_complete_latency_ms,
            "e2e_delay_incl_tx_ms": e2e_incl_tx_ms,
            "e2e_delay_no_processing_ms": e2e_no_processing_ms,
            "predicted_completion_time_no_proc_ms": e2e_no_processing_ms,
            "latency_ms": observed_complete_latency_ms,
            "complete_latency_ms": complete_latency_ms,
            "observed_complete_latency_ms": observed_complete_latency_ms,
            "cpu_util_percent": util["cpu"],
            "mem_util_percent": util["memory"],
            "sto_util_percent": util["storage"],
            "selected_tier_cpu_util_percent": selected_util["cpu"],
            "selected_tier_mem_util_percent": selected_util["memory"],
            "selected_tier_congestion_percent": selected_util["congestion"],
            "selected_projected_utilization_percent": (
                future.get(selected_action, "") if selected_action is not None else ""
            ),
            "chosen_cpu_util": util["cpu"],
            "chosen_mem_util": util["memory"],
            "oracle_cpu_util": "",
            "oracle_mem_util": "",
            "Connected": int(bool(getattr(task, "wireless_ap", ""))),
            "reward": round(float(reward), 6),
            "epsilon": getattr(self.policy, "epsilon", ""),
            "cost": cost,
            "cost_accounting_perspective": cost_breakdown.get("accounting_perspective", ""),
            "cost_complete": cost_breakdown.get("complete", ""),
            "cost_incomplete_reason": getattr(task, "cost_incomplete_reason", "") if realized_basis else "",
            "tx_cost": cost_breakdown.get("tx_cost", cost_breakdown.get("communication_cost", "")),
            "proc_cost": cost_breakdown.get("proc_cost", cost_breakdown.get("processing_cost", "")),
            "processing_cost": cost_breakdown.get("proc_cost", cost_breakdown.get("processing_cost", "")),
            "communication_cost": cost_breakdown.get("tx_cost", cost_breakdown.get("communication_cost", "")),
            "cooling_cost": cost_breakdown.get("cooling_cost", ""),
            "operation_energy_cost": cost_breakdown.get("operation_energy_cost", ""),
            "propulsion_cost": cost_breakdown.get("propulsion_cost", ""),
            "device_wait_cost": cost_breakdown.get("device_wait_cost", ""),
            "operational_co2e_g": operational_co2e_g,
            "energy": energy,
            "tx_energy_proxy": energy_breakdown.get("device_tx_energy_j", energy),
            "proc_energy_j": energy_breakdown.get("infra_compute_energy_j", ""),
            "total_energy_j": energy_breakdown.get("total_energy_j", energy),
            "vehicle_battery_energy_j": getattr(task, "battery_remaining_j", ""),
            "device_tx_energy_j": energy_breakdown.get("device_tx_energy_j", ""),
            "device_rx_energy_j": energy_breakdown.get("device_rx_energy_j", ""),
            "device_local_compute_energy_j": energy_breakdown.get("device_local_compute_energy_j", ""),
            "device_wait_energy_j": energy_breakdown.get("device_wait_energy_j", ""),
            "device_propulsion_energy_j": energy_breakdown.get("device_propulsion_energy_j", ""),
            "device_paid_energy_j": getattr(task, "device_paid_energy_j", energy_breakdown.get("device_battery_energy_j", "")),
            "infra_compute_energy_j": energy_breakdown.get("infra_compute_energy_j", ""),
            "backhaul_energy_j": energy_breakdown.get("backhaul_energy_j", ""),
            "server_hover_energy_j": energy_breakdown.get("server_hover_energy_j", ""),
            "device_battery_energy_j": energy_breakdown.get("device_battery_energy_j", ""),
            "server_battery_energy_j": energy_breakdown.get("server_battery_energy_j", ""),
            "communication_energy_j": energy_breakdown.get("communication_energy_j", ""),
            "infrastructure_network_energy_j": energy_breakdown.get("infrastructure_network_energy_j", ""),
            "facility_overhead_energy_j": energy_breakdown.get("facility_overhead_energy_j", ""),
            # IT-only compute energy = infra_compute (IT+overhead) - facility overhead,
            # so compute_it + facility_overhead are DISJOINT and sum without double
            # counting. infra_compute_energy_j remains the (aggregate) total.
            "compute_it_energy_j": (
                float(energy_breakdown.get("infra_compute_energy_j", 0.0) or 0.0)
                - float(energy_breakdown.get("facility_overhead_energy_j", 0.0) or 0.0)
            ) if energy_breakdown.get("infra_compute_energy_j", "") != "" else "",
            "objective_energy_j": energy_breakdown.get("objective_energy_j", energy),
            "battery_draw_j": getattr(task, "battery_draw_j", ""),
            "battery_capacity_j": getattr(task, "battery_capacity_j", ""),
            "battery_remaining_j": getattr(task, "battery_remaining_j", ""),
            "device_battery_before_j": getattr(task, "device_battery_before_j", ""),
            "device_battery_after_j": getattr(task, "device_battery_after_j", getattr(task, "battery_remaining_j", "")),
            "device_battery_ratio": (
                max(0.0, min(1.0, float(getattr(task, "battery_remaining_j", 0.0) or 0.0) / float(getattr(task, "battery_capacity_j", 1.0) or 1.0)))
                if float(getattr(task, "battery_capacity_j", 0.0) or 0.0) > 0.0 else ""
            ),
            "server_battery_before_j": getattr(task, "server_battery_before_j", ""),
            "server_battery_after_j": getattr(task, "server_battery_after_j", ""),
            "server_battery_ratio": (context.get("server_battery_ratio", {}) or {}).get(action, "") if action is not None else "",
            "server_battery_projected_ratio": (context.get("projected_server_battery_ratio", {}) or {}).get(action, "") if action is not None else "",
            "local_cpu_capacity": getattr(task, "local_cpu_capacity", ""),
            "local_memory_capacity": getattr(task, "local_memory_capacity", ""),
            "local_power_w": getattr(task, "local_power_w", ""),
            "local_compute_used": int(action == ACTION_LOCAL),
            "handover_risk": getattr(task, "handover_risk", ""),
            "mobility_violation": mobility_violation,
            # These are LINK offered-load / network congestion. Compute-resource
            # pressure belongs in the selected-tier utilization/congestion fields
            # (selected_tier_congestion_percent), NOT here. Populated only from
            # this task's own observed shared-link reservations (queue delay,
            # transmission delay, on-wire bytes vs. declared link bit-rate --
            # see _offered_load_and_congestion_percent); blank whenever no hop
            # ever reserved service (e.g. pure local execution, or every
            # attempt ended in channel outage before a reservation formed),
            # never a fabricated zero.
            "offered_load_percent": "" if offered_load_percent is None else offered_load_percent,
            "network_congestion_percent": "" if network_congestion_percent is None else network_congestion_percent,
            "task_arrival_rate": "",
            "arrival_rate_rps": "",
            "rho": "",
            "load_factor": "",
            "throughput_mbps": (
                context.get("throughput_mbps", {}).get(action, "")
                if action is not None else ""
            ),
            "future_util": future_util,
            "u_ve": "",
            "u_local": float(future.get(ACTION_LOCAL, 0.0)) / 100.0,
            "u_mec": float(future.get(ACTION_MEC, 100.0)) / 100.0,
            "u_collab": float(future.get(ACTION_MEC_COLLAB, 100.0)) / 100.0,
            "u_met": float(future.get(ACTION_METRO, 100.0)) / 100.0,
            "u_cloud": float(future.get(ACTION_CLOUD, 100.0)) / 100.0,
            "u_uav": float(future.get(ACTION_UAV_BS, 100.0)) / 100.0,
            "reward_deadline_miss": deadline_missed,
            "reward_mobility_violation": mobility_violation,
            "reward_resource_overload": int(float(future_util or 0.0) >= 100.0) if future_util != "" else 0,
            "reward_collab_needed": int(action == ACTION_MEC_COLLAB),
            "reward_energy_norm": reward_info.get("energy_pen", ""),
            "reward_cost_norm": reward_info.get("cost_pen", ""),
            "reward_congestion_norm": util["congestion"] / 100.0,
            "reward_throughput_norm": "",
            "reward_capacity_norm": max(0.0, (100.0 - float(future_util or 100.0)) / 100.0) if future_util != "" else "",
            "reward_compute_capacity_norm": reward_info.get("compute_score", ""),
            "reward_shared_capacity_externality": reward_info.get("shared_capacity_externality_pen", ""),
            "reward_shared_capacity_externality_weight": reward_info.get("shared_capacity_externality_weight", ""),
            "current_util": reward_info.get("current_util", ""),
            "reward_cloud_overuse": int(action == ACTION_CLOUD and priority == 1),
            "reward_deadline_success": reward_info.get("deadline_success", ""),
            "reward_device_energy_penalty": reward_info.get("device_energy_pen", ""),
            "reward_server_battery_penalty": (
                max(0.0, 1.0 - float(reward_info["server_battery_score"]))
                if reward_info.get("server_battery_score", "") != ""
                else ""
            ),
            "reward_device_battery_score": reward_info.get("device_battery_score", ""),
            "reward_server_battery_score": reward_info.get("server_battery_score", ""),
            "reward_cost_penalty": reward_info.get("cost_pen", ""),
            "reward_capacity_adequacy": (
                (context.get("capacity_adequacy", {}) or {}).get(action, "")
                if action is not None
                else ""
            ),
            "reward_cost_normalizer": reward_info.get("cost_normalizer", ""),
            "reward_device_energy_normalizer_j": reward_info.get(
                "device_energy_normalizer_j", ""
            ),
            "battery_aware_action_flag": int(action == ACTION_LOCAL or action == ACTION_UAV_BS),
            "resource_balance_index": "",
            "utilization_stability_index": "",
            "decision_stability_index": "",
            "policy_consistency": "",
            "action_change_flag": action_change_flag,
            "action_change_rate": round(float(action_change_rate), 6),
            "priority_fairness_index": round(float(priority_fairness_index), 6),
            "energy_efficiency": round(float(energy_efficiency), 6) if energy_efficiency != "" else "",
            "cost_efficiency": round(float(cost_efficiency), 6) if cost_efficiency != "" else "",
            "useful_bits_per_j": round(float(energy_efficiency), 6) if realized_basis and energy_efficiency != "" else "",
            "unit_data_cost_usd_per_byte": unit_data_cost,
            "wire_efficiency": wire_efficiency,
            "resource_utilization_efficiency": "",
            "ve_future_cpu": "",
            "ve_future_mem": "",
            "local_future_cpu": future.get(ACTION_LOCAL, ""),
            "local_future_mem": future.get(ACTION_LOCAL, ""),
            "mec_future_cpu": future.get(ACTION_MEC, ""),
            "mec_future_mem": future.get(ACTION_MEC, ""),
            "collab_future_cpu": future.get(ACTION_MEC_COLLAB, ""),
            "collab_future_mem": future.get(ACTION_MEC_COLLAB, ""),
            "metro_future_cpu": future.get(ACTION_METRO, ""),
            "metro_future_mem": future.get(ACTION_METRO, ""),
            "cloud_future_cpu": future.get(ACTION_CLOUD, ""),
            "cloud_future_mem": future.get(ACTION_CLOUD, ""),
            "uav_future_cpu": future.get(ACTION_UAV_BS, ""),
            "uav_future_mem": future.get(ACTION_UAV_BS, ""),
            "ve_peak_future": "",
            "local_peak_future": future.get(ACTION_LOCAL, ""),
            "mec_peak_future": future.get(ACTION_MEC, ""),
            "collab_peak_future": future.get(ACTION_MEC_COLLAB, ""),
            "metro_peak_future": future.get(ACTION_METRO, ""),
            "cloud_peak_future": future.get(ACTION_CLOUD, ""),
            "uav_peak_future": future.get(ACTION_UAV_BS, ""),
            "local_compute_capable": getattr(task, "local_compute_capable", False),
            "mec_available": int(physical_mask[ACTION_MEC] == 1) if len(physical_mask) > ACTION_MEC else 0,
            "metro_available": int(physical_mask[ACTION_METRO] == 1) if len(physical_mask) > ACTION_METRO else 0,
            "cloud_available": int(physical_mask[ACTION_CLOUD] == 1) if len(physical_mask) > ACTION_CLOUD else 0,
            "uav_available": int(physical_mask[ACTION_UAV_BS] == 1) if len(physical_mask) > ACTION_UAV_BS else 0,
            "bs_ok": int(physical_mask[ACTION_MEC] == 1) if len(physical_mask) > ACTION_MEC else 0,
            "me_ok": int(physical_mask[ACTION_METRO] == 1) if len(physical_mask) > ACTION_METRO else 0,
            "success": int(status in ("assigned", "completed") and not deadline_missed),
            "dropped": int(status not in ("assigned", "completed")),
            "deadline_missed": deadline_missed,
            "status": status,
            "zone_id": getattr(task, "zone_id", ""),
            "sat_available": getattr(task, "sat_available", ""),
            "sat_elevation_deg": getattr(task, "sat_elevation_deg", ""),
            "sat_slant_range_m": getattr(task, "sat_slant_range_m", ""),
            "sat_access_node": getattr(task, "sat_access_node", ""),
            "sat_selected": int(action == ACTION_SAT) if action is not None else 0,
            "reward_forced_single_action": int(reward_info.get("forced_single_action", 0) or 0),
            "reward_deadline_penalty_exempt": int(reward_info.get("deadline_penalty_exempt", 0) or 0),
        }
        return row

    def save(self):
        model_dir = Path(getattr(self.config, "CHECKPOINT_ROOT", "trained_model")) / REWARD_SCHEMA_VERSION
        model_dir.mkdir(parents=True, exist_ok=True)
        self.policy.save(model_dir / f"{self.model_name}_comparison_policy.pt")

    def load(self):
        if self.model_name in ("random", "rule_based"):
            return
        path = Path(getattr(self.config, "CHECKPOINT_ROOT", "trained_model")) / REWARD_SCHEMA_VERSION / f"{self.model_name}_comparison_policy.pt"
        if not path.exists():
            raise FileNotFoundError(f"Train a fresh {REWARD_SCHEMA_VERSION} policy before evaluation: {path}")
        self.policy.load(path)

    def save_logs(self, output_dir=None, include_plots=None):
        out_dir = Path(output_dir) if output_dir is not None else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        # Plots are produced only by the campaign plot section (plots/), never
        # inside a model folder.
        include_plots = False

        if self.train_mode or bool(getattr(self.config, "WRITE_EVALUATION_TRAINING_METRICS", False)):
            TrainingReporter(out_dir).save_offloading_training(
                self.model_name,
                reward_log=self.reward_log,
                epsilon_log=self.epsilon_log,
                action_dist_log=self.action_dist_log,
                loss_log=self.loss_log,
                include_plots=include_plots,
                visualizer=self.visualizer,
            )
    def _write_diagnostics(self, output_dir):
        # Per-run policy diagnostics. Best-effort; never breaks a run.
        import json
        try:
            out_dir = Path(output_dir) if output_dir is not None else self.output_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            policy_diag = {}
            getter = getattr(self.policy, "get_diagnostics", None)
            if callable(getter):
                policy_diag = getter() or {}
            payload = {
                "model": self.model_name,
                "phase": self.phase,
                "train_mode": bool(self.train_mode),
                "seed": int(self._seed),
                "decisions": self._diag_decisions,
                "policy_diagnostics": policy_diag,
            }
            with open(out_dir / f"diagnostics_{self.model_name}.json", "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except Exception as exc:
            print(f"[DIAG] Failed to write diagnostics for {self.model_name}: {exc}", flush=True)

    def finalize(self):
        try:
            if self.train_mode:
                self.save()
            self.save_logs(output_dir=self.output_dir, include_plots=False)
            if bool(getattr(self.config, "WRITE_DIAGNOSTICS_JSON", False)):
                self._write_diagnostics(self.output_dir)
        finally:
            self._close_csv()


def create_comparison_scheduler(model_name, config=default_config, output_dir=None, train_mode=False, seed=None, logger=print):
    scheduler_type = str(model_name).strip().lower()
    if seed is not None:
        set_global_seed(seed)
    try:
        import config as cfg
        cfg.SCHEDULER_TYPE = scheduler_type
    except Exception:
        pass
    scheduler = ComparisonRunner(
        scheduler_type,
        config=config,
        output_dir=output_dir,
        seed=seed,
    )
    if not bool(train_mode) and hasattr(scheduler, "load"):
        scheduler.load()
    scheduler.set_train_mode(bool(train_mode))
    return scheduler
