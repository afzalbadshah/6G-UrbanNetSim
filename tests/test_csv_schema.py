"""Task 52 schema tests: canonical resource, consumer-price, disjoint-energy, and
perspective-safe CSV reporting, plus full-precision server_results.

`test_schema_and_formats` is a fast static check. `test_live_csv_values` runs one
tiny deterministic evaluation and checks the new columns are populated and
internally consistent against the realized execution.
"""

import csv
import os
import tempfile
import unittest
from unittest.mock import patch

from scheduler.common.comparison_runner import ComparisonRunner
from core.reporter import TASK_FIELDS

NEW_DETAIL_COLUMNS = (
    "compute_work_cycles", "allocated_compute_rate_cycles_per_s",
    "memory_working_set_bytes", "storage_footprint_bytes", "input_payload_bytes",
    "resource_conversion_profile_id", "local_compute_capacity_cycles_per_s",
    "compute_it_energy_j",
    "legacy_objective_delay_ms", "legacy_complete_latency_ms",
    "sla_profile_id", "sla_generation", "radio_latency_target_ms",
    "application_deadline_ms", "sla_maturity", "sla_source_id",
    "reward_device_battery_weight", "reward_server_battery_weight",
    "reward_compute_weight", "reward_mobility_weight",
    "reward_device_battery_score", "reward_server_battery_score",
    "reward_cost_penalty", "reward_capacity_adequacy",
    "reward_cost_normalizer", "reward_device_energy_normalizer_j",
    "reward_overutilization_penalty", "reward_overutilization_weight",
    "selected_projected_utilization_percent",
)


class TestCsvSchemaStatic(unittest.TestCase):
    def test_schema_and_formats(self):
        # New detailed-CSV columns exist, and existing columns are preserved.
        for col in NEW_DETAIL_COLUMNS:
            self.assertIn(col, ComparisonRunner.FIELDNAMES, f"missing detail column {col}")
        for kept in ("cost", "cost_accounting_perspective", "on_wire_bytes",
                     "infra_compute_energy_j", "facility_overhead_energy_j"):
            self.assertIn(kept, ComparisonRunner.FIELDNAMES)
        # server_results monetary/energy fields keep full precision (Task 52).
        fmt = {key: f for key, _label, f in TASK_FIELDS}
        for key in ("tx_cost", "proc_cost", "energy"):
            self.assertEqual(fmt[key], "{:.10g}", f"{key} must retain full precision")


class TestLiveCsvValues(unittest.TestCase):
    def test_live_csv_values(self):
        import config as cfg
        from scheduler.common import common_config
        from scheduler.common.comparison_runner import create_comparison_scheduler
        from scheduler.common.policy_utils import set_global_seed
        from core.simulator import Simulator

        tmp = tempfile.mkdtemp(prefix="csvschema_")
        cfg.apply_workload_profile(train_mode=False,
                                   overrides={"start_devices": 1, "max_devices": 4,
                                              "increment": 1, "rounds": 2})
        set_global_seed(42)
        sched = create_comparison_scheduler("random", config=common_config,
                                            output_dir=tmp, train_mode=False, seed=42,
                                            logger=lambda *a, **k: None)
        sim = Simulator(
            scheduler=sched,
            output_dir=tmp,
            workload=cfg.WORKLOAD,
            deployment_profile="experimental_6g_sla_16",
        )
        sim.run(train_mode=False, progress_callback=None)

        results = os.path.join(tmp, "offloading", "random", "results.csv")
        with open(results, newline="", encoding="utf-8") as result_file:
            rows = list(csv.DictReader(result_file, delimiter=";"))
        self.assertTrue(rows, "results.csv has no rows")
        detailed = os.path.join(
            tmp, "offloading", "random", sched.offloading_filename
        )
        with open(detailed, newline="", encoding="utf-8") as detailed_file:
            detailed_rows = list(csv.DictReader(detailed_file, delimiter=";"))
        self.assertEqual(rows, detailed_rows)
        header = rows[0].keys()
        for col in NEW_DETAIL_COLUMNS:
            self.assertIn(col, header)

        for r in rows:
            # Canonical resource input is always present.
            self.assertNotEqual(r["compute_work_cycles"], "")
            # Resolved SLA identity and provenance are present in every result.
            for field in (
                "sla_profile_id", "sla_generation",
                "radio_latency_target_ms", "application_deadline_ms",
                "sla_maturity", "sla_source_id",
            ):
                self.assertNotEqual(r[field], "", field)
            self.assertEqual(r["sla_generation"], "6G")
            self.assertEqual(
                float(r["deadline_ms"]),
                float(r["application_deadline_ms"]),
            )
            # Task 20: link offered-load/congestion are real, observed shared
            # -link reservation measurements (queue delay, transmission
            # delay, on-wire bytes vs. declared link bit-rate), never a
            # fabricated number. Compute-resource pressure stays in the
            # selected-tier fields, not here. Blank whenever this task never
            # reserved a shared link (e.g. pure local execution) -- when
            # present, values must be well-formed and congestion bounded.
            if r["local_compute_used"] == "1":
                self.assertEqual(r["offered_load_percent"], "")
                self.assertEqual(r["network_congestion_percent"], "")
            if r["network_congestion_percent"] != "":
                self.assertGreaterEqual(float(r["network_congestion_percent"]), 0.0)
                self.assertLessEqual(float(r["network_congestion_percent"]), 100.0)
            if r["offered_load_percent"] != "":
                self.assertGreaterEqual(float(r["offered_load_percent"]), 0.0)
            # Disjoint energy: compute_it + facility_overhead == infra_compute total.
            it, fo, agg = r["compute_it_energy_j"], r["facility_overhead_energy_j"], r["infra_compute_energy_j"]
            if it != "" and fo != "" and agg != "":
                self.assertAlmostEqual(float(it) + float(fo), float(agg), places=9)
        self.assertTrue(
            any(r["network_congestion_percent"] != "" for r in rows),
            "expected at least one realized network hop to populate "
            "network_congestion_percent in this run",
        )

    def test_late_task_is_completed_and_fully_reported(self):
        """An application SLA miss must not erase the physical observation."""

        import config as cfg
        import workload.generator as workload_generator
        from core.simulator import Simulator
        from scheduler.common.common_action import ACTION_CLOUD
        from scheduler.common.comparison_runner import create_comparison_scheduler

        workload = {
            "start_devices": 1,
            "max_devices": 1,
            "increment": 1,
            "start_data_per_device_kb": 1.0,
            "max_data_per_device_kb": 1.0,
            "data_increment_kb": 0.0,
            "data_per_device_kb": 1.0,
            "cpu_per_kb": 4.0,
            "memory_per_kb": 0.2,
            "rounds": 1,
            "profile": "late_task_reporting_test",
        }
        configured_get_sla = workload_generator.get_sla

        def tiny_application_sla(service_class, generation):
            sla = dict(configured_get_sla(service_class, generation))
            sla["application_deadline_ms"] = 0.001
            return sla

        with tempfile.TemporaryDirectory(prefix="late_task_csv_") as tmp:
            scheduler = create_comparison_scheduler(
                "random",
                output_dir=tmp,
                train_mode=False,
                seed=17,
                logger=lambda *_: None,
            )
            scheduler.policy.select_action = (
                lambda _state, mask: ACTION_CLOUD
                if int(mask[ACTION_CLOUD]) == 1
                else next(index for index, enabled in enumerate(mask) if enabled)
            )
            with patch.object(
                workload_generator, "get_sla", side_effect=tiny_application_sla
            ), patch.object(cfg, "OUTAGE_PROBABILITY", 0.0), patch.object(
                cfg, "HANDOVER_FAILURE_PROB", 0.0
            ):
                simulator = Simulator(
                    scheduler,
                    output_dir=tmp,
                    workload=workload,
                    deployment_profile="legacy_sparse",
                )
                simulator.mobility.handover_scheduler.load = lambda: None
                device = simulator.mobility.entities[0]
                access = next(
                    server for server in simulator.servers if server.name == "MEC_1"
                )
                device.mobility = None
                device.x, device.y = access.x, access.y
                device.position = [device.x, device.y]
                device.attached_server = access
                simulator.run(train_mode=False)

            results = os.path.join(tmp, "offloading", "random", "results.csv")
            with open(results, newline="", encoding="utf-8") as result_file:
                row = next(csv.DictReader(result_file, delimiter=";"))

        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["network_status"], "delivered")
        self.assertEqual(row["deadline_missed"], "1")
        self.assertEqual(row["dropped"], "0")
        self.assertGreater(float(row["delay_ms"]), float(row["deadline_ms"]))
        self.assertGreater(float(row["network_delay_ms"]), 0.0)
        self.assertGreater(float(row["processing_delay_ms"]), 0.0)
        self.assertGreater(int(row["on_wire_bytes"]), int(row["payload_bytes"]))
        self.assertGreater(float(row["total_energy_j"]), 0.0)
        self.assertGreater(float(row["compute_it_energy_j"]), 0.0)
        self.assertGreater(float(row["cost"]), 0.0)
        self.assertEqual(row["cost_complete"], "True")


class TestOfferedLoadCongestionDerivation(unittest.TestCase):
    """Task 20 hand-check: pin the offered-load/congestion formulas against
    real, observed ``LinkReservation`` timing -- not a mock.

    Two tasks are sent to the same MEC access link back-to-back at the same
    simulated instant so the second one's packet genuinely queues behind the
    first's on the shared FIFO resource (``network/links.py``). The expected
    percentages are then recomputed independently, directly from that
    reservation's own timestamps and the link's declared nominal bit-rate
    (``config/radio.get_link_profile``), and compared against
    ``ComparisonRunner._offered_load_and_congestion_percent``.
    """

    @staticmethod
    def _task(task_id, data_size_kb):
        from types import SimpleNamespace

        return SimpleNamespace(
            id=task_id,
            device_id=task_id,
            entity_id=task_id,
            data_size_kb=data_size_kb,
            cpu_demand=1.0,
            memory_demand=0.0,
            storage_demand=0.1,
            deadline_ms=5000.0,
            service_class="URLLC",
            application_type="safety_upload",
            tx_power_w=1.3,
            interference_dbm="",
            handover_delay_ms=0.0,
            handover_interruption_ms=0.0,
            coverage_outage_flag=0,
            coverage_outage_duration_ms=0.0,
        )

    def test_congestion_and_offered_load_match_observed_reservation_timing(self):
        from types import SimpleNamespace

        from config.radio import get_link_profile
        from config.topology import get_topology
        from core.task_execution import TaskExecutionRuntime
        from metrics.kpi import bounded_ratio, offered_load_ratio
        from network import PhysicalTopology, Position3D

        topology = PhysicalTopology.from_mapping(
            "smartcity_default", get_topology("smartcity_default")
        )
        runtime = TaskExecutionRuntime(topology, experiment_seed=17)
        mec = SimpleNamespace(name="MEC_1", tier="MEC")
        position = Position3D(5000.0, 7400.0, 1.5)

        # A first, larger transfer occupies MEC_1's access SharedLink so the
        # second task's packet observably queues behind it: a real FIFO
        # backlog, not a fabricated one.
        first = runtime.execute(
            self._task("T1", data_size_kb=50.0),
            mec,
            current_time_ms=0.0,
            processing_duration_ms=1.0,
            utilization_before_ratio=0.0,
            utilization_after_ratio=0.1,
            source_position=position,
            access_node_id="MEC_1",
        )
        second = runtime.execute(
            self._task("T2", data_size_kb=0.1),
            mec,
            current_time_ms=0.0,
            processing_duration_ms=1.0,
            utilization_before_ratio=0.1,
            utilization_after_ratio=0.2,
            source_position=position,
            access_node_id="MEC_1",
        )
        self.assertTrue(first.application_completed)
        self.assertTrue(second.application_completed)

        network_execution = second.network_execution
        self.assertIsNotNone(network_execution)
        route = network_execution.transmission.route
        # MEC_1 is both the access node and the destination, so the route is
        # exactly the one radio-access hop.
        self.assertEqual(len(route.hops), 1)

        # Independently walk every reservation this task's own packets made
        # (mirrors, but does not call, ComparisonRunner's traversal) and pick
        # the one with the highest observed queue-delay share, exactly as
        # specified: "the single hop with the highest observed
        # congestion_ratio".
        best = None
        for packet_route in network_execution.transmission.packet_routes:
            for index, hop_result in enumerate(packet_route.hop_transmissions):
                resolved_hop = route.hops[index]
                nominal_bit_rate_bps = float(
                    get_link_profile(resolved_hop.link.link_profile_id)[
                        "nominal_bit_rate_bps"
                    ]
                )
                for attempt in hop_result.attempts:
                    reservation = attempt.reservation
                    if reservation is None:
                        continue
                    congestion_ratio = bounded_ratio(
                        reservation.queue_delay_s,
                        reservation.queue_delay_s + reservation.transmission_delay_s,
                    )
                    offered_ratio = (
                        offered_load_ratio(
                            reservation.on_wire_bytes * 8.0,
                            nominal_bit_rate_bps,
                            reservation.transmission_delay_s,
                        )
                        if reservation.transmission_delay_s > 0.0
                        else 0.0
                    )
                    if best is None or congestion_ratio > best[0]:
                        best = (congestion_ratio, offered_ratio)

        self.assertIsNotNone(best, "expected at least one real reservation")
        expected_congestion_ratio, expected_offered_ratio = best
        self.assertGreater(
            expected_congestion_ratio,
            0.0,
            "test setup must produce a real, observed queueing backlog",
        )

        offered_percent, congestion_percent = (
            ComparisonRunner._offered_load_and_congestion_percent(network_execution)
        )
        self.assertEqual(congestion_percent, round(expected_congestion_ratio * 100.0, 4))
        self.assertEqual(offered_percent, round(expected_offered_ratio * 100.0, 4))
        self.assertGreaterEqual(congestion_percent, 0.0)
        self.assertLessEqual(congestion_percent, 100.0)
        self.assertGreater(offered_percent, 0.0)

    def test_no_reservation_reports_blank_not_zero(self):
        # A None network_execution (pure local task -- see
        # RealizedTaskExecution.network_execution) must report blank, never
        # a fabricated 0.
        offered, congestion = ComparisonRunner._offered_load_and_congestion_percent(None)
        self.assertIsNone(offered)
        self.assertIsNone(congestion)


if __name__ == "__main__":
    unittest.main()
