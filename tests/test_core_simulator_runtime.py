import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config as cfg
from core.simulator import Simulator
from scheduler.common.common_action import ACTION_CLOUD
from scheduler.common.comparison_runner import create_comparison_scheduler


class LivePhysicalExecutionIntegrationTests(unittest.TestCase):
    @staticmethod
    def _one_task_workload():
        return {
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
            "profile": "integration_smoke",
        }

    @staticmethod
    def _pin_device_attachment(simulator, device, access):
        """Force ``device`` to stay attached to ``access`` for the whole run.

        ``mobility.update_all()`` re-decides attachment via the handover
        scheduler every round (its checkpoint never loads -- shape mismatch --
        so its choice is arbitrary), which silently overwrites a one-time
        ``device.attached_server`` assignment before task generation reads it.
        Re-pinning after each update makes a test actually exercise the link
        it sets up, regardless of the handover scheduler's unrelated
        checkpoint problem.
        """

        device.mobility = None
        device.x, device.y = access.x, access.y
        device.position = [device.x, device.y]
        device.attached_server = access
        original_update_all = simulator.mobility.update_all

        def _pinned_update_all(*args, **kwargs):
            result = original_update_all(*args, **kwargs)
            device.attached_server = access
            device.x, device.y = access.x, access.y
            device.position = [device.x, device.y]
            return result

        simulator.mobility.update_all = _pinned_update_all

    def test_cloud_decision_emits_realized_packet_measurements(self):
        workload = self._one_task_workload()

        with tempfile.TemporaryDirectory() as tmp:
            scheduler = create_comparison_scheduler(
                "random",
                output_dir=tmp,
                train_mode=False,
                seed=17,
                logger=lambda _message: None,
            )
            # Make the integration gate deterministic and exercise the longest
            # declared route: device -> serving access node -> Metro -> Cloud.
            scheduler.policy.select_action = (
                lambda _state, mask: ACTION_CLOUD
                if int(mask[ACTION_CLOUD]) == 1
                else next(index for index, enabled in enumerate(mask) if enabled)
            )
            simulator = Simulator(scheduler, output_dir=tmp, workload=workload)
            access = next(server for server in simulator.servers if server.name == "MEC_1")
            device = simulator.mobility.entities[0]
            self._pin_device_attachment(simulator, device, access)

            with patch.object(cfg, "OUTAGE_PROBABILITY", 0.0), patch.object(
                cfg, "HANDOVER_FAILURE_PROB", 0.0
            ):
                simulator.run(train_mode=False)

            self.assertEqual(simulator.current_time, 5000.0)
            self.assertEqual(len(simulator.realized_execution_by_task_id), 1)
            execution = next(iter(simulator.realized_execution_by_task_id.values()))
            self.assertGreater(execution.payload_bytes, 0)
            self.assertGreater(execution.packet_count, 0)
            self.assertGreater(execution.packet_hop_count, 0)
            self.assertGreater(execution.delay.network_elapsed_ms, 0.0)
            self.assertGreater(execution.energy.total_energy_j, 0.0)
            self.assertTrue(execution.operator_cost.complete)
            self.assertGreaterEqual(execution.operator_cost.total, 0.0)

            results_path = Path(tmp) / "offloading" / "random" / "results.csv"
            with results_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter=";"))

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["measurement_basis"], "realized_execution")
            self.assertEqual(row["allocated_tier"], "CLOUD")
            self.assertEqual(int(row["payload_bytes"]), execution.payload_bytes)
            self.assertGreater(int(row["on_wire_bytes"]), execution.payload_bytes)
            self.assertGreater(int(row["packet_hop_count"]), 0)
            self.assertGreater(float(row["network_delay_ms"]), 0.0)
            self.assertGreater(float(row["total_energy_j"]), 0.0)
            self.assertEqual(row["cost_accounting_perspective"], "operator_cost")
            self.assertEqual(row["cost_complete"], "True")

    def test_all_scheduler_families_run_with_canonical_resource_wiring(self):
        model_names = (
            "random",
            "rule_based",
            "q_learning",
            "dqn",
            "double_dqn",
            "ppo",
            "a2c",
        )
        with tempfile.TemporaryDirectory() as tmp:
            for model_name in model_names:
                with self.subTest(model=model_name):
                    output_dir = str(Path(tmp) / model_name)
                    scheduler = create_comparison_scheduler(
                        model_name,
                        output_dir=output_dir,
                        train_mode=False,
                        seed=23,
                        logger=lambda _message: None,
                    )
                    scheduler.policy.select_action = (
                        lambda _state, mask: ACTION_CLOUD
                        if int(mask[ACTION_CLOUD]) == 1
                        else next(
                            index for index, enabled in enumerate(mask) if enabled
                        )
                    )
                    simulator = Simulator(
                        scheduler,
                        output_dir=output_dir,
                        workload=self._one_task_workload(),
                    )
                    access = next(
                        server for server in simulator.servers if server.name == "MEC_1"
                    )
                    device = simulator.mobility.entities[0]
                    self._pin_device_attachment(simulator, device, access)
                    with patch.object(cfg, "OUTAGE_PROBABILITY", 0.0), patch.object(
                        cfg, "HANDOVER_FAILURE_PROB", 0.0
                    ):
                        simulator.run(train_mode=False)

                    self.assertEqual(len(simulator.realized_execution_by_task_id), 1)
                    execution = next(
                        iter(simulator.realized_execution_by_task_id.values())
                    )
                    self.assertTrue(execution.application_completed)
                    self.assertTrue(
                        all(
                            hasattr(server, "compute_rate_cycles_per_s")
                            for server in simulator.servers
                        )
                    )


if __name__ == "__main__":
    unittest.main()
