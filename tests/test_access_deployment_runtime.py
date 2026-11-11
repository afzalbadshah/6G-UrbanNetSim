"""Live integration test for separate access and compute-site counts."""

import csv
import tempfile
import unittest
from pathlib import Path

from config import SERVER_CONFIG
from core.simulator import Simulator
from scheduler.common.common_action import ACTION_CLOUD, ACTION_MEC
from scheduler.common.comparison_runner import create_comparison_scheduler


class AccessDeploymentRuntimeTests(unittest.TestCase):
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
            "profile": "separate_access_integration",
        }

    def test_access_only_bs_reaches_cloud_without_changing_mec_count(self):
        with tempfile.TemporaryDirectory(prefix="access_deployment_") as tmp:
            scheduler = create_comparison_scheduler(
                "random",
                output_dir=tmp,
                train_mode=False,
                seed=23,
                logger=lambda *_: None,
            )
            scheduler.policy.select_action = (
                lambda _state, mask: ACTION_CLOUD
                if int(mask[ACTION_CLOUD]) == 1
                else next(index for index, enabled in enumerate(mask) if enabled)
            )
            simulator = Simulator(
                scheduler,
                output_dir=tmp,
                workload=self._one_task_workload(),
                deployment_profile="fr1_density_sweep_16",
            )
            mec_count = int(SERVER_CONFIG["MEC"]["num_datacenters"])
            # MEC sites are now upgraded ground BS positions (see
            # core.deployment_builder), so the access-only layer is the
            # remaining ground BS count, not the full deployment profile count.
            self.assertEqual(len(simulator.access_points), 16 - mec_count)
            self.assertEqual(
                sum(server.tier == "MEC" for server in simulator.servers),
                mec_count,
            )

            # The existing handover checkpoint has the legacy six-point input
            # dimension.  This test exercises deployment wiring, so suppress its
            # optional load and place the device exactly at a new BS.
            simulator.mobility.handover_scheduler.load = lambda: None
            access_point = simulator.access_points[0]
            device = simulator.mobility.entities[0]
            device.mobility = None
            device.x, device.y = access_point.x, access_point.y
            device.position = [device.x, device.y]
            device.attached_server = access_point

            simulator.run(train_mode=False, progress_callback=None)
            self.assertEqual(len(simulator.realized_execution_by_task_id), 1)
            execution = next(iter(simulator.realized_execution_by_task_id.values()))
            self.assertTrue(execution.application_completed)
            self.assertEqual(execution.network_status.value, "delivered")
            self.assertGreater(execution.packet_count, 0)
            self.assertGreater(execution.packet_hop_count, 0)

            result_path = Path(tmp) / "offloading" / "random" / "results.csv"
            with result_path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle, delimiter=";"))
            self.assertTrue(row["wireless_ap"].startswith("BS_"))
            self.assertEqual(row["processing_server"], "Cloud_1")
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["network_status"], "delivered", row)

    def test_deployed_ground_access_layer_is_used_as_wireless_candidate(self):
        workload = self._one_task_workload()
        workload["start_devices"] = 1
        workload["max_devices"] = 1
        with tempfile.TemporaryDirectory(prefix="access_candidate_") as tmp:
            scheduler = create_comparison_scheduler(
                "random",
                output_dir=tmp,
                train_mode=False,
                seed=31,
                logger=lambda *_: None,
            )
            simulator = Simulator(
                scheduler,
                output_dir=tmp,
                workload=workload,
                deployment_profile="fr1_density_sweep_16",
            )
            simulator.mobility.handover_scheduler.load = lambda: None

            # The fixed actor inventory includes ground BS and UAV-BS, but a
            # usable ground BS must be selected before UAV fallback.
            #
            # With this deployment profile's ground BS layer in place, MEC's
            # site count is filled by UPGRADING that many ground BS positions
            # to compute (see core.deployment_builder.build_deployed_topology),
            # and an upgraded MEC's topology node carries
            # provides_radio_access=True. Such a promoted MEC is a genuine
            # radio endpoint exactly like an is_access_only=True AccessPoint,
            # so it now legitimately appears in the candidate list too --
            # this is the intended fix for the bug where a promoted MEC's
            # radio access was silently dropped from the candidate list
            # (see mobility.handover_scheduler.candidate_access_points).
            candidates = simulator.mobility._attachable_servers()
            self.assertTrue(candidates)
            self.assertTrue(
                all(
                    bool(getattr(server, "is_access_only", False))
                    or bool(getattr(server, "provides_radio_access", False))
                    or str(getattr(server, "tier", "")).upper() == "UAV_BS"
                    for server in candidates
                )
            )
            device = simulator.mobility.entities[0]
            ground_bs = simulator.access_points[0]
            uav = next(server for server in simulator.servers if server.tier == "UAV_BS")
            device.mobility = None
            device.x, device.y = ground_bs.x, ground_bs.y
            device.position = [device.x, device.y]
            device.attached_server = uav
            simulator.run(train_mode=False)
            attached = device.attached_server
            # Accepts either a plain access-only BS or a promoted MEC -- this
            # test's own setup above already establishes both are equally
            # legitimate ground candidates (provides_radio_access is "a
            # genuine radio endpoint exactly like an is_access_only=True
            # AccessPoint"). Ground eligibility is also now anchor-aware
            # (dual-connectivity coverage fallback, mobility/handover_scheduler.py),
            # so a ground candidate reachable only via its anchor can
            # legitimately be selected over the exact-colocated one; the
            # invariant this test protects -- ground before UAV fallback --
            # still holds, just not literally down to which ground site.
            self.assertTrue(
                bool(getattr(attached, "is_access_only", False))
                or bool(getattr(attached, "provides_radio_access", False))
            )

    def test_uav_fallback_can_route_a_task_to_near_edge_mec(self):
        with tempfile.TemporaryDirectory(prefix="uav_to_mec_") as tmp:
            scheduler = create_comparison_scheduler(
                "random",
                output_dir=tmp,
                train_mode=False,
                seed=41,
                logger=lambda *_: None,
            )
            scheduler.policy.select_action = (
                lambda _state, mask: ACTION_MEC
                if int(mask[ACTION_MEC]) == 1
                else next(index for index, enabled in enumerate(mask) if enabled)
            )
            simulator = Simulator(
                scheduler,
                output_dir=tmp,
                workload=self._one_task_workload(),
                deployment_profile="legacy_sparse",
            )
            simulator.mobility.handover_scheduler.load = lambda: None

            device = simulator.mobility.entities[0]
            device.mobility = None
            uav = next(server for server in simulator.servers if server.tier == "UAV_BS")
            uav.mobility = None
            device.x, device.y = uav.x, uav.y
            device.position = [device.x, device.y]
            device.attached_server = uav
            # Keep the MEC operational for processing, but put its radio far
            # outside receiver sensitivity so UAV is the genuine fallback.
            for server in simulator.servers:
                if server.tier == "MEC":
                    server.x, server.y = 1_000_000.0, 1_000_000.0

            simulator.run(train_mode=False)
            result_path = Path(tmp) / "offloading" / "random" / "results.csv"
            with result_path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle, delimiter=";"))
            self.assertTrue(row["wireless_ap"].startswith("UAV_BS_"))
            self.assertTrue(row["processing_server"].startswith("MEC_"))
            self.assertEqual(row["physical_mask"].split(",")[ACTION_MEC], "1")
            self.assertGreater(int(row["packet_hop_count"]), 0)
            self.assertNotIn("route", row["network_failure_reason"].lower())

    def test_legacy_simulator_can_activate_interactive_access_layer_before_run(self):
        # This test starts from the empty (legacy) layout and then activates the
        # interactive access layer, so force legacy_sparse rather than relying on
        # the owner-set default deployment_profile (config.py §1), which may be a
        # coverage profile.
        from config import SIMULATION
        previous_profile = SIMULATION["deployment_profile"]
        SIMULATION["deployment_profile"] = "legacy_sparse"
        try:
            with tempfile.TemporaryDirectory(prefix="access_activation_") as tmp:
                scheduler = create_comparison_scheduler(
                    "random",
                    output_dir=tmp,
                    train_mode=False,
                    seed=37,
                    logger=lambda *_: None,
                )
                simulator = Simulator(
                    scheduler,
                    output_dir=tmp,
                    workload=self._one_task_workload(),
                )
                self.assertEqual(simulator.access_points, [])
                simulator.mobility.handover_scheduler.load = lambda: None
                simulator.activate_deployment_profile("fr1_density_sweep_16")
                mec_count = int(SERVER_CONFIG["MEC"]["num_datacenters"])
                self.assertEqual(len(simulator.access_points), 16 - mec_count)
                self.assertEqual(
                    sum(server.tier == "MEC" for server in simulator.servers),
                    mec_count,
                )
                simulator.run(train_mode=False)
        finally:
            SIMULATION["deployment_profile"] = previous_profile


if __name__ == "__main__":
    unittest.main()
