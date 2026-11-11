import unittest
from types import SimpleNamespace

from config.servers import SERVER_CONFIG
from config.simulation import SIMULATION
from config.topology import get_deployment_profile, get_topology
from core.deployment_builder import build_deployed_topology
from core.task_execution import TaskExecutionRuntime
from network import DeliveryStatus, PhysicalTopology, Position3D


def task(**overrides):
    values = {
        "id": "T0",
        "device_id": 0,
        "entity_id": 0,
        "data_size_kb": 0.1,
        "cpu_demand": 1.0,
        "memory_demand": 0.0,
        "storage_demand": 0.1,
        "deadline_ms": 1000.0,
        "service_class": "URLLC",
        "application_type": "safety_upload",
        "tx_power_w": 1.3,
        "idle_power_w": 0.02,
        "local_power_w": 2.0,
        "interference_dbm": "",
        "handover_delay_ms": 0.0,
        "handover_interruption_ms": 0.0,
        "coverage_outage_flag": 0,
        "coverage_outage_duration_ms": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def server(name, tier):
    return SimpleNamespace(name=name, tier=tier)


class TaskExecutionRuntimeTests(unittest.TestCase):
    def setUp(self):
        topology = PhysicalTopology.from_mapping(
            "smartcity_default", get_topology("smartcity_default")
        )
        self.runtime = TaskExecutionRuntime(topology, experiment_seed=17)

    def test_remote_execution_is_packetized_and_measured(self):
        result = self.runtime.execute(
            task(),
            server("MEC_1", "MEC"),
            current_time_ms=0.0,
            processing_duration_ms=10.0,
            utilization_before_ratio=0.0,
            utilization_after_ratio=0.1,
            source_position=Position3D(5000.0, 7400.0, 1.5),
            access_node_id="MEC_1",
        )

        self.assertTrue(result.application_completed)
        self.assertEqual(result.network_status, DeliveryStatus.DELIVERED)
        self.assertEqual(result.payload_bytes, 100)
        self.assertEqual(result.packet_count, 1)
        self.assertGreater(result.on_wire_bytes, result.payload_bytes)
        self.assertGreater(result.delay.wireless_transmission_ms, 0.0)
        self.assertGreater(result.delay.wireless_propagation_ms, 0.0)
        self.assertEqual(result.delay.wired_transmission_ms, 0.0)
        self.assertAlmostEqual(
            result.delay.total_ms,
            result.delay.pre_access_wait_ms
            + result.delay.network_elapsed_ms
            + result.delay.processing_ms,
        )
        self.assertAlmostEqual(
            result.delay.legacy_objective_delay_ms,
            result.delay.propagation_ms
            + result.delay.handover_ms,
        )
        self.assertTrue(result.channel_results)
        self.assertGreater(result.energy.device_tx_energy_j, 0.0)
        access_active_s = sum(
            attempt.reservation.transmission_delay_s
            for packet_route in result.network_execution.transmission.packet_routes
            for attempt in packet_route.hop_transmissions[0].attempts
            if attempt.reservation is not None
        )
        self.assertAlmostEqual(
            result.energy.device_tx_energy_j,
            1.3 * access_active_s,
        )
        device_wait_s = (
            result.delay.pre_access_wait_ms
            + result.delay.network_elapsed_ms
            + result.delay.processing_ms
        ) / 1000.0 - access_active_s
        self.assertAlmostEqual(
            result.energy.device_wait_energy_j,
            0.02 * device_wait_s,
        )
        self.assertGreater(result.energy.infrastructure_network_energy_j, 0.0)
        self.assertGreater(result.energy.compute_it_energy_j, 0.0)
        self.assertTrue(result.operator_cost.complete)

    def test_local_execution_bypasses_network_and_has_zero_service_price(self):
        result = self.runtime.execute(
            task(),
            server("LocalDevice_0", "LOCAL_DEVICE"),
            current_time_ms=5.0,
            processing_duration_ms=8.0,
            utilization_before_ratio=0.0,
            utilization_after_ratio=0.5,
            source_position=Position3D(10.0, 10.0, 1.5),
            access_node_id=None,
        )

        self.assertTrue(result.application_completed)
        self.assertIsNone(result.network_status)
        self.assertEqual(result.packet_count, 0)
        self.assertEqual(result.on_wire_bytes, 0)
        self.assertEqual(result.delay.handover_ms, 0.0)
        self.assertEqual(result.delay.coverage_outage_ms, 0.0)
        self.assertEqual(result.delay.network_elapsed_ms, 0.0)
        self.assertGreater(result.energy.compute_it_energy_j, 0.0)
        self.assertAlmostEqual(result.energy.compute_it_energy_j, 2.0 * 0.008)
        self.assertAlmostEqual(
            result.energy.device_battery_energy_j,
            result.energy.compute_it_energy_j,
        )
        self.assertEqual(result.operator_cost_total, 0.0)

    def test_access_outage_misses_sla_but_task_still_completes(self):
        result = self.runtime.execute(
            task(
                deadline_ms=100.0,
                coverage_outage_flag=1,
                coverage_outage_duration_ms=5000.0,
            ),
            server("MEC_1", "MEC"),
            current_time_ms=20.0,
            processing_duration_ms=10.0,
            utilization_before_ratio=0.0,
            utilization_after_ratio=0.1,
            source_position=Position3D(5000.0, 7400.0, 1.5),
            access_node_id="MEC_1",
        )

        self.assertTrue(result.application_completed)
        self.assertTrue(result.deadline_missed)
        self.assertEqual(result.network_status, DeliveryStatus.DELIVERED)
        self.assertGreater(result.finished_at_ms, 5020.0)
        self.assertGreater(result.packet_count, 0)
        self.assertGreater(result.on_wire_bytes, result.payload_bytes)
        self.assertEqual(result.delay.handover_ms, 0.0)
        self.assertEqual(result.delay.coverage_outage_ms, 5000.0)
        self.assertGreater(result.delay.transmission_ms, 0.0)
        self.assertEqual(result.delay.processing_ms, 10.0)
        self.assertGreater(result.energy.device_wait_energy_j, 0.0)
        self.assertGreater(result.energy.compute_it_energy_j, 0.0)
        self.assertGreater(result.operator_cost_total, 0.0)
        self.assertIsNone(result.failure_reason)

    def test_remote_processing_that_exceeds_sla_is_reported_as_late_completion(self):
        result = self.runtime.execute(
            task(deadline_ms=1.0),
            server("MEC_1", "MEC"),
            current_time_ms=0.0,
            processing_duration_ms=10.0,
            utilization_before_ratio=0.0,
            utilization_after_ratio=0.1,
            source_position=Position3D(5000.0, 7400.0, 1.5),
            access_node_id="MEC_1",
        )

        self.assertTrue(result.application_completed)
        self.assertTrue(result.deadline_missed)
        self.assertEqual(result.network_status, DeliveryStatus.DELIVERED)
        self.assertGreater(result.delay.total_ms, 10.0)
        self.assertGreater(result.energy.compute_it_energy_j, 0.0)
        self.assertGreater(result.operator_cost.compute_cost, 0.0)
        self.assertGreater(result.operator_cost.transport_cost, 0.0)

    def test_handover_interruption_is_counted_once_as_pre_access_wait(self):
        result = self.runtime.execute(
            task(handover_delay_ms=5.0, handover_interruption_ms=25.0),
            server("MEC_1", "MEC"),
            current_time_ms=0.0,
            processing_duration_ms=10.0,
            utilization_before_ratio=0.0,
            utilization_after_ratio=0.1,
            source_position=Position3D(5000.0, 7400.0, 1.5),
            access_node_id="MEC_1",
        )

        self.assertEqual(result.delay.pre_access_wait_ms, 25.0)
        self.assertEqual(result.delay.handover_ms, 25.0)
        self.assertEqual(result.delay.coverage_outage_ms, 0.0)
        self.assertGreater(result.delay.total_ms, 35.0)

    def test_ground_bs_to_uav_executes_and_keeps_late_measurements(self):
        deployment = get_deployment_profile(SIMULATION["deployment_profile"])
        deployed = build_deployed_topology(
            "smartcity_default",
            get_topology("smartcity_default"),
            deployment,
            compute_tier_counts={
                "MEC": SERVER_CONFIG["MEC"]["num_datacenters"],
                "UAV_BS": SERVER_CONFIG["UAV_BS"]["num_datacenters"],
            },
        )
        topology = PhysicalTopology.from_mapping(deployed.profile_id, deployed.mapping)
        runtime = TaskExecutionRuntime(topology, experiment_seed=17)
        access = topology.node("BS_1").position
        uav = topology.node("UAV_BS_1").position

        result = runtime.execute(
            task(deadline_ms=1.0, data_size_kb=2.0),
            server("UAV_BS_1", "UAV_BS"),
            current_time_ms=0.0,
            processing_duration_ms=10.0,
            utilization_before_ratio=0.0,
            utilization_after_ratio=0.1,
            source_position=Position3D(access.x_m, access.y_m, 1.5),
            access_node_id="BS_1",
            position_overrides={"UAV_BS_1": uav},
        )

        self.assertTrue(result.application_completed)
        self.assertEqual(result.network_status, DeliveryStatus.DELIVERED)
        self.assertTrue(result.deadline_missed)
        self.assertGreater(result.delay.network_elapsed_ms, 0.0)
        self.assertEqual(result.delay.processing_ms, 10.0)
        self.assertGreater(result.energy.device_battery_energy_j, 0.0)
        self.assertGreater(result.energy.compute_it_energy_j, 0.0)
        self.assertGreater(result.operator_cost_total, 0.0)
        self.assertIsNone(result.failure_reason)


if __name__ == "__main__":
    unittest.main()
