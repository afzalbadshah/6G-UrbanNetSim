"""End-to-end check that a changed radio channel reaches realized metrics."""

import unittest
from types import SimpleNamespace

from config.topology import get_topology
from core.task_execution import TaskExecutionRuntime
from network import PhysicalTopology, Position3D


class MetricChannelSensitivityTests(unittest.TestCase):
    def test_interference_changes_realized_delay_energy_and_operator_cost(self):
        topology = PhysicalTopology.from_mapping(
            "smartcity_default", get_topology("smartcity_default")
        )

        def execute(interference_dbm):
            task = SimpleNamespace(
                id="T0",
                device_id=0,
                entity_id=0,
                data_size_kb=0.1,
                cpu_demand=1.0,
                memory_demand=0.0,
                storage_demand=0.1,
                deadline_ms=1000.0,
                service_class="URLLC",
                application_type="safety_upload",
                tx_power_w=1.3,
                interference_dbm=interference_dbm,
                handover_delay_ms=0.0,
                handover_interruption_ms=0.0,
                coverage_outage_flag=0,
                coverage_outage_duration_ms=0.0,
            )
            runtime = TaskExecutionRuntime(topology, experiment_seed=17)
            return runtime.execute(
                task,
                SimpleNamespace(name="MEC_1", tier="MEC"),
                current_time_ms=0.0,
                processing_duration_ms=10.0,
                utilization_before_ratio=0.0,
                utilization_after_ratio=0.1,
                source_position=Position3D(5000.0, 7400.0, 1.5),
                access_node_id="MEC_1",
            )

        clean = execute(-200.0)
        interfered = execute(-90.0)

        self.assertTrue(clean.application_completed)
        self.assertTrue(interfered.application_completed)
        self.assertLess(
            interfered.channel_results[0].sinr_db,
            clean.channel_results[0].sinr_db,
        )
        self.assertLess(
            interfered.channel_results[0].achievable_rate_bps,
            clean.channel_results[0].achievable_rate_bps,
        )
        self.assertGreater(
            interfered.delay.wireless_transmission_ms,
            clean.delay.wireless_transmission_ms,
        )
        self.assertGreater(
            interfered.energy.device_tx_energy_j,
            clean.energy.device_tx_energy_j,
        )
        self.assertGreater(interfered.operator_cost_total, clean.operator_cost_total)


if __name__ == "__main__":
    unittest.main()
