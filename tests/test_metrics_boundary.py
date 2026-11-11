"""Regression guards for the staged ``metrics`` package migration (Task 28)."""

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

import metrics
from metrics import handover_channel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestMetricsPackageBoundary(unittest.TestCase):
    def test_package_publishes_only_verified_formulas(self):
        self.assertEqual(
            metrics.__all__,
            (
                "ApplicationObservation",
                "AccountingPerspective",
                "ComputeEnergyBreakdown",
                "IncrementalComputeEnergyBreakdown",
                "CostLine",
                "CostMeasurement",
                "CostSnapshot",
                "DelayDistribution",
                "KpiSnapshot",
                "PacketDelayBreakdown",
                "PacketRouteDelay",
                "TaskDelayBreakdown",
                "TaskEnergyBreakdown",
                "PriceBasis",
                "RadioEnergyBreakdown",
                "RadioPacketEnergyModel",
                "RewardBreakdown",
                "RewardObservation",
                "SixGChannelModel",
                "TariffProfile",
                "TransportEnergyBreakdown",
                "TransportPacketEnergyModel",
                "UavEnergyBreakdown",
                "aggregate_kpis",
                "aggregate_cost_lines",
                "calculate_task_delay",
                "calculate_task_energy",
                "calculate_cost_line",
                "compute_energy",
                "energy_from_power_duration_j",
                "incremental_compute_energy",
                "linear_cost",
                "observe_route",
                "packet_delay_breakdown",
                "packet_radio_energy",
                "packet_route_delay",
                "percentile_type7",
                "propagation_delay_s",
                "radio_energy",
                "measure_cost",
                "evaluate_reward",
                "marginal_shared_capacity_pressure",
                "round_cost_for_presentation",
                "route_completion_delay_s",
                "transmission_delay_s",
                "sum_costs",
                "sum_delay_components_s",
                "transport_energy",
                "uav_energy",
            ),
        )
        self.assertIs(metrics.SixGChannelModel, metrics.channel.SixGChannelModel)
        self.assertFalse(hasattr(metrics, "calculate_total_delay"))
        self.assertFalse(hasattr(metrics, "calculate_total_cost"))
        self.assertFalse(hasattr(metrics, "TaskMetrics"))
        self.assertFalse(hasattr(metrics, "compose_task_metrics"))

    def test_configuration_has_no_metric_formula_module(self):
        self.assertIsNone(importlib.util.find_spec("config.metrics"))
        self.assertIsNone(importlib.util.find_spec("metrics.legacy_runtime"))

    def test_import_does_not_eagerly_load_handover_adapter(self):
        check = (
            "import sys; import metrics; "
            "raise SystemExit(1 if 'metrics.handover_channel' in sys.modules else 0)"
        )
        result = subprocess.run(
            [sys.executable, "-c", check],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

    def test_handover_adapter_exposes_only_radio_observations(self):
        for name in (
            "calculate_rssi_dbm",
            "calculate_noise_power_dbm",
            "calculate_interference_power_dbm",
            "calculate_snr_db",
            "calculate_sinr_db",
        ):
            self.assertTrue(callable(getattr(handover_channel, name)))
        self.assertFalse(hasattr(handover_channel, "calculate_total_cost"))
        self.assertFalse(hasattr(handover_channel, "calculate_total_delay"))


if __name__ == "__main__":
    unittest.main()
