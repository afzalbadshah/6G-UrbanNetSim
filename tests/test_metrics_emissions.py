import unittest

from metrics.emissions import (
    DeploymentEmissionsSummary,
    EmbodiedEmissionsLine,
    operational_emissions_g,
    summarize_deployment_embodied_emissions,
)


class OperationalEmissionsTests(unittest.TestCase):
    def test_operational_emissions_is_energy_times_intensity(self):
        self.assertAlmostEqual(
            operational_emissions_g(energy_j=1000.0, carbon_intensity_g_per_j=2.0e-4),
            1000.0 * 2.0e-4,
        )

    def test_zero_energy_is_zero_emissions(self):
        self.assertEqual(operational_emissions_g(energy_j=0.0, carbon_intensity_g_per_j=1.0), 0.0)

    def test_negative_inputs_rejected(self):
        with self.assertRaises(ValueError):
            operational_emissions_g(energy_j=-1.0, carbon_intensity_g_per_j=1.0)
        with self.assertRaises(ValueError):
            operational_emissions_g(energy_j=1.0, carbon_intensity_g_per_j=-1.0)


class EmbodiedEmissionsLineTests(unittest.TestCase):
    def test_valid_line_constructs(self):
        line = EmbodiedEmissionsLine(tier="mec", unit_count=6, annualized_kg_co2e=1410.0)
        self.assertEqual(line.tier, "mec")
        self.assertEqual(line.unit_count, 6)
        self.assertAlmostEqual(line.annualized_kg_co2e, 1410.0)

    def test_invalid_fields_rejected(self):
        with self.assertRaises(ValueError):
            EmbodiedEmissionsLine(tier="", unit_count=1, annualized_kg_co2e=1.0)
        with self.assertRaises(ValueError):
            EmbodiedEmissionsLine(tier="mec", unit_count=-1, annualized_kg_co2e=1.0)
        with self.assertRaises(ValueError):
            EmbodiedEmissionsLine(tier="mec", unit_count=1, annualized_kg_co2e=-1.0)


class DeploymentEmissionsSummaryTests(unittest.TestCase):
    def test_total_sums_lines(self):
        summary = DeploymentEmissionsSummary(
            lines=(
                EmbodiedEmissionsLine(tier="mec", unit_count=6, annualized_kg_co2e=1410.0),
                EmbodiedEmissionsLine(tier="ground_bs", unit_count=81, annualized_kg_co2e=35640.0),
            )
        )
        self.assertAlmostEqual(summary.total_annualized_kg_co2e, 1410.0 + 35640.0)

    def test_empty_summary_totals_zero(self):
        self.assertEqual(DeploymentEmissionsSummary(lines=()).total_annualized_kg_co2e, 0.0)

    def test_rejects_non_embodied_line_values(self):
        with self.assertRaises(ValueError):
            DeploymentEmissionsSummary(lines=({"tier": "mec"},))


class SummarizeDeploymentEmbodiedEmissionsTests(unittest.TestCase):
    def test_builds_one_line_per_nonzero_tier(self):
        summary = summarize_deployment_embodied_emissions(
            {"mec": 6, "uav_bs": 4, "metro": 0, "ground_bs": 81}
        )
        tiers = {line.tier for line in summary.lines}
        # metro had a zero count -- omitted, not a zero row.
        self.assertEqual(tiers, {"mec", "uav_bs", "ground_bs"})
        self.assertGreater(summary.total_annualized_kg_co2e, 0.0)

    def test_matches_direct_per_tier_calculation(self):
        from config.emissions import annualized_embodied_kg_co2e

        summary = summarize_deployment_embodied_emissions({"mec": 6})
        self.assertAlmostEqual(
            summary.total_annualized_kg_co2e,
            annualized_embodied_kg_co2e("mec", unit_count=6),
        )

    def test_unknown_tier_raises(self):
        with self.assertRaises(KeyError):
            summarize_deployment_embodied_emissions({"not-a-real-tier": 1})

    def test_non_mapping_input_rejected(self):
        with self.assertRaises(ValueError):
            summarize_deployment_embodied_emissions([("mec", 6)])


if __name__ == "__main__":
    unittest.main()
