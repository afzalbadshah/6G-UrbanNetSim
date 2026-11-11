import unittest

from config.emissions import (
    CARBON_INTENSITY_G_CO2_PER_JOULE,
    CARBON_INTENSITY_G_CO2_PER_KWH,
    EFFECTIVE_GRID_CARBON_INTENSITY_G_CO2_PER_KWH,
    EMBODIED_CARBON_PROFILES,
    RENEWABLE_ENERGY_SHARE_RATIO,
    annualized_embodied_kg_co2e,
    get_embodied_carbon_profile,
)


class GridCarbonIntensityTests(unittest.TestCase):
    def test_effective_intensity_nets_out_renewable_share(self):
        expected = CARBON_INTENSITY_G_CO2_PER_KWH * (1.0 - RENEWABLE_ENERGY_SHARE_RATIO)
        self.assertAlmostEqual(EFFECTIVE_GRID_CARBON_INTENSITY_G_CO2_PER_KWH, expected)
        # Renewable share nets emissions down, never up or negative.
        self.assertLess(EFFECTIVE_GRID_CARBON_INTENSITY_G_CO2_PER_KWH, CARBON_INTENSITY_G_CO2_PER_KWH)
        self.assertGreater(EFFECTIVE_GRID_CARBON_INTENSITY_G_CO2_PER_KWH, 0.0)

    def test_per_joule_intensity_matches_per_kwh_conversion(self):
        expected = EFFECTIVE_GRID_CARBON_INTENSITY_G_CO2_PER_KWH / 3.6e6
        self.assertAlmostEqual(CARBON_INTENSITY_G_CO2_PER_JOULE, expected)

    def test_renewable_share_is_a_valid_ratio(self):
        self.assertGreaterEqual(RENEWABLE_ENERGY_SHARE_RATIO, 0.0)
        self.assertLessEqual(RENEWABLE_ENERGY_SHARE_RATIO, 1.0)


class EmbodiedCarbonProfileTests(unittest.TestCase):
    def test_every_expected_tier_is_registered(self):
        self.assertEqual(
            set(EMBODIED_CARBON_PROFILES),
            {"onboard", "mec", "metro", "cloud", "ground_bs", "uav_bs"},
        )
        for tier, profile in EMBODIED_CARBON_PROFILES.items():
            self.assertGreater(profile["embodied_kg_co2e"], 0.0, tier)
            self.assertGreater(profile["lifetime_years"], 0.0, tier)
            self.assertIn("provenance", profile)

    def test_compute_tiers_scale_with_the_projects_own_capacity_ratios(self):
        # Metro and Cloud must be strictly heavier than MEC, in the same
        # relative order as config/servers.py's own tier capacity ratios
        # (Onboard < UAV_BS-compute-share < MEC < Metro < Cloud).
        mec = get_embodied_carbon_profile("mec")["embodied_kg_co2e"]
        metro = get_embodied_carbon_profile("metro")["embodied_kg_co2e"]
        cloud = get_embodied_carbon_profile("cloud")["embodied_kg_co2e"]
        self.assertLess(mec, metro)
        self.assertLess(metro, cloud)

    def test_unknown_tier_raises(self):
        with self.assertRaises(KeyError):
            get_embodied_carbon_profile("not-a-real-tier")

    def test_tier_name_is_case_and_whitespace_insensitive(self):
        self.assertIs(get_embodied_carbon_profile("MEC"), get_embodied_carbon_profile(" mec "))


class AnnualizedEmbodiedEmissionsTests(unittest.TestCase):
    def test_amortization_divides_by_lifetime_and_scales_with_unit_count(self):
        profile = get_embodied_carbon_profile("ground_bs")
        per_unit_per_year = profile["embodied_kg_co2e"] / profile["lifetime_years"]

        self.assertAlmostEqual(annualized_embodied_kg_co2e("ground_bs", unit_count=1), per_unit_per_year)
        self.assertAlmostEqual(
            annualized_embodied_kg_co2e("ground_bs", unit_count=16), per_unit_per_year * 16
        )

    def test_zero_units_is_zero(self):
        self.assertEqual(annualized_embodied_kg_co2e("mec", unit_count=0), 0.0)

    def test_negative_or_non_integer_unit_count_rejected(self):
        with self.assertRaises(ValueError):
            annualized_embodied_kg_co2e("mec", unit_count=-1)
        with self.assertRaises(ValueError):
            annualized_embodied_kg_co2e("mec", unit_count=1.5)


if __name__ == "__main__":
    unittest.main()
