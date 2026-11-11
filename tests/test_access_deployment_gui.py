"""Regression checks for the access-only Live Map representation."""

import unittest

from config import SIMULATION
from entities.access_point import AccessPoint
from live.tiers import TIER_COLORS, tier_from_server_obj


class AccessDeploymentGuiTests(unittest.TestCase):
    def test_interactive_default_creates_a_visible_access_layer(self):
        self.assertEqual(
            SIMULATION["deployment_profile"],
            "fr3_city_coverage_6g",
        )
        self.assertEqual(
            SIMULATION["interactive_deployment_profile"],
            "fr3_city_coverage_6g",
        )

    def test_access_point_has_a_distinct_display_category(self):
        access_point = AccessPoint(
            name="BS_1",
            index=0,
            x=100.0,
            y=200.0,
            z=25.0,
            profile_id="nr-fr1-access",
            compute_anchor_id="MEC_1",
        )
        self.assertEqual(tier_from_server_obj(access_point), "ACCESS_POINT")
        self.assertIn("ACCESS_POINT", TIER_COLORS)
        self.assertNotEqual(TIER_COLORS["ACCESS_POINT"], TIER_COLORS["MEC"])


if __name__ == "__main__":
    unittest.main()
