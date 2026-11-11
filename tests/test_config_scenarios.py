"""Tests for config/scenarios.py (Task 27, 6G scenario DATA).

Verifies: the six IMT-2030 scenarios, unique priority ints, back-compat with the
existing 3 classes (deadlines match PRIORITY_PROFILES), and that scenario `plane`
values are valid against Codex's network.Plane enum (cross-module sync check).
"""

import unittest

from config import workload as scenarios
from config import PRIORITY_PROFILES
from network import Plane


class TestScenarioProfiles(unittest.TestCase):
    def test_six_scenarios_present(self):
        self.assertEqual(
            set(scenarios.SCENARIO_PROFILES),
            {"URLLC", "mMTC", "eMBB", "ISAC", "AIComm", "Ubiquitous"},
        )

    def test_priority_ints_unique_and_1_to_6(self):
        ints = sorted(p["priority_int"] for p in scenarios.SCENARIO_PROFILES.values())
        self.assertEqual(ints, [1, 2, 3, 4, 5, 6])

    def test_three_new_scenarios_flagged(self):
        new = {k for k, v in scenarios.SCENARIO_PROFILES.items() if v["is_new_6g"]}
        self.assertEqual(new, {"ISAC", "AIComm", "Ubiquitous"})

    def test_backcompat_deadlines_match_priority_profiles(self):
        # The first three scenarios must not diverge from the authoritative
        # PRIORITY_PROFILES deadlines (single source of truth for the 3 classes).
        for key, prio in (("URLLC", 1), ("mMTC", 2), ("eMBB", 3)):
            with self.subTest(scenario=key):
                self.assertEqual(
                    scenarios.SCENARIO_PROFILES[key]["deadline_ms"],
                    PRIORITY_PROFILES[prio]["deadline_ms"],
                )

    def test_plane_values_valid_against_network_enum(self):
        valid_planes = {p.value for p in Plane}
        for key, profile in scenarios.SCENARIO_PROFILES.items():
            with self.subTest(scenario=key):
                self.assertIn(profile["plane"], valid_planes)

    def test_isac_uses_sensing_plane(self):
        self.assertEqual(scenarios.SCENARIO_PROFILES["ISAC"]["plane"], Plane.SENSING.value)

    def test_priority_int_to_scenario_mapping(self):
        self.assertEqual(scenarios.PRIORITY_INT_TO_SCENARIO[1], "URLLC")
        self.assertEqual(scenarios.PRIORITY_INT_TO_SCENARIO[4], "ISAC")

    def test_get_scenario_unknown_raises(self):
        with self.assertRaises(KeyError):
            scenarios.get_scenario("6G-TV")

    def test_runtime_sla_profiles_cover_current_classes_and_generations(self):
        self.assertEqual(set(scenarios.SLA_PROFILES), {"URLLC", "mMTC", "eMBB"})
        for service_class in scenarios.SLA_PROFILES:
            with self.subTest(service_class=service_class):
                self.assertEqual(
                    set(scenarios.SLA_PROFILES[service_class]),
                    {"5G", "6G"},
                )

    def test_6g_targets_are_tighter_but_5g_deadlines_stay_compatible(self):
        for service_class, priority in (("URLLC", 1), ("mMTC", 2), ("eMBB", 3)):
            with self.subTest(service_class=service_class):
                five_g = scenarios.get_sla(service_class, "5g")
                six_g = scenarios.get_sla(service_class, "6G")
                self.assertEqual(
                    five_g["application_deadline_ms"],
                    PRIORITY_PROFILES[priority]["deadline_ms"],
                )
                self.assertLess(
                    six_g["application_deadline_ms"],
                    five_g["application_deadline_ms"],
                )
                self.assertLess(
                    six_g["radio_latency_target_ms"],
                    five_g["radio_latency_target_ms"],
                )
                self.assertIn("experimental", six_g["maturity"])

    def test_sla_lookup_rejects_unknown_class_or_generation(self):
        with self.assertRaises(KeyError):
            scenarios.get_sla("ISAC", "6G")
        with self.assertRaises(ValueError):
            scenarios.get_sla("URLLC", "7G")

    def test_delay_weight_matches_class_role(self):
        # Corrected per owner: the CONTINUOUS delay reward is the defining
        # objective for URLLC (latency-critical) and must be high; for the
        # non-latency classes (mMTC battery-led, eMBB compute-led) it stays a
        # small shaping term. The binary SLA-violation flag carries no reward
        # weight in either case.
        for priority, profile in PRIORITY_PROFILES.items():
            with self.subTest(priority=priority):
                delay_weight = profile["objective_weights"]["delay"]
                self.assertGreater(delay_weight, 0.0)
        # URLLC (priority 1): delay is dominant (at least half of the weights).
        urllc = PRIORITY_PROFILES[1]["objective_weights"]
        self.assertGreaterEqual(
            urllc["delay"] / sum(abs(weight) for weight in urllc.values()), 0.5
        )
        # mMTC remains battery-led; eMBB remains compute-led. Delay may still
        # act as a secondary shaping objective for either class.
        self.assertGreater(
            PRIORITY_PROFILES[2]["objective_weights"]["device_battery"],
            PRIORITY_PROFILES[2]["objective_weights"]["delay"],
        )
        self.assertGreater(
            PRIORITY_PROFILES[3]["objective_weights"]["compute"],
            PRIORITY_PROFILES[3]["objective_weights"]["delay"],
        )


if __name__ == "__main__":
    unittest.main()
