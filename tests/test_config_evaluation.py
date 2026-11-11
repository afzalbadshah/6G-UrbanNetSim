"""Tests for config/evaluation.py (Task 33 / R4 evaluation DATA).

Verifies contract §15 (EvaluationPlan fields + reproducibility metadata), §11
(visionary excluded from baseline), SI units, and deep immutability.
"""

import unittest

from config import evaluation as ev
from config._provenance import MATURITIES

PROV_FIELDS = {"source_id", "source_uri", "source_version", "valid_from",
               "valid_to", "validity_notes", "maturity"}


class TestEvaluationPlans(unittest.TestCase):
    def test_plans_present(self):
        self.assertEqual(set(ev.EVALUATION_PLANS), {"default", "handcheck_smoke"})

    def test_required_fields(self):
        for name, plan in ev.EVALUATION_PLANS.items():
            with self.subTest(plan=name):
                self.assertEqual(set(plan), ev.REQUIRED_KEYS)

    def test_reproducibility_metadata(self):
        p = ev.get_evaluation_plan("default")
        self.assertIsInstance(p["master_seed"], int)
        # Task 57: >=3 training seeds (owner compute-budget cap). At n=3 the plan
        # reports the observed min-max range + every seed, not a parametric CI.
        self.assertGreaterEqual(p["repetitions"], 3)
        self.assertTrue(0.0 < p["confidence_level"] < 1.0)
        self.assertIn(p["confidence_interval_method"], ev.CI_METHODS)
        self.assertIn(p["aggregate_metric"], ev.AGGREGATE_METRICS)
        # CRN policy must reference the §17 PER key.
        self.assertIn("attempt_number", p["common_random_number_policy"])
        self.assertIn("per", p["common_random_number_policy"])

    def test_resolved_seeds_deterministic_and_disjoint(self):
        # Reproducible from the one disclosed master seed.
        first = ev.resolved_seeds("default", purpose="eval")
        again = ev.resolved_seeds("default", purpose="eval")
        self.assertEqual(first, again)
        self.assertEqual(len(first), ev.get_evaluation_plan("default")["repetitions"])
        # All seeds are valid 32-bit non-negative ints.
        for s in first:
            self.assertIsInstance(s, int)
            self.assertTrue(0 <= s < 2 ** 32)
        # Train and eval streams are disjoint (held-out evaluation).
        train = ev.resolved_seeds("default", purpose="train")
        self.assertFalse(set(first) & set(train))
        # count overrides repetitions for cheap eval reps.
        self.assertEqual(len(ev.resolved_seeds("default", purpose="eval", count=10)), 10)

    def test_units_seconds(self):
        for plan in ev.EVALUATION_PLANS.values():
            self.assertGreaterEqual(plan["warmup_window_s"], 0.0)
            self.assertIn(plan["measurement_window_s"], (None,)) if plan["measurement_window_s"] is None \
                else self.assertGreater(plan["measurement_window_s"], 0.0)

    def test_enabled_kpis_subset(self):
        for plan in ev.EVALUATION_PLANS.values():
            for kpi in plan["enabled_kpis"]:
                self.assertIn(kpi, ev.KPI_NAMES)

    def test_visionary_excluded_from_baseline(self):
        for plan in ev.EVALUATION_PLANS.values():
            self.assertNotIn("visionary", plan["baseline_allowed_maturities"])
            for m in plan["baseline_allowed_maturities"]:
                self.assertIn(m, MATURITIES)

    def test_speculative_flags_off_by_default(self):
        flags = ev.get_evaluation_plan("default")["enabled_feature_flags"]
        for flag in ("stochastic_los", "isac_sensing", "ai_as_a_service", "ntn"):
            self.assertFalse(flags[flag])

    def test_calibration_and_provenance(self):
        for plan in ev.EVALUATION_PLANS.values():
            self.assertTrue(plan["calibration_references"])
            prov = plan["provenance"]
            self.assertEqual(set(prov), PROV_FIELDS)
            self.assertIn(prov["maturity"], MATURITIES)

    def test_deep_immutable(self):
        p = ev.get_evaluation_plan("default")
        with self.assertRaises(TypeError):
            p["master_seed"] = 0
        with self.assertRaises(TypeError):
            p["enabled_feature_flags"]["isac_sensing"] = True

    def test_unknown_plan_raises(self):
        with self.assertRaises(KeyError):
            ev.get_evaluation_plan("mega_run")


if __name__ == "__main__":
    unittest.main()
