"""Tests for tools/aggregate_seeds.py (Task 57 across-seed aggregation)."""
import csv
import os
import tempfile
import unittest

from tools import aggregate_seeds as agg

_COLS = ("network_status", "reward", "observed_complete_latency_ms",
         "total_energy_j", "cost")


def _write_results(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLS, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def _make_campaign(root, spec):
    """spec = {seed_label: {model: [row_dicts]}} -> writes the folder tree."""
    for seed_label, models in spec.items():
        for model, rows in models.items():
            _write_results(os.path.join(root, seed_label, "offloading", model,
                                        "results.csv"), rows)


class TestSummarizeRun(unittest.TestCase):
    def test_summary_math(self):
        tmp = tempfile.mkdtemp(prefix="agg57_")
        path = os.path.join(tmp, "results.csv")
        _write_results(path, [
            {"network_status": "delivered", "reward": "1.0",
             "observed_complete_latency_ms": "10", "total_energy_j": "0.2", "cost": "0.1"},
            {"network_status": "dropped", "reward": "-10.0",
             "observed_complete_latency_ms": "0", "total_energy_j": "0.0", "cost": "0.3"},
        ])
        s = agg.summarize_run(path)
        self.assertEqual(s["n_rows"], 2.0)
        self.assertEqual(s["n_completed"], 1.0)
        self.assertAlmostEqual(s["completion_ratio"], 0.5)
        self.assertAlmostEqual(s["mean_reward"], (1.0 - 10.0) / 2)
        # delay averaged over DELIVERED rows only.
        self.assertAlmostEqual(s["mean_e2e_delay_ms_completed"], 10.0)
        self.assertAlmostEqual(s["mean_total_energy_j"], 0.1)
        self.assertAlmostEqual(s["mean_cost"], 0.2)


class TestAggregateModel(unittest.TestCase):
    def test_mean_min_max_across_seeds(self):
        tmp = tempfile.mkdtemp(prefix="agg57_")
        _make_campaign(tmp, {
            "seed_1": {"dqn": [{"network_status": "delivered", "reward": "2.0",
                                "observed_complete_latency_ms": "20",
                                "total_energy_j": "0.4", "cost": "0.1"}]},
            "seed_2": {"dqn": [{"network_status": "delivered", "reward": "4.0",
                                "observed_complete_latency_ms": "40",
                                "total_energy_j": "0.6", "cost": "0.3"}]},
        })
        a = agg.aggregate_model(tmp, "dqn")
        self.assertEqual(a["seeds"], ["seed_1", "seed_2"])
        self.assertEqual(a["per_seed"]["mean_reward"], [2.0, 4.0])
        self.assertAlmostEqual(a["mean"]["mean_reward"], 3.0)
        self.assertAlmostEqual(a["min"]["mean_reward"], 2.0)
        self.assertAlmostEqual(a["max"]["mean_reward"], 4.0)

    def test_seed_dir_ordering_is_numeric(self):
        tmp = tempfile.mkdtemp(prefix="agg57_")
        for label in ("seed_10", "seed_2", "seed_1"):
            _write_results(os.path.join(tmp, label, "offloading", "m", "results.csv"),
                           [{"network_status": "delivered", "reward": "0",
                             "observed_complete_latency_ms": "0",
                             "total_energy_j": "0", "cost": "0"}])
        dirs = [os.path.basename(d) for d in agg.discover_seed_dirs(tmp)]
        self.assertEqual(dirs, ["seed_1", "seed_2", "seed_10"])  # numeric, not lexical


class TestWriteAggregate(unittest.TestCase):
    def test_paired_reward_wins_vs_baseline(self):
        tmp = tempfile.mkdtemp(prefix="agg57_")
        base = {"network_status": "delivered", "reward": "0.0",
                "observed_complete_latency_ms": "5", "total_energy_j": "0.1", "cost": "0.1"}
        better = dict(base, reward="5.0")
        _make_campaign(tmp, {
            "seed_1": {"random": [base], "dqn": [better]},
            "seed_2": {"random": [base], "dqn": [better]},
        })
        agg.write_aggregate(tmp, models=["random", "dqn"], baseline="random")
        comp = os.path.join(tmp, "aggregate", "comparison_summary.csv")
        self.assertTrue(os.path.exists(comp))
        with open(comp, encoding="utf-8") as fh:
            rows = {r["model"]: r for r in csv.DictReader(fh)}
        # dqn beats random on reward in BOTH seeds; random beats itself in none.
        self.assertEqual(rows["dqn"]["reward_seeds_beating_random"], "2")
        self.assertEqual(rows["dqn"]["seeds_compared"], "2")
        self.assertEqual(rows["random"]["reward_seeds_beating_random"], "0")
        # per-model detail file exists too.
        self.assertTrue(os.path.exists(os.path.join(tmp, "aggregate", "dqn_summary.csv")))


if __name__ == "__main__":
    unittest.main()
