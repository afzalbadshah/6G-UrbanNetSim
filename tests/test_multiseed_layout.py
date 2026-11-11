"""Tests for the multi-seed campaign layout in applications/MainApplication.py
(Task 57): seed-folder naming, job fan-out, and manifest provenance. These test
the pure orchestration helpers only — they do NOT run a simulation.
"""
import json
import os
import tempfile
import unittest

import applications.MainApplication as app
from config.evaluation import get_evaluation_plan, resolved_seeds


class TestSeedDir(unittest.TestCase):
    def test_seed_dir_is_one_based(self):
        self.assertTrue(app._seed_dir("/x/evaluation", 0).endswith(os.path.join("evaluation", "seed_1")))
        self.assertTrue(app._seed_dir("/x/evaluation", 2).endswith(os.path.join("evaluation", "seed_3")))


class TestBuildJobs(unittest.TestCase):
    def test_job_fanout_and_isolation(self):
        models = ["random", "dqn"]
        seeds = resolved_seeds("default", purpose="eval", count=3)
        phase = os.path.join("root", "evaluation")
        jobs = app._build_jobs(models, seeds, train=False, phase_dir=phase)

        self.assertEqual(len(jobs), len(seeds) * len(models))  # full grid
        # Each seed maps to its own folder; each job carries the right seed value.
        # Run settings (workload, models) are read from config.py §1 by the worker,
        # so a job only carries model/seed/output — no duplicated config fields.
        for job in jobs:
            self.assertEqual(job["output_dir"], app._seed_dir(phase, job["seed_index"]))
            self.assertEqual(job["seed"], seeds[job["seed_index"]])
            self.assertFalse(job["train"])
            self.assertNotIn("overrides", job)     # config not duplicated into jobs
            self.assertNotIn("run_models", job)
        # Distinct (seed, model) pairs, no collisions.
        pairs = {(j["seed_index"], j["model_name"]) for j in jobs}
        self.assertEqual(len(pairs), len(jobs))


class TestManifest(unittest.TestCase):
    def test_manifest_records_seeds_and_methodology(self):
        tmp = tempfile.mkdtemp(prefix="ms57_")
        plan = get_evaluation_plan("default")
        seeds = resolved_seeds(plan, purpose="eval")
        path = app._write_manifest(tmp, ["random", "dqn"], seeds, train=False, plan=plan)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["master_seed"], plan["master_seed"])
        self.assertEqual(data["repetitions"], len(seeds))
        self.assertEqual(data["seed_purpose"], "eval")
        self.assertEqual(set(data["seeds"].values()), set(seeds))
        self.assertEqual(data["aggregate_metric"], plan["aggregate_metric"])
        self.assertEqual(data["confidence_interval_method"], plan["confidence_interval_method"])
        self.assertIn("models", data)


if __name__ == "__main__":
    unittest.main()
