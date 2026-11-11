"""Regression tests for versioned campaign output and plot-source selection."""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import config as cfg
from core.run_history import create_run, latest_run, run_number, select_phase
from visualization import plot_config


class RunHistoryTests(unittest.TestCase):
    def test_runs_increment_and_write_recognizable_manifests(self):
        with tempfile.TemporaryDirectory(prefix="run_history_") as folder:
            root = Path(folder) / "output"
            first = create_run(
                root,
                deployment_profile="fr3_city_coverage_6g",
                mode="training",
                created_utc=datetime(2026, 9, 30, tzinfo=timezone.utc),
            )
            second = create_run(
                root,
                deployment_profile="fr1_city_coverage",
                mode="evaluation",
            )

            self.assertEqual(first.name, "run_001_fr3_city_coverage_6g_training")
            self.assertEqual(second.name, "run_002_fr1_city_coverage_evaluation")
            self.assertEqual(run_number(second), 2)
            self.assertEqual(latest_run(root), second)
            manifest = json.loads((first / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["run_number"], 1)
            self.assertEqual(manifest["deployment_profile"], "fr3_city_coverage_6g")
            self.assertEqual(manifest["phase"], "exploration")

    def test_plot_selection_uses_highest_run_or_explicit_pin(self):
        original_root = cfg.OUTPUT_DIR
        original_selections = dict(plot_config.VISUALIZATION_RUNS)
        try:
            with tempfile.TemporaryDirectory(prefix="run_history_") as folder:
                root = Path(folder) / "output"
                first = create_run(
                    root,
                    deployment_profile="first",
                    mode="evaluation",
                )
                second = create_run(
                    root,
                    deployment_profile="second",
                    mode="evaluation",
                )
                cfg.OUTPUT_DIR = str(root)
                plot_config.VISUALIZATION_RUNS["evaluation"] = None
                self.assertEqual(
                    plot_config.plot_phase_dir("evaluation"),
                    second / "evaluation",
                )

                plot_config.VISUALIZATION_RUNS["evaluation"] = first.name
                self.assertEqual(
                    plot_config.plot_phase_dir("evaluation"),
                    first / "evaluation",
                )
        finally:
            cfg.OUTPUT_DIR = original_root
            plot_config.VISUALIZATION_RUNS.clear()
            plot_config.VISUALIZATION_RUNS.update(original_selections)

    def test_each_phase_uses_its_highest_compatible_run(self):
        with tempfile.TemporaryDirectory(prefix="run_history_") as folder:
            root = Path(folder) / "output"
            evaluation = create_run(
                root,
                deployment_profile="evaluation_scenario",
                mode="evaluation",
            )
            training = create_run(
                root,
                deployment_profile="training_scenario",
                mode="training",
            )

            self.assertEqual(select_phase(root, "evaluation"), evaluation / "evaluation")
            self.assertEqual(select_phase(root, "exploration"), training / "exploration")

    def test_training_and_evaluation_can_be_pinned_independently(self):
        original_root = cfg.OUTPUT_DIR
        original_selections = dict(plot_config.VISUALIZATION_RUNS)
        try:
            with tempfile.TemporaryDirectory(prefix="run_history_") as folder:
                root = Path(folder) / "output"
                evaluation = create_run(root, deployment_profile="eval", mode="evaluation")
                training = create_run(root, deployment_profile="train", mode="training")
                cfg.OUTPUT_DIR = str(root)
                plot_config.VISUALIZATION_RUNS.update({
                    "exploration": training.name,
                    "evaluation": str(evaluation),
                })

                self.assertEqual(
                    plot_config.plot_phase_dir("exploration"),
                    training / "exploration",
                )
                self.assertEqual(
                    plot_config.plot_phase_dir("evaluation"),
                    evaluation / "evaluation",
                )
        finally:
            cfg.OUTPUT_DIR = original_root
            plot_config.VISUALIZATION_RUNS.clear()
            plot_config.VISUALIZATION_RUNS.update(original_selections)

    def test_pinned_run_must_contain_the_requested_phase(self):
        with tempfile.TemporaryDirectory(prefix="run_history_") as folder:
            root = Path(folder) / "output"
            training = create_run(root, deployment_profile="train", mode="training")
            with self.assertRaisesRegex(FileNotFoundError, "does not contain"):
                select_phase(root, "evaluation", training.name)

    def test_legacy_phase_is_used_before_first_versioned_run(self):
        with tempfile.TemporaryDirectory(prefix="run_history_") as folder:
            root = Path(folder) / "output"
            self.assertEqual(select_phase(root, "evaluation"), root / "evaluation")


if __name__ == "__main__":
    unittest.main()
