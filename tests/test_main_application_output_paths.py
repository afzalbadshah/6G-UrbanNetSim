"""Regression tests for phase-specific application output roots."""

import os
import tempfile
import unittest
from pathlib import Path

import applications.MainApplication as app
from scheduler.common import common_config


class TestMainApplicationOutputPaths(unittest.TestCase):
    def test_evaluation_uses_evaluation_root(self):
        original = app.cfg.RUN_TRAINING
        try:
            app.cfg.RUN_TRAINING = False
            app.RUNTIME.clear()
            app._refresh_runtime_flags()

            expected = os.path.join(app.ROOT_DIR, app.cfg.OUTPUT_DIR, "evaluation")
            self.assertEqual(app._phase_output_dir(), expected)
            self.assertEqual(common_config.OUTPUT_DIR, expected)
        finally:
            app.cfg.RUN_TRAINING = original
            app.RUNTIME.clear()
            app._refresh_runtime_flags()

    def test_training_uses_exploration_root(self):
        original = app.cfg.RUN_TRAINING
        try:
            app.cfg.RUN_TRAINING = True
            app.RUNTIME.clear()
            app._refresh_runtime_flags()

            expected = os.path.join(app.ROOT_DIR, app.cfg.OUTPUT_DIR, "exploration")
            self.assertEqual(app._phase_output_dir(), expected)
            self.assertEqual(common_config.OUTPUT_DIR, expected)
        finally:
            app.cfg.RUN_TRAINING = original
            app.RUNTIME.clear()
            app._refresh_runtime_flags()

    def test_launch_creates_one_versioned_run_shared_by_all_phases(self):
        original_output = app.cfg.OUTPUT_DIR
        original_training = app.cfg.RUN_TRAINING
        try:
            with tempfile.TemporaryDirectory(prefix="app_runs_") as folder:
                app.cfg.OUTPUT_DIR = str(Path(folder) / "output")
                app.cfg.RUN_TRAINING = True
                app.RUNTIME.clear()
                app._refresh_runtime_flags()

                first = Path(app._start_run_output())
                self.assertTrue(first.name.startswith("run_001_"))
                self.assertTrue(first.name.endswith("_training"))
                self.assertEqual(app._start_run_output(), str(first))
                self.assertEqual(
                    Path(app._phase_output_dir()),
                    first / "exploration",
                )
                self.assertEqual(
                    Path(common_config.OUTPUT_DIR),
                    first / "exploration",
                )

                app.RUNTIME.clear()
                app._refresh_runtime_flags()
                second = Path(app._start_run_output())
                self.assertTrue(second.name.startswith("run_002_"))
        finally:
            app.cfg.OUTPUT_DIR = original_output
            app.cfg.RUN_TRAINING = original_training
            app.RUNTIME.clear()
            app._refresh_runtime_flags()


if __name__ == "__main__":
    unittest.main()
