"""Tests for learning schedule orchestration."""

import unittest

import config as cfg


class TestLearningSchedule(unittest.TestCase):
    def setUp(self):
        self._workload = dict(cfg.WORKLOAD)
        self._decay_steps = cfg.EPSILON_DECAY_STEPS

    def tearDown(self):
        cfg.WORKLOAD.clear()
        cfg.WORKLOAD.update(self._workload)
        cfg.EPSILON_DECAY_STEPS = self._decay_steps

    def test_training_run_decays_exploration_across_full_run(self):
        workload = cfg.apply_workload_profile(
            train_mode=True,
            overrides={
                "start_devices": 1,
                "max_devices": 100,
                "increment": 1,
                "rounds": 500,
            },
        )

        self.assertEqual(cfg.total_rounds(workload), 500)
        self.assertEqual(cfg.EPSILON_DECAY_STEPS, 500)

    def test_short_training_run_uses_its_full_length(self):
        workload = cfg.apply_workload_profile(
            train_mode=True,
            overrides={
                "start_devices": 1,
                "max_devices": 40,
                "increment": 1,
                "rounds": 80,
            },
        )

        self.assertEqual(cfg.total_rounds(workload), 80)
        self.assertEqual(cfg.EPSILON_DECAY_STEPS, 80)


if __name__ == "__main__":
    unittest.main()
