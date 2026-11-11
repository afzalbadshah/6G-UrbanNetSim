"""Tests for the Task 61 multi-seed and all-model training figures."""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from visualization.supportive import automatic_plotting
from visualization.supportive import requested_training_plots as plots


ACTIONS = ("LOCAL_DEVICE", "MEC", "MEC_COLLAB", "METRO", "CLOUD", "UAV_BS")


def _write_csv(path, fieldnames, rows, delimiter=","):
    """Write one small fixture CSV using the simulator's folder conventions."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def _write_run(root, seed_label, model, rewards, include_epsilon=True):
    """Create the four authoritative per-seed inputs consumed by the plotter."""
    model_dir = Path(root) / seed_label / "offloading" / model
    reward_rows = []
    action_rows = []
    result_rows = []
    for round_number, reward in enumerate(rewards, start=1):
        priority = ((round_number - 1) % 3) + 1
        action = (round_number - 1) % len(ACTIONS)
        reward_rows.append(
            {"round": round_number, "reward": reward, "action": action, "priority": priority}
        )
        shares = {name: 1.0 if index == action else 0.0 for index, name in enumerate(ACTIONS)}
        action_rows.append(
            {
                "round": round_number,
                "priority": "ALL",
                "priority_class": "ALL",
                **shares,
                "avg_reward": reward,
            }
        )
        action_rows.append(
            {
                "round": round_number,
                "priority": priority,
                "priority_class": {1: "URLLC", 2: "mMTC", 3: "eMBB"}[priority],
                **shares,
                "avg_reward": reward,
            }
        )
        result_rows.append(
            {
                "round": round_number,
                "status": "completed",
                "network_status": "delivered",
                "observed_complete_latency_ms": 10.0 + round_number,
                "deadline_missed": int(round_number == len(rewards)),
            }
        )
    _write_csv(
        model_dir / f"{model}_reward_log.csv",
        ("round", "reward", "action", "priority"),
        reward_rows,
    )
    _write_csv(
        model_dir / f"{model}_action_distribution.csv",
        ("round", "priority", "priority_class", *ACTIONS, "avg_reward"),
        action_rows,
    )
    _write_csv(
        model_dir / "results.csv",
        (
            "round",
            "status",
            "network_status",
            "observed_complete_latency_ms",
            "deadline_missed",
        ),
        result_rows,
        delimiter=";",
    )
    if include_epsilon:
        _write_csv(
            model_dir / f"{model}_epsilon_decay.csv",
            ("round", "epsilon"),
            [
                {"round": round_number, "epsilon": 1.0 - 0.1 * round_number}
                for round_number in range(1, len(rewards) + 1)
            ],
        )


class TestSeedCurveAggregation(unittest.TestCase):
    """Validate equal-seed weighting and categorical-share conservation."""

    def test_reward_is_averaged_within_seed_before_across_seeds(self):
        root = tempfile.mkdtemp(prefix="plots61_")
        model = "dqn"
        model_one = Path(root) / "seed_1" / "offloading" / model
        model_two = Path(root) / "seed_2" / "offloading" / model
        _write_csv(
            model_one / f"{model}_reward_log.csv",
            ("round", "reward", "action", "priority"),
            [
                {"round": 1, "reward": 1.0, "action": 0, "priority": 1},
                {"round": 1, "reward": 3.0, "action": 1, "priority": 2},
            ],
        )
        _write_csv(
            model_two / f"{model}_reward_log.csv",
            ("round", "reward", "action", "priority"),
            [{"round": 1, "reward": 6.0, "action": 2, "priority": 3}],
        )
        curves = plots.load_reward_curves(root, model)
        aggregate = plots.aggregate_seed_curves(curves)
        self.assertAlmostEqual(aggregate.loc[1, "mean"], 4.0)
        self.assertAlmostEqual(aggregate.loc[1, "minimum"], 2.0)
        self.assertAlmostEqual(aggregate.loc[1, "maximum"], 6.0)

    def test_mean_action_rows_remain_probabilities(self):
        frames = {
            "seed_1": pd.DataFrame(
                {"round": [1, 2], "LOCAL_DEVICE": [1.0, 0.0], "MEC": [0.0, 1.0]}
            ),
            "seed_2": pd.DataFrame(
                {"round": [1, 2], "LOCAL_DEVICE": [0.0, 1.0], "MEC": [1.0, 0.0]}
            ),
        }
        mean_frame = plots.mean_action_frame(frames, moving_average_window=2)
        row_sums = mean_frame[["LOCAL_DEVICE", "MEC"]].sum(axis=1)
        self.assertTrue(all(abs(value - 1.0) < 1e-12 for value in row_sums))


class TestFigureGeneration(unittest.TestCase):
    """Validate dynamic seed discovery and the complete requested figure set."""

    def test_three_seed_two_model_campaign_writes_all_figures(self):
        root = tempfile.mkdtemp(prefix="plots61_")
        for seed_index in range(1, 4):
            _write_run(root, f"seed_{seed_index}", "dqn", [1.0, 2.0, 3.0])
            _write_run(
                root,
                f"seed_{seed_index}",
                "ppo",
                [2.0, 2.5, 3.5],
                include_epsilon=False,
            )

        written = plots.generate_training_figures(
            root,
            models=("dqn", "ppo"),
            moving_average_window=2,
            dpi=60,
        )
        expected = {
            Path(root) / "plots" / "by_model" / "dqn" / "training_action_by_class.png",
            Path(root) / "plots" / "by_model" / "dqn" / "training_epsilon.png",
            Path(root) / "plots" / "by_model" / "dqn" / "training_reward_by_class.png",
            Path(root) / "plots" / "by_model" / "dqn" / "training_average_reward.png",
            Path(root) / "plots" / "by_model" / "ppo" / "training_epsilon.png",
            Path(root) / "plots" / "comparison" / "all_models_main_reward.png",
            Path(root) / "plots" / "plot_manifest.json",
        }
        self.assertTrue(expected.issubset(set(written)))
        for path in expected:
            self.assertTrue(path.exists(), path)
            self.assertGreater(path.stat().st_size, 0)

        with open(Path(root) / "plots" / "plot_manifest.json", encoding="utf-8") as file_handle:
            manifest = json.load(file_handle)
        self.assertEqual(manifest["models"], ["dqn", "ppo"])
        self.assertEqual(manifest["moving_average_window_rounds"], 2)


class TestAutomaticPlotting(unittest.TestCase):
    """Validate campaign timing without running simulation workers."""

    def test_model_plot_waits_for_all_seeds_and_final_plot_waits_for_finalize(self):
        messages = []
        with patch.object(
            automatic_plotting, "generate_model_figures", return_value=[]
        ) as generate_model, patch.object(
            automatic_plotting, "generate_campaign_summary", return_value=[]
        ) as generate_summary:
            coordinator = automatic_plotting.CampaignPlotCoordinator(
                phase_dir="output/training",
                models=("dqn", "ppo"),
                seed_count=2,
                logger=messages.append,
            )
            coordinator.record_job_completion(
                {"model": "dqn", "seed_index": 0}
            )
            self.assertEqual(generate_model.call_count, 0)
            coordinator.record_job_completion(
                {"model": "dqn", "seed_index": 1}
            )
            coordinator.record_job_completion(
                {"model": "ppo", "seed_index": 0}
            )
            coordinator.finalize()

        generate_model.assert_called_once()
        self.assertEqual(generate_model.call_args.args[1], "dqn")
        generate_summary.assert_called_once()
        self.assertTrue(any("ppo has only 1/2" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
