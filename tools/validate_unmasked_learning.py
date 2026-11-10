"""Independent acceptance checks for unmasked, class-led offloading learning.

Run focused checks with: python tools/validate_unmasked_learning.py
After production is ready: python tools/validate_unmasked_learning.py --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import config as cfg
from metrics.reward import RewardObservation, evaluate_reward
from scheduler.common.common_action import ACTION_LABELS
from scheduler.common.common_mask import CommonActionMask
from scheduler.common.common_servers import projected_utilization
from scheduler.common.common_state import CommonStateBuilder
from scheduler.common.comparison_runner import ComparisonRunner
from scheduler.common.model_factory import create_policy

MODELS = ("random", "q_learning", "dqn", "double_dqn", "ppo", "a2c")
LEARNERS = MODELS[1:]


def observation(priority=1, **changes):
    base = RewardObservation(
        priority=priority, deadline_ms=1000.0, complete_latency_ms=1.0,
        execution_failed=False, deadline_missed=False,
        device_battery_capacity_j=100.0, device_battery_remaining_j=80.0,
        device_battery_energy_j=0.00001, total_energy_j=0.001, cost=0.0,
        current_utilization_percent=10.0, projected_utilization_percent=20.0,
        resource_fit_ratio=0.5, tier_capacity_ratio=0.5,
        projected_server_battery_ratio=0.8, mobility_violation=False,
        shared_capacity=True,
    )
    return replace(base, **changes)


def reward(item):
    return evaluate_reward(item).reward


def snapshot(policy):
    if hasattr(policy, "q"):
        return {key: value.copy() for key, value in policy.q.items()}
    return {key: value.detach().clone() for key, value in policy.net.state_dict().items()}


def changed(policy, before):
    after = snapshot(policy)
    if before.keys() != after.keys():
        return True
    if hasattr(policy, "q"):
        return any(not np.array_equal(before[key], after[key]) for key in before)
    return any(not torch.equal(before[key], after[key]) for key in before)


def isolated_config(root):
    values = {name: getattr(cfg, name) for name in dir(cfg) if name.isupper()}
    values["CHECKPOINT_ROOT"] = str(root)
    return SimpleNamespace(**values)


class RewardAcceptance(unittest.TestCase):
    def test_class_objectives_are_pure(self):
        for priority, component in ((1, "delay"), (2, "device_battery"), (3, "compute")):
            with self.subTest(priority=priority):
                weights = cfg.PRIORITY_PROFILES[priority]["objective_weights"]
                self.assertEqual(weights[component], 1.0)
                self.assertTrue(all(value == 0.0 for key, value in weights.items() if key != component))

    def test_battery_state_does_not_change_reward(self):
        for priority in (1, 2, 3):
            baseline = observation(priority)
            for capacity, remaining, server in ((100, 0, 0), (100, 100, 1), (10000, 200, 0.1)):
                with self.subTest(priority=priority, capacity=capacity, remaining=remaining):
                    altered = replace(baseline, device_battery_capacity_j=capacity,
                                      device_battery_remaining_j=remaining,
                                      projected_server_battery_ratio=server)
                    self.assertAlmostEqual(reward(baseline), reward(altered), places=12)

    def test_urllc_lower_delay_wins_even_above_target(self):
        for low, high in ((0.1, 1), (10, 100), (200, 800)):
            with self.subTest(low=low, high=high):
                better = reward(observation(1, complete_latency_ms=low))
                worse = reward(observation(1, complete_latency_ms=high))
                self.assertGreater(better, worse)
                self.assertGreater(worse, 0)

    def test_mmtc_lower_incremental_energy_wins_without_saturation(self):
        for low, high in ((1e-6, 2e-6), (0.001, 0.01), (1, 10)):
            with self.subTest(low=low, high=high):
                self.assertGreater(reward(observation(2, device_battery_energy_j=low)),
                                   reward(observation(2, device_battery_energy_j=high)))

    def test_embb_more_useful_spare_resources_wins(self):
        base = observation(3)
        self.assertGreater(reward(replace(base, resource_fit_ratio=0.8)), reward(base))
        self.assertGreater(reward(replace(base, tier_capacity_ratio=0.8)), reward(base))

    def test_unrelated_metrics_do_not_change_pure_objectives(self):
        variants = {
            1: dict(device_battery_energy_j=10, resource_fit_ratio=0.1, tier_capacity_ratio=0.1),
            2: dict(complete_latency_ms=900, resource_fit_ratio=0.1, tier_capacity_ratio=0.1),
            3: dict(complete_latency_ms=900, device_battery_energy_j=10),
        }
        for priority, changes in variants.items():
            with self.subTest(priority=priority):
                base = observation(priority)
                self.assertAlmostEqual(reward(base), reward(replace(
                    base, **changes, cost=100, mobility_violation=True,
                    total_energy_j=100, device_propulsion_energy_j=50,
                    server_hover_energy_j=50)), places=12)

    def test_failure_is_negative_even_with_legacy_exemption(self):
        for priority in (1, 2, 3):
            with self.subTest(priority=priority):
                self.assertEqual(reward(observation(priority, execution_failed=True,
                                                   deadline_penalty_exempt=True)), -10.0)

    def test_nominal_overload_retained_before_score_clipping(self):
        for priority in (1, 2, 3):
            for nominal in (95, 100, 125, 250):
                with self.subTest(priority=priority, nominal=nominal):
                    base = observation(priority)
                    result = evaluate_reward(replace(base, selected_projected_utilization_percent=nominal))
                    self.assertEqual(result.selected_future_util, nominal)
                    deduction = reward(base) - result.reward
                    self.assertGreaterEqual(deduction, 0)
                    self.assertLessEqual(deduction, 2.0 + 1e-12)


class StateAcceptance(unittest.TestCase):
    def test_full_policy_mask_preserves_physical_observations(self):
        for physical in ([0] * 7, [0, 0, 0, 0, 0, 0, 1], [1] * 7):
            context = {"physical_mask": list(physical), "capacity_mask": list(physical)}
            with self.subTest(physical=physical):
                self.assertEqual(CommonActionMask(7).build(context), [1] * 7)
                self.assertEqual(context["physical_mask"], physical)

    def test_state_layout_and_absence_are_distinct_from_overload(self):
        task = SimpleNamespace(priority=1, local_compute_capable=False, zone_id="remote")
        state, context = CommonStateBuilder(cfg.STATE_DIM).build(task, [])
        self.assertEqual(state.shape, (26 + 10 * len(ACTION_LABELS),))
        self.assertTrue(np.isfinite(state).all())
        self.assertEqual(context["physical_mask"], [0] * 7)
        self.assertEqual(context["learning_mask"], [1] * 7)
        self.assertEqual(list(state[22:25]), [0, 0, 1])
        for action, present in enumerate(context["physical_mask"]):
            with self.subTest(action=action):
                self.assertEqual(state[26 + 10 * action], present)
                self.assertNotEqual(context["future_util"].get(action), 100.0)

    def test_nominal_utilization_is_not_lost_in_projection(self):
        server = SimpleNamespace(cpu_capacity=100, memory_capacity=100, storage_capacity=100,
                                 available_cpu=20, available_memory=100, available_storage=100)
        task = SimpleNamespace(cpu_demand=70, memory_demand=1, storage_demand=1)
        self.assertAlmostEqual(projected_utilization(server, task)["peak"], 150.0)

    def test_q_key_tracks_zone_utilization_and_congestion(self):
        policy = create_policy("q_learning", cfg.STATE_DIM, cfg.ACTION_DIM, cfg)
        base = np.zeros(cfg.STATE_DIM, dtype=np.float32)
        base[19] = 1
        base[22] = 1
        key = policy._key(base)
        self.assertLess(len(key), cfg.STATE_DIM, "96-dimensional fallback defeats the intended state abstraction")
        for index in (27, 28):
            altered = base.copy()
            altered[index] = 0.9
            with self.subTest(index=index):
                self.assertNotEqual(policy._key(altered), key)
        remote = base.copy()
        remote[22] = 0
        remote[24] = 1
        self.assertNotEqual(policy._key(remote), key)


class RunnerAcceptance(unittest.TestCase):
    @staticmethod
    def context(physical):
        return {
            "physical_mask": list(physical), "capacity_mask": list(physical),
            "learning_mask": [1] * 7, "mask": [1] * 7,
            "deadline_ms": 1000, "action_servers": {},
            "current_util": {a: 0.0 for a in range(7)},
            "future_util": {a: 0.0 for a in range(7)},
            "objective_delay": {a: 1.0 for a in range(7)},
            "observed_complete_delay": {a: 1.0 for a in range(7)},
            "resource_fit": {a: 0.8 for a in range(7)},
            "tier_capacity_ratio": {a: 0.8 for a in range(7)},
        }

    def runner(self, context, selected):
        runner = ComparisonRunner("random", config=cfg)
        runner.set_train_mode(True)
        runner.policy = SimpleNamespace(select_action=Mock(return_value=selected),
                                        learn=Mock(return_value=None), epsilon=0.0,
                                        on_training_round=Mock())
        runner.state_builder = SimpleNamespace(build=Mock(return_value=(np.zeros(cfg.STATE_DIM), context)))
        runner._append_row = Mock()
        runner._row = Mock(return_value={})
        return runner

    @staticmethod
    def task():
        task = SimpleNamespace(priority=1, device_id=1, id=1, battery_remaining_j=80,
                               battery_capacity_j=100, local_compute_capable=False, status="pending")
        task.fail = lambda: setattr(task, "status", "failed")
        task.complete = lambda **kwargs: setattr(task, "status", "completed")
        return task

    def assert_transition(self, runner, action, negative):
        runner.policy.select_action.assert_called_once()
        runner.policy.learn.assert_called_once()
        item = runner.policy.learn.call_args.args[0]
        self.assertEqual(item["action"], action)
        self.assertTrue(item["done"])
        self.assertEqual(list(item["mask"]), [1] * 7)
        self.assertEqual(item["state"].shape, (cfg.STATE_DIM,))
        self.assertTrue(math.isfinite(item["reward"]))
        if negative:
            self.assertLess(item["reward"], 0)

    def test_unavailable_attempts_each_learn_once(self):
        for action in range(cfg.ACTION_DIM):
            with self.subTest(action=action):
                context = self.context([0] * 7)
                runner = self.runner(context, action)
                runner.schedule([self.task()], [], 0)
                self.assert_transition(runner, action, negative=True)
                self.assertIsNone(context.get("executed_action"))

    def test_admission_failure_learns_once(self):
        context = self.context([0, 1, 0, 0, 0, 0, 0])
        server = SimpleNamespace(name="validation_mec")
        context["action_servers"][1] = server
        runner = self.runner(context, 1)
        runner._allocate = Mock(return_value=False)
        runner.schedule([self.task()], [server], 0)
        self.assert_transition(runner, 1, negative=True)

    def test_selected_action_cannot_spill_into_another_tier(self):
        context = self.context([0, 1, 0, 0, 1, 0, 0])
        context["future_util"][1] = 150
        context["action_servers"] = {1: SimpleNamespace(name="MEC"), 4: SimpleNamespace(name="Cloud")}
        runner = self.runner(context, 1)
        action, _, _ = runner._select_allocatable_action(1, context, self.task())
        self.assertIn(action, (None, 1))

    def test_single_physical_choice_keeps_reward_and_zone_has_no_term(self):
        context = self.context([0, 1, 0, 0, 0, 0, 0])
        task = self.task()
        expected = ComparisonRunner._evaluate_reward(task, 1, context)["reward"]
        self.assertGreater(expected, 0)
        for zone in ("cellular", "uav", "remote"):
            with self.subTest(zone=zone):
                task.zone_id = zone
                context["forced_single_action"] = True
                self.assertEqual(ComparisonRunner._evaluate_reward(task, 1, context)["reward"], expected)


class PolicyAcceptance(unittest.TestCase):
    def test_actor_critic_exploration_is_on_policy(self):
        for model in ("a2c", "ppo"):
            with self.subTest(model=model):
                policy = create_policy(model, cfg.STATE_DIM, cfg.ACTION_DIM, cfg)
                policy.set_train_mode(True)
                policy.on_training_round(100)
                self.assertEqual(policy.epsilon, 0.0)
                self.assertFalse(hasattr(policy, "epsilon_scheduler"))

    def test_failed_terminal_actions_update_each_learning_policy(self):
        torch.set_num_threads(1)
        for model in LEARNERS:
            with self.subTest(model=model):
                policy = create_policy(model, cfg.STATE_DIM, cfg.ACTION_DIM, cfg)
                policy.set_train_mode(True)
                before = snapshot(policy)
                for index in range(64):
                    state = np.zeros(cfg.STATE_DIM, dtype=np.float32)
                    state[19 + index % 3] = 1
                    state[22 + index % 3] = 1
                    transition = dict(state=state, action=index % cfg.ACTION_DIM,
                                      reward=-10.0, next_state=state.copy(), done=True,
                                      mask=[1] * cfg.ACTION_DIM, next_mask=[1] * cfg.ACTION_DIM)
                    loss = policy.learn(transition)
                    if loss is not None:
                        self.assertTrue(math.isfinite(float(loss)))
                self.assertTrue(changed(policy, before), f"{model} did not update on failed actions")
                self.assertEqual(getattr(policy, "_updates_discarded", 0), 0)

    def test_runner_rejects_legacy_checkpoint_location_and_roundtrips_fresh(self):
        with tempfile.TemporaryDirectory(prefix="unmasked_checkpoint_", dir=ROOT / "output") as folder:
            for model in LEARNERS:
                with self.subTest(model=model):
                    config = isolated_config(folder)
                    runner = ComparisonRunner(model, config=config)
                    policy = runner.policy
                    path = Path(folder) / f"{model}_comparison_policy.pt"
                    if model == "q_learning":
                        with path.open("wb") as handle:
                            pickle.dump({}, handle)
                    else:
                        torch.save(policy.net.state_dict(), path)
                    with self.assertRaises(FileNotFoundError):
                        runner.load()
                    runner.save()
                    versioned = Path(folder) / cfg.REWARD_SCHEMA_VERSION / path.name
                    self.assertTrue(versioned.is_file())
                    loaded = ComparisonRunner(model, config=config)
                    loaded.load()
                    self.assertFalse(changed(loaded.policy, snapshot(policy)))


def attach_audit(runner, training):
    audit = {"attempts": 0, "learn_calls": 0, "rows": 0, "zones": {}, "failed": 0,
             "unavailable": 0, "errors": []}
    original_select = runner.policy.select_action
    original_learn = runner.policy.learn
    original_row = runner._row
    pending = []

    def select(state, mask=None):
        if state.shape != (cfg.STATE_DIM,) or not np.isfinite(state).all():
            audit["errors"].append("invalid state shape or nonfinite state")
        if list(mask) != [1] * cfg.ACTION_DIM:
            audit["errors"].append("policy received restricted action mask")
        action = original_select(state, mask)
        audit["attempts"] += 1
        pending.append(int(action))
        return action

    def learn(transition):
        audit["learn_calls"] += 1
        if not pending or int(transition["action"]) != pending.pop(0):
            audit["errors"].append("learn did not match selected action exactly once")
        if not transition.get("done"):
            audit["errors"].append("nonterminal independent task transition")
        value = original_learn(transition)
        if value is not None and not math.isfinite(float(value)):
            audit["errors"].append("nonfinite loss")
        return value

    def row(task, selected, executed, server, route, context, mask, reward_value, status, reward_info=None):
        audit["rows"] += 1
        zone = str(getattr(task, "zone_id", ""))
        audit["zones"][zone] = audit["zones"].get(zone, 0) + 1
        if executed is not None and executed != selected:
            audit["errors"].append("selected and executed action differ")
        if status == "failed":
            audit["failed"] += 1
            if reward_value >= 0:
                audit["errors"].append("failed attempt has nonnegative reward")
        if executed is None:
            audit["unavailable"] += 1
            if float((reward_info or {}).get("overutilization_pen", 0)) != 0:
                audit["errors"].append("unavailable action charged an overload penalty")
        return original_row(task, selected, executed, server, route, context, mask,
                            reward_value, status, reward_info=reward_info)

    runner.policy.select_action = select
    runner.policy.learn = learn
    runner._row = row
    return audit


def smoke(args):
    from core.simulator import Simulator

    if not (2 <= args.rounds <= 200 and 8 <= args.devices <= 64):
        raise ValueError("bounded validation requires 2..200 rounds and 8..64 devices")
    if args.rounds * args.devices > 12_800:
        raise ValueError("bounded validation permits at most 12,800 decisions per phase")
    models = tuple(args.models or MODELS)
    unknown = sorted(set(models).difference(MODELS))
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    if not hasattr(cfg, "REWARD_SCHEMA_VERSION"):
        raise RuntimeError("production checkpoint schema guard is not ready")
    torch.set_num_threads(1)
    root = ROOT / "output" / (
        "validation_unmasked_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_" + uuid4().hex[:8]
    )
    root.mkdir(exist_ok=False)
    report = {"root": str(root), "schema": cfg.REWARD_SCHEMA_VERSION,
              "train_seed": args.seed, "eval_seed": args.seed + 1,
              "rounds": args.rounds, "devices": args.devices,
              "models": {}, "scope": "bounded integration/convergence validation"}
    original_cwd = Path.cwd()
    try:
        for model in models:
            home = root / model
            home.mkdir(exist_ok=False)
            os.chdir(home)
            checkpoints = home / "checkpoints"
            checkpoints.mkdir(exist_ok=False)
            trained_snapshot = None
            for training in (True, False):
                phase = "train" if training else "heldout"
                seed = args.seed if training else args.seed + 1
                cfg.RANDOM_SEED = seed
                workload = cfg.apply_workload_profile(train_mode=training, overrides={
                    "start_devices": args.devices, "max_devices": args.devices,
                    "increment": 0, "rounds": args.rounds,
                })
                config = isolated_config(checkpoints)
                output = home / phase
                output.mkdir(exist_ok=False)
                runner = ComparisonRunner(model, config=config, output_dir=output, seed=seed)
                if not training:
                    runner.load()
                    if model in LEARNERS and changed(runner.policy, trained_snapshot):
                        raise AssertionError(f"{model}: loaded checkpoint differs from trained weights")
                runner.set_train_mode(training)
                before = snapshot(runner.policy) if model in LEARNERS else None
                audit = attach_audit(runner, training)
                sim = Simulator(scheduler=runner, output_dir=output, workload=workload)
                # Handover historically uses an absolute repository checkpoint
                # directory; explicitly redirect it before run() can save/load.
                handover = sim.mobility.handover_scheduler
                handover.model_dir = str(checkpoints / "mobility")
                Path(handover.model_dir).mkdir(exist_ok=True)
                print(f"[VALIDATE] {model} {phase} seed={seed} output={output}", flush=True)
                try:
                    sim.run(train_mode=training)
                finally:
                    runner._close_csv()
                expected = args.rounds * args.devices
                if audit["attempts"] != expected or audit["rows"] != expected:
                    audit["errors"].append(f"expected {expected} decisions and rows")
                if audit["learn_calls"] != (expected if training else 0):
                    audit["errors"].append("incorrect learn calls for phase")
                if set(audit["zones"]) != {"cellular", "uav", "remote"}:
                    audit["errors"].append("all three zones were not observed")
                if training and model in LEARNERS:
                    audit["weights_updated"] = changed(runner.policy, before)
                    if not audit["weights_updated"]:
                        audit["errors"].append("training did not update policy weights")
                    trained_snapshot = snapshot(runner.policy)
                if not training and model in LEARNERS:
                    after = snapshot(runner.policy)
                    equal = np.array_equal if model == "q_learning" else torch.equal
                    if any(not equal(value, after[key]) for key, value in before.items()):
                        audit["errors"].append("evaluation modified learned values")
                audit["errors"] = sorted(set(audit["errors"]))
                report["models"].setdefault(model, {})[phase] = audit
                (root / "validation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(f"[VALIDATE] {model} {phase}: {json.dumps(audit)}", flush=True)
    finally:
        os.chdir(original_cwd)
    errors = [error for phases in report["models"].values() for phase in phases.values() for error in phase["errors"]]
    print(f"[VALIDATE] summary={root / 'validation_summary.json'}", flush=True)
    if errors:
        raise AssertionError(f"smoke found {len(errors)} audit failures; see validation_summary.json")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--smoke", action="store_true")
        parser.add_argument("--rounds", type=int, default=8)
        parser.add_argument("--devices", type=int, default=24)
        parser.add_argument("--seed", type=int, default=47219)
        parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
        smoke(parser.parse_args())
    else:
        unittest.main(verbosity=2)
