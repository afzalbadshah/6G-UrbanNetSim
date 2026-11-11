import importlib.util
import unittest

from config.reward import PRIORITY_PROFILES, REWARD_SCALE
from metrics.reward import (
    RewardObservation,
    evaluate_reward,
    marginal_shared_capacity_pressure,
    overutilization_pressure,
)


def observation(**changes):
    values = {
        "priority": 3,
        "deadline_ms": 1_000.0,
        "complete_latency_ms": 100.0,
        "execution_failed": False,
        "deadline_missed": False,
        "device_battery_capacity_j": 10_000.0,
        "device_battery_remaining_j": 10_000.0,
        "device_battery_energy_j": 0.1,
        "total_energy_j": 1.0,
        "cost": 0.0,
        "current_utilization_percent": 20.0,
        "projected_utilization_percent": 30.0,
        "resource_fit_ratio": 1.0,
        "tier_capacity_ratio": 1.0,
        "projected_server_battery_ratio": 1.0,
        "mobility_violation": False,
        "shared_capacity": True,
    }
    values.update(changes)
    return RewardObservation(**values)


class RewardMetricTests(unittest.TestCase):
    def test_completed_after_deadline_uses_complete_delay_and_failure_floor(self):
        result = evaluate_reward(
            observation(
                complete_latency_ms=90_000.0,
                deadline_missed=True,
            )
        )

        self.assertEqual(result.delay_ms, 90_000.0)
        self.assertEqual(result.deadline_success, 0.0)
        self.assertEqual(result.reward, -REWARD_SCALE)

    def test_completion_within_deadline_is_not_forced_to_failure_floor(self):
        result = evaluate_reward(observation())

        self.assertEqual(result.deadline_success, 1.0)
        self.assertGreater(result.reward, -REWARD_SCALE)

    def test_capacity_pressure_is_continuous_across_former_50_percent_knee(self):
        p49 = marginal_shared_capacity_pressure(40.0, 49.0)
        p50 = marginal_shared_capacity_pressure(40.0, 50.0)
        p51 = marginal_shared_capacity_pressure(40.0, 51.0)

        self.assertLess(p49, p50)
        self.assertLess(p50, p51)
        self.assertLess(abs((p51 - p50) - (p50 - p49)), 0.001)

    def test_embb_prefers_lower_marginal_pressure_when_other_outcomes_match(self):
        lower = evaluate_reward(
            observation(
                current_utilization_percent=10.0,
                projected_utilization_percent=20.0,
            )
        )
        higher = evaluate_reward(
            observation(
                current_utilization_percent=70.0,
                projected_utilization_percent=80.0,
            )
        )

        self.assertGreater(lower.reward, higher.reward)
        self.assertLess(
            lower.shared_capacity_externality_pen,
            higher.shared_capacity_externality_pen,
        )

    def test_every_class_is_penalized_only_above_spillover_threshold(self):
        self.assertEqual(overutilization_pressure(95.0), 0.0)
        self.assertGreater(overutilization_pressure(96.0), 0.0)
        self.assertEqual(overutilization_pressure(100.0), 1.0)

        for priority in (1, 2, 3):
            below = evaluate_reward(
                observation(
                    priority=priority,
                    selected_projected_utilization_percent=94.0,
                )
            )
            above = evaluate_reward(
                observation(
                    priority=priority,
                    selected_projected_utilization_percent=99.0,
                )
            )
            self.assertEqual(below.overutilization_pen, 0.0)
            self.assertGreater(above.overutilization_pen, 0.0)
            self.assertGreater(below.reward, above.reward)

    def test_mmtc_can_prefer_mec_energy_savings_despite_more_delay(self):
        mec = evaluate_reward(
            observation(
                priority=2,
                deadline_ms=600.0,
                complete_latency_ms=350.0,
                device_battery_energy_j=0.000015,
                total_energy_j=0.02,
                cost=0.00000004,
                shared_capacity=True,
            )
        )
        cloud = evaluate_reward(
            observation(
                priority=2,
                deadline_ms=600.0,
                complete_latency_ms=250.0,
                device_battery_energy_j=0.00002,
                total_energy_j=0.02,
                cost=0.00000014,
                shared_capacity=True,
            )
        )

        self.assertGreater(mec.delay_ms, cloud.delay_ms)
        self.assertGreater(mec.reward, cloud.reward)
        self.assertEqual(mec.cost_normalizer, 0.000001)
        self.assertEqual(mec.device_energy_normalizer_j, 0.00002)

    def test_delay_target_shapes_preference_without_deadline_failure(self):
        fast = evaluate_reward(
            observation(
                priority=1,
                deadline_ms=100.0,
                delay_target_ms=1.0,
                complete_latency_ms=1.0,
            )
        )
        slow = evaluate_reward(
            observation(
                priority=1,
                deadline_ms=100.0,
                delay_target_ms=1.0,
                complete_latency_ms=1.4,
            )
        )

        self.assertEqual(fast.deadline_success, 1.0)
        self.assertEqual(slow.deadline_success, 1.0)
        self.assertGreater(fast.delay_score, slow.delay_score)
        self.assertGreater(fast.reward, slow.reward)

    def test_deadline_safety_remains_separate_from_delay_target(self):
        inside_deadline = evaluate_reward(
            observation(
                priority=1,
                deadline_ms=100.0,
                delay_target_ms=1.0,
                complete_latency_ms=2.0,
            )
        )
        missed_deadline = evaluate_reward(
            observation(
                priority=1,
                deadline_ms=100.0,
                delay_target_ms=1.0,
                complete_latency_ms=150.0,
            )
        )

        self.assertEqual(inside_deadline.deadline_success, 1.0)
        self.assertGreater(inside_deadline.reward, -REWARD_SCALE)
        self.assertEqual(missed_deadline.deadline_success, 0.0)
        self.assertEqual(missed_deadline.reward, -REWARD_SCALE)

    def test_reward_profiles_contain_objectives_not_tier_policy(self):
        forbidden = {
            "tier_order",
            "allowed_tiers",
            "avoid_tiers",
            "tier_thresholds",
            "deadline_weight",
            "energy_weight",
            "cost_weight",
        }
        required_weights = {
            "delay",
            "device_battery",
            "server_battery",
            "compute",
            "cost",
            "mobility",
            "shared_capacity_externality",
        }
        for profile in PRIORITY_PROFILES.values():
            self.assertTrue(forbidden.isdisjoint(profile))
            self.assertEqual(set(profile["objective_weights"]), required_weights)

    def test_profile_weights_are_shares_of_the_reward_scale(self):
        # Weights sum to 1.0 per class so each is a share of the reward scale.
        # OVERUTILIZATION_PENALTY_WEIGHT is applied on that same scale, so a
        # weight sum far from 1.0 would silently rescale it.
        for priority, profile in PRIORITY_PROFILES.items():
            with self.subTest(priority=priority):
                total = sum(abs(w) for w in profile["objective_weights"].values())
                self.assertAlmostEqual(total, 1.0)

    def test_on_time_delivery_never_scores_as_failure(self):
        # Failure is the strict floor: even at the deadline, with the maximum
        # over-utilization penalty, a delivered task must score above it.
        for priority, deadline_ms in ((1, 100.0), (2, 300.0), (3, 500.0)):
            for delay_ms in (1.0, 0.5 * deadline_ms, deadline_ms):
                with self.subTest(priority=priority, delay_ms=delay_ms):
                    result = evaluate_reward(
                        observation(
                            priority=priority,
                            deadline_ms=deadline_ms,
                            complete_latency_ms=delay_ms,
                            selected_projected_utilization_percent=100.0,
                        )
                    )
                    self.assertEqual(result.deadline_success, 1.0)
                    self.assertGreater(result.reward, -REWARD_SCALE)

    def test_urllc_separates_sub_millisecond_delay_differences(self):
        # Measured URLLC tiers deliver within ~1-2.4 ms end to end.  Scored
        # against the deadline this difference is ~0.02 reward points.
        fast = evaluate_reward(
            observation(priority=1, deadline_ms=100.0, complete_latency_ms=1.0)
        )
        slow = evaluate_reward(
            observation(priority=1, deadline_ms=100.0, complete_latency_ms=1.3)
        )
        self.assertGreater(fast.reward - slow.reward, 0.5)

    def test_runner_scores_delay_against_reward_target_not_radio_target(self):
        from types import SimpleNamespace

        from scheduler.common.common_action import ACTION_LOCAL
        from scheduler.common.comparison_runner import ComparisonRunner

        # A 0.1 ms radio target divides every end-to-end delay into the score
        # floor; the reward must use its own end-to-end target instead.
        task = SimpleNamespace(
            priority=1, deadline_ms=100.0, radio_latency_target_ms=0.1
        )
        context = {"deadline_ms": 100.0, "objective_delay": {ACTION_LOCAL: 1.0}}
        result = ComparisonRunner._evaluate_reward(task, ACTION_LOCAL, context)
        self.assertEqual(
            result["delay_target_ms"], PRIORITY_PROFILES[1]["delay_target_ms"]
        )

    def test_old_scheduler_reward_module_is_deleted(self):
        self.assertIsNone(importlib.util.find_spec("scheduler.common.common_reward"))


if __name__ == "__main__":
    unittest.main()
