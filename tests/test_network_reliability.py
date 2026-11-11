import unittest

from network.reliability import CounterBasedSampler, RetryPolicy


class CounterBasedSamplerTests(unittest.TestCase):
    def test_same_key_is_reproducible_and_order_independent(self):
        sampler = CounterBasedSampler(experiment_seed=20260915)
        first = sampler.sample(
            0.5,
            purpose="per",
            unit_id="packet-7",
            hop_id="hop-2",
            attempt_number=3,
        )
        sampler.sample(
            0.5,
            purpose="per",
            unit_id="unrelated",
            hop_id="hop-x",
            attempt_number=1,
        )
        repeated = sampler.sample(
            0.5,
            purpose="per",
            unit_id="packet-7",
            hop_id="hop-2",
            attempt_number=3,
        )

        self.assertEqual(first, repeated)
        self.assertEqual(first.uniform_value, 0.9086746744560832)
        self.assertGreaterEqual(first.uniform_value, 0.0)
        self.assertLess(first.uniform_value, 1.0)

    def test_purpose_and_attempt_are_separate_random_streams(self):
        sampler = CounterBasedSampler(experiment_seed=9)
        per = sampler.sample(0.5, purpose="per", unit_id="p", hop_id="h", attempt_number=1)
        los = sampler.sample(0.5, purpose="los", unit_id="p", hop_id="h", attempt_number=1)
        retry = sampler.sample(0.5, purpose="per", unit_id="p", hop_id="h", attempt_number=2)

        self.assertNotEqual(per.uniform_value, los.uniform_value)
        self.assertNotEqual(per.uniform_value, retry.uniform_value)

    def test_probability_boundaries_are_exact(self):
        sampler = CounterBasedSampler(experiment_seed="experiment-a")
        never = sampler.sample(0.0, purpose="per", unit_id="p", hop_id="h", attempt_number=1)
        always = sampler.sample(1.0, purpose="per", unit_id="p", hop_id="h", attempt_number=1)

        self.assertFalse(never.occurred)
        self.assertTrue(always.occurred)
        self.assertEqual(never.uniform_value, always.uniform_value)

    def test_invalid_sample_inputs_are_rejected(self):
        sampler = CounterBasedSampler(experiment_seed=1)
        for probability in (-0.1, 1.1):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                sampler.sample(
                    probability,
                    purpose="per",
                    unit_id="p",
                    hop_id="h",
                    attempt_number=1,
                )
        with self.assertRaises(ValueError):
            sampler.sample(0.5, purpose="", unit_id="p", hop_id="h", attempt_number=1)
        with self.assertRaises(ValueError):
            sampler.sample(0.5, purpose="per", unit_id="p", hop_id="h", attempt_number=0)

    def test_seed_validation(self):
        for seed in (-1, True, ""):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                CounterBasedSampler(experiment_seed=seed)


class RetryPolicyTests(unittest.TestCase):
    def test_next_attempt_uses_fixed_backoff_until_limit(self):
        policy = RetryPolicy(max_attempts=3, backoff_s=0.25, outage_observation_s=0.01)

        self.assertAlmostEqual(policy.next_attempt_at_s(1.0, completed_attempts=1), 1.25)
        self.assertAlmostEqual(policy.next_attempt_at_s(1.0, completed_attempts=2), 1.25)
        self.assertIsNone(policy.next_attempt_at_s(1.0, completed_attempts=3))

    def test_policy_validation(self):
        with self.assertRaises(ValueError):
            RetryPolicy(max_attempts=0)
        with self.assertRaises(ValueError):
            RetryPolicy(backoff_s=-0.1)
        with self.assertRaises(ValueError):
            RetryPolicy(outage_observation_s=-0.1)


if __name__ == "__main__":
    unittest.main()
