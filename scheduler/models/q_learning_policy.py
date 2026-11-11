import os
import pickle
from collections import defaultdict

import numpy as np

from scheduler.common.policy_utils import EpsilonScheduler, masked_argmax, masked_random_action, valid_actions_from_mask
from scheduler.models.base_policy import BasePolicy


class QLearningPolicy(BasePolicy):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.q = defaultdict(lambda: np.zeros(self.action_dim, dtype=np.float32))
        self.alpha = float(getattr(config, "LEARNING_RATE", 0.001))
        self.gamma = float(getattr(config, "DISCOUNT_FACTOR", 0.95))
        self.epsilon_scheduler = EpsilonScheduler(
            start=getattr(config, "EPSILON_START", 1.0),
            minimum=getattr(config, "EPSILON_MIN", 0.05),
            decay=getattr(config, "EPSILON_DECAY", 0.995),
            decay_steps=getattr(config, "EPSILON_DECAY_STEPS", 0),
        )
        self.epsilon = self.epsilon_scheduler.value
        self.epsilon_granularity = str(getattr(config, "EPSILON_DECAY_GRANULARITY", "decision")).lower()
        # --- Phase 0 instrumentation (no behavior change): generalization
        # diagnostics. A "hit" means the greedy state key was already present
        # in the Q-table from a prior decision; a "miss" means it was seen for
        # the first time (so masked_argmax runs over a freshly-zeroed vector and
        # the policy has learned nothing usable for that state). A near-zero
        # hit rate is direct evidence the state key never generalizes.
        self._cache_hits = 0
        self._cache_misses = 0
        # Number of buckets each [0,1]-normalised continuous feature is coarsened
        # to when forming the tabular state key (Phase 2 generalisation fix).
        self._state_bins = max(2, int(getattr(config, "QLEARNING_STATE_BINS", 4)))

    def _lookup(self, key, record):
        # Membership test BEFORE indexing: self.q is a defaultdict, so the
        # indexing that follows would itself create the (all-zero) entry.
        if record:
            if key in self.q:
                self._cache_hits += 1
            else:
                self._cache_misses += 1
        return self.q[key]

    def get_diagnostics(self):
        looked_up = self._cache_hits + self._cache_misses
        return {
            "policy": "q_learning",
            "q_table_size": len(self.q),
            "greedy_lookups": looked_up,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": (self._cache_hits / looked_up) if looked_up else 0.0,
        }

    def set_train_mode(self, is_training):
        super().set_train_mode(is_training)
        self.epsilon = self.epsilon_scheduler.reset(self.train_mode)

    def on_training_round(self, round_id):
        if self.train_mode and self.epsilon_granularity == "round":
            self.epsilon = self.epsilon_scheduler.set_step(round_id)

    @staticmethod
    def _bin(value, bins):
        # Map a [0,1]-normalised feature into one of `bins` integer buckets.
        v = float(value)
        if v <= 0.0:
            return 0
        if v >= 1.0:
            return bins - 1
        return min(bins - 1, int(v * bins))

    def _key(self, state):
        # Generalisation-friendly state key (Phase 2 fix).
        #
        # The previous key rounded all STATE_DIM (=82) continuous features to 2
        # decimals and hashed the whole vector, so nearly every decision hashed
        # to a unique bucket (measured train cache-hit rate: 0/1048) and the
        # Q-table never reused a value. Here we instead key on a small set of
        # decision-relevant, coarsely-binned features so that many similar
        # states collapse to the same bucket and Bellman updates accumulate.
        #
        # Layout comes from CommonStateBuilder.build(): a 22-feature task/
        # mobility prefix followed by action_dim blocks of 10 per-tier features.
        # Relevant prefix indices:  4=priority/3, 5=deadline/1000, 7=|signal|,
        # 8=local_compute_capable, 12=handover_risk, 18=battery_low,
        # 19/20/21=priority one-hot.  Per-tier block (base = 22 + 10*a):
        # +0=capacity_mask, +4=future_util, +5=congestion.
        s = np.asarray(state, dtype=np.float32)
        n = s.shape[0]
        bins = self._state_bins
        prefix = n - 10 * self.action_dim
        if prefix == 22 and n >= 22:
            if s[19] or s[20] or s[21]:
                priority = int(np.argmax(s[19:22])) + 1
            else:
                priority = int(round(float(s[4]) * 3.0))
            parts = [
                priority,
                int(round(float(s[8]))),          # local-compute capable
                int(round(float(s[18]))),         # device battery low
                self._bin(s[5], bins),            # deadline bucket
                self._bin(s[7], bins),            # link-quality (|signal|) bucket
                self._bin(s[12], bins),           # handover-risk bucket
            ]
            for a in range(self.action_dim):
                base = 22 + 10 * a
                feasible = int(round(float(s[base]))) if base < n else 0
                util_b = self._bin(s[base + 4], bins) if base + 4 < n else 0
                cong_b = self._bin(s[base + 5], bins) if base + 5 < n else 0
                parts.append((feasible, util_b, cong_b))
            return tuple(parts)
        # Layout-agnostic fallback: coarse-bin every feature (still far more
        # general than rounding to 2 decimals) if the expected layout changes.
        return tuple(self._bin(v, bins) for v in s.tolist())

    def select_action(self, state, mask=None):
        if self.train_mode and np.random.random() < self.epsilon:
            return masked_random_action(mask, self.action_dim)
        # record=True: this greedy lookup is the decision-time generalization
        # signal the Phase 0 diagnostics summarize.
        return masked_argmax(self._lookup(self._key(state), True), mask)

    def learn(self, transition):
        if not self.train_mode:
            return None
        state = self._key(transition["state"])
        action = int(transition["action"])
        reward = float(transition["reward"])
        next_state = transition.get("next_state")
        done = bool(transition.get("done", False))
        target = reward
        # Terminal transitions (each offloading decision) do not bootstrap, so
        # credit stays one-step and never chains across unrelated tasks.
        if next_state is not None and not done:
            next_values = self.q[self._key(next_state)]
            valid_next = valid_actions_from_mask(transition.get("next_mask"), self.action_dim)
            if valid_next:
                target += self.gamma * float(np.max(next_values[valid_next]))
        self.q[state][action] += self.alpha * (target - self.q[state][action])
        if self.epsilon_granularity != "round":
            self.epsilon = self.epsilon_scheduler.step()
        return None

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(dict(self.q), fh)

    def load(self, path):
        max_mb = float(os.environ.get("SMARTCITY_MAX_QTABLE_LOAD_MB", "50") or 50)
        size_mb = os.path.getsize(path) / (1024.0 * 1024.0)
        if max_mb > 0 and size_mb > max_mb:
            raise RuntimeError(
                f"Q-learning checkpoint is {size_mb:.1f} MB; skipping load. "
                "Set SMARTCITY_MAX_QTABLE_LOAD_MB=0 to force loading it."
            )
        with open(path, "rb") as fh:
            self.q.update(pickle.load(fh) or {})
