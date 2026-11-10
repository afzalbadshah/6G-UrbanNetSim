import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scheduler.common.policy_utils import EpsilonScheduler, NoFeasibleActionError, ReplayBuffer, load_checkpoint, masked_random_action, save_checkpoint, to_tensor, valid_actions_from_mask
from scheduler.models.base_policy import BasePolicy


class DoubleDQNNetwork(nn.Module):
    """Identical architecture to dqn_policy.DQNNetwork -- the comparison
    with plain DQN needs to isolate the target-network/Double-Q mechanism as
    the only variable, not also change network capacity."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


class DoubleDQNPolicy(BasePolicy):
    """Same as DQNPolicy (scheduler/models/dqn_policy.py), except the
    bootstrapped target is computed with a separate, periodically-synced
    target network, and action selection/evaluation are decoupled (Double
    DQN): the online network picks the next action, the target network
    scores it. Kept as a fully separate model rather than patched into
    DQNPolicy so the two remain independently selectable for a controlled
    before/after comparison under identical reward/environment/workload
    conditions.
    """

    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.net = DoubleDQNNetwork(state_dim, action_dim)
        self.target_net = DoubleDQNNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.net.parameters(), lr=float(getattr(config, "LEARNING_RATE", 0.001)))
        self.criterion = nn.MSELoss()
        self.gamma = float(getattr(config, "DISCOUNT_FACTOR", 0.95))
        self.memory = ReplayBuffer(int(getattr(config, "MEMORY_SIZE", 10000)))
        self.batch_size = int(getattr(config, "BATCH_SIZE", 32))
        self.epsilon_scheduler = EpsilonScheduler(
            start=getattr(config, "EPSILON_START", 1.0),
            minimum=getattr(config, "EPSILON_MIN", 0.05),
            decay=getattr(config, "EPSILON_DECAY", 0.995),
            decay_steps=getattr(config, "EPSILON_DECAY_STEPS", 0),
        )
        self.epsilon = self.epsilon_scheduler.value
        self.epsilon_granularity = str(getattr(config, "EPSILON_DECAY_GRANULARITY", "decision")).lower()
        # Same learn()-call throttling as DQNPolicy -- every transition is
        # still stored every call, only the gradient step is throttled.
        self.learn_every = max(1, int(getattr(config, "LEARN_EVERY_N_DECISIONS", 4)))
        self._learn_call_count = 0
        # Target-network sync cadence: counts actual gradient steps taken
        # (not raw learn() calls), since that's the quantity the classic
        # DQN update_frequency convention is defined against.
        self.target_sync_every = max(1, int(getattr(config, "TARGET_NETWORK_SYNC_EVERY", 250)))
        self._gradient_step_count = 0

    def set_train_mode(self, is_training):
        super().set_train_mode(is_training)
        self.epsilon = self.epsilon_scheduler.reset(self.train_mode)

    def on_training_round(self, round_id):
        if self.train_mode and self.epsilon_granularity == "round":
            self.epsilon = self.epsilon_scheduler.set_step(round_id)

    def select_action(self, state, mask=None):
        valid = valid_actions_from_mask(mask, self.action_dim)
        if not valid:
            raise NoFeasibleActionError("no physically feasible action")
        if self.train_mode and np.random.random() < self.epsilon:
            return masked_random_action(mask, self.action_dim)
        state_t = to_tensor(state).float().unsqueeze(0)
        with torch.no_grad():
            q_values = torch.nan_to_num(self.net(state_t).squeeze(0), nan=0.0, posinf=0.0, neginf=0.0)
            mask_t = torch.zeros(self.action_dim, dtype=torch.float32)
            mask_t[valid] = 1.0
            q_values = q_values + (mask_t - 1.0) * 1e9
        return int(torch.argmax(q_values).item())

    def learn(self, transition):
        if not self.train_mode:
            return None
        self.memory.append(dict(transition))
        self._learn_call_count += 1
        if self.epsilon_granularity != "round":
            self.epsilon = self.epsilon_scheduler.step()
        if len(self.memory) < self.batch_size:
            return None
        if self._learn_call_count % self.learn_every != 0:
            return None
        batch = self.memory.sample(self.batch_size)
        states = to_tensor([t["state"] for t in batch]).float()
        actions = torch.as_tensor([int(t["action"]) for t in batch], dtype=torch.long).unsqueeze(1)
        rewards = torch.as_tensor([float(t["reward"]) for t in batch], dtype=torch.float32)
        done = torch.as_tensor([float(bool(t.get("done", False))) for t in batch], dtype=torch.float32)
        current = self.net(states).gather(1, actions).squeeze(1)
        targets = rewards.clone()
        if any(t.get("next_state") is not None for t in batch):
            next_states = to_tensor([
                t.get("next_state") if t.get("next_state") is not None else t["state"]
                for t in batch
            ]).float()
            with torch.no_grad():
                # Double DQN: the ONLINE network selects the next action...
                next_q_online = self.net(next_states)
                bootstrap_allowed = torch.ones(len(batch), dtype=torch.float32)
                for row_idx, transition in enumerate(batch):
                    next_mask = transition.get("next_mask")
                    if next_mask is not None:
                        if not valid_actions_from_mask(next_mask, self.action_dim):
                            next_q_online[row_idx].zero_()
                            bootstrap_allowed[row_idx] = 0.0
                            continue
                        mask_t = torch.as_tensor(next_mask, dtype=torch.float32)
                        if mask_t.numel() < self.action_dim:
                            mask_t = torch.cat([mask_t, torch.zeros(self.action_dim - mask_t.numel())])
                        mask_t = mask_t[: self.action_dim]
                        next_q_online[row_idx] = next_q_online[row_idx] + (mask_t - 1.0) * 1e9
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                # ...but the TARGET network (a stale, periodically-synced
                # copy) evaluates how good that action actually is -- this
                # decoupling is what corrects the single-network overestimation
                # bias that plain DQN (dqn_policy.py) is subject to.
                next_q_target = self.target_net(next_states)
                next_q = next_q_target.gather(1, next_actions).squeeze(1)
                targets = rewards + (1.0 - done) * bootstrap_allowed * self.gamma * next_q
        loss = self.criterion(current, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        self._gradient_step_count += 1
        if self._gradient_step_count % self.target_sync_every == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        return float(loss.item())

    def save(self, path):
        save_checkpoint(path, {"double_dqn": self.net.state_dict()})

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        data = load_checkpoint(path, map_location="cpu")
        state = data.get("double_dqn", data) if isinstance(data, dict) else data
        self.net.load_state_dict(state)
        self.target_net.load_state_dict(state)
