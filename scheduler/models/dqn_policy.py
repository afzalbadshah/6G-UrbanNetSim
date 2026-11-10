import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scheduler.common.policy_utils import EpsilonScheduler, NoFeasibleActionError, ReplayBuffer, load_checkpoint, masked_random_action, save_checkpoint, to_tensor, valid_actions_from_mask
from scheduler.models.base_policy import BasePolicy


class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


class DQNPolicy(BasePolicy):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.net = DQNNetwork(state_dim, action_dim)
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
        self.learn_every = max(1, int(getattr(config, "LEARN_EVERY_N_DECISIONS", 1)))
        self._decisions_since_learn = 0

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
        self._decisions_since_learn += 1
        if self.epsilon_granularity != "round":
            self.epsilon = self.epsilon_scheduler.step()
        if len(self.memory) < self.batch_size:
            return None
        if self._decisions_since_learn < self.learn_every:
            return None
        self._decisions_since_learn = 0
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
                next_q = self.net(next_states)
                for row_idx, transition in enumerate(batch):
                    next_mask = transition.get("next_mask")
                    if next_mask is not None:
                        if not valid_actions_from_mask(next_mask, self.action_dim):
                            next_q[row_idx].zero_()
                            continue
                        mask_t = torch.as_tensor(next_mask, dtype=torch.float32)
                        if mask_t.numel() < self.action_dim:
                            mask_t = torch.cat([mask_t, torch.zeros(self.action_dim - mask_t.numel())])
                        mask_t = mask_t[: self.action_dim]
                        next_q[row_idx] = next_q[row_idx] + (mask_t - 1.0) * 1e9
                targets = rewards + (1.0 - done) * self.gamma * next_q.max(dim=1).values
        loss = self.criterion(current, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.item())

    def save(self, path):
        save_checkpoint(path, {"dqn": self.net.state_dict()})

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        data = load_checkpoint(path, map_location="cpu")
        self.net.load_state_dict(data.get("dqn", data) if isinstance(data, dict) else data)
