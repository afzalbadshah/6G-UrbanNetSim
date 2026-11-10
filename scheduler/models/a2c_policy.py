import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scheduler.common.policy_utils import NoFeasibleActionError, load_checkpoint, masked_argmax, save_checkpoint, to_tensor
from scheduler.models.base_policy import BasePolicy


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.policy = nn.Linear(128, action_dim)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        features = self.body(x)
        return self.policy(features), self.value(features).squeeze(-1)


def masked_logits(logits, masks):
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
    if masks is None:
        return logits
    mask_t = torch.as_tensor(masks, dtype=torch.float32, device=logits.device)
    if mask_t.dim() == 1:
        mask_t = mask_t.unsqueeze(0)
    if mask_t.size(-1) < logits.size(-1):
        pad_shape = list(mask_t.shape)
        pad_shape[-1] = logits.size(-1) - mask_t.size(-1)
        mask_t = torch.cat([mask_t, torch.zeros(*pad_shape, dtype=mask_t.dtype, device=mask_t.device)], dim=-1)
    mask_t = mask_t[..., : logits.size(-1)]
    empty = mask_t.sum(dim=-1, keepdim=True) <= 0.0
    if bool(empty.any()):
        raise NoFeasibleActionError("no physically feasible action")
    return logits.masked_fill(mask_t <= 0.0, -1e9)


class A2CPolicy(BasePolicy):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.net = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=float(getattr(config, "LEARNING_RATE", 0.001)))
        self.gamma = float(getattr(config, "DISCOUNT_FACTOR", 0.99))
        self.batch_size = int(getattr(config, "BATCH_SIZE", 32))
        self.value_weight = float(getattr(config, "A2C_VALUE_WEIGHT", 0.5))
        self.entropy_weight = float(getattr(config, "ENTROPY_WEIGHT", 0.01))
        # Shared actor-critic logit stabilizer (A2C and PPO). An L2 pull on the
        # raw policy logits bounds their magnitude and prevents the late-training
        # policy collapse both otherwise show on harder state->action mappings.
        self.logit_l2 = float(getattr(config, "PPO_LOGIT_L2", 1e-3))
        self.rollout = []
        # Actor-critic exploration must remain on-policy. Training actions are
        # sampled from the actor's categorical distribution, with entropy
        # regularization keeping that distribution exploratory. A separate
        # epsilon-random branch would give A2C/PPO actions whose behavior
        # probability does not match the log-probability used by their loss.
        self.epsilon = 0.0
        # --- Phase 0 instrumentation (no behavior change): count how many
        # gradient updates were actually applied vs. discarded because the loss
        # went non-finite (the `if not torch.isfinite(loss)` bailout). A high
        # discard rate means the network is being starved of updates and its
        # weights drift toward / stay at their divergent (often NaN) state.
        self._updates_applied = 0
        self._updates_discarded = 0

    def get_diagnostics(self):
        attempted = self._updates_applied + self._updates_discarded
        return {
            "policy": "a2c",
            "updates_applied": self._updates_applied,
            "updates_discarded": self._updates_discarded,
            "update_attempts": attempted,
            "discard_rate": (self._updates_discarded / attempted) if attempted else 0.0,
        }

    def set_train_mode(self, is_training):
        super().set_train_mode(is_training)
        self.epsilon = 0.0

    def on_training_round(self, round_id):
        return None

    def select_action(self, state, mask=None):
        state_t = to_tensor(state).float().unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(state_t)
            logits = torch.nan_to_num(masked_logits(logits, mask).squeeze(0), nan=0.0, posinf=0.0, neginf=-1e9)
            if not self.train_mode:
                return masked_argmax(logits.cpu().numpy(), mask)
            probs = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
            if float(probs.sum()) <= 0.0:
                return masked_argmax(logits.cpu().numpy(), mask)
            return int(torch.distributions.Categorical(probs=probs).sample().item())

    def _returns(self, rewards, dones=None):
        # Discounted returns that RESET at episode boundaries. A terminal
        # transition (done=True) starts a fresh return, so credit never leaks
        # across unrelated decisions batched together. With the offloading
        # environment marking every decision terminal this reduces to the
        # correct one-step target (return == immediate reward).
        values = []
        running = 0.0
        for idx in reversed(range(len(rewards))):
            if dones is not None and bool(dones[idx]):
                running = 0.0
            running = float(rewards[idx]) + self.gamma * running
            values.append(running)
        values.reverse()
        return torch.as_tensor(values, dtype=torch.float32)

    def learn(self, transition):
        if not self.train_mode:
            return None
        self.rollout.append(dict(transition))
        if len(self.rollout) < self.batch_size:
            return None
        batch = self.rollout
        self.rollout = []
        states = to_tensor([t["state"] for t in batch]).float()
        actions = torch.as_tensor([int(t["action"]) for t in batch], dtype=torch.long)
        masks = [t.get("mask") for t in batch]
        # Returns keep their true scale so the critic has a stable target; only
        # the ADVANTAGES are normalized (per-batch return normalization made the
        # value target move every batch and starved the critic of a fit).
        returns = self._returns(
            [float(t["reward"]) for t in batch],
            [bool(t.get("done", False)) for t in batch],
        )
        raw_logits, values = self.net(states)
        logits = masked_logits(raw_logits, masks)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.logits
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)
        loss = (
            -(selected_log_probs * advantages).mean()
            + self.value_weight * F.mse_loss(values, returns)
            - self.entropy_weight * dist.entropy().mean()
            + self.logit_l2 * raw_logits.pow(2).mean()
        )
        if not torch.isfinite(loss):
            self._updates_discarded += 1
            self.rollout = []
            return None
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        self._updates_applied += 1
        return float(loss.item())

    def save(self, path):
        save_checkpoint(path, {"actor_critic": self.net.state_dict()})

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        data = load_checkpoint(path, map_location="cpu")
        self.net.load_state_dict(data.get("actor_critic", data) if isinstance(data, dict) else data)
