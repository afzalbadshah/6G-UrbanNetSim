from scheduler.common.policy_utils import masked_random_action
from scheduler.models.base_policy import BasePolicy


class RandomPolicy(BasePolicy):
    def select_action(self, state, mask=None):
        return masked_random_action(mask, self.action_dim)
