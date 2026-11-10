from scheduler.common.policy_utils import masked_random_action
from scheduler.models.base_policy import BasePolicy


class RuleBasedPolicy(BasePolicy):
    """Compatibility baseline that samples only physically feasible actions.

    It contains no service-class ladder or utilization threshold. Experiments
    needing a hand-authored heuristic must implement and label it separately.
    """
    def select_action(self, state, mask=None):
        return masked_random_action(mask, self.action_dim)
