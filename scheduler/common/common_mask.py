"""Keep the full action space open for every learned policy."""

from scheduler.common.common_action import get_action_dim


class CommonActionMask:
    """Availability belongs in observations, not policy action suppression."""

    def __init__(self, action_dim=None):
        self.action_dim = int(action_dim or get_action_dim())

    def build(self, context):
        return [1] * self.action_dim
