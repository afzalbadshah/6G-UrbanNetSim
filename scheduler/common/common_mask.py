"""Normalize the physical action-feasibility mask for every policy."""

from scheduler.common.common_action import get_action_dim


class CommonActionMask:
    """Return physical feasibility unchanged, including an all-zero mask."""

    def __init__(self, action_dim=None):
        self.action_dim = int(action_dim or get_action_dim())

    def build(self, context):
        context = context or {}
        base = [
            int(value)
            for value in context.get(
                "physical_mask",
                context.get("capacity_mask", []),
            )
        ]
        if len(base) < self.action_dim:
            base.extend([0] * (self.action_dim - len(base)))
        return base[: self.action_dim]
