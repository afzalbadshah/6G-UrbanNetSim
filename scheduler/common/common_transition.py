def build_transition(state, action, reward, next_state=None, done=False, mask=None, next_mask=None, task=None, selected_tier=None, metrics=None):
    return {
        "state": state,
        "action": int(action),
        "reward": float(reward),
        "next_state": next_state,
        "done": bool(done),
        "mask": mask,
        "next_mask": next_mask,
        "task_id": getattr(task, "id", ""),
        "device_id": int(getattr(task, "device_id", getattr(task, "entity_id", -1)) or -1),
        "device_type": getattr(task, "device_type", "generic"),
        "priority": int(getattr(task, "priority", 2) or 2),
        "selected_tier": selected_tier,
        "metrics": dict(metrics or {}),
    }
