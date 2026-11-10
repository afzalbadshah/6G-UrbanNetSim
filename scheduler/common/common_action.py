ACTION_LOCAL = 0
ACTION_MEC = 1
ACTION_MEC_COLLAB = 2
ACTION_METRO = 3
ACTION_CLOUD = 4
ACTION_UAV_BS = 5
# Satellite-IoT scenario: device -> LEO satellite -> ground gateway -> Cloud.
ACTION_SAT = 6

ACTION_LABELS = {
    ACTION_LOCAL: "LOCAL_DEVICE",
    ACTION_MEC: "MEC",
    ACTION_MEC_COLLAB: "MEC_COLLAB",
    ACTION_METRO: "METRO",
    ACTION_CLOUD: "CLOUD",
    ACTION_UAV_BS: "UAV_BS",
    ACTION_SAT: "SAT",
}

ACTION_ALIASES = {
    "LOCAL": "LOCAL_DEVICE",
    "LOCAL_DEVICE": "LOCAL_DEVICE",
    "DEVICE": "LOCAL_DEVICE",
    "MEC": "MEC",
    "BS": "MEC",
    "MEC_COLLAB": "MEC_COLLAB",
    "COLLAB": "MEC_COLLAB",
    "METRO": "METRO",
    "ME": "METRO",
    "CLOUD": "CLOUD",
    "CC": "CLOUD",
    "UAV_BS": "UAV_BS",
    "UAV": "UAV_BS",
    "UAVBS": "UAV_BS",
    "AIR": "UAV_BS",
    "SAT": "SAT",
    "LEO": "SAT",
    "LEO_SAT": "SAT",
    "SATELLITE": "SAT",
}

LABEL_TO_ACTION = {label: action for action, label in ACTION_LABELS.items()}
LABEL_TO_ACTION.update({alias: LABEL_TO_ACTION[target] for alias, target in ACTION_ALIASES.items()})


def normalize_tier_label(label):
    key = str(label or "").strip().upper()
    return ACTION_ALIASES.get(key, key)


def label_to_action(label):
    key = normalize_tier_label(label)
    if key not in LABEL_TO_ACTION:
        raise ValueError(f"Unknown SmartCity offloading action: {label}")
    return int(LABEL_TO_ACTION[key])


def action_to_label(action):
    return ACTION_LABELS.get(int(action), "NA")


def get_action_dim():
    return len(ACTION_LABELS)


def get_all_actions():
    return list(ACTION_LABELS.keys())
