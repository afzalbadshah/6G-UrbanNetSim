"""Plot task allocation at decision grain for one training run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from scheduler.scheduler_helper.training_visualization_helper import (
    ACTION_LABEL_LIST,
    PRIORITY_LABELS,
    TIER_COLORS,
)


ACTION_FALLBACK_COLUMNS = (
    "selected_action",
    "rl_action",
    "action_final",
    "selected_tier",
    "allocated_tier",
    "execution_tier",
    "rl_tier_name",
    "offloading_tier",
)


def _read_available_columns(path: Path, requested: tuple[str, ...]) -> pd.DataFrame:
    header = pd.read_csv(path, sep=";", nrows=0)
    columns = [column for column in requested if column in header.columns]
    if not columns:
        return pd.DataFrame()
    return pd.read_csv(path, sep=";", usecols=columns)


def _normalise_action(value: object) -> str:
    text = str(value).strip().upper()
    replacements = {
        "LOCAL": "LOCAL_DEVICE",
        "DEVICE": "LOCAL_DEVICE",
        "LOCAL_DEVICE": "LOCAL_DEVICE",
        "MEC": "MEC",
        "EDGE": "MEC",
        "MEC_COLLAB": "MEC_COLLAB",
        "COLLAB": "MEC_COLLAB",
        "METRO": "METRO",
        "CLOUD": "CLOUD",
        "UAV": "UAV_BS",
        "UAV_BS": "UAV_BS",
    }
    return replacements.get(text, text)


def _choose_action_column(frame: pd.DataFrame) -> str:
    for column in ACTION_FALLBACK_COLUMNS:
        if column in frame.columns and frame[column].notna().any():
            return column
    raise ValueError("No action/tier column was found in the results CSV.")


def plot_decision_allocation(
    results_csv: str | Path,
    output_path: str | Path,
    rolling_decisions: int = 500,
    dpi: int = 180,
) -> Path:
    """Write a three-panel rolling action-share plot using individual decisions."""
    results_path = Path(results_csv)
    requested = (
        "round",
        "priority",
        "priority_class",
        "class_name",
        *ACTION_FALLBACK_COLUMNS,
    )
    data = _read_available_columns(results_path, requested)
    if data.empty:
        raise ValueError(f"No usable data was found in {results_path}")

    action_column = _choose_action_column(data)
    data = data.copy()
    data["action"] = data[action_column].map(_normalise_action)
    data = data[data["action"].isin(ACTION_LABEL_LIST)].copy()
    if data.empty:
        raise ValueError(f"No recognized action values were found in {results_path}")

    if "priority_class" in data.columns:
        data["service_class"] = data["priority_class"].astype(str)
    elif "class_name" in data.columns:
        data["service_class"] = data["class_name"].astype(str)
    elif "priority" in data.columns:
        priority_to_label = {int(priority): label for priority, label in PRIORITY_LABELS.items()}
        data["service_class"] = pd.to_numeric(data["priority"], errors="coerce").map(priority_to_label)
    else:
        raise ValueError("No service-class column was found in the results CSV.")

    data["service_class_norm"] = data["service_class"].astype(str).str.lower()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(15.6, 4.6), sharey=True)
    for axis, label in zip(axes, PRIORITY_LABELS.values()):
        class_data = data[data["service_class_norm"] == label.lower()].copy()
        if class_data.empty:
            axis.axis("off")
            axis.set_title(label)
            axis.text(0.5, 0.5, "No decisions", ha="center", va="center")
            continue

        class_data["decision_in_class"] = range(1, len(class_data) + 1)
        one_hot = pd.get_dummies(class_data["action"]).reindex(columns=ACTION_LABEL_LIST, fill_value=0)
        rolling = one_hot.rolling(
            window=max(1, int(rolling_decisions)),
            min_periods=1,
        ).mean()

        axis.stackplot(
            class_data["decision_in_class"],
            [rolling[action].to_numpy(dtype=float) for action in ACTION_LABEL_LIST],
            labels=ACTION_LABEL_LIST,
            colors=[TIER_COLORS.get(action, "#94a3b8") for action in ACTION_LABEL_LIST],
            alpha=0.9,
        )
        axis.set_title(f"{label} ({len(class_data):,} decisions)")
        axis.set_xlabel("Decision within class")
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, alpha=0.25)
        axis.set_axisbelow(True)

    axes[0].set_ylabel(f"Rolling action share ({int(rolling_decisions):,} decisions)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    figure.suptitle("DQN task allocation by training decision")
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--rolling-decisions", type=int, default=500)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    written = plot_decision_allocation(
        args.results_csv,
        args.output_path,
        rolling_decisions=args.rolling_decisions,
        dpi=args.dpi,
    )
    print(written)


if __name__ == "__main__":
    main()
