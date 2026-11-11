from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scheduler.common.common_action import ACTION_LABELS


ACTION_LABEL_LIST = [ACTION_LABELS[i] for i in range(len(ACTION_LABELS))]
PRIORITY_LABELS = {1: "URLLC", 2: "mMTC", 3: "eMBB"}
TIER_COLORS = {
    "LOCAL_DEVICE": "#64748b",  # on-board computing: slate gray
    "MEC": "#2563eb",
    "MEC_COLLAB": "#7c3aed",
    "METRO": "#ea580c",
    "CLOUD": "#16a34a",
    "UAV_BS": "#dc2626",        # UAV base station: red
}


class TrainingVisualizationHelper:
    FIG_SIZE = (6.4, 4.2)

    def __init__(self, model_name, output_root="output/exploration", moving_average_window=10):
        self.model_name = str(model_name).strip().lower()
        self.output_dir = Path(output_root)
        self.moving_average_window = int(max(1, moving_average_window))
        self.reward_history = []
        self.priority_reward_history = []
        self.epsilon_history = []
        self.action_history = []
        self.loss_history = []

    def log_step(self, round_id, reward=None, action=None, epsilon=None, loss=None, priority=None):
        if reward is not None:
            self.reward_history.append({"round": int(round_id), "reward": float(reward)})
            if priority is not None:
                self.priority_reward_history.append({"round": int(round_id), "priority": int(priority), "reward": float(reward)})
        if action is not None:
            entry = {"round": int(round_id), "action": int(action)}
            if priority is not None:
                entry["priority"] = int(priority)
            self.action_history.append(entry)
        if epsilon is not None:
            self.epsilon_history.append({"round": int(round_id), "epsilon": float(epsilon)})
        if loss is not None:
            self.loss_history.append({"round": int(round_id), "loss": float(loss)})

    def _prepare_output_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _style_axis(ax):
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    @staticmethod
    def _save_note(path, title, note):
        fig, ax = plt.subplots(figsize=TrainingVisualizationHelper.FIG_SIZE)
        ax.axis("off")
        ax.set_title(title)
        ax.text(0.5, 0.5, note, ha="center", va="center", wrap=True)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def save_reward_curve(self):
        path = self.output_dir / "reward_curve.png"
        if not self.reward_history:
            self._save_note(path, "Reward progression", "No training reward data recorded.")
            return
        df = pd.DataFrame(self.reward_history).groupby("round", as_index=False)["reward"].mean()
        df["moving_average"] = df["reward"].rolling(self.moving_average_window, min_periods=1).mean()
        fig, ax = plt.subplots(figsize=self.FIG_SIZE)
        ax.plot(df["round"], df["reward"], marker="o", markersize=2, linewidth=0.8, label="Reward")
        ax.plot(df["round"], df["moving_average"], linewidth=1.2, label="Moving average")
        ax.set_xlabel("Training round")
        ax.set_ylabel("Reward")
        ax.set_title("Reward progression")
        ax.legend()
        self._style_axis(ax)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def save_epsilon_decay(self):
        path = self.output_dir / "epsilon_decay.png"
        if not self.epsilon_history:
            self._save_note(path, "Epsilon decay", f"{self.model_name} does not use epsilon exploration.")
            return
        df = pd.DataFrame(self.epsilon_history).groupby("round", as_index=False)["epsilon"].mean()
        fig, ax = plt.subplots(figsize=self.FIG_SIZE)
        ax.plot(df["round"], df["epsilon"], linewidth=1.0)
        ax.set_xlabel("Training round")
        ax.set_ylabel("Epsilon")
        ax.set_title("Epsilon decay")
        self._style_axis(ax)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _action_distribution_percentages(self, df):
        df = df.copy()
        df["action_label"] = df["action"].apply(lambda action: ACTION_LABELS.get(int(action), "NA"))
        counts = df.groupby(["round", "action_label"]).size().unstack(fill_value=0)
        percentages = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)
        # Smooth over a rolling window of rounds so the plot matches its
        # "Rolling action share" label -- raw per-round percentages are noisy
        # blocky steps early in a run when few devices are active yet (e.g.
        # round 2 with 2 decisions can only land on 0%/50%/100%). Rolling
        # mean is linear, so each already-normalized row (sums to 1) keeps
        # summing to 1 after smoothing -- no renormalization needed.
        return percentages.rolling(self.moving_average_window, min_periods=1).mean()

    def _plot_action_distribution(self, ax, percentages, title):
        values = [
            percentages[label].values if label in percentages.columns else np.zeros(len(percentages))
            for label in ACTION_LABEL_LIST
        ]
        colors = [TIER_COLORS.get(label) for label in ACTION_LABEL_LIST]
        ax.stackplot(percentages.index, values, labels=ACTION_LABEL_LIST, colors=colors)
        ax.set_xlabel("Training round")
        ax.set_ylabel("Rolling action share")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="lower right")
        self._style_axis(ax)

    def save_action_distribution(self):
        """
        2x2 grid: the collective (all-priorities) action distribution plus
        one subplot per priority class, so it's visible where each
        workload type's tasks actually land (e.g. does URLLC really stay
        off Cloud/Metro, does mMTC actually favor local/low-energy tiers)
        instead of only seeing the blended average across all three.
        """
        path = self.output_dir / "action_distribution.png"
        if not self.action_history:
            self._save_note(path, "Action distribution", "No training actions recorded.")
            return
        df = pd.DataFrame(self.action_history)
        fig, axes = plt.subplots(2, 2, figsize=(self.FIG_SIZE[0] * 2, self.FIG_SIZE[1] * 2))
        axes = axes.ravel()

        self._plot_action_distribution(
            axes[0], self._action_distribution_percentages(df), "Action distribution -- all priorities"
        )

        has_priority = "priority" in df.columns
        for ax, (priority, label) in zip(axes[1:], PRIORITY_LABELS.items()):
            subset = df[df["priority"] == priority] if has_priority else df.iloc[0:0]
            if subset.empty:
                ax.axis("off")
                ax.set_title(f"{label}: no data" if has_priority else f"{label}: priority not recorded")
                continue
            self._plot_action_distribution(
                ax, self._action_distribution_percentages(subset), f"Action distribution -- {label}"
            )

        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def save_priority_reward_curve(self):
        path = self.output_dir / "priority_reward_curve.png"
        if not self.priority_reward_history:
            self._save_note(path, "Priority reward progression", "No priority-specific reward data recorded.")
            return
        df = pd.DataFrame(self.priority_reward_history).groupby(["priority", "round"], as_index=False)["reward"].mean()
        df["moving_average"] = df.groupby("priority")["reward"].transform(lambda x: x.rolling(self.moving_average_window, min_periods=1).mean())
        fig, ax = plt.subplots(figsize=self.FIG_SIZE)
        for priority, label in PRIORITY_LABELS.items():
            entries = df[df["priority"] == priority]
            if entries.empty:
                continue
            ax.plot(entries["round"], entries["reward"], marker="o", markersize=2, linewidth=0.8, label=f"{label} reward")
            ax.plot(entries["round"], entries["moving_average"], linewidth=1.2, label=f"{label} MA")
        ax.set_xlabel("Training round")
        ax.set_ylabel("Reward")
        ax.set_title("Reward progression by priority class")
        ax.legend()
        self._style_axis(ax)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def save_all_plots(self):
        self._prepare_output_dir()
        self.save_reward_curve()
        self.save_priority_reward_curve()
        self.save_epsilon_decay()
        self.save_action_distribution()
