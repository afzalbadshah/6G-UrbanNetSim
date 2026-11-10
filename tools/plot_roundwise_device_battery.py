from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "visualization" / "plots" / "paper_round_priority_summary.csv"
OUT_DIR = ROOT / "visualization" / "plots" / "paper_figures"

MODEL_ORDER = ["random", "q_learning", "dqn", "double_dqn", "ppo", "a2c"]
MODEL_LABELS = {
    "random": "Random",
    "q_learning": "Q-learning",
    "dqn": "DQN",
    "double_dqn": "Double DQN",
    "ppo": "PPO",
    "a2c": "A2C",
}
PRIORITY_ORDER = ["URLLC", "mMTC", "eMBB"]


def smooth(series: pd.Series, window: int = 25) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def plot_all_models(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), sharex=True)
    colors = plt.get_cmap("tab10").colors
    color_map = {m: colors[i % len(colors)] for i, m in enumerate(MODEL_ORDER)}

    for col_idx, priority in enumerate(PRIORITY_ORDER):
        subp = df[df["priority_class"] == priority]
        ax_top = axes[0, col_idx]
        ax_bottom = axes[1, col_idx]
        for model in MODEL_ORDER:
            sub = subp[subp["model"] == model].sort_values("round")
            if sub.empty:
                continue
            label = MODEL_LABELS.get(model, model)
            color = color_map[model]
            ax_top.plot(
                sub["round"],
                smooth(sub["mean_device_battery_ratio"] * 100),
                lw=1.5,
                color=color,
                label=label,
            )
            ax_bottom.plot(
                sub["round"],
                smooth(sub["mean_battery_drain_j"]),
                lw=1.2,
                color=color,
                label=label,
            )

        ax_top.set_title(priority)
        ax_top.set_ylabel("Battery remaining (%)")
        ax_top.grid(axis="y", alpha=0.25)
        ax_bottom.set_xlabel("Round")
        ax_bottom.set_ylabel("Battery drain (J/task)")
        ax_bottom.grid(axis="y", alpha=0.25)

    axes[0, 0].legend(fontsize=8, ncol=2, loc="best")
    fig.suptitle("Round-wise device battery behavior by priority and model", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT_DIR / "roundwise_device_battery_behavior_all_models.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_dqn_focus(df: pd.DataFrame) -> Path:
    sub = df[df["model"] == "dqn"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    colors = {"URLLC": "#ef4444", "mMTC": "#2563eb", "eMBB": "#16a34a"}

    for priority in PRIORITY_ORDER:
        psub = sub[sub["priority_class"] == priority].sort_values("round")
        if psub.empty:
            continue
        axes[0].plot(
            psub["round"],
            smooth(psub["mean_device_battery_ratio"] * 100),
            lw=1.8,
            color=colors[priority],
            label=priority,
        )
        axes[1].plot(
            psub["round"],
            smooth(psub["mean_battery_drain_j"]),
            lw=1.8,
            color=colors[priority],
            label=priority,
        )

    axes[0].set_title("Remaining battery")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Battery remaining (%)")
    axes[1].set_title("Battery drain")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Battery drain (J/task)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("DQN round-wise device battery behavior", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = OUT_DIR / "roundwise_device_battery_behavior_dqn.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SUMMARY)
    for col in ["round", "mean_device_battery_ratio", "mean_device_battery_after_j", "mean_battery_drain_j"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["round", "mean_device_battery_ratio", "mean_battery_drain_j"])

    out_all = plot_all_models(df)
    out_dqn = plot_dqn_focus(df)

    table = (
        df.groupby(["model", "priority_class"])
        .agg(
            first_round=("round", "min"),
            last_round=("round", "max"),
            start_battery_pct=("mean_device_battery_ratio", "first"),
            end_battery_pct=("mean_device_battery_ratio", "last"),
            mean_battery_drain_j=("mean_battery_drain_j", "mean"),
            mean_battery_after_j=("mean_device_battery_after_j", "mean"),
        )
        .reset_index()
    )
    table["start_battery_pct"] *= 100
    table["end_battery_pct"] *= 100
    table["battery_drop_pct_points"] = table["start_battery_pct"] - table["end_battery_pct"]
    summary_path = OUT_DIR / "roundwise_device_battery_behavior_summary.csv"
    table.to_csv(summary_path, index=False)

    print(out_all)
    print(out_dqn)
    print(summary_path)


if __name__ == "__main__":
    main()
