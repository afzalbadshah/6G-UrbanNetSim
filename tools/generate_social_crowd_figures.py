from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "visualization" / "plots"
OUT = PLOTS / "paper_figures"
MODEL_SUMMARY = PLOTS / "paper_model_priority_summary.csv"
TIER_SUMMARY = PLOTS / "paper_tier_priority_summary.csv"
OFFLOADING = ROOT / "output" / "evaluation" / "offloading"

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
TIER_ORDER = ["LOCAL_DEVICE", "MEC", "UAV_BS", "CLOUD"]
TIER_LABELS = {
    "LOCAL_DEVICE": "Local",
    "MEC": "MEC",
    "UAV_BS": "UAV-BS",
    "CLOUD": "Cloud",
}
COLORS = {
    "Random": "#6b7280",
    "Q-learning": "#f59e0b",
    "DQN": "#2563eb",
    "Double DQN": "#7c3aed",
    "PPO": "#16a34a",
    "A2C": "#dc2626",
    "Local": "#4b5563",
    "MEC": "#2563eb",
    "UAV-BS": "#16a34a",
    "Cloud": "#dc2626",
}


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 220,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def ordered_summary() -> pd.DataFrame:
    df = pd.read_csv(MODEL_SUMMARY)
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model_label"])
    df["priority_class"] = pd.Categorical(df["priority_class"], PRIORITY_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    return df.sort_values(["priority_class", "model"])


def plot_grouped_bars(ax, df, value_col, ylabel, title):
    x = np.arange(len(PRIORITY_ORDER))
    width = 0.12
    offsets = (np.arange(len(MODEL_ORDER)) - (len(MODEL_ORDER) - 1) / 2) * width
    for i, model in enumerate(MODEL_ORDER):
        sub = df[df["model"] == model].set_index("priority_class")
        vals = [sub.loc[p, value_col] if p in sub.index else np.nan for p in PRIORITY_ORDER]
        label = MODEL_LABELS[model]
        ax.bar(x + offsets[i], vals, width=width, label=label, color=COLORS.get(label))
    ax.set_xticks(x)
    ax.set_xticklabels(PRIORITY_ORDER)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_delay_energy_success() -> list[Path]:
    df = ordered_summary()
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    plot_grouped_bars(ax, df, "mean_delay_ms", "Mean objective delay (ms)", "Delay by priority and model")
    ax.legend(ncol=3, frameon=True)
    fig.tight_layout()
    path = OUT / "social_crowd_figure1_delay_by_priority_model.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    plot_grouped_bars(axes[0], df, "mean_energy", "Mean objective energy (J)", "Energy by priority and model")
    plot_grouped_bars(axes[1], df, "mean_cost", "Mean cost", "Cost by priority and model")
    axes[0].legend(ncol=2, frameon=True)
    fig.tight_layout()
    path = OUT / "social_crowd_figure2_energy_cost_by_priority_model.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), sharex=True)
    plot_grouped_bars(axes[0], df, "success_rate", "Success rate (%)", "Deadline success by priority and model")
    plot_grouped_bars(axes[1], df, "deadline_missed_rate", "Miss rate (%)", "Deadline miss by priority and model")
    axes[0].set_ylim(0, 105)
    axes[1].set_ylim(0, max(5, df["deadline_missed_rate"].max() * 1.15))
    axes[0].legend(ncol=2, frameon=True)
    fig.tight_layout()
    path = OUT / "social_crowd_figure5_deadline_success_miss_rate.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def plot_tier_share() -> Path:
    df = pd.read_csv(TIER_SUMMARY)
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df.get("model_label", df["model"]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, priority in zip(axes, PRIORITY_ORDER):
        sub = df[df["priority_class"] == priority]
        bottoms = np.zeros(len(MODEL_ORDER))
        x = np.arange(len(MODEL_ORDER))
        for tier in TIER_ORDER:
            vals = []
            for model in MODEL_ORDER:
                hit = sub[(sub["model"] == model) & (sub["allocated_tier"] == tier)]
                vals.append(float(hit["percentage"].iloc[0]) if not hit.empty else 0.0)
            label = TIER_LABELS[tier]
            ax.bar(x, vals, bottom=bottoms, label=label, color=COLORS[label])
            bottoms += np.array(vals)
        ax.set_title(priority)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=35, ha="right")
        ax.set_ylim(0, 105)
        if ax is axes[0]:
            ax.set_ylabel("% of decisions")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    fig.suptitle("Tier share by priority and model", y=1.03)
    fig.tight_layout()
    path = OUT / "social_crowd_figure4_tier_share_by_priority.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def crowd_bin(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return (np.round(vals / 50.0) * 50).astype("Int64")


def scan_crowd_metrics() -> pd.DataFrame:
    rows = []
    usecols = [
        "active_devices_this_round",
        "priority_class",
        "allocated_tier",
        "u_mec",
        "u_uav",
        "chosen_rssi_dbm",
        "SINR_dB",
        "coverage_outage_flag",
    ]
    for model in MODEL_ORDER:
        path = OFFLOADING / model / "results.csv"
        if not path.exists():
            continue
        chunks = pd.read_csv(path, sep=";", usecols=lambda c: c in usecols, chunksize=200_000)
        for chunk in chunks:
            chunk["model"] = model
            chunk["model_label"] = MODEL_LABELS[model]
            chunk["crowd_bin"] = crowd_bin(chunk["active_devices_this_round"])
            chunk["uav_share_flag"] = (chunk["allocated_tier"] == "UAV_BS").astype(float) * 100.0
            chunk["outage_pct"] = pd.to_numeric(chunk["coverage_outage_flag"], errors="coerce").fillna(0.0) * 100.0
            numeric_cols = ["u_mec", "u_uav", "chosen_rssi_dbm", "SINR_dB", "uav_share_flag", "outage_pct"]
            for col in numeric_cols:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
            grouped = chunk.groupby(["model", "model_label", "priority_class", "crowd_bin"], observed=True)[numeric_cols].agg(["sum", "count"])
            for idx, rec in grouped.iterrows():
                row = {
                    "model": idx[0],
                    "model_label": idx[1],
                    "priority_class": idx[2],
                    "crowd_bin": int(idx[3]) if not pd.isna(idx[3]) else np.nan,
                }
                for col in numeric_cols:
                    row[f"{col}_sum"] = rec[(col, "sum")]
                    row[f"{col}_count"] = rec[(col, "count")]
                rows.append(row)
    agg = pd.DataFrame(rows)
    if agg.empty:
        return agg
    sum_cols = [c for c in agg.columns if c.endswith("_sum") or c.endswith("_count")]
    agg = agg.groupby(["model", "model_label", "priority_class", "crowd_bin"], as_index=False)[sum_cols].sum()
    for col in ["u_mec", "u_uav", "chosen_rssi_dbm", "SINR_dB", "uav_share_flag", "outage_pct"]:
        agg[col] = agg[f"{col}_sum"] / agg[f"{col}_count"].replace(0, np.nan)
    return agg.sort_values(["priority_class", "model", "crowd_bin"])


def line_by_model(ax, df, value, title, ylabel, models=MODEL_ORDER):
    for model in models:
        sub = df[df["model"] == model].sort_values("crowd_bin")
        if sub.empty:
            continue
        label = MODEL_LABELS[model]
        ax.plot(sub["crowd_bin"], sub[value], marker="o", linewidth=1.8, label=label, color=COLORS.get(label))
    ax.set_title(title)
    ax.set_xlabel("Crowd size")
    ax.set_ylabel(ylabel)


def plot_crowd_figures(agg: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    if agg.empty:
        return paths

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, priority in zip(axes, PRIORITY_ORDER):
        sub = agg[agg["priority_class"] == priority]
        for model in MODEL_ORDER:
            ms = sub[sub["model"] == model].sort_values("crowd_bin")
            if ms.empty:
                continue
            label = MODEL_LABELS[model]
            ax.plot(ms["crowd_bin"], ms["u_mec"], marker="o", linewidth=1.6, color=COLORS[label], label=f"{label} MEC")
            ax.plot(ms["crowd_bin"], ms["u_uav"], marker="s", linestyle="--", linewidth=1.3, color=COLORS[label], label=f"{label} UAV")
        ax.set_title(priority)
        ax.set_xlabel("Crowd size")
        if ax is axes[0]:
            ax.set_ylabel("Utilization (%)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, ncol=1)
    fig.suptitle("Resource utilization vs crowd", y=1.03)
    fig.tight_layout()
    path = OUT / "social_crowd_figure3_resource_utilization_vs_crowd.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    overall = agg.groupby(["model", "model_label", "crowd_bin"], as_index=False).agg(
        chosen_rssi_dbm=("chosen_rssi_dbm", "mean"),
        SINR_dB=("SINR_dB", "mean"),
        outage_pct=("outage_pct", "mean"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    line_by_model(axes[0], overall, "chosen_rssi_dbm", "RSSI support", "Mean RSSI (dBm)")
    line_by_model(axes[1], overall, "SINR_dB", "SINR support", "Mean SINR (dB)")
    line_by_model(axes[2], overall, "outage_pct", "Outage pressure", "Outage rate (%)")
    axes[0].legend(ncol=2, frameon=True)
    fig.suptitle("UAV-BS coverage support vs crowd", y=1.03)
    fig.tight_layout()
    path = OUT / "social_crowd_figure6_uav_bs_coverage_support_vs_crowd.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    overall = agg.groupby(["model", "model_label", "crowd_bin"], as_index=False).agg(
        uav_share_flag=("uav_share_flag", "mean"),
        u_uav=("u_uav", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    line_by_model(axes[0], overall, "uav_share_flag", "UAV-BS decision share", "% decisions to UAV-BS")
    line_by_model(axes[1], overall, "u_uav", "UAV-BS utilization", "Utilization (%)")
    axes[0].legend(ncol=2, frameon=True)
    fig.suptitle("UAV-BS collaboration vs crowd", y=1.03)
    fig.tight_layout()
    path = OUT / "social_crowd_figure7_uav_bs_collaboration_vs_crowd.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def main() -> None:
    setup()
    paths = []
    paths.extend(plot_delay_energy_success())
    paths.append(plot_tier_share())
    agg = scan_crowd_metrics()
    agg_path = PLOTS / "social_crowd_crowd_level_summary.csv"
    agg.to_csv(agg_path, index=False)
    paths.extend(plot_crowd_figures(agg))
    print("Wrote:")
    for path in paths:
        print(path)
    print(agg_path)


if __name__ == "__main__":
    main()

