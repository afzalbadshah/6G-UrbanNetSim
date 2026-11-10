from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output" / "evaluation" / "offloading"
OUT_DIR = ROOT / "visualization" / "plots" / "paper_figures"
MODEL_LABELS = {
    "random": "Random",
    "q_learning": "Q-learning",
    "dqn": "DQN",
    "double_dqn": "Double DQN",
    "ppo": "PPO",
    "a2c": "A2C",
}
MODEL_ORDER = ["random", "q_learning", "dqn", "double_dqn", "ppo", "a2c"]


def load_summary() -> pd.DataFrame:
    cols = [
        "wireless_ap",
        "chosen_rssi_dbm",
        "SINR_dB",
        "coverage_outage_flag",
        "active_devices_this_round",
        "priority_class",
        "class_name",
    ]
    rows = []
    for model in MODEL_ORDER:
        path = BASE / model / "results.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, sep=";", usecols=lambda c: c in cols)
        df["cls"] = df["priority_class"].fillna(df["class_name"])
        df["wireless_group"] = (
            df["wireless_ap"].astype(str).str.extract(r"^(UAV_BS|MEC)", expand=False)
        )
        df = df[
            (df["cls"] == "eMBB")
            & (df["active_devices_this_round"] >= 250)
            & (df["wireless_group"].isin(["MEC", "UAV_BS"]))
        ].copy()
        if df.empty:
            continue
        for col in ["chosen_rssi_dbm", "SINR_dB", "coverage_outage_flag"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        group = (
            df.groupby("wireless_group")
            .agg(
                rows=("wireless_ap", "size"),
                rssi=("chosen_rssi_dbm", "mean"),
                sinr=("SINR_dB", "mean"),
                outage_pct=("coverage_outage_flag", lambda s: s.mean() * 100),
            )
            .reset_index()
        )
        group["model"] = model
        group["model_label"] = MODEL_LABELS.get(model, model)
        rows.append(group)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    summary = load_summary()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "all_models_high_crowd_embb_coverage_gain_summary.csv"
    summary.to_csv(summary_path, index=False)

    if summary.empty:
        print("No data")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    metrics = [
        ("rssi", "Mean RSSI (dBm)", "RSSI"),
        ("sinr", "Mean SINR (dB)", "SINR"),
        ("outage_pct", "Outage rate (%)", "Outage"),
    ]
    colors = {"MEC": "#2563eb", "UAV_BS": "#16a34a"}
    present_models = [m for m in MODEL_ORDER if m in set(summary["model"])]
    x = range(len(present_models))
    width = 0.36

    for ax, (metric, ylabel, title) in zip(axes, metrics):
        for i, group in enumerate(["MEC", "UAV_BS"]):
            vals = []
            for model in present_models:
                hit = summary[
                    (summary["model"] == model) & (summary["wireless_group"] == group)
                ]
                vals.append(float(hit[metric].iloc[0]) if not hit.empty else float("nan"))
            offset = -width / 2 if group == "MEC" else width / 2
            ax.bar(
                [v + offset for v in x],
                vals,
                width=width,
                color=colors[group],
                label="MEC AP" if group == "MEC" else "UAV-BS AP",
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(x))
        ax.set_xticklabels([MODEL_LABELS[m] for m in present_models], rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if metric == "sinr":
            ax.axhline(0, color="gray", lw=0.8)
        if metric == "outage_pct":
            ax.set_ylim(0, max(70, summary[metric].max() * 1.15))
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "High-crowd eMBB coverage: MEC AP vs UAV-BS AP across models",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = OUT_DIR / "all_models_high_crowd_embb_coverage_gain.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(summary.round(3).to_string(index=False))
    print(out_path)
    print(summary_path)


if __name__ == "__main__":
    main()
