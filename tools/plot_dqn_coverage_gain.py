from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "output" / "evaluation" / "offloading" / "dqn" / "results.csv"
OUT_DIR = ROOT / "visualization" / "plots" / "paper_figures"


def main() -> None:
    cols = [
        "wireless_ap",
        "chosen_rssi_dbm",
        "SINR_dB",
        "coverage_outage_flag",
        "active_devices_this_round",
        "priority_class",
        "class_name",
    ]
    df = pd.read_csv(RESULTS, sep=";", usecols=cols)
    df["cls"] = df["priority_class"].fillna(df["class_name"])
    df["wireless_group"] = (
        df["wireless_ap"].astype(str).str.extract(r"^(UAV_BS|MEC)", expand=False)
    )
    df = df[
        (df["cls"] == "eMBB")
        & (df["active_devices_this_round"] >= 250)
        & (df["wireless_group"].isin(["MEC", "UAV_BS"]))
    ].copy()

    for col in ["chosen_rssi_dbm", "SINR_dB", "coverage_outage_flag"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    summary = (
        df.groupby("wireless_group")
        .agg(
            rows=("wireless_ap", "size"),
            rssi=("chosen_rssi_dbm", "mean"),
            sinr=("SINR_dB", "mean"),
            outage_pct=("coverage_outage_flag", lambda s: s.mean() * 100),
        )
        .reindex(["MEC", "UAV_BS"])
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "dqn_high_crowd_embb_coverage_gain_summary.csv"
    summary.to_csv(summary_path)

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.0))
    colors = ["#2563eb", "#16a34a"]
    labels = ["MEC AP", "UAV-BS AP"]
    metrics = [
        ("rssi", "Mean RSSI (dBm)", "Higher is better"),
        ("sinr", "Mean SINR (dB)", "Higher is better"),
        ("outage_pct", "Outage rate (%)", "Lower is better"),
    ]

    for ax, (col, ylabel, subtitle) in zip(axes, metrics):
        vals = summary[col].values
        bars = ax.bar(labels, vals, color=colors, width=0.58)
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        for bar, val in zip(bars, vals):
            va = "bottom" if val >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val,
                f"{val:.2f}",
                ha="center",
                va=va,
                fontsize=9,
            )

    axes[0].set_ylim(min(summary["rssi"].min() - 2, -105), max(summary["rssi"].max() + 2, -85))
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[2].set_ylim(0, max(65, summary["outage_pct"].max() * 1.15))

    fig.suptitle(
        "DQN: High-crowd eMBB coverage gain from UAV-BS wireless association",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = OUT_DIR / "dqn_high_crowd_embb_coverage_gain.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(summary.round(3).to_string())
    print(out_path)
    print(summary_path)


if __name__ == "__main__":
    main()
