"""Run a fresh DQN evaluation sample and plot zone-first paper figures.

This utility is intentionally self-contained: it patches the runtime config in
memory, runs only DQN, and writes sample plots without changing config/run.py.

Outputs:
    output/run_XXX_zoned_satellite_6g_evaluation/
        offloading/dqn/results.csv
        plots/fresh_dqn_zone_sample/
            fig1_dqn_qos_8panel.png
            fig2_dqn_energy_8panel.png
            fig3_dqn_scalability_8panel.png
            fig4_dqn_action_share_4panel.png
            sample_manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
from applications import MainApplication as app
from scheduler.scheduler_helper.training_visualization_helper import (
    ACTION_LABEL_LIST,
    TIER_COLORS,
)


ZONE_ORDER = ("cellular", "uav", "remote", "all")
ZONE_LABELS = {
    "cellular": "Cellular",
    "uav": "UAV edge",
    "remote": "Remote satellite",
    "all": "All zones",
}
MODEL_COLOR = "#D62728"
OUT_SUBDIR = Path("plots") / "fresh_dqn_zone_sample"
PNG_DPI = 220
TIME_CREATED = datetime.now(timezone.utc).isoformat()

NEEDED_COLUMNS = (
    "round",
    "active_devices_this_round",
    "zone_id",
    "class_name",
    "service_class",
    "priority_class",
    "status",
    "success",
    "deadline_missed",
    "observed_complete_latency_ms",
    "battery_draw_j",
    "total_energy_j",
    "device_battery_ratio",
    "throughput_mbps",
    "allocated_tier",
    "selected_tier",
    "execution_tier",
    "offloading_tier",
)


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.45,
    }
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fresh DQN-only sample run plus zone-first plots."
    )
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--max-devices", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Run-history root relative to 6G-UrbanNetSim unless absolute.",
    )
    return parser.parse_args()


def _sample_increment(rounds: int, max_devices: int) -> int:
    if rounds <= 1:
        return max(1, max_devices - 1)
    return max(1, int(math.ceil((max_devices - 1) / float(rounds - 1))))


def _configure_fresh_dqn_sample(args: argparse.Namespace) -> None:
    """Patch runtime config in memory only; do not edit config/run.py."""
    cfg.RANDOM_SEED = int(args.seed)
    cfg.RUN_TRAINING = False
    cfg.RUN_MODELS = ["dqn"]
    cfg.COMPARISON_ENABLED = True
    cfg.MAIN_SCHEDULER = "dqn"
    cfg.WORKLOAD_PROFILE = "custom"
    cfg.ROUNDS = int(args.rounds)
    cfg.MAX_DEVICES = int(args.max_devices)
    cfg.WORKLOAD_OVERRIDES = {
        "start_devices": 1,
        "max_devices": int(args.max_devices),
        "rounds": int(args.rounds),
        "increment": _sample_increment(args.rounds, args.max_devices),
    }
    cfg.OUTPUT_DIR = str(args.output_dir)
    cfg.TRAINING_GUI = False
    cfg.EVALUATION_GUI = False
    cfg.MULTISEED = False
    cfg.MULTISEED_COUNT = 1
    cfg.MULTISEED_PARALLEL = False


def _result_path(run_dir: Path) -> Path:
    candidates = [
        run_dir / "offloading" / "dqn" / "results.csv",
        run_dir / "seed_1" / "offloading" / "dqn" / "results.csv",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(f"No DQN results.csv found under {run_dir}")


def _read_results(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, sep=";", nrows=0)
    usecols = [column for column in NEEDED_COLUMNS if column in header.columns]
    df = pd.read_csv(path, sep=";", usecols=usecols)

    if "class_name" not in df.columns:
        if "service_class" in df.columns:
            df["class_name"] = df["service_class"]
        elif "priority_class" in df.columns:
            df["class_name"] = df["priority_class"]

    if "zone_id" not in df.columns:
        df["zone_id"] = "cellular"

    df["class_name"] = df["class_name"].astype(str)
    df["zone_id"] = df["zone_id"].astype(str).str.lower()

    numeric = (
        "round",
        "active_devices_this_round",
        "success",
        "deadline_missed",
        "observed_complete_latency_ms",
        "battery_draw_j",
        "total_energy_j",
        "device_battery_ratio",
        "throughput_mbps",
    )
    for column in numeric:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _completed(df: pd.DataFrame) -> pd.Series:
    if "status" in df.columns:
        return df["status"].astype(str).str.lower().eq("completed")
    if "success" in df.columns:
        return df["success"].fillna(0).astype(float).ge(1)
    return pd.Series(True, index=df.index)


def _tier(df: pd.DataFrame) -> pd.Series:
    for column in ("selected_tier", "allocated_tier", "execution_tier", "offloading_tier"):
        if column in df.columns:
            values = df[column].astype(str).str.strip().str.upper()
            if values.ne("").any():
                return values.replace({"LOCAL": "LOCAL_DEVICE", "UAV": "UAV_BS"})
    return pd.Series("UNKNOWN", index=df.index)


def _zone_subset(df: pd.DataFrame, zone: str) -> pd.DataFrame:
    if zone == "all":
        return df.copy()
    return df[df["zone_id"] == zone].copy()


def _smooth_curve(curve: pd.Series, window: int = 3) -> pd.Series:
    return curve.sort_index().rolling(window, min_periods=1).mean()


def _rate_curve(df: pd.DataFrame, column: str) -> pd.Series:
    if df.empty or column not in df.columns:
        return pd.Series(dtype=float)
    work = df.dropna(subset=["active_devices_this_round", column]).copy()
    if work.empty:
        return pd.Series(dtype=float)
    return _smooth_curve(work.groupby("active_devices_this_round")[column].mean() * 100.0)


def _mean_curve(df: pd.DataFrame, column: str, scale: float = 1.0) -> pd.Series:
    if df.empty or column not in df.columns:
        return pd.Series(dtype=float)
    work = df.dropna(subset=["active_devices_this_round", column]).copy()
    if work.empty:
        return pd.Series(dtype=float)
    return _smooth_curve(work.groupby("active_devices_this_round")[column].mean() * scale)


def _quantile_curve(df: pd.DataFrame, column: str, q: float) -> pd.Series:
    if df.empty or column not in df.columns:
        return pd.Series(dtype=float)
    work = df.dropna(subset=["active_devices_this_round", column]).copy()
    if work.empty:
        return pd.Series(dtype=float)
    return _smooth_curve(work.groupby("active_devices_this_round")[column].quantile(q))


def _draw_line(ax: plt.Axes, curve: pd.Series, title: str, ylabel: str, ylim=None) -> None:
    ax.set_title(title)
    if curve.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No recorded data", ha="center", va="center", transform=ax.transAxes)
        return
    ax.plot(curve.index, curve.values, color=MODEL_COLOR, linewidth=1.65, marker="o", markersize=2.8)
    ax.set_xlabel("Active devices")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _eight_panel(
    df: pd.DataFrame,
    service_class: str,
    top_title: str,
    bottom_title: str,
    top_ylabel: str,
    bottom_ylabel: str,
    top_curve,
    bottom_curve,
    filename: Path,
    suptitle: str,
    top_ylim=None,
    bottom_ylim=None,
) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 6.2), sharex=False)
    for col, zone in enumerate(ZONE_ORDER):
        zone_df = _zone_subset(df, zone)
        zone_df = zone_df[zone_df["class_name"] == service_class].copy()
        _draw_line(
            axes[0, col],
            top_curve(zone_df),
            f"({chr(97 + col)}) {ZONE_LABELS[zone]}",
            top_ylabel if col == 0 else "",
            ylim=top_ylim,
        )
        _draw_line(
            axes[1, col],
            bottom_curve(zone_df),
            f"({chr(101 + col)}) {ZONE_LABELS[zone]}",
            bottom_ylabel if col == 0 else "",
            ylim=bottom_ylim,
        )
    fig.suptitle(f"{suptitle}\nTop row: {top_title}; bottom row: {bottom_title}", fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88), h_pad=2.6, w_pad=1.0)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return filename


def _plot_qos(df: pd.DataFrame, out_dir: Path) -> Path:
    def top(zone_df: pd.DataFrame) -> pd.Series:
        return _rate_curve(zone_df, "success")

    def bottom(zone_df: pd.DataFrame) -> pd.Series:
        completed = zone_df.loc[_completed(zone_df)].copy()
        return _quantile_curve(completed, "observed_complete_latency_ms", 0.95)

    return _eight_panel(
        df,
        "URLLC",
        "Main parameter: deadline success rate",
        "Sub parameter: P95 completion latency",
        "Success (%)",
        "P95 latency (ms)",
        top,
        bottom,
        out_dir / "fig1_dqn_qos_8panel.png",
        "DQN sample QoS evaluation by zone scenario (URLLC)",
        top_ylim=(0, 105),
    )


def _plot_energy(df: pd.DataFrame, out_dir: Path) -> Path:
    energy_col = "battery_draw_j" if "battery_draw_j" in df.columns else "total_energy_j"

    def top(zone_df: pd.DataFrame) -> pd.Series:
        return _mean_curve(zone_df, energy_col)

    def bottom(zone_df: pd.DataFrame) -> pd.Series:
        return _mean_curve(zone_df, "device_battery_ratio", scale=100.0)

    return _eight_panel(
        df,
        "mMTC",
        f"Main parameter: mean battery draw ({energy_col})",
        "Sub parameter: remaining device battery ratio",
        "Energy (J)",
        "Battery left (%)",
        top,
        bottom,
        out_dir / "fig2_dqn_energy_8panel.png",
        "DQN sample energy evaluation by zone scenario (mMTC)",
        bottom_ylim=(0, 105),
    )


def _plot_scalability(df: pd.DataFrame, out_dir: Path) -> Path:
    def top(zone_df: pd.DataFrame) -> pd.Series:
        if zone_df.empty:
            return pd.Series(dtype=float)
        work = zone_df.copy()
        work["completed"] = _completed(work).astype(float)
        return _rate_curve(work, "completed")

    def bottom(zone_df: pd.DataFrame) -> pd.Series:
        completed = zone_df.loc[_completed(zone_df)].copy()
        return _mean_curve(completed, "throughput_mbps")

    return _eight_panel(
        df,
        "eMBB",
        "Main parameter: application completion ratio",
        "Sub parameter: mean throughput",
        "Completed (%)",
        "Throughput (Mbit/s)",
        top,
        bottom,
        out_dir / "fig3_dqn_scalability_8panel.png",
        "DQN sample scalability evaluation by zone scenario (eMBB)",
        top_ylim=(0, 105),
    )


def _action_share_curve(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.dropna(subset=["active_devices_this_round"]).copy()
    work["tier"] = _tier(work)
    actions = [a for a in ACTION_LABEL_LIST if a in set(work["tier"])]
    extras = sorted(set(work["tier"]).difference(actions).difference({"", "NAN"}))
    actions.extend(extras)
    if not actions:
        return pd.DataFrame()

    counts = (
        work.groupby(["active_devices_this_round", "tier"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=actions, fill_value=0)
        .sort_index()
    )
    shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    shares = shares.rolling(3, min_periods=1).mean()
    shares = shares.div(shares.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return shares


def _plot_actions(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.2), sharex=False, sharey=True)
    flat = axes.ravel()
    for idx, zone in enumerate(ZONE_ORDER):
        ax = flat[idx]
        shares = _action_share_curve(_zone_subset(df, zone))
        ax.set_title(f"({chr(97 + idx)}) {ZONE_LABELS[zone]}")
        if shares.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "No recorded data", ha="center", va="center", transform=ax.transAxes)
            continue
        xs = shares.index.to_numpy(dtype=float)
        values = [shares[col].to_numpy(dtype=float) for col in shares.columns]
        colors = [TIER_COLORS.get(col, "#94a3b8") for col in shares.columns]
        ax.stackplot(xs, values, labels=[c.replace("_", " ") for c in shares.columns], colors=colors, alpha=0.92)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Active devices")
        if idx in (0, 2):
            ax.set_ylabel("Action share")
    legend_items = {}
    for ax in flat:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            legend_items.setdefault(label, handle)
    if legend_items:
        labels = list(legend_items)
        handles = [legend_items[label] for label in labels]
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False)
    fig.suptitle("DQN sample zone-aware action allocation (all service classes)", fontsize=11)
    fig.tight_layout(rect=(0.0, 0.07 if legend_items else 0.0, 1.0, 0.95))
    path = out_dir / "fig4_dqn_action_share_4panel.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def generate_sample_plots(run_dir: Path) -> list[Path]:
    result_csv = _result_path(run_dir)
    df = _read_results(result_csv)
    out_dir = run_dir / OUT_SUBDIR
    written = [
        _plot_qos(df, out_dir),
        _plot_energy(df, out_dir),
        _plot_scalability(df, out_dir),
        _plot_actions(df, out_dir),
    ]
    manifest = {
        "created_utc": TIME_CREATED,
        "run_dir": str(run_dir),
        "result_csv": str(result_csv),
        "model": "dqn",
        "zones": list(ZONE_ORDER),
        "figures": [str(path.relative_to(run_dir)) for path in written],
        "note": "Fresh compact DQN-only sample; use full multi-seed run for final paper claims.",
    }
    manifest_path = out_dir / "sample_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    written.append(manifest_path)
    return written


def main() -> None:
    args = _parse_args()
    _configure_fresh_dqn_sample(args)
    app.run_configured_models()
    run_dir = Path(app.RUNTIME["run_dir"])
    written = generate_sample_plots(run_dir)
    print(f"[fresh_dqn_zone_sample] run_dir={run_dir}")
    for path in written:
        print(f"[fresh_dqn_zone_sample] {path}")


if __name__ == "__main__":
    main()
