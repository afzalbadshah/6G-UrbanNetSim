"""Summarize whether policies are meaningfully separated in an evaluation run.

Usage:
    python tools/analyze_run_similarity.py output/run_003_zoned_satellite_6g_evaluation
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.supportive.eval_plots import (  # noqa: E402
    _completed,
    _ordered_models,
    _tier_series,
    load_zone_results,
)


ZONES = ("cellular", "uav", "remote", "all")
METRIC_COLUMNS = (
    "urllc_success_mean",
    "urllc_p95_mean",
    "mmtc_draw_mean",
    "mmtc_final_batt_mean",
    "embb_completion_mean",
    "embb_tput_mean",
)


def _zone_subset(df: pd.DataFrame, zone: str) -> pd.DataFrame:
    return df if zone == "all" else df[df["zone_id"] == zone]


def _seed_stats(values: list[float]) -> tuple[float, float, int]:
    values = [float(value) for value in values if pd.notna(value)]
    if not values:
        return np.nan, np.nan, 0
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return float(np.mean(values)), std, len(values)


def _per_seed_metric(df: pd.DataFrame, fn) -> tuple[float, float, int]:
    values = []
    for _seed, seed_df in df.groupby("seed", observed=True):
        try:
            values.append(fn(seed_df))
        except Exception:
            values.append(np.nan)
    return _seed_stats(values)


def _numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    models = _ordered_models(df["model"].dropna().astype(str).unique())
    for model in models:
        model_df = df[df["model"].astype(str) == model]
        for zone in ZONES:
            zone_df = _zone_subset(model_df, zone)
            if zone_df.empty:
                continue
            urllc = zone_df[zone_df["class_name"] == "URLLC"]
            mmtc = zone_df[zone_df["class_name"] == "mMTC"]
            embb = zone_df[zone_df["class_name"] == "eMBB"]

            urllc_success = _per_seed_metric(
                urllc,
                lambda seed_df: 100.0 * seed_df["success"].astype(float).mean(),
            )
            urllc_p95 = _per_seed_metric(
                urllc,
                lambda seed_df: np.nanpercentile(
                    seed_df.loc[_completed(seed_df), "observed_complete_latency_ms"].astype(float),
                    95,
                ),
            )
            mmtc_draw = _per_seed_metric(
                mmtc,
                lambda seed_df: seed_df["battery_draw_j"].astype(float).mean(),
            )
            mmtc_final_batt = _per_seed_metric(
                mmtc,
                lambda seed_df: 100.0
                * seed_df.sort_values("round")
                .groupby("device_id", observed=True)["device_battery_ratio"]
                .last()
                .mean(),
            )
            embb_completion = _per_seed_metric(
                embb,
                lambda seed_df: 100.0 * _completed(seed_df).mean(),
            )
            embb_throughput = _per_seed_metric(
                embb,
                lambda seed_df: seed_df.loc[_completed(seed_df), "throughput_mbps"].astype(float).mean(),
            )

            rows.append({
                "model": model,
                "zone": zone,
                "urllc_success_mean": urllc_success[0],
                "urllc_success_std": urllc_success[1],
                "urllc_p95_mean": urllc_p95[0],
                "urllc_p95_std": urllc_p95[1],
                "mmtc_draw_mean": mmtc_draw[0],
                "mmtc_draw_std": mmtc_draw[1],
                "mmtc_final_batt_mean": mmtc_final_batt[0],
                "mmtc_final_batt_std": mmtc_final_batt[1],
                "embb_completion_mean": embb_completion[0],
                "embb_completion_std": embb_completion[1],
                "embb_tput_mean": embb_throughput[0],
                "embb_tput_std": embb_throughput[1],
            })
    return pd.DataFrame(rows)


def _action_tables(df: pd.DataFrame) -> None:
    df = df.copy()
    df["action"] = _tier_series(df)
    models = _ordered_models(df["model"].dropna().astype(str).unique())
    print("\nACTION SHARES (%) BY ZONE/MODEL")
    for zone in ZONES:
        zone_df = _zone_subset(df, zone)
        rows = []
        for model in models:
            shares = (
                zone_df[zone_df["model"].astype(str) == model]["action"]
                .value_counts(normalize=True)
                .mul(100.0)
            )
            rows.append(pd.Series({"model": model, **{str(k): float(v) for k, v in shares.items()}}))
        table = pd.DataFrame(rows).fillna(0.0).set_index("model")
        print(f"\n{zone}")
        print(table.round(1).to_string())
        if "dqn" in table.index:
            dqn = table.loc["dqn"] / 100.0
            total_variation = table.div(100.0).sub(dqn, axis=1).abs().sum(axis=1).div(2.0)
            print("total variation distance from DQN:", total_variation.round(3).to_dict())


def _feasibility_signals(phase: Path) -> None:
    paths = list(phase.glob("seed_*/offloading/*/results.csv"))
    columns = (
        "selected_tier",
        "allocated_tier",
        "execution_tier",
        "offloading_tier",
        "spillover_used",
        "valid_count",
        "physical_valid_count",
        "learning_valid_count",
        "final_valid_count",
        "only_cloud_feasible",
        "reward_forced_single_action",
    )
    frames = []
    for path in paths:
        try:
            frame = pd.read_csv(path, sep=";", usecols=lambda col: col in columns, low_memory=False)
        except Exception:
            continue
        frame["model"] = path.parent.name
        frame["seed"] = int(path.parents[2].name.split("_")[-1])
        frames.append(frame)
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    available = [column for column in columns if column in data.columns]
    print("\nFEASIBILITY SIGNALS BY MODEL")
    numeric_available = [
        column
        for column in available
        if column not in {"selected_tier", "allocated_tier", "execution_tier", "offloading_tier"}
    ]
    for column in numeric_available:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    print(data.groupby("model", observed=True)[numeric_available].mean().round(3).to_string())

    if {"selected_tier", "allocated_tier"}.issubset(data.columns):
        changed = data["selected_tier"].astype(str) != data["allocated_tier"].astype(str)
        changed = changed & data["selected_tier"].astype(str).ne("") & data["allocated_tier"].astype(str).ne("")
        data["_selected_differs_from_allocated"] = changed.astype(float)
        print("\nSELECTED VS ALLOCATED OVERRIDE RATE")
        print(data.groupby("model", observed=True)["_selected_differs_from_allocated"].mean().mul(100).round(2).to_string())

        print("\nSELECTED TIER SHARES (%)")
        selected = (
            data.groupby("model", observed=True)["selected_tier"]
            .value_counts(normalize=True)
            .mul(100.0)
            .unstack(fill_value=0.0)
            .round(1)
        )
        print(selected.to_string())

        print("\nALLOCATED TIER SHARES (%)")
        allocated = (
            data.groupby("model", observed=True)["allocated_tier"]
            .value_counts(normalize=True)
            .mul(100.0)
            .unstack(fill_value=0.0)
            .round(1)
        )
        print(allocated.to_string())

    if "valid_count" in data.columns:
        print("\nVALID ACTION COUNT DISTRIBUTION (%)")
        dist = (
            data.groupby("model", observed=True)["valid_count"]
            .value_counts(normalize=True)
            .mul(100.0)
            .unstack(fill_value=0.0)
            .round(1)
        )
        print(dist.to_string())


def main() -> None:
    phase = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "output" / "run_003_zoned_satellite_6g_evaluation"
    if not phase.is_absolute():
        phase = ROOT / phase

    df = load_zone_results(phase)
    models = _ordered_models(df["model"].dropna().astype(str).unique())
    print("DATASET")
    print(f"phase: {phase}")
    print(f"rows: {len(df):,}")
    print(f"models: {models}")
    print(f"seeds: {sorted(df['seed'].unique().tolist())}")
    print(f"zones: {sorted(df['zone_id'].unique().tolist())}")

    summary = _numeric_summary(df)
    print("\nOVERALL BY MODEL (mean +/- seed std)")
    print(summary[summary["zone"] == "all"].round(4).to_string(index=False))

    print("\nZONE RANGES ACROSS MODELS")
    for zone in ZONES:
        zone_summary = summary[summary["zone"] == zone]
        ranges = {
            column: round(float(zone_summary[column].max() - zone_summary[column].min()), 4)
            for column in METRIC_COLUMNS
        }
        print(f"{zone}: {ranges}")

    print("\nKEY ZONE METRIC MATRICES")
    for column in ("urllc_p95_mean", "mmtc_final_batt_mean", "embb_tput_mean"):
        matrix = summary.pivot(index="model", columns="zone", values=column).round(2)
        print(f"\n{column}")
        print(matrix.to_string())

    _action_tables(df)
    _feasibility_signals(phase)


if __name__ == "__main__":
    main()
