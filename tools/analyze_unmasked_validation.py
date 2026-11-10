"""Summarize held-out unmasked-learning validation results by model/class/zone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS = (
    "reward", "observed_complete_latency_ms", "device_battery_energy_j",
    "reward_capacity_adequacy", "throughput_mbps",
    "reward_overutilization_deduction_points",
)


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";").copy()
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df.get(column), errors="coerce")
    df["completed"] = df["status"].astype(str).str.lower().eq("completed")
    df["rejected"] = df["execution_tier"].fillna("").astype(str).eq("")
    return df


def summarize(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    actions = []
    exploration = []
    for path in sorted(root.glob("*/heldout/offloading/*/results.csv")):
        model = path.parts[-2]
        df = load_results(path)
        for (service, zone), group in df.groupby(["class_name", "zone_id"], dropna=False):
            completed = group[group["completed"]]
            rows.append({
                "model": model,
                "class": service,
                "zone": zone,
                "attempts": len(group),
                "completion_pct": 100.0 * group["completed"].mean(),
                "rejected_pct": 100.0 * group["rejected"].mean(),
                "mean_reward": group["reward"].mean(),
                "p95_latency_ms": completed["observed_complete_latency_ms"].quantile(0.95),
                "mean_device_energy_j": completed["device_battery_energy_j"].mean(),
                "mean_capacity_adequacy": completed["reward_capacity_adequacy"].mean(),
                "mean_throughput_mbps": completed["throughput_mbps"].mean(),
                "mean_overload_deduction": group["reward_overutilization_deduction_points"].mean(),
            })
            counts = group["selected_tier"].fillna("NONE").value_counts()
            for tier, count in counts.items():
                actions.append({
                    "model": model, "class": service, "zone": zone,
                    "selected_tier": tier, "count": int(count),
                    "share_pct": 100.0 * count / len(group),
                })
    for path in sorted(root.glob("*/train/offloading/*/results.csv")):
        model = path.parts[-2]
        df = load_results(path)
        for (service, zone, tier), group in df.groupby(
            ["class_name", "zone_id", "selected_tier"], dropna=False
        ):
            completed = group[group["completed"]]
            exploration.append({
                "model": model,
                "class": service,
                "zone": zone,
                "selected_tier": tier,
                "attempts": len(group),
                "completion_pct": 100.0 * group["completed"].mean(),
                "mean_reward": group["reward"].mean(),
                "median_latency_ms": completed["observed_complete_latency_ms"].median(),
                "p95_latency_ms": completed["observed_complete_latency_ms"].quantile(0.95),
                "mean_device_energy_j": completed["device_battery_energy_j"].mean(),
                "mean_capacity_adequacy": completed["reward_capacity_adequacy"].mean(),
                "mean_overload_deduction": group["reward_overutilization_deduction_points"].mean(),
            })
    return pd.DataFrame(rows), pd.DataFrame(actions), pd.DataFrame(exploration)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    metrics, actions, exploration = summarize(args.root.resolve())
    metrics_path = args.root / "heldout_class_zone_metrics.csv"
    actions_path = args.root / "heldout_class_zone_actions.csv"
    exploration_path = args.root / "training_action_objectives.csv"
    metrics.to_csv(metrics_path, index=False)
    actions.to_csv(actions_path, index=False)
    exploration.to_csv(exploration_path, index=False)
    json_metrics = metrics.astype(object).where(pd.notna(metrics), None)
    json_actions = actions.astype(object).where(pd.notna(actions), None)
    payload = {
        "metrics": json_metrics.to_dict(orient="records"),
        "actions": json_actions.to_dict(orient="records"),
        "training_action_objectives": exploration.astype(object).where(
            pd.notna(exploration), None
        ).to_dict(orient="records"),
    }
    json_path = args.root / "heldout_class_zone_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"metrics={metrics_path}")
    print(f"actions={actions_path}")
    print(f"training_action_objectives={exploration_path}")
    print(f"json={json_path}")


if __name__ == "__main__":
    main()
