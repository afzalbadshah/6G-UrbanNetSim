"""Internal CSV comparison reporting retained for simulator compatibility."""
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from visualization.plot_config import (COLORS_MODELS,MODEL_ALIASES,ACTIVE_MODELS,SERVICE_CLASSES,plot_phase_dir)

PRIORITY_ORDER = SERVICE_CLASSES
MODEL_ORDER = ACTIVE_MODELS
TIER_ORDER = ["LOCAL_DEVICE", "MEC", "MEC_COLLAB", "UAV_BS", "METRO", "CLOUD"]
CONNECTION_ORDER = ["MEC", "UAV_BS", "METRO", "CLOUD", "LOCAL_DEVICE", "FAILED", "OTHER"]
TIER_COLORS = {
    "LOCAL_DEVICE": "#4b5563",
    "MEC": "#2563eb",
    "MEC_COLLAB": "#f97316",
    "UAV_BS": "#16a34a",
    "METRO": "#7c3aed",
    "CLOUD": "#dc2626",
    "FAILED": "#be185d",  # distinct from OTHER -- this is a task that never got assigned a server, not an unrecognized value
    "OTHER": "#94a3b8",
}

NUMERIC_MEAN_COLUMNS = [
    "reward", "delay_ms", "cost", "energy", "battery_remaining_j",
    "device_battery_before_j", "device_battery_after_j", "device_battery_ratio", "battery_capacity_j",
    "device_tx_energy_j", "device_rx_energy_j", "device_local_compute_energy_j",
    "device_energy_j", "processing_delay_ms", "service_time_ms", "valid_count",
    "mec_peak_future", "uav_peak_future", "cloud_peak_future",
]

RATE_COLUMNS = [
    "deadline_missed", "success", "mobility_violation", "completion_before_exit",
    "handover_failure_flag",
]

CSV_COLUMNS = {
    "round", "priority", "priority_class", "allocated_tier", "offloading_tier", "selected_tier",
    "connected_bs", "selected_server_type", "allocated_server", "selected_server_id",
    "reward", "delay_ms", "cost", "energy", "battery_remaining_j",
    "device_battery_before_j", "device_battery_after_j", "device_battery_ratio", "battery_capacity_j",
    "device_tx_energy_j", "device_rx_energy_j", "device_local_compute_energy_j",
    "processing_delay_ms", "proc_delay_ms", "service_time_ms", "valid_count",
    "deadline_missed", "success", "status", "mobility_violation", "completion_before_exit",
    "handover_executed", "chosen_rssi_dbm", "RSSI_dBm", "distance_to_bs", "distance_to_bs_m",
    "handover_failure_flag", "mec_peak_future", "uav_peak_future", "cloud_peak_future",
}

DECISION_QUALITY_COLUMNS = {
    "decision_idx", "time_ms", "vehicle_id", "chosen_bs", "oracle_bs",
    "chosen_rssi_dbm", "oracle_rssi_dbm", "regret_sig_norm", "speed_norm",
    "distance_m", "handover_flag", "pingpong_flag", "prev_bs", "target_bs",
    "target_rssi_dbm", "target_rssi_gain_db", "best_available_bs",
    "best_available_rssi_dbm", "chosen_cpu_util", "chosen_mem_util",
    "oracle_cpu_util", "oracle_mem_util", "load_regret", "q_margin",
}
SAMPLE_COLUMNS = [
    "model", "priority_class", "reward", "delay_ms", "battery_remaining_j",
    "device_energy_j", "processing_delay_ms", "service_time_ms", "valid_count",
]


def _norm_path(path):
    return Path(path).expanduser().resolve()


def _display_model(model):
    labels = {
        "random": "Random",
        "q_learning": "Q-learning",
        "dqn": "DQN",
        "double_dqn": "Double DQN",
        "ppo": "PPO",
        "a2c": "A2C",
    }
    return labels.get(str(model), str(model))


def _normalize_priority(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    low = text.lower()
    if low in {"1", "1.0", "urllc", "urlc", "ultra_reliable_low_latency"}:
        return "URLLC"
    if low in {"2", "2.0", "mmtc", "massive_machine_type"}:
        return "mMTC"
    if low in {"3", "3.0", "embb", "enhanced_mobile_broadband"}:
        return "eMBB"
    if "urllc" in low or "urlc" in low:
        return "URLLC"
    if "mmtc" in low:
        return "mMTC"
    if "embb" in low:
        return "eMBB"
    return text


def _normalize_tier(value):
    if pd.isna(value):
        return "UNKNOWN"
    text = str(value).strip().upper().replace(" ", "_").replace("-", "_")
    aliases = {
        "LOCAL": "LOCAL_DEVICE", "LOCAL_DEVICE": "LOCAL_DEVICE", "DEVICE": "LOCAL_DEVICE",
        "MEC": "MEC", "EDGE": "MEC", "MEC_COLLAB": "MEC_COLLAB",
        "MEC_COLLABORATION": "MEC_COLLAB", "UAV": "UAV_BS", "UAV_BS": "UAV_BS",
        "UAV_BASE_STATION": "UAV_BS", "METRO": "METRO", "CLOUD": "CLOUD",
    }
    return aliases.get(text, text)



def _normalize_connection_type(value):
    # Blank/NaN means the task never got assigned a server (see
    # comparison_runner.py's task.fail() path -- selected_server_type is
    # only ever set on success) -- that's a real, meaningful category
    # ("FAILED"), not the same thing as a value that IS present but
    # doesn't match any known tier ("OTHER").
    if pd.isna(value):
        return "FAILED"
    text = str(value).strip().upper().replace(" ", "_").replace("-", "_")
    if not text or text in {"NAN", "NONE", "NULL"}:
        return "FAILED"
    if text.startswith("MEC") or text == "EDGE":
        return "MEC"
    if text.startswith("UAV"):
        return "UAV_BS"
    if text.startswith("METRO"):
        return "METRO"
    if text.startswith("CLOUD"):
        return "CLOUD"
    if text.startswith("LOCAL") or text == "DEVICE":
        return "LOCAL_DEVICE"
    tier = _normalize_tier(text)
    return tier if tier in CONNECTION_ORDER else "OTHER"
def _mobility_root(evaluation_dir):
    root = Path(evaluation_dir)
    if root.name.lower() == "offloading":
        return root.parent / "mobility"
    if (root / "mobility").is_dir():
        return root / "mobility"
    return root.parent / "mobility"


def _fallback_model_root(model):
    return Path(tempfile.gettempdir()) / "SmartCityEdgeSim" / "evaluation_fallback" / model


def _decision_quality_path(evaluation_dir, model):
    return _mobility_root(evaluation_dir) / model / "decision_quality.csv"


def _decision_quality_paths(evaluation_dir, model):
    primary = _decision_quality_path(evaluation_dir, model)
    fallback = _fallback_model_root(model) / "mobility" / "decision_quality.csv"
    paths = [fallback, primary] if fallback.is_file() else [primary, fallback]
    seen = set()
    out = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _model_csv_path(evaluation_dir, model):
    return Path(evaluation_dir) / model / f"{model}_offloading_results.csv"


def _model_csv_paths(evaluation_dir, model):
    primary = _model_csv_path(evaluation_dir, model)
    fallback = _fallback_model_root(model) / "offloading" / f"{model}_offloading_results.csv"
    paths = [fallback, primary] if fallback.is_file() else [primary, fallback]
    seen = set()
    out = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _discover_models(evaluation_dir, requested=None):
    candidates = requested if requested else MODEL_ORDER
    return [m for m in candidates if any(path.is_file() for path in _model_csv_paths(evaluation_dir, m))]


def _coerce_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _prepare_chunk(df, model):
    df = df.copy()
    df["model"] = model
    if "priority_class" in df.columns:
        priority = df["priority_class"]
    elif "priority" in df.columns:
        priority = df["priority"]
    else:
        priority = pd.Series(np.nan, index=df.index)
    df["priority_class"] = priority.map(_normalize_priority)
    df = df[df["priority_class"].isin(PRIORITY_ORDER)].copy()
    if df.empty:
        return df

    if "round" not in df.columns:
        df["round"] = np.nan
    df["round"] = pd.to_numeric(df["round"], errors="coerce")

    if "processing_delay_ms" not in df.columns and "proc_delay_ms" in df.columns:
        df["processing_delay_ms"] = df["proc_delay_ms"]

    for col in NUMERIC_MEAN_COLUMNS + RATE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    _coerce_numeric(df, NUMERIC_MEAN_COLUMNS + RATE_COLUMNS)

    tx = df["device_tx_energy_j"].fillna(0.0)
    rx = df["device_rx_energy_j"].fillna(0.0)
    local = df["device_local_compute_energy_j"].fillna(0.0)
    df["device_energy_j"] = tx + rx + local
    df["energy"] = df["energy"].fillna(df["device_energy_j"])

    if "success" in df.columns:
        df["success"] = df["success"].fillna(0.0)
    if "status" in df.columns:
        status_success = df["status"].astype(str).str.lower().isin({"success", "completed", "done"})
        df["success"] = df["success"].fillna(status_success.astype(float))

    tier_source = None
    for col in ("allocated_tier", "offloading_tier", "selected_tier"):
        if col in df.columns:
            tier_source = df[col]
            break
    df["allocated_tier"] = tier_source.map(_normalize_tier) if tier_source is not None else "UNKNOWN"
    return df


def _read_chunks(evaluation_dir, model, chunksize):
    last_exc = None
    for path in _model_csv_paths(evaluation_dir, model):
        if not path.is_file():
            continue
        try:
            for chunk in pd.read_csv(path, sep=";", chunksize=chunksize, usecols=lambda col: col in CSV_COLUMNS, low_memory=False):
                prepared = _prepare_chunk(chunk, model)
                if not prepared.empty:
                    yield prepared
            return
        except OSError as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        print(f"[PLOT] Skipped unreadable CSV for {model}: {last_exc}", flush=True)


def _sample_by_group(df, sample_per_group, per_chunk=False):
    if df.empty:
        return df
    limit = max(50, min(250, sample_per_group // 8)) if per_chunk else sample_per_group
    sampled = []
    for _, group in df.groupby(["model", "priority_class"], observed=True):
        sampled.append(group.sample(n=min(len(group), limit), random_state=42))
    return pd.concat(sampled, ignore_index=True) if sampled else df.iloc[0:0].copy()


def _aggregate_evaluation(evaluation_dir, models, chunksize=50000, sample_per_group=4000):
    summary_parts, tier_parts, sample_parts, round_parts, connection_parts = [], [], [], [], []
    for model in models:
        for chunk in _read_chunks(evaluation_dir, model, chunksize):
            for col in [
                "device_battery_before_j", "device_battery_after_j", "device_battery_ratio",
                "handover_executed", "chosen_rssi_dbm", "RSSI_dBm", "distance_to_bs", "distance_to_bs_m",
            ]:
                if col not in chunk.columns:
                    chunk[col] = np.nan
            _coerce_numeric(chunk, [
                "device_battery_before_j", "device_battery_after_j", "device_battery_ratio",
                "handover_executed", "chosen_rssi_dbm", "RSSI_dBm", "distance_to_bs", "distance_to_bs_m",
            ])

            grouped = chunk.groupby(["model", "priority_class"], observed=True)
            part = grouped[NUMERIC_MEAN_COLUMNS + RATE_COLUMNS].sum(min_count=1)
            part["decisions"] = grouped.size()
            summary_parts.append(part.reset_index())

            tier_parts.append(
                chunk.groupby(["model", "priority_class", "allocated_tier"], observed=True)
                .size().reset_index(name="count")
            )

            connection_source = None
            for col in ("selected_server_type", "connected_bs", "allocated_server", "selected_server_id"):
                if col in chunk.columns:
                    candidate = chunk[col].astype(str).replace({"nan": "", "None": ""})
                    if candidate.str.len().sum() > 0:
                        connection_source = candidate
                        break
            if connection_source is not None:
                conn = chunk[["model", "priority_class"]].copy()
                conn["connection_type"] = connection_source.map(_normalize_connection_type)
                connection_parts.append(
                    conn.groupby(["model", "priority_class", "connection_type"], observed=True)
                    .size().reset_index(name="count")
                )

            if "round" in chunk.columns and chunk["round"].notna().any():
                round_data = chunk.dropna(subset=["round"]).copy()
                round_data["round"] = round_data["round"].astype(int)
                round_data["device_battery_after_j"] = round_data["device_battery_after_j"].fillna(round_data["battery_remaining_j"])
                round_data["device_battery_before_j"] = round_data["device_battery_before_j"].fillna(round_data["device_battery_after_j"])
                round_data["battery_drain_j"] = (round_data["device_battery_before_j"] - round_data["device_battery_after_j"]).clip(lower=0.0)
                round_data["chosen_rssi_dbm"] = round_data["chosen_rssi_dbm"].fillna(round_data["RSSI_dBm"])
                round_data["distance_to_bs_m"] = round_data["distance_to_bs_m"].fillna(round_data["distance_to_bs"])
                round_parts.append(
                    round_data.groupby(["model", "priority_class", "round"], observed=True)
                    .agg(
                        delay_sum=("delay_ms", "sum"),
                        delay_count=("delay_ms", "count"),
                        battery_sum=("battery_remaining_j", "sum"),
                        battery_count=("battery_remaining_j", "count"),
                        device_battery_after_sum=("device_battery_after_j", "sum"),
                        device_battery_after_count=("device_battery_after_j", "count"),
                        device_battery_ratio_sum=("device_battery_ratio", "sum"),
                        device_battery_ratio_count=("device_battery_ratio", "count"),
                        battery_drain_sum=("battery_drain_j", "sum"),
                        battery_drain_count=("battery_drain_j", "count"),
                        handover_sum=("handover_executed", "sum"),
                        handover_count=("handover_executed", "count"),
                        handover_failure_sum=("handover_failure_flag", "sum"),
                        handover_failure_count=("handover_failure_flag", "count"),
                        rssi_sum=("chosen_rssi_dbm", "sum"),
                        rssi_count=("chosen_rssi_dbm", "count"),
                        distance_sum=("distance_to_bs_m", "sum"),
                        distance_count=("distance_to_bs_m", "count"),
                    )
                    .reset_index()
                )

            sample_cols = [col for col in SAMPLE_COLUMNS if col in chunk.columns]
            value_cols = [c for c in sample_cols if c not in {"model", "priority_class"}]
            reduced = chunk[sample_cols].dropna(how="all", subset=value_cols)
            if not reduced.empty:
                sample_parts.append(_sample_by_group(reduced, sample_per_group, per_chunk=True))

    if not summary_parts:
        raise FileNotFoundError(f"No offloading result rows found under {evaluation_dir}")

    raw_summary = pd.concat(summary_parts, ignore_index=True)
    sum_cols = NUMERIC_MEAN_COLUMNS + RATE_COLUMNS + ["decisions"]
    summary = raw_summary.groupby(["model", "priority_class"], observed=True)[sum_cols].sum(min_count=1).reset_index()
    for col in NUMERIC_MEAN_COLUMNS:
        summary[f"mean_{col}"] = summary[col] / summary["decisions"].replace(0, np.nan)
    for col in RATE_COLUMNS:
        summary[f"{col}_rate"] = 100.0 * summary[col] / summary["decisions"].replace(0, np.nan)
    summary["model_label"] = summary["model"].map(_display_model)
    summary["priority_class"] = pd.Categorical(summary["priority_class"], PRIORITY_ORDER, ordered=True)
    summary = summary.sort_values(["priority_class", "model"])

    if tier_parts:
        tier_counts = pd.concat(tier_parts, ignore_index=True)
        tier_counts = tier_counts.groupby(["model", "priority_class", "allocated_tier"], observed=True)["count"].sum().reset_index()
        totals = tier_counts.groupby(["model", "priority_class"], observed=True)["count"].transform("sum")
        tier_counts["percentage"] = 100.0 * tier_counts["count"] / totals.replace(0, np.nan)
        tier_counts["model_label"] = tier_counts["model"].map(_display_model)
    else:
        tier_counts = pd.DataFrame(columns=["model", "priority_class", "allocated_tier", "count", "percentage", "model_label"])

    if connection_parts:
        connection_counts = pd.concat(connection_parts, ignore_index=True)
        connection_counts = connection_counts.groupby(["model", "priority_class", "connection_type"], observed=True)["count"].sum().reset_index()
        totals = connection_counts.groupby(["model", "priority_class"], observed=True)["count"].transform("sum")
        connection_counts["percentage"] = 100.0 * connection_counts["count"] / totals.replace(0, np.nan)
        connection_counts["model_label"] = connection_counts["model"].map(_display_model)
    else:
        connection_counts = pd.DataFrame(columns=["model", "priority_class", "connection_type", "count", "percentage", "model_label"])

    if sample_parts:
        samples = _sample_by_group(pd.concat(sample_parts, ignore_index=True), sample_per_group)
        samples["model_label"] = samples["model"].map(_display_model)
        samples["priority_class"] = pd.Categorical(samples["priority_class"], PRIORITY_ORDER, ordered=True)
    else:
        samples = pd.DataFrame(columns=SAMPLE_COLUMNS + ["model_label"])

    if round_parts:
        round_summary = pd.concat(round_parts, ignore_index=True)
        round_summary = round_summary.groupby(["model", "priority_class", "round"], observed=True).sum(min_count=1).reset_index()
        round_summary["mean_delay_ms"] = round_summary["delay_sum"] / round_summary["delay_count"].replace(0, np.nan)
        round_summary["mean_battery_remaining_j"] = round_summary["battery_sum"] / round_summary["battery_count"].replace(0, np.nan)
        round_summary["mean_device_battery_after_j"] = round_summary["device_battery_after_sum"] / round_summary["device_battery_after_count"].replace(0, np.nan)
        round_summary["mean_device_battery_ratio"] = round_summary["device_battery_ratio_sum"] / round_summary["device_battery_ratio_count"].replace(0, np.nan)
        round_summary["mean_battery_drain_j"] = round_summary["battery_drain_sum"] / round_summary["battery_drain_count"].replace(0, np.nan)
        round_summary["handover_rate"] = 100.0 * round_summary["handover_sum"] / round_summary["handover_count"].replace(0, np.nan)
        round_summary["handover_failure_rate"] = 100.0 * round_summary["handover_failure_sum"] / round_summary["handover_failure_count"].replace(0, np.nan)
        round_summary["mean_chosen_rssi_dbm"] = round_summary["rssi_sum"] / round_summary["rssi_count"].replace(0, np.nan)
        round_summary["mean_distance_to_bs_m"] = round_summary["distance_sum"] / round_summary["distance_count"].replace(0, np.nan)
        round_summary["model_label"] = round_summary["model"].map(_display_model)
        round_summary["priority_class"] = pd.Categorical(round_summary["priority_class"], PRIORITY_ORDER, ordered=True)
        round_summary = round_summary.sort_values(["priority_class", "model", "round"])
    else:
        round_summary = pd.DataFrame(columns=[
            "model", "priority_class", "round", "mean_delay_ms", "mean_battery_remaining_j",
            "mean_device_battery_after_j", "mean_device_battery_ratio", "mean_battery_drain_j",
            "handover_rate", "handover_failure_rate", "mean_chosen_rssi_dbm", "mean_distance_to_bs_m", "model_label",
        ])
    return summary, tier_counts, samples, round_summary, connection_counts

def write_model_comparison_summary(root, run_models=None, logger=print):
    root = Path(root)
    evaluation_dir = root / "offloading" if (root / "offloading").is_dir() else root
    models = _discover_models(evaluation_dir, run_models)
    if not models:
        if logger:
            logger(f"No model comparison CSVs found under {evaluation_dir}")
        return None
    summary, tier_counts, _, _, _ = _aggregate_evaluation(evaluation_dir, models, chunksize=50000, sample_per_group=1000)
    out_dir = Path("visualization") / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "model_comparison_priority_summary.csv"
    tier_path = out_dir / "model_comparison_tier_priority_summary.csv"
    summary.to_csv(summary_path, index=False)
    tier_counts.to_csv(tier_path, index=False)
    if logger:
        logger(f"Wrote priority comparison summary: {summary_path}")
        logger(f"Wrote priority tier summary: {tier_path}")
    return summary_path

