"""Create campaign figures that compare seeds and models without changing simulation.

The simulator writes one folder per seed and model.  This module reads those completed
CSV logs, reduces every seed to one value per simulation round, and then displays the
individual seed curves together with their mean and observed min--max range.

How to extend: add a focused ``plot_*`` function and register it in
``generate_training_figures``; never recalculate simulator physics in this module.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scheduler.scheduler_helper.training_visualization_helper import (
    ACTION_LABEL_LIST,
    PRIORITY_LABELS,
    TIER_COLORS,
)


FIGURE_SIZE = (8.4, 5.2)
MODEL_ORDER = (
    "random",
    "rule_based",
    "q_learning",
    "dqn",
    "double_dqn",
    "ppo",
    "a2c",
)
MODEL_COLORS = {
    "random": "#64748b",
    "rule_based": "#0f766e",
    "q_learning": "#9333ea",
    "dqn": "#2563eb",
    "double_dqn": "#0891b2",
    "ppo": "#dc2626",
    "a2c": "#ea580c",
}


def _seed_number(seed_dir: Path) -> int:
    """Return the numeric part of ``seed_<n>`` for stable human ordering."""
    suffix = seed_dir.name.rsplit("seed_", 1)[-1]
    return int(suffix) if suffix.isdigit() else 1 << 30


def discover_seed_runs(phase_dir: str | Path, model: str) -> dict[str, Path]:
    """Return ordered seed labels and model folders that contain training output."""
    phase_path = Path(phase_dir)
    runs: dict[str, Path] = {}
    flat_model_dir = phase_path / "offloading" / str(model)
    if flat_model_dir.is_dir():
        # Flat single-seed training run: <run>/offloading/<model>.
        runs["seed_1"] = flat_model_dir
    for seed_dir in sorted(phase_path.glob("seed_*"), key=_seed_number):
        model_dir = seed_dir / "offloading" / str(model)
        if model_dir.is_dir():
            runs[seed_dir.name] = model_dir
    return runs


def discover_models(phase_dir: str | Path) -> list[str]:
    """Return model names found under any seed, using the standard display order."""
    names: set[str] = set()
    for seed_dir in [Path(phase_dir), *Path(phase_dir).glob("seed_*")]:
        offloading_dir = seed_dir / "offloading"
        if offloading_dir.is_dir():
            names.update(path.name for path in offloading_dir.iterdir() if path.is_dir())
    order = {name: index for index, name in enumerate(MODEL_ORDER)}
    return sorted(names, key=lambda name: (order.get(name, len(order)), name))


def _read_available_columns(path: Path, delimiter: str, requested: Iterable[str]) -> pd.DataFrame:
    """Read only requested CSV columns that exist, returning an empty frame if absent."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        header = pd.read_csv(path, sep=delimiter, nrows=0)
        available = [name for name in requested if name in header.columns]
        if not available:
            return pd.DataFrame()
        return pd.read_csv(path, sep=delimiter, usecols=available)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _read_training_metrics(model_dir: Path, metric_group: str, requested: Iterable[str]) -> pd.DataFrame:
    """Read one metric group from consolidated training_metrics.csv."""
    columns = ("metric_group", *tuple(requested))
    frame = _read_available_columns(model_dir / "training_metrics.csv", ",", columns)
    if frame.empty or "metric_group" not in frame.columns:
        return pd.DataFrame()
    group_rows = frame["metric_group"].astype(str) == str(metric_group)
    frame = frame.loc[group_rows, [name for name in requested if name in frame.columns]]
    return frame.reset_index(drop=True)


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return one CSV column as numeric values, coercing blanks to missing values."""
    if name not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _round_series(frame: pd.DataFrame, value_column: str) -> pd.Series:
    """Reduce task rows to one mean value per round for equal seed weighting."""
    if frame.empty or "round" not in frame.columns or value_column not in frame.columns:
        return pd.Series(dtype=float)
    reduced = frame.copy()
    reduced["round"] = _numeric_column(reduced, "round")
    reduced[value_column] = _numeric_column(reduced, value_column)
    reduced = reduced.dropna(subset=["round", value_column])
    if reduced.empty:
        return pd.Series(dtype=float)
    return reduced.groupby("round")[value_column].mean().sort_index()


def load_reward_curves(
    phase_dir: str | Path,
    model: str,
    priority: int | None = None,
) -> dict[str, pd.Series]:
    """Load per-seed round-average reward, optionally for one service priority."""
    curves: dict[str, pd.Series] = {}
    for seed_label, model_dir in discover_seed_runs(phase_dir, model).items():
        frame = _read_training_metrics(model_dir, "reward", ("round", "reward", "priority"))
        if frame.empty:
            reward_path = model_dir / f"{model}_reward_log.csv"
            frame = _read_available_columns(
                reward_path, ",", ("round", "reward", "priority")
            )
        if frame.empty:
            frame = _read_available_columns(
                model_dir / "results.csv", ";", ("round", "reward", "priority")
            )
        if priority is not None:
            if "priority" not in frame.columns:
                continue
            priority_values = _numeric_column(frame, "priority")
            frame = frame.loc[priority_values == int(priority)]
        curve = _round_series(frame, "reward")
        if not curve.empty:
            curves[seed_label] = curve
    return curves


def load_exploration_curves(phase_dir: str | Path, model: str) -> dict[str, pd.Series]:
    """Load the actual epsilon trace recorded by each seed, when the model provides it."""
    curves: dict[str, pd.Series] = {}
    for seed_label, model_dir in discover_seed_runs(phase_dir, model).items():
        frame = _read_training_metrics(model_dir, "epsilon", ("round", "epsilon"))
        if frame.empty:
            frame = _read_available_columns(
                model_dir / f"{model}_epsilon_decay.csv", ",", ("round", "epsilon")
            )
        curve = _round_series(frame, "epsilon")
        if not curve.empty:
            curves[seed_label] = curve
    return curves


def load_action_frames(phase_dir: str | Path, model: str) -> dict[str, pd.DataFrame]:
    """Load one all-priority action-share frame per seed from existing training logs."""
    frames: dict[str, pd.DataFrame] = {}
    requested = ("round", "priority", "priority_class", *ACTION_LABEL_LIST)
    for seed_label, model_dir in discover_seed_runs(phase_dir, model).items():
        frame = _read_training_metrics(model_dir, "action_distribution", requested)
        if frame.empty:
            path = model_dir / f"{model}_action_distribution.csv"
            frame = _read_available_columns(path, ",", requested)
        if frame.empty or "round" not in frame.columns:
            continue
        if "priority_class" in frame.columns:
            all_rows = frame["priority_class"].astype(str).str.upper() == "ALL"
            frame = frame.loc[all_rows]
        elif "priority" in frame.columns:
            frame = frame.loc[frame["priority"].astype(str).str.upper() == "ALL"]
        action_columns = [name for name in ACTION_LABEL_LIST if name in frame.columns]
        if not action_columns:
            continue
        clean = pd.DataFrame({"round": _numeric_column(frame, "round")})
        for action in action_columns:
            clean[action] = _numeric_column(frame, action).fillna(0.0)
        clean = clean.dropna(subset=["round"])
        clean = clean.groupby("round", as_index=False)[action_columns].mean()
        frames[seed_label] = clean.sort_values("round")
    return frames


def _completed_rows(frame: pd.DataFrame) -> pd.Series:
    """Identify application completion without treating network-only status as universal."""
    if "status" in frame.columns:
        return frame["status"].astype(str).str.lower() == "completed"
    if "network_status" in frame.columns:
        status = frame["network_status"].astype(str).str.lower()
        return status.isin(("delivered", "local_bypass"))
    return pd.Series(False, index=frame.index)


def load_delay_curves(
    phase_dir: str | Path,
    model: str,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Load completed-task delay and all-task deadline-miss rate for every seed."""
    delay_curves: dict[str, pd.Series] = {}
    miss_curves: dict[str, pd.Series] = {}
    requested = (
        "round",
        "status",
        "network_status",
        "observed_complete_latency_ms",
        "deadline_missed",
    )
    for seed_label, model_dir in discover_seed_runs(phase_dir, model).items():
        frame = _read_available_columns(model_dir / "results.csv", ";", requested)
        if frame.empty or "round" not in frame.columns:
            continue
        frame["round"] = _numeric_column(frame, "round")
        completed = frame.loc[_completed_rows(frame)].copy()
        if "observed_complete_latency_ms" in completed.columns:
            completed["observed_complete_latency_ms"] = _numeric_column(
                completed, "observed_complete_latency_ms"
            )
            curve = _round_series(completed, "observed_complete_latency_ms")
            if not curve.empty:
                delay_curves[seed_label] = curve
        if "deadline_missed" in frame.columns:
            miss_values = frame["deadline_missed"].replace(
                {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}
            )
            frame["deadline_missed"] = pd.to_numeric(miss_values, errors="coerce")
            curve = _round_series(frame, "deadline_missed")
            if not curve.empty:
                miss_curves[seed_label] = curve
    return delay_curves, miss_curves


def align_seed_curves(
    curves: Mapping[str, pd.Series],
    moving_average_window: int = 1,
) -> pd.DataFrame:
    """Align seed curves by round and smooth each seed without filling missing rounds."""
    window = max(1, int(moving_average_window))
    aligned_parts: list[pd.Series] = []
    for seed_label, curve in curves.items():
        clean = pd.to_numeric(curve, errors="coerce").sort_index()
        clean = clean.rolling(window, min_periods=1).mean()
        aligned_parts.append(clean.rename(seed_label))
    if not aligned_parts:
        return pd.DataFrame()
    return pd.concat(aligned_parts, axis=1).sort_index()


def aggregate_seed_curves(
    curves: Mapping[str, pd.Series],
    moving_average_window: int = 1,
) -> pd.DataFrame:
    """Return per-round mean, observed minimum, maximum, and available seed count."""
    aligned = align_seed_curves(curves, moving_average_window)
    if aligned.empty:
        return pd.DataFrame(columns=("mean", "minimum", "maximum", "seed_count"))
    return pd.DataFrame(
        {
            "mean": aligned.mean(axis=1, skipna=True),
            "minimum": aligned.min(axis=1, skipna=True),
            "maximum": aligned.max(axis=1, skipna=True),
            "seed_count": aligned.count(axis=1),
        }
    )


def mean_action_frame(
    seed_frames: Mapping[str, pd.DataFrame],
    moving_average_window: int = 10,
) -> pd.DataFrame:
    """Return the across-seed mean rolling action share for each training round."""
    actions = [
        action
        for action in ACTION_LABEL_LIST
        if any(action in frame.columns for frame in seed_frames.values())
    ]
    if not actions:
        return pd.DataFrame(columns=("round",))
    per_action: dict[str, pd.Series] = {}
    for action in actions:
        curves: dict[str, pd.Series] = {}
        for seed_label, frame in seed_frames.items():
            if action not in frame.columns:
                continue
            curves[seed_label] = frame.set_index("round")[action]
        aligned = align_seed_curves(curves, moving_average_window)
        per_action[action] = aligned.mean(axis=1, skipna=True)
    result = pd.DataFrame(per_action).fillna(0.0).clip(lower=0.0)
    row_totals = result.sum(axis=1).replace(0.0, np.nan)
    result = result.div(row_totals, axis=0).fillna(0.0)
    result.index.name = "round"
    return result.reset_index()


def _style_axis(axis: plt.Axes) -> None:
    """Apply the shared readable style used by every aggregate figure."""
    axis.grid(True, alpha=0.25)
    axis.set_axisbelow(True)


def _save_note(path: Path, title: str, note: str, dpi: int) -> Path:
    """Write an explanatory figure when an algorithm does not record a metric."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, note, ha="center", va="center", wrap=True)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def _validate_plot_options(moving_average_window: int, dpi: int) -> None:
    """Reject invalid rendering options before any output files are created."""
    if moving_average_window < 1:
        raise ValueError("moving_average_window must be at least 1")
    if dpi < 1:
        raise ValueError("dpi must be positive")


def generate_model_figures(
    phase_dir: str | Path,
    model: str,
    moving_average_window: int = 10,
    dpi: int = 180,
) -> list[Path]:
    """Generate the four aggregate figures (epsilon, reward, reward per class, actions) for one completed model campaign."""
    _validate_plot_options(moving_average_window, dpi)
    phase_path = Path(phase_dir)
    if not discover_seed_runs(phase_path, model):
        raise ValueError(f"No seed output found for model {model!r} under {phase_path}")

    model_output = phase_path / "plots" / "by_model" / model
    return [
        plot_exploration_decay(phase_path, model, model_output, dpi),
        plot_main_reward_across_seeds(
            phase_path, model, model_output, moving_average_window, dpi
        ),
        plot_reward_by_service_class(
            phase_path, model, model_output, moving_average_window, dpi
        ),
        plot_action_share_across_seeds(
            phase_path, model, model_output, moving_average_window, dpi
        ),
    ]


def generate_campaign_summary(
    phase_dir: str | Path,
    models: Iterable[str] | None = None,
    moving_average_window: int = 10,
    dpi: int = 180,
) -> list[Path]:
    """Generate the cross-model reward figure and a manifest of campaign plots."""
    _validate_plot_options(moving_average_window, dpi)
    phase_path = Path(phase_dir)
    selected_models = list(models) if models is not None else discover_models(phase_path)
    if not selected_models:
        raise ValueError(f"No offloading model folders found under {phase_path}")

    plot_root = phase_path / "plots"
    comparison_path = plot_all_models_main_reward(
        phase_path,
        selected_models,
        plot_root / "comparison",
        moving_average_window,
        dpi,
    )
    all_figures = sorted((plot_root / "by_model").glob("*/*.png"))
    all_figures.append(comparison_path)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase_dir": str(phase_path),
        "models": selected_models,
        "moving_average_window_rounds": int(moving_average_window),
        "range_definition": "observed min-max across available seed-level round means",
        "figures": [str(path.relative_to(phase_path)) for path in all_figures],
    }
    manifest_path = plot_root / "plot_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as file_handle:
        json.dump(manifest, file_handle, indent=2)
    return [comparison_path, manifest_path]


def generate_training_figures(
    phase_dir: str | Path,
    models: Iterable[str] | None = None,
    moving_average_window: int = 10,
    dpi: int = 180,
) -> list[Path]:
    """Generate all model-level and cross-model figures for a completed campaign."""
    phase_path = Path(phase_dir)
    selected_models = list(models) if models is not None else discover_models(phase_path)
    if not selected_models:
        raise ValueError(f"No offloading model folders found under {phase_path}")
    _validate_plot_options(moving_average_window, dpi)

    written: list[Path] = []
    for model in selected_models:
        written.extend(
            generate_model_figures(
                phase_path,
                model,
                moving_average_window=moving_average_window,
                dpi=dpi,
            )
        )
    written.extend(
        generate_campaign_summary(
            phase_path,
            selected_models,
            moving_average_window=moving_average_window,
            dpi=dpi,
        )
    )
    return written


def _plot_seed_curves(
    axis: plt.Axes,
    curves: Mapping[str, pd.Series],
    moving_average_window: int,
    title: str,
    y_label: str,
) -> None:
    """Draw individual seed curves, their equal-seed mean, and observed range."""
    aligned = align_seed_curves(curves, moving_average_window)
    if aligned.empty:
        axis.axis("off")
        axis.set_title(title)
        axis.text(0.5, 0.5, "No recorded data", ha="center", va="center")
        return
    colors = plt.get_cmap("tab10")
    for index, seed_label in enumerate(aligned.columns):
        axis.plot(
            aligned.index,
            aligned[seed_label],
            color=colors(index % 10),
            alpha=0.72,
            linewidth=1.0,
            label=seed_label.replace("_", " ").title(),
        )
    aggregate = aggregate_seed_curves(curves, moving_average_window)
    axis.fill_between(
        aggregate.index,
        aggregate["minimum"].to_numpy(dtype=float),
        aggregate["maximum"].to_numpy(dtype=float),
        color="#0f172a",
        alpha=0.10,
        label="Observed seed range",
    )
    axis.plot(
        aggregate.index,
        aggregate["mean"],
        color="#0f172a",
        linewidth=2.2,
        label="Seed mean",
    )
    axis.set_xlabel("Simulation round")
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.legend(fontsize=8)
    _style_axis(axis)


def plot_exploration_decay(
    phase_dir: str | Path,
    model: str,
    output_dir: str | Path,
    dpi: int = 180,
) -> Path:
    """Plot recorded epsilon traces; write an honest note when none were recorded."""
    path = Path(output_dir) / "exploration_decay.png"
    curves = load_exploration_curves(phase_dir, model)
    if not curves:
        return _save_note(
            path,
            f"{model.upper()} exploration",
            "This model did not record epsilon exploration; no decay curve was fabricated.",
            dpi,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    _plot_seed_curves(axis, curves, 1, f"{model.upper()} exploration decay", "Epsilon")
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def plot_main_reward_across_seeds(
    phase_dir: str | Path,
    model: str,
    output_dir: str | Path,
    moving_average_window: int = 10,
    dpi: int = 180,
) -> Path:
    """Plot every seed's main reward with equal-seed mean and min--max range."""
    path = Path(output_dir) / "main_reward_across_seeds.png"
    curves = load_reward_curves(phase_dir, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    _plot_seed_curves(
        axis,
        curves,
        moving_average_window,
        f"{model.upper()} main reward across seeds",
        "Main reward",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def plot_reward_by_service_class(
    phase_dir: str | Path,
    model: str,
    output_dir: str | Path,
    moving_average_window: int = 10,
    dpi: int = 180,
) -> Path:
    """Plot the real main reward grouped by service priority, never a synthetic reward."""
    path = Path(output_dir) / "reward_by_service_class.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15.6, 4.6), sharex=True, sharey=True)
    for axis, (priority, label) in zip(axes, PRIORITY_LABELS.items()):
        curves = load_reward_curves(phase_dir, model, priority=priority)
        _plot_seed_curves(
            axis,
            curves,
            moving_average_window,
            label,
            "Main reward" if axis is axes[0] else "",
        )
    figure.suptitle(f"{model.upper()} reward by service class")
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def _smoothed_action_frame(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Smooth one seed's categorical shares while preserving a unit row sum."""
    actions = [name for name in ACTION_LABEL_LIST if name in frame.columns]
    if not actions:
        return pd.DataFrame(columns=("round",))
    smoothed = frame[["round", *actions]].copy().set_index("round")
    smoothed = smoothed.rolling(max(1, int(window)), min_periods=1).mean()
    totals = smoothed.sum(axis=1).replace(0.0, np.nan)
    smoothed = smoothed.div(totals, axis=0).fillna(0.0)
    return smoothed.reset_index()


def _plot_action_stack(axis: plt.Axes, frame: pd.DataFrame, title: str) -> None:
    """Draw rolling categorical shares as a stacked area rather than numeric action IDs."""
    actions = [name for name in ACTION_LABEL_LIST if name in frame.columns]
    if frame.empty or not actions:
        axis.axis("off")
        axis.set_title(title)
        axis.text(0.5, 0.5, "No recorded actions", ha="center", va="center")
        return
    values = [frame[action].to_numpy(dtype=float) for action in actions]
    colors = [TIER_COLORS.get(action, "#94a3b8") for action in actions]
    axis.stackplot(frame["round"], values, labels=actions, colors=colors, alpha=0.9)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Simulation round")
    axis.set_ylabel("Rolling action share")
    axis.set_title(title)
    _style_axis(axis)


def plot_action_share_across_seeds(
    phase_dir: str | Path,
    model: str,
    output_dir: str | Path,
    moving_average_window: int = 10,
    dpi: int = 180,
) -> Path:
    """Plot one action-share panel per seed plus an equal-seed aggregate panel."""
    path = Path(output_dir) / "action_share_across_seeds.png"
    frames = load_action_frames(phase_dir, model)
    panel_count = max(1, len(frames) + 1)
    column_count = 2 if panel_count > 1 else 1
    row_count = math.ceil(panel_count / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(8.0 * column_count, 4.5 * row_count),
        squeeze=False,
    )
    flat_axes = list(axes.ravel())
    panel_index = 0
    for seed_label, frame in frames.items():
        _plot_action_stack(
            flat_axes[panel_index],
            _smoothed_action_frame(frame, moving_average_window),
            seed_label.replace("_", " ").title(),
        )
        panel_index += 1
    _plot_action_stack(
        flat_axes[panel_index],
        mean_action_frame(frames, moving_average_window),
        "Equal-seed mean",
    )
    panel_index += 1
    for axis in flat_axes[panel_index:]:
        axis.axis("off")
    handles: list[object] = []
    labels: list[str] = []
    for axis in flat_axes[:panel_index]:
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            break
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
        figure.subplots_adjust(bottom=0.12)
    figure.suptitle(f"{model.upper()} rolling action shares across seeds")
    figure.tight_layout(rect=(0.0, 0.06 if handles else 0.0, 1.0, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def plot_delay_across_seeds(
    phase_dir: str | Path,
    model: str,
    output_dir: str | Path,
    moving_average_window: int = 10,
    dpi: int = 180,
) -> Path:
    """Plot completed-task delay beside the all-task deadline-miss rate."""
    path = Path(output_dir) / "delay_across_seeds.png"
    delay_curves, miss_curves = load_delay_curves(phase_dir, model)
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.0))
    _plot_seed_curves(
        axes[0],
        delay_curves,
        moving_average_window,
        "Completed applications",
        "Completion delay (ms)",
    )
    _plot_seed_curves(
        axes[1],
        miss_curves,
        moving_average_window,
        "All offered tasks",
        "Deadline-miss rate",
    )
    if axes[1].axison:
        axes[1].set_ylim(-0.02, 1.02)
    figure.suptitle(f"{model.upper()} delay and deadline behavior across seeds")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def plot_all_models_main_reward(
    phase_dir: str | Path,
    models: Iterable[str],
    output_dir: str | Path,
    moving_average_window: int = 10,
    dpi: int = 180,
) -> Path:
    """Plot one equal-seed mean main-reward curve per model with light seed ranges."""
    path = Path(output_dir) / "all_models_main_reward.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10.2, 6.0))
    plotted = 0
    fallback_colors = plt.get_cmap("tab10")
    for index, model in enumerate(models):
        curves = load_reward_curves(phase_dir, model)
        aggregate = aggregate_seed_curves(curves, moving_average_window)
        if aggregate.empty:
            continue
        color = MODEL_COLORS.get(model, fallback_colors(index % 10))
        axis.fill_between(
            aggregate.index,
            aggregate["minimum"].to_numpy(dtype=float),
            aggregate["maximum"].to_numpy(dtype=float),
            color=color,
            alpha=0.08,
        )
        axis.plot(
            aggregate.index,
            aggregate["mean"],
            color=color,
            linewidth=2.0,
            label=model.replace("_", " ").upper(),
        )
        plotted += 1
    if plotted == 0:
        axis.axis("off")
        axis.text(0.5, 0.5, "No model reward logs found", ha="center", va="center")
    else:
        axis.set_xlabel("Simulation round")
        axis.set_ylabel("Mean main reward across seeds")
        axis.legend(fontsize=8, ncol=2)
        _style_axis(axis)
    axis.set_title("All-model reward comparison")
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path
