"""Create the requested training figures for completed campaigns.

This module is intentionally narrow: it reads the simulator's existing training
CSV files and writes four figures per model: action by class, epsilon, reward by
class, and average reward.
"""

from __future__ import annotations

import json
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
from visualization.supportive.training_aggregate_plots import (
    FIGURE_SIZE,
    aggregate_seed_curves,
    align_seed_curves,
    discover_seed_runs,
    load_exploration_curves,
    load_reward_curves,
    plot_all_models_main_reward,
)

CLASS_COLORS = {
    "URLLC": "#2563eb",
    "mMTC": "#16a34a",
    "eMBB": "#dc2626",
}
OBSOLETE_MODEL_FIGURES = (
    "action_share_across_seeds.png",
    "exploration_decay.png",
    "main_reward_across_seeds.png",
    "reward_by_service_class.png",
)
ACTION_SOURCE_COLUMNS = (
    "selected_action",
    "rl_action",
    "action_final",
    "selected_tier",
    "allocated_tier",
    "execution_tier",
    "rl_tier_name",
    "offloading_tier",
)


def _read_training_csv(path: Path, delimiter: str, columns: Iterable[str]) -> pd.DataFrame:
    """Read available columns from one CSV; return empty data if it is absent."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        header = pd.read_csv(path, sep=delimiter, nrows=0)
        available = [column for column in columns if column in header.columns]
        if not available:
            return pd.DataFrame()
        return pd.read_csv(path, sep=delimiter, usecols=available)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric version of one column."""
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _normalise_action(value: object) -> str:
    """Map recorded action/tier values to the canonical action labels."""
    if pd.isna(value):
        return ""
    text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(numeric) and float(numeric).is_integer():
        index = int(numeric)
        if 0 <= index < len(ACTION_LABEL_LIST):
            return ACTION_LABEL_LIST[index]
    replacements = {
        "LOCAL": "LOCAL_DEVICE",
        "DEVICE": "LOCAL_DEVICE",
        "LOCAL_DEVICE": "LOCAL_DEVICE",
        "MEC": "MEC",
        "EDGE": "MEC",
        "MEC_COLLAB": "MEC_COLLAB",
        "COLLAB": "MEC_COLLAB",
        "METRO": "METRO",
        "CLOUD": "CLOUD",
        "UAV": "UAV_BS",
        "UAV_BS": "UAV_BS",
    }
    return replacements.get(text, text)


def _choose_action_column(frame: pd.DataFrame) -> str | None:
    """Return the first populated action/tier column available in a result log."""
    for column in ACTION_SOURCE_COLUMNS:
        if column in frame.columns and frame[column].notna().any():
            return column
    return None


def _decision_frames(phase_dir: str | Path, model: str) -> dict[str, pd.DataFrame]:
    """Load individual decision rows from each seed's simulator result log."""
    frames: dict[str, pd.DataFrame] = {}
    requested = (
        "round",
        "reward",
        "epsilon",
        "priority",
        "priority_class",
        "class_name",
        *ACTION_SOURCE_COLUMNS,
    )
    for seed_label, model_dir in discover_seed_runs(phase_dir, model).items():
        frame = _read_training_csv(model_dir / "results.csv", ";", requested)
        if frame.empty:
            frame = _read_training_csv(
                model_dir / f"{model}_offloading_results.csv",
                ";",
                requested,
            )
        if frame.empty:
            continue
        clean = frame.copy()
        clean["decision"] = range(1, len(clean) + 1)
        action_column = _choose_action_column(clean)
        if action_column is not None:
            clean["action_label"] = clean[action_column].map(_normalise_action)
        frames[seed_label] = clean
    return frames


def _filter_decision_class(
    frame: pd.DataFrame,
    priority: int,
    label: str,
) -> pd.DataFrame:
    """Return only the decision rows for one service class."""
    if "priority_class" in frame.columns:
        mask = frame["priority_class"].astype(str).str.lower() == label.lower()
    elif "class_name" in frame.columns:
        mask = frame["class_name"].astype(str).str.lower() == label.lower()
    elif "priority" in frame.columns:
        mask = _numeric(frame, "priority") == int(priority)
    else:
        return pd.DataFrame(columns=frame.columns)
    return frame.loc[mask].copy()


def _decision_value_curves(
    phase_dir: str | Path,
    model: str,
    value_column: str,
    priority: int | None = None,
    label: str | None = None,
) -> dict[str, pd.Series]:
    """Return per-seed value curves indexed by decision ordinal."""
    curves: dict[str, pd.Series] = {}
    for seed_label, frame in _decision_frames(phase_dir, model).items():
        data = frame
        if priority is not None and label is not None:
            data = _filter_decision_class(data, priority, label)
        if data.empty or value_column not in data.columns:
            continue
        values = _numeric(data, value_column).dropna()
        if values.empty:
            continue
        values.index = range(1, len(values) + 1)
        curves[seed_label] = values
    return curves


def _style_axis(axis: plt.Axes) -> None:
    """Apply the shared training figure style."""
    axis.grid(True, alpha=0.25)
    axis.set_axisbelow(True)


def _save_note(path: Path, title: str, note: str, dpi: int) -> Path:
    """Write a note figure when a metric is unavailable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, note, ha="center", va="center", wrap=True)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def _plot_seed_curves(
    axis: plt.Axes,
    curves: Mapping[str, pd.Series],
    moving_average_window: int,
    title: str,
    y_label: str,
    x_label: str = "Training decision",
) -> None:
    """Draw seed curves plus their equal-seed mean and observed range."""
    aligned = align_seed_curves(curves, moving_average_window)
    if aligned.empty:
        axis.axis("off")
        axis.set_title(title)
        axis.text(0.5, 0.5, "No recorded data", ha="center", va="center")
        return
    seed_colors = plt.get_cmap("tab10")
    for index, seed_label in enumerate(aligned.columns):
        axis.plot(
            aligned.index,
            aligned[seed_label],
            color=seed_colors(index % 10),
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
        linewidth=0,
        label="Observed seed range",
    )
    axis.plot(
        aggregate.index,
        aggregate["mean"],
        color="#0f172a",
        linewidth=2.2,
        label="Seed mean",
    )
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.legend(fontsize=8)
    _style_axis(axis)


def _action_frames_for_class(
    phase_dir: str | Path,
    model: str,
    priority: int,
    label: str,
) -> dict[str, pd.DataFrame]:
    """Load action-share frames for one service class from each seed."""
    frames: dict[str, pd.DataFrame] = {}
    requested = ("round", "priority", "priority_class", *ACTION_LABEL_LIST)
    for seed_label, model_dir in discover_seed_runs(phase_dir, model).items():
        frame = _read_training_csv(
            model_dir / f"{model}_action_distribution.csv",
            ",",
            requested,
        )
        if frame.empty or "round" not in frame.columns:
            continue
        if "priority_class" in frame.columns:
            class_rows = frame["priority_class"].astype(str).str.lower() == label.lower()
            frame = frame.loc[class_rows]
        elif "priority" in frame.columns:
            frame = frame.loc[_numeric(frame, "priority") == int(priority)]
        else:
            continue
        action_columns = [name for name in ACTION_LABEL_LIST if name in frame.columns]
        if not action_columns:
            continue
        clean = pd.DataFrame({"round": _numeric(frame, "round")})
        for action in action_columns:
            clean[action] = _numeric(frame, action).fillna(0.0)
        clean = clean.dropna(subset=["round"])
        clean = clean.groupby("round", as_index=False)[action_columns].mean()
        frames[seed_label] = clean.sort_values("round")
    return frames


def _action_decision_frames_for_class(
    phase_dir: str | Path,
    model: str,
    priority: int,
    label: str,
) -> dict[str, pd.DataFrame]:
    """Load one-hot action frames indexed by individual decisions."""
    frames: dict[str, pd.DataFrame] = {}
    for seed_label, frame in _decision_frames(phase_dir, model).items():
        class_rows = _filter_decision_class(frame, priority, label)
        if class_rows.empty or "action_label" not in class_rows.columns:
            continue
        class_rows = class_rows[class_rows["action_label"].isin(ACTION_LABEL_LIST)]
        if class_rows.empty:
            continue
        one_hot = pd.get_dummies(class_rows["action_label"]).reindex(
            columns=ACTION_LABEL_LIST,
            fill_value=0,
        )
        one_hot.insert(0, "decision", range(1, len(one_hot) + 1))
        frames[seed_label] = one_hot
    return frames


def _mean_action_frame(
    seed_frames: Mapping[str, pd.DataFrame],
    moving_average_window: int,
    x_column: str = "round",
) -> pd.DataFrame:
    """Return equal-seed mean rolling action shares by x-position."""
    actions = [
        action
        for action in ACTION_LABEL_LIST
        if any(action in frame.columns for frame in seed_frames.values())
    ]
    if not actions:
        return pd.DataFrame(columns=(x_column,))
    per_action: dict[str, pd.Series] = {}
    for action in actions:
        curves: dict[str, pd.Series] = {}
        for seed_label, frame in seed_frames.items():
            if action in frame.columns and x_column in frame.columns:
                curves[seed_label] = frame.set_index(x_column)[action]
        aligned = align_seed_curves(curves, moving_average_window)
        per_action[action] = aligned.mean(axis=1, skipna=True)
    result = pd.DataFrame(per_action).fillna(0.0).clip(lower=0.0)
    row_totals = result.sum(axis=1).replace(0.0, np.nan)
    result = result.div(row_totals, axis=0).fillna(0.0)
    result.index.name = x_column
    return result.reset_index()


def mean_action_frame(
    seed_frames: Mapping[str, pd.DataFrame],
    moving_average_window: int = 10,
) -> pd.DataFrame:
    """Public wrapper for tests and ad-hoc checks."""
    return _mean_action_frame(seed_frames, moving_average_window)


def _plot_action_stack(axis: plt.Axes, frame: pd.DataFrame, title: str) -> None:
    """Draw one stacked action-share panel."""
    actions = [name for name in ACTION_LABEL_LIST if name in frame.columns]
    if frame.empty or not actions:
        axis.axis("off")
        axis.set_title(title)
        axis.text(0.5, 0.5, "No action data", ha="center", va="center")
        return
    x_column = "decision" if "decision" in frame.columns else "round"
    x_label = "Decision within class" if x_column == "decision" else "Simulation round"
    axis.stackplot(
        frame[x_column],
        [frame[action].to_numpy(dtype=float) for action in actions],
        labels=actions,
        colors=[TIER_COLORS.get(action, "#94a3b8") for action in actions],
        alpha=0.9,
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Rolling action share")
    axis.set_title(title)
    _style_axis(axis)


def plot_action_by_class(
    phase_dir: str | Path,
    model: str,
    output_dir: str | Path,
    moving_average_window: int,
    dpi: int,
) -> Path:
    """Write the action figure: three service-class panels."""
    path = Path(output_dir) / "training_action_by_class.png"
    figure, axes = plt.subplots(1, 3, figsize=(15.6, 4.6), sharex=False, sharey=True)
    for axis, (priority, label) in zip(axes, PRIORITY_LABELS.items()):
        frames = _action_decision_frames_for_class(phase_dir, model, priority, label)
        x_column = "decision"
        if not frames:
            frames = _action_frames_for_class(phase_dir, model, priority, label)
            x_column = "round"
        mean_frame = _mean_action_frame(frames, moving_average_window, x_column=x_column)
        _plot_action_stack(axis, mean_frame, label)
    handles: list[object] = []
    labels: list[str] = []
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            break
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    figure.suptitle(f"{model.upper()} action selection by class")
    figure.tight_layout(rect=(0.0, 0.08 if handles else 0.0, 1.0, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def plot_epsilon(
    phase_dir: str | Path,
    model: str,
    output_dir: str | Path,
    dpi: int,
) -> Path:
    """Write the epsilon figure."""
    path = Path(output_dir) / "training_epsilon.png"
    curves = _decision_value_curves(phase_dir, model, "epsilon")
    x_label = "Training decision"
    if not curves:
        curves = load_exploration_curves(phase_dir, model)
        x_label = "Simulation round"
    if not curves:
        return _save_note(
            path,
            f"{model.upper()} epsilon",
            "This model did not record epsilon exploration.",
            dpi,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    _plot_seed_curves(axis, curves, 1, f"{model.upper()} epsilon", "Epsilon", x_label=x_label)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def plot_reward_by_class(
    phase_dir: str | Path,
    model: str,
    output_dir: str | Path,
    moving_average_window: int,
    dpi: int,
) -> Path:
    """Write the reward figure: one panel with one line per class."""
    path = Path(output_dir) / "training_reward_by_class.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    plotted = False
    for priority, label in PRIORITY_LABELS.items():
        curves = _decision_value_curves(phase_dir, model, "reward", priority, label)
        x_label = "Decision within class"
        if not curves:
            curves = load_reward_curves(phase_dir, model, priority=priority)
            x_label = "Simulation round"
        aggregate = aggregate_seed_curves(curves, moving_average_window)
        if aggregate.empty:
            continue
        color = CLASS_COLORS.get(label)
        axis.fill_between(
            aggregate.index,
            aggregate["minimum"].to_numpy(dtype=float),
            aggregate["maximum"].to_numpy(dtype=float),
            color=color,
            alpha=0.08,
            linewidth=0,
        )
        axis.plot(
            aggregate.index,
            aggregate["mean"],
            color=color,
            linewidth=2.0,
            label=label,
        )
        plotted = True
    if plotted:
        axis.set_xlabel(x_label)
        axis.set_ylabel("Reward")
        axis.legend(fontsize=8)
        _style_axis(axis)
    else:
        axis.axis("off")
        axis.text(0.5, 0.5, "No class reward data", ha="center", va="center")
    axis.set_title(f"{model.upper()} reward by class")
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def plot_average_reward(
    phase_dir: str | Path,
    model: str,
    output_dir: str | Path,
    moving_average_window: int,
    dpi: int,
) -> Path:
    """Write the average reward figure."""
    path = Path(output_dir) / "training_average_reward.png"
    curves = _decision_value_curves(phase_dir, model, "reward")
    x_label = "Training decision"
    if not curves:
        curves = load_reward_curves(phase_dir, model)
        x_label = "Simulation round"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    _plot_seed_curves(
        axis,
        curves,
        moving_average_window,
        f"{model.upper()} average reward",
        "Average reward",
        x_label=x_label,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def generate_model_figures(
    phase_dir: str | Path,
    model: str,
    moving_average_window: int = 500,
    dpi: int = 180,
) -> list[Path]:
    """Generate the four requested model-level training figures."""
    phase_path = Path(phase_dir)
    if not discover_seed_runs(phase_path, model):
        raise ValueError(f"No seed output found for model {model!r} under {phase_path}")
    output_dir = phase_path / "plots" / "by_model" / model
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in OBSOLETE_MODEL_FIGURES:
        (output_dir / filename).unlink(missing_ok=True)
    return [
        plot_action_by_class(phase_path, model, output_dir, moving_average_window, dpi),
        plot_epsilon(phase_path, model, output_dir, dpi),
        plot_reward_by_class(phase_path, model, output_dir, moving_average_window, dpi),
        plot_average_reward(phase_path, model, output_dir, moving_average_window, dpi),
    ]


def generate_campaign_summary(
    phase_dir: str | Path,
    models: Iterable[str] | None = None,
    moving_average_window: int = 500,
    dpi: int = 180,
) -> list[Path]:
    """Generate the cross-model reward figure and plot manifest."""
    phase_path = Path(phase_dir)
    selected_models = list(models) if models is not None else []
    comparison_window_rounds = max(1, min(int(moving_average_window), 10))
    comparison_path = plot_all_models_main_reward(
        phase_path,
        selected_models,
        phase_path / "plots" / "comparison",
        comparison_window_rounds,
        dpi,
    )
    all_figures = sorted((phase_path / "plots" / "by_model").glob("*/*.png"))
    all_figures.append(comparison_path)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase_dir": str(phase_path),
        "models": selected_models,
        "moving_average_window_decisions": int(moving_average_window),
        "comparison_moving_average_window_rounds": comparison_window_rounds,
        "moving_average_window_rounds": comparison_window_rounds,
        "figures": [str(path.relative_to(phase_path)) for path in all_figures],
    }
    manifest_path = phase_path / "plots" / "plot_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as file_handle:
        json.dump(manifest, file_handle, indent=2)
    return [comparison_path, manifest_path]


def generate_training_figures(
    phase_dir: str | Path,
    models: Iterable[str],
    moving_average_window: int = 500,
    dpi: int = 180,
) -> list[Path]:
    """Generate requested figures for every selected model plus manifest."""
    written: list[Path] = []
    for model in models:
        written.extend(
            generate_model_figures(
                phase_dir,
                str(model),
                moving_average_window=moving_average_window,
                dpi=dpi,
            )
        )
    written.extend(
        generate_campaign_summary(
            phase_dir,
            models,
            moving_average_window=moving_average_window,
            dpi=dpi,
        )
    )
    return written
