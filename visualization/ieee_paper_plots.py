"""Generate the individual IEEE paper panels used by this study.

Each PNG contains one publishable panel: no figure title and no ``(a)`` style
panel label. The filename and JSON manifest carry that context for LaTeX.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from visualization.plot_config import (
    ACTIVE_MODELS,
    COLORS_MODELS,
    MODEL_ALIASES,
    SERVICE_CLASSES,
    load_model_seeds,
    model_result_paths,
    plot_phase_dir,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "ieee_figures" / "zone_compact"
PANEL_SIZE = (3.5, 2.65)
HEATMAP_SIZE = (3.5, 2.9)
PNG_DPI = 300
ZONES = ("cellular", "uav", "remote", "all")
ZONE_SLUGS = {"cellular": "city", "uav": "uav_edge", "remote": "remote_satellite", "all": "all_zones"}
CLASS_LABELS = {"URLLC": "HRLLC", "mMTC": "MC", "eMBB": "IC"}
MARKERS = {"dqn": "o", "double_dqn": "s", "ppo": "^", "a2c": "D", "q_learning": "v", "random": "P"}
MODEL_SHORT = {"double_dqn": "DDQN", "q_learning": "QL", "random": "Ran."}
ALLOCATION_TIERS = ("LOCAL_DEVICE", "MEC", "CLOUD", "UAV_BS", "SAT")
TIER_COLORS = {
    "LOCAL_DEVICE": "#718096", "MEC": "#2563eb", "CLOUD": "#16a34a",
    "UAV_BS": "#dc2626", "SAT": "#f59e0b", "FAILED": "#9e9e9e",
}
TIER_ORDER = tuple(TIER_COLORS)

NEEDED_COLUMNS = (
    "round", "device_id", "class_name", "status", "success",
    "observed_complete_latency_ms", "total_energy_j", "battery_draw_j",
    "device_battery_ratio", "throughput_mbps", "active_devices_this_round",
    "allocated_tier", "payload_bytes", "operational_co2e_g", "zone_id",
)
NUMERIC_COLUMNS = (
    "round", "success", "observed_complete_latency_ms", "total_energy_j",
    "battery_draw_j", "device_battery_ratio", "throughput_mbps",
    "active_devices_this_round", "payload_bytes", "operational_co2e_g",
)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.2,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#b0b0b0",
    "grid.alpha": 0.32,
    "grid.linewidth": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def _model_label(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def _model_short_label(model: str) -> str:
    return MODEL_SHORT.get(model, _model_label(model))


def _model_color(model: str) -> str:
    return COLORS_MODELS.get(model, "#64748b")


def _completed(frame: pd.DataFrame) -> pd.Series:
    if "status" in frame:
        return frame["status"].astype(str).str.lower().eq("completed")
    return frame["success"].fillna(0).ge(1)


def load_results(phase: str | Path | None = None) -> tuple[pd.DataFrame, list[str], Path]:
    phase_path = plot_phase_dir("evaluation", explicit_phase=phase)
    models = [model for model in ACTIVE_MODELS if model_result_paths(model, phase_path)]
    if not models:
        raise FileNotFoundError(f"no configured model results found under {phase_path}")
    frames = []
    for model in models:
        frame = load_model_seeds(model, columns=NEEDED_COLUMNS, phase=phase_path)
        if frame is None or frame.empty:
            continue
        frame["model"] = model
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no result rows found under {phase_path}")
    data = pd.concat(frames, ignore_index=True)
    for column in NUMERIC_COLUMNS:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data["class_name"] = data["class_name"].astype(str)
    data["zone_id"] = data.get("zone_id", "cellular").astype(str).str.lower()
    data["allocated_tier"] = (
        data.get("allocated_tier", "UNKNOWN").fillna("UNKNOWN").astype(str).str.strip().str.upper()
        .replace({"LOCAL": "LOCAL_DEVICE", "UAV": "UAV_BS", "": "UNKNOWN", "NAN": "UNKNOWN"})
    )
    return data, models, phase_path


def _zone(frame: pd.DataFrame, zone: str) -> pd.DataFrame:
    return frame if zone == "all" else frame[frame["zone_id"] == zone]


def _value_curve(
    frame: pd.DataFrame,
    value_column: str,
    reducer: str | Callable[[pd.Series], float] = "mean",
    completed_only: bool = False,
    multiplier: float = 1.0,
) -> pd.Series:
    if frame.empty or value_column not in frame:
        return pd.Series(dtype=float)
    working = frame.loc[_completed(frame)] if completed_only else frame
    working = working.dropna(subset=["active_devices_this_round", value_column])
    if working.empty:
        return pd.Series(dtype=float)
    curve = working.groupby("active_devices_this_round", observed=True)[value_column].agg(reducer)
    return curve.sort_index().astype(float).mul(multiplier).rolling(3, min_periods=1).mean()


def _deadline_success_curve(frame: pd.DataFrame) -> pd.Series:
    working = frame.copy()
    working["_value"] = working["success"].fillna(0).mul(100.0)
    return _value_curve(working, "_value")


def _completion_curve(frame: pd.DataFrame) -> pd.Series:
    working = frame.copy()
    working["_value"] = _completed(working).astype(float).mul(100.0)
    return _value_curve(working, "_value")


def _p95_latency_curve(frame: pd.DataFrame) -> pd.Series:
    return _value_curve(
        frame, "observed_complete_latency_ms",
        reducer=lambda values: float(np.nanpercentile(values, 95)), completed_only=True,
    )


def _battery_draw_curve(frame: pd.DataFrame) -> pd.Series:
    column = "battery_draw_j" if frame["battery_draw_j"].notna().any() else "total_energy_j"
    return _value_curve(frame, column)


def _battery_left_curve(frame: pd.DataFrame) -> pd.Series:
    return _value_curve(frame, "device_battery_ratio", multiplier=100.0)


def _throughput_curve(frame: pd.DataFrame) -> pd.Series:
    return _value_curve(frame, "throughput_mbps", completed_only=True)


def _seed_curve_stats(
    frame: pd.DataFrame,
    service_class: str,
    curve_builder: Callable[[pd.DataFrame], pd.Series],
) -> pd.DataFrame:
    rows = []
    selected = frame[frame["class_name"] == service_class]
    for (model, seed), group in selected.groupby(["model", "seed"], observed=True):
        for x_value, value in curve_builder(group).items():
            if pd.notna(x_value) and pd.notna(value):
                rows.append({"model": str(model), "seed": int(seed), "x": float(x_value), "value": float(value)})
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows).groupby(["model", "x"], observed=True)["value"]
        .agg(mean="mean", std="std", count="count").reset_index().fillna({"std": 0.0})
    )


def _local_legend(ax: plt.Axes, handles=None, ncol: int = 2, loc: str = "best") -> None:
    legend = ax.legend(
        handles=handles, loc=loc, ncol=ncol, frameon=True, framealpha=0.88,
        facecolor="white", edgecolor="#bbbbbb", handlelength=1.8,
        handletextpad=0.35, columnspacing=0.7, borderpad=0.3,
    )
    if legend is not None:
        legend.get_frame().set_linewidth(0.5)


def _line_panel(
    frame: pd.DataFrame,
    models: list[str],
    service_class: str,
    curve_builder: Callable[[pd.DataFrame], pd.Series],
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=PANEL_SIZE, layout="constrained")
    stats = _seed_curve_stats(frame, service_class, curve_builder)
    draw_order = (["random"] if "random" in models else []) + [model for model in models if model != "random"]
    for draw_index, model in enumerate(draw_order):
        curve = stats[stats["model"] == model].sort_values("x") if not stats.empty else pd.DataFrame()
        if curve.empty:
            continue
        x = curve["x"].to_numpy(dtype=float)
        mean = curve["mean"].to_numpy(dtype=float)
        std = curve["std"].to_numpy(dtype=float)
        color = _model_color(model)
        marker_step = max(len(x) // 9, 1)
        line_width = 0.82 if model == "random" else 1.0
        line_alpha = 0.82 if model == "random" else 1.0
        ax.plot(
            x, mean, color=color, linestyle="-", linewidth=line_width, alpha=line_alpha,
            marker=MARKERS.get(model, "o"), markevery=(draw_index % marker_step, marker_step),
            markersize=2.5, markerfacecolor="white", markeredgewidth=0.65,
            zorder=2 + draw_index,
        )
        ax.fill_between(x, mean - std, mean + std, color=color,
                        alpha=0.09 if model == "random" else 0.14, linewidth=0,
                        zorder=1 + draw_index)
    ax.set_xlabel("Active devices")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.margins(x=0.03)
    handles = [Line2D([0], [0], color=_model_color(model), linestyle="-", linewidth=1.0,
                      marker=MARKERS.get(model, "o"), markersize=3.2, markerfacecolor="white",
                      markeredgewidth=0.65, label=_model_short_label(model)) for model in models]
    _local_legend(ax, handles=handles)
    return fig


def _action_seed_stats(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    for (model, seed), group in frame.groupby(["model", "seed"], observed=True):
        # A missing tier means that the task failed before an allocation was made.
        # Keep that outcome in the denominator so every bar represents all tasks.
        outcomes = group["allocated_tier"].where(
            group["allocated_tier"].isin(ALLOCATION_TIERS), "FAILED"
        )
        shares = outcomes.value_counts(normalize=True).mul(100.0)
        for action, value in shares.items():
            rows.append({"model": str(model), "seed": int(seed), "action": str(action), "value": float(value)})
    if not rows:
        return pd.DataFrame(), []
    stats = (
        pd.DataFrame(rows).groupby(["model", "action"], observed=True)["value"]
        .mean().reset_index(name="mean")
    )
    present = set(stats["action"])
    actions = [action for action in TIER_ORDER if action in present]
    actions.extend(sorted(present.difference(actions)))
    return stats, actions


def _action_panel(frame: pd.DataFrame, models: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=PANEL_SIZE, layout="constrained")
    stats, actions = _action_seed_stats(frame)
    x = np.arange(len(models))
    bottom = np.zeros(len(models), dtype=float)
    handles = []
    for action in actions:
        values = []
        for model in models:
            row = stats[(stats["model"] == model) & (stats["action"] == action)]
            values.append(float(row["mean"].iloc[0]) if not row.empty else 0.0)
        bars = ax.bar(
            x, values, bottom=bottom, width=0.72,
            color=TIER_COLORS.get(action, "#94a3b8"), edgecolor="white", linewidth=0.35,
            label=action.replace("_", " "),
        )
        handles.append(bars)
        if action == "FAILED":
            for bar, value, base in zip(bars, values, bottom):
                if value >= 4.0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        base + value / 2,
                        f"{value:.0f}%",
                        ha="center", va="center", fontsize=5.2, color="#222222",
                    )
        bottom += np.asarray(values)
    ax.set_ylim(0.0, 116.0)
    ax.set_yticks(np.arange(0.0, 101.0, 20.0))
    ax.set_xticks(x)
    ax.set_xticklabels([_model_short_label(model) for model in models], rotation=0, ha="center")
    ax.set_ylabel("Action share (%)")
    ax.grid(axis="x", visible=False)
    action_labels = {
        "LOCAL_DEVICE": "Local", "MEC": "MEC", "CLOUD": "Cloud",
        "UAV_BS": "UAV", "SAT": "Sat.", "FAILED": "Failed",
    }
    legend = ax.legend(
        handles=handles, labels=[action_labels.get(action, action) for action in actions],
        loc="upper center", ncol=max(len(actions), 1), fontsize=5.2,
        frameon=True, framealpha=0.9, facecolor="white", edgecolor="#bbbbbb",
        handlelength=1.15, handletextpad=0.25, columnspacing=0.45,
        borderpad=0.25, labelspacing=0.2,
    )
    legend.get_frame().set_linewidth(0.5)
    return fig


ENVIRONMENT_METRICS = (
    ("energy_per_success_j", "Energy per deadline success (J)", "YlOrBr"),
    ("co2e_per_success_mg", r"Operational CO$_2$e per deadline success (mg)", "OrRd"),
    ("ontime_mbit_per_j", "On-time payload efficiency (Mbit/J)", "YlGnBu"),
    ("carbon_efficiency_mbit_per_g", r"Carbon efficiency (Mbit/g CO$_2$e)", "PuBuGn"),
)


def _environment_values(group: pd.DataFrame) -> dict[str, float]:
    success = group["success"].fillna(0).ge(1)
    successes = int(success.sum())
    energy = float(group["total_energy_j"].sum(min_count=1))
    co2e = float(group["operational_co2e_g"].sum(min_count=1))
    ontime_mbit = float(group.loc[success, "payload_bytes"].fillna(0).sum()) * 8.0e-6
    return {
        "deadline_success_pct": 100.0 * successes / len(group) if len(group) else np.nan,
        "energy_per_success_j": energy / successes if successes else np.nan,
        "co2e_per_success_mg": 1000.0 * co2e / successes if successes else np.nan,
        "ontime_mbit_per_j": ontime_mbit / energy if energy > 0 else np.nan,
        "carbon_efficiency_mbit_per_g": ontime_mbit / co2e if co2e > 0 else np.nan,
    }


def _environment_seed_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, service_class, seed), group in frame.groupby(
        ["model", "class_name", "seed"], observed=True
    ):
        rows.append({
            "model": str(model), "class_name": str(service_class), "seed": int(seed),
            **_environment_values(group),
        })
    return pd.DataFrame(rows)


def _metric_stats(seed_table: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        seed_table.dropna(subset=[metric]).groupby(["model", "class_name"], observed=True)[metric]
        .agg(mean="mean", minimum="min", maximum="max", count="count").reset_index()
    )


def _number(value: float) -> str:
    magnitude = abs(value)
    if magnitude and (magnitude < 0.01 or magnitude >= 1000):
        return f"{value:.1e}"
    if magnitude < 1:
        return f"{value:.3f}"
    if magnitude < 100:
        return f"{value:.2f}"
    return f"{value:.1f}"


def _heatmap_panel(seed_table: pd.DataFrame, models: list[str], metric: str,
                   colorbar_label: str, cmap_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=HEATMAP_SIZE, layout="constrained")
    stats = _metric_stats(seed_table, metric)
    classes = [service_class for service_class in SERVICE_CLASSES
               if service_class in set(stats["class_name"].astype(str))]
    means = np.full((len(models), len(classes)), np.nan)
    minima = np.full_like(means, np.nan)
    maxima = np.full_like(means, np.nan)
    counts = np.zeros_like(means)
    for i, model in enumerate(models):
        for j, service_class in enumerate(classes):
            row = stats[(stats["model"] == model) & (stats["class_name"] == service_class)]
            if row.empty:
                continue
            means[i, j] = float(row["mean"].iloc[0])
            minima[i, j] = float(row["minimum"].iloc[0])
            maxima[i, j] = float(row["maximum"].iloc[0])
            counts[i, j] = int(row["count"].iloc[0])
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#eeeeee")
    image_values = np.ma.masked_invalid(means)
    norm = None
    if metric == "carbon_efficiency_mbit_per_g":
        positive = means[np.isfinite(means) & (means > 0)]
        if positive.size:
            vmin = 10.0 ** np.floor(np.log10(float(positive.min())))
            vmax = 10.0 ** np.ceil(np.log10(float(positive.max())))
            if np.isclose(vmin, vmax):
                vmax = vmin * 10.0
            norm = LogNorm(vmin=vmin, vmax=vmax)
            image_values = np.ma.masked_where(~np.isfinite(means) | (means <= 0), means)
    im = ax.imshow(image_values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([CLASS_LABELS.get(service_class, service_class) for service_class in classes])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([_model_label(model) for model in models])
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for (i, j), mean in np.ndenumerate(means):
        if not np.isfinite(mean):
            continue
        text = _number(mean)
        if counts[i, j] > 1:
            text += f"\n[{_number(minima[i, j])}, {_number(maxima[i, j])}]"
        shade = im.norm(mean)
        ax.text(j, i, text, ha="center", va="center", fontsize=5.2,
                color="white" if np.isfinite(shade) and shade >= 0.58 else "#111111")
    colorbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.06)
    colorbar.set_label(colorbar_label, fontsize=7)
    colorbar.ax.tick_params(labelsize=6, length=2)
    colorbar.outline.set_visible(False)
    return fig


def _tradeoff_panel(seed_table: pd.DataFrame, models: list[str], service_class: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=PANEL_SIZE, layout="constrained")
    x_values = []
    for index, model in enumerate(models):
        rows = seed_table[(seed_table["model"] == model) &
                          (seed_table["class_name"] == service_class)].dropna(
                              subset=["co2e_per_success_mg", "deadline_success_pct"]
                          )
        if rows.empty:
            continue
        xs = rows["co2e_per_success_mg"].to_numpy(dtype=float)
        ys = rows["deadline_success_pct"].to_numpy(dtype=float)
        x, y = float(xs.mean()), float(ys.mean())
        x_values.append(x)
        ax.errorbar(
            x, y,
            xerr=[[x - float(xs.min())], [float(xs.max()) - x]],
            yerr=[[y - float(ys.min())], [float(ys.max()) - y]],
            fmt=MARKERS.get(model, "o"), markersize=max(7.5 - 0.5 * index, 4.5),
            markerfacecolor=_model_color(model), markeredgecolor="#222222",
            markeredgewidth=0.55, ecolor=_model_color(model), elinewidth=0.75,
            capsize=2, label=_model_label(model), zorder=3 + index,
        )
    positive = [value for value in x_values if value > 0]
    if len(positive) >= 2 and max(positive) / min(positive) >= 5:
        ax.set_xscale("log")
    ax.set_xlabel(r"Operational CO$_2$e per deadline success (mg)")
    ax.set_ylabel("Deadline success (%)")
    ax.set_ylim(0.0, 102.0)
    ax.margins(x=0.15)
    _local_legend(ax)
    return fig


def _save(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def _remove_grouped_outputs(output_dir: Path) -> None:
    obsolete = (
        "fig1_zone_qos_reliability_8panel_std.png",
        "fig2_zone_energy_sustainability_8panel_std.png",
        "fig3_zone_scalability_8panel_std.png",
        "fig4_zone_action_allocation_4panel.png",
        "fig5_environmental_performance_4panel_range.png",
        "fig6_carbon_qos_tradeoff_3panel_range.png",
        "zone_compact_manifest.json",
    )
    for filename in obsolete:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def generate_ieee_panels(
    phase: str | Path | None = None,
    output_dir: str | Path = OUTPUT_DIR,
) -> list[Path]:
    data, models, phase_path = load_results(phase)
    destination = Path(output_dir)
    _remove_grouped_outputs(destination)
    written: list[Path] = []
    labels: dict[str, str] = {}

    line_specs = (
        ("qos_deadline_success", "URLLC", _deadline_success_curve, "Deadline success (%)", (0.0, 100.0)),
        ("qos_p95_latency", "URLLC", _p95_latency_curve, "P95 latency (ms)", None),
        ("energy_battery_draw", "mMTC", _battery_draw_curve, "Battery draw (J)", None),
        ("energy_battery_remaining", "mMTC", _battery_left_curve, "Battery remaining (%)", (0.0, 100.0)),
        ("scalability_completion_ratio", "eMBB", _completion_curve, "Completion ratio (%)", (0.0, 100.0)),
        ("scalability_throughput", "eMBB", _throughput_curve, "Throughput (Mbit/s)", None),
    )
    for metric_slug, service_class, builder, ylabel, ylim in line_specs:
        for zone in ZONES:
            filename = f"{metric_slug}_{ZONE_SLUGS[zone]}.png"
            panel_data = _zone(data, zone)
            written.append(_save(
                _line_panel(panel_data, models, service_class, builder, ylabel, ylim),
                destination, filename,
            ))
            labels[filename] = f"{ylabel} for {CLASS_LABELS[service_class]} in {ZONE_SLUGS[zone].replace('_', ' ')}"

    for zone in ZONES:
        filename = f"action_allocation_{ZONE_SLUGS[zone]}.png"
        written.append(_save(_action_panel(_zone(data, zone), models), destination, filename))
        labels[filename] = f"Action allocation in {ZONE_SLUGS[zone].replace('_', ' ')}"

    seed_table = _environment_seed_table(data)
    metric_slugs = {
        "energy_per_success_j": "environment_energy_per_deadline_success",
        "co2e_per_success_mg": "environment_operational_co2e_per_deadline_success",
        "ontime_mbit_per_j": "environment_ontime_payload_efficiency",
        "carbon_efficiency_mbit_per_g": "environment_carbon_efficiency",
    }
    for metric, colorbar_label, cmap_name in ENVIRONMENT_METRICS:
        filename = f"{metric_slugs[metric]}.png"
        written.append(_save(
            _heatmap_panel(seed_table, models, metric, colorbar_label, cmap_name),
            destination, filename,
        ))
        labels[filename] = colorbar_label.replace("$", "")

    for service_class in SERVICE_CLASSES:
        class_slug = CLASS_LABELS.get(service_class, service_class).lower()
        filename = f"tradeoff_operational_co2e_vs_deadline_success_{class_slug}.png"
        written.append(_save(_tradeoff_panel(seed_table, models, service_class), destination, filename))
        labels[filename] = f"Operational CO2e versus deadline success for {CLASS_LABELS.get(service_class, service_class)}"

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase_dir": str(phase_path),
        "models": models,
        "seed_count": int(data["seed"].nunique()),
        "aggregation": {
            "load_curves": "line is the cross-seed mean; band is +/-1 seed standard deviation",
            "environment_heatmaps": "cell is the cross-seed mean; brackets are observed seed minimum and maximum",
            "tradeoff": "point is the cross-seed mean; bars are observed seed minimum and maximum",
        },
        "panels": labels,
    }
    manifest_path = destination / "ieee_panel_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    written.append(manifest_path)
    return written


def main() -> None:
    phase = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    written = generate_ieee_panels(phase)
    print(f"Wrote {len(written) - 1} individual IEEE panels to:")
    print(f"  {OUTPUT_DIR}")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()




