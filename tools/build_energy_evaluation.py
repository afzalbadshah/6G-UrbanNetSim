"""Build a compact, source-backed energy evaluation from simulator CSV results.

The report deliberately separates device-battery debit from whole-system energy.
It is intended for fresh evaluation runs, not training logs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


SERVICE_CLASSES = ("URLLC", "mMTC", "eMBB")
MODEL_LABELS = {
    "dqn": "DQN",
    "double_dqn": "Double DQN",
    "ppo": "PPO",
    "a2c": "A2C",
    "q_learning": "Q-Learning",
    "random": "Random",
}
NUMERIC_COLUMNS = (
    "round",
    "battery_draw_j",
    "battery_capacity_j",
    "device_battery_before_j",
    "device_battery_after_j",
    "device_battery_ratio",
    "battery_remaining_j",
    "vehicle_battery_energy_j",
    "device_battery_energy_j",
    "device_paid_energy_j",
    "device_tx_energy_j",
    "device_rx_energy_j",
    "device_wait_energy_j",
    "device_local_compute_energy_j",
    "device_propulsion_energy_j",
    "infrastructure_network_energy_j",
    "backhaul_energy_j",
    "infra_compute_energy_j",
    "proc_energy_j",
    "compute_it_energy_j",
    "facility_overhead_energy_j",
    "server_battery_energy_j",
    "server_hover_energy_j",
    "communication_energy_j",
    "objective_energy_j",
    "total_energy_j",
    "energy",
    "tx_energy_proxy",
)
TEXT_COLUMNS = (
    "device_id",
    "device_type",
    "service_class",
    "processing_tier",
)
REQUIRED_COLUMNS = set(NUMERIC_COLUMNS + TEXT_COLUMNS)


INK = colors.HexColor("#202020")
MUTED = colors.HexColor("#6A6761")
LINE = colors.HexColor("#DED9D0")
PAPER = colors.HexColor("#FFFEFC")
GREEN = colors.HexColor("#009B1A")
RED = colors.HexColor("#C7362F")
BLUE = colors.HexColor("#356A94")
WARM = colors.HexColor("#F5F1EA")


def _fmt_energy(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    if value == 0:
        return "0"
    if abs(value) < 0.001:
        return f"{value:.2e}"
    if abs(value) < 0.1:
        return f"{value:.5f}"
    return f"{value:.3f}"


def _fmt_percent(value: float, digits: int = 1) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}%"


def _read_manifest(run_dir: Path) -> dict:
    path = run_dir / "evaluation" / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _result_paths(run_dir: Path) -> dict[str, list[Path]]:
    paths: dict[str, list[Path]] = {}
    for path in sorted((run_dir / "evaluation").glob("seed_*/offloading/*/results.csv")):
        paths.setdefault(path.parent.name, []).append(path)
    return paths


def _load_model(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        seed = path.parents[2].name
        frame = pd.read_csv(
            path,
            sep=";",
            usecols=lambda column: column in REQUIRED_COLUMNS,
            low_memory=False,
        )
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        frame["seed"] = seed
        frame["source_path"] = str(path)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    for column in NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _class_summary(frame: pd.DataFrame, service_class: str) -> dict:
    group = frame.loc[frame["service_class"].eq(service_class)].copy()
    radio = group["device_tx_energy_j"] + group["device_rx_energy_j"]
    system_components = (
        group["device_tx_energy_j"]
        + group["device_rx_energy_j"]
        + group["device_wait_energy_j"]
        + group["infrastructure_network_energy_j"]
        + group["device_local_compute_energy_j"]
        + group["compute_it_energy_j"]
        + group["facility_overhead_energy_j"]
    )
    battery_components = (
        radio
        + group["device_wait_energy_j"]
        + group["device_local_compute_energy_j"]
        + group["device_propulsion_energy_j"]
    )

    ordered = group.sort_values(["seed", "device_id", "round"], kind="stable")
    lifecycle = ordered.groupby(["seed", "device_id"], sort=False).agg(
        service_class=("service_class", "first"),
        device_type=("device_type", "first"),
        capacity_j=("battery_capacity_j", "first"),
        before_j=("device_battery_before_j", "first"),
        after_j=("device_battery_after_j", "last"),
        task_draw_j=("battery_draw_j", "sum"),
        tasks=("round", "size"),
    )
    valid_capacity = lifecycle["capacity_j"].gt(0)
    lifecycle = lifecycle.loc[valid_capacity].copy()
    lifecycle["start_pct"] = lifecycle["before_j"] / lifecycle["capacity_j"] * 100.0
    lifecycle["end_pct"] = lifecycle["after_j"] / lifecycle["capacity_j"] * 100.0
    lifecycle["depletion_pct_points"] = lifecycle["start_pct"] - lifecycle["end_pct"]
    lifecycle["observed_depletion_j"] = lifecycle["before_j"] - lifecycle["after_j"]
    lifecycle["non_task_depletion_j"] = lifecycle["observed_depletion_j"] - lifecycle["task_draw_j"]

    means = {
        "radio": float(radio.mean()),
        "wait": float(group["device_wait_energy_j"].mean()),
        "network": float(group["infrastructure_network_energy_j"].mean()),
        "compute_it": float(group["compute_it_energy_j"].mean()),
        "facility": float(group["facility_overhead_energy_j"].mean()),
        "local_compute": float(group["device_local_compute_energy_j"].mean()),
        "propulsion": float(group["device_propulsion_energy_j"].mean()),
        "battery": float(group["battery_draw_j"].mean()),
        "objective": float(group["objective_energy_j"].mean()),
        "total": float(group["total_energy_j"].mean()),
    }
    battery_share = means["battery"] / means["total"] * 100.0 if means["total"] else math.nan
    return {
        "tasks": int(len(group)),
        "rounds": int(group["round"].max()),
        "means": means,
        "battery_share_pct": battery_share,
        "battery_component_max_abs_error_j": float((group["battery_draw_j"] - battery_components).abs().max()),
        "system_component_max_abs_error_j": float((group["total_energy_j"] - system_components).abs().max()),
        "device_instances": int(len(lifecycle)),
        "lifecycle_start_mean_pct": float(lifecycle["start_pct"].mean()),
        "lifecycle_end_mean_pct": float(lifecycle["end_pct"].mean()),
        "lifecycle_depletion_mean_pp": float(lifecycle["depletion_pct_points"].mean()),
        "lifecycle_depletion_median_pp": float(lifecycle["depletion_pct_points"].median()),
        "empty_device_pct": float(lifecycle["end_pct"].le(0.01).mean() * 100.0),
        "total_task_battery_draw_j": float(group["battery_draw_j"].sum()),
        "empty_device_types": lifecycle.loc[lifecycle["end_pct"].le(0.01), "device_type"].value_counts().to_dict(),
        "non_task_depletion_j": float(lifecycle["non_task_depletion_j"].sum()),
        "lifecycle_draw_reconciliation_max_abs_j": float(
            (lifecycle["observed_depletion_j"] - lifecycle["task_draw_j"]).abs().max()
        ),
        "energy_field_means": {
            column: float(group[column].mean()) for column in NUMERIC_COLUMNS if "energy" in column or column in {"battery_draw_j", "proc_energy_j"}
        },
        "alias_max_abs_errors_j": {
            "battery_draw_vs_device_battery": float((group["battery_draw_j"] - group["device_battery_energy_j"]).abs().max()),
            "device_paid_vs_device_battery": float((group["device_paid_energy_j"] - group["device_battery_energy_j"]).abs().max()),
            "proc_vs_infra_compute": float((group["proc_energy_j"] - group["infra_compute_energy_j"]).abs().max()),
            "backhaul_vs_infra_network": float((group["backhaul_energy_j"] - group["infrastructure_network_energy_j"]).abs().max()),
            "energy_vs_objective": float((group["energy"] - group["objective_energy_j"]).abs().max()),
            "tx_proxy_vs_device_tx": float((group["tx_energy_proxy"] - group["device_tx_energy_j"]).abs().max()),
            "vehicle_battery_vs_remaining": float((group["vehicle_battery_energy_j"] - group["battery_remaining_j"]).abs().max()),
        },
    }


def _model_summary(frame: pd.DataFrame) -> dict:
    return {
        "tasks": int(len(frame)),
        "rounds": int(frame["round"].max()),
        "seeds": sorted(frame["seed"].unique().tolist()),
        "classes": {service: _class_summary(frame, service) for service in SERVICE_CLASSES},
        "null_energy_values": int(frame[list(NUMERIC_COLUMNS)].isna().sum().sum()),
        "negative_energy_values": int((frame[list(NUMERIC_COLUMNS)] < 0).sum().sum()),
    }


def _energy_identity(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    keys = ["seed", "round", "device_id", "device_type", "service_class"]
    metrics = [
        "battery_draw_j",
        "device_battery_after_j",
        "device_battery_ratio",
        "device_tx_energy_j",
        "device_rx_energy_j",
        "device_wait_energy_j",
        "device_local_compute_energy_j",
        "infrastructure_network_energy_j",
        "compute_it_energy_j",
        "facility_overhead_energy_j",
        "objective_energy_j",
        "total_energy_j",
    ]
    l = left.sort_values(keys, kind="stable").reset_index(drop=True)
    r = right.sort_values(keys, kind="stable").reset_index(drop=True)
    same_shape = len(l) == len(r)
    same_keys = same_shape and l[keys].equals(r[keys])
    if not same_keys:
        return {"same_rows": False, "all_energy_metrics_identical": False, "max_abs_difference_j": math.nan}
    differences = (l[metrics] - r[metrics]).abs().to_numpy(dtype=float)
    max_difference = float(np.nanmax(differences)) if differences.size else 0.0
    same_tier = l["processing_tier"].fillna("").equals(r["processing_tier"].fillna(""))
    return {
        "same_rows": True,
        "same_processing_tiers": bool(same_tier),
        "all_energy_metrics_identical": bool(max_difference == 0.0),
        "max_abs_difference_j": max_difference,
    }


def analyze(run_dir: Path, focus_model: str) -> tuple[dict, dict[str, pd.DataFrame]]:
    manifest = _read_manifest(run_dir)
    paths = _result_paths(run_dir)
    if focus_model not in paths:
        raise FileNotFoundError(f"No results.csv found for focus model {focus_model!r} in {run_dir}")
    frames = {model: _load_model(model_paths) for model, model_paths in paths.items()}
    summaries = {model: _model_summary(frame) for model, frame in frames.items()}
    expected_models = list(manifest.get("models", []))
    expected_seeds = [f"seed_{i}" for i in range(1, int(manifest.get("repetitions", 0)) + 1)]
    found_seeds = sorted({seed for summary in summaries.values() for seed in summary["seeds"]})
    quality = {
        "expected_models": expected_models,
        "found_models": sorted(frames),
        "missing_models": sorted(set(expected_models).difference(frames)),
        "expected_seeds": expected_seeds,
        "found_seeds": found_seeds,
        "missing_seeds": sorted(set(expected_seeds).difference(found_seeds)),
        "incomplete_round_models": {
            model: summary["rounds"] for model, summary in summaries.items() if summary["rounds"] < 500
        },
    }
    identity = None
    if "dqn" in frames and "double_dqn" in frames:
        identity = _energy_identity(frames["dqn"], frames["double_dqn"])
    analysis = {
        "run_dir": str(run_dir.resolve()),
        "manifest": manifest,
        "focus_model": focus_model,
        "models": summaries,
        "quality": quality,
        "dqn_double_dqn_identity": identity,
    }
    return analysis, frames


def _set_font(pdf: canvas.Canvas, size: float, bold: bool = False, mono: bool = False, color=INK) -> None:
    if mono:
        font = "Courier-Bold" if bold else "Courier"
    else:
        font = "Helvetica-Bold" if bold else "Helvetica"
    pdf.setFont(font, size)
    pdf.setFillColor(color)


def _draw_wrapped(pdf: canvas.Canvas, text: str, x: float, y: float, width: float, size: float, leading: float, color=MUTED, bold=False) -> float:
    words = text.split()
    lines: list[str] = []
    current = ""
    font = "Helvetica-Bold" if bold else "Helvetica"
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or stringWidth(candidate, font, size) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    _set_font(pdf, size, bold=bold, color=color)
    for line in lines:
        pdf.drawString(x, y, line)
        y -= leading
    return y


def _page_footer(pdf: canvas.Canvas, page: int, source: str) -> None:
    _set_font(pdf, 7.2, mono=True, color=MUTED)
    pdf.drawString(42, 24, f"SOURCE  {source[:96]}")
    pdf.drawRightString(570, 24, f"PAGE {page}")


def _draw_header(pdf: canvas.Canvas, title: str, subtitle: str, metadata: str) -> None:
    _set_font(pdf, 24, bold=True)
    pdf.drawString(42, 748, title)
    _draw_wrapped(pdf, subtitle, 42, 722, 525, 11.3, 16, color=MUTED)
    _set_font(pdf, 8.3, mono=True, color=INK)
    pdf.drawString(42, 682, metadata)


def _draw_energy_card(pdf: canvas.Canvas, y: float, service: str, summary: dict) -> None:
    x, width, height = 42, 528, 168
    pdf.setFillColor(PAPER)
    pdf.setStrokeColor(LINE)
    pdf.roundRect(x, y, width, height, 9, fill=1, stroke=1)
    means = summary["means"]
    _set_font(pdf, 14, bold=True)
    pdf.drawString(x + 15, y + height - 26, service)
    _set_font(pdf, 10.5, bold=True, mono=True)
    pdf.drawRightString(x + width - 15, y + height - 24, f"{_fmt_energy(means['total'])} J/task system")
    _set_font(pdf, 9.2, mono=True, color=GREEN)
    pdf.drawString(x + 15, y + height - 50, f"BATTERY DEBIT  {_fmt_energy(means['battery'])} J/task")
    _set_font(pdf, 8.5, mono=True, color=MUTED)
    pdf.drawRightString(x + width - 15, y + height - 50, f"{_fmt_percent(summary['battery_share_pct'])} of system energy")

    rows = [
        ("Device radio (TX + RX)", means["radio"]),
        ("Device waiting", means["wait"]),
        ("Network / backhaul", means["network"]),
        ("Local compute (device)", means["local_compute"]),
        ("Remote compute IT", means["compute_it"]),
        ("Remote facility overhead", means["facility"]),
    ]
    row_y = y + height - 74
    for label, value in rows:
        _set_font(pdf, 8.7, color=MUTED)
        pdf.drawString(x + 15, row_y, label)
        _set_font(pdf, 8.7, mono=True, color=INK)
        pdf.drawRightString(x + 300, row_y, f"{_fmt_energy(value)} J/task")
        row_y -= 15

    _set_font(pdf, 8.5, color=MUTED)
    pdf.drawString(x + 326, y + height - 78, "Battery-powered instances")
    _set_font(pdf, 8.7, mono=True, color=INK)
    pdf.drawRightString(x + width - 15, y + height - 78, f"{summary['device_instances']:,}")
    _set_font(pdf, 8.5, color=MUTED)
    pdf.drawString(x + 326, y + height - 105, "Mean end-of-run battery")
    _set_font(pdf, 8.7, mono=True, color=INK)
    pdf.drawRightString(x + width - 15, y + height - 105, _fmt_percent(summary["lifecycle_end_mean_pct"]))
    _set_font(pdf, 8.5, color=MUTED)
    pdf.drawString(x + 326, y + height - 132, "Mean lifecycle depletion")
    _set_font(pdf, 8.7, mono=True, color=INK)
    pdf.drawRightString(x + width - 15, y + height - 132, f"{summary['lifecycle_depletion_mean_pp']:.2f} pp")


def _draw_lifecycle_card(pdf: canvas.Canvas, x: float, y: float, width: float, service: str, summary: dict) -> None:
    height = 126
    pdf.setFillColor(PAPER)
    pdf.setStrokeColor(LINE)
    pdf.roundRect(x, y, width, height, 8, fill=1, stroke=1)
    _set_font(pdf, 12.5, bold=True)
    pdf.drawString(x + 12, y + height - 22, service)
    _set_font(pdf, 7.4, mono=True, color=MUTED)
    pdf.drawRightString(x + width - 12, y + height - 21, f"n={summary['device_instances']:,}")
    rows = [
        ("Start battery", _fmt_percent(summary["lifecycle_start_mean_pct"])),
        ("End battery", _fmt_percent(summary["lifecycle_end_mean_pct"])),
        ("Depletion", f"{summary['lifecycle_depletion_mean_pp']:.2f} pp"),
        ("Empty devices", _fmt_percent(summary["empty_device_pct"])),
        ("Task debit total", f"{_fmt_energy(summary['total_task_battery_draw_j'])} J"),
    ]
    row_y = y + height - 44
    for label, value in rows:
        _set_font(pdf, 7.8, color=MUTED)
        pdf.drawString(x + 12, row_y, label)
        _set_font(pdf, 7.8, mono=True, color=INK)
        pdf.drawRightString(x + width - 12, row_y, value)
        row_y -= 16


def _draw_comparison_table(pdf: canvas.Canvas, x: float, y_top: float, width: float, analysis: dict) -> float:
    _set_font(pdf, 14, bold=True)
    pdf.drawString(x, y_top, "Battery consumption by policy")
    _set_font(pdf, 8.3, color=MUTED)
    pdf.drawString(x, y_top - 17, "Mean originating-device battery debit (J/task); lower is better.")
    y = y_top - 38
    columns = [x, x + 142, x + 237, x + 332, x + 427]
    headers = ["Model", "URLLC", "mMTC", "eMBB", "eMBB total"]
    pdf.setFillColor(WARM)
    pdf.rect(x, y - 6, width, 23, fill=1, stroke=0)
    for column, header in zip(columns, headers):
        _set_font(pdf, 7.8, bold=True, mono=(header != "Model"), color=INK)
        pdf.drawString(column + 6, y + 2, header)
    y -= 20
    order = ["dqn", "double_dqn", "ppo", "q_learning", "random", "a2c"]
    for index, model in enumerate(order):
        if model not in analysis["models"]:
            continue
        summary = analysis["models"][model]
        if index % 2 == 0:
            pdf.setFillColor(colors.HexColor("#FAF8F4"))
            pdf.rect(x, y - 6, width, 20, fill=1, stroke=0)
        row_color = RED if model == analysis["focus_model"] else INK
        values = [
            MODEL_LABELS.get(model, model),
            _fmt_energy(summary["classes"]["URLLC"]["means"]["battery"]),
            _fmt_energy(summary["classes"]["mMTC"]["means"]["battery"]),
            _fmt_energy(summary["classes"]["eMBB"]["means"]["battery"]),
            _fmt_energy(summary["classes"]["eMBB"]["means"]["total"]),
        ]
        for column, value in zip(columns, values):
            _set_font(pdf, 7.8, bold=(model == analysis["focus_model"]), mono=(column != columns[0]), color=row_color)
            pdf.drawString(column + 6, y, value)
        if summary["rounds"] < 500:
            _set_font(pdf, 6.8, mono=True, color=MUTED)
            pdf.drawRightString(x + width - 6, y, f"R{summary['rounds']}")
        y -= 20
    pdf.setStrokeColor(LINE)
    pdf.line(x, y + 11, x + width, y + 11)
    return y


def _relative_lower(a: float, b: float) -> float:
    return (b - a) / b * 100.0 if b else math.nan


def _draw_metric_inventory(pdf: canvas.Canvas, analysis: dict, focus_summary: dict, metadata: str) -> None:
    _draw_header(
        pdf,
        "Energy Metric Inventory",
        "Every reported DQN energy quantity is shown below with its accounting scope. Aliases are identified so they are not accidentally added twice.",
        metadata,
    )
    x, width, y = 42, 528, 642
    columns = [x, x + 278, x + 368, x + 458]
    pdf.setFillColor(WARM)
    pdf.rect(x, y - 6, width, 24, fill=1, stroke=0)
    for column, header in zip(columns, ["Canonical metric", "URLLC", "mMTC", "eMBB"]):
        _set_font(pdf, 7.8, bold=True, mono=(header != "Canonical metric"))
        pdf.drawString(column + 6, y + 2, header)
    y -= 21
    metric_rows = [
        ("battery_draw_j", "Device task battery debit"),
        ("device_tx_energy_j", "Device radio TX"),
        ("device_rx_energy_j", "Device radio RX"),
        ("device_wait_energy_j", "Device idle/wait"),
        ("device_local_compute_energy_j", "Local device compute"),
        ("device_propulsion_energy_j", "Task-row propulsion"),
        ("infrastructure_network_energy_j", "Infrastructure network"),
        ("compute_it_energy_j", "Remote compute IT"),
        ("facility_overhead_energy_j", "Remote facility/PUE"),
        ("server_battery_energy_j", "Server battery debit (subset)"),
        ("server_hover_energy_j", "Task-row server hover"),
        ("communication_energy_j", "Radio + network subtotal"),
        ("objective_energy_j", "RL optimization energy"),
        ("total_energy_j", "Whole-system task total"),
    ]
    for index, (field, label) in enumerate(metric_rows):
        if index % 2 == 0:
            pdf.setFillColor(colors.HexColor("#FAF8F4"))
            pdf.rect(x, y - 6, width, 20, fill=1, stroke=0)
        _set_font(pdf, 7.6, color=INK)
        pdf.drawString(x + 6, y, label)
        _set_font(pdf, 6.5, mono=True, color=MUTED)
        pdf.drawString(x + 137, y, field)
        for column, service in zip(columns[1:], SERVICE_CLASSES):
            value = focus_summary["classes"][service]["energy_field_means"].get(field, math.nan)
            _set_font(pdf, 7.6, mono=True, color=INK)
            pdf.drawString(column + 6, y, _fmt_energy(value))
        y -= 20
    pdf.setStrokeColor(LINE)
    pdf.line(x, y + 11, x + width, y + 11)

    _set_font(pdf, 13, bold=True)
    pdf.drawString(42, y - 16, "How to interpret the battery fields")
    notes = [
        ("battery_draw_j", "Actual per-task debit from the originating device battery. This is the primary battery-consumption metric."),
        ("device_battery_before_j / after_j", "Battery stock immediately before and after the task; their difference equals battery_draw_j for task charging."),
        ("device_battery_ratio", "State of charge after the task, equal to after_j divided by capacity_j."),
        ("vehicle_battery_energy_j", "Despite its name, this is remaining battery stock (an alias of battery_remaining_j), not energy consumed."),
        ("Round-level propulsion / hover", "Charged separately from task battery_draw_j. It can make lifecycle depletion exceed the sum of task debits."),
    ]
    note_y = y - 38
    for field, explanation in notes:
        _set_font(pdf, 7.7, bold=True, mono=True, color=INK)
        pdf.drawString(48, note_y, field)
        note_y = _draw_wrapped(pdf, explanation, 226, note_y, 334, 7.7, 10, color=MUTED)
        note_y -= 7

    pdf.setFillColor(WARM)
    pdf.setStrokeColor(LINE)
    pdf.roundRect(42, 62, 528, 72, 8, fill=1, stroke=1)
    _set_font(pdf, 9.2, bold=True)
    pdf.drawString(55, 116, "Alias and double-counting guardrail")
    alias_text = (
        "device_paid_energy_j = device_battery_energy_j = battery_draw_j; "
        "backhaul_energy_j = infrastructure_network_energy_j; proc_energy_j = infra_compute_energy_j; "
        "energy = objective_energy_j; tx_energy_proxy = device_tx_energy_j. server_battery_energy_j, server_hover_energy_j, and "
        "communication_energy_j are overlapping or separately charged quantities, not extra components of total_energy_j."
    )
    _draw_wrapped(pdf, alias_text, 55, 99, 500, 7.8, 11, color=MUTED)
    _page_footer(pdf, 3, "metrics/energy.py + scheduler/common/comparison_runner.py")


def build_pdf(analysis: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    focus = analysis["focus_model"]
    focus_summary = analysis["models"][focus]
    manifest = analysis["manifest"]
    seeds = focus_summary["seeds"]
    metadata = (
        f"MODEL {MODEL_LABELS.get(focus, focus)}   PHASE evaluation   BAND FR3 (6G)   "
        f"ROUNDS {focus_summary['rounds']}   SEED(S) {','.join(seeds)}   TASKS {focus_summary['tasks']:,}"
    )
    pdf = canvas.Canvas(str(output_path), pagesize=letter)
    pdf.setTitle("6G Energy Anatomy - Fresh Evaluation")
    pdf.setAuthor("SmartCityEdgeSim evaluation")

    _draw_header(
        pdf,
        "6G Energy Anatomy",
        "Where DQN energy goes, separated into originating-device battery consumption and whole-system task energy - the fresh evaluation run, not training.",
        metadata,
    )
    card_y = [494, 315, 136]
    for y, service in zip(card_y, SERVICE_CLASSES):
        _draw_energy_card(pdf, y, service, focus_summary["classes"][service])
    _set_font(pdf, 7.4, color=MUTED)
    pdf.drawString(42, 115, "System ledger is additive: radio + waiting + network/backhaul + local compute + remote IT + facility overhead.")
    pdf.drawString(42, 103, "Battery debit includes radio + waiting + local compute only; infrastructure and remote compute do not deplete the originating device.")
    _page_footer(pdf, 1, f"run_004/evaluation/seed_1/offloading/{focus}/results.csv")
    pdf.showPage()

    _draw_header(
        pdf,
        "DQN Battery Depletion & Policy Position",
        "Lifecycle battery loss is measured per device from its first observed before-state to its last observed after-state. Cross-policy values remain provisional until the planned three-seed run is complete.",
        metadata,
    )
    card_width = 166
    for index, service in enumerate(SERVICE_CLASSES):
        _draw_lifecycle_card(pdf, 42 + index * 181, 530, card_width, service, focus_summary["classes"][service])
    table_bottom = _draw_comparison_table(pdf, 42, 498, 528, analysis)

    dqn = analysis["models"].get("dqn")
    ql = analysis["models"].get("q_learning")
    rnd = analysis["models"].get("random")
    findings: list[tuple[str, str, object]] = []
    if dqn and ql:
        q_values = []
        for service in SERVICE_CLASSES:
            a = dqn["classes"][service]["means"]["battery"]
            b = ql["classes"][service]["means"]["battery"]
            q_values.append(f"{service} {_relative_lower(a, b):.1f}%")
        findings.append(("DQN vs Q-Learning battery debit", ", ".join(q_values) + " lower", GREEN))
    if dqn and rnd:
        r_values = []
        for service in SERVICE_CLASSES:
            a = dqn["classes"][service]["means"]["battery"]
            b = rnd["classes"][service]["means"]["battery"]
            r_values.append(f"{service} {_relative_lower(a, b):.1f}%")
        findings.append(("DQN vs Random battery debit", ", ".join(r_values) + " lower", GREEN))
    if dqn:
        embb = dqn["classes"]["eMBB"]
        findings.append(("eMBB trade-off", f"battery is {_fmt_percent(embb['battery_share_pct'])} of {_fmt_energy(embb['means']['total'])} J/task total system energy", BLUE))
        urllc = dqn["classes"]["URLLC"]
        if urllc["empty_device_types"]:
            empty = ", ".join(f"{name} ({count})" for name, count in urllc["empty_device_types"].items())
            findings.append(("Lifecycle depletion outlier", f"{empty} reaches 0%; {urllc['non_task_depletion_j']:,.0f} J is outside task debits and reflects round-level energy", RED))
    identity = analysis.get("dqn_double_dqn_identity") or {}
    if identity.get("all_energy_metrics_identical"):
        findings.append(("DQN and Double DQN", "energy rows and processing tiers are exactly identical in this result set", RED))

    _set_font(pdf, 14, bold=True)
    y = table_bottom - 12
    pdf.drawString(42, y, "What the fresh result says")
    y -= 24
    for label, value, color in findings:
        pdf.setFillColor(color)
        pdf.circle(48, y + 3, 3, fill=1, stroke=0)
        _set_font(pdf, 8.8, bold=True, color=INK)
        pdf.drawString(59, y, label)
        y = _draw_wrapped(pdf, value, 229, y, 337, 8.8, 12, color=MUTED)
        y -= 7

    quality = analysis["quality"]
    pdf.setFillColor(WARM)
    pdf.setStrokeColor(LINE)
    pdf.roundRect(42, 72, 528, 86, 8, fill=1, stroke=1)
    _set_font(pdf, 9.4, bold=True, color=INK)
    pdf.drawString(55, 139, "Result-status warning")
    warning = (
        f"Manifest expects {len(quality['expected_seeds'])} seeds and {len(quality['expected_models'])} models. "
        f"Available: {len(quality['found_seeds'])} seed and {len(quality['found_models'])} models; "
        f"missing seeds: {', '.join(quality['missing_seeds']) or 'none'}; "
        f"missing models: {', '.join(quality['missing_models']) or 'none'}. "
        f"Incomplete rounds: {quality['incomplete_round_models'] or 'none'}. Descriptive DQN values are valid for the observed seed, "
        "but no across-seed confidence interval or significance claim is justified yet."
    )
    _draw_wrapped(pdf, warning, 55, 123, 500, 7.9, 11, color=MUTED)
    _page_footer(pdf, 2, f"manifest {manifest.get('experiment_id', 'unknown')} | run_004 fresh evaluation")
    pdf.showPage()

    _draw_metric_inventory(pdf, analysis, focus_summary, metadata)
    pdf.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", default="dqn", choices=sorted(MODEL_LABELS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()
    analysis, _ = analyze(args.run_dir, args.model)
    build_pdf(analysis, args.output)
    if args.print_summary:
        print(json.dumps(analysis, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
