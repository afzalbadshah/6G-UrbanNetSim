"""Seed-aware data loader for multi-seed / multi-model result plots.

One ``results.csv`` per (seed, model) already holds every metric as a column
(cost, energy, delay, RSSI, tier, battery, ...). Every sub-plotter reads the
SAME files through this module and only selects its own columns; this module
turns the per-seed values into a mean + observed min-max range so the plotters
can draw the seed band the training figures use.

Works for any phase dir laid out as::

    <phase>/seed_1/offloading/<model>/results.csv
    <phase>/seed_2/offloading/<model>/results.csv
    ...
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd


def _seed_index(path: str) -> int:
    try:
        return int(os.path.basename(path).split("seed_", 1)[-1])
    except (ValueError, IndexError):
        return 0


def discover(phase_dir: str) -> dict[str, list[str]]:
    """Return ``{model: [seed_1_results.csv, seed_2_results.csv, ...]}`` for a phase."""
    models: dict[str, list[str]] = {}
    for seed_dir in sorted(glob.glob(os.path.join(phase_dir, "seed_*")), key=_seed_index):
        for model_dir in sorted(glob.glob(os.path.join(seed_dir, "offloading", "*"))):
            if os.path.isdir(model_dir):
                models.setdefault(os.path.basename(model_dir), []).append(
                    os.path.join(model_dir, "results.csv")
                )
    return models


def load(paths: list[str], cols) -> pd.DataFrame:
    """Concatenate every seed's rows for one model, tolerant of missing columns.

    Adds a 1-based ``seed`` column. Only the requested ``cols`` are read (fast).
    """
    wanted = set(cols)
    frames = []
    for i, path in enumerate(paths, start=1):
        if not os.path.exists(path):
            continue
        try:
            frame = pd.read_csv(path, sep=";", usecols=lambda c: c in wanted)
        except (ValueError, OSError):
            continue
        frame["seed"] = i
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def per_seed_scalar(df: pd.DataFrame, value: str, agg: str = "mean") -> pd.Series:
    """Collapse each seed to one number (e.g. mean energy per task) -> Series by seed."""
    if df.empty or value not in df.columns or "seed" not in df.columns:
        return pd.Series(dtype=float)
    numeric = pd.to_numeric(df[value], errors="coerce")
    return numeric.groupby(df["seed"]).agg(agg).dropna()


def band(series: pd.Series) -> tuple[float, float, float]:
    """Return (mean, min, max) across seeds from a per-seed Series."""
    if series.empty:
        return (float("nan"), float("nan"), float("nan"))
    return float(series.mean()), float(series.min()), float(series.max())


def per_seed_curve(df: pd.DataFrame, x: str, value: str, agg: str = "mean") -> dict[int, pd.Series]:
    """For line plots: {seed: Series(value indexed by x)} — e.g. energy vs device-load."""
    out: dict[int, pd.Series] = {}
    if df.empty or x not in df.columns or value not in df.columns:
        return out
    v = pd.to_numeric(df[value], errors="coerce")
    xb = pd.to_numeric(df[x], errors="coerce")
    for seed, sub in df.assign(_x=xb, _v=v).groupby("seed"):
        out[int(seed)] = sub.groupby("_x")["_v"].agg(agg).dropna()
    return out
