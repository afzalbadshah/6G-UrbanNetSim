"""Aggregate a multi-seed campaign into across-seed statistics (Task 57).

A campaign written by ``MainApplication`` looks like::

    output/<phase>/seed_1/offloading/<model>/results.csv
    output/<phase>/seed_2/offloading/<model>/results.csv
    ...
    output/<phase>/manifest.json

This tool discovers the ``seed_*`` folders, summarizes each run into a few
research KPIs, and reports the across-seed **mean and observed min-max range**
plus the **per-seed values** for every model. At the small seed counts this
project uses (3), reporting every seed + the range is the honest choice — a
parametric confidence interval or bootstrap would over-state precision (see
Henderson 2018, Colas 2018, Agarwal/rliable 2021, summarized in the paper's
methodology paragraph).

How to extend: add a ``(name, fn)`` pair to ``_RUN_METRICS`` to report another
per-run KPI. Each ``fn(rows, completed)`` returns a float or ``None`` (missing).
Nothing here computes physics — it only reduces the authoritative CSV that the
simulator already wrote.

Usage:
    python -m tools.aggregate_seeds output/evaluation
    python -m tools.aggregate_seeds output/evaluation --baseline random
"""
import argparse
import csv
import glob
import os
import statistics


def _fnum(row, key):
    """Parse one CSV cell as float, or None when blank/non-numeric."""
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values):
    nums = [v for v in values if v is not None]
    return (sum(nums) / len(nums)) if nums else None


def _is_delivered(row):
    return str(row.get("network_status", "")).strip().lower() == "delivered"


# Per-run KPIs. Each fn receives (all_rows, delivered_rows) and returns float|None.
_RUN_METRICS = (
    ("n_rows", lambda rows, done: float(len(rows))),
    ("n_completed", lambda rows, done: float(len(done))),
    ("completion_ratio", lambda rows, done: (len(done) / len(rows)) if rows else None),
    ("mean_reward", lambda rows, done: _mean(_fnum(r, "reward") for r in rows)),
    ("mean_e2e_delay_ms_completed",
     lambda rows, done: _mean(_fnum(r, "observed_complete_latency_ms") for r in done)),
    ("mean_total_energy_j", lambda rows, done: _mean(_fnum(r, "total_energy_j") for r in rows)),
    ("mean_cost", lambda rows, done: _mean(_fnum(r, "cost") for r in rows)),
)
METRIC_NAMES = tuple(name for name, _ in _RUN_METRICS)


def discover_seed_dirs(phase_dir):
    """Return the seed_* run folders under a phase dir, ordered seed_1, seed_2, ..."""
    found = glob.glob(os.path.join(phase_dir, "seed_*"))
    dirs = [d for d in found if os.path.isdir(d)]

    def _index(path):
        tail = os.path.basename(path).split("seed_", 1)[-1]
        return int(tail) if tail.isdigit() else 1 << 30

    return sorted(dirs, key=_index)


def discover_models(phase_dir):
    """Model names present under any seed's offloading/ folder."""
    models = set()
    for seed_dir in discover_seed_dirs(phase_dir):
        off = os.path.join(seed_dir, "offloading")
        if os.path.isdir(off):
            models.update(name for name in os.listdir(off)
                          if os.path.isdir(os.path.join(off, name)))
    return sorted(models)


def summarize_run(results_csv_path):
    """Reduce one run's detailed results.csv to the per-run KPI dict."""
    with open(results_csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    delivered = [r for r in rows if _is_delivered(r)]
    return {name: fn(rows, delivered) for name, fn in _RUN_METRICS}


def aggregate_model(phase_dir, model):
    """Per-seed values + across-seed mean/min/max for one model.

    Returns {"seeds": [seed_1_label, ...], "per_seed": {metric: [v_per_seed]},
             "mean"/"min"/"max": {metric: value}}.
    """
    seed_dirs = discover_seed_dirs(phase_dir)
    seed_labels, per_seed = [], {name: [] for name in METRIC_NAMES}
    for seed_dir in seed_dirs:
        path = os.path.join(seed_dir, "offloading", model, "results.csv")
        if not os.path.exists(path):
            continue
        seed_labels.append(os.path.basename(seed_dir))
        summary = summarize_run(path)
        for name in METRIC_NAMES:
            per_seed[name].append(summary[name])

    agg_mean, agg_min, agg_max = {}, {}, {}
    for name in METRIC_NAMES:
        vals = [v for v in per_seed[name] if v is not None]
        agg_mean[name] = statistics.mean(vals) if vals else None
        agg_min[name] = min(vals) if vals else None
        agg_max[name] = max(vals) if vals else None
    return {"seeds": seed_labels, "per_seed": per_seed,
            "mean": agg_mean, "min": agg_min, "max": agg_max}


def _fmt(value):
    return "" if value is None else f"{value:.10g}"


def write_aggregate(phase_dir, models=None, baseline=None):
    """Write aggregate/<model>_summary.csv and aggregate/comparison_summary.csv.

    Returns the aggregation dict keyed by model. ``baseline`` (a model name)
    enables a paired per-seed ordering count on mean_reward: how many seeds each
    model beats the baseline on (the honest 'consistent across seeds' evidence).
    """
    models = list(models) if models is not None else discover_models(phase_dir)
    out_dir = os.path.join(phase_dir, "aggregate")
    os.makedirs(out_dir, exist_ok=True)

    results = {model: aggregate_model(phase_dir, model) for model in models}

    # Per-model detail: one row per metric, a column per seed, then mean/min/max.
    for model, agg in results.items():
        path = os.path.join(out_dir, f"{model}_summary.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["metric", *agg["seeds"], "mean", "min", "max"])
            for name in METRIC_NAMES:
                writer.writerow([name, *[_fmt(v) for v in agg["per_seed"][name]],
                                 _fmt(agg["mean"][name]), _fmt(agg["min"][name]),
                                 _fmt(agg["max"][name])])

    # Cross-model comparison: mean/min/max per metric, plus paired reward wins.
    comp_path = os.path.join(out_dir, "comparison_summary.csv")
    header = ["model"]
    for name in METRIC_NAMES:
        header += [f"{name}_mean", f"{name}_min", f"{name}_max"]
    if baseline in results:
        header += [f"reward_seeds_beating_{baseline}", "seeds_compared"]
    with open(comp_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        base_reward = results[baseline]["per_seed"]["mean_reward"] if baseline in results else None
        for model, agg in results.items():
            row = [model]
            for name in METRIC_NAMES:
                row += [_fmt(agg["mean"][name]), _fmt(agg["min"][name]), _fmt(agg["max"][name])]
            if base_reward is not None:
                model_reward = agg["per_seed"]["mean_reward"]
                pairs = [(m, b) for m, b in zip(model_reward, base_reward)
                         if m is not None and b is not None]
                wins = sum(1 for m, b in pairs if m > b)
                row += [wins, len(pairs)]
            writer.writerow(row)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate a multi-seed campaign.")
    parser.add_argument("phase_dir", help="e.g. output/evaluation (holds seed_* folders)")
    parser.add_argument("--baseline", default="random",
                        help="baseline model for paired per-seed reward comparison")
    args = parser.parse_args(argv)

    seed_dirs = discover_seed_dirs(args.phase_dir)
    models = discover_models(args.phase_dir)
    if not seed_dirs:
        raise SystemExit(f"No seed_* folders under {args.phase_dir!r}")
    print(f"[AGG] {len(seed_dirs)} seeds, {len(models)} models: {models}")
    write_aggregate(args.phase_dir, models=models, baseline=args.baseline)
    print(f"[AGG] wrote {os.path.join(args.phase_dir, 'aggregate')}")


if __name__ == "__main__":
    main()
