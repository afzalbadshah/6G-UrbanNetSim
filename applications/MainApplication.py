import os
import sys
import json
import traceback
from datetime import datetime, timezone
from functools import partial

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config as cfg
from config.evaluation import get_evaluation_plan, resolved_seeds
from applications.system_awake import run_while_system_awake
from core.run_history import create_run
from core.simulator import Simulator
from live.live_map import run_live_map
from live.training_progress import (
    finish_training_progress,
    make_training_progress_callback,
    start_training_progress,
)
from scheduler.common import common_config
from scheduler.common.comparison_runner import create_comparison_scheduler
from scheduler.common.policy_utils import set_global_seed
try:
    from visualization.smartcities_plotter import write_model_comparison_summary
except ModuleNotFoundError:
    # Optional post-run comparison summary; the module may be absent in trimmed
    # checkouts. Per-model results.csv is still written without it.
    write_model_comparison_summary = None

print = partial(print, flush=True)

# ==========================================================
#  LAUNCHER - all run control lives in config.py §1
# ==========================================================
# MainApplication is only the launcher: it READS run settings from config.py §1
# (RANDOM_SEED, RUN_TRAINING, RUN_MODELS, MULTISEED*, GUI flags, OUTPUT_DIR,
# ACTIVE_DEVICES_ONLY, WORKLOAD_OVERRIDES/ROUNDS/MAX_DEVICES) and defines none of
# them here. To change a run, edit config.py §1 — not this file.

RUNTIME = {}

# Checkpoint training logs (reward/epsilon/action CSVs) every N rounds so an
# interrupted long run still leaves usable data. Plots are written at finalize.
CHECKPOINT_EVERY_ROUNDS = 25


def _configured_output_root():
    configured = str(cfg.OUTPUT_DIR).strip() or "output"
    return os.path.abspath(
        configured if os.path.isabs(configured) else os.path.join(ROOT_DIR, configured)
    )



def _refresh_runtime_flags():
    train = bool(cfg.RUN_TRAINING)
    if hasattr(cfg, "apply_workload_profile"):
        cfg.apply_workload_profile(train_mode=train)   # reads §1 overrides itself
    run_models = [str(model).strip().lower() for model in (cfg.RUN_MODELS or []) if str(model).strip()]
    common_config.COMPARISON_ENABLED = bool(cfg.COMPARISON_ENABLED)
    common_config.RUN_MODELS = run_models
    common_config.RANDOM_SEED = int(cfg.RANDOM_SEED)
    RUNTIME.clear()
    RUNTIME.update({
        "train": train,
        "evaluation": not train,
        "training_gui": bool(cfg.TRAINING_GUI),
        "evaluation_gui": bool(cfg.EVALUATION_GUI),
        "use_gui": bool(cfg.EVALUATION_GUI) and not train,
        "output_phase": "exploration" if train else "evaluation",
        "run_mode_label": "TRAIN" if train else "EVAL",
        "base_seed": int(cfg.RANDOM_SEED),
        "main_scheduler": str(cfg.MAIN_SCHEDULER).strip().lower(),
        "output_root": _configured_output_root(),
        "run_dir": "",
    })
    # Keep legacy scheduler/reporting consumers on the same phase-specific
    # root as the explicit simulator output path.
    common_config.OUTPUT_DIR = os.path.join(
        RUNTIME["output_root"], RUNTIME["output_phase"]
    )


def _runtime(name):
    if not RUNTIME:
        _refresh_runtime_flags()
    return RUNTIME[name]


def _start_run_output():
    """Create exactly one versioned run folder for this application launch."""

    if RUNTIME.get("run_dir"):
        return RUNTIME["run_dir"]
    deployment = str(
        (getattr(cfg, "SIMULATION", {}) or {}).get("deployment_profile", "scenario")
    )
    mode = "training" if _runtime("train") else "evaluation"
    run_dir = create_run(
        _runtime("output_root"),
        deployment_profile=deployment,
        mode=mode,
    )
    RUNTIME["run_dir"] = str(run_dir)
    common_config.OUTPUT_DIR = str(run_dir)
    print(f"[RUN HISTORY] {run_dir.name} -> {run_dir}")
    return RUNTIME["run_dir"]


def _phase_output_dir(output_phase=None):
    output_phase_default = _runtime("output_phase")
    phase = str(output_phase or output_phase_default).replace("\\", "/").strip()
    root_phase = phase.split("/", 1)[0] if phase else output_phase_default
    if root_phase not in ("exploration", "evaluation"):
        root_phase = "evaluation" if not _runtime("train") else "exploration"
        phase = root_phase
    run_root = RUNTIME.get("run_dir") or _runtime("output_root")
    # Flat layout: the versioned run folder IS the phase root, so the leading
    # "exploration"/"evaluation" token is not repeated as a sub-folder.
    rest = [part for part in phase.split("/")[1:] if part]
    if not RUNTIME.get("run_dir"):
        return os.path.join(run_root, root_phase, *rest)
    return os.path.join(run_root, *rest)


def _summary_root():
    return _phase_output_dir()


def _console_progress_callback(model_name):
    def _callback(**payload):
        round_id = int(payload.get("round_id", 0) or 0)
        total_rounds = int(payload.get("total_rounds", cfg.total_rounds()) or cfg.total_rounds())
        event = payload.get("event", "")
        step = max(1, total_rounds // 10)
        if event == "round_end" and (round_id == 1 or round_id == total_rounds or round_id % step == 0):
            print(f"[PROGRESS] {model_name} round {round_id}/{total_rounds}")
    return _callback


def _build_simulator(model_name, seed, output_phase=None):
    scheduler_type = str(model_name).strip().lower()
    set_global_seed(seed)
    final_output_dir = _phase_output_dir(output_phase)
    scheduler = create_comparison_scheduler(
        scheduler_type,
        config=common_config,
        output_dir=final_output_dir,
        train_mode=_runtime("train"),
        seed=seed,
        logger=print,
    )
    print(
        f"[RUN] {scheduler_type} | mode={_runtime('run_mode_label')} "
        f"| training_gui={_runtime('training_gui') if _runtime('train') else False} "
        f"| eval_gui={_runtime('use_gui') if _runtime('evaluation') else False} | seed={seed}"
    )
    sim = Simulator(scheduler=scheduler, output_dir=final_output_dir, workload=cfg.WORKLOAD)
    print(f"[OUT] {scheduler_type} -> {sim.output_dir}")
    return sim


def _run_simulator(sim, allow_gui, training_progress=None, model_index=0):
    model_name = str(getattr(getattr(sim, "scheduler", None), "model_name", "") or "").strip().lower()
    console_callback = _console_progress_callback(model_name)

    if _runtime("train"):
        total_rounds = cfg.total_rounds()
        if training_progress is not None:
            training_progress.model_started(model_name, model_index, total_rounds=total_rounds)
        else:
            print("[GUI] Training progress window unavailable; training continues in console.")
        callback = make_training_progress_callback(training_progress, model_name, model_index, total_rounds)
        sim.run(train_mode=True, progress_callback=callback or console_callback)
        if training_progress is not None:
            training_progress.model_finished(model_name, model_index, total_rounds=total_rounds)
    elif allow_gui and _runtime("use_gui"):
        run_live_map(sim, model_name=model_name, refresh_ms=100, logger=print)
    else:
        reason = []
        if not allow_gui:
            reason.append("comparison model is headless")
        if not _runtime("use_gui"):
            reason.append("GUI is disabled")
        print(f"[GUI] Disabled: {', '.join(reason) or 'headless run requested'}.")
        sim.run(train_mode=False, progress_callback=console_callback)


def run_once(model_name, seed=None, allow_gui=True, output_phase=None, training_progress=None, model_index=0):
    sim = _build_simulator(
        model_name,
        seed=_runtime("base_seed") if seed is None else seed,
        output_phase=output_phase,
    )
    _run_simulator(
        sim,
        allow_gui=allow_gui,
        training_progress=training_progress,
        model_index=model_index,
    )
    print(f"[DONE] {str(model_name).strip().lower()} completed.")


def _run_ordered_models(ordered_models, allow_gui):
    ordered_models = [str(model).strip().lower() for model in (ordered_models or [])]
    progress = None
    if _runtime("train"):
        progress = start_training_progress(
            ordered_models,
            total_rounds=cfg.total_rounds(),
            enabled=_runtime("training_gui"),
            logger=print,
        )
    try:
        for model_index, model_name in enumerate(ordered_models):
            run_once(
                model_name,
                seed=getattr(common_config, "RANDOM_SEED", _runtime("base_seed")),
                allow_gui=allow_gui,
                training_progress=progress,
                model_index=model_index,
            )
    except Exception:
        finish_training_progress(progress, failed=True, message="Training stopped")
        raise
    else:
        finish_training_progress(progress, failed=False)


# ==========================================================
#  MULTI-SEED CAMPAIGN (Task 57) - headless, per-seed folders
# ==========================================================

def _seed_dir(phase_dir, seed_index, flat=False):
    """Per-seed output root: <run>/seed_<k> (1-based, readable).

    ``flat=True`` (single-seed training) writes straight into the run folder.
    """
    if flat:
        return phase_dir
    return os.path.join(phase_dir, f"seed_{seed_index + 1}")


def _multiseed_progress_callback(job):
    """Console progress for a parallel worker (the Tk GUI can't span processes).

    Prints a compact line at ~10% steps so a long training run does not look
    frozen, tagged with the seed so interleaved output from several workers stays
    readable, e.g. ``[PROG] seed_1 3/3 model=dqn round 120/200``.
    """
    tag = f"seed_{job['seed_index'] + 1}"
    queue = job.get("progress_queue")

    def _callback(**payload):
        if payload.get("event") != "round_end":
            return
        round_id = int(payload.get("round_id", 0) or 0)
        total_rounds = int(payload.get("total_rounds", cfg.total_rounds()) or cfg.total_rounds())
        # Feed the GUI every round (smooth bar); print to console at ~10% steps.
        if queue is not None:
            try:
                queue.put({"seed_index": job["seed_index"], "model": job["model_name"],
                           "round": round_id, "total": total_rounds, "done": False})
            except Exception:
                pass
        step = max(1, total_rounds // 10)
        if round_id == 1 or round_id == total_rounds or round_id % step == 0:
            print(f"[PROG] {tag} ({job['seed_index'] + 1}/{job['seed_total']}) "
                  f"model={job['model_name']} round {round_id}/{total_rounds}")

    return _callback


def _run_one_job(job):
    """Worker: run ONE (seed, model) in its own process, entirely from `job`.

    On Windows `spawn`, a child re-imports this module fresh, so parent-process
    global mutations (workload profile, common_config) are NOT inherited. The
    worker therefore re-applies every run setting from the picklable `job` before
    building the simulator, then writes into its own seed/model folder. Returns a
    small picklable summary. No GUI, no shared state between jobs.
    """
    train = bool(job["train"])
    cfg.apply_workload_profile(train_mode=train)      # §1 overrides read from config
    common_config.COMPARISON_ENABLED = True
    common_config.RUN_MODELS = list(cfg.RUN_MODELS)
    common_config.RANDOM_SEED = int(job["seed"])

    set_global_seed(int(job["seed"]))
    os.makedirs(job["output_dir"], exist_ok=True)
    scheduler = create_comparison_scheduler(
        job["model_name"], config=common_config, output_dir=job["output_dir"],
        train_mode=train, seed=int(job["seed"]), logger=print,
    )
    sim = Simulator(scheduler=scheduler, output_dir=job["output_dir"], workload=cfg.WORKLOAD)
    print(f"[JOB] seed={job['seed']} ({job['seed_index'] + 1}/{job['seed_total']}) "
          f"model={job['model_name']} -> {job['output_dir']}")

    # Checkpoint training logs every N rounds so an interrupted run still leaves
    # usable reward/epsilon/action CSVs (plots are written at finalize). Cheap:
    # CSV-only, no plots mid-run.
    progress = _multiseed_progress_callback(job)

    def _callback(**payload):
        progress(**payload)
        if train and payload.get("event") == "round_end":
            rid = int(payload.get("round_id", 0) or 0)
            if rid and rid % CHECKPOINT_EVERY_ROUNDS == 0:
                try:
                    scheduler.save_logs(include_plots=False)
                except Exception:
                    pass

    sim.run(train_mode=train, progress_callback=_callback)
    queue = job.get("progress_queue")
    if queue is not None:
        try:
            queue.put({"seed_index": job["seed_index"], "model": job["model_name"],
                       "round": 0, "total": 0, "done": True})
        except Exception:
            pass
    return {"model": job["model_name"], "seed": int(job["seed"]),
            "seed_index": int(job["seed_index"]), "output_dir": str(sim.output_dir)}


def _build_jobs(run_models, seeds, train, phase_dir):
    # Workers re-read run settings from config.py §1 (spawn re-imports config),
    # so a job only carries what is unique to it: which model, which seed, where.
    # Training is single-seed and flat (files directly in the run folder);
    # evaluation keeps one seed_<k> folder per seed.
    flat = bool(train) and len(seeds) == 1
    jobs = []
    for seed_index, seed in enumerate(seeds):
        out_dir = _seed_dir(phase_dir, seed_index, flat=flat)
        for model_name in run_models:
            jobs.append({
                "model_name": model_name,
                "seed": int(seed),
                "seed_index": seed_index,
                "seed_total": len(seeds),
                "train": bool(train),
                "output_dir": out_dir,
            })
    return jobs


def _write_manifest(phase_dir, run_models, seeds, train, plan):
    """Immutable run provenance for the paper: seeds, methodology, versions."""
    os.makedirs(phase_dir, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": plan["experiment_id"],
        "master_seed": int(plan["master_seed"]),
        "seed_purpose": "train" if train else "eval",
        "seeds": {f"seed_{i + 1}": int(s) for i, s in enumerate(seeds)},
        "repetitions": len(seeds),
        "models": list(run_models),
        "mode": "train" if train else "eval",
        "aggregate_metric": plan["aggregate_metric"],
        "confidence_interval_method": plan["confidence_interval_method"],
        "confidence_level": float(plan["confidence_level"]),
        "profile_versions": dict(plan["profile_versions"]),
        "workload": {k: cfg.WORKLOAD.get(k)
                     for k in ("start_devices", "max_devices", "increment", "rounds")},
        "note": ("Task 57 multi-seed campaign. Seeds derived from master_seed; "
                 "valid independent samples, not bitwise-paired CRN (see Task 54)."),
    }
    path = os.path.join(phase_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return path


class _MultiSeedProgressGUI:
    """One progress bar per (seed, model) job, updated from a worker queue.

    Parallel workers run in separate processes and cannot drive Tk, so each
    reports round progress onto a shared queue that this window (in the main
    process) drains and renders. One row per job: label, bar, round text.
    """

    def __init__(self, jobs):
        import tkinter as tk
        from tkinter import ttk
        self.root = tk.Tk()
        self.root.title("SmartCityEdgeSim - Multi-seed Training")
        tk.Label(self.root, text="Multi-seed Training Progress",
                 font=("Segoe UI", 13, "bold")).grid(row=0, column=0, columnspan=3,
                                                      sticky="w", padx=10, pady=(10, 4))
        self.total_jobs = len(jobs)
        self.done_count = 0
        self.status = tk.Label(self.root, text=f"0/{self.total_jobs} jobs done", anchor="w")
        self.status.grid(row=1, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 6))
        self.rows = {}
        r = 2
        for job in jobs:
            key = (job["seed_index"], job["model_name"])
            tk.Label(self.root, text=f"seed_{job['seed_index'] + 1} · {job['model_name']}",
                     width=22, anchor="w").grid(row=r, column=0, sticky="w", padx=(10, 6), pady=2)
            bar = ttk.Progressbar(self.root, length=260, maximum=100)
            bar.grid(row=r, column=1, padx=6, pady=2)
            txt = tk.Label(self.root, text="waiting", width=12, anchor="w")
            txt.grid(row=r, column=2, sticky="w", padx=(6, 10), pady=2)
            self.rows[key] = (bar, txt)
            r += 1

    def update(self, ev):
        row = self.rows.get((ev["seed_index"], ev["model"]))
        if not row:
            return
        bar, txt = row
        if ev.get("done"):
            bar["value"] = 100
            txt.config(text="done ✓")
            self.done_count += 1
            self.status.config(text=f"{self.done_count}/{self.total_jobs} jobs done")
        else:
            total = ev.get("total") or 1
            bar["value"] = 100.0 * ev.get("round", 0) / total
            txt.config(text=f"{ev.get('round', 0)}/{total}")


def _run_jobs_parallel_gui(jobs, max_workers, on_job_complete=None):
    """Run jobs across processes while a Tk window shows per-job progress bars.

    Falls back to the headless parallel path if Tk is unavailable.
    """
    import multiprocessing as mp
    import concurrent.futures as cf
    import queue as qmod
    try:
        manager = mp.Manager()
        progress_queue = manager.Queue()
        for job in jobs:
            job["progress_queue"] = progress_queue
        gui = _MultiSeedProgressGUI(jobs)
    except Exception as exc:
        print(f"[GUI] progress window unavailable ({exc}); running headless.")
        for job in jobs:
            job.pop("progress_queue", None)
        _run_jobs_parallel_headless(
            jobs, max_workers, on_job_complete=on_job_complete
        )
        return

    pool = _process_pool(max_workers)
    futures = [pool.submit(_run_one_job, job) for job in jobs]
    handled_futures = set()

    def pump():
        try:
            while True:
                gui.update(progress_queue.get_nowait())
        except qmod.Empty:
            pass
        for future in futures:
            if not future.done() or future in handled_futures:
                continue
            handled_futures.add(future)
            try:
                result = future.result()
                print(f"[DONE] seed={result['seed']} model={result['model']}")
                if on_job_complete is not None:
                    on_job_complete(result)
            except Exception as exc:
                print(f"[ERR] job failed: {exc}")
        if len(handled_futures) == len(futures):
            pool.shutdown(wait=False)
            gui.status.config(text=f"All {len(jobs)} jobs complete. Closing…")
            gui.root.after(1500, gui.root.destroy)
            return
        gui.root.after(150, pump)

    gui.root.after(300, pump)
    gui.root.mainloop()


def _process_pool(max_workers):
    """Process pool that recycles each worker after one job (Python >= 3.11),
    so a finished model's packet ledgers never accumulate in memory."""
    import concurrent.futures as cf
    try:
        return cf.ProcessPoolExecutor(max_workers=max_workers, max_tasks_per_child=1)
    except TypeError:
        return cf.ProcessPoolExecutor(max_workers=max_workers)


def _run_jobs_parallel_headless(jobs, max_workers, on_job_complete=None):
    import concurrent.futures as cf
    with _process_pool(max_workers) as pool:
        futures = [pool.submit(_run_one_job, job) for job in jobs]
        for fut in cf.as_completed(futures):
            res = fut.result()
            print(f"[DONE] seed={res['seed']} model={res['model']}")
            if on_job_complete is not None:
                on_job_complete(res)


def _run_multiseed_campaign(run_models):
    # Import plotting only in the parent process. Spawned simulation workers remain
    # independent of matplotlib/pandas and never compete for shared figure files.
    from visualization.supportive.automatic_plotting import CampaignPlotCoordinator

    train = _runtime("train")
    plan = get_evaluation_plan(cfg.EVAL_PLAN_NAME)
    purpose = "train" if train else "eval"
    # Training uses TRAINING_SEED_COUNT; evaluation uses MULTISEED_COUNT.
    # Multi-seed campaigns write one seed_<k> folder per seed under one run.
    seed_count = (
        int(getattr(cfg, "TRAINING_SEED_COUNT", 1) or 1)
        if train
        else int(cfg.MULTISEED_COUNT)
    )
    seeds = resolved_seeds(plan, purpose=purpose, count=seed_count,
                           master_seed=int(cfg.RANDOM_SEED))
    phase_dir = _phase_output_dir()
    os.makedirs(phase_dir, exist_ok=True)
    jobs = _build_jobs(run_models, seeds, train, phase_dir)
    print(f"[MULTISEED] {len(seeds)} seeds x {len(run_models)} models = {len(jobs)} jobs "
          f"| parallel={cfg.MULTISEED_PARALLEL} | phase_dir={phase_dir}")
    _write_manifest(phase_dir, run_models, seeds, train, plan)
    # Aggregate plots are a training-only product; evaluation writes no figures.
    plot_coordinator = (
        CampaignPlotCoordinator(
            phase_dir=phase_dir,
            models=run_models,
            seed_count=len(seeds),
            logger=print,
        )
        if train
        else None
    )
    on_complete = plot_coordinator.record_job_completion if plot_coordinator else None

    if cfg.MULTISEED_PARALLEL and len(jobs) > 1:
        max_workers = cfg.MULTISEED_MAX_WORKERS or min(os.cpu_count() or 1, len(jobs))
        print(f"[MULTISEED] ProcessPoolExecutor(max_workers={max_workers})")
        if train and _runtime("training_gui"):
            _run_jobs_parallel_gui(
                jobs,
                max_workers,
                on_job_complete=on_complete,
            )  # progress bar per (seed, model)
        else:
            _run_jobs_parallel_headless(
                jobs,
                max_workers,
                on_job_complete=on_complete,
            )
    else:
        for job in jobs:
            res = _run_one_job(job)
            print(f"[DONE] seed={res['seed']} model={res['model']}")
            if on_complete is not None:
                on_complete(res)

    # Per-seed comparison summaries; across-seed aggregation is tools/aggregate_seeds.py.
    if not train and write_model_comparison_summary is not None:
        for seed_index in range(len(seeds)):
            try:
                write_model_comparison_summary(_seed_dir(phase_dir, seed_index),
                                               run_models, logger=print)
            except Exception as exc:
                print(f"[WARN] per-seed summary failed for seed_{seed_index + 1}: {exc}")
    if plot_coordinator is not None:
        plot_coordinator.finalize()
    print(f"[MULTISEED] campaign complete -> {phase_dir}")


def run_configured_models():
    _refresh_runtime_flags()
    _start_run_output()
    print(
        f"[MODE] RUN_TRAINING={_runtime('train')} | "
        f"TRAINING_GUI={_runtime('training_gui')} | "
        f"EVALUATION_GUI={_runtime('evaluation_gui')} | "
        f"phase={_runtime('output_phase')}"
    )

    if bool(getattr(common_config, "COMPARISON_ENABLED", False)):
        run_models = [str(model).strip().lower() for model in (getattr(common_config, "RUN_MODELS", []) or [])]
        if not run_models:
            raise ValueError("COMPARISON_ENABLED is True but RUN_MODELS is empty.")
        print(
            f"[SIZE] rounds={cfg.total_rounds()} | "
            f"decisions/model={cfg.decisions_per_model()} | "
            f"models={len(run_models)} | "
            f"total_decisions={cfg.decisions_per_model() * len(run_models)}"
        )

        if cfg.MULTISEED:
            print("[MULTISEED] Enabled: forcing headless multi-seed campaign.")
            _run_multiseed_campaign(run_models)
        elif _runtime("use_gui") and not _runtime("train"):
            print("[GUI] Evaluation will run all configured models in order.")
            for model_index, model_name in enumerate(run_models):
                run_once(
                    model_name,
                    seed=getattr(common_config, "RANDOM_SEED", _runtime("base_seed")),
                    allow_gui=True,
                    output_phase=_runtime("output_phase"),
                    model_index=model_index,
                )
            if write_model_comparison_summary is not None:
                write_model_comparison_summary(_summary_root(), run_models, logger=print)
        else:
            _run_ordered_models(run_models, allow_gui=False)
            if not _runtime("train") and write_model_comparison_summary is not None:
                write_model_comparison_summary(_summary_root(), run_models, logger=print)
    else:
        _run_ordered_models([_runtime("main_scheduler")], allow_gui=True)


def main():
    try:
        run_while_system_awake(run_configured_models, logger=print)
    except Exception as exc:
        print(f"[ERROR] Scheduler run failed: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
