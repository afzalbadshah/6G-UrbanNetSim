"""
Evaluation plans (Task 33 / R4 — config side, DATA only).

How to extend: to add or change an experiment plan, edit `EVALUATION_PLANS`
(one place); enabled KPIs must be a subset of `KPI_NAMES`.

STATUS: sourced configuration DATA; no runtime KPI computation, no formulas. Codex's
`metrics/kpi.py` (R4) computes the §15 KPIs against one authoritative ledger; this
file only declares the immutable `EvaluationPlan`s (methodology + reproducibility
metadata) that a run records.

Ownership: Claude. Per IMPLEMENTATION_CONTRACT.md §15 an EvaluationPlan records:
experiment/run IDs, master seed, warm-up/measurement windows (s), repetitions,
common-random-number policy, CI method/level, scenario/profile versions, enabled
feature flags, calibration references, and output maturity labels. §11 visionary
values are excluded from default baseline claims.

Units follow §2: windows in seconds (`_window_s`); seed/repetitions are ints;
ci_level is a unitless ratio. Everything is deep-immutable.
"""

import hashlib

from config._provenance import deep_freeze
from config._references import cite

# Aggregate + interval choices a plan may declare (§15 methodology metadata).
# "range" (observed min-max across seeds) is the honest report for very small
# seed counts, where a parametric CI or bootstrap would over-state precision;
# "student_t"/"stratified_bootstrap" become appropriate at larger repetitions.
AGGREGATE_METRICS = ("mean", "iqm", "median")
CI_METHODS = ("range", "student_t", "stratified_bootstrap")

# The §15 KPI names (authoritative denominators live in metrics/kpi.py). An
# EvaluationPlan's `enabled_kpis` must be a subset of these.
KPI_NAMES = (
    "packet_hop_delivery_ratio",
    "application_completion_ratio",
    "deadline_success_ratio",
    "goodput",
    "wire_efficiency",
    "energy_efficiency",
    "unit_data_cost",
    "completion_delay_percentile",
)

# Maturity levels permitted in DEFAULT baseline claims (§11: visionary excluded).
BASELINE_ALLOWED_MATURITIES = ("validated", "standards-aligned", "experimental")

_P_METHOD = cite(
    "eval_methodology", maturity="experimental",
    validity_notes="warm-up + multiple seeds + CI over seed means; Student-t interval; "
                   "type-7 percentiles; deadline success excludes late deliveries. "
                   "Common practice, not an applicable standard -> experimental.")


def _plan(experiment_id, *, master_seed, warmup_window_s, measurement_window_s,
          repetitions, ci_level, delay_percentiles, enabled_kpis,
          enabled_feature_flags, calibration_references, prov,
          crn_policy="per = experiment_seed + purpose('per') + unit_id + hop_id + attempt_number",
          ci_method="student_t", aggregate_metric="mean",
          late_deliveries_in_percentiles=False):
    for kpi in enabled_kpis:
        if kpi not in KPI_NAMES:
            raise ValueError(f"enabled KPI {kpi!r} not in {KPI_NAMES}")
    if not (isinstance(master_seed, int) and not isinstance(master_seed, bool) and master_seed >= 0):
        raise ValueError("master_seed must be a non-negative int")
    if not (isinstance(repetitions, int) and repetitions >= 1):
        raise ValueError("repetitions must be a positive int")
    if ci_method not in CI_METHODS:
        raise ValueError(f"ci_method {ci_method!r} not in {CI_METHODS}")
    if aggregate_metric not in AGGREGATE_METRICS:
        raise ValueError(f"aggregate_metric {aggregate_metric!r} not in {AGGREGATE_METRICS}")
    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must be within (0, 1)")
    if warmup_window_s < 0.0:
        raise ValueError("warmup_window_s must be non-negative")
    if measurement_window_s is not None and measurement_window_s <= 0.0:
        raise ValueError("measurement_window_s must be positive or None (whole run)")
    return {
        "experiment_id": experiment_id,
        "master_seed": int(master_seed),
        "warmup_window_s": float(warmup_window_s),
        "measurement_window_s": (None if measurement_window_s is None
                                 else float(measurement_window_s)),
        "repetitions": int(repetitions),
        "common_random_number_policy": crn_policy,
        "confidence_interval_method": ci_method,
        "aggregate_metric": aggregate_metric,
        "confidence_level": float(ci_level),
        "profile_versions": {
            "bands": "1.2", "scenarios": "1.2", "topology": "1.2",
            "power": "1.2", "tariffs": "2024.1", "contract": "0.2.0",
        },
        "enabled_feature_flags": dict(enabled_feature_flags),
        "enabled_kpis": tuple(enabled_kpis),
        "delay_percentiles": tuple(delay_percentiles),
        "late_deliveries_in_percentiles": bool(late_deliveries_in_percentiles),
        "baseline_allowed_maturities": BASELINE_ALLOWED_MATURITIES,
        "calibration_references": tuple(calibration_references),
        "provenance": prov,
    }


_RAW_PLANS = {
    # Default publication-grade plan.
    "default": _plan(
        "smartcity_default",
        master_seed=42,
        warmup_window_s=0.0,                 # device ramp starts small; no warm-up drop by default
        measurement_window_s=None,           # whole run
        repetitions=3,                       # 3 training seeds (compute-budget cap; see Task 57)
        ci_level=0.95,
        ci_method="range",                   # report observed min-max + every seed at n=3
        aggregate_metric="mean",             # mean of the 3 seeds (IQM/bootstrap need >=5-10)
        delay_percentiles=(50.0, 95.0, 99.0),
        enabled_kpis=KPI_NAMES,              # all §15 KPIs
        enabled_feature_flags={
            "stochastic_los": False,         # deterministic expected LoS (RL-stable)
            "shadowing_fading_kpi": False,   # seeded opt-in only
            "isac_sensing": False,           # visionary/experimental; off in baseline
            "ai_as_a_service": False,
            "ntn": False,
        },
        calibration_references=(
            "FR1 channel regression reference (regression anchor)",
            "ns-3 / 5G-LENA cross-validation (planned, contract P5)",
        ),
        prov=_P_METHOD),

    # Minimal deterministic plan for the headless smoke/hand-check.
    "handcheck_smoke": _plan(
        "handcheck_smoke",
        master_seed=42,
        warmup_window_s=0.0,
        measurement_window_s=None,
        repetitions=1,
        ci_level=0.95,
        delay_percentiles=(50.0, 95.0, 99.0),
        enabled_kpis=("application_completion_ratio", "deadline_success_ratio",
                      "goodput", "wire_efficiency"),
        enabled_feature_flags={"stochastic_los": False, "shadowing_fading_kpi": False,
                               "isac_sensing": False, "ai_as_a_service": False, "ntn": False},
        calibration_references=("FR1 channel regression reference",),
        prov=_P_METHOD),
}

EVALUATION_PLANS = deep_freeze(_RAW_PLANS)

REQUIRED_KEYS = deep_freeze({
    "experiment_id", "master_seed", "warmup_window_s", "measurement_window_s",
    "repetitions", "common_random_number_policy", "confidence_interval_method",
    "aggregate_metric", "confidence_level", "profile_versions",
    "enabled_feature_flags", "enabled_kpis",
    "delay_percentiles", "late_deliveries_in_percentiles",
    "baseline_allowed_maturities", "calibration_references", "provenance",
})


def get_evaluation_plan(name):
    """Return a deep-immutable evaluation plan by name (KeyError lists valid names)."""
    if name not in EVALUATION_PLANS:
        raise KeyError(f"Unknown evaluation plan {name!r}. Valid: {sorted(EVALUATION_PLANS)}")
    return EVALUATION_PLANS[name]


def _derive_seed(master_seed, purpose, repetition):
    """Deterministically derive a 32-bit run seed from one disclosed master seed.

    Uses SHA-256 over ``master:purpose:repetition`` so the derivation is stable
    across processes, machines, and Python runs (unlike the salted built-in
    ``hash``). Distinct ``purpose`` labels (e.g. ``"train"`` vs ``"eval"``) yield
    disjoint streams by construction, so a policy is never evaluated on the exact
    conditions it trained on. Range is ``[0, 2**32)`` — valid for ``random``,
    ``numpy``, and ``torch`` seeding.
    """
    key = f"{int(master_seed)}:{purpose}:{int(repetition)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def resolved_seeds(plan, purpose="eval", count=None, master_seed=None):
    """Return the tuple of per-repetition run seeds a campaign should use.

    ``plan`` is an EvaluationPlan (or its name). ``purpose`` selects a disjoint
    stream ("train" / "eval"). ``count`` overrides the plan's ``repetitions``
    (e.g. many cheap eval reps per trained agent). ``master_seed`` overrides the
    plan's own seed so the single run-control authority (``config.RANDOM_SEED``)
    drives the campaign; the derived seeds stay fully reproducible from it.
    """
    if isinstance(plan, str):
        plan = get_evaluation_plan(plan)
    n = int(plan["repetitions"] if count is None else count)
    if n < 1:
        raise ValueError("count/repetitions must be >= 1")
    master = int(plan["master_seed"] if master_seed is None else master_seed)
    return tuple(_derive_seed(master, purpose, rep) for rep in range(n))
