"""Workload / traffic configuration -- how many tasks arrive and how big.

Device count and per-device data size both RAMP across the run: they start at
the ``start_*`` value, grow by the ``*_increment`` each round, and hold at the
``max_*`` value once reached (see workload/generator.py). Set start == max (or
the increment to 0) to keep a quantity flat.

Sizing note: ``cpu_per_kb`` is calibrated against SERVER_CONFIG["Onboard"] --
a 500 KB task (x4 = 2,000 CPU units) fits the on-device tier (3,000) with
headroom, uses ~1/6 of a UAV_BS (12,000) and ~0.7% of a MEC site (280,000).

Per-class 5G/6G SLA deadlines live in config/scenarios.py (SLA_PROFILES); the
reward emphasis per class lives in config/reward.py.

To change the traffic (device ramp or per-task size), edit the dicts below --
they are the one place workload is defined.
"""

# --- Base profile (shared device scaling + per-task sizing) --------------
BASE_WORKLOAD = {
    "start_devices": 1,
    "max_devices": 500,
    "increment": 1,
    "start_data_per_device_kb": 1,
    "max_data_per_device_kb": 400,
    "data_increment_kb": 1,
    "data_per_device_kb": 400,   # fallback for code reading this key directly
    "cpu_per_kb": 4,
    "memory_per_kb": 0.2,
    "rounds": 500,
    "profile": "default",
}

# --- Per-class payload profiles -----------------------------------------
# URLLC and mMTC carry small, FLAT (non-growing) payloads: a safety alert or a
# sensor reading stays the same size no matter how many other devices join the
# city (32-200 B for URLLC per 3GPP HRLLC targets; under 200 B for mMTC
# telemetry per typical NB-IoT smart-meter reports). eMBB is the one class
# whose payload genuinely grows (streaming/video), so it inherits the base
# profile's ramp unchanged. Total traffic for every class still grows over the
# run regardless, via start_devices -> max_devices -- that ramp is independent
# of payload size and is not changed here.
URLLC_WORKLOAD = {
    "start_devices": 1,
    "max_devices": 500,
    "increment": 1,
    "start_data_per_device_kb": 0.2,
    "max_data_per_device_kb": 0.2,
    "data_increment_kb": 0.0,
    "data_per_device_kb": 0.2,
    "cpu_per_kb": 4,
    "memory_per_kb": 0.2,
    "rounds": 200,
    "profile": "urllc",
}

# mMTC: one canonical smart-city telemetry task.  It represents a small batch
# of measurements plus local preprocessing: heavier than a URLLC alert, but far
# below an eMBB video/analytics task.  Physical device differences (battery,
# MCU and radio power) remain in devices_config.py; they do not create separate
# mMTC workload classes.
mMTC_WORKLOAD = {
    **BASE_WORKLOAD,
    "start_data_per_device_kb": 2.0,
    "max_data_per_device_kb": 2.0,
    "data_increment_kb": 0.0,
    "data_per_device_kb": 2.0,
    "apply_device_data_multiplier": False,
    "compute_work_cycles": 40_000_000.0,
    "memory_working_set_bytes": 2_000_000,
    "profile": "mmtc",
}

eMBB_WORKLOAD = dict(BASE_WORKLOAD)
eMBB_WORKLOAD["profile"] = "embb"

# Priority class -> workload profile (1 = URLLC, 2 = mMTC, 3 = eMBB).
PRIORITY_WORKLOADS = {
    1: URLLC_WORKLOAD,
    2: mMTC_WORKLOAD,
    3: eMBB_WORKLOAD,
}

# --- Train vs eval profiles ---------------------------------------------
# Evaluation uses the base as-is; training inherits it and overrides only what
# should differ, e.g. TRAIN_WORKLOAD = {**BASE_WORKLOAD, "max_devices": 150}.
EVAL_WORKLOAD = dict(BASE_WORKLOAD)
TRAIN_WORKLOAD = dict(BASE_WORKLOAD)

# Active workload. apply_workload_profile() (config/config.py) refreshes this in
# place for the current run, so every module that imported the object sees it.
WORKLOAD = dict(EVAL_WORKLOAD)


# --- Density profiles: pick ONE in config/run.py (WORKLOAD_PROFILE) -------
# Each profile is a named device-density ramp (devices grow 1 -> max_devices
# over `rounds`). The run selects one; "custom" instead uses the device count
# set directly in config/run.py (MAX_DEVICES / ROUNDS). To add a density, add
# an entry here (one place).
WORKLOAD_PROFILES = {
    "sparse_urban": {"start_devices": 1, "max_devices": 50,  "increment": 1, "rounds": 200},
    "urban":        {"start_devices": 1, "max_devices": 200, "increment": 1, "rounds": 200},
    "dense_urban":  {"start_devices": 1, "max_devices": 500, "increment": 1, "rounds": 500},
}


def density_overrides(profile_name):
    """Return the device-ramp overrides for a named density profile.

    ``"custom"`` (or an empty/unknown name) returns ``{}`` so the run falls back
    to the MAX_DEVICES / ROUNDS set in config/run.py.
    """
    return dict(WORKLOAD_PROFILES.get(str(profile_name or "custom"), {}))


# =====================================================================
# 6G IMT-2030 service scenarios and generation-aware SLA deadlines
# (merged from the former config/scenarios.py). SLA_PROFILES is the single
# runtime source of per-class 5G/6G application deadlines; SCENARIO_PROFILES is
# the forward-looking descriptor layer. To add/change a deadline, edit
# SLA_PROFILES below (one place).
# =====================================================================
# Descriptor registry for the six IMT-2030 usage scenarios.
# Each entry is DATA a researcher can edit in one place.
SCENARIO_PROFILES = {
    # ---- Evolved from the three 5G classes (kept as the first three) ----------
    "URLLC": {
        "priority_int": 1,
        "imt2030_name": "Hyper-Reliable & Low-Latency Communication (HRLLC)",
        "evolves_5g_class": "URLLC",
        "is_new_6g": False,
        "plane": "user",
        "optimization_intent": "delay",        # protect delay / deadline
        "deadline_ms": 300,                     # matches PRIORITY_PROFILES[1]
        "notes": "Safety/emergency; tighten toward 0.1 ms + 1e-7 reliability in 6G.",
    },
    "mMTC": {
        "priority_int": 2,
        "imt2030_name": "Massive Communication",
        "evolves_5g_class": "mMTC",
        "is_new_6g": False,
        "plane": "user",
        "optimization_intent": "device_energy",  # protect battery
        "deadline_ms": 600,                       # matches PRIORITY_PROFILES[2]
        "notes": "Massive IoT; scale toward 1e6-1e8 devices/km^2 in 6G.",
    },
    "eMBB": {
        "priority_int": 3,
        "imt2030_name": "Immersive Communication",
        "evolves_5g_class": "eMBB",
        "is_new_6g": False,
        "plane": "user",
        "optimization_intent": "throughput_compute",  # bandwidth/compute-heavy
        "deadline_ms": 1000,                           # matches PRIORITY_PROFILES[3]
        "notes": "Immersive XR/holographic; hundreds of Mbps user rate in 6G.",
    },
    # ---- New 6G scenarios (the three genuinely new IMT-2030 additions) --------
    "ISAC": {
        "priority_int": 4,
        "imt2030_name": "Integrated Sensing & Communication",
        "evolves_5g_class": None,
        "is_new_6g": True,
        "plane": "sensing",
        "optimization_intent": "sensing_accuracy",  # comms-vs-sensing trade-off
        "deadline_ms": 200,
        "notes": "Network senses the city; couples a sensing-accuracy KPI to delivery "
                 "via TrafficObject.measurement_id.",
    },
    "AIComm": {
        "priority_int": 5,
        "imt2030_name": "AI & Communication (AI-as-a-Service)",
        "evolves_5g_class": None,
        "is_new_6g": True,
        "plane": "user",
        "optimization_intent": "compute_placement",  # model-serving / split inference
        "deadline_ms": 400,
        "notes": "Split/federated inference; uses TrafficObject.flow_role "
                 "(inference/gradient/activation).",
    },
    "Ubiquitous": {
        "priority_int": 6,
        "imt2030_name": "Ubiquitous Connectivity",
        "evolves_5g_class": None,
        "is_new_6g": True,
        "plane": "user",
        "optimization_intent": "coverage",  # connect-the-unconnected (NTN)
        "deadline_ms": 1500,
        "notes": "Non-terrestrial (LEO/HAPS) reach; ties to the deferred NTN tier.",
    },
}

SUPPORTED_ACCESS_GENERATIONS = ("5G", "6G")

# Runtime SLA contract for the three service classes currently generated by the
# workload module. Radio latency is a technology target; application deadline is
# the end-to-end budget used by the simulator and reward.
SLA_PROFILES = {
    "URLLC": {
        "5G": {
            "profile_id": "5g-urllc-v1",
            "radio_latency_target_ms": 1.0,
            "application_deadline_ms": 300.0,
            "maturity": "standard_radio_target_with_legacy_application_deadline",
            "source_id": "ITU-R-M.2410-0",
        },
        "6G": {
            "profile_id": "6g-hrllc-v1",
            "radio_latency_target_ms": 0.1,
            "application_deadline_ms": 100.0,
            "maturity": "experimental_calibrated",
            "source_id": "ITU-R-M.2160-0-plus-simulator-calibration-v1",
        },
    },
    "mMTC": {
        "5G": {
            "profile_id": "5g-mmtc-v1",
            "radio_latency_target_ms": 100.0,
            "application_deadline_ms": 600.0,
            "maturity": "experimental_modeling_assumption",
            "source_id": "SmartCityEdgeSim-SLA-calibration-v1",
        },
        "6G": {
            "profile_id": "6g-massive-v1",
            "radio_latency_target_ms": 10.0,
            "application_deadline_ms": 300.0,
            "maturity": "experimental_calibrated",
            "source_id": "ITU-R-M.2160-0-plus-simulator-calibration-v1",
        },
    },
    "eMBB": {
        "5G": {
            "profile_id": "5g-embb-v1",
            "radio_latency_target_ms": 4.0,
            "application_deadline_ms": 1000.0,
            "maturity": "standard_radio_target_with_legacy_application_deadline",
            "source_id": "ITU-R-M.2410-0",
        },
        "6G": {
            "profile_id": "6g-immersive-v1",
            "radio_latency_target_ms": 1.0,
            "application_deadline_ms": 500.0,
            "maturity": "experimental_calibrated",
            "source_id": "ITU-R-M.2160-0-plus-simulator-calibration-v1",
        },
    },
}


def normalize_access_generation(generation):
    """Return canonical 5G/6G text or fail loudly."""
    normalized = str(generation or "").strip().upper()
    if normalized not in SUPPORTED_ACCESS_GENERATIONS:
        raise ValueError(
            f"Unknown access generation {generation!r}. "
            f"Valid generations: {list(SUPPORTED_ACCESS_GENERATIONS)}"
        )
    return normalized


def get_sla(service_class, generation):
    """Resolve one service-class/generation SLA descriptor."""
    canonical_service = next(
        (
            name
            for name in SLA_PROFILES
            if name.lower() == str(service_class or "").strip().lower()
        ),
        None,
    )
    if canonical_service is None:
        raise KeyError(
            f"Unknown SLA service class {service_class!r}. "
            f"Valid classes: {sorted(SLA_PROFILES)}"
        )
    return SLA_PROFILES[canonical_service][normalize_access_generation(generation)]


def _validate_sla_profiles():
    profile_ids = set()
    for service_class, generations in SLA_PROFILES.items():
        if set(generations) != set(SUPPORTED_ACCESS_GENERATIONS):
            raise ValueError(f"{service_class} must define both 5G and 6G SLA profiles")
        for generation, profile in generations.items():
            if float(profile["radio_latency_target_ms"]) <= 0.0:
                raise ValueError(f"{service_class}/{generation} radio target must be positive")
            if float(profile["application_deadline_ms"]) <= 0.0:
                raise ValueError(f"{service_class}/{generation} deadline must be positive")
            profile_id = str(profile["profile_id"])
            if profile_id in profile_ids:
                raise ValueError(f"Duplicate SLA profile_id {profile_id!r}")
            profile_ids.add(profile_id)
        if (
            generations["6G"]["radio_latency_target_ms"]
            >= generations["5G"]["radio_latency_target_ms"]
            or generations["6G"]["application_deadline_ms"]
            >= generations["5G"]["application_deadline_ms"]
        ):
            raise ValueError(f"{service_class} 6G targets must be tighter than 5G")
        if (
            float(SCENARIO_PROFILES[service_class]["deadline_ms"])
            != float(generations["5G"]["application_deadline_ms"])
        ):
            raise ValueError(f"{service_class} legacy deadline must match its 5G SLA")


_validate_sla_profiles()

# Convenience: legacy priority int -> scenario key (URLLC=1, mMTC=2, eMBB=3, ...).
PRIORITY_INT_TO_SCENARIO = {
    profile["priority_int"]: key for key, profile in SCENARIO_PROFILES.items()
}


def get_scenario(name):
    """Return the descriptor for a scenario key (e.g. 'URLLC', 'ISAC').

    Pure lookup; raises KeyError listing valid scenarios if unknown so a
    misconfigured experiment fails loudly.
    """
    if name not in SCENARIO_PROFILES:
        raise KeyError(
            f"Unknown scenario {name!r}. Valid scenarios: {sorted(SCENARIO_PROFILES)}"
        )
    return SCENARIO_PROFILES[name]
