"""
Energy / power profiles (config side, DATA only).

How to extend: to add or change a radio/compute/transport/UAV power profile, edit
`POWER_PROFILES` (one place); cite the source via `config/_references.py`.

STATUS: sourced configuration DATA; no runtime energy state, no formulas. Codex's
`metrics/energy.py` (R3) computes energy = power_w * duration_s from these
immutable parameters. Per IMPLEMENTATION_CONTRACT.md §13, radio/compute/transport/
UAV components are kept SEPARATE and each is independently reportable before any
total is formed.

Ownership: Claude (sourced parameter DATA). Every profile carries a §11 provenance
record. Units follow §2: power in watts (`_w`), ratios unitless (`_ratio`), energy
joules (`_j`), duration seconds (`_s`).

KEY REALISM RULES (§13):
- A radio transmission's PA input power = `radiated_tx_power_w / pa_efficiency_ratio`
  (formula lives in metrics/energy.py, NOT here); plus circuit power over airtime.
- Received signal power (dBm) is NEVER receiver energy — RX energy uses an explicit
  `rx_circuit_power_w`.
- Idle/sleep/cooling/propulsion/radio/transport/compute stay separate.

Maturity honesty (§11): experimental candidate values are NOT calibrated. Only
`radiated_tx_power_w` (3GPP UE power class) and PUE (Uptime/Green Grid) are
standards-aligned; front-end circuit powers and UAV/transport values are
experimental until measurement calibration.
"""

from config._provenance import deep_freeze
from config._references import cite

# ---- Provenance records -----------------------------------------------------
# Provenance (contract §11). Sources are cited by short id from config/_references.py;
# `validity_notes` documents THIS value. A MIXED profile (one standards-aligned
# sub-value + uncalibrated estimates) is `experimental` as a whole, with the
# standards-aligned sub-value named in its notes.
_P_DEVICE_RADIO = cite(
    "3gpp_ue_power", maturity="experimental",
    validity_notes=("radiated_tx_power_w=0.2 W (23 dBm) is 3GPP UE-class "
                    "standards-aligned; PA efficiency and TX/RX/idle/sleep circuit "
                    "powers are uncalibrated research estimates"))
_P_ONBOARD = cite(
    "krishnan2021_uav", maturity="experimental",
    validity_notes="Jetson-class embedded SoC ~3 W active; academic research estimate, not a standard")
_P_COMPUTE_TIER = cite(
    "green_grid_pue", maturity="experimental",
    validity_notes=("pue_ratio is Uptime/Green-Grid standards-aligned; idle/max/"
                    "dynamic server power values are uncalibrated research estimates"))
_P_DC = cite(
    "dc_server_power", maturity="experimental",
    validity_notes="per-tier server active/idle power; scenario estimate, uncalibrated")
_P_ACCESS_RADIO = cite(
    "access_radio", maturity="experimental",
    validity_notes="BS TX radiated ~6.3 W (38 dBm) small-cell class; higher circuit "
                   "power than a handset; uncalibrated")
_P_UAV_RADIO = cite(
    "uav_radio", maturity="experimental",
    validity_notes="aerial BS radio, separate from propulsion and onboard compute; "
                   "TX radiated ~1 W (30 dBm); uncalibrated")
_P_UAV = cite(
    "sim_default", maturity="experimental",
    validity_notes="UAV-BS hover power ~650 W heavy-lift multi-Orin payload; not measured")
_P_FIBRE = cite(
    "fibre_energy", maturity="experimental",
    validity_notes="fibre/backhaul active-power + energy-per-bit; order-of-magnitude")


def _radio(radiated_tx_power_w, radiated_tx_power_max_w, pa_efficiency_ratio,
           tx_circuit_power_w, rx_circuit_power_w, idle_power_w, sleep_power_w, prov,
           transition_energy_j=0.0, transition_duration_s=0.0):
    if radiated_tx_power_max_w < radiated_tx_power_w:
        raise ValueError("radiated_tx_power_max_w must be >= radiated_tx_power_w")
    return {
        "kind": "radio",
        "radiated_tx_power_w": float(radiated_tx_power_w),      # default operating point
        "radiated_tx_power_max_w": float(radiated_tx_power_max_w),
        "pa_efficiency_ratio": float(pa_efficiency_ratio),
        "tx_circuit_power_w": float(tx_circuit_power_w),
        "rx_circuit_power_w": float(rx_circuit_power_w),
        "idle_power_w": float(idle_power_w),
        "sleep_power_w": float(sleep_power_w),
        "transition_energy_j": float(transition_energy_j),
        "transition_duration_s": float(transition_duration_s),
        "provenance": prov,
    }


def _compute(idle_power_w, max_power_w, dynamic_power_w, utilization_exponent,
             pue_ratio, prov):
    return {
        "kind": "compute",
        "idle_power_w": float(idle_power_w),
        "max_power_w": float(max_power_w),
        "dynamic_power_w": float(dynamic_power_w),
        "utilization_exponent": float(utilization_exponent),
        "pue_ratio": float(pue_ratio),
        "provenance": prov,
    }


_TRANSPORT_METHODS = ("energy_per_bit", "active_power")


def _transport(accounting_method, active_power_w, idle_power_w, energy_per_bit_j,
               technology, prov):
    """One transport profile. `accounting_method` (§13) disambiguates which basis a
    consumer uses so the two are never added (double-count). The selected basis's
    required input must be positive."""
    if accounting_method not in _TRANSPORT_METHODS:
        raise ValueError(f"accounting_method must be one of {_TRANSPORT_METHODS}")
    if accounting_method == "energy_per_bit" and not (energy_per_bit_j > 0.0):
        raise ValueError("energy_per_bit method requires energy_per_bit_j > 0")
    if accounting_method == "active_power" and not (active_power_w > 0.0):
        raise ValueError("active_power method requires active_power_w > 0")
    return {
        "kind": "transport",
        "accounting_method": accounting_method,
        "active_power_w": float(active_power_w),
        "idle_power_w": float(idle_power_w),
        "energy_per_bit_j": float(energy_per_bit_j),
        "technology": technology,
        "provenance": prov,
    }


def _uav(hover_power_w, propulsion_power_w, mobility_power_w, prov):
    return {
        "kind": "uav",
        "hover_power_w": float(hover_power_w),
        "propulsion_power_w": float(propulsion_power_w),
        "mobility_power_w": float(mobility_power_w),
        "provenance": prov,
    }


_RAW_POWER_PROFILES = {
    # -- Radio front-ends -----------------------------------------------------
    # Device UE radio: radiated 0.2 W (=23 dBm), PA 35%, explicit TX/RX circuit.
    "device_radio_default": _radio(
        radiated_tx_power_w=0.2, radiated_tx_power_max_w=0.2, pa_efficiency_ratio=0.35,
        tx_circuit_power_w=0.1, rx_circuit_power_w=0.05,
        idle_power_w=0.01, sleep_power_w=0.001, prov=_P_DEVICE_RADIO),
    # Ground MEC/BS access radio — higher radiated power + circuit power than a handset.
    "access_radio_default": _radio(
        radiated_tx_power_w=6.3, radiated_tx_power_max_w=10.0, pa_efficiency_ratio=0.40,
        tx_circuit_power_w=5.0, rx_circuit_power_w=3.0,
        idle_power_w=1.0, sleep_power_w=0.2, prov=_P_ACCESS_RADIO),
    # UAV-BS access/backhaul radio electronics — separate from propulsion & compute.
    "uav_bs_radio_default": _radio(
        radiated_tx_power_w=1.0, radiated_tx_power_max_w=2.0, pa_efficiency_ratio=0.35,
        tx_circuit_power_w=2.0, rx_circuit_power_w=1.5,
        idle_power_w=0.3, sleep_power_w=0.05, prov=_P_UAV_RADIO),

    # -- Compute tiers (idle/max/dynamic power + PUE boundary) -----------------
    "onboard_compute": _compute(
        idle_power_w=0.1, max_power_w=3.0, dynamic_power_w=2.9,
        utilization_exponent=1.0, pue_ratio=1.0, prov=_P_ONBOARD),
    "mec_compute": _compute(
        idle_power_w=300.0, max_power_w=1000.0, dynamic_power_w=700.0,
        utilization_exponent=1.0, pue_ratio=1.35, prov=_P_COMPUTE_TIER),
    "metro_compute": _compute(
        idle_power_w=3000.0, max_power_w=10000.0, dynamic_power_w=7000.0,
        utilization_exponent=1.0, pue_ratio=1.45, prov=_P_COMPUTE_TIER),
    "cloud_compute": _compute(
        idle_power_w=8000.0, max_power_w=30000.0, dynamic_power_w=22000.0,
        utilization_exponent=1.0, pue_ratio=1.20, prov=_P_COMPUTE_TIER),
    "uav_bs_compute": _compute(
        idle_power_w=10.0, max_power_w=60.0, dynamic_power_w=50.0,
        utilization_exponent=1.0, pue_ratio=1.10, prov=_P_DC),

    # -- Transport (wired). accounting_method selects the single basis a consumer
    #    uses (§13) so active-power and energy-per-bit are never double-counted.
    "transport_fibre_default": _transport(
        accounting_method="energy_per_bit",
        active_power_w=50.0, idle_power_w=20.0, energy_per_bit_j=1.0e-8,
        technology="fibre_backhaul_core", prov=_P_FIBRE),

    # -- UAV mobility (kept separate from radio and compute) ------------------
    "uav_bs_mobility": _uav(
        hover_power_w=650.0, propulsion_power_w=650.0, mobility_power_w=700.0,
        prov=_P_UAV),
}

POWER_PROFILES = deep_freeze(_RAW_POWER_PROFILES)

# Required keys per kind (published, immutable schema contract for R3 consumers).
REQUIRED_KEYS = deep_freeze({
    "radio": {"radiated_tx_power_w", "radiated_tx_power_max_w", "pa_efficiency_ratio",
              "tx_circuit_power_w", "rx_circuit_power_w", "idle_power_w", "sleep_power_w",
              "transition_energy_j", "transition_duration_s", "provenance", "kind"},
    "compute": {"idle_power_w", "max_power_w", "dynamic_power_w",
                "utilization_exponent", "pue_ratio", "provenance", "kind"},
    "transport": {"accounting_method", "active_power_w", "idle_power_w",
                  "energy_per_bit_j", "technology", "provenance", "kind"},
    "uav": {"hover_power_w", "propulsion_power_w", "mobility_power_w",
            "provenance", "kind"},
})

# The required input for each transport accounting method (§13 selector).
TRANSPORT_METHOD_REQUIRED_INPUT = deep_freeze({
    "energy_per_bit": "energy_per_bit_j",
    "active_power": "active_power_w",
})


def get_power(name):
    """Return a deep-immutable power profile by name (KeyError lists valid names)."""
    if name not in POWER_PROFILES:
        raise KeyError(f"Unknown power profile {name!r}. Valid: {sorted(POWER_PROFILES)}")
    return POWER_PROFILES[name]
