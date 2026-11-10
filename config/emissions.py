"""
Carbon-intensity and embodied-emissions profiles (config side, DATA only).

Companion to config/energy.py: that file prices electrical energy in watts and
joules; this file converts energy into operational grams of CO2-equivalent,
and declares each infrastructure tier's manufacturing (embodied) footprint.

STATUS: sourced configuration DATA; no runtime emissions state, no simulation
formulas. metrics/emissions.py computes:
  * operational CO2e   = grid-drawn energy (J) x effective grid intensity (g/J)
  * embodied CO2e/year = per-tier manufacturing footprint (kg) / hardware
                          lifetime (years), summed across a deployment's
                          declared unit counts

Ownership: sourced parameter DATA, cited via config/_references.py. Every
profile carries a provenance record; see llm_collab/MESSAGES.md for the
research pass that grounded these values (EPA eGRID, GSMA, IEA, Ericsson,
Dell/Boavizta, academic drone-LCA review).

Maturity honesty: the grid-average carbon intensity and the ~24% renewable
share are each a single real published figure applied uniformly to this
simulated city -- standards-aligned as data, experimental as an assumption
about THIS deployment, so profiles below are conservatively labeled
`experimental` throughout (same convention config/energy.py already uses for
mixed standards-aligned/estimated profiles).
"""

from config._provenance import deep_freeze
from config._references import cite

# ---- Provenance records -----------------------------------------------------
_P_GRID_INTENSITY = cite(
    "epa_egrid_2023", maturity="experimental",
    validity_notes=(
        "US eGRID national average CO2 emission rate for 2023 is a real "
        "published federal figure (~370 gCO2/kWh); applying one national "
        "average to this simulated city, rather than a location-specific or "
        "time-varying grid mix, is a modeling simplification."))
_P_RENEWABLE_SHARE = cite(
    "gsma_mobile_net_zero_2026", maturity="experimental",
    validity_notes=(
        "GSMA Mobile Net Zero 2026 (2024 data, 110+ operators, 85% of global "
        "mobile connections): ~24% of purchased/generated electricity is "
        "renewable. Independently cross-validated by IEA's ~24% figure for "
        "US data-centre physical grid draw (see iea_data_centres_2024) -- two "
        "different industries, two different methodologies, same figure, so "
        "one shared ratio is used here rather than a fabricated per-tier "
        "spread. This is a location-based (physical-draw) figure, not "
        "market-based accounting (power-purchase agreements/RECs can push an "
        "individual operator's contractual renewable share much higher)."))
_P_SERVER_EMBODIED = cite(
    "server_pcf_2024", maturity="experimental",
    validity_notes=(
        "Dell R740 (2x24-core CPU, 384GB RAM, 400GB SSD) product carbon "
        "footprint ~910 kgCO2e, cross-validated by Boavizta's independent "
        "component model at ~970 kgCO2e; ~940 kgCO2e used as the midpoint. "
        "Anchors the MEC tier (same server class); Metro/Cloud are scaled by "
        "this project's own existing tier capacity ratios "
        "(config/servers.py: Onboard 1x, UAV_BS 4x, MEC 93x, Metro 2,100x, "
        "Cloud 7,500x) -- a modeling assumption, not an independent citation "
        "for those two tiers."))
_P_RAN_EMBODIED = cite(
    "ran_pcf_ericsson", maturity="experimental",
    validity_notes=(
        "Ericsson RBS 6201 radio base station published product carbon "
        "footprint: 4.4 tCO2e (raw materials 41%, manufacturing/assembly "
        "36%, transport 15%, support 9%). Applied here as the ground-access "
        "(BS_*) embodied figure; a specific product's footprint, not an "
        "industry average."))
_P_DEVICE_EMBODIED = cite(
    "device_pcf_2025", maturity="experimental",
    validity_notes=(
        "Smartphone-class published environmental reports average ~55 "
        "kgCO2e (Pixel 10 60.2 kg, iPhone 17 50.7 kg). Used as the Onboard/ "
        "end-device embodied figure; this project's mMTC-class devices "
        "(sensors/meters) are far lighter than a smartphone, so this is a "
        "conservative (high) proxy, not a device-specific LCA."))
_P_UAV_EMBODIED = cite(
    "drone_lca_review", maturity="experimental",
    validity_notes=(
        "Academic drone-LCA review reports ~28.5 kgCO2e for a small consumer/ "
        "delivery-class airframe -- a SIZE-CLASS MISMATCH against this "
        "simulator's UAV_BS (650 W hover, heavy-lift multi-Orin payload per "
        "config/servers.py), which has no clean citable match. Kept "
        "separate from and added to a scaled onboard-compute embodied "
        "estimate (see EMBODIED_CARBON_PROFILES['uav_bs']); flagged "
        "explicitly rather than silently used as if size-appropriate."))
_P_LIFETIME = cite(
    "hardware_refresh_cycle", maturity="experimental",
    validity_notes=(
        "Server/device hardware refresh cycles are typically 3-5 years; RAN "
        "equipment is typically longer-lived (~10 years, consistent with "
        "the 7-20 year range reported for earlier-generation base "
        "stations). Midpoints used: 4 years (server/device), 10 years (RAN), "
        "3 years (UAV airframe, shorter due to flight-hours wear -- no "
        "citation, scenario assumption)."))


# ---- Grid carbon intensity ---------------------------------------------------
# US eGRID national average (2023). Paired with config.servers.
# ELECTRICITY_PRICE_PER_JOULE, which is also US/EIA-anchored, for geographic
# consistency between the existing price model and this new emissions model.
CARBON_INTENSITY_G_CO2_PER_KWH = 370.0

# Renewable share nets out of the grid-average intensity: the fraction of
# drawn electricity assumed carbon-free. One shared ratio across every tier
# (see _P_RENEWABLE_SHARE) -- not a fabricated per-tier spread.
RENEWABLE_ENERGY_SHARE_RATIO = 0.24

EFFECTIVE_GRID_CARBON_INTENSITY_G_CO2_PER_KWH = CARBON_INTENSITY_G_CO2_PER_KWH * (
    1.0 - RENEWABLE_ENERGY_SHARE_RATIO
)
# 1 kWh = 3.6e6 J.
CARBON_INTENSITY_G_CO2_PER_JOULE = EFFECTIVE_GRID_CARBON_INTENSITY_G_CO2_PER_KWH / 3.6e6

GRID_CARBON_INTENSITY_PROVENANCE = _P_GRID_INTENSITY
RENEWABLE_SHARE_PROVENANCE = _P_RENEWABLE_SHARE


# ---- Embodied (manufacturing) carbon, amortized per year of hardware life --
_SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0

# Tier capacity ratios already established in config/servers.py's own
# docstring ("Tier ratios follow published deployment classes"), reused here
# instead of inventing a second, independent scaling scheme.
_TIER_CAPACITY_RATIO = {"onboard": 1.0, "uav_bs": 4.0, "mec": 93.0, "metro": 2_100.0, "cloud": 7_500.0}
_SERVER_ANCHOR_KG_CO2E = 940.0  # Dell R740-class server, midpoint of 910-970 kg


def _scaled_server_kg(tier: str) -> float:
    return _SERVER_ANCHOR_KG_CO2E * (_TIER_CAPACITY_RATIO[tier] / _TIER_CAPACITY_RATIO["mec"])


_RAW_EMBODIED_CARBON_PROFILES = {
    # End device (originating device / LOCAL_DEVICE compute). Reported for
    # completeness; not currently amortized per-task (see module docstring
    # and metrics/emissions.py -- attributing a shared or owned device's
    # embodied footprint per task risks double-counting across many tasks
    # from the same device, so this is a deployment-level, not per-row,
    # figure, same as every other entry here).
    "onboard": {
        "embodied_kg_co2e": 55.0,
        "lifetime_years": 4.0,
        "provenance": _P_DEVICE_EMBODIED,
    },
    "mec": {
        "embodied_kg_co2e": _scaled_server_kg("mec"),
        "lifetime_years": 4.0,
        "provenance": _P_SERVER_EMBODIED,
    },
    "metro": {
        "embodied_kg_co2e": _scaled_server_kg("metro"),
        "lifetime_years": 4.0,
        "provenance": _P_SERVER_EMBODIED,
    },
    "cloud": {
        "embodied_kg_co2e": _scaled_server_kg("cloud"),
        "lifetime_years": 4.0,
        "provenance": _P_SERVER_EMBODIED,
    },
    # Ground access radio (BS_*/promoted-MEC radio role): Ericsson RBS 6201
    # published footprint, independent of the MEC compute-server figure above
    # (a ground site's compute and radio hardware are manufactured and
    # amortized separately).
    "ground_bs": {
        "embodied_kg_co2e": 4_400.0,
        "lifetime_years": 10.0,
        "provenance": _P_RAN_EMBODIED,
    },
    # UAV-BS: airframe (drone-LCA figure, flagged size-mismatch) + a scaled
    # onboard-compute share, kept as separate addends so the mismatch stays
    # visible rather than hidden inside one blended number.
    "uav_bs": {
        "embodied_kg_co2e": 28.5 + _scaled_server_kg("uav_bs"),
        "lifetime_years": 3.0,
        "provenance": _P_UAV_EMBODIED,
    },
}

EMBODIED_CARBON_PROFILES = deep_freeze(_RAW_EMBODIED_CARBON_PROFILES)
LIFETIME_PROVENANCE = _P_LIFETIME


def get_embodied_carbon_profile(tier: str):
    """Return the embodied-carbon profile for a named tier (KeyError if unknown)."""
    key = str(tier or "").strip().lower()
    if key not in EMBODIED_CARBON_PROFILES:
        raise KeyError(
            f"Unknown embodied-carbon tier {tier!r}. Valid: {sorted(EMBODIED_CARBON_PROFILES)}"
        )
    return EMBODIED_CARBON_PROFILES[key]


def annualized_embodied_kg_co2e(tier: str, *, unit_count: int) -> float:
    """Return this tier's total amortized embodied CO2e (kg) for one year.

    ``unit_count`` is how many physical units of that tier are deployed (e.g.
    SERVER_CONFIG["MEC"]["num_datacenters"], or a deployment profile's
    ground_bs_count). Deployment-level, not per-task -- see module docstring.
    """
    if isinstance(unit_count, bool) or not isinstance(unit_count, int) or unit_count < 0:
        raise ValueError("unit_count must be a non-negative integer")
    profile = get_embodied_carbon_profile(tier)
    per_unit_per_year = float(profile["embodied_kg_co2e"]) / float(profile["lifetime_years"])
    return per_unit_per_year * unit_count
