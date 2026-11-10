"""Mobility, propagation, and handover configuration.

Covers where devices move (the city area and movement settings), how fast
signals propagate (wireless/wired speeds used for propagation delay), and how
robust handovers are.

To change movement, the city size, or handover behaviour, edit the values
below -- this is the one place these are defined.

PROVENANCE (audited 2026-10-01, market/standards references below each
sourced value). This file predates the Claude-owned `cite()`/provenance-dict
system used by config/topology.py, power.py, etc. (see config/_references.py);
it keeps its existing plain-comment citation style rather than converting to
that structure, but every physics/reliability value below now carries an
explicit source and maturity note, same as that system requires.

Audit summary: the handover and outage constants below were inherited from
5G-NR/LTE-era assumptions, not 6G ones -- they were internally consistent
(the arithmetic was correct) but cited the wrong generation's physics for a
project making 6G claims. Values are revised to the cited 6G/5G-Advanced
baselines; where no official 6G standard number exists yet, the nearest
defensible floor is used and flagged `experimental`, never invented.
"""

from config.servers import SERVER_CONFIG

# --- City area (the world devices move in) -------------------------------
CITY_AREA = {
    "xmin": 0.0, "xmax": 10000.0,
    "ymin": 0.0, "ymax": 10000.0,
}

# --- Movement ------------------------------------------------------------
# num_entities/time_step_ms are a simulation SCALE choice (compute budget and
# position-update cadence), not a physics parameter with a market reference.
# For context only: IMT-2030 (6G) research targets a connection density of
# 10^6-10^8 devices/km^2 (ITU-R M.2160-0) -- at this city's 100 km^2 extent
# that is 10^8-10^10 devices, far beyond any feasible simulation population,
# so num_entities is sized for compute budget, not literal density matching.
MOBILITY_CONFIG = {
    "enabled": True,
    "apply_to_servers": False,   # infra is usually fixed; True makes it mobile too
    "num_entities": 50,
    "time_step_ms": 5000,        # larger step => larger per-round movement (5 s)
    # Base/signaling handover latency. 5G NR legacy (break-before-make) RRC
    # reconfiguration processing alone is ~10 ms (3GPP TS 38.331) before any
    # radio interruption; that is the wrong generation's number to cite here.
    # 6G URLLC research assumes sub-millisecond-to-single-millisecond handover
    # latency (e.g. arXiv:2508.20205, "A Comprehensive Survey of 5G URLLC and
    # Challenges in the 6G Era"). 1 ms is the conservative (slower) end of
    # that band -- experimental, no single official 6G standard figure exists
    # yet to cite more precisely.
    "handover_latency_ms": 1,
    # A3-event hysteresis margin: 3.0 dB is the literature-typical value for
    # LTE/NR handover hysteresis (commonly cited range 1-3 dB; see e.g.
    # TelecomTrainer, "Understanding Event A3"). Confirmed correct as-is;
    # no change made.
    "handover_threshold_db": 3,
    "area": CITY_AREA,           # keep entities inside the city area
}

# --- Propagation (distances + speeds -> propagation delay) ---------------
# WIRELESS_PROP_SPEED / WIRED_PROP_SPEED are physical constants (speed of
# light in air; ~c/1.5 for a typical fibre refractive index), not market
# values -- confirmed correct, unchanged.
#
# The four constants below (BS_CLOUD_DISTANCE_M, WIRELESS_DELAY_SCALE,
# WIRED_DELAY_SCALE, OB_PROP_DELAY_MS) are NOT read by any live delay
# computation (verified: no import/reference anywhere outside this file and
# tests) -- they are inert leftovers from the pre-metrics/delay.py legacy
# formula path and do not contribute to any of the audited delay numbers.
# Kept for now rather than deleted unasked; flag if you want them removed.
BS_CLOUD_DISTANCE_M = 1_000_000.0
# BS (ground MEC) -> Metro aggregation backbone distance: a real wired hop.
BS_METRO_DISTANCE_M = SERVER_CONFIG.get("Metro", {}).get("distance", 10_000.0)
WIRELESS_PROP_SPEED  = 3.0e8     # ~speed of light in air
WIRED_PROP_SPEED     = 2.0e8     # ~fiber
WIRELESS_DELAY_SCALE = 1000
WIRED_DELAY_SCALE    = 10        # with the 1,000 km backbone gives ~50 ms wired path delay
OB_PROP_DELAY_MS = 0.20

# --- Handover robustness -------------------------------------------------
# Radio-link interruption while a handover executes. 3GPP Release 16 DAPS
# (Dual Active Protocol Stack) handover -- already standardized, not a future
# target -- keeps source and target cell active simultaneously and reduces
# interruption from the legacy 30-60 ms break-before-make figure to
# "effectively 0 ms" (source: RF Essentials, "DAPS handover"; Ericsson,
# "Reducing mobility interruption time in 5G networks"). 6G URLLC research
# assumes sub-ms-to-1ms handover latency more broadly (arXiv:2508.20205).
# 1.0 ms models DAPS-class make-before-break handover as a conservative
# nonzero value within that band -- experimental (no single official 6G
# figure to cite), but no longer citing 4G/legacy-5G break-before-make
# physics (was 25 ms, 25-50x higher than DAPS/6G assumptions).
HANDOVER_INTERRUPTION_MS = 1.0
# Probability an attempted handover fails. 5G NR KPI targets: Handover
# Preparation and Execution Success Rate both > 98%, i.e. failure < 2%
# (telecomhall.net, "Understanding 5G Handover Failures"; cnrs attribution
# pending vendor KPI docs). 1% sits below that 5G ceiling -- 6G is expected
# to improve on 5G via AI-assisted/predictive handover, but no official 6G
# HOF target was found, so this is held at "better than 5G" rather than
# invented further. Was 3%, above the 5G target ceiling.
HANDOVER_FAILURE_PROB = 0.01
DWELL_TIME_ERROR_STD = 0.15       # relative std-dev of dwell-time prediction error
                                   # (simulator-internal prediction-noise assumption;
                                   # no market/standards benchmark exists for this
                                   # quantity, so left unchanged rather than citing
                                   # a figure that would not be real)
# Transient coverage-outage probability per round. 5G/6G URLLC reliability
# baseline is 99.999% ("five nines"), i.e. <=1e-5 outage probability; 6G
# xURLLC research pushes to 1e-6-1e-8 (arXiv:2510.08080, "URLLC for 6G
# Enabled Industry 5.0"; search summary citing 6G targets up to 99.99999%
# reliability). 1e-5 matches the five-nines floor -- the least aggressive
# defensible 6G/URLLC figure, not the tightest one found. Was 0.01 (1e-2),
# 1,000x looser than even the baseline five-nines target.
#
# NOTE: this probability still gates a full MOBILITY_CONFIG["time_step_ms"]
# (5 s) duration per firing when it does trigger (see handover_scheduler.py)
# -- that duration-to-round-length coupling is a separate, still-open issue
# flagged previously; lowering the probability here reduces how often it
# fires but does not fix that coupling. Ask if you want that decoupled too.
OUTAGE_PROBABILITY = 0.00001
DWELL_EXPIRE_THRESHOLD_MS = 1.0   # remaining dwell below this => "dwell_expired"
                                   # (numerical epsilon for trigger logic, not a
                                   # market-referenced quantity; unchanged)
