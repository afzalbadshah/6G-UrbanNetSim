"""Simulation-run toggles and timing/energy scales.

Print/log toggles for a run, the CSV flush cadence, the processing-time scales
that turn CPU demand into a duration, and the per-task energy-amortization and
congestion knobs. To change a run's logging or a timing scale, edit it here
(one place).
"""

# --- Per-round print / log toggles --------------------------------------
SIMULATION = {
    "rounds": 1,
    # Radio-access density is independent of SERVER_CONFIG MEC compute count.
    # Satellite-IoT scenario (config/zones.py): 6G towers in the city zone,
    # UAV-BS edge in the UAV zone, LEO satellite outside the city.
    # Use "fr3_city_coverage_6g" (and ZONES_ENABLED=False) for the original city.
    "deployment_profile": "zoned_satellite_6g",
    "interactive_deployment_profile": "zoned_satellite_6g",
    "print_cost": True,
    "print_congestion": True,
    "log_assignments": True,
    "print_metrics": False,
}

# Flush the wide per-decision offloading CSV every N rows (always flushed at
# model finish). Larger = faster during long runs.
SMARTCITY_CSV_FLUSH_INTERVAL = 1000

# --- Processing-time scales (CPU demand -> duration) --------------------
CPU_PROCESSING_TIME_SCALE_MS = 1000.0
MIN_PROCESSING_TIME_MS = 1.0
# On-device compute uses a smaller scale so an embedded device is not made to
# look slower than distant tiers.
LOCAL_PROCESSING_TIME_SCALE_MS = 80.0
LOCAL_MIN_PROCESSING_TIME_MS = 1.0

# --- Per-task energy amortization ---------------------------------------
# UAV hover/propulsion is continuous; share it across served tasks instead of
# charging a full 650 W round to every task.
PROPULSION_ENERGY_AMORTIZATION_TASKS = 1.0
UAV_BS_HOVER_ENERGY_AMORTIZATION_TASKS = 25.0

# --- Congestion-aware delay ---------------------------------------------
# effective_ms = base_ms / max(1 - u_avg, CONGESTION_HEADROOM_FLOOR).
# Smaller floor = sharper penalty near full capacity (0.05 caps at 20x).
CONGESTION_HEADROOM_FLOOR = 0.05
