"""Physical scheduler availability and battery-operating settings.

Learned service-class preferences belong only to the reward specification.
The learned runner attempts exactly the selected action. Legacy spillover
settings are retained for configuration compatibility, but never replace a
learned choice. The operating threshold only scales the overload penalty.
"""

# --- Physical availability ----------------------------------------------
LOCAL_DEVICE_ENABLED = True

# --- Compute spillover --------------------------------------------------
# A selected tier is still physically admissible up to 100%.  At this softer
# operating threshold the runtime first tries the next outward tier so that a
# task does not push a shared resource into the unstable saturation region.
# Unavailable tiers are skipped. Cloud is terminal and therefore accepts any
# task that still satisfies its physical (100%) capacity constraint.
COMPUTE_SPILLOVER_ENABLED = False
COMPUTE_SPILLOVER_THRESHOLD_PERCENT = 95.0
COMPUTE_SPILLOVER_ORDER = (
    "LOCAL_DEVICE",
    "MEC",
    "UAV_BS",
    "METRO",
    "CLOUD",
    "SAT",       # satellite-IoT zones: UAV/local overflow outside the city
)
COMPUTE_SPILLOVER_TERMINAL_TIER = "CLOUD"

# --- Battery operation ---------------------------------------------------
BATTERY_MIN_LOCAL_EXEC_RATIO = 0.20
BATTERY_LOW_RATIO = 0.30
URLLC_LOCAL_MIN_BATTERY_RATIO = 0.25
UAV_BS_MIN_BATTERY_RATIO = 0.25
# Rounds a UAV_BS stays grounded (return-to-base swap) after its battery drops
# below UAV_BS_MIN_BATTERY_RATIO, then it resets to full charge.
UAV_BS_RECHARGE_ROUNDS = 20
