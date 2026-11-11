# Representative smart-city device classes.
#
# Basis for these profiles:
# - 3GPP/ITU 5G service families: URLLC, mMTC, eMBB.
#   https://www.itu.int/rec/R-REC-M.2083
#   https://www.3gpp.org/technologies/5g-system-overview
# - Smart-city application classes commonly reported in surveys:
#   traffic control, video surveillance, utilities/meters, parking,
#   environmental monitoring, smart lighting, public safety, transport.
#   https://arxiv.org/abs/1909.08747
#   https://arxiv.org/abs/2202.04312
# - TALQ smart-city domains include lighting, parking, waste, traffic,
#   energy, and environmental data collection.
#   https://www.talq-consortium.org/
#
# Notes:
# - Priority class deadlines and tier order are intentionally not defined here;
#   they are centralized in config.PRORITY_PROFILES.
# - CPU/memory are simulator capacity units, not physical MHz/GFLOPS.
# - battery.capacity_j = 0.0 means mains-powered or externally powered.
# - On-device compute capacity/power is PER-DEVICE-TYPE, sized to the real
#   hardware class each profile represents (embedded MCU vs. automotive SoC
#   vs. edge-AI camera chip, etc.) via local_compute.cpu/memory/power_w.
#   config.SERVER_CONFIG["Onboard"] is now only the fallback default used if
#   a capable profile omits its own spec. local_compute.capable still gates
#   whether this device class can run the offloadable workload at all
#   (MCU-class sensors cannot, regardless of any cpu/memory/power_w value).

DEVICE_PROFILES = {
    "traffic_controller": {
        "service_class": "URLLC",
        "weight": 0.012,
        "application_type": "traffic_signal_control",
        "power_source": "mains_with_backup",
        "radio_type": "5g_urllc",
        "reporting_pattern": {"period_s": (0.05, 0.5), "burst_probability": 0.20},
        "workload": {"data_multiplier": 0.20, "cpu_multiplier": 0.80, "memory_per_kb": 0.20},
        # Embedded industrial controller (signal timing + sensor fusion) --
        # weakest capable device on the list, but still real control logic.
        "local_compute": {"capable": True, "cpu": 2_000, "memory": 1_536, "power_w": 2.0},
        "battery": {"capacity_j": 222.7, "initial_j": 213.8},
        "radio_energy": {"tx_power_w": 0.70, "rx_power_w": 0.25, "idle_power_w": 0.05},
        "mobility": {"profile": "fixed", "kind": "stationary", "speed_m_s": (0.0, 0.0), "pause_s": (600, 3600), "handover_sensitivity": 0.05, "dwell_weight": 0.1},
    },
    "emergency_vehicle": {
        "service_class": "URLLC",
        "weight": 0.030,
        "application_type": "emergency_response",
        "power_source": "vehicle_battery",
        "radio_type": "5g_urllc",
        "reporting_pattern": {"period_s": (0.05, 0.2), "burst_probability": 0.50},
        "workload": {"data_multiplier": 0.40, "cpu_multiplier": 1.40, "memory_per_kb": 0.28},
        # Ruggedized MDT + sensor-fusion computer -- sized to keep meeting
        # URLLC's deadline through the full workload ramp on its own chip.
        "local_compute": {"capable": True, "cpu": 12_000, "memory": 8_000, "power_w": 15.0},
        "battery": {"capacity_j": 85.5, "initial_j": 82.1},
        "radio_energy": {"tx_power_w": 1.20, "rx_power_w": 0.40, "idle_power_w": 0.08},
        # Speed up to ~120 km/h (33 m/s), matching Pakistani motorway/emergency-response speeds.
        "mobility": {"profile": "vehicle_urban", "kind": "vehicle", "speed_m_s": (10.0, 33.0), "pause_s": (0, 3), "handover_sensitivity": 0.85, "dwell_weight": 1.6},
    },
    "connected_car": {
        "service_class": "URLLC",
        "weight": 0.060,
        "application_type": "connected_vehicle_safety",
        "power_source": "vehicle_battery",
        "radio_type": "5g_v2x",
        "reporting_pattern": {"period_s": (0.1, 1.0), "burst_probability": 0.25},
        "workload": {"data_multiplier": 0.30, "cpu_multiplier": 1.00, "memory_per_kb": 0.24},
        # Automotive ADAS/infotainment SoC -- sized to keep meeting URLLC's
        # deadline through the workload ramp, with headroom.
        "local_compute": {"capable": True, "cpu": 6_000, "memory": 4_096, "power_w": 8.0},
        "battery": {"capacity_j": 548.8, "initial_j": 521.3},
        "radio_energy": {"tx_power_w": 1.00, "rx_power_w": 0.35, "idle_power_w": 0.07},
        # Speed up to ~120 km/h (33 m/s) on motorways; slower in city traffic.
        "mobility": {"profile": "vehicle_urban", "kind": "vehicle", "speed_m_s": (6.0, 33.0), "pause_s": (0, 8), "handover_sensitivity": 0.75, "dwell_weight": 1.4},
    },
    "public_transport_bus": {
        "service_class": "eMBB",
        "weight": 0.288,
        "application_type": "bus_telemetry_video",
        "power_source": "vehicle_battery",
        "radio_type": "5g_embb",
        "reporting_pattern": {"period_s": (0.5, 5.0), "burst_probability": 0.35},
        # eMBB is compute-HEAVY (real-time multi-camera video encode/analytics):
        # its compute demand far exceeds what an embedded device can serve in
        # real time, so cpu_multiplier is large. This is what makes LOCAL read as
        # under-resourced for eMBB (resource_fit -> low) and pushes it to MEC/
        # Metro/Cloud through outcomes, with no mask.
        "workload": {"data_multiplier": 1.80, "cpu_multiplier": 16.0, "memory_per_kb": 0.30},
        # Bus telemetry + onboard video system -- continuous multi-sensor/
        # camera processing, same class as a city gateway.
        "local_compute": {"capable": True, "cpu": 8_000, "memory": 6_144, "power_w": 10.0},
        "battery": {"capacity_j": 123.5, "initial_j": 117.4},
        "radio_energy": {"tx_power_w": 1.10, "rx_power_w": 0.40, "idle_power_w": 0.09},
        "mobility": {"profile": "bus_route", "kind": "vehicle", "speed_m_s": (4.0, 20.0), "pause_s": (5, 45), "handover_sensitivity": 0.55, "dwell_weight": 1.2},
    },
    "drone": {
        "service_class": "URLLC",
        "weight": 0.020,
        "application_type": "aerial_monitoring",
        "power_source": "battery",
        "radio_type": "5g_embb_urllc",
        "reporting_pattern": {"period_s": (0.05, 0.5), "burst_probability": 0.45},
        "workload": {"data_multiplier": 0.35, "cpu_multiplier": 1.80, "memory_per_kb": 0.32},
        # Jetson-class companion computer [R6] -- deliberately NOT sized up
        # like the other URLLC vehicles: a task-generating drone's SWaP
        # budget keeps it at the same class as the old universal Onboard
        # default, so it realistically keeps escalating heavy tasks to the
        # dedicated UAV_BS payload (also Jetson-class, but heavy-lift/multi-
        # Orin) rather than always self-serving.
        "local_compute": {"capable": True, "cpu": 3_000, "memory": 2_048, "power_w": 5.0},
        # ~75Wh small multi-rotor pack (physically real scale -- the previous
        # 13.1J placeholder made any realistic propulsion wattage deplete it
        # in a fraction of one round). At propulsion_power_w=150 (typical
        # hover draw for a sensor-payload quadrotor) this gives ~360 rounds
        # (~30 min real flight time) of pure-flight endurance; comm/compute
        # energy is negligible on top of that (sub-0.01J per task).
        "battery": {"capacity_j": 270_000.0, "initial_j": 245_000.0},
        "radio_energy": {"tx_power_w": 1.30, "rx_power_w": 0.40, "idle_power_w": 0.08},
        # Continuous flight/propulsion power draw (W) -- independent of
        # whether the drone transmits/computes this round. Only URLLC uses
        # this "uav" mobility kind, so drone battery only ever drains for
        # URLLC traffic, matching the other two service classes never using it.
        "propulsion_power_w": 150.0,
        # UAVs roam freely in x/y and also vary altitude (see UAVFlight).
        "mobility": {"profile": "aerial", "kind": "uav", "speed_m_s": (8.0, 22.0), "pause_s": (0, 3), "altitude_range_m": (50.0, 150.0), "handover_sensitivity": 0.90, "dwell_weight": 1.8},
    },
    "smart_camera": {
        "service_class": "eMBB",
        "weight": 0.182,
        "application_type": "video_analytics",
        "power_source": "mains",
        "radio_type": "5g_embb",
        "reporting_pattern": {"period_s": (0.03, 0.2), "burst_probability": 0.40},
        # Compute-heavy real-time video analytics (deep inference per frame):
        # demand exceeds the onboard NPU's real-time budget, so it must offload.
        "workload": {"data_multiplier": 1.80, "cpu_multiplier": 16.0, "memory_per_kb": 0.32},
        # Edge-AI camera chip (onboard NPU, e.g. Ambarella/Hailo-class) --
        # needs real inference compute, more than a simple sensor/controller.
        "local_compute": {"capable": True, "cpu": 5_000, "memory": 4_096, "power_w": 6.0},
        "battery": {"capacity_j": 0.0, "initial_j": 0.0},
        "radio_energy": {"tx_power_w": 0.90, "rx_power_w": 0.25, "idle_power_w": 0.03},
        "mobility": {"profile": "fixed", "kind": "stationary", "speed_m_s": (0.0, 0.0), "pause_s": (600, 3600), "handover_sensitivity": 0.03, "dwell_weight": 0.1},
    },
    "environment_sensor": {
        "service_class": "mMTC",
        "weight": 0.083,
        "application_type": "environment_sensing",
        "power_source": "battery_or_solar",
        "radio_type": "nb_iot",
        "reporting_pattern": {"period_s": (30, 600), "burst_probability": 0.02},
        "workload": {"data_multiplier": 0.18, "cpu_multiplier": 0.25, "memory_per_kb": 0.10},
        # Low-power environmental MCU. It can attempt the canonical mMTC task;
        # its modest rate and power let the simulator measure the local-vs-MEC
        # battery trade-off instead of masking the local action completely.
        "local_compute": {"capable": True, "cpu": 400, "memory": 256, "power_w": 0.12},
        "battery": {"capacity_j": 0.77, "initial_j": 0.74},
        "radio_energy": {"tx_power_w": 0.10, "rx_power_w": 0.03, "idle_power_w": 0.002},
        "mobility": {"profile": "fixed", "kind": "stationary", "speed_m_s": (0.0, 0.0), "pause_s": (600, 3600), "handover_sensitivity": 0.02, "dwell_weight": 0.05},
    },
    "smart_meter": {
        "service_class": "mMTC",
        "weight": 0.071,
        "application_type": "utility_metering",
        "power_source": "mains_or_long_life_battery",
        "radio_type": "nb_iot",
        "reporting_pattern": {"period_s": (300, 3600), "burst_probability": 0.01},
        "workload": {"data_multiplier": 0.10, "cpu_multiplier": 0.18, "memory_per_kb": 0.08},
        "local_compute": {"capable": True, "cpu": 300, "memory": 256, "power_w": 0.08},
        "battery": {"capacity_j": 0.22, "initial_j": 0.21},
        "radio_energy": {"tx_power_w": 0.08, "rx_power_w": 0.02, "idle_power_w": 0.001},
        "mobility": {"profile": "fixed", "kind": "stationary", "speed_m_s": (0.0, 0.0), "pause_s": (600, 3600), "handover_sensitivity": 0.01, "dwell_weight": 0.05},
    },
    "smart_parking_sensor": {
        "service_class": "mMTC",
        "weight": 0.048,
        "application_type": "parking_occupancy",
        "power_source": "battery",
        "radio_type": "lora_or_nb_iot",
        "reporting_pattern": {"period_s": (60, 900), "burst_probability": 0.04},
        "workload": {"data_multiplier": 0.08, "cpu_multiplier": 0.12, "memory_per_kb": 0.06},
        "local_compute": {"capable": True, "cpu": 250, "memory": 128, "power_w": 0.06},
        "battery": {"capacity_j": 0.35, "initial_j": 0.33},
        "radio_energy": {"tx_power_w": 0.06, "rx_power_w": 0.015, "idle_power_w": 0.001},
        "mobility": {"profile": "fixed", "kind": "stationary", "speed_m_s": (0.0, 0.0), "pause_s": (600, 3600), "handover_sensitivity": 0.01, "dwell_weight": 0.05},
    },
    "street_light_controller": {
        "service_class": "mMTC",
        "weight": 0.048,
        "application_type": "smart_lighting_control",
        "power_source": "mains",
        "radio_type": "lte_m_or_nb_iot",
        "reporting_pattern": {"period_s": (10, 300), "burst_probability": 0.06},
        "workload": {"data_multiplier": 0.16, "cpu_multiplier": 0.25, "memory_per_kb": 0.10},
        # Basic MCU/PLC (on/off + dimming logic) -- weakest device on the
        # list, no real compute task beyond simple control loops.
        "local_compute": {"capable": True, "cpu": 800, "memory": 512, "power_w": 1.0},
        "battery": {"capacity_j": 0.0, "initial_j": 0.0},
        "radio_energy": {"tx_power_w": 0.12, "rx_power_w": 0.04, "idle_power_w": 0.004},
        "mobility": {"profile": "fixed", "kind": "stationary", "speed_m_s": (0.0, 0.0), "pause_s": (600, 3600), "handover_sensitivity": 0.02, "dwell_weight": 0.05},
    },
    "city_gateway": {
        "service_class": "eMBB",
        "weight": 0.030,
        "application_type": "iot_aggregation_gateway",
        "power_source": "mains",
        "radio_type": "5g_embb",
        "reporting_pattern": {"period_s": (0.2, 5.0), "burst_probability": 0.30},
        # Compute-heavy aggregation + analytics over many upstream streams:
        # exceeds the small edge computer's real-time capacity -> offloads.
        "workload": {"data_multiplier": 1.25, "cpu_multiplier": 16.0, "memory_per_kb": 0.26},
        # IoT aggregation gateway (small edge computer, e.g. industrial
        # Raspberry-Pi-class) -- aggregates many devices' data, needs more
        # headroom than a single sensor/controller.
        "local_compute": {"capable": True, "cpu": 8_000, "memory": 6_144, "power_w": 10.0},
        "battery": {"capacity_j": 0.0, "initial_j": 0.0},
        "radio_energy": {"tx_power_w": 1.00, "rx_power_w": 0.35, "idle_power_w": 0.08},
        "mobility": {"profile": "fixed", "kind": "stationary", "speed_m_s": (0.0, 0.0), "pause_s": (600, 3600), "handover_sensitivity": 0.03, "dwell_weight": 0.08},
    },
    "public_safety_wearable": {
        "service_class": "URLLC",
        "weight": 0.128,
        "application_type": "public_safety_bodycam_health",
        "power_source": "battery",
        "radio_type": "5g_urllc_embb",
        "reporting_pattern": {"period_s": (0.1, 2.0), "burst_probability": 0.35},
        "workload": {"data_multiplier": 0.25, "cpu_multiplier": 1.10, "memory_per_kb": 0.24},
        "local_compute": {"capable": False},
        "battery": {"capacity_j": 889.9, "initial_j": 794.6},
        "radio_energy": {"tx_power_w": 0.65, "rx_power_w": 0.25, "idle_power_w": 0.04},
        # Pedestrians walk along a horizontal sidewalk only (no vertical movement).
        "mobility": {"profile": "pedestrian", "kind": "pedestrian", "speed_m_s": (0.5, 2.2), "pause_s": (1, 20), "handover_sensitivity": 0.35, "dwell_weight": 1.0},
    },
}


def normalize_legacy_profile(profile):
    """Flatten nested profile sections for older simulator code paths."""
    workload = profile.get("workload", {}) or {}
    local = profile.get("local_compute", {}) or {}
    mobility = profile.get("mobility", {}) or {}
    out = dict(profile)
    out.update({
        "data_multiplier": workload.get("data_multiplier", 1.0),
        "cpu_multiplier": workload.get("cpu_multiplier", 1.0),
        "memory_per_kb": workload.get("memory_per_kb"),
        "local_compute_capable": bool(local.get("capable", False)),
        "mobility_profile": mobility.get("profile", "fixed"),
        "mobility_kind": mobility.get("kind", "stationary"),
    })
    return out


DEVICE_TYPE_PROFILES = {
    name: normalize_legacy_profile(profile)
    for name, profile in DEVICE_PROFILES.items()
}
