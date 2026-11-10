"""Service zones and the LEO satellite layer (satellite-enabled IoT scenario).

The 10 km x 10 km map is split into three service zones. A zone says only which
infrastructure physically EXISTS there; it never tells the scheduler what to
choose. Inside a zone the RL policy still learns every decision from the
measured delay, energy and cost of the actions that are physically possible.

    cellular : 6G ground base stations + MEC, Metro and Cloud over fibre (the city)
    uav      : no towers; UAV-BS edge cells patrol here; LEO satellite overhead
    remote   : forest / desert; no ground infrastructure; LEO satellite only

To change the layout, edit ZONES (one place). Set ZONES_ENABLED = False to
get the original single-zone city back.

Satellite model (deliberately minimal): one serving LEO satellite is always
overhead (dense constellation), transparent/bent-pipe payload, device ->
satellite -> ground gateway -> Cloud. The serving satellite's elevation follows a
periodic pass profile, and the slant range is computed from that elevation
(spherical Earth), so propagation delay and path loss change over the run.
Orbit propagation, inter-satellite links, on-board processing and satellite
handover signalling are out of scope.

References
- 3GPP TR 38.821 (NTN solutions), Set-1 LEO-600 parameters; TR 38.811 (NTN
  channel model): elevation-dependent slant range, FSPL, scintillation,
  atmospheric and polarization losses.
- 3GPP Rel-17 IoT-NTN (NB-IoT/eMTC over NTN, S-band n256).
"""

import math

ZONES_ENABLED = True

# Rectangles in metres on the CITY_AREA frame (x_min, x_max, y_min, y_max).
# "actions" = which offloading actions are physically AVAILABLE in the zone.
# LOCAL_DEVICE is always available when the device has onboard compute.
ZONES = {
    "cellular": {
        "label": "Cellular 6G (city)",
        "extent_m": (0.0, 4000.0, 0.0, 10000.0),
        "actions": ("LOCAL_DEVICE", "MEC", "METRO", "CLOUD", "UAV_BS"),
        "color": "#dbeafe",
    },
    "uav": {
        "label": "UAV edge zone (no towers)",
        "extent_m": (4000.0, 7000.0, 0.0, 10000.0),
        "actions": ("LOCAL_DEVICE", "UAV_BS", "SAT"),
        "color": "#ede9fe",
    },
    "remote": {
        "label": "Remote (forest / desert)",
        "extent_m": (7000.0, 10000.0, 0.0, 10000.0),
        "actions": ("LOCAL_DEVICE", "SAT"),
        "color": "#dcfce7",
    },
}
ZONE_ORDER = ("cellular", "uav", "remote")
DEFAULT_ZONE = "cellular"

# Which device types spawn in each zone (weights). None -> the normal city mix
# from devices_config.DEVICE_PROFILES. Mobile devices can still drive/walk/fly
# across zone borders during the run.
ZONE_DEVICE_WEIGHTS = {
    "cellular": None,
    "uav": {
        "environment_sensor": 0.40,       # mMTC soil / weather / water sensor
        "disaster_alert_sensor": 0.20,    # URLLC flood / wildfire / landslide alert
        "camera_trap": 0.15,              # eMBB wildlife / fire camera
        "public_safety_wearable": 0.15,   # URLLC ranger / rescue team
        "drone": 0.10,                    # URLLC monitoring drone
    },
    "remote": {
        "environment_sensor": 0.45,
        "disaster_alert_sensor": 0.20,
        "camera_trap": 0.20,
        "public_safety_wearable": 0.10,
        "drone": 0.05,
    },
}
# Share of devices spawned in each zone (normalised).
ZONE_DEVICE_SHARE = {"cellular": 0.40, "uav": 0.30, "remote": 0.30}


def zone_at(x_m, y_m):
    """Return the zone id containing (x, y); the default zone if none matches."""
    if not ZONES_ENABLED:
        return DEFAULT_ZONE
    x, y = float(x_m), float(y_m)
    for zone_id in ZONE_ORDER:
        x0, x1, y0, y1 = ZONES[zone_id]["extent_m"]
        if x0 <= x <= x1 and y0 <= y <= y1:
            return zone_id
    return DEFAULT_ZONE


def zone_index(zone_id):
    return ZONE_ORDER.index(zone_id) if zone_id in ZONE_ORDER else 0


def zone_area(zone_id):
    """Zone extent in the {xmin,xmax,ymin,ymax} shape mobility code uses."""
    x0, x1, y0, y1 = ZONES[zone_id]["extent_m"]
    return {"xmin": x0, "xmax": x1, "ymin": y0, "ymax": y1}


def zone_allows(zone_id, tier_label):
    """True when ``tier_label`` (e.g. "MEC", "SAT") physically exists in the zone."""
    if not ZONES_ENABLED:
        return tier_label != "SAT"
    return tier_label in ZONES.get(zone_id, ZONES[DEFAULT_ZONE])["actions"]


# ---------------------------------------------------------------------------
# LEO satellite layer
# ---------------------------------------------------------------------------
EARTH_RADIUS_M = 6_371_000.0

SATELLITE_CONFIG = {
    "enabled": True,
    "altitude_m": 600_000.0,          # TR 38.821 Set-1 LEO-600
    "min_elevation_deg": 25.0,        # serving-satellite handover threshold
    "max_elevation_deg": 88.0,
    "pass_period_s": 480.0,           # one serving satellite pass (~8 min)
    # Ground gateway (teleport) location on the map and fibre to the Cloud DC.
    "gateway_xy_m": (9800.0, 200.0),
    "gateway_height_m": 10.0,
    "gateway_cloud_fibre_m": 300_000.0,
    # Device terminal type per service class: NB-IoT NTN (S-band) for small
    # sensor/alert traffic, Ka-band broadband terminal for eMBB imagery.
    "terminal_by_service_class": {"URLLC": "s_band", "mMTC": "s_band", "eMBB": "ka_band"},
    "access_node_by_terminal": {"s_band": "LEO_SAT_S", "ka_band": "LEO_SAT_KA"},
    # Satellite capacity is billed per byte (~1 USD/GB), far above fibre.
    "transport_price_per_byte": 1.0e-9,
}

SAT_NODE_IDS = tuple(SATELLITE_CONFIG["access_node_by_terminal"].values())
SAT_GATEWAY_ID = "SAT_GW_1"


def satellite_elevation_deg(time_ms):
    """Serving-satellite elevation at simulation time ``time_ms``.

    Periodic pass: rises from the minimum elevation, culminates, sets, and the
    next satellite of the constellation takes over at the minimum elevation.
    """
    cfg = SATELLITE_CONFIG
    period_s = max(1.0, float(cfg["pass_period_s"]))
    phase = (float(time_ms) / 1000.0 % period_s) / period_s
    e_min, e_max = float(cfg["min_elevation_deg"]), float(cfg["max_elevation_deg"])
    return e_min + (e_max - e_min) * math.sin(math.pi * phase)


def slant_range_m(elevation_deg, altitude_m=None):
    """Ground-to-satellite slant range for a spherical Earth (TR 38.811 eq.)."""
    h = float(SATELLITE_CONFIG["altitude_m"] if altitude_m is None else altitude_m)
    r = EARTH_RADIUS_M
    e = math.radians(float(elevation_deg))
    return math.sqrt((r + h) ** 2 - (r * math.cos(e)) ** 2) - r * math.sin(e)


def satellite_terminal(service_class):
    return SATELLITE_CONFIG["terminal_by_service_class"].get(str(service_class), "s_band")


def satellite_access_node(service_class):
    return SATELLITE_CONFIG["access_node_by_terminal"][satellite_terminal(service_class)]
