"""Compute-tier configuration -- the ONE place that defines every server tier.

Each tier block holds EVERYTHING about that tier in one place, so a student can
read it top-to-bottom and know the tier completely:

    * capacity  : cpu / memory / storage / bandwidth / latency
    * placement : count (num_datacenters) and optional fixed positions
    * price     : compute_price_per_cpu_s (processing) and
                  transport_price_per_byte (transmission)
    * mobility/energy extras for the aerial tier (battery, hover power)

To change what a tier costs or how big it is, edit ITS block below -- nothing
else. The cost engine (core/task_execution.py) reads the prices straight from
here, so there is a single source of truth and no duplicated price constants.

Tier ratios follow published deployment classes (Onboard ~1x, UAV_BS ~4x,
MEC ~93x, Metro ~2,100x, Cloud ~7,500x). Cost direction [R10]: hyperscale
compute is cheaper per unit than edge, while WAN/backhaul transport is the
expensive leg -- so Cloud has the LOWEST compute price but a higher transport
price, and MEC the reverse.
"""

# One shared operator input price. US commercial electricity ~0.13 USD/kWh
# = 3.6e-8 USD/J (EIA Electric Power Monthly, Table 5.6.A).
ELECTRICITY_PRICE_PER_JOULE = 3.6e-8


SERVER_CONFIG = {
    # ------------------------------------------------------------------
    # Onboard -- on-device compute (the LOCAL_DEVICE action). One per
    # task-generating device; never instantiated as a standalone Server.
    # Sized as a smartphone / vehicle-OBU / Jetson-class SoC (~1% of a MEC
    # site). It is the device's own hardware, so it is FREE to the operator:
    # no compute price, no transport hop, no cooling.
    # ------------------------------------------------------------------
    "Onboard": {
        "cpu": 3_000,
        "memory": 2_048,
        "storage": 4_000,
        "bandwidth": 0,          # executes in place -- no network hop
        "latency": 0,
        "distance": 0,
        "power_w": 3.0,          # active compute power draw (embedded SoC)
        "compute_price_per_cpu_s": 0.0,     # own hardware -> free to operator
        "transport_price_per_byte": 0.0,    # no hop
    },

    # ------------------------------------------------------------------
    # MEC -- ground edge site (10s of racks of dual-Xeon servers). Cheap,
    # low-latency radio-access hop; pricier compute per unit than cloud.
    # ------------------------------------------------------------------
    "MEC": {
        "num_datacenters": 6,
        "cpu": 280_000,
        "memory": 300_000,
        "storage": 800_000,
        "bandwidth": 1_600_000,  # ~10 Gbps fiber backhaul
        "latency": 5,
        "distance": 1_000,
        "compute_price_per_cpu_s": 2.0e-5,     # edge compute (pricier/unit)
        "transport_price_per_byte": 1.0e-11,   # cheap radio-access hop
        # Positions are assigned at runtime (evenly spread across CITY_AREA).
    },

    # ------------------------------------------------------------------
    # UAV_BS -- aerial edge (mobile drone base station with a Jetson-class
    # payload). Value is coverage/proximity, not capacity. Wireless mmWave
    # backhaul. Drains battery every round for hover.
    # ------------------------------------------------------------------
    "UAV_BS": {
        "num_datacenters": 4,
        "cpu": 12_000,
        "memory": 8_000,
        "storage": 8_000,
        "bandwidth": 12_000,     # mmWave backhaul, ~1/10 of MEC fiber
        "latency": 8,
        "distance": 1_500,
        "compute_price_per_cpu_s": 2.0e-5,     # edge-class compute
        "transport_price_per_byte": 1.0e-11,   # radio-access hop
        "altitude_range_m": (220.0, 380.0),
        "battery_capacity_j": 350_000.0,
        "battery_initial_j": 350_000.0,
        "battery_backed": True,
        "hover_power_w": 650.0,
        "patrol_speed_m_s": 14.0,
    },

    # ------------------------------------------------------------------
    # Metro -- regional aggregation datacenter (~23x a MEC site). Reached
    # over the metro backhaul (more expensive transport than the radio hop).
    # ------------------------------------------------------------------
    "Metro": {
        "num_datacenters": 0,    # live action; positions below
        "cpu": 6_400_000,
        "memory": 10_240_000,
        "storage": 40_000_000,
        "bandwidth": 16_000_000, # ~100 Gbps metro aggregation
        "latency": 50,
        "distance": 10_000,
        "compute_price_per_cpu_s": 2.0e-5,     # edge-class compute
        "transport_price_per_byte": 3.0e-11,   # metro backhaul (pricier)
        "positions": [
            [2700.0, 5000.0],    # Metro_1  West
            [7300.0, 5000.0],    # Metro_2  East
        ],
    },

    # ------------------------------------------------------------------
    # Cloud -- large regional/enterprise datacenter (~80x a MEC site).
    # Cheapest compute per unit (economies of scale) but the most expensive
    # transport (WAN/backhaul) and the highest latency.
    # ------------------------------------------------------------------
    "Cloud": {
        "num_datacenters": 1,
        "cpu": 22_400_000,
        "memory": 24_000_000,
        "storage": 64_000_000,
        "bandwidth": 128_000_000,
        "latency": 100,          # distant-cloud user RTT 50-150 ms
        "distance": 1_000_000,
        "compute_price_per_cpu_s": 1.0e-5,     # cheapest compute per unit
        "transport_price_per_byte": 3.0e-11,   # WAN/backhaul (pricier)
        "positions": [
            [1_000_000.0, 1_000_000.0],   # Cloud_1
        ],
    },
}


# Map the scheduler's tier names (LOCAL_DEVICE/MEC/UAV_BS/METRO/CLOUD) to the
# SERVER_CONFIG keys above, so the cost/energy code can look a tier up directly.
TIER_TO_CONFIG_KEY = {
    "LOCAL_DEVICE": "Onboard",
    "MEC": "MEC",
    "UAV_BS": "UAV_BS",
    "METRO": "Metro",
    "CLOUD": "Cloud",
}


def tier_prices(compute_tier):
    """Return (compute_price_per_cpu_s, transport_price_per_byte) for a tier.

    ``compute_tier`` is a scheduler tier name (e.g. "MEC", "CLOUD"). Unknown
    tiers price at zero so a missing mapping never silently inflates cost.
    """

    spec = SERVER_CONFIG.get(TIER_TO_CONFIG_KEY.get(compute_tier, ""), {})
    return (
        float(spec.get("compute_price_per_cpu_s", 0.0) or 0.0),
        float(spec.get("transport_price_per_byte", 0.0) or 0.0),
    )
