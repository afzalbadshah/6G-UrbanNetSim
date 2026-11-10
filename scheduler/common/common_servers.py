"""Shared server selection, admission, utilization, and service-time helpers.

Canonical processing uses cycles divided by cycles/s. Compatibility fallbacks
remain only for external task/server objects that predate the resource contract.

How to extend: add physical resource dimensions in ``entities/resources.py``
and update admission plus projected utilization together here.
"""

import math

import config as cfg
from entities.resources import bounded_utilization_ratio
from metrics.delay import congestion_multiplier, processing_delay_s

from scheduler.common.common_action import (
    ACTION_CLOUD,
    ACTION_LOCAL,
    ACTION_MEC,
    ACTION_MEC_COLLAB,
    ACTION_METRO,
    ACTION_UAV_BS,
    ACTION_SAT,
)


_KNOWN_TIERS = {"MEC", "METRO", "CLOUD", "UAV_BS", "LOCAL_DEVICE"}


def server_is_operational(server):
    """Return whether a configured compute server can currently serve work."""

    if server is None:
        return False
    if hasattr(server, "enabled") and not bool(getattr(server, "enabled")):
        return False
    if hasattr(server, "operational") and not bool(getattr(server, "operational")):
        return False
    if bool(getattr(server, "recharging", False)):
        return False
    if bool(getattr(server, "battery_backed", False)):
        if float(getattr(server, "battery_remaining_j", 0.0) or 0.0) <= 0.0:
            return False
    return True


def local_has_capacity(task):
    """Check physical onboard capability without applying a class preference."""

    if not bool(getattr(task, "local_compute_capable", False)):
        return False
    battery_capacity = float(getattr(task, "battery_capacity_j", 0.0) or 0.0)
    if battery_capacity > 0.0 and float(getattr(task, "battery_remaining_j", 0.0) or 0.0) <= 0.0:
        return False
    allocation = getattr(task, "resource_allocation", None)
    if allocation is not None:
        return (
            float(
                getattr(
                    task,
                    "local_available_compute_rate_cycles_per_s",
                    getattr(task, "local_compute_rate_cycles_per_s", 0.0),
                )
                or 0.0
            )
            >= float(allocation.allocated_compute_rate_cycles_per_s)
            and int(
                getattr(
                    task,
                    "local_available_memory_bytes",
                    getattr(task, "local_memory_capacity_bytes", 0),
                )
                or 0
            )
            >= int(allocation.memory_allocated_bytes)
        )
    return (
        float(getattr(task, "local_cpu_capacity", 0.0) or 0.0)
        >= float(getattr(task, "cpu_demand", 0.0) or 0.0)
        and float(getattr(task, "local_memory_capacity", 0.0) or 0.0)
        >= float(getattr(task, "memory_demand", 0.0) or 0.0)
    )


def tier_of(server):
    # Prefer the explicit tier set at construction (entities/server.py) --
    # robust against multi-token tier names like "UAV_BS" that a naive
    # name.split('_')[0] would mis-parse.
    tier_attr = str(getattr(server, "tier", "") or "").upper()
    if tier_attr in _KNOWN_TIERS:
        return tier_attr

    name = str(getattr(server, "name", ""))
    if name.startswith("LocalDevice"):
        return "LOCAL_DEVICE"
    if name.startswith("UAV_BS"):
        return "UAV_BS"
    if name.startswith("MEC"):
        return "MEC"
    if name.startswith("Metro"):
        return "METRO"
    if name.startswith("Cloud"):
        return "CLOUD"
    return "OTHER"


def server_utilization(server):
    if server is None:
        return {"cpu": 0.0, "storage": 0.0, "memory": 0.0, "congestion": 0.0}
    if hasattr(server, "utilization"):
        cpu, storage, memory = server.utilization()
    else:
        cpu = storage = memory = 0.0
    congestion = float(server.congestion()) if hasattr(server, "congestion") else max(cpu, memory)
    return {"cpu": float(cpu), "storage": float(storage), "memory": float(memory), "congestion": congestion}


def has_capacity(server, task, util_threshold=100.0):
    if server is None:
        return False
    try:
        resource_demand = getattr(task, "resource_demand", None)
        if resource_demand is not None and hasattr(server, "can_allocate_resources"):
            ok = server.can_allocate_resources(resource_demand)
        else:
            ok = server.can_allocate(
                task.cpu_demand,
                task.storage_demand,
                task.memory_demand,
            )
    except Exception:
        ok = False
    if not ok:
        return False
    util = projected_utilization(server, task)
    return max(util["cpu"], util["memory"]) <= float(util_threshold)


def projected_utilization(server, task):
    if server is None:
        return {"cpu": 100.0, "memory": 100.0, "storage": 100.0, "peak": 100.0}
    allocation = getattr(task, "resource_allocation", None)
    if allocation is not None and hasattr(server, "compute_rate_cycles_per_s"):
        cpu_cap = float(server.compute_rate_cycles_per_s)
        mem_cap = float(server.memory_capacity_bytes)
        sto_cap = float(server.storage_capacity_bytes)
        def used_capacity(capacity, available, name):
            capacity = float(capacity)
            available = float(available)
            tolerance = max(1e-9, abs(capacity) * 1e-12)
            if not math.isfinite(available):
                raise ValueError(f"{name} availability must be finite")
            if available < -tolerance or available > capacity + tolerance:
                raise ValueError(
                    f"{name} availability {available} is outside [0, {capacity}]"
                )
            return capacity - min(capacity, max(0.0, available))

        cpu_used = used_capacity(
            cpu_cap,
            server.available_compute_rate_cycles_per_s,
            "compute",
        )
        mem_used = used_capacity(mem_cap, server.available_memory_bytes, "memory")
        sto_used = used_capacity(sto_cap, server.available_storage_bytes, "storage")
        cpu = 100.0 * bounded_utilization_ratio(cpu_used, cpu_cap)
        mem = 100.0 * bounded_utilization_ratio(mem_used, mem_cap)
        sto = 100.0 * bounded_utilization_ratio(sto_used, sto_cap)
        def projected_percent(used, capacity):
            if capacity > 0.0:
                return 100.0 * used / capacity
            return 100.0 if used > 0.0 else 0.0

        cpu_after = projected_percent(
            cpu_used + allocation.allocated_compute_rate_cycles_per_s, cpu_cap
        )
        mem_after = projected_percent(
            mem_used + allocation.memory_allocated_bytes, mem_cap
        )
        sto_after = projected_percent(
            sto_used + allocation.storage_allocated_bytes, sto_cap
        )
    else:
        cpu_cap = max(float(getattr(server, "cpu_capacity", 0.0) or 0.0), 1e-9)
        mem_cap = max(float(getattr(server, "memory_capacity", 0.0) or 0.0), 1e-9)
        sto_cap = max(float(getattr(server, "storage_capacity", 0.0) or 0.0), 1e-9)
        cpu = 100.0 * (1.0 - float(getattr(server, "available_cpu", 0.0)) / cpu_cap)
        mem = 100.0 * (1.0 - float(getattr(server, "available_memory", 0.0)) / mem_cap)
        sto = 100.0 * (1.0 - float(getattr(server, "available_storage", 0.0)) / sto_cap)
        cpu_after = cpu + 100.0 * float(task.cpu_demand) / cpu_cap
        mem_after = mem + 100.0 * float(task.memory_demand) / mem_cap
        sto_after = sto + 100.0 * float(task.storage_demand) / sto_cap
    return {"cpu": cpu_after, "memory": mem_after, "storage": sto_after, "peak": max(cpu_after, mem_after, sto_after)}


def processing_time_ms(server, task):
    """
    Physical service time is task cycles divided by server cycles/s. Existing
    utilization then applies a bounded M/M/1-style waiting approximation. An
    idle server has no artificial congestion penalty from the arriving task.
    """
    if server is None:
        return 0.0
    min_ms = float(getattr(cfg, "MIN_PROCESSING_TIME_MS", 1.0) or 1.0)
    work_cycles = getattr(task, "compute_work_cycles", None)
    compute_rate = getattr(server, "compute_rate_cycles_per_s", None)
    if work_cycles is None or compute_rate is None or float(compute_rate) <= 0.0:
        # No cycle-rate available -> no physical processing time to compute.
        # (The legacy scale-based cpu_demand/cpu_cap fallback is removed; the
        #  single processing-delay formula lives in metrics.delay.)
        return min_ms
    # Single physical formula: work / rate, sourced from metrics.delay.
    base_ms = max(min_ms, processing_delay_s(
        compute_work_cycles=float(work_cycles),
        compute_rate_cycles_per_s=float(compute_rate),
    ) * 1000.0)
    available_rate = float(
        getattr(server, "available_compute_rate_cycles_per_s", compute_rate)
    )
    u_before = bounded_utilization_ratio(
        max(0.0, float(compute_rate) - available_rate),
        float(compute_rate),
    )
    floor = float(getattr(cfg, "CONGESTION_HEADROOM_FLOOR", 0.05) or 0.05)
    return base_ms * congestion_multiplier(utilization_ratio=u_before, headroom_floor=floor)


def service_time_ms(server, task):
    if server is None:
        return 0.0
    return float(getattr(server, "latency", 0.0) or 0.0) + processing_time_ms(server, task)


def distance_between(a, b):
    if a is None or b is None:
        return 0.0
    if not all(hasattr(obj, attr) for obj in (a, b) for attr in ("x", "y")):
        return 0.0
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def _task_access_position(task):
    """Return the task's serving-access position, with device position fallback.

    Access-only BS objects are deliberately absent from the scheduler's compute
    server list.  The simulator therefore copies the serving BS coordinates
    onto each task so server selection can still choose the geographically
    appropriate MEC or UAV-BS instead of tie-selecting the first server.
    """

    for x_name, y_name in (
        ("wireless_ap_x_m", "wireless_ap_y_m"),
        ("pos_x", "pos_y"),
    ):
        x = getattr(task, x_name, None)
        y = getattr(task, y_name, None)
        if (
            isinstance(x, (int, float))
            and not isinstance(x, bool)
            and isinstance(y, (int, float))
            and not isinstance(y, bool)
            and math.isfinite(float(x))
            and math.isfinite(float(y))
        ):
            return float(x), float(y)
    return None


def nearest_server(servers, tier, task=None, exclude_name=None):
    candidates = [s for s in servers if tier_of(s) == tier and str(getattr(s, "name", "")) != str(exclude_name)]
    if not candidates:
        return None
    origin = None
    if task is not None:
        ap_name = str(getattr(task, "wireless_ap", "") or "")
        origin = next((s for s in servers if s.name == ap_name), None)
    if origin is None:
        return min(candidates, key=lambda s: max(server_utilization(s)["cpu"], server_utilization(s)["memory"]))
    if str(tier).upper() == "METRO":
        max_dist = max(distance_between(origin, s) for s in candidates) or 1.0
        def metro_score(server):
            util = server_utilization(server)
            load = max(util["cpu"], util["memory"]) / 100.0
            distance = distance_between(origin, server) / max_dist
            return (0.70 * load) + (0.30 * distance)
        return min(candidates, key=metro_score)
    return min(candidates, key=lambda s: distance_between(origin, s))


def action_server(action, servers, task):
    action = int(action)
    ap_name = str(getattr(task, "wireless_ap", "") or "")
    if action == ACTION_LOCAL:
        return None
    if action == ACTION_MEC:
        # `wireless_ap` may now name a UAV_BS (device handed over to an
        # aerial cell) -- match by name AND tier so an ACTION_MEC choice
        # never gets silently routed to a UAV_BS just because that's what
        # the device happens to be attached to this round.
        ap_srv = next((s for s in servers if s.name == ap_name), None)
        if ap_srv is not None and str(getattr(ap_srv, "tier", "MEC")).upper() != "MEC":
            ap_srv = None
        return ap_srv or nearest_server(servers, "MEC", task)
    if action == ACTION_MEC_COLLAB:
        # Disabled for now -- no distinct cost/delay structure exists yet to
        # make collaboration a real, differentiated choice (it currently
        # priced identically to a plain MEC hop), so it's treated as
        # structurally unavailable (no server) rather than a soft
        # preference -- same category as a zero-instance tier.
        return None
    if action == ACTION_METRO:
        return nearest_server(servers, "METRO", task)
    if action == ACTION_CLOUD:
        return nearest_server(servers, "CLOUD", task)
    if action == ACTION_UAV_BS:
        # Same tier-checked preference as ACTION_MEC: use the device's
        # actual attached UAV_BS (from the handover scheduler) when it is
        # in fact one, else fall back to nearest.
        ap_srv = next((s for s in servers if s.name == ap_name), None)
        if ap_srv is not None and str(getattr(ap_srv, "tier", "UAV_BS")).upper() != "UAV_BS":
            ap_srv = None
        return ap_srv or nearest_server(servers, "UAV_BS", task)
    return None


def _zone_allows_action(task, action):
    """Physical availability of an action in the device's current zone."""
    from config import zones as zone_config
    from scheduler.common.common_action import action_to_label

    if action == ACTION_SAT and not zone_config.ZONES_ENABLED:
        return False
    zone_id = getattr(task, "zone_id", None)
    if not zone_config.ZONES_ENABLED or zone_id is None:
        return action != ACTION_SAT
    return zone_config.zone_allows(zone_id, action_to_label(action))


def allocatable_action_server(
    action,
    servers,
    task,
    *,
    reachable_destinations=None,
    require_route=False,
):
    """Resolve an operational, reachable server that can admit ``task``.

    The resolver examines every instance in a tier before ranking candidates,
    so a full nearest server does not incorrectly hide another feasible server.
    No service-class order or utilization threshold is applied.
    """

    action = int(action)
    if action in (ACTION_LOCAL, ACTION_MEC_COLLAB):
        return None
    # Satellite-IoT zones: an action exists only where its infrastructure does.
    if not _zone_allows_action(task, action):
        return None
    if action == ACTION_SAT:
        # Bent-pipe LEO path terminates at the Cloud; the satellite route
        # (satellite -> gateway -> Cloud) is declared in the topology, so no
        # terrestrial reachability from the device's ground attachment applies.
        if not bool(getattr(task, "sat_available", 0)):
            return None
        clouds = [
            server for server in servers or []
            if tier_of(server) == "CLOUD"
            and server_is_operational(server)
            and has_capacity(server, task, util_threshold=100.0)
        ]
        if not clouds:
            return None
        return min(clouds, key=lambda server: max(
            server_utilization(server)["cpu"], server_utilization(server)["memory"]
        ))
    tier_by_action = {
        ACTION_MEC: "MEC",
        ACTION_METRO: "METRO",
        ACTION_CLOUD: "CLOUD",
        ACTION_UAV_BS: "UAV_BS",
    }
    tier = tier_by_action.get(action)
    if tier is None:
        return None

    access_name = str(getattr(task, "wireless_ap", "") or "")
    reachable = set(reachable_destinations or ())
    if require_route and not access_name:
        return None

    candidates = []
    for server in servers or []:
        if tier_of(server) != tier or not server_is_operational(server):
            continue
        server_name = str(getattr(server, "name", "") or "")
        if require_route and server_name != access_name and server_name not in reachable:
            continue
        if not has_capacity(server, task, util_threshold=100.0):
            continue
        candidates.append(server)
    if not candidates:
        return None

    if action in (ACTION_MEC, ACTION_UAV_BS):
        attached = next(
            (server for server in candidates if str(getattr(server, "name", "")) == access_name),
            None,
        )
        if attached is not None:
            return attached

    origin = next(
        (server for server in (servers or []) if str(getattr(server, "name", "")) == access_name),
        None,
    )
    access_position = _task_access_position(task) if origin is None else None
    if origin is None and access_position is None:
        return min(
            candidates,
            key=lambda server: max(
                server_utilization(server)["cpu"],
                server_utilization(server)["memory"],
            ),
        )

    def distance_from_access(server):
        if origin is not None:
            return distance_between(origin, server)
        return math.hypot(
            float(getattr(server, "x", 0.0)) - access_position[0],
            float(getattr(server, "y", 0.0)) - access_position[1],
        )

    if tier == "METRO":
        max_dist = max(distance_from_access(server) for server in candidates) or 1.0

        def metro_score(server):
            utilization = server_utilization(server)
            load = max(utilization["cpu"], utilization["memory"]) / 100.0
            return (0.70 * load) + (0.30 * distance_from_access(server) / max_dist)

        return min(candidates, key=metro_score)
    return min(candidates, key=distance_from_access)
