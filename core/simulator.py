# core/simulator.py
"""
SmartCityEdgeSimPy — Educational Release

Main simulator. Builds servers/devices, runs scheduler, reports metrics,
and (now) publishes MEC->Metro pairs each round for the GUI.

How to extend: compose new domain services during initialization; keep formulas
in ``metrics/`` and mutable resource behavior in the owning entity/network.
"""

import math
import os
import random
from pathlib import Path

from entities.server          import Server
from workload.generator       import WorkloadGenerator
from scheduler.base_scheduler import BaseScheduler
from core.reporter            import Reporter, TASK_FIELDS
from core.task_execution      import TaskExecutionRuntime
from mobility.manager         import MobilityManager
from mobility.handover_scheduler import candidate_access_points
from mobility.mobile_entity   import MobileEntity
from mobility.pedestrian_walk import PedestrianWalk
from mobility.uav_flight      import UAVFlight
from mobility.uav_bs_patrol   import UAVBSPatrol
from mobility.vehicle_road    import VehicleRoadMobility
from mobility.road_network    import distribute_coverage_cells
from metrics.delay            import remaining_coverage_time_s
from metrics.handover_channel import calculate_rssi_dbm

_MOBILITY_CLASS_BY_KIND = {
    "pedestrian": PedestrianWalk,
    "vehicle": VehicleRoadMobility,
    "uav": UAVFlight,
    # "stationary" intentionally omitted: no mobility model is attached.
}

from collections import Counter, defaultdict

import config as cfg
from config import CITY_AREA
from config import SERVER_CONFIG, WORKLOAD, MOBILITY_CONFIG, SIMULATION
from config.radio import get_link_profile
from config.topology          import get_topology
from config.topology import get_deployment_profile
from core.deployment_builder  import build_deployed_topology
from network                  import PhysicalTopology, Position3D
from metrics.handover_channel  import calculate_interference_power_dbm
try:
    from config import DEVICE_TYPE_PROFILES
except Exception:
    DEVICE_TYPE_PROFILES = {}

# ───────────────────────── helpers ─────────────────────────

def _euclid(ax, ay, bx, by) -> float:
    return math.hypot(float(ax) - float(bx), float(ay) - float(by))

def _entity_altitude_m(obj) -> float:
    """Height above ground (m): a UAV device or UAV_BS server keeps its
    altitude on its mobility model (.z), not on itself directly."""
    z = getattr(getattr(obj, "mobility", None), "z", None)
    return float(z) if z is not None else 0.0

def _nearest_wp_distance(x: float, y: float, servers):
    """Return the nearest configured ground or aerial wireless point.

    Candidate selection is delegated to
    ``mobility.handover_scheduler.candidate_access_points`` -- the "one
    canonical, fixed-order radio candidate list" -- instead of keeping a
    second, independent copy of the same ground/UAV/promoted-MEC filter
    here. The two copies previously drifted (a promoted MEC's
    provides_radio_access=True was recognized nowhere), which is exactly the
    duplicate-maintenance risk delegating removes.
    """
    candidates = candidate_access_points(servers)
    wireless = [(s, getattr(s, "x", None), getattr(s, "y", None)) for s in candidates]
    wireless = [(s, sx, sy) for (s, sx, sy) in wireless if sx is not None and sy is not None]
    if not wireless:
        return None, float("inf")
    s, sx, sy = min(wireless, key=lambda t: _euclid(x, y, t[1], t[2]))
    return s, _euclid(x, y, sx, sy)

def _compute_tier_counts():
    """Return the current MEC/UAV_BS site counts driving topology generation.

    Reading straight from ``SERVER_CONFIG`` means changing ``num_datacenters``
    is the only edit needed to resize a tier -- ``build_deployed_topology``
    regenerates that tier's nodes/links/routes to match automatically.
    """

    return {
        "MEC": int(SERVER_CONFIG.get("MEC", {}).get("num_datacenters", 0) or 0),
        "UAV_BS": int(SERVER_CONFIG.get("UAV_BS", {}).get("num_datacenters", 0) or 0),
    }


def _reachability_from_topology(mapping):
    """Build {source_node_id: set(reachable destination_node_id)} from the
    deployed topology's declared routes. Used to gate compute-tier choices to
    destinations the packet engine can actually route to."""
    reach = {}
    try:
        for route in (mapping or {}).get("routes", []) or []:
            src = str(route.get("source_node_id", ""))
            dst = str(route.get("destination_node_id", ""))
            if src and dst:
                reach.setdefault(src, set()).add(dst)
    except Exception:
        return {}
    return reach


def _install_reachability(scheduler, reachability):
    """Hand the reachability map to a scheduler's state builder if it has one."""
    builder = getattr(scheduler, "state_builder", None)
    if builder is not None and hasattr(builder, "set_reachability"):
        builder.set_reachability(reachability)


def _clamp_city(x, y):
    return (
        min(max(float(x), float(CITY_AREA["xmin"])), float(CITY_AREA["xmax"])),
        min(max(float(y), float(CITY_AREA["ymin"])), float(CITY_AREA["ymax"])),
    )

def _choose_device_profile():
    profiles = DEVICE_TYPE_PROFILES or {"generic": {"weight": 1.0}}
    names = list(profiles.keys())
    weights = [float((profiles.get(name, {}) or {}).get("weight", 1.0) or 0.0) for name in names]
    if sum(weights) <= 0.0:
        weights = [1.0] * len(names)
    name = random.choices(names, weights=weights, k=1)[0]
    return name, profiles.get(name, {}) or {}

# Device types that must spawn at a real road-grid junction rather than
# anywhere in the city (a traffic light only ever sits at an intersection).
JUNCTION_ONLY_DEVICE_TYPES = {"traffic_controller"}

def _random_junction_point():
    """A random road-grid intersection (x, y) inside CITY_AREA -- every
    (v_road, h_road) pair is a real junction (build_road_grid already snaps
    the outermost line on each axis to the city border), so any combination
    is valid."""
    from mobility.road_network import build_road_grid
    h_roads, v_roads = build_road_grid(CITY_AREA)
    return random.choice(v_roads), random.choice(h_roads)

def _mobility_speed_range(profile):
    mobility = (profile or {}).get("mobility", {}) or {}
    return tuple(mobility.get("speed_m_s", (0.0, 0.0)))

def _mobility_pause_range(profile):
    mobility = (profile or {}).get("mobility", {}) or {}
    return tuple(mobility.get("pause_s", (0.0, 0.0)))

def _estimate_dwell_ms(distance_to_bs, speed_m_s, sensitivity=0.0, dwell_weight=1.0):
    mec_radius = max(1.0, float(SERVER_CONFIG.get("MEC", {}).get("distance", 1000.0) or 1000.0))
    remaining_m = max(0.0, mec_radius - float(distance_to_bs or 0.0))
    speed = max(0.1, float(speed_m_s or 0.0))
    dwell_ms = (
        remaining_coverage_time_s(remaining_distance_m=remaining_m, relative_speed_m_s=speed)
        * 1000.0
        * max(0.1, float(dwell_weight or 1.0))
    )
    risk = 1.0 - min(1.0, remaining_m / mec_radius)
    risk = min(1.0, max(0.0, risk * (1.0 + float(sensitivity or 0.0))))
    return dwell_ms, risk

def place_servers_in_city(servers):
    """
    Assign x,y to MEC_ / Metro_ / Cloud_ in a simple grid inside CITY_AREA
    if they don't already have explicit positions from SERVER_CONFIG.
    """
    map_servers = [s for s in servers
                   if (s.name.startswith("MEC_") or s.name.startswith("Metro_")
                       or s.name.startswith("Cloud_") or s.name.startswith("UAV_BS_"))]
    unplaced = [s for s in map_servers if not (hasattr(s, "x") and hasattr(s, "y"))]
    if not unplaced:
        return

    xmin, xmax = CITY_AREA["xmin"], CITY_AREA["xmax"]
    ymin, ymax = CITY_AREA["ymin"], CITY_AREA["ymax"]

    n = len(unplaced)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    xs = [xmin + (i + 0.5) * (xmax - xmin) / cols for i in range(cols)]
    ys = [ymin + (j + 0.5) * (ymax - ymin) / rows for j in range(rows)]

    k = 0
    for j in range(rows):
        for i in range(cols):
            if k >= n:
                break
            s = unplaced[k]
            s.x, s.y = xs[i], ys[j]
            k += 1



class Simulator:
    """
    Orchestrates simulation rounds and integrates a scheduler.
    Publishes gui_links each round for the LiveMap (MEC->Metro pairs).
    """

    def __init__(self, scheduler: BaseScheduler, output_dir="output", workload=None,
                 deployment_profile=None):
        self.scheduler = scheduler
        self.stop_requested = False
        self.current_time = 0
        self.servers = []
        self.access_points = []
        self.tasks = []
        # Active workload settings (device sweep + task sizing), wired down
        # from the launcher; defaults to the shared config.WORKLOAD profile.
        self.workload = workload if workload is not None else WORKLOAD
        self.gui_links = []  # [{'MEC': 'MEC_1','Metro':'Metro_1','count':N}, ...]
        self.live_processing_server_by_entity = {}  # entity_id -> server currently processing its task
        self.live_local_cpu_util_by_entity = {}  # entity_id -> fraction of local_cpu_capacity used, this round's on-device tasks only
        self.live_server_cpu_util_by_name = {}  # server_name -> this round task-demand/capacity fraction
        self.live_active_device_count = int(self.workload.get("start_devices", 0) or 0)
        topology_profile = str(SIMULATION.get("topology_profile", "smartcity_default"))
        deployment = get_deployment_profile(
            deployment_profile or SIMULATION.get("deployment_profile")
        )
        deployed_topology = build_deployed_topology(
            topology_profile,
            get_topology(topology_profile),
            deployment,
            compute_tier_counts=_compute_tier_counts(),
        )
        self.physical_topology = PhysicalTopology.from_mapping(
            deployed_topology.profile_id,
            deployed_topology.mapping,
        )
        self.access_points = list(deployed_topology.access_points)
        self.deployment_profile_id = deployment.profile_id
        self.access_generation = deployment.access_generation
        self.ground_access_band_name = str(
            get_link_profile(deployment.ground_bs_profile_id)["band"]
        )
        # Reachability map {source_node: set(destinations)} from the deployed
        # topology's declared routes, so the scheduler only offers compute tiers
        # the packet engine can actually route to from a device's radio
        # attachment (e.g. MEC is reachable from a ground BS but not from a UAV).
        self.reachability = _reachability_from_topology(deployed_topology.mapping)
        _install_reachability(self.scheduler, self.reachability)
        self.generator = WorkloadGenerator(
            self.workload,
            access_generation=self.access_generation,
        )
        self.task_execution_runtime = TaskExecutionRuntime(
            self.physical_topology,
            experiment_seed=getattr(self.scheduler, "_seed", 0),
        )
        self.realized_execution_by_task_id = {}
        self.output_dir = Path(output_dir).resolve()
        model_name = str(getattr(self.scheduler, "model_name", "") or "common").strip().lower()
        # Mirrors DriveNetSim's layout: each run phase (exploration/evaluation,
        # i.e. whatever `output_dir` points at) gets its own "offloading" and
        # "mobility" subtrees, each with one folder per model -- so the main
        # scheduler's CSVs and the handover scheduler's CSVs/figures never mix
        # in the same folder the way they used to when both pointed at the
        # same model_dir.
        self.offloading_dir = self.output_dir / "offloading" / model_name
        self.mobility_dir = self.output_dir / "mobility" / model_name
        os.makedirs(self.offloading_dir, exist_ok=True)
        os.makedirs(self.mobility_dir, exist_ok=True)
        self.model_dir = self.offloading_dir  # back-compat alias -- offloading output, same as before
        self.task_reporter = Reporter(
            TASK_FIELDS,
            str(self.offloading_dir / "server_results.csv"),
            print_console=bool(SIMULATION.get("print_metrics", False)),
        )
        if hasattr(self.scheduler, "set_output_dirs"):
            self.scheduler.set_output_dirs(self.offloading_dir, self.mobility_dir)

        # Build servers and attach configured coordinates
        self._initialize_servers()
        place_servers_in_city(self.servers)

        # ------------------------- DEVICE ENTITIES SETUP -------------------------
        device_entities = []

        # Devices spawn at genuinely random locations spread across the
        # whole city, not clustered tightly around a handful of MEC anchor
        # points. Vehicles/pedestrians still snap onto their nearest
        # road/block right after (see mobility.on_spawn/_start_move below),
        # so a plain uniform point over the city area is all that's needed
        # here -- each device kind's own mobility class handles the rest.
        MECs = [s for s in self.servers if str(getattr(s, "name", "")).startswith("MEC_")]
        base_list = self.access_points or MECs or self.servers
        n_base = len(base_list) if base_list else 1

        for i in range(self.workload["max_devices"]):
            # Profile is chosen before position: a handful of device types
            # (traffic_controller) are constrained to spawn at a real road
            # junction instead of anywhere in the city. This only changes
            # *where* that device lands, not whether it gets chosen --
            # _choose_device_profile()'s weighted random.choices call (and
            # so the overall per-type device ratio) is untouched.
            device_type, profile = _choose_device_profile()

            if device_type in JUNCTION_ONLY_DEVICE_TYPES:
                init_x, init_y = _random_junction_point()
            else:
                init_x = random.uniform(CITY_AREA["xmin"], CITY_AREA["xmax"])
                init_y = random.uniform(CITY_AREA["ymin"], CITY_AREA["ymax"])
            init_x, init_y = _clamp_city(init_x, init_y)

            ent = MobileEntity(
                entity_id=i,
                init_pos=[init_x, init_y],
                speed=0.0,
            )

            # Seed the initial serving cell to the nearest configured wireless
            # point (an access-only BS, MEC, or UAV-BS).
            serving0, _ = _nearest_wp_distance(
                init_x,
                init_y,
                self.servers + self.access_points,
            )
            ent.attached_server = serving0 if serving0 is not None else base_list[i % n_base]

            ent.apply_device_profile(device_type, profile)
            mobility_kind = ent.mobility_kind

            # Every device belongs to exactly one mobility kinematics: a
            # stationary sensor/gateway (no model), a horizontal-only
            # pedestrian walk, a Manhattan-grid road vehicle, or a free-roam
            # (+altitude) UAV flight. See mobility/{pedestrian_walk,
            # vehicle_road,uav_flight}.py.
            mobility_cls = _MOBILITY_CLASS_BY_KIND.get(mobility_kind)
            if mobility_cls is not None:
                kwargs = {}
                if mobility_kind == "uav":
                    kwargs["altitude_range"] = getattr(ent, "altitude_range_m", (50.0, 150.0)) or (50.0, 150.0)
                ent.mobility = mobility_cls(
                    MOBILITY_CONFIG.get("area", CITY_AREA),
                    _mobility_speed_range(profile),
                    _mobility_pause_range(profile),
                    **kwargs,
                )
                ent.mobility.pos = (init_x, init_y)
                if hasattr(ent.mobility, "on_spawn"):
                    ent.mobility.on_spawn(init_x, init_y)
                ent.mobility._start_move()
            device_entities.append(ent)

        # Mobility manager gets both servers and devices
        # Mobility and handover see access-only BSs as well as compute-backed
        # radio points.  The offloading scheduler still receives only compute
        # servers, so access density cannot silently change MEC capacity.
        self.mobility = MobilityManager(
            self.servers + self.access_points,
            device_entities,
            deployment_profile_id=self.deployment_profile_id,
        )
        self.mobility.out_dir = str(self.mobility_dir)
        self.scheduler.mobility = self.mobility
        self._scheduler_execution_hook_enabled = hasattr(
            self.scheduler, "set_execution_runtime"
        )
        if not self._scheduler_execution_hook_enabled:
            raise TypeError("scheduler must support realized task execution")
        self.scheduler.set_execution_runtime(self._execute_scheduled_task)

    def request_stop(self):
        self.stop_requested = True

    def activate_deployment_profile(self, profile_name: str) -> None:
        """Activate a declared access-density profile before a run starts.

        The launcher constructs the simulator before opening the Live Map.  This
        seam lets the interactive UI opt into access-only BSs without changing
        the legacy headless/regression default.  It is intentionally rejected
        after simulation time has advanced because replacing the topology would
        invalidate queues, routes, and already-realized measurements.
        """
        if self.current_time != 0 or self.realized_execution_by_task_id:
            raise RuntimeError("deployment profile must be selected before simulation starts")

        deployment = get_deployment_profile(profile_name)
        if deployment.profile_id == getattr(self, "deployment_profile_id", None):
            return

        topology_profile = str(SIMULATION.get("topology_profile", "smartcity_default"))
        deployed_topology = build_deployed_topology(
            topology_profile,
            get_topology(topology_profile),
            deployment,
            compute_tier_counts=_compute_tier_counts(),
        )
        self.physical_topology = PhysicalTopology.from_mapping(
            deployed_topology.profile_id,
            deployed_topology.mapping,
        )
        self.access_points = list(deployed_topology.access_points)
        self.deployment_profile_id = deployment.profile_id
        self.access_generation = deployment.access_generation
        self.ground_access_band_name = str(
            get_link_profile(deployment.ground_bs_profile_id)["band"]
        )
        self.reachability = _reachability_from_topology(deployed_topology.mapping)
        _install_reachability(self.scheduler, self.reachability)
        self.generator.set_access_generation(self.access_generation)
        self.task_execution_runtime = TaskExecutionRuntime(
            self.physical_topology,
            experiment_seed=getattr(self.scheduler, "_seed", 0),
        )
        self.realized_execution_by_task_id = {}

        # Rebind mobility/handover to the same entities and the new radio set.
        entities = list(getattr(getattr(self, "mobility", None), "entities", []) or [])
        self.mobility = MobilityManager(
            self.servers + self.access_points,
            entities,
            deployment_profile_id=self.deployment_profile_id,
        )
        self.mobility.out_dir = str(self.mobility_dir)
        self.scheduler.mobility = self.mobility
        if self._scheduler_execution_hook_enabled:
            self.scheduler.set_execution_runtime(self._execute_scheduled_task)

        for entity in entities:
            x, y = entity.get_position()
            serving, _ = _nearest_wp_distance(x, y, self.servers + self.access_points)
            if serving is not None:
                entity.attached_server = serving

    # ───────────────────────────── Setup helpers ─────────────────────────────

    def _initialize_servers(self):
        """Build Server objects from SERVER_CONFIG and attach x,y if provided."""
        self.servers = []
        for tier, spec in SERVER_CONFIG.items():
            if tier == "Onboard":
                # Universal per-device compute, not a real server tier --
                # MobileEntity/Task read SERVER_CONFIG["Onboard"]'s values
                # directly at spawn time (see apply_device_profile()/
                # workload/generator.py) for whichever devices happen to
                # roll a local-compute-capable profile at runtime. There is
                # no fixed count to instantiate here, so this tier is
                # skipped outright rather than trusting a "num_datacenters"
                # key that may or may not be present on its spec.
                continue
            count = int(spec.get("num_datacenters", 1))
            area = MOBILITY_CONFIG.get("area", CITY_AREA)
            uav_cells = None
            if tier == "UAV_BS":
                # Each UAV-BS patrols a loop around its own coverage cell --
                # the same cell layout MEC's topology nodes use (see
                # core.deployment_builder._regenerate_mec_sites), so ground and
                # aerial base stations divide the city the same coverage-
                # maximizing way for any count (see UAVBSPatrol for the
                # overlap at shared cell edges).
                uav_cells = distribute_coverage_cells(count, area)
            for i in range(count):
                srv = Server(
                    name=f"{tier}_{i+1}",
                    index=i,
                    cpu_capacity=spec["cpu"],
                    memory_capacity=spec["memory"],
                    storage_capacity=spec["storage"],
                    bandwidth=spec["bandwidth"],
                    latency=spec["latency"],
                    tier=tier,
                )
                srv.available_cpu = srv.cpu_capacity
                srv.available_memory = srv.memory_capacity
                srv.available_storage = srv.storage_capacity
                try:
                    physical_node = self.physical_topology.node(srv.name)
                except KeyError as exc:
                    raise ValueError(
                        f"configured runtime server {srv.name!r} is missing from "
                        f"topology {self.physical_topology.profile_id!r}"
                    ) from exc
                # The topology node is the physical authority for position:
                # for MEC/Metro/Cloud it's this tier's coverage-optimized site
                # (core.deployment_builder regenerates it from num_datacenters);
                # for UAV_BS it's overwritten below by the live patrol position.
                srv.x = physical_node.position.x_m
                srv.y = physical_node.position.y_m
                srv.z = physical_node.position.z_m
                # Every radio-capable runtime server carries the topology's
                # explicit link profile. RSSI/SINR code must resolve this
                # profile and never fall back to a process-wide band.
                srv.profile_id = physical_node.profile_id
                srv.anchor_profile_id = physical_node.anchor_profile_id
                # A promoted MEC's topology node carries provides_radio_access
                # =True (see core.deployment_builder); Metro/Cloud/UAV_BS and
                # independently-placed MEC nodes don't carry the field at all,
                # so default False. Consumed by
                # mobility.handover_scheduler.candidate_access_points() to
                # decide whether this server is a genuine radio candidate.
                srv.provides_radio_access = bool(
                    getattr(physical_node, "provides_radio_access", False)
                )

                # UAV-BS is an aerial, always-mobile edge tier: it patrols a
                # loop around its assigned coverage cell (see UAVBSPatrol),
                # overlapping slightly with neighboring UAV-BS cells so a
                # device near a shared border can reach either one, and
                # moves regardless of the generic MOBILITY_CONFIG.apply_to_servers
                # toggle (which is meant for static-by-default tiers like
                # MEC/Metro/Cloud).
                if tier == "UAV_BS":
                    if uav_cells and i < len(uav_cells):
                        cell_bounds = uav_cells[i]["bounds"]
                    else:
                        cell_bounds = (area["xmin"], area["xmax"], area["ymin"], area["ymax"])
                    srv.mobility = UAVBSPatrol(
                        area=area,
                        bounds=cell_bounds,
                        speed_m_s=spec.get("patrol_speed_m_s", 14.0),
                        altitude_range=spec.get("altitude_range_m", (220.0, 380.0)),
                        start_waypoint_index=(2 if i % 2 == 1 else 0),
                    )
                    srv.x, srv.y = srv.mobility.pos
                    srv.force_mobile = True

                print(f"Initialized {srv.name} at ({getattr(srv,'x',None)}, {getattr(srv,'y',None)})")
                self.servers.append(srv)

    def _live_position_overrides(self):
        overrides = {}
        for server in self.servers:
            if str(getattr(server, "tier", "")).upper() != "UAV_BS":
                continue
            overrides[server.name] = Position3D(
                float(server.x),
                float(server.y),
                _entity_altitude_m(server),
            )
        return overrides

    def _execute_scheduled_task(self, *, task, server, current_time, action, context):
        """Run one chosen action before the scheduler learns or reports it."""

        # Processing time is computed HERE, in the matrix (common_servers), not
        # read from the state builder. service_time_ms = server.latency +
        # work_cycles / server_rate, with a bounded congestion factor. This keeps
        # common_state formula-free and makes on-device compute cost real time
        # (a heavy task on a small device is slow, so it is learned-avoided).
        from scheduler.common.common_servers import service_time_ms as _service_time_ms
        service_ms = max(0.0, float(_service_time_ms(server, task)))
        is_local = str(getattr(server, "name", "")).startswith("LocalDevice")
        processing_ms = (
            service_ms
            if is_local
            else max(0.0, service_ms - float(getattr(server, "latency", 0.0) or 0.0))
        )
        if is_local:
            capacity = max(1e-9, float(getattr(task, "local_cpu_capacity", 0.0) or 0.0))
            utilization_before = 0.0
            utilization_after = min(
                1.0, max(0.0, float(getattr(task, "cpu_demand", 0.0) or 0.0) / capacity)
            )
        else:
            capacity = max(1e-9, float(getattr(server, "cpu_capacity", 0.0) or 0.0))
            available = max(0.0, float(getattr(server, "available_cpu", 0.0) or 0.0))
            utilization_before = min(1.0, max(0.0, 1.0 - available / capacity))
            utilization_after = min(
                1.0,
                utilization_before
                + max(0.0, float(getattr(task, "cpu_demand", 0.0) or 0.0)) / capacity,
            )
        entity_index = int(getattr(task, "entity_id", getattr(task, "device_id", 0)) or 0)
        device = self.mobility.entities[entity_index]
        x, y = device.get_position()
        source_position = Position3D(float(x), float(y), _entity_altitude_m(device))
        access_node_id = None if is_local else str(getattr(task, "wireless_ap", "") or "")
        result = self.task_execution_runtime.execute(
            task,
            server,
            current_time_ms=float(current_time),
            processing_duration_ms=processing_ms,
            utilization_before_ratio=utilization_before,
            utilization_after_ratio=utilization_after,
            source_position=source_position,
            access_node_id=access_node_id,
            position_overrides=self._live_position_overrides(),
        )
        self.realized_execution_by_task_id[str(task.id)] = result
        return result

    # ────────────────────────── GUI links helper ──────────────────────────

    def _update_gui_links(self, assignments):
        """
        Build (MEC|UAV_BS)->(Metro|Cloud) pairs for GUI.
        Publishes avg_delay_ms per link and distinct-device counts per tier.
        """
        pairs = Counter()
        dev_ids_by_tier = defaultdict(set)
        link_delay_sum = defaultdict(float)   # (MEC, dest, tier) -> ms
        link_delay_cnt = Counter()            # (MEC, dest, tier) -> n
        link_kb_sum = defaultdict(float)      # (MEC, dest, tier) -> kb transferred this round

        for task, srv in assignments:
            sname  = str(getattr(srv, "name", ""))
            ent_id = getattr(task, "entity_id", -1)

            if sname.startswith("Metro"):
                tier = "Metro"
            elif sname.startswith("Cloud"):
                tier = "Cloud"
            elif sname.startswith("UAV_BS"):
                # aerial edge tier: local service, no backhaul arrow, count device
                dev_ids_by_tier["UAV_BS"].add(ent_id)
                continue
            elif sname.startswith("MEC"):
                # local service at MEC -> no backhaul arrow, but count device
                dev_ids_by_tier["MEC"].add(ent_id)
                continue
            else:
                continue

            # Serving wireless AP -- a ground MEC or an aerial UAV_BS, both of
            # which can have their workload overflow to Metro/Cloud when
            # over capacity. Previously this only recognized an "MEC_"
            # prefix, so a UAV_BS-attached device's overflow to Metro/Cloud
            # never got a backhaul arrow, device tally, or delay stats --
            # the overflow happened in the scheduler, it just never got
            # drawn.
            serving_ap = str(getattr(task, "wireless_ap", "") or "")
            if not serving_ap and hasattr(self, "mobility"):
                try:
                    dev = self.mobility.entities[ent_id]
                    ap  = getattr(dev, "attached_server", None)
                    serving_ap = str(getattr(ap, "name", "") or "")
                except Exception:
                    serving_ap = ""

            if serving_ap.startswith("MEC") or serving_ap.startswith("UAV_BS"):
                key = (serving_ap, sname, tier)
                pairs[key] += 1
                dev_ids_by_tier[tier].add(ent_id)

                if getattr(task, "measurement_basis", "") != "realized_execution":
                    raise RuntimeError("link delay requires realized packet execution")
                link_delay_sum[key] += float(getattr(task, "network_delay_ms", 0.0) or 0.0)
                link_delay_cnt[key] += 1
                link_kb_sum[key] += float(getattr(task, "data_size_kb", 0.0) or 0.0)

        # publish links (with avg_delay_ms). total_kb is the actual data
        # volume carried over this link this round -- the GUI sizes the
        # backhaul arrow's width off this, not off task count, so a link
        # carrying a few large payloads reads as "busier" than one carrying
        # many tiny ones.
        self.gui_links = []
        for (e, d, t), c in pairs.items():
            n = link_delay_cnt[(e, d, t)]
            avg_ms = (link_delay_sum[(e, d, t)] / n) if n else 0.0
            self.gui_links.append({
                "MEC": e, "to": d, "tier": t, "count": c,
                "avg_delay_ms": round(avg_ms, 1),
                "total_kb": round(link_kb_sum[(e, d, t)], 2),
            })

        # legend counts
        self.gui_tier_counts = {
            "MEC":   len(dev_ids_by_tier["MEC"]),
            "Metro": len(dev_ids_by_tier["Metro"]),
            "Cloud": len(dev_ids_by_tier["Cloud"]),
            "UAV_BS": len(dev_ids_by_tier["UAV_BS"]),
        }




    # ───────────────────────────────── Main loop ─────────────────────────────────
    def run(self, train_mode=None, progress_callback=None):
        """
        Each round:
          1) Update mobility (handovers, etc.)
          2) Release completed tasks
          3) Generate tasks and attach mobility context
          4) Ask scheduler → (task, server) pairs
          5) Apply allocations
          6) Report metrics
          7) Advance clock (included via HO delays)
        """
        total_rounds = cfg.total_rounds(self.workload)

        if train_mode is not None and hasattr(self.scheduler, "set_train_mode"):
            self.scheduler.set_train_mode(bool(train_mode))
        if train_mode is not None and hasattr(self.mobility, "set_train_mode"):
            self.mobility.set_train_mode(bool(train_mode))
        self._mobility_train_mode = bool(train_mode) if train_mode is not None else False
        handover_scheduler = getattr(self.mobility, "handover_scheduler", None)
        if not self._mobility_train_mode and handover_scheduler is not None and hasattr(handover_scheduler, "load"):
            # Mirrors create_comparison_scheduler()'s load-before-eval pattern
            # for the main offloading scheduler: an evaluation run must use
            # the actor-critic weights trained on a prior run, not a fresh
            # randomly-initialized network.
            try:
                handover_scheduler.load()
            except Exception as exc:
                print(f"[LOAD] No trained handover model loaded: {exc}")

        round_duration_ms = float(MOBILITY_CONFIG.get("time_step_ms", 1000.0) or 1000.0)

        def _active_device_count(round_id):
            count = int(self.workload["start_devices"]) + ((int(round_id) - 1) * int(self.workload["increment"]))
            return min(int(self.workload["max_devices"]), max(0, count))

        for round_no in range(1, total_rounds + 1):
            if self.stop_requested:
                break
            if progress_callback is not None:
                progress_callback(round_id=round_no, total_rounds=total_rounds, event="round_start")
            # 1) Mobility update
            active_count = _active_device_count(round_no)
            self.live_active_device_count = int(active_count)
            mobility_active_count = active_count if bool(getattr(cfg, "ACTIVE_DEVICES_ONLY", False)) else None
            ho_count, ho_delay = self.mobility.update_all(active_count=mobility_active_count)
            # Handover interruptions occur independently per device. Summing them
            # into the single global clock serialized parallel radio events and
            # inflated every task's creation time. The realized execution service
            # now applies each device's own interruption on its own timeline.
            mob_metrics = self.mobility.get_metrics()

            # 2) Release resources from finished tasks
            for srv in self.servers:
                srv.release_completed_tasks(self.current_time)

            # 3) Generate tasks and tag with mobility context
            self.tasks = self.generator.generate_tasks(round_no, entities=self.mobility.entities)
            num_devices = len(self.tasks)

            # Co-channel cells that transmit on the access band (ground MEC BSs
            # and aerial UAV_BS cells). Their aggregate power at the device is
            # the interference term that turns SNR into SINR on the access hop.
            interference_cells = [
                s for s in (self.servers + self.access_points)
                if str(getattr(s, "tier", "")).upper() in ("MEC", "UAV_BS")
                and hasattr(s, "x") and hasattr(s, "y")
            ]

            for idx, task in enumerate(self.tasks):
                task.entity_id = idx
                if hasattr(task, "set_round"):
                    task.set_round(round_no)

                dev = self.mobility.entities[idx]
                x, y = dev.get_position()

                serving_bs = getattr(dev, "attached_server", None)
                if serving_bs is None or not (hasattr(serving_bs, "x") and hasattr(serving_bs, "y")):
                    serving_bs, _ = _nearest_wp_distance(
                        x,
                        y,
                        self.servers + self.access_points,
                    )

                if serving_bs is not None and hasattr(serving_bs, "x") and hasattr(serving_bs, "y"):
                    # A serving UAV_BS (or an airborne task-generating UAV)
                    # is not at ground level -- use the true 3D distance so
                    # its altitude isn't dropped from the link budget.
                    dev_z = _entity_altitude_m(dev)
                    bs_is_airborne = str(getattr(serving_bs, "tier", "")).upper() == "UAV_BS"
                    if dev_z > 0.0 or bs_is_airborne:
                        bs_z = _entity_altitude_m(serving_bs)
                        dist_bs = math.dist((x, y, dev_z), (serving_bs.x, serving_bs.y, bs_z))
                    else:
                        dist_bs = _euclid(x, y, serving_bs.x, serving_bs.y)
                else:
                    dist_bs = float(SERVER_CONFIG.get("MEC", {}).get("distance", 2000.0))

                # UAV_BS is a direct aerial radio link (not routed via the
                # ground MEC), and the UAV moves every round, so recompute
                # the device's live distance to the nearest one each round.
                uav_servers = [s for s in self.servers if str(getattr(s, "tier", "")) == "UAV_BS"]
                if uav_servers:
                    dev_z = _entity_altitude_m(dev)
                    dist_uav_bs = min(math.dist((x, y, dev_z), (s.x, s.y, _entity_altitude_m(s))) for s in uav_servers)
                else:
                    dist_uav_bs = float(SERVER_CONFIG.get("UAV_BS", {}).get("distance", 1500.0))

                # RF/mobility decision-quality signals are computed once per
                # device per round by mobility.handover_scheduler (see
                # mobility/handover_scheduler.py::select_server). Fall back to
                # the simple proxy only if that hasn't run yet (e.g. no MECs).
                sig_db = getattr(dev, "chosen_rssi_dbm", None)
                if sig_db is None:
                    try:
                        sig_db = (
                            dev.signal_strength(serving_bs)
                            if serving_bs is not None
                            else calculate_rssi_dbm(
                                dist_bs,
                                band_name=self.ground_access_band_name,
                            )
                        )
                    except Exception:
                        sig_db = calculate_rssi_dbm(
                            dist_bs,
                            band_name=self.ground_access_band_name,
                        )

                ho_ms = float(getattr(dev, "last_handover_ms", 0.0))
                speed_m_s = float(getattr(dev, "movement_speed_m_s", getattr(dev, "speed", 0.0)) or 0.0)

                dwell_ms = getattr(dev, "predicted_dwell_time_ms", None)
                remaining_dwell_ms = getattr(dev, "remaining_dwell_time_ms", None)
                if dwell_ms is None or remaining_dwell_ms is None:
                    dwell_ms, handover_risk = _estimate_dwell_ms(
                        dist_bs, speed_m_s,
                        getattr(dev, "handover_sensitivity", 0.0),
                        getattr(dev, "dwell_weight", 1.0),
                    )
                    remaining_dwell_ms = dwell_ms
                else:
                    mec_radius = max(1.0, float(SERVER_CONFIG.get("MEC", {}).get("distance", 1000.0) or 1000.0))
                    handover_risk = min(1.0, max(0.0, 1.0 - min(1.0, remaining_dwell_ms / max(mec_radius, 1.0))))

                task.wireless_ap = getattr(serving_bs, "name", "")
                # The offloading scheduler receives compute servers only, not
                # access-only BS objects.  Preserve the serving point's live
                # coordinates on the task so MEC/UAV candidate selection can
                # use the real access origin instead of defaulting to the first
                # equally-loaded server in a tier.
                task.wireless_ap_x_m = (
                    float(serving_bs.x) if serving_bs is not None and hasattr(serving_bs, "x") else None
                )
                task.wireless_ap_y_m = (
                    float(serving_bs.y) if serving_bs is not None and hasattr(serving_bs, "y") else None
                )
                task.set_mobility(
                    x=x,
                    y=y,
                    distance_to_bs=dist_bs,
                    signal_dbm=sig_db,
                    handover_ms=ho_ms,
                    speed_m_s=speed_m_s,
                    predicted_dwell_time_ms=dwell_ms,
                    remaining_dwell_time_ms=remaining_dwell_ms,
                    handover_risk=handover_risk,
                )
                task.distance_to_uav_bs = dist_uav_bs
                task.true_dwell_time_ms = getattr(dev, "true_dwell_time_ms", dwell_ms)
                task.oracle_bs = getattr(dev, "oracle_bs", "")
                task.oracle_rssi_dbm = getattr(dev, "oracle_rssi_dbm", "")
                task.rssi_regret_db = getattr(dev, "rssi_regret_db", "")
                task.snr_db = getattr(dev, "snr_db", "")
                task.sinr_db = getattr(dev, "sinr_db", "")
                task.noise_dbm = getattr(dev, "noise_dbm", "")
                # The handover scheduler sets a numeric interference value
                # during mobility runs; in offloading runs it is unset, so
                # compute it here from the co-channel cells so that realized
                # SINR (not just SNR) drives the access-hop transmission delay.
                dev_interference_dbm = getattr(dev, "interference_dbm", "")
                if isinstance(dev_interference_dbm, (int, float)):
                    task.interference_dbm = dev_interference_dbm
                else:
                    task.interference_dbm = calculate_interference_power_dbm(
                        (x, y),
                        serving_bs,
                        interference_cells,
                        band_name=str(
                            get_link_profile(serving_bs.profile_id)["band"]
                        ),
                    )
                task.handover_executed = getattr(dev, "handover_executed", 0)
                task.handover_failure_flag = getattr(dev, "handover_failure_flag", 0)
                task.handover_interruption_ms = getattr(dev, "handover_interruption_ms", 0.0)
                task.handover_trigger_reason = getattr(dev, "handover_trigger_reason", "")
                task.pingpong_flag = getattr(dev, "pingpong_flag", 0)
                task.bs_residence_time_ms = getattr(dev, "bs_residence_time_ms", "")
                task.coverage_outage_flag = getattr(dev, "coverage_outage_flag", 0)
                task.coverage_outage_duration_ms = getattr(dev, "coverage_outage_duration_ms", "")
                task.prev_attached_bs = getattr(dev, "prev_attached_bs", "")
                task.handover_q_margin = getattr(dev, "handover_q_margin", "")
                task.handover_prev_rssi_dbm = getattr(dev, "handover_prev_rssi_dbm", "")
                task.handover_target_bs = getattr(dev, "handover_target_bs", "")
                task.handover_target_rssi_dbm = getattr(dev, "handover_target_rssi_dbm", "")
                task.handover_target_rssi_gain_db = getattr(dev, "handover_target_rssi_gain_db", "")
                task.handover_best_available_bs = getattr(dev, "handover_best_available_bs", "")
                task.handover_best_available_rssi_dbm = getattr(dev, "handover_best_available_rssi_dbm", "")
                task.handover_chosen_cpu_util = getattr(dev, "chosen_cpu_util", "")
                task.handover_chosen_mem_util = getattr(dev, "chosen_mem_util", "")
                task.handover_oracle_cpu_util = getattr(dev, "oracle_cpu_util", "")
                task.handover_oracle_mem_util = getattr(dev, "oracle_mem_util", "")

            # 4) Scheduler → assignments
            self.scheduler.mobility = self.mobility
            self.scheduler.active_devices_this_round = num_devices
            assignments = self.scheduler.schedule(self.tasks, self.servers, self.current_time)


            # ---- MEC↔MEC collaboration aggregates for this round ----
            name_to_srv = {
                s.name: s for s in (self.servers + self.access_points)
            }
            collab_in = defaultdict(int)          # tasks collaborated into this MEC
            wp_local  = defaultdict(int)          # tasks served by their own WP-MEC
            collab_dist_sum   = defaultdict(float)
            collab_dist_count = defaultdict(int)

            for task, srv in assignments:
                sname = str(getattr(srv, "name", ""))
                if not sname.startswith("MEC"):
                    continue

                wp = str(getattr(task, "wireless_ap", "") or "")
                if not wp:
                    # fall back to mobility attachment if needed
                    try:
                        dev = self.mobility.entities[task.entity_id]
                        ap  = getattr(dev, "attached_server", None)
                        wp  = str(getattr(ap, "name", "") or "")
                    except Exception:
                        wp = ""

                if wp == sname:
                    wp_local[sname] += 1
                else:
                    collab_in[sname] += 1
                    src = name_to_srv.get(wp)
                    if src and hasattr(src, "x") and hasattr(src, "y") and hasattr(srv, "x") and hasattr(srv, "y"):
                        d = _euclid(src.x, src.y, srv.x, srv.y)
                        collab_dist_sum[sname]   += d
                        collab_dist_count[sname] += 1


            # Publish MEC->Metro pairs for GUI
            self._update_gui_links(assignments)

            # Publish per-device processing links for the GUI (dotted device
            # -> server line). Local-device processing never leaves the
            # device, so there's nothing to draw a connection to.
            #
            # Devices are only ever radio-connected to a wireless point (MEC
            # or UAV_BS) -- never directly to Metro/Cloud. So when a task's
            # actual processing server IS Metro/Cloud (the wireless point's
            # own compute was full and the workload spilled over), the
            # visual line from the device still terminates at its wireless
            # AP; the existing MEC->Metro/Cloud backhaul arrow (built in
            # _update_gui_links, keyed off task.wireless_ap) draws the
            # remaining Metro/Cloud hop, so the two segments together show
            # device -> AP -> Cloud instead of a line straight to the cloud.
            def _visual_ap(task, server):
                sname = str(getattr(server, "name", ""))
                if sname.startswith("Metro") or sname.startswith("Cloud"):
                    wp_name = str(getattr(task, "wireless_ap", "") or "")
                    return name_to_srv.get(wp_name, server)
                return server

            self.live_processing_server_by_entity = {
                task.entity_id: _visual_ap(task, server)
                for task, server in assignments
                if server is not None and not str(getattr(server, "name", "")).startswith("LocalDevice")
            }
            live_server_load = {}
            for task, server in assignments:
                if server is None or str(getattr(server, "name", "")).startswith("LocalDevice"):
                    continue
                sname = str(getattr(server, "name", ""))
                cap = max(1e-9, float(getattr(server, "cpu_capacity", 0.0) or 0.0))
                live_server_load[sname] = live_server_load.get(sname, 0.0) + max(0.0, float(getattr(task, "cpu_demand", 0.0) or 0.0)) / cap
            self.live_server_cpu_util_by_name = {name: max(0.0, min(1.0, value)) for name, value in live_server_load.items()}
            # On-device compute utilization for the GUI: fraction of the
            # device's own local_cpu_capacity its task demanded, for
            # whichever devices actually processed locally this round (a
            # device's local compute isn't a standing reservation like a
            # server's -- it's this round's instantaneous demand/capacity).
            self.live_local_cpu_util_by_entity = {
                task.entity_id: max(0.0, min(1.0, float(getattr(task, "cpu_demand", 0.0) or 0.0)
                                              / max(1e-9, float(getattr(task, "local_cpu_capacity", 0.0) or 0.0))))
                for task, server in assignments
                if server is not None and str(getattr(server, "name", "")).startswith("LocalDevice")
            }

            initial_count  = num_devices
            assigned_count = len(assignments)
            failed_count   = initial_count - assigned_count

            # 5) Remove assigned tasks from pending list
            self.tasks = [t for t in self.tasks if not t.is_assigned()]

            # 6) Every chosen action must have its observed packet/compute result.
            for task, _server in assignments:
                if getattr(task, "measurement_basis", "") != "realized_execution":
                    raise RuntimeError("assigned task lacks realized execution metrics")


            # 7) Per-server metrics (mobility-aware propagation)
            for srv in self.servers:
                assigned_here = [t for t, s in assignments if s is srv]
                if not assigned_here:
                    continue

                total_data = sum(t.data_size_kb for t in assigned_here)

                tx_delay_ms   = 0.0
                prop_delay_ms = 0.0
                tx_cost       = 0.0
                proc_cost     = 0.0
                energy        = 0.0
                signals       = []
                ap_names      = []
                MEC_Metro_dists = []

                tier = getattr(srv, "tier", None) or srv.name.split("_")[0]

                for t in assigned_here:
                    sig = getattr(t, "signal_dbm", None)
                    dev_t = self.mobility.entities[t.entity_id]
                    if sig is None:
                        ap_name = getattr(t, "wireless_ap", "")
                        ap_srv = name_to_srv.get(ap_name) if ap_name else None
                        try:
                            if ap_srv is not None:
                                sig = dev_t.signal_strength(ap_srv)   # ✅ serving AP
                            else:
                                sig = calculate_rssi_dbm(
                                    getattr(t, "distance_to_bs", 1.0),
                                    band_name=self.ground_access_band_name,
                                )
                        except Exception:
                            sig = calculate_rssi_dbm(
                                getattr(t, "distance_to_bs", 1.0),
                                band_name=self.ground_access_band_name,
                            )
                    signals.append(sig)

                    if getattr(t, "measurement_basis", "") == "realized_execution":
                        tx_delay_ms += float(getattr(t, "network_transmission_delay_ms", 0.0) or 0.0)
                        prop_delay_ms += float(getattr(t, "network_propagation_delay_ms", 0.0) or 0.0)
                        tx_cost += float(getattr(t, "communication_cost", 0.0) or 0.0)
                        proc_cost += float(getattr(t, "processing_cost", 0.0) or 0.0)
                        energy += float(getattr(t, "total_energy_j", 0.0) or 0.0)
                    else:
                        raise RuntimeError("server metrics require realized execution")

                    ap_name = getattr(t, "wireless_ap", "")
                    if not ap_name:
                        ap = getattr(dev_t, "attached_server", None)
                        ap_name = getattr(ap, "name", "")
                    if not ap_name:
                        dx, dy = dev_t.get_position()
                        ap_fallback, _ = _nearest_wp_distance(dx, dy, self.servers + self.access_points)
                        ap_name = getattr(ap_fallback, "name", "")
                    if ap_name:
                        ap_names.append(ap_name)

                    if tier == "Metro" and ap_name:
                        ap_srv = next((s for s in self.servers if s.name == ap_name), None)
                        if ap_srv is not None and hasattr(ap_srv, "x") and hasattr(ap_srv, "y") \
                           and hasattr(srv, "x") and hasattr(srv, "y"):
                            MEC_Metro_dists.append(_euclid(ap_srv.x, ap_srv.y, srv.x, srv.y))

                wireless_ap = Counter(ap_names).most_common(1)[0][0] if ap_names else ""
                avg_MEC2Metro = round(sum(MEC_Metro_dists) / len(MEC_Metro_dists), 1) if MEC_Metro_dists else ""

                cpu_util, storage_util, memory_util = srv.utilization()
                # Network pressure is measured by packet/link queues. This
                # compatibility field reports bounded compute-resource pressure,
                # never cumulative KB divided by an unspecified bandwidth value.
                congestion = srv.congestion()

               # base metrics (what you already have)
                metrics = {
                    "round_no": round_no,
                    "devices": initial_count,
                    "workload": total_data,
                    **mob_metrics,
                    "avg_signal": round(sum(signals) / len(signals), 2) if signals else 0.0,
                    "wireless_ap": wireless_ap,
                    "MEC2Metro_m": avg_MEC2Metro,
                    "cpu_util": cpu_util,
                    "memory_util": memory_util,
                    "storage_util": storage_util,
                    "layer": srv.name,
                    "avg_tx":   round(tx_delay_ms   / max(1, len(assigned_here)), 2),  # guard ÷0
                    "avg_prop": round(prop_delay_ms / max(1, len(assigned_here)), 2),
                    # Full precision retained in the stored CSV (Task 52): small
                    # nonzero costs/energy must not be rounded to 0. Presentation
                    # rounding, if any, is a display-only concern in the formatter.
                    "tx_cost":  tx_cost,
                    "proc_cost": proc_cost,
                    "energy":   energy,
                    "congestion": congestion,
                    "flag": assigned_here[0].priority if assigned_here else "",
                    "failed": initial_count - len(assignments),
                }

                # decide if this row is an MEC server
                is_MEC = srv.name.startswith("MEC")  # or: (_tier_of(srv) == "MEC")

                if is_MEC:
                    eci = collab_in.get(srv.name, 0)      # collaborated into this MEC (computed earlier)
                    ewp = wp_local.get(srv.name, 0)       # served by own WP-MEC
                    denom = (eci + ewp) or 1
                    ratio = round(eci / denom, 3)

                    if collab_dist_count.get(srv.name, 0) > 0:
                        avg_collab_d = round(collab_dist_sum[srv.name] / collab_dist_count[srv.name], 1)
                    else:
                        avg_collab_d = ""

                    metrics.update({
                        "MEC_collab_in":    eci,
                        "MEC_wp_local":     ewp,
                        "MEC_collab_ratio": ratio,
                        "MEC_collab_dist_m": avg_collab_d,
                    })
                else:
                    # keep Reporter happy: always provide the same keys
                    metrics.update({
                        "MEC_collab_in":    "",
                        "MEC_wp_local":     "",
                        "MEC_collab_ratio": "",
                        "MEC_collab_dist_m": "",
                    })

                self.task_reporter.report(metrics)

            if progress_callback is not None:
                progress_callback(round_id=round_no, total_rounds=total_rounds, event="round_end")

            # Advance the discrete simulation clock. Resources are released
            # when their configured service time has elapsed.
            self.current_time += round_duration_ms

            if self.stop_requested:
                break

        if hasattr(self.scheduler, "finalize"):
            self.scheduler.finalize()
        if self._mobility_train_mode:
            handover_scheduler = getattr(self.mobility, "handover_scheduler", None)
            if handover_scheduler is not None and hasattr(handover_scheduler, "save"):
                handover_scheduler.save()
        if hasattr(self.mobility, "save_logs"):
            self.mobility.save_logs(output_dir=str(self.mobility_dir))
        if hasattr(self.task_reporter, "close"):
            self.task_reporter.close()


