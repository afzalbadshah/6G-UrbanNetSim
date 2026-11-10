"""Compose an explicit access layer on top of a declared physical topology.

The builder keeps the immutable base topology unchanged and returns a derived
mapping containing access-only ground BS nodes plus explicit wired/backhaul routes
to every processing destination.  This is the composition seam between deployment
configuration and the packet engine; it does not invent shortest paths at runtime.

It is also the ONE place that reconciles a compute tier's site COUNT (declared in
``config/servers.py``'s ``num_datacenters``) with the physical topology graph
(declared in ``config/topology.py``). ``config/topology.py`` is deliberately pure,
provenance-tracked DATA with a fixed node list -- it does not know about
``SERVER_CONFIG``. So when ``num_datacenters`` for MEC or UAV_BS changes, this
module regenerates that tier's nodes/links/routes to match, at coverage-optimizing
positions, instead of requiring a hand-edit of the frozen topology file.

How to extend: add another placement strategy in ``_positions`` and a matching
validated deployment profile.  Every new access point must receive a physical node,
one shared radio resource (created by the normal runtime registry), a compute
anchor, and routes to every action tier that the scheduler can select. To let
another compute tier's count drive the topology the same way MEC/UAV_BS do, add
its site count to the ``compute_tier_counts`` mapping passed into
``build_deployed_topology`` and extend the regeneration calls below.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Mapping

from config.topology import DeploymentProfile
from config import zones as zone_config
from entities.access_point import AccessPoint
from mobility.road_network import distribute_coverage_positions, farthest_first_order
from network import PhysicalTopology, Position3D


def _thaw(value):
    """Convert the frozen configuration snapshot to mutable containers."""

    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, list):
        return [_thaw(item) for item in value]
    return value


def _area_from_extent(extent: Mapping[str, float]) -> dict[str, float]:
    """Adapt the topology's ``{x_min,x_max,y_min,y_max}`` extent to the
    ``{xmin,xmax,ymin,ymax}`` shape ``mobility.road_network`` expects."""

    return {
        "xmin": float(extent["x_min"]),
        "xmax": float(extent["x_max"]),
        "ymin": float(extent["y_min"]),
        "ymax": float(extent["y_max"]),
    }


def _node_map(raw: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    return {str(item["node_id"]): item for item in raw["nodes"]}


def _nearest_mec(x_m: float, y_m: float, nodes: Mapping[str, Mapping[str, object]]) -> str:
    candidates = {
        node_id: item
        for node_id, item in nodes.items()
        if str(item.get("node_kind", "")).lower() == "mec"
    }
    if not candidates:
        raise ValueError("an access deployment requires at least one MEC compute site")
    return min(
        candidates,
        key=lambda node_id: hypot(
            x_m - float(candidates[node_id]["x_m"]),
            y_m - float(candidates[node_id]["y_m"]),
        ),
    )


def _route_by_endpoints(raw: Mapping[str, object]) -> dict[tuple[str, str], Mapping[str, object]]:
    return {
        (str(item["source_node_id"]), str(item["destination_node_id"])): item
        for item in raw["routes"]
    }


def _provenance(raw: Mapping[str, object], profile_id: str) -> Mapping[str, object]:
    """Reuse a declared provenance record without inventing source claims."""

    for item in raw["links"]:
        if item.get("link_profile_id") == profile_id:
            return item["provenance"]
    for item in raw["nodes"]:
        if item.get("profile_id") == profile_id:
            return item["provenance"]
    # The base profile always contains at least one sourced record; this fallback
    # is only for a custom profile whose radio metadata is intentionally reused.
    return raw["nodes"][0]["provenance"]


def _nearest_node_of_kind(
    x_m: float, y_m: float, nodes: Mapping[str, Mapping[str, object]], node_kind: str
) -> str:
    """Return the id of the nearest node of ``node_kind`` to ``(x_m, y_m)``."""

    candidates = {
        node_id: item
        for node_id, item in nodes.items()
        if str(item.get("node_kind", "")).lower() == node_kind
    }
    if not candidates:
        raise ValueError(f"a topology regeneration requires at least one {node_kind!r} node")
    return min(
        candidates,
        key=lambda node_id: hypot(
            x_m - float(candidates[node_id]["x_m"]),
            y_m - float(candidates[node_id]["y_m"]),
        ),
    )


def _capture_tier_template(raw: Mapping[str, object], node_kind: str) -> tuple[float, str, Mapping[str, object]]:
    """Return ``(z_m, profile_id, provenance)`` from an existing node of ``node_kind``.

    Regenerating a tier reuses ITS OWN declared altitude/profile/provenance rather
    than inventing new ones, so a site-count change never drifts from the sourced
    metadata ``config/topology.py`` already carries for that tier.
    """

    for item in raw["nodes"]:
        if str(item.get("node_kind", "")).lower() == node_kind:
            return float(item["z_m"]), str(item["profile_id"]), item["provenance"]
    raise ValueError(
        f"cannot regenerate {node_kind!r} sites: the base topology has no "
        f"{node_kind!r} node to source altitude/profile/provenance from"
    )


def _drop_nodes_of_kind(raw: dict, node_kind: str) -> set[str]:
    """Remove every node of ``node_kind`` plus any link/route touching it.

    Returns the set of removed node ids. Regeneration always replaces the whole
    tier rather than trying to patch individual sites, so counting down or up
    both work the same way.
    """

    removed = {
        str(item["node_id"])
        for item in raw["nodes"]
        if str(item.get("node_kind", "")).lower() == node_kind
    }
    if not removed:
        return removed
    raw["nodes"] = [item for item in raw["nodes"] if str(item["node_id"]) not in removed]
    raw["links"] = [
        item
        for item in raw["links"]
        if str(item["source_node_id"]) not in removed and str(item["destination_node_id"]) not in removed
    ]
    raw["routes"] = [
        item
        for item in raw["routes"]
        if str(item["source_node_id"]) not in removed and str(item["destination_node_id"]) not in removed
    ]
    return removed


def _regenerate_mec_sites(
    raw: dict,
    positions: list[tuple[float, float]],
    access_profile_id: str,
    anchor_profile_id: str,
    promoted: bool = False,
) -> None:
    """Resize the MEC tier to sites at the given ``(x_m, y_m)`` positions.

    ``positions`` is usually a coverage-optimized subset of the SAME ground BS
    grid the access layer below is built from (see ``build_deployed_topology``)
    -- upgrading existing sites with compute -- or, when no ground BS layer is
    requested, MEC's own independent coverage-optimizing placement.

    Rebuilds MEC's wired backhaul to its nearest METRO and the composed MEC->
    METRO->CLOUD route for every site, mirroring the pattern ``config/topology.py``
    declares for its own fixed MEC sites -- just generalized to any count.
    """

    z_m, profile_id, node_provenance = _capture_tier_template(raw, "mec")
    fibre_prov = _provenance(raw, "fibre-backhaul")
    nodes_before = _node_map(raw)
    metro_to_cloud = {
        str(item["source_node_id"]): item
        for item in raw["links"]
        if str(nodes_before.get(item["destination_node_id"], {}).get("node_kind", "")).lower()
        == "cloud_gateway"
    }
    _drop_nodes_of_kind(raw, "mec")
    if not positions:
        return

    for index, (x_m, y_m) in enumerate(positions):
        mec_id = f"MEC_{index + 1}"
        raw["nodes"].append(
            {
                "node_id": mec_id, "node_kind": "mec",
                "x_m": float(x_m), "y_m": float(y_m), "z_m": z_m,
                "profile_id": access_profile_id, "anchor_profile_id": anchor_profile_id,
                "provenance": node_provenance,
                "provides_radio_access": promoted,
            }
        )
        nodes = _node_map(raw)
        metro_id = _nearest_node_of_kind(x_m, y_m, nodes, "metro")
        metro_node = nodes[metro_id]
        backhaul_link_id = f"{mec_id}->{metro_id}"
        raw["links"].append(
            {
                "link_id": backhaul_link_id,
                "source_node_id": mec_id, "destination_node_id": metro_id,
                "link_profile_id": "fibre-backhaul",
                "route_distance_m": hypot(x_m - float(metro_node["x_m"]), y_m - float(metro_node["y_m"])),
                "medium": "wired", "provenance": fibre_prov,
            }
        )
        raw["routes"].append(
            {
                "route_id": f"r_{mec_id.lower()}_{metro_id.lower()}",
                "source_node_id": mec_id, "destination_node_id": metro_id,
                "hop_ids": [backhaul_link_id], "provenance": fibre_prov,
            }
        )
        cloud_link = metro_to_cloud.get(metro_id)
        if cloud_link is not None:
            raw["routes"].append(
                {
                    "route_id": f"r_{mec_id.lower()}_{cloud_link['destination_node_id'].lower()}",
                    "source_node_id": mec_id, "destination_node_id": cloud_link["destination_node_id"],
                    "hop_ids": [backhaul_link_id, cloud_link["link_id"]], "provenance": fibre_prov,
                }
            )


def _regenerate_uav_sites(
    raw: dict, count: int, area: Mapping[str, float], access_profile_id: str, anchor_profile_id: str
) -> None:
    """Resize the UAV_BS tier to ``count`` sites at coverage-optimizing positions.

    Must run AFTER ``_regenerate_mec_sites`` -- each UAV-BS backhauls through its
    nearest MEC (reverse mmWave hop), so it needs the final MEC site set to anchor
    to, then composes METRO/CLOUD routes through that MEC's own routes.
    """

    if count < 0:
        raise ValueError("UAV_BS site count must be non-negative")
    z_m, profile_id, node_provenance = _capture_tier_template(raw, "uav_bs")
    air_prov = _provenance(raw, "nr-mmwave-backhaul")
    _drop_nodes_of_kind(raw, "uav_bs")
    if count == 0:
        return

    positions = distribute_coverage_positions(count, area)
    for index, (x_m, y_m) in enumerate(positions):
        uav_id = f"UAV_BS_{index + 1}"
        raw["nodes"].append(
            {
                "node_id": uav_id, "node_kind": "uav_bs",
                "x_m": float(x_m), "y_m": float(y_m), "z_m": z_m,
                "profile_id": access_profile_id, "anchor_profile_id": anchor_profile_id,
                "provenance": node_provenance,
            }
        )
        nodes = _node_map(raw)
        mec_id = _nearest_mec(x_m, y_m, nodes)
        mec_node = nodes[mec_id]
        backhaul_link_id = f"{uav_id}->{mec_id}"
        raw["links"].append(
            {
                "link_id": backhaul_link_id,
                "source_node_id": uav_id, "destination_node_id": mec_id,
                "link_profile_id": "nr-mmwave-backhaul",
                "route_distance_m": Position3D(x_m, y_m, z_m).distance_to(
                    Position3D(float(mec_node["x_m"]), float(mec_node["y_m"]), float(mec_node["z_m"]))
                ),
                "medium": "radio", "provenance": air_prov,
            }
        )
        # The backhaul terminates at a real MEC compute site, so declare that
        # MEC as a first-class processing destination.  Previously the link
        # existed physically but only composed METRO/CLOUD routes were added;
        # the action mask therefore (correctly, from the incomplete route
        # table) rejected MEC whenever a device happened to attach to UAV-BS.
        raw["routes"].append(
            {
                "route_id": f"r_{uav_id.lower()}_{mec_id.lower()}",
                "source_node_id": uav_id,
                "destination_node_id": mec_id,
                "hop_ids": [backhaul_link_id],
                "provenance": air_prov,
            }
        )
        # Compose through whatever METRO/CLOUD routes this MEC already has
        # (its own regenerated routes, computed just above for every MEC site).
        for (source_id, destination_id), route in _route_by_endpoints(raw).items():
            if source_id != mec_id:
                continue
            raw["routes"].append(
                {
                    "route_id": f"r_{uav_id.lower()}_{destination_id.lower()}",
                    "source_node_id": uav_id, "destination_node_id": destination_id,
                    "hop_ids": [backhaul_link_id, *list(route["hop_ids"])],
                    "provenance": air_prov,
                }
            )


def _add_ntn_layer(raw: dict) -> None:
    """Add the LEO satellite access nodes, ground gateway and routes to Cloud.

    Transparent (bent-pipe) payload: device -> satellite (radio access, added at
    runtime) -> gateway (Ka feeder, radio) -> Cloud (fibre). One satellite node
    per terminal type (S-band IoT-NTN, Ka-band broadband) so each has its own
    shared radio resource/queue. The satellite's live position (slant range from
    the pass elevation) is supplied at runtime as a position override; the
    nominal position here is the zenith point above the map centre.
    """

    sat_cfg = zone_config.SATELLITE_CONFIG
    nodes = _node_map(raw)
    cloud_ids = [
        node_id for node_id, item in nodes.items()
        if str(item.get("node_kind", "")).lower() == "cloud_gateway"
    ]
    if not cloud_ids:
        raise ValueError("the satellite layer needs a Cloud gateway node to route to")
    cloud_id = cloud_ids[0]
    extent = raw["coordinate_frame"]["extent_m"]
    cx = (float(extent["x_min"]) + float(extent["x_max"])) / 2.0
    cy = (float(extent["y_min"]) + float(extent["y_max"])) / 2.0
    altitude = float(sat_cfg["altitude_m"])
    prov = _provenance(raw, "fibre-core")
    gw_x, gw_y = (float(v) for v in sat_cfg["gateway_xy_m"])
    gw_z = float(sat_cfg["gateway_height_m"])
    gw_id = zone_config.SAT_GATEWAY_ID
    raw["nodes"].append({
        "node_id": gw_id, "node_kind": "sat_gateway",
        "x_m": gw_x, "y_m": gw_y, "z_m": gw_z,
        "profile_id": "ntn-feeder", "provenance": prov,
    })
    access_profile = {"LEO_SAT_S": "ntn-s-access", "LEO_SAT_KA": "ntn-ka-access"}
    for sat_id in zone_config.SAT_NODE_IDS:
        # One gateway->Cloud transport channel per satellite carrier (separate
        # modem chain / VLAN), so the FIFO packet engine never makes a fast Ka
        # flow wait behind slow S-band IoT packets on the same reservation queue.
        gw_cloud_link = f"{gw_id}->{cloud_id}::{sat_id}"
        raw["links"].append({
            "link_id": gw_cloud_link, "source_node_id": gw_id, "destination_node_id": cloud_id,
            "link_profile_id": "fibre-core",
            "route_distance_m": float(sat_cfg["gateway_cloud_fibre_m"]),
            "medium": "wired", "provenance": prov,
        })
        raw["nodes"].append({
            "node_id": sat_id, "node_kind": "ntn",
            "x_m": cx, "y_m": cy, "z_m": altitude,
            "profile_id": access_profile[sat_id], "provenance": prov,
        })
        feeder_id = f"{sat_id}->{gw_id}"
        raw["links"].append({
            "link_id": feeder_id, "source_node_id": sat_id, "destination_node_id": gw_id,
            "link_profile_id": "ntn-feeder",
            "route_distance_m": Position3D(cx, cy, altitude).distance_to(Position3D(gw_x, gw_y, gw_z)),
            "medium": "radio", "provenance": prov,
        })
        raw["routes"].append({
            "route_id": f"r_{sat_id.lower()}_{cloud_id.lower()}",
            "source_node_id": sat_id, "destination_node_id": cloud_id,
            "hop_ids": [feeder_id, gw_cloud_link], "provenance": prov,
        })


@dataclass(frozen=True, slots=True)
class DeploymentTopology:
    """Derived topology and runtime access-point objects for one profile."""

    profile_id: str
    mapping: Mapping[str, object]
    access_points: tuple[AccessPoint, ...]


def build_deployed_topology(
    base_profile_id: str,
    base_mapping: Mapping[str, object],
    deployment: DeploymentProfile,
    *,
    compute_tier_counts: Mapping[str, int] | None = None,
) -> DeploymentTopology:
    """Add ground BSs and explicit routes to a base topology; upgrade some to MEC.

    ``compute_tier_counts`` (e.g. ``{"MEC": 6, "UAV_BS": 4}``), when given,
    resizes those compute tiers to ``config/servers.py``'s current
    ``num_datacenters`` BEFORE the rest of the access layer is built. A tier
    left out of the mapping keeps exactly the sites already in ``base_mapping``.
    Metro/Cloud are not resized this way: Metro stays exactly as declared, and
    Cloud is always the single routed endpoint.

    MEC does not get an independently-placed set of sites: when this deployment
    has any ground BS sites at all, MEC's site count is filled by UPGRADING that
    many of those same ground BS positions to compute, chosen by coverage
    (``farthest_first_order`` -- most evenly spread first). Growing the MEC count
    later only promotes more of the remaining ground BS sites; it never moves or
    demotes one already promoted. The rest of the ground BS grid is generated as
    plain access-only towers exactly as before, each with a declared fibre-
    backhaul hop to its nearest MEC and routes composed through that MEC's own
    routes. A reverse MEC-to-UAV backhaul hop is added where needed so a ground
    access point can legally reach a UAV compute action as well. Only when NO
    ground BS layer is requested does MEC fall back to its own independent
    coverage-optimizing placement (see ``_regenerate_mec_sites``).
    """

    if not isinstance(deployment, DeploymentProfile):
        raise ValueError("deployment must be a DeploymentProfile")
    raw = _thaw(base_mapping)
    if not isinstance(raw, dict) or not raw.get("nodes") or not raw.get("links"):
        raise ValueError("base topology mapping must contain nodes and links")

    area = _area_from_extent(raw["coordinate_frame"]["extent_m"])
    # Zoned scenario: ground access (and MEC) only inside the cellular zone,
    # UAV-BS cells only inside the UAV zone. Without zones both use the city.
    access_area = (
        zone_config.zone_area(deployment.access_zone)
        if deployment.access_zone and zone_config.ZONES_ENABLED
        else area
    )
    uav_area = (
        zone_config.zone_area(deployment.uav_zone)
        if deployment.uav_zone and zone_config.ZONES_ENABLED
        else area
    )
    ground_bs_positions = (
        distribute_coverage_positions(deployment.ground_bs_count, access_area)
        if deployment.ground_bs_count > 0
        else []
    )

    # MEC first: UAV_BS backhauls through its nearest MEC, so it needs the
    # final MEC site set already in place before it can anchor to one.
    counts = dict(compute_tier_counts or {})
    promoted_positions: set[tuple[float, float]] = set()
    if "MEC" in counts:
        mec_count = int(counts["MEC"])
        if ground_bs_positions:
            # Upgrade the mec_count best-spread of the SAME ground BS sites
            # generated below to compute, instead of inventing a second,
            # unrelated set of MEC positions -- one site grid, one ranking.
            mec_positions = farthest_first_order(ground_bs_positions)[:mec_count]
            promoted_positions = set(mec_positions)
        else:
            # No ground-BS layer requested: nothing to promote from, so MEC
            # keeps its own independent coverage-optimizing placement.
            mec_positions = distribute_coverage_positions(mec_count, access_area)
        _regenerate_mec_sites(
            raw,
            mec_positions,
            deployment.ground_bs_profile_id,
            deployment.anchor_profile_id,
            promoted=bool(ground_bs_positions),
        )
    if "UAV_BS" in counts:
        _regenerate_uav_sites(
            raw, int(counts["UAV_BS"]), uav_area, deployment.uav_profile_id, deployment.anchor_profile_id
        )
    if deployment.ntn_enabled:
        _add_ntn_layer(raw)

    if deployment.ground_bs_count == 0 and deployment.hotspot_count == 0:
        return DeploymentTopology(base_profile_id, raw, ())

    nodes = _node_map(raw)
    locations = [position for position in ground_bs_positions if position not in promoted_positions]
    fibre_prov = _provenance(raw, "fibre-backhaul")
    air_prov = _provenance(raw, "nr-mmwave-backhaul")
    routes = _route_by_endpoints(raw)
    mec_ids = [node_id for node_id, item in nodes.items() if str(item.get("node_kind", "")).lower() == "mec"]
    metro_ids = [node_id for node_id, item in nodes.items() if str(item.get("node_kind", "")).lower() == "metro"]
    cloud_ids = [node_id for node_id, item in nodes.items() if str(item.get("node_kind", "")).lower() == "cloud_gateway"]
    uav_ids = [node_id for node_id, item in nodes.items() if str(item.get("node_kind", "")).lower() == "uav_bs"]
    if not mec_ids:
        raise ValueError("ground access deployment requires at least one MEC")

    links = raw["links"]
    route_items = raw["routes"]
    access_points: list[AccessPoint] = []

    def add_route(route_id: str, source: str, destination: str, hop_ids: list[str]) -> None:
        if (source, destination) in routes:
            return
        route_item = {
            "route_id": route_id,
            "source_node_id": source,
            "destination_node_id": destination,
            "hop_ids": hop_ids,
            "provenance": fibre_prov,
        }
        route_items.append(route_item)
        routes[(source, destination)] = route_item

    def _place_access_site(access_id: str, index: int, x_m: float, y_m: float, profile_id: str) -> None:
        """One access-only ground site: node, AccessPoint, MEC backhaul links/
        routes, composed Metro/Cloud routes, and reverse UAV backhaul routes.
        Shared by the main coverage grid and any hotspot overlay -- identical
        wiring, the only difference is which radio ``profile_id`` (band) the
        site uses, so a hotspot band never gets a lesser-connected node."""
        anchor_id = _nearest_mec(x_m, y_m, nodes)
        nodes[access_id] = {
            "node_id": access_id,
            "node_kind": "access_tower",
            "x_m": x_m,
            "y_m": y_m,
            "z_m": float(deployment.ground_bs_height_m),
            "profile_id": profile_id,
            "anchor_profile_id": deployment.anchor_profile_id,
            "provenance": _provenance(raw, profile_id),
        }
        access_points.append(
            AccessPoint(
                name=access_id,
                index=index,
                x=x_m,
                y=y_m,
                z=float(deployment.ground_bs_height_m),
                profile_id=profile_id,
                compute_anchor_id=anchor_id,
            )
        )

        mec_hops: dict[str, str] = {}
        for mec_id in mec_ids:
            link_id = f"{access_id}->{mec_id}"
            mec_hops[mec_id] = link_id
            mec_node = nodes[mec_id]
            links.append(
                {
                    "link_id": link_id,
                    "source_node_id": access_id,
                    "destination_node_id": mec_id,
                    "link_profile_id": "fibre-backhaul",
                    "route_distance_m": hypot(
                        x_m - float(mec_node["x_m"]),
                        y_m - float(mec_node["y_m"]),
                    ),
                    "medium": "wired",
                    "provenance": fibre_prov,
                }
            )
            add_route(f"r_{access_id.lower()}_{mec_id.lower()}", access_id, mec_id, [link_id])

        # Compose routes through the MEC's already-declared backbone paths.
        for mec_id in mec_ids:
            first_hop = mec_hops[mec_id]
            for destination_id in (*metro_ids, *cloud_ids):
                base_route = routes.get((mec_id, destination_id))
                if base_route is not None:
                    add_route(
                        f"r_{access_id.lower()}_{destination_id.lower()}",
                        access_id,
                        destination_id,
                        [first_hop, *list(base_route["hop_ids"])],
                    )

        # Ground access to an aerial compute endpoint uses an explicit reverse
        # backhaul hop; no hidden shortest path is introduced.
        for uav_id in uav_ids:
            mec_node = nodes[anchor_id]
            uav_node = nodes[uav_id]
            reverse_id = f"{anchor_id}->{uav_id}::reverse-backhaul"
            if not any(item.get("link_id") == reverse_id for item in links):
                links.append(
                    {
                        "link_id": reverse_id,
                        "source_node_id": anchor_id,
                        "destination_node_id": uav_id,
                        "link_profile_id": "nr-mmwave-backhaul",
                        "route_distance_m": Position3D(
                            float(mec_node["x_m"]), float(mec_node["y_m"]), float(mec_node["z_m"])
                        ).distance_to(
                            Position3D(float(uav_node["x_m"]), float(uav_node["y_m"]), float(uav_node["z_m"]))
                        ),
                        "medium": "radio",
                        "provenance": air_prov,
                    }
                )
            add_route(
                f"r_{access_id.lower()}_{uav_id.lower()}",
                access_id,
                uav_id,
                [mec_hops[anchor_id], reverse_id],
            )

    for index, (x_m, y_m) in enumerate(locations):
        _place_access_site(f"BS_{index + 1}", index, x_m, y_m, deployment.ground_bs_profile_id)

    if deployment.hotspot_count > 0:
        # A small, independently-banded overlay (e.g. mmWave/subTHz small
        # cells scattered on top of the FR1/FR3 macro grid above) -- same
        # coverage-spreading placement as the main grid, just at a much
        # smaller count, and identically wired to MEC/Metro/Cloud/UAV via
        # the same helper. "HOTSPOT_" keeps ids distinct from "BS_*".
        hotspot_positions = distribute_coverage_positions(deployment.hotspot_count, access_area)
        for index, (x_m, y_m) in enumerate(hotspot_positions):
            _place_access_site(
                f"HOTSPOT_{index + 1}", index, x_m, y_m, deployment.hotspot_profile_id
            )

    raw["nodes"] = list(nodes.values())
    raw["links"] = links
    raw["routes"] = route_items
    return DeploymentTopology(
        profile_id=f"{base_profile_id}+{deployment.profile_id}",
        mapping=raw,
        access_points=tuple(access_points),
    )
