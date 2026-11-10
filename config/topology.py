"""
Physical topology profiles (Task 33 / R2 — config side, DATA only).

How to extend: to add or move a node, link, or route, edit `TOPOLOGY_PROFILES`
(one place); to add a whole layout, add a new named entry to it.

STATUS: sourced configuration DATA; no runtime types, no simulator wiring. It
supplies immutable city/node/link/route DATA in SI units that Codex's runtime
`network/topology.py` + `network/routing.py` (R2) will read and validate into the
reserved objects `Position3D` / `TopologyNode` / `TopologyLink` / `Route`
(IMPLEMENTATION_CONTRACT.md §12). This file therefore uses PLAIN dicts with the
contract's field names and does NOT import `network/`.

Ownership: Claude (sourced parameter DATA). Every default (node, link, route, and
coordinate frame) carries a `provenance` record with the §11 fields (source_id,
source_uri, source_version, valid_from, valid_to, validity_notes, maturity). Units
follow §2 (metres, `_m` suffix).

IMMUTABILITY (§10 guardrail 6, reproducibility): the published `TOPOLOGY_PROFILES`
graph is DEEP-FROZEN — dicts become read-only `MappingProxyType` and lists become
tuples — so a caller cannot mutate a node/link/route/extent/provenance and silently
change later runs. `get_topology()` returns the frozen snapshot.

REALISM (§10): Cloud is a ROUTED endpoint reached via wired access/backhaul/core
hops, never a remote radio tower used to manufacture latency. Radio path distance
is the 3-D Euclidean distance between antenna positions at runtime; wired hops use
`route_distance_m`.
"""

from config._provenance import provenance, deep_freeze as _deep_freeze  # shared helpers (DRY)
from config._references import cite  # central citation registry (declutters code)
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping
from config.workload import normalize_access_generation

# Allowed node kinds (align with contract §12 examples).
NODE_KINDS = (
    "device",
    "access_tower",   # ground radio access point / MEC-collocated tower
    "mec",
    "metro",
    "cloud_gateway",  # routed endpoint (wired), NOT a radio tower
    "uav_bs",
    "ntn",            # LEO satellite access node (see config/zones.py)
    "sat_gateway",    # satellite ground gateway (teleport) -> fibre to Cloud
)

def _node(node_id, node_kind, x_m, y_m, z_m, profile_id, prov):
    if node_kind not in NODE_KINDS:
        raise ValueError(f"node_kind {node_kind!r} not in {NODE_KINDS}")
    return {
        "node_id": node_id, "node_kind": node_kind,
        "x_m": float(x_m), "y_m": float(y_m), "z_m": float(z_m),
        "profile_id": profile_id, "provenance": prov,
    }


def _link(link_id, src, dst, link_profile_id, route_distance_m, medium, prov):
    return {
        "link_id": link_id, "source_node_id": src, "destination_node_id": dst,
        "link_profile_id": link_profile_id,
        "route_distance_m": float(route_distance_m),
        "medium": medium,               # "radio" | "wired"
        "provenance": prov,
    }


def _route(route_id, src, dst, hop_ids, prov):
    return {"route_id": route_id, "source_node_id": src,
            "destination_node_id": dst, "hop_ids": list(hop_ids), "provenance": prov}


def _frame(extent_m, prov):
    return {"type": "local_cartesian", "unit": "m",
            "z_meaning": "height_above_ground_m",
            "extent_m": dict(extent_m), "provenance": prov}


# ---- Provenance records (reused) -------------------------------------------
# Provenance cited by short id (config/_references.py); notes document each value.
_P_38901 = cite(
    "tr38901", maturity="standards-aligned",
    validity_notes="UMa/UMi antenna heights (BS 25 m, UT 1.5 m)")
_P_UAV = cite(
    "att_flying_cow", maturity="experimental",
    validity_notes="aerial BS altitude 220-380 m (heavy-lift); Flying COW ~137 m tethered")
_P_EDGE = cite(
    "stl_edge", maturity="experimental",
    validity_notes="metro aggregation / regional edge geography; distances are scenario assumptions")
# An existing simulator assumption is NOT externally validated merely because it
# already existed (contract §11) -> experimental, not validated.
_P_SIMDEF = cite(
    "sim_default", maturity="experimental",
    validity_notes="existing CITY_AREA/positions carried forward; not measurement-calibrated")
_P_ROUTE = cite(
    "example_route", maturity="experimental",
    validity_notes="illustrative access->metro->core path; structural default")


# ---- Topology profiles (raw, then deep-frozen) ------------------------------
_CITY_EXTENT_M = {"x_min": 0.0, "x_max": 10_000.0, "y_min": 0.0, "y_max": 10_000.0}
_HANDCHECK_EXTENT_M = {"x_min": 0.0, "x_max": 1000.0, "y_min": 0.0, "y_max": 1000.0}
# 3-D UAV-BS -> ground-anchor MEC distance: sqrt(2500^2 + 0^2 + (300-25)^2).
# Must equal the initial geometry (network.topology validates radio links).
_UAV_MEC_M = 2515.079521605629

_RAW_PROFILES = {
    # Full 10 km x 10 km smart-city reference layout. Infrastructure coordinates
    # here are the AUTHORITY the R5 adapter aligns to; UAV positions are nominal
    # (runtime supplies live positions as dynamic overrides). 4 UAV-BS match the
    # runtime SERVER_CONFIG["UAV_BS"] count.
    "smartcity_default": {
        "coordinate_frame": _frame(_CITY_EXTENT_M, _P_SIMDEF),
        "nodes": [
            _node("MEC_1", "mec", 5000.0, 7500.0, 25.0, "nr-fr1-access", _P_38901),
            _node("MEC_2", "mec", 5000.0, 2500.0, 25.0, "nr-fr1-access", _P_38901),
            _node("Metro_1", "metro", 2700.0, 5000.0, 0.0, "fibre-backhaul", _P_SIMDEF),
            _node("Metro_2", "metro", 7300.0, 5000.0, 0.0, "fibre-backhaul", _P_SIMDEF),
            _node("UAV_BS_1", "uav_bs", 2500.0, 2500.0, 300.0, "nr-fr1-air", _P_UAV),
            _node("UAV_BS_2", "uav_bs", 7500.0, 7500.0, 300.0, "nr-fr1-air", _P_UAV),
            _node("UAV_BS_3", "uav_bs", 2500.0, 7500.0, 300.0, "nr-fr1-air", _P_UAV),
            _node("UAV_BS_4", "uav_bs", 7500.0, 2500.0, 300.0, "nr-fr1-air", _P_UAV),
            # Cloud: geographic position recorded, but reached via wired core only.
            _node("Cloud_1", "cloud_gateway", 5000.0, 5000.0, 0.0, "fibre-core", _P_EDGE),
        ],
        "links": [
            # Wired backhaul: MEC -> Metro; wired core: Metro -> Cloud.
            _link("MEC_1->Metro_1", "MEC_1", "Metro_1", "fibre-backhaul", 10_000.0, "wired", _P_SIMDEF),
            _link("MEC_2->Metro_2", "MEC_2", "Metro_2", "fibre-backhaul", 10_000.0, "wired", _P_SIMDEF),
            _link("Metro_1->Cloud_1", "Metro_1", "Cloud_1", "fibre-core", 50_000.0, "wired", _P_EDGE),
            _link("Metro_2->Cloud_1", "Metro_2", "Cloud_1", "fibre-core", 50_000.0, "wired", _P_EDGE),
            # Aerial mmWave backhaul: each UAV-BS -> nearest ground-anchor MEC (radio;
            # route_distance_m == initial 3-D geometry, validated by network.topology).
            _link("UAV_BS_1->MEC_2", "UAV_BS_1", "MEC_2", "nr-mmwave-backhaul", _UAV_MEC_M, "radio", _P_UAV),
            _link("UAV_BS_2->MEC_1", "UAV_BS_2", "MEC_1", "nr-mmwave-backhaul", _UAV_MEC_M, "radio", _P_UAV),
            _link("UAV_BS_3->MEC_1", "UAV_BS_3", "MEC_1", "nr-mmwave-backhaul", _UAV_MEC_M, "radio", _P_UAV),
            _link("UAV_BS_4->MEC_2", "UAV_BS_4", "MEC_2", "nr-mmwave-backhaul", _UAV_MEC_M, "radio", _P_UAV),
        ],
        # Explicit routes for every access anchor x enabled remote processing tier
        # (Metro, Cloud). MEC/UAV_BS/LOCAL processing terminates at the access node,
        # so needs no infra route. No hidden shortest path.
        "routes": [
            _route("r_mec1_metro", "MEC_1", "Metro_1", ["MEC_1->Metro_1"], _P_ROUTE),
            _route("r_mec2_metro", "MEC_2", "Metro_2", ["MEC_2->Metro_2"], _P_ROUTE),
            _route("r_mec1_cloud", "MEC_1", "Cloud_1", ["MEC_1->Metro_1", "Metro_1->Cloud_1"], _P_ROUTE),
            _route("r_mec2_cloud", "MEC_2", "Cloud_1", ["MEC_2->Metro_2", "Metro_2->Cloud_1"], _P_ROUTE),
            _route("r_uav1_metro", "UAV_BS_1", "Metro_2", ["UAV_BS_1->MEC_2", "MEC_2->Metro_2"], _P_ROUTE),
            _route("r_uav1_cloud", "UAV_BS_1", "Cloud_1", ["UAV_BS_1->MEC_2", "MEC_2->Metro_2", "Metro_2->Cloud_1"], _P_ROUTE),
            _route("r_uav2_metro", "UAV_BS_2", "Metro_1", ["UAV_BS_2->MEC_1", "MEC_1->Metro_1"], _P_ROUTE),
            _route("r_uav2_cloud", "UAV_BS_2", "Cloud_1", ["UAV_BS_2->MEC_1", "MEC_1->Metro_1", "Metro_1->Cloud_1"], _P_ROUTE),
            _route("r_uav3_metro", "UAV_BS_3", "Metro_1", ["UAV_BS_3->MEC_1", "MEC_1->Metro_1"], _P_ROUTE),
            _route("r_uav3_cloud", "UAV_BS_3", "Cloud_1", ["UAV_BS_3->MEC_1", "MEC_1->Metro_1", "Metro_1->Cloud_1"], _P_ROUTE),
            _route("r_uav4_metro", "UAV_BS_4", "Metro_2", ["UAV_BS_4->MEC_2", "MEC_2->Metro_2"], _P_ROUTE),
            _route("r_uav4_cloud", "UAV_BS_4", "Cloud_1", ["UAV_BS_4->MEC_2", "MEC_2->Metro_2", "Metro_2->Cloud_1"], _P_ROUTE),
        ],
    },

    # Minimal deterministic hand-check topology for integration tests.
    # D0 (ground, 1.5 m) <-> A0 (300 m east, 25 m high):
    #   3-D radio distance = sqrt(300^2 + 0^2 + (25-1.5)^2) = 300.919009... m
    # Wired hop A0 -> core with route_distance_m = 1000 m.
    "handcheck_min": {
        "coordinate_frame": _frame(_HANDCHECK_EXTENT_M, _P_SIMDEF),
        "nodes": [
            _node("D0", "device", 0.0, 0.0, 1.5, "nr-fr1-access", _P_38901),
            _node("A0", "access_tower", 300.0, 0.0, 25.0, "nr-fr1-access", _P_38901),
            _node("C0", "cloud_gateway", 500.0, 0.0, 0.0, "fibre-core", _P_EDGE),
        ],
        "links": [
            _link("D0->A0", "D0", "A0", "nr-fr1-access", 300.919009, "radio", _P_38901),
            _link("A0->C0", "A0", "C0", "fibre-core", 1000.0, "wired", _P_EDGE),
        ],
        "routes": [
            _route("r_d0_core", "D0", "C0", ["D0->A0", "A0->C0"], _P_ROUTE),
        ],
    },
}

# Public, deep-immutable snapshot.
TOPOLOGY_PROFILES = _deep_freeze(_RAW_PROFILES)


def get_topology(name):
    """Return a deep-immutable topology profile by name (KeyError lists valid names)."""
    if name not in TOPOLOGY_PROFILES:
        raise KeyError(
            f"Unknown topology {name!r}. Valid: {sorted(TOPOLOGY_PROFILES)}"
        )
    return TOPOLOGY_PROFILES[name]


# =====================================================================
# Deployment profiles (merged from the former config/deployment.py):
# how many ground radio-access base stations and where, separate from MEC
# compute count. To add a layout, add a DeploymentProfile to
# DEPLOYMENT_PROFILES below (one place).
# =====================================================================
@dataclass(frozen=True, slots=True)
class DeploymentProfile:
    """Immutable access-density profile used by the simulator composition root.

    ``ground_bs_count`` is a count of access-only ground radio nodes.  It is not a
    count of MEC compute sites.  ``ground_bs_height_m`` is the antenna height in
    metres; ``ground_bs_profile_id`` resolves to a physical radio link profile.

    ``hotspot_count``/``hotspot_profile_id`` add a SECOND, independently-sized,
    independently-banded set of ground sites on top of the main
    ``ground_bs_count`` grid -- e.g. a sparse mmWave/subTHz small-cell overlay
    scattered across an otherwise-FR1-covered city, matching how those bands
    are actually deployed (hotspot/small-cell, not blanket macro coverage; see
    llm_collab/MESSAGES.md band-density analysis). Both fields are either both
    set or both left at their zero/None defaults -- there is no partial state.
    Hotspot sites use the SAME coverage-spreading placement as the main grid,
    just at a much smaller count, and anchor to the same MEC/backbone routes.
    """

    profile_id: str
    ground_bs_count: int
    ground_bs_profile_id: str = "nr-fr1-access"
    uav_profile_id: str = "nr-fr1-air"
    ground_bs_height_m: float = 25.0
    placement_strategy: str = "grid"
    access_generation: str = "5G"
    anchor_profile_id: str = "nr-lowband-access"
    hotspot_count: int = 0
    hotspot_profile_id: str | None = None
    # Satellite-IoT scenario (config/zones.py). When set, ground BS/MEC are
    # placed only inside ``access_zone``, UAV-BS cells only inside ``uav_zone``,
    # and ``ntn_enabled`` adds the LEO satellite + gateway layer.
    access_zone: str | None = None
    uav_zone: str | None = None
    ntn_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("profile_id must be a non-empty string")
        if (
            isinstance(self.ground_bs_count, bool)
            or not isinstance(self.ground_bs_count, int)
            or self.ground_bs_count < 0
        ):
            raise ValueError("ground_bs_count must be a non-negative integer")
        if not isinstance(self.ground_bs_profile_id, str) or not self.ground_bs_profile_id.strip():
            raise ValueError("ground_bs_profile_id must be a non-empty string")
        if not isinstance(self.uav_profile_id, str) or not self.uav_profile_id.strip():
            raise ValueError("uav_profile_id must be a non-empty string")
        if not isinstance(self.anchor_profile_id, str) or not self.anchor_profile_id.strip():
            raise ValueError("anchor_profile_id must be a non-empty string")
        if (
            isinstance(self.ground_bs_height_m, bool)
            or not isinstance(self.ground_bs_height_m, (int, float))
            or not isfinite(float(self.ground_bs_height_m))
            or float(self.ground_bs_height_m) < 0.0
        ):
            raise ValueError("ground_bs_height_m must be finite and non-negative")
        if self.placement_strategy != "grid":
            raise ValueError("the initial deployment builder supports only 'grid'")
        if (
            isinstance(self.hotspot_count, bool)
            or not isinstance(self.hotspot_count, int)
            or self.hotspot_count < 0
        ):
            raise ValueError("hotspot_count must be a non-negative integer")
        has_hotspots = self.hotspot_count > 0
        has_hotspot_profile = (
            self.hotspot_profile_id is not None
            and isinstance(self.hotspot_profile_id, str)
            and self.hotspot_profile_id.strip() != ""
        )
        if has_hotspots != has_hotspot_profile:
            raise ValueError(
                "hotspot_count and hotspot_profile_id must be set together "
                "(a hotspot overlay needs both a count and a band, or neither)"
            )
        object.__setattr__(
            self,
            "access_generation",
            normalize_access_generation(self.access_generation),
        )


DEPLOYMENT_PROFILES: Mapping[str, DeploymentProfile] = MappingProxyType(
    {
        # Preserves the v1.1/v1.2 topology and is the safe regression default.
        "legacy_sparse": DeploymentProfile(
            profile_id="legacy_sparse",
            ground_bs_count=0,
        ),
        # Conservative ideal-grid starting point for the current FR1 link budget
        # over the 10 km x 10 km city.  It is a scenario, not a claim that every
        # production network requires exactly 81 sites.
        "fr1_city_coverage": DeploymentProfile(
            profile_id="fr1_city_coverage",
            ground_bs_count=81,
        ),
        # A smaller, intentionally partial-coverage profile useful for density
        # sensitivity studies without constructing the full city grid.
        "fr1_density_sweep_16": DeploymentProfile(
            profile_id="fr1_density_sweep_16",
            ground_bs_count=16,
        ),
        # SLA sensitivity profile at a 16-site density, now wired to a real
        # FR3-band 6G access profile (ground and aerial) instead of inheriting
        # the FR1 defaults.
        "experimental_6g_sla_16": DeploymentProfile(
            profile_id="experimental_6g_sla_16",
            ground_bs_count=16,
            ground_bs_profile_id="nr-fr3-access",
            uav_profile_id="nr-mmwave-air",
            access_generation="6G",
        ),
        # Mirrors "fr1_city_coverage" at the same 81-site density, but on a real
        # 6G FR3 access profile, so the two can be run back-to-back for a direct
        # 5G-vs-6G comparison at equal site density.
        "fr3_city_coverage_6g": DeploymentProfile(
            profile_id="fr3_city_coverage_6g",
            ground_bs_count=81,
            ground_bs_profile_id="nr-fr3-access",
            uav_profile_id="nr-mmwave-air",
            access_generation="6G",
        ),
        # FR3's own sensitivity-limited link budget (metrics/handover_channel.py
        # max_coverage_radius_m) implies an inter-site distance of ~322 m for
        # gap-free coverage of the 10 km x 10 km city -- ~964 sites, not 81.
        # "fr3_city_coverage_6g" deliberately holds site count equal to FR1 to
        # isolate the band effect (a controlled comparison); this profile is
        # the honest, formula-matched density for an actual "6G macro coverage"
        # claim. Site count is a decision, not a physics constant -- see
        # llm_collab/MESSAGES.md for the derivation.
        "fr3_city_coverage_realistic": DeploymentProfile(
            profile_id="fr3_city_coverage_realistic",
            ground_bs_count=964,
            ground_bs_profile_id="nr-fr3-access",
            uav_profile_id="nr-mmwave-air",
            access_generation="6G",
        ),
        # mmWave's own formula-ideal macro density (~13,177 sites/100km^2) is not
        # a credible blanket deployment -- real mmWave is deployed as a sparse
        # small-cell overlay on top of wide-area mid/low-band coverage, not a
        # replacement for it. This profile keeps "fr1_city_coverage"'s 81-site
        # FR1 macro grid (already realistically sized -- see the FR1 profile's
        # own comment) and scatters a small mmWave hotspot layer on top of it,
        # matching real-world small-cell placement practice.
        "mmwave_hotspot_overlay": DeploymentProfile(
            profile_id="mmwave_hotspot_overlay",
            ground_bs_count=81,
            ground_bs_profile_id="nr-fr1-access",
            uav_profile_id="nr-fr1-air",
            access_generation="6G",
            hotspot_count=12,
            hotspot_profile_id="nr-mmwave-access",
        ),
        # Same reasoning as "mmwave_hotspot_overlay": subTHz's formula-ideal
        # macro density (~255,000 sites/100km^2) only makes sense as a sparse
        # small-cell/hotspot overlay, never blanket coverage. Same FR1 macro
        # base, same hotspot count, so the two overlay profiles are directly
        # comparable band-for-band.
        "subthz_hotspot_overlay": DeploymentProfile(
            profile_id="subthz_hotspot_overlay",
            ground_bs_count=81,
            ground_bs_profile_id="nr-fr1-access",
            uav_profile_id="nr-fr1-air",
            access_generation="6G",
            hotspot_count=12,
            hotspot_profile_id="nr-subthz-access",
        ),
        # Satellite-enabled IoT scenario: the same FR3/6G macro density as
        # "fr3_city_coverage_6g" (81 sites / 100 km^2 -> ~32 sites on the 40 km^2
        # cellular zone), UAV-BS cells over the UAV zone, LEO satellite overhead.
        "zoned_satellite_6g": DeploymentProfile(
            profile_id="zoned_satellite_6g",
            ground_bs_count=32,
            ground_bs_profile_id="nr-fr3-access",
            uav_profile_id="nr-fr1-a2g",
            access_generation="6G",
            access_zone="cellular",
            uav_zone="uav",
            ntn_enabled=True,
        ),
    }
)


def get_deployment_profile(name: str) -> DeploymentProfile:
    """Return one explicitly named, validated deployment profile."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("deployment profile name must be a non-empty string")
    key = name.strip()
    try:
        return DEPLOYMENT_PROFILES[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown deployment profile {key!r}; "
            f"valid profiles: {sorted(DEPLOYMENT_PROFILES)}"
        ) from exc
