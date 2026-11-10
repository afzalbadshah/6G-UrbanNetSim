"""Represent and validate an immutable physical topology in metres.

How to extend: add nodes, links, and explicit routes in ``config/topology.py``;
use ``PhysicalTopology.from_mapping`` to validate them before a run. Do not add
route-search policy or mutable queue state here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isclose, isfinite, sqrt
from types import MappingProxyType
from typing import Mapping

from network.contracts import Direction, Hop


def _nonempty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _nonnegative(name: str, value: float) -> float:
    numeric = _finite(name, value)
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


class MaturityLevel(str, Enum):
    """Evidence maturity attached to a topology assumption."""

    VALIDATED = "validated"
    STANDARDS_ALIGNED = "standards-aligned"
    EXPERIMENTAL = "experimental"
    VISIONARY = "visionary"


class NodeKind(str, Enum):
    """Physical role of a topology node."""

    DEVICE = "device"
    ACCESS_TOWER = "access_tower"
    MEC = "mec"
    METRO = "metro"
    CLOUD_GATEWAY = "cloud_gateway"
    UAV_BS = "uav_bs"
    NTN = "ntn"
    SAT_GATEWAY = "sat_gateway"


class Medium(str, Enum):
    """Radio or wired physical link medium."""

    RADIO = "radio"
    WIRED = "wired"


@dataclass(frozen=True, slots=True)
class Provenance:
    """Source, validity, and maturity metadata for one configured assumption."""

    source_id: str
    source_uri: str
    source_version: str
    valid_from: str | None
    valid_to: str | None
    validity_notes: str
    maturity: MaturityLevel

    def __post_init__(self) -> None:
        _nonempty("source_id", self.source_id)
        for name in ("source_uri", "source_version", "validity_notes"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")
        for name in ("valid_from", "valid_to"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or None")
        if not isinstance(self.maturity, MaturityLevel):
            raise ValueError("maturity must be a MaturityLevel")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "Provenance":
        """Build validated provenance from configuration data."""

        try:
            return cls(
                source_id=raw["source_id"],
                source_uri=raw.get("source_uri", ""),
                source_version=raw.get("source_version", ""),
                valid_from=raw.get("valid_from"),
                valid_to=raw.get("valid_to"),
                validity_notes=raw.get("validity_notes", ""),
                maturity=MaturityLevel(raw["maturity"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid provenance record: {exc}") from exc


@dataclass(frozen=True, slots=True)
class Position3D:
    """Cartesian position in metres within the topology coordinate frame."""

    x_m: float
    y_m: float
    z_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_m", _finite("x_m", self.x_m))
        object.__setattr__(self, "y_m", _finite("y_m", self.y_m))
        object.__setattr__(self, "z_m", _nonnegative("z_m", self.z_m))

    def distance_to(self, other: "Position3D") -> float:
        """Return three-dimensional Euclidean distance in metres."""

        if not isinstance(other, Position3D):
            raise ValueError("other must be a Position3D")
        return sqrt(
            (self.x_m - other.x_m) ** 2
            + (self.y_m - other.y_m) ** 2
            + (self.z_m - other.z_m) ** 2
        )


@dataclass(frozen=True, slots=True)
class Extent2D:
    """Horizontal topology bounds in metres."""

    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float

    def __post_init__(self) -> None:
        for name in ("x_min_m", "x_max_m", "y_min_m", "y_max_m"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.x_max_m <= self.x_min_m or self.y_max_m <= self.y_min_m:
            raise ValueError("topology extent maxima must exceed minima")

    def contains(self, position: Position3D) -> bool:
        """Return whether a position lies inside the horizontal bounds."""

        return (
            self.x_min_m <= position.x_m <= self.x_max_m
            and self.y_min_m <= position.y_m <= self.y_max_m
        )


@dataclass(frozen=True, slots=True)
class CoordinateFrame:
    """Coordinate-system identity, units, z-axis meaning, and extent."""

    frame_type: str
    unit: str
    z_meaning: str
    extent: Extent2D
    provenance: Provenance

    def __post_init__(self) -> None:
        _nonempty("frame_type", self.frame_type)
        if self.unit != "m":
            raise ValueError("coordinate frame unit must be canonical metres ('m')")
        _nonempty("z_meaning", self.z_meaning)


@dataclass(frozen=True, slots=True)
class TopologyNode:
    """Immutable infrastructure/device anchor with a position and profile."""

    node_id: str
    kind: NodeKind
    position: Position3D
    profile_id: str
    provenance: Provenance
    anchor_profile_id: str | None = None
    provides_radio_access: bool = False

    def __post_init__(self) -> None:
        _nonempty("node_id", self.node_id)
        _nonempty("profile_id", self.profile_id)
        if self.anchor_profile_id is not None and (
            not isinstance(self.anchor_profile_id, str) or not self.anchor_profile_id.strip()
        ):
            raise ValueError("anchor_profile_id must be a non-empty string when provided")
        if not isinstance(self.provides_radio_access, bool):
            raise ValueError("provides_radio_access must be a bool")
        if not isinstance(self.kind, NodeKind):
            raise ValueError("kind must be a NodeKind")


@dataclass(frozen=True, slots=True)
class TopologyLink:
    """Immutable directed physical link with medium and route distance."""

    link_id: str
    source_node_id: str
    destination_node_id: str
    link_profile_id: str
    route_distance_m: float
    medium: Medium
    direction: Direction
    provenance: Provenance

    def __post_init__(self) -> None:
        for name in (
            "link_id",
            "source_node_id",
            "destination_node_id",
            "link_profile_id",
        ):
            _nonempty(name, getattr(self, name))
        object.__setattr__(
            self,
            "route_distance_m",
            _nonnegative("route_distance_m", self.route_distance_m),
        )
        if self.source_node_id == self.destination_node_id:
            raise ValueError("a topology link cannot connect a node to itself")
        if not isinstance(self.medium, Medium):
            raise ValueError("medium must be a Medium")
        if not isinstance(self.direction, Direction):
            raise ValueError("direction must be a Direction")

    def to_hop(self) -> Hop:
        """Return the link's logical packet-engine hop."""

        return Hop(
            hop_id=self.link_id,
            source_node_id=self.source_node_id,
            destination_node_id=self.destination_node_id,
            link_profile_id=self.link_profile_id,
            direction=self.direction,
        )


@dataclass(frozen=True, slots=True)
class Route:
    """Explicit ordered link IDs between one source and destination."""

    route_id: str
    source_node_id: str
    destination_node_id: str
    hop_ids: tuple[str, ...]
    provenance: Provenance

    def __post_init__(self) -> None:
        for name in ("route_id", "source_node_id", "destination_node_id"):
            _nonempty(name, getattr(self, name))
        if not self.hop_ids:
            raise ValueError("route must contain at least one hop")
        for hop_id in self.hop_ids:
            _nonempty("hop_id", hop_id)


def _direction_for(
    raw: Mapping[str, object],
    medium: Medium,
    source: TopologyNode,
    destination: TopologyNode,
) -> Direction:
    configured = raw.get("direction")
    if configured is not None:
        try:
            return Direction(configured)
        except ValueError as exc:
            raise ValueError(f"invalid topology link direction {configured!r}") from exc
    if medium is Medium.WIRED:
        return Direction.BACKHAUL
    if source.kind is NodeKind.DEVICE:
        return Direction.UPLINK
    if destination.kind is NodeKind.DEVICE:
        return Direction.DOWNLINK
    return Direction.ACCESS


@dataclass(frozen=True, slots=True)
class PhysicalTopology:
    """Validated immutable nodes, links, and explicit routes for one profile."""

    profile_id: str
    coordinate_frame: CoordinateFrame
    nodes: tuple[TopologyNode, ...]
    links: tuple[TopologyLink, ...]
    routes: tuple[Route, ...]
    _node_by_id: Mapping[str, TopologyNode] = field(init=False, repr=False, compare=False)
    _link_by_id: Mapping[str, TopologyLink] = field(init=False, repr=False, compare=False)
    _route_by_id: Mapping[str, Route] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _nonempty("profile_id", self.profile_id)
        node_by_id = self._unique("node", self.nodes, lambda item: item.node_id)
        link_by_id = self._unique("link", self.links, lambda item: item.link_id)
        route_by_id = self._unique("route", self.routes, lambda item: item.route_id)
        object.__setattr__(self, "_node_by_id", MappingProxyType(node_by_id))
        object.__setattr__(self, "_link_by_id", MappingProxyType(link_by_id))
        object.__setattr__(self, "_route_by_id", MappingProxyType(route_by_id))

        for node in self.nodes:
            if not self.coordinate_frame.extent.contains(node.position):
                raise ValueError(f"node {node.node_id!r} lies outside the coordinate extent")
        for link in self.links:
            if link.source_node_id not in node_by_id or link.destination_node_id not in node_by_id:
                raise ValueError(f"link {link.link_id!r} references an unknown node")
            source = node_by_id[link.source_node_id]
            destination = node_by_id[link.destination_node_id]
            if link.medium is Medium.RADIO and (
                source.kind is NodeKind.CLOUD_GATEWAY
                or destination.kind is NodeKind.CLOUD_GATEWAY
            ):
                raise ValueError("Cloud gateway cannot be incident to a radio link")
            if link.medium is Medium.RADIO:
                geometric_m = source.position.distance_to(destination.position)
                if not isclose(
                    link.route_distance_m,
                    geometric_m,
                    rel_tol=1e-6,
                    abs_tol=1e-3,
                ):
                    raise ValueError(
                        f"radio link {link.link_id!r} route distance disagrees with initial 3-D geometry"
                    )
        for route in self.routes:
            try:
                route_links = [link_by_id[hop_id] for hop_id in route.hop_ids]
            except KeyError as exc:
                raise ValueError(
                    f"route {route.route_id!r} references unknown hop {exc.args[0]!r}"
                ) from exc
            if route_links[0].source_node_id != route.source_node_id:
                raise ValueError(f"route {route.route_id!r} starts at the wrong node")
            if route_links[-1].destination_node_id != route.destination_node_id:
                raise ValueError(f"route {route.route_id!r} ends at the wrong node")
            for previous, following in zip(route_links, route_links[1:]):
                if previous.destination_node_id != following.source_node_id:
                    raise ValueError(f"route {route.route_id!r} is not contiguous")

    @staticmethod
    def _unique(kind: str, values: tuple, key):
        result = {}
        for value in values:
            identifier = key(value)
            if identifier in result:
                raise ValueError(f"duplicate {kind} ID {identifier!r}")
            result[identifier] = value
        return result

    @classmethod
    def from_mapping(
        cls,
        profile_id: str,
        raw: Mapping[str, object],
    ) -> "PhysicalTopology":
        """Build and fully validate a topology from configuration data."""

        try:
            frame_raw = raw["coordinate_frame"]
            extent_raw = frame_raw["extent_m"]
            frame = CoordinateFrame(
                frame_type=frame_raw["type"],
                unit=frame_raw["unit"],
                z_meaning=frame_raw["z_meaning"],
                extent=Extent2D(
                    x_min_m=extent_raw["x_min"],
                    x_max_m=extent_raw["x_max"],
                    y_min_m=extent_raw["y_min"],
                    y_max_m=extent_raw["y_max"],
                ),
                provenance=Provenance.from_mapping(frame_raw["provenance"]),
            )
            nodes = tuple(
                TopologyNode(
                    node_id=item["node_id"],
                    kind=NodeKind(item["node_kind"]),
                    position=Position3D(item["x_m"], item["y_m"], item["z_m"]),
                    profile_id=item["profile_id"],
                    provenance=Provenance.from_mapping(item["provenance"]),
                    anchor_profile_id=item.get("anchor_profile_id"),
                    provides_radio_access=bool(item.get("provides_radio_access", False)),
                )
                for item in raw["nodes"]
            )
            node_by_id = {item.node_id: item for item in nodes}
            links = tuple(
                cls._link_from_mapping(item, node_by_id)
                for item in raw["links"]
            )
            routes = tuple(
                Route(
                    route_id=item["route_id"],
                    source_node_id=item["source_node_id"],
                    destination_node_id=item["destination_node_id"],
                    hop_ids=tuple(item["hop_ids"]),
                    provenance=Provenance.from_mapping(item["provenance"]),
                )
                for item in raw["routes"]
            )
            return cls(
                profile_id=profile_id,
                coordinate_frame=frame,
                nodes=nodes,
                links=links,
                routes=routes,
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith(("duplicate ", "node ", "link ", "route ", "Cloud ")):
                raise
            raise ValueError(f"invalid topology profile {profile_id!r}: {exc}") from exc

    @staticmethod
    def _link_from_mapping(
        item: Mapping[str, object],
        node_by_id: Mapping[str, TopologyNode],
    ) -> TopologyLink:
        source_id = item["source_node_id"]
        destination_id = item["destination_node_id"]
        if source_id not in node_by_id or destination_id not in node_by_id:
            raise ValueError(f"link {item.get('link_id')!r} references an unknown node")
        medium = Medium(item["medium"])
        return TopologyLink(
            link_id=item["link_id"],
            source_node_id=source_id,
            destination_node_id=destination_id,
            link_profile_id=item["link_profile_id"],
            route_distance_m=item["route_distance_m"],
            medium=medium,
            direction=_direction_for(
                item,
                medium,
                node_by_id[source_id],
                node_by_id[destination_id],
            ),
            provenance=Provenance.from_mapping(item["provenance"]),
        )

    def node(self, node_id: str) -> TopologyNode:
        """Return one node by ID or raise ``KeyError``."""

        try:
            return self._node_by_id[node_id]
        except KeyError as exc:
            raise KeyError(f"unknown topology node {node_id!r}") from exc

    def link(self, link_id: str) -> TopologyLink:
        """Return one physical link by ID or raise ``KeyError``."""

        try:
            return self._link_by_id[link_id]
        except KeyError as exc:
            raise KeyError(f"unknown topology link {link_id!r}") from exc

    def route(self, route_id: str) -> Route:
        """Return one explicit route by ID or raise ``KeyError``."""

        try:
            return self._route_by_id[route_id]
        except KeyError as exc:
            raise KeyError(f"unknown topology route {route_id!r}") from exc

    def link_distance_m(
        self,
        link_id: str,
        *,
        position_overrides: Mapping[str, Position3D] | None = None,
    ) -> float:
        """Return wired route distance or live 3-D radio distance in metres."""

        link = self.link(link_id)
        overrides = dict(position_overrides or {})
        for node_id, position in overrides.items():
            if node_id not in self._node_by_id:
                raise KeyError(f"position override references unknown node {node_id!r}")
            if not isinstance(position, Position3D):
                raise ValueError("position overrides must contain Position3D values")
        if link.medium is Medium.WIRED:
            return link.route_distance_m
        source = overrides.get(link.source_node_id, self.node(link.source_node_id).position)
        destination = overrides.get(
            link.destination_node_id,
            self.node(link.destination_node_id).position,
        )
        return source.distance_to(destination)
