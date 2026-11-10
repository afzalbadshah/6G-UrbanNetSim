"""Resolve declared routes and calculate their physical hop distances.

How to extend: declare a new route in ``config/topology.py``. This resolver is
intentionally not a shortest-path algorithm; absent routes fail closed so a
simulation cannot invent connectivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from network.contracts import Hop
from network.topology import Medium, PhysicalTopology, Position3D, TopologyLink


@dataclass(frozen=True, slots=True)
class ResolvedHop:
    """One declared hop paired with its physical link and distance in metres."""

    hop: Hop
    link: TopologyLink
    distance_m: float

    def __post_init__(self) -> None:
        if self.hop.hop_id != self.link.link_id:
            raise ValueError("resolved hop and topology link identities must match")
        if self.distance_m < 0.0:
            raise ValueError("resolved hop distance must be non-negative")


@dataclass(frozen=True, slots=True)
class ResolvedRoute:
    """Ordered, contiguous physical hops from source to destination."""

    route_id: str
    source_node_id: str
    destination_node_id: str
    hops: tuple[ResolvedHop, ...]

    def __post_init__(self) -> None:
        if not self.hops:
            raise ValueError("resolved route must contain at least one hop")

    @property
    def total_distance_m(self) -> float:
        """Return the full route distance in metres."""

        return sum(item.distance_m for item in self.hops)

    @property
    def radio_distance_m(self) -> float:
        """Return the sum of radio-hop distances in metres."""

        return sum(item.distance_m for item in self.hops if item.link.medium is Medium.RADIO)

    @property
    def wired_distance_m(self) -> float:
        """Return the sum of wired-hop distances in metres."""

        return sum(item.distance_m for item in self.hops if item.link.medium is Medium.WIRED)


class RouteResolver:
    """Resolve configured routes; no hidden shortest-path or tier-distance proxy."""

    def __init__(self, topology: PhysicalTopology) -> None:
        if not isinstance(topology, PhysicalTopology):
            raise ValueError("topology must be a PhysicalTopology")
        self._topology = topology

    def resolve(
        self,
        route_id: str,
        *,
        position_overrides: Mapping[str, Position3D] | None = None,
    ) -> ResolvedRoute:
        """Resolve one named route with optional live position overrides."""

        route = self._topology.route(route_id)
        hops = tuple(
            ResolvedHop(
                hop=self._topology.link(link_id).to_hop(),
                link=self._topology.link(link_id),
                distance_m=self._topology.link_distance_m(
                    link_id,
                    position_overrides=position_overrides,
                ),
            )
            for link_id in route.hop_ids
        )
        return ResolvedRoute(
            route_id=route.route_id,
            source_node_id=route.source_node_id,
            destination_node_id=route.destination_node_id,
            hops=hops,
        )

    def resolve_between(
        self,
        source_node_id: str,
        destination_node_id: str,
        *,
        position_overrides: Mapping[str, Position3D] | None = None,
    ) -> ResolvedRoute:
        """Resolve the single declared route between two nodes; fail if absent."""

        candidates = [
            route
            for route in self._topology.routes
            if route.source_node_id == source_node_id
            and route.destination_node_id == destination_node_id
        ]
        if not candidates:
            raise KeyError(
                f"no explicit route from {source_node_id!r} to {destination_node_id!r}"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"multiple explicit routes from {source_node_id!r} to {destination_node_id!r}; "
                "select a route_id"
            )
        return self.resolve(
            candidates[0].route_id,
            position_overrides=position_overrides,
        )
