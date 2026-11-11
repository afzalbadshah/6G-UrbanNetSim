"""Assemble the physical packet-network services used by the live simulator.

This module creates one shared queue/channel/energy model per physical resource,
adds a device's live access hop to an explicitly declared backbone route, and
derives per-hop RF inputs from endpoint profiles and three-dimensional geometry.
It never chooses an offloading action or invents a shortest path.

How to extend: add topology/routes in ``config/topology.py``, link behavior in
``config/radio.py``, and endpoint power in ``config/energy.py``. Keep
those tunable values out of these composition functions. The full relationship
map is in ``docs/PACKET_EXECUTION_GUIDE.md``.
"""

from __future__ import annotations

from math import atan2, degrees, hypot, isfinite
from typing import Mapping

from config import radio as radio_config
from config.energy import POWER_PROFILES
from metrics.channel import SixGChannelModel
from metrics.energy import RadioPacketEnergyModel, TransportPacketEnergyModel

from network import (
    ChannelModel,
    ChannelRequest,
    Direction,
    FixedMtuPacketizer,
    FixedRateChannelModel,
    Hop,
    HopEnergyModel,
    HopTransmissionParameters,
    Medium,
    NetworkEngine,
    NodeKind,
    PhysicalTopology,
    Position3D,
    ResolvedHop,
    ResolvedRoute,
    RetryPolicy,
    RouteResolver,
    SharedLink,
    TopologyLink,
    TopologyNode,
    TransmissionLedger,
    instantiate_link_profile,
)


_ACCESS_NODE_KINDS = frozenset(
    {NodeKind.ACCESS_TOWER, NodeKind.MEC, NodeKind.UAV_BS}
)

_RADIO_POWER_PROFILE_BY_NODE_KIND = {
    NodeKind.DEVICE: "device_radio_default",
    NodeKind.ACCESS_TOWER: "access_radio_default",
    NodeKind.MEC: "access_radio_default",
    NodeKind.UAV_BS: "uav_bs_radio_default",
}


def access_resource_id(access_node_id: str, *, tier: str = "main") -> str:
    """Return the stable shared-resource ID for one radio access point.

    ``tier`` selects which band's resource ID to return: ``"main"`` (the
    default) is the node's primary ``profile_id`` band and is the only tier
    that existed before the anchor fallback -- its output is unchanged from
    before. ``"anchor"`` is the node's low-band ``anchor_profile_id`` fallback
    layer and is registered as a distinct resource so it never shares a FIFO
    queue or channel model with the main band.
    """

    if not isinstance(access_node_id, str) or not access_node_id.strip():
        raise ValueError("access_node_id must be a non-empty string")
    if tier not in ("main", "anchor"):
        raise ValueError("tier must be 'main' or 'anchor'")
    if tier == "main":
        return f"radio-access:{access_node_id}"
    return f"radio-access:anchor:{access_node_id}"


def anchor_resource_id(access_node_id: str) -> str:
    """Return the stable shared-resource ID for one access point's anchor tier."""

    return access_resource_id(access_node_id, tier="anchor")


def build_shared_link_registry(
    topology: PhysicalTopology,
) -> dict[str, SharedLink]:
    """Create one mutable FIFO service resource per physical link and access AP.

    The topology is immutable; the returned ``SharedLink`` instances hold the
    per-run queue state. Invalid topology/resource identities raise ``ValueError``.
    """

    if not isinstance(topology, PhysicalTopology):
        raise ValueError("topology must be a PhysicalTopology")
    registry: dict[str, SharedLink] = {}
    for link in topology.links:
        registry[link.link_id] = SharedLink(
            instantiate_link_profile(
                link.link_profile_id,
                shared_resource_id=link.link_id,
            )
        )
    for node in topology.nodes:
        if node.kind not in _ACCESS_NODE_KINDS:
            continue
        resource_id = access_resource_id(node.node_id)
        if resource_id in registry:
            raise ValueError(f"duplicate runtime resource ID {resource_id!r}")
        registry[resource_id] = SharedLink(
            instantiate_link_profile(
                node.profile_id,
                shared_resource_id=resource_id,
            )
        )
        if (
            node.anchor_profile_id is not None
            and node.anchor_profile_id != node.profile_id
        ):
            anchor_id = anchor_resource_id(node.node_id)
            if anchor_id in registry:
                raise ValueError(f"duplicate runtime resource ID {anchor_id!r}")
            registry[anchor_id] = SharedLink(
                instantiate_link_profile(
                    node.anchor_profile_id,
                    shared_resource_id=anchor_id,
                )
            )
    return registry


def build_channel_model_registry(
    topology: PhysicalTopology,
    shared_links: Mapping[str, SharedLink],
) -> dict[str, ChannelModel]:
    """Create channel providers and index them by link-profile ID.

    Radio profiles share one ``SixGChannelModel`` snapshot and wired profiles
    share one fixed-rate provider. A profile cannot be both radio and wired.
    """

    if not isinstance(topology, PhysicalTopology):
        raise ValueError("topology must be a PhysicalTopology")
    if not shared_links:
        raise ValueError("shared_links must not be empty")
    profiles = {}
    for resource_id, link in shared_links.items():
        if not isinstance(resource_id, str) or not resource_id.strip():
            raise ValueError("shared-link resource IDs must be non-empty strings")
        if not isinstance(link, SharedLink):
            raise ValueError("shared_links values must be SharedLink objects")
        profiles.setdefault(link.profile.profile_id, link.profile)

    radio_profile_ids = {
        link.link_profile_id
        for link in topology.links
        if link.medium is Medium.RADIO
    }
    radio_profile_ids.update(
        node.profile_id
        for node in topology.nodes
        if node.kind in _ACCESS_NODE_KINDS
    )
    radio_profile_ids.update(
        node.anchor_profile_id
        for node in topology.nodes
        if node.kind in _ACCESS_NODE_KINDS and node.anchor_profile_id is not None
    )
    wired_profile_ids = {
        link.link_profile_id
        for link in topology.links
        if link.medium is Medium.WIRED
    }
    overlap = radio_profile_ids & wired_profile_ids
    if overlap:
        raise ValueError(f"link profiles cannot be both radio and wired: {sorted(overlap)}")
    unclassified = set(profiles) - radio_profile_ids - wired_profile_ids
    if unclassified:
        raise ValueError(f"unclassified runtime link profiles: {sorted(unclassified)}")

    registry: dict[str, ChannelModel] = {}
    if radio_profile_ids:
        radio_profiles = {profile_id: profiles[profile_id] for profile_id in radio_profile_ids}
        first_radio = radio_profiles[sorted(radio_profiles)[0]]
        radio_model = SixGChannelModel(first_radio, profiles=radio_profiles)
        registry.update({profile_id: radio_model for profile_id in radio_profile_ids})
    if wired_profile_ids:
        wired_profiles = {profile_id: profiles[profile_id] for profile_id in wired_profile_ids}
        first_wired = wired_profiles[sorted(wired_profiles)[0]]
        wired_model = FixedRateChannelModel(first_wired, profiles=wired_profiles)
        registry.update({profile_id: wired_model for profile_id in wired_profile_ids})
    return registry


def build_energy_model_registry(
    topology: PhysicalTopology,
    shared_links: Mapping[str, SharedLink],
    *,
    power_profiles: Mapping[str, Mapping[str, object]] = POWER_PROFILES,
) -> dict[str, HopEnergyModel]:
    """Pair every physical resource with its endpoint energy model.

    Radio resources receive explicit transmitter/receiver profiles; wired
    resources receive the configured transport basis. Missing endpoint mappings
    or profiles raise ``ValueError`` before a run starts.
    """

    if not isinstance(topology, PhysicalTopology):
        raise ValueError("topology must be a PhysicalTopology")
    if not isinstance(power_profiles, Mapping):
        raise ValueError("power_profiles must be a mapping")
    required = {
        "device_radio_default",
        "access_radio_default",
        "uav_bs_radio_default",
        "transport_fibre_default",
    }
    missing = sorted(required - set(power_profiles))
    if missing:
        raise ValueError(f"missing runtime power profiles: {missing}")

    radio_by_kind = {
        kind: power_profiles[profile_id]
        for kind, profile_id in _RADIO_POWER_PROFILE_BY_NODE_KIND.items()
    }
    topology_links = {link.link_id: link for link in topology.links}
    access_nodes: dict[str, TopologyNode] = {}
    for node in topology.nodes:
        if node.kind not in _ACCESS_NODE_KINDS:
            continue
        access_nodes[access_resource_id(node.node_id)] = node
        if (
            node.anchor_profile_id is not None
            and node.anchor_profile_id != node.profile_id
        ):
            access_nodes[anchor_resource_id(node.node_id)] = node
    models: dict[str, HopEnergyModel] = {}
    for resource_id in shared_links:
        if resource_id in access_nodes:
            receiver = access_nodes[resource_id]
            models[resource_id] = RadioPacketEnergyModel(
                radio_by_kind[NodeKind.DEVICE],
                radio_by_kind[receiver.kind],
            )
            continue
        try:
            topology_link = topology_links[resource_id]
        except KeyError as exc:
            raise ValueError(f"resource {resource_id!r} has no physical endpoint mapping") from exc
        if topology_link.medium is Medium.WIRED:
            models[resource_id] = TransportPacketEnergyModel(
                power_profiles["transport_fibre_default"]
            )
            continue
        source = topology.node(topology_link.source_node_id)
        destination = topology.node(topology_link.destination_node_id)
        try:
            transmitter_profile = radio_by_kind[source.kind]
            receiver_profile = radio_by_kind[destination.kind]
        except KeyError as exc:
            raise ValueError(
                f"no radio power profile for topology node kind {exc.args[0].value!r}"
            ) from exc
        models[resource_id] = RadioPacketEnergyModel(
            transmitter_profile,
            receiver_profile,
        )
    return models


class RuntimeHopParameterBuilder:
    """Derive channel-request inputs from endpoint profiles and live geometry.

    The legacy device ``tx_power_w`` field is intentionally not accepted here:
    it represented an energy-draw parameter and can exceed the UE's physical RF
    maximum. Radiated power comes only from the selected endpoint radio profile.
    Output power is watts, bandwidth is hertz, interference is dBm, and elevation
    is degrees.
    """

    def __init__(
        self,
        topology: PhysicalTopology,
        *,
        power_profiles: Mapping[str, Mapping[str, object]] = POWER_PROFILES,
    ) -> None:
        if not isinstance(topology, PhysicalTopology):
            raise ValueError("topology must be a PhysicalTopology")
        if not isinstance(power_profiles, Mapping):
            raise ValueError("power_profiles must be a mapping")
        required = set(_RADIO_POWER_PROFILE_BY_NODE_KIND.values())
        missing = sorted(required - set(power_profiles))
        if missing:
            raise ValueError(f"missing radio power profiles: {missing}")
        self._topology = topology
        self._power_profiles = {
            profile_id: dict(power_profiles[profile_id])
            for profile_id in required
        }
        for profile_id, profile in self._power_profiles.items():
            if profile.get("kind") != "radio":
                raise ValueError(f"power profile {profile_id!r} must be a radio profile")
            default_w = self._finite_nonnegative(
                f"{profile_id}.radiated_tx_power_w",
                profile.get("radiated_tx_power_w"),
            )
            maximum_w = self._finite_nonnegative(
                f"{profile_id}.radiated_tx_power_max_w",
                profile.get("radiated_tx_power_max_w"),
            )
            if default_w > maximum_w:
                raise ValueError(
                    f"{profile_id}.radiated_tx_power_w exceeds its maximum"
                )

    @staticmethod
    def _finite_nonnegative(name: str, value: object) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value < 0.0
        ):
            raise ValueError(f"{name} must be a finite non-negative number")
        return float(value)

    @staticmethod
    def _optional_finite(name: str, value: object | None) -> float | None:
        if value is None or value == "":
            return None
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
        ):
            raise ValueError(f"{name} must be finite when provided")
        return float(value)

    def _position(
        self,
        node_id: str,
        *,
        source_node_id: str,
        source_position: Position3D,
        overrides: Mapping[str, Position3D],
    ) -> Position3D:
        if node_id == source_node_id:
            return source_position
        return overrides.get(node_id, self._topology.node(node_id).position)

    def _radio_tx_power_w(self, node_kind: NodeKind) -> float:
        try:
            profile_id = _RADIO_POWER_PROFILE_BY_NODE_KIND[node_kind]
        except KeyError as exc:
            raise ValueError(
                f"no runtime radio power profile for node kind {node_kind.value!r}"
            ) from exc
        return float(self._power_profiles[profile_id]["radiated_tx_power_w"])

    @staticmethod
    def _full_band_tdm_grant_hz(link_profile_id: str) -> float:
        """Return the full channel grant used by the current FIFO/TDM service.

        A queued packet owns the shared resource while it is serialized, so it
        receives the configured channel bandwidth. Contention appears as queue
        time; dividing this bandwidth by the queue length would count sharing
        twice. A future OFDMA scheduler may replace this explicit grant.
        """

        profile = radio_config.get_link_profile(link_profile_id)
        bandwidth_hz = float(
            radio_config.get_band(str(profile["band"]))["bandwidth_hz"]
        )
        if not isfinite(bandwidth_hz) or bandwidth_hz <= 0.0:
            raise ValueError(
                f"bandwidth for radio profile {link_profile_id!r} must be positive"
            )
        return bandwidth_hz

    def build(
        self,
        route: ResolvedRoute,
        *,
        source_position: Position3D,
        position_overrides: Mapping[str, Position3D] | None = None,
        access_interference_dbm: float | None = None,
        access_line_of_sight: bool | None = None,
    ) -> dict[str, HopTransmissionParameters]:
        """Build validated parameters for every hop in ``route``.

        ``source_position`` and overrides are metres in the topology frame.
        Optional access interference is dBm. Invalid or incomplete geometry,
        profiles, or duplicate hops raise ``ValueError``.
        """

        if not isinstance(route, ResolvedRoute):
            raise ValueError("route must be a ResolvedRoute")
        if not isinstance(source_position, Position3D):
            raise ValueError("source_position must be a Position3D")
        if access_line_of_sight is not None and not isinstance(
            access_line_of_sight, bool
        ):
            raise ValueError("access_line_of_sight must be a bool or None")
        interference_dbm = self._optional_finite(
            "access_interference_dbm", access_interference_dbm
        )
        overrides = dict(position_overrides or {})
        for node_id, position in overrides.items():
            self._topology.node(node_id)
            if not isinstance(position, Position3D):
                raise ValueError("position overrides must contain Position3D values")
            if not self._topology.coordinate_frame.extent.contains(position):
                raise ValueError(f"position override for {node_id!r} lies outside the extent")

        parameters: dict[str, HopTransmissionParameters] = {}
        for index, resolved_hop in enumerate(route.hops):
            hop = resolved_hop.hop
            if hop.hop_id in parameters:
                raise ValueError(f"route repeats hop identifier {hop.hop_id!r}")
            if resolved_hop.link.medium is Medium.WIRED:
                parameters[hop.hop_id] = HopTransmissionParameters(tx_power_w=0.0)
                continue

            if hop.source_node_id == route.source_node_id:
                source_kind = NodeKind.DEVICE
            else:
                source_kind = self._topology.node(hop.source_node_id).kind
            source = self._position(
                hop.source_node_id,
                source_node_id=route.source_node_id,
                source_position=source_position,
                overrides=overrides,
            )
            destination = self._position(
                hop.destination_node_id,
                source_node_id=route.source_node_id,
                source_position=source_position,
                overrides=overrides,
            )
            horizontal_m = hypot(destination.x_m - source.x_m, destination.y_m - source.y_m)
            vertical_m = abs(destination.z_m - source.z_m)
            elevation_deg = degrees(atan2(vertical_m, horizontal_m))
            is_access_hop = index == 0 and hop.hop_id.startswith("radio-access:")
            # The declared MEC<->UAV backhaul is an engineered directional
            # point-to-point radio path.  Treating its LoS state as unknown made
            # the urban channel proxy apply an almost entirely NLoS exponent to
            # every 2.8+ km hop, producing deterministic outage before UAV
            # compute could begin.  Access links still use their observed LoS
            # input; only this explicit backhaul profile is fixed LoS.
            is_directional_uav_backhaul = (
                resolved_hop.link.link_profile_id == "nr-mmwave-backhaul"
            )
            parameters[hop.hop_id] = HopTransmissionParameters(
                tx_power_w=self._radio_tx_power_w(source_kind),
                allocated_bandwidth_hz=self._full_band_tdm_grant_hz(
                    resolved_hop.link.link_profile_id
                ),
                interference_dbm=interference_dbm if is_access_hop else None,
                line_of_sight=(
                    True
                    if is_directional_uav_backhaul
                    else access_line_of_sight if is_access_hop else None
                ),
                elevation_angle_deg=elevation_deg,
            )
        if set(parameters) != {item.hop.hop_id for item in route.hops}:
            raise ValueError("runtime parameters must cover the route exactly")
        return parameters


def build_network_engine(
    topology: PhysicalTopology,
    *,
    experiment_seed: int | str = 0,
    retry_policy: RetryPolicy | None = None,
    ledger: TransmissionLedger | None = None,
    energy_models: Mapping[str, HopEnergyModel] | None = None,
    enable_energy_accounting: bool = True,
) -> NetworkEngine:
    """Compose a packet engine and all per-run physical resource state.

    ``experiment_seed`` controls reproducible packet-error draws. The returned
    engine owns shared queues and a ledger; callers should reuse it for the run.
    """

    if not isinstance(enable_energy_accounting, bool):
        raise ValueError("enable_energy_accounting must be a bool")
    shared_links = build_shared_link_registry(topology)
    if energy_models is not None and not enable_energy_accounting:
        raise ValueError("energy_models cannot be supplied when energy accounting is disabled")
    selected_energy_models = energy_models
    if selected_energy_models is None and enable_energy_accounting:
        selected_energy_models = build_energy_model_registry(topology, shared_links)
    return NetworkEngine(
        packetizer=FixedMtuPacketizer(),
        channel_model=build_channel_model_registry(topology, shared_links),
        shared_links=shared_links,
        retry_policy=retry_policy,
        experiment_seed=experiment_seed,
        ledger=ledger,
        energy_models=selected_energy_models,
    )


class DynamicAccessRouteBuilder:
    """Connect a live device to an explicitly declared infrastructure route.

    The first hop uses the device's current 3-D position. Remaining hops come
    only from ``RouteResolver``; no direct-to-Cloud or hidden shortest path is
    created.
    """

    def __init__(
        self,
        topology: PhysicalTopology,
        *,
        channel_models: Mapping[str, ChannelModel] | None = None,
    ) -> None:
        if not isinstance(topology, PhysicalTopology):
            raise ValueError("topology must be a PhysicalTopology")
        if channel_models is not None and not isinstance(channel_models, Mapping):
            raise ValueError("channel_models must be a mapping or None")
        self._topology = topology
        self._resolver = RouteResolver(topology)
        # Anchor-tier fallback probing is opt-in: existing callers that do not
        # pass channel_models get exactly the pre-existing behavior below (the
        # main band, unconditionally, with no extra ChannelModel.evaluate call).
        self._channel_models = channel_models

    def build(
        self,
        *,
        source_node_id: str,
        source_position: Position3D,
        access_node_id: str,
        destination_node_id: str,
        position_overrides: Mapping[str, Position3D] | None = None,
        access_interference_dbm: float | None = None,
    ) -> ResolvedRoute:
        """Return ``device -> access -> destination`` with distances in metres.

        Position overrides support moving infrastructure such as UAV base
        stations. Unknown nodes, invalid extents, or absent declared backbone
        routes raise ``ValueError``/``KeyError``.

        ``access_interference_dbm`` is the same live co-channel interference
        estimate the caller will use for the actual access-hop transmission
        (see ``RuntimeHopParameterBuilder.build``'s identically-named
        parameter). Passing it here lets the anchor-tier probe below judge
        the main band under the same conditions the real transmission will
        face; omitting it makes the probe implicitly assume an interference-
        free best case, which can miss main-band outages that only occur
        under real co-channel load.
        """

        if not isinstance(source_node_id, str) or not source_node_id.strip():
            raise ValueError("source_node_id must be a non-empty string")
        if not isinstance(source_position, Position3D):
            raise ValueError("source_position must be a Position3D")
        if not self._topology.coordinate_frame.extent.contains(source_position):
            raise ValueError("dynamic source lies outside the topology extent")
        try:
            self._topology.node(source_node_id)
        except KeyError:
            pass
        else:
            raise ValueError("dynamic source_node_id collides with a topology node")

        access_node = self._topology.node(access_node_id)
        destination_node = self._topology.node(destination_node_id)
        if access_node.kind not in _ACCESS_NODE_KINDS:
            raise ValueError(f"node {access_node_id!r} is not a radio access endpoint")
        if destination_node.kind is NodeKind.DEVICE:
            raise ValueError("uplink processing destination cannot be a device node")

        overrides = dict(position_overrides or {})
        for node_id, position in overrides.items():
            self._topology.node(node_id)
            if not isinstance(position, Position3D):
                raise ValueError("position overrides must contain Position3D values")
            if not self._topology.coordinate_frame.extent.contains(position):
                raise ValueError(f"position override for {node_id!r} lies outside the extent")
        access_position = overrides.get(access_node_id, access_node.position)
        access_id = access_resource_id(access_node_id)
        distance_m = source_position.distance_to(access_position)

        # Default (and, when channel_models was not supplied, the ONLY
        # possible) outcome: the main band, exactly as before this feature.
        link_profile_id = access_node.profile_id
        resource_id = access_id
        if self._channel_models is not None and access_node.anchor_profile_id is not None:
            if self._main_band_in_outage(
                access_node, distance_m, access_id, access_interference_dbm
            ):
                link_profile_id = access_node.anchor_profile_id
                resource_id = anchor_resource_id(access_node_id)

        access_link = TopologyLink(
            link_id=resource_id,
            source_node_id=source_node_id,
            destination_node_id=access_node_id,
            link_profile_id=link_profile_id,
            route_distance_m=distance_m,
            medium=Medium.RADIO,
            direction=Direction.UPLINK,
            provenance=access_node.provenance,
        )
        access_hop = ResolvedHop(
            hop=Hop(
                hop_id=resource_id,
                source_node_id=source_node_id,
                destination_node_id=access_node_id,
                link_profile_id=link_profile_id,
                direction=Direction.UPLINK,
            ),
            link=access_link,
            distance_m=distance_m,
        )

        if destination_node_id == access_node_id:
            backbone_hops: tuple[ResolvedHop, ...] = ()
        else:
            backbone = self._resolver.resolve_between(
                access_node_id,
                destination_node_id,
                position_overrides=overrides,
            )
            backbone_hops = backbone.hops
            if backbone_hops[0].hop.source_node_id != access_node_id:
                raise ValueError("declared backbone route does not begin at the access node")

        return ResolvedRoute(
            route_id=(
                f"runtime:{source_node_id}->{access_node_id}->{destination_node_id}"
            ),
            source_node_id=source_node_id,
            destination_node_id=destination_node_id,
            hops=(access_hop, *backbone_hops),
        )

    def _main_band_in_outage(
        self,
        access_node: TopologyNode,
        distance_m: float,
        main_hop_id: str,
        access_interference_dbm: float | None = None,
    ) -> bool:
        """Read-only probe: is the access node's main band in outage at this range?

        Only called when ``channel_models`` was supplied and the node declares
        an anchor tier (see ``__init__``); never affects the packet engine's
        retry state, RNG, or ledgers, since ``ChannelModel.evaluate`` is pure.

        ``access_interference_dbm`` must mirror the live estimate the real
        access-hop transmission will use (see ``build``'s parameter of the
        same name). ``ChannelModel.evaluate`` is a deterministic function of
        its inputs, so leaving interference out here does not make the probe
        a conservative worst case -- it makes it an interference-free best
        case, silently missing any main-band outage that only occurs under
        real co-channel load. That gap was confirmed live: a probe run with
        no interference returned "in range" for every one of 584 access
        decisions in a seed_1 evaluation trace, while the same run's actual
        per-attempt transmissions on that same main band still produced
        11,890 real outages -- the anchor never engaged for a single one.
        """

        try:
            channel_model = self._channel_models[access_node.profile_id]
        except KeyError as exc:
            raise ValueError(
                f"no ChannelModel registered for profile {access_node.profile_id!r}"
            ) from exc
        # Same config source RuntimeHopParameterBuilder uses for a device
        # source's tx_power_w: the device radio profile's radiated_tx_power_w.
        device_profile = POWER_PROFILES[_RADIO_POWER_PROFILE_BY_NODE_KIND[NodeKind.DEVICE]]
        probe_request = ChannelRequest(
            link_profile_id=access_node.profile_id,
            hop_id=main_hop_id,
            requested_at_s=0.0,
            # Same blank-tolerant coercion RuntimeHopParameterBuilder.build()
            # applies to this identically-named field: an unmeasured "" (e.g.
            # before a device's first interference reading) becomes None, not
            # a validation error.
            interference_dbm=RuntimeHopParameterBuilder._optional_finite(
                "access_interference_dbm", access_interference_dbm
            ),
            distance_m=distance_m,
            on_wire_bytes=1,
            direction=Direction.UPLINK,
            tx_power_w=float(device_profile["radiated_tx_power_w"]),
        )
        return channel_model.evaluate(probe_request).outage
