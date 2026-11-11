import unittest

from config.topology import get_topology
from network import (
    Direction,
    PhysicalTopology,
    Position3D,
    RouteResolver,
)
from tests.test_network_topology import mutable_handcheck


class RouteResolverTests(unittest.TestCase):
    def setUp(self):
        self.topology = PhysicalTopology.from_mapping(
            "handcheck_min", get_topology("handcheck_min")
        )
        self.resolver = RouteResolver(self.topology)

    def test_resolves_contiguous_hops_and_physical_distances(self):
        resolved = self.resolver.resolve("r_d0_core")

        self.assertEqual([item.hop.hop_id for item in resolved.hops], ["D0->A0", "A0->C0"])
        self.assertEqual(resolved.hops[0].hop.direction, Direction.UPLINK)
        self.assertEqual(resolved.hops[1].hop.direction, Direction.BACKHAUL)
        self.assertAlmostEqual(resolved.radio_distance_m, 300.919009, places=3)
        self.assertEqual(resolved.wired_distance_m, 1_000.0)
        self.assertAlmostEqual(
            resolved.total_distance_m,
            resolved.radio_distance_m + resolved.wired_distance_m,
        )

    def test_dynamic_positions_flow_into_resolved_route(self):
        initial = self.resolver.resolve("r_d0_core")
        moved = self.resolver.resolve(
            "r_d0_core",
            position_overrides={"D0": Position3D(100.0, 0.0, 1.5)},
        )
        self.assertLess(moved.radio_distance_m, initial.radio_distance_m)
        self.assertEqual(moved.wired_distance_m, initial.wired_distance_m)

    def test_resolve_between_requires_one_explicit_route(self):
        route = self.resolver.resolve_between("D0", "C0")
        self.assertEqual(route.route_id, "r_d0_core")
        with self.assertRaises(KeyError):
            self.resolver.resolve_between("C0", "D0")

    def test_noncontiguous_and_unknown_route_hops_are_rejected_at_load(self):
        noncontiguous = mutable_handcheck()
        noncontiguous["routes"][0]["hop_ids"] = ["A0->C0", "D0->A0"]
        with self.assertRaises(ValueError):
            PhysicalTopology.from_mapping("noncontiguous", noncontiguous)

        missing = mutable_handcheck()
        missing["routes"][0]["hop_ids"] = ["missing"]
        with self.assertRaises(ValueError):
            PhysicalTopology.from_mapping("missing-hop", missing)

    def test_ambiguous_route_pair_is_rejected(self):
        raw = mutable_handcheck()
        duplicate_route = dict(raw["routes"][0])
        duplicate_route["route_id"] = "r_d0_core_2"
        raw["routes"].append(duplicate_route)
        topology = PhysicalTopology.from_mapping("ambiguous", raw)
        with self.assertRaises(ValueError):
            RouteResolver(topology).resolve_between("D0", "C0")


if __name__ == "__main__":
    unittest.main()
