import math
import unittest
from dataclasses import FrozenInstanceError

from config.topology import get_topology
from network import Medium, NodeKind, PhysicalTopology, Position3D


class PhysicalTopologyTests(unittest.TestCase):
    def setUp(self):
        self.topology = PhysicalTopology.from_mapping(
            "handcheck_min", get_topology("handcheck_min")
        )

    def test_loads_immutable_typed_objects(self):
        self.assertEqual(self.topology.profile_id, "handcheck_min")
        self.assertEqual(self.topology.node("D0").kind, NodeKind.DEVICE)
        self.assertEqual(self.topology.link("D0->A0").medium, Medium.RADIO)
        self.assertEqual(self.topology.link("A0->C0").medium, Medium.WIRED)
        with self.assertRaises(FrozenInstanceError):
            self.topology.node("D0").position.x_m = 5.0

    def test_radio_uses_3d_geometry_and_wired_uses_route_length(self):
        expected_radio_m = math.sqrt(300.0**2 + (25.0 - 1.5) ** 2)
        self.assertAlmostEqual(
            self.topology.link_distance_m("D0->A0"), expected_radio_m
        )
        self.assertEqual(self.topology.link_distance_m("A0->C0"), 1_000.0)

    def test_dynamic_position_override_changes_only_radio_distance(self):
        moved = {"D0": Position3D(x_m=100.0, y_m=0.0, z_m=1.5)}
        initial = self.topology.link_distance_m("D0->A0")
        dynamic = self.topology.link_distance_m("D0->A0", position_overrides=moved)

        self.assertLess(dynamic, initial)
        self.assertAlmostEqual(dynamic, math.sqrt(200.0**2 + 23.5**2))
        self.assertEqual(
            self.topology.link_distance_m("A0->C0", position_overrides=moved),
            1_000.0,
        )
        self.assertEqual(self.topology.node("D0").position.x_m, 0.0)

    def test_unknown_ids_and_invalid_overrides_are_rejected(self):
        with self.assertRaises(KeyError):
            self.topology.node("missing")
        with self.assertRaises(KeyError):
            self.topology.link_distance_m("missing")
        with self.assertRaises(KeyError):
            self.topology.link_distance_m(
                "D0->A0", position_overrides={"missing": Position3D(0.0, 0.0, 0.0)}
            )
        with self.assertRaises(ValueError):
            self.topology.link_distance_m("D0->A0", position_overrides={"D0": (0, 0, 0)})

    def test_cloud_cannot_be_incident_to_radio_link(self):
        raw = mutable_handcheck()
        raw["links"][1]["medium"] = "radio"
        with self.assertRaises(ValueError):
            PhysicalTopology.from_mapping("bad-cloud-radio", raw)

    def test_duplicate_nodes_and_broken_links_are_rejected(self):
        duplicate = mutable_handcheck()
        duplicate["nodes"].append(dict(duplicate["nodes"][0]))
        with self.assertRaises(ValueError):
            PhysicalTopology.from_mapping("duplicate", duplicate)

        broken = mutable_handcheck()
        broken["links"][0]["destination_node_id"] = "missing"
        with self.assertRaises(ValueError):
            PhysicalTopology.from_mapping("broken", broken)

    def test_invalid_position_and_extent_are_rejected(self):
        with self.assertRaises(ValueError):
            Position3D(x_m=0.0, y_m=0.0, z_m=-1.0)
        raw = mutable_handcheck()
        raw["coordinate_frame"]["extent_m"]["x_max"] = -1.0
        with self.assertRaises(ValueError):
            PhysicalTopology.from_mapping("bad-extent", raw)


def mutable_handcheck():
    frozen = get_topology("handcheck_min")

    def thaw(value):
        if hasattr(value, "items"):
            return {key: thaw(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [thaw(item) for item in value]
        return value

    return thaw(frozen)


if __name__ == "__main__":
    unittest.main()
