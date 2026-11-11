"""Tests for config/topology.py (Task 33 / R2 topology DATA).

Verifies contract §11 (provenance on every default) and §12 (typed topology
meanings), SI units, node-kind validity, Cloud-as-routed-endpoint (not a radio
tower), route contiguity, and the deterministic hand-check geometry.
"""

import math
import unittest

from config import topology as topo

PROV_FIELDS = {"source_id", "source_uri", "source_version", "valid_from",
               "valid_to", "validity_notes", "maturity"}
MATURITIES = {"validated", "standards-aligned", "experimental", "visionary"}


def _all_carriers(profile):
    """Every node, link, route, AND the coordinate frame carries provenance (§11)."""
    return (list(profile["nodes"]) + list(profile["links"])
            + list(profile["routes"]) + [profile["coordinate_frame"]])


class TestTopologyProfiles(unittest.TestCase):
    def test_profiles_present(self):
        self.assertEqual(set(topo.TOPOLOGY_PROFILES),
                         {"smartcity_default", "handcheck_min"})

    def test_node_schema_units_and_kinds(self):
        for name, profile in topo.TOPOLOGY_PROFILES.items():
            for n in profile["nodes"]:
                with self.subTest(profile=name, node=n["node_id"]):
                    self.assertIn(n["node_kind"], topo.NODE_KINDS)
                    for k in ("x_m", "y_m", "z_m"):
                        self.assertIsInstance(n[k], float)   # SI metres
                    self.assertGreaterEqual(n["z_m"], 0.0)   # height above ground
                    self.assertTrue(n["profile_id"])

    def test_provenance_on_every_default(self):
        for name, profile in topo.TOPOLOGY_PROFILES.items():
            for item in _all_carriers(profile):
                label = item.get("node_id") or item.get("link_id") or item.get("route_id") or "frame"
                with self.subTest(profile=name, item=label):
                    prov = item["provenance"]
                    self.assertEqual(set(prov), PROV_FIELDS)
                    self.assertIn(prov["maturity"], MATURITIES)
                    self.assertTrue(prov["source_id"])

    def test_no_default_falsely_labeled_validated(self):
        # §11: an existing simulator assumption is not 'validated' without disclosed
        # calibration. None of our shipped defaults claim measurement calibration.
        for profile in topo.TOPOLOGY_PROFILES.values():
            for item in _all_carriers(profile):
                self.assertNotEqual(item["provenance"]["maturity"], "validated")

    def test_published_graph_is_deep_immutable(self):
        prof = topo.get_topology("handcheck_min")
        # Mutating a node field must raise (MappingProxyType is read-only).
        with self.assertRaises(TypeError):
            prof["nodes"][0]["x_m"] = 999.0
        # Reassigning a profile key must raise.
        with self.assertRaises(TypeError):
            prof["nodes"] = ()
        # Nodes/links are tuples (no append).
        with self.assertRaises(AttributeError):
            prof["nodes"].append({})
        # Provenance dicts are frozen too.
        with self.assertRaises(TypeError):
            prof["nodes"][0]["provenance"]["maturity"] = "validated"

    def test_snapshot_stable_across_calls(self):
        # Same immutable object each call; a mutation attempt cannot leak into it.
        a = topo.get_topology("smartcity_default")
        b = topo.get_topology("smartcity_default")
        self.assertIs(a, b)
        try:
            a["nodes"][0]["x_m"] = -1.0
        except TypeError:
            pass
        self.assertGreaterEqual(topo.get_topology("smartcity_default")["nodes"][0]["x_m"], 0.0)

    def test_cloud_is_routed_endpoint_not_radio(self):
        # A cloud_gateway must never be the destination of a radio link.
        for name, profile in topo.TOPOLOGY_PROFILES.items():
            clouds = {n["node_id"] for n in profile["nodes"]
                      if n["node_kind"] == "cloud_gateway"}
            for link in profile["links"]:
                if link["destination_node_id"] in clouds:
                    with self.subTest(profile=name, link=link["link_id"]):
                        self.assertEqual(link["medium"], "wired")

    def test_positions_within_extent(self):
        for name, profile in topo.TOPOLOGY_PROFILES.items():
            ext = profile["coordinate_frame"]["extent_m"]
            for n in profile["nodes"]:
                with self.subTest(profile=name, node=n["node_id"]):
                    self.assertGreaterEqual(n["x_m"], ext["x_min"])
                    self.assertLessEqual(n["x_m"], ext["x_max"])
                    self.assertGreaterEqual(n["y_m"], ext["y_min"])
                    self.assertLessEqual(n["y_m"], ext["y_max"])

    def test_routes_are_contiguous(self):
        for name, profile in topo.TOPOLOGY_PROFILES.items():
            links = {l["link_id"]: l for l in profile["links"]}
            for route in profile["routes"]:
                hops = [links[h] for h in route["hop_ids"]]
                self.assertEqual(hops[0]["source_node_id"], route["source_node_id"])
                self.assertEqual(hops[-1]["destination_node_id"], route["destination_node_id"])
                for a, b in zip(hops, hops[1:]):
                    self.assertEqual(a["destination_node_id"], b["source_node_id"])

    def test_links_reference_existing_nodes(self):
        for name, profile in topo.TOPOLOGY_PROFILES.items():
            ids = {n["node_id"] for n in profile["nodes"]}
            for link in profile["links"]:
                self.assertIn(link["source_node_id"], ids)
                self.assertIn(link["destination_node_id"], ids)
                self.assertGreaterEqual(link["route_distance_m"], 0.0)

    def test_handcheck_geometry(self):
        prof = topo.get_topology("handcheck_min")
        nodes = {n["node_id"]: n for n in prof["nodes"]}
        d0, a0 = nodes["D0"], nodes["A0"]
        radio_3d = math.sqrt((a0["x_m"] - d0["x_m"])**2 + (a0["y_m"] - d0["y_m"])**2
                             + (a0["z_m"] - d0["z_m"])**2)
        self.assertAlmostEqual(radio_3d, 300.919009, places=3)
        wired = next(l for l in prof["links"] if l["link_id"] == "A0->C0")
        self.assertEqual(wired["route_distance_m"], 1000.0)
        self.assertEqual(wired["medium"], "wired")

    def test_provenance_helper_rejects_bad_maturity(self):
        with self.assertRaises(ValueError):
            topo.provenance("x", maturity="totally-real-6g")

    def test_get_topology_unknown_raises(self):
        with self.assertRaises(KeyError):
            topo.get_topology("mars_mesh")

    def test_four_uav_bs_match_runtime_count(self):
        nodes = topo.get_topology("smartcity_default")["nodes"]
        uavs = [n for n in nodes if n["node_kind"] == "uav_bs"]
        self.assertEqual(len(uavs), 4)  # matches SERVER_CONFIG["UAV_BS"] num_datacenters

    def test_routes_cover_mec_and_uav_to_cloud(self):
        routes = {r["route_id"] for r in topo.get_topology("smartcity_default")["routes"]}
        for rid in ("r_mec1_cloud", "r_mec2_cloud",
                    "r_uav1_cloud", "r_uav2_cloud", "r_uav3_cloud", "r_uav4_cloud"):
            self.assertIn(rid, routes)

    def test_consumable_by_network_runtime(self):
        # The strongest check: my config must satisfy ALL of network.topology's §12
        # validation (radio distance == geometry, contiguity, no radio->Cloud, extent).
        from network.topology import PhysicalTopology
        from network.routing import RouteResolver
        for name in ("smartcity_default", "handcheck_min"):
            phys = PhysicalTopology.from_mapping(name, topo.get_topology(name))
            resolver = RouteResolver(phys)
            for route in phys.routes:
                resolved = resolver.resolve(route.route_id)
                self.assertGreater(resolved.total_distance_m, 0.0)


if __name__ == "__main__":
    unittest.main()
