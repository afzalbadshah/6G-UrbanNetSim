"""Tests for independent access-density deployment composition."""

import unittest

from config.topology import DeploymentProfile, get_deployment_profile
from config.topology import get_topology
from core.deployment_builder import build_deployed_topology
from network import PhysicalTopology


class DeploymentProfileTests(unittest.TestCase):
    def test_profile_selection_requires_an_explicit_name(self):
        with self.assertRaises(TypeError):
            get_deployment_profile()
        for invalid_name in (None, "", "   "):
            with self.subTest(invalid_name=invalid_name):
                with self.assertRaises(ValueError):
                    get_deployment_profile(invalid_name)

    def test_named_profiles_validate_and_keep_mec_independent(self):
        sparse = get_deployment_profile("legacy_sparse")
        dense = get_deployment_profile("fr1_city_coverage")

        self.assertEqual(sparse.ground_bs_count, 0)
        self.assertEqual(dense.ground_bs_count, 81)
        self.assertEqual(sparse.access_generation, "5G")
        self.assertEqual(
            get_deployment_profile("experimental_6g_sla_16").access_generation,
            "6G",
        )
        self.assertNotIn("num_datacenters", dense.__slots__)

    def test_invalid_profile_values_are_rejected(self):
        with self.assertRaises(ValueError):
            DeploymentProfile("bad", -1)
        with self.assertRaises(ValueError):
            DeploymentProfile("bad", 1, ground_bs_height_m=-1.0)
        with self.assertRaises(ValueError):
            DeploymentProfile("bad", 1, placement_strategy="random")
        with self.assertRaises(ValueError):
            DeploymentProfile("bad", 1, access_generation="7G")

    def test_hotspot_count_and_profile_id_must_be_set_together(self):
        with self.assertRaises(ValueError):
            DeploymentProfile("bad", 1, hotspot_count=5)  # count without a band
        with self.assertRaises(ValueError):
            DeploymentProfile("bad", 1, hotspot_profile_id="nr-subthz-access")  # band without a count
        with self.assertRaises(ValueError):
            DeploymentProfile("bad", 1, hotspot_count=-1, hotspot_profile_id="nr-subthz-access")
        # Both set together, or both left at the zero/None default, are fine.
        DeploymentProfile("ok_a", 1, hotspot_count=5, hotspot_profile_id="nr-subthz-access")
        DeploymentProfile("ok_b", 1)

    def test_new_density_and_hotspot_profiles_are_registered(self):
        realistic_fr3 = get_deployment_profile("fr3_city_coverage_realistic")
        mmwave_hotspot = get_deployment_profile("mmwave_hotspot_overlay")
        subthz_hotspot = get_deployment_profile("subthz_hotspot_overlay")

        self.assertEqual(realistic_fr3.ground_bs_count, 964)
        self.assertEqual(realistic_fr3.ground_bs_profile_id, "nr-fr3-access")
        self.assertEqual(realistic_fr3.hotspot_count, 0)

        for hotspot_profile, band_id in (
            (mmwave_hotspot, "nr-mmwave-access"),
            (subthz_hotspot, "nr-subthz-access"),
        ):
            self.assertEqual(hotspot_profile.ground_bs_count, 81)
            self.assertEqual(hotspot_profile.ground_bs_profile_id, "nr-fr1-access")
            self.assertEqual(hotspot_profile.hotspot_count, 12)
            self.assertEqual(hotspot_profile.hotspot_profile_id, band_id)


class DeployedTopologyTests(unittest.TestCase):
    def test_legacy_profile_is_byte_for_byte_shape_compatible(self):
        deployment = get_deployment_profile("legacy_sparse")
        result = build_deployed_topology(
            "smartcity_default",
            get_topology("smartcity_default"),
            deployment,
        )
        topology = PhysicalTopology.from_mapping(result.profile_id, result.mapping)

        self.assertEqual(result.profile_id, "smartcity_default")
        self.assertEqual(result.access_points, ())
        self.assertEqual(len(topology.nodes), 9)

    def test_access_points_have_resources_and_routes_to_each_action_tier(self):
        deployment = get_deployment_profile("fr1_density_sweep_16")
        result = build_deployed_topology(
            "smartcity_default",
            get_topology("smartcity_default"),
            deployment,
        )
        topology = PhysicalTopology.from_mapping(result.profile_id, result.mapping)

        self.assertEqual(len(result.access_points), 16)
        self.assertEqual(
            sum(node.kind.value == "access_tower" for node in topology.nodes),
            16,
        )
        for access_point in result.access_points:
            node = topology.node(access_point.name)
            self.assertEqual(node.kind.value, "access_tower")
            self.assertEqual(node.profile_id, "nr-fr1-access")
            self.assertEqual(access_point.compute_anchor_id, "MEC_1" if access_point.y > 5000 else "MEC_2")
            for destination in ("MEC_1", "MEC_2", "Metro_1", "Cloud_1", "UAV_BS_1"):
                routes = [
                    route
                    for route in topology.routes
                    if route.source_node_id == access_point.name
                    and route.destination_node_id == destination
                ]
                self.assertEqual(len(routes), 1, (access_point.name, destination))

    def test_each_uav_has_a_direct_route_to_its_nearest_mec_anchor(self):
        result = build_deployed_topology(
            "smartcity_default",
            get_topology("smartcity_default"),
            get_deployment_profile("fr1_density_sweep_16"),
            compute_tier_counts={"MEC": 2, "UAV_BS": 3},
        )
        links = {link["link_id"]: link for link in result.mapping["links"]}
        routes = list(result.mapping["routes"])

        for uav_index in range(1, 4):
            uav_id = f"UAV_BS_{uav_index}"
            mec_routes = [
                route
                for route in routes
                if route["source_node_id"] == uav_id
                and route["destination_node_id"].startswith("MEC_")
            ]
            self.assertEqual(len(mec_routes), 1, uav_id)
            self.assertEqual(len(mec_routes[0]["hop_ids"]), 1)
            backhaul = links[mec_routes[0]["hop_ids"][0]]
            self.assertEqual(backhaul["source_node_id"], uav_id)
            self.assertEqual(
                backhaul["destination_node_id"],
                mec_routes[0]["destination_node_id"],
            )

    def test_hotspot_overlay_adds_a_second_band_on_top_of_the_macro_grid(self):
        deployment = get_deployment_profile("mmwave_hotspot_overlay")
        result = build_deployed_topology(
            "smartcity_default",
            get_topology("smartcity_default"),
            deployment,
        )
        topology = PhysicalTopology.from_mapping(result.profile_id, result.mapping)

        self.assertEqual(len(result.access_points), 81 + 12)
        macro_ids = {ap.name for ap in result.access_points if ap.name.startswith("BS_")}
        hotspot_ids = {ap.name for ap in result.access_points if ap.name.startswith("HOTSPOT_")}
        self.assertEqual(len(macro_ids), 81)
        self.assertEqual(len(hotspot_ids), 12)
        # No id collision between the two site sets.
        self.assertEqual(macro_ids & hotspot_ids, set())

        for access_point in result.access_points:
            node = topology.node(access_point.name)
            expected_band = "nr-mmwave-access" if access_point.name.startswith("HOTSPOT_") else "nr-fr1-access"
            self.assertEqual(node.profile_id, expected_band)

        # A hotspot site is wired exactly like a macro site: full route set.
        hotspot_point = next(ap for ap in result.access_points if ap.name == "HOTSPOT_1")
        for destination in ("MEC_1", "MEC_2", "Metro_1", "Cloud_1", "UAV_BS_1"):
            routes = [
                route
                for route in topology.routes
                if route.source_node_id == hotspot_point.name
                and route.destination_node_id == destination
            ]
            self.assertEqual(len(routes), 1, (hotspot_point.name, destination))

    def test_realistic_fr3_density_profile_builds_at_scale(self):
        deployment = get_deployment_profile("fr3_city_coverage_realistic")
        result = build_deployed_topology(
            "smartcity_default",
            get_topology("smartcity_default"),
            deployment,
        )
        topology = PhysicalTopology.from_mapping(result.profile_id, result.mapping)

        self.assertEqual(len(result.access_points), 964)
        self.assertTrue(
            all(ap.profile_id == "nr-fr3-access" for ap in result.access_points)
        )
        self.assertEqual(
            sum(node.kind.value == "access_tower" for node in topology.nodes),
            964,
        )
        # Spot-check one site has the full route set, same as the smaller profiles.
        sample = result.access_points[0]
        for destination in ("MEC_1", "MEC_2", "Metro_1", "Cloud_1", "UAV_BS_1"):
            routes = [
                route
                for route in topology.routes
                if route.source_node_id == sample.name
                and route.destination_node_id == destination
            ]
            self.assertEqual(len(routes), 1, (sample.name, destination))


if __name__ == "__main__":
    unittest.main()
