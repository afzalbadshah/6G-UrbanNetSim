import math
import unittest

from config.topology import get_topology
from core.network_runtime import (
    DynamicAccessRouteBuilder,
    RuntimeHopParameterBuilder,
    access_resource_id,
    build_channel_model_registry,
    build_energy_model_registry,
    build_network_engine,
    build_shared_link_registry,
)
from core.network_adapter import LegacyUnitBoundary, NetworkTaskAdapter
from network import PhysicalTopology, Position3D
from network import DeliveryStatus, HopTransmissionParameters
from types import SimpleNamespace


def smartcity():
    return PhysicalTopology.from_mapping(
        "smartcity_default", get_topology("smartcity_default")
    )


class SharedLinkCompositionTests(unittest.TestCase):
    def test_registry_has_one_resource_per_physical_link_and_access_point(self):
        topology = smartcity()

        registry = build_shared_link_registry(topology)

        self.assertEqual(
            set(registry),
            {
                "MEC_1->Metro_1",
                "MEC_2->Metro_2",
                "Metro_1->Cloud_1",
                "Metro_2->Cloud_1",
                "UAV_BS_1->MEC_2",
                "UAV_BS_2->MEC_1",
                "UAV_BS_3->MEC_1",
                "UAV_BS_4->MEC_2",
                "radio-access:MEC_1",
                "radio-access:MEC_2",
                "radio-access:UAV_BS_1",
                "radio-access:UAV_BS_2",
                "radio-access:UAV_BS_3",
                "radio-access:UAV_BS_4",
            },
        )
        self.assertEqual(
            registry["radio-access:MEC_1"].profile.shared_resource_id,
            "radio-access:MEC_1",
        )
        self.assertIsNot(
            registry["MEC_1->Metro_1"], registry["radio-access:MEC_1"]
        )

    def test_channel_registry_covers_every_runtime_link_profile(self):
        topology = smartcity()
        shared_links = build_shared_link_registry(topology)

        channels = build_channel_model_registry(topology, shared_links)

        self.assertEqual(
            set(channels),
            {link.profile.profile_id for link in shared_links.values()},
        )

    def test_energy_registry_covers_every_resource_with_endpoint_specific_models(self):
        topology = smartcity()
        shared_links = build_shared_link_registry(topology)

        energy = build_energy_model_registry(topology, shared_links)

        self.assertEqual(set(energy), set(shared_links))


class DynamicAccessRouteBuilderTests(unittest.TestCase):
    def test_prepends_live_access_hop_to_declared_backbone(self):
        subject = DynamicAccessRouteBuilder(smartcity())

        route = subject.build(
            source_node_id="device:T0",
            source_position=Position3D(5000.0, 7400.0, 1.5),
            access_node_id="MEC_1",
            destination_node_id="Cloud_1",
        )

        self.assertEqual(route.source_node_id, "device:T0")
        self.assertEqual(route.destination_node_id, "Cloud_1")
        self.assertEqual(
            tuple(item.hop.hop_id for item in route.hops),
            (
                "radio-access:MEC_1",
                "MEC_1->Metro_1",
                "Metro_1->Cloud_1",
            ),
        )
        self.assertAlmostEqual(
            route.hops[0].distance_m,
            math.sqrt(100.0**2 + (25.0 - 1.5) ** 2),
        )
        self.assertEqual(route.hops[1].distance_m, 10_000.0)
        self.assertEqual(route.hops[2].distance_m, 50_000.0)

    def test_access_only_destination_needs_no_backbone_route(self):
        subject = DynamicAccessRouteBuilder(smartcity())

        route = subject.build(
            source_node_id="device:T0",
            source_position=Position3D(5000.0, 7400.0, 1.5),
            access_node_id="MEC_1",
            destination_node_id="MEC_1",
        )

        self.assertEqual(len(route.hops), 1)
        self.assertEqual(route.hops[0].hop.hop_id, access_resource_id("MEC_1"))

    def test_live_access_position_override_changes_3d_distance(self):
        subject = DynamicAccessRouteBuilder(smartcity())

        route = subject.build(
            source_node_id="device:T0",
            source_position=Position3D(5000.0, 7400.0, 1.5),
            access_node_id="MEC_1",
            destination_node_id="Cloud_1",
            position_overrides={"MEC_1": Position3D(5000.0, 7300.0, 100.0)},
        )

        self.assertAlmostEqual(
            route.hops[0].distance_m,
            math.sqrt(100.0**2 + (100.0 - 1.5) ** 2),
        )
        self.assertEqual(route.hops[1].distance_m, 10_000.0)

    def test_missing_backbone_route_fails_closed(self):
        subject = DynamicAccessRouteBuilder(smartcity())

        with self.assertRaises(KeyError):
            subject.build(
                source_node_id="device:T0",
                source_position=Position3D(5000.0, 2600.0, 1.5),
                access_node_id="MEC_1",
                destination_node_id="Metro_2",
            )

    def test_cloud_cannot_be_used_as_a_radio_access_point(self):
        subject = DynamicAccessRouteBuilder(smartcity())

        with self.assertRaises(ValueError):
            subject.build(
                source_node_id="device:T0",
                source_position=Position3D(5000.0, 5000.0, 1.5),
                access_node_id="Cloud_1",
                destination_node_id="Cloud_1",
            )

    def test_two_devices_share_the_same_access_resource_identity(self):
        subject = DynamicAccessRouteBuilder(smartcity())
        first = subject.build(
            source_node_id="device:T0",
            source_position=Position3D(5000.0, 7400.0, 1.5),
            access_node_id="MEC_1",
            destination_node_id="MEC_1",
        )
        second = subject.build(
            source_node_id="device:T1",
            source_position=Position3D(5010.0, 7400.0, 1.5),
            access_node_id="MEC_1",
            destination_node_id="MEC_1",
        )

        self.assertEqual(first.hops[0].hop.hop_id, second.hops[0].hop.hop_id)
        self.assertNotEqual(
            first.hops[0].hop.source_node_id,
            second.hops[0].hop.source_node_id,
        )

    def test_real_composition_executes_packets_and_exposes_shared_queueing(self):
        topology = smartcity()
        routes = DynamicAccessRouteBuilder(topology)
        engine = build_network_engine(topology, experiment_seed=17)
        adapter = NetworkTaskAdapter(
            engine=engine,
            units=LegacyUnitBoundary(bytes_per_legacy_kb=1000),
        )
        parameters = {
            "radio-access:MEC_1": HopTransmissionParameters(tx_power_w=0.2),
            "MEC_1->Metro_1": HopTransmissionParameters(tx_power_w=0.0),
            "Metro_1->Cloud_1": HopTransmissionParameters(tx_power_w=0.0),
        }
        first_route = routes.build(
            source_node_id="device:T0",
            source_position=Position3D(5000.0, 7400.0, 1.5),
            access_node_id="MEC_1",
            destination_node_id="Cloud_1",
        )
        second_route = routes.build(
            source_node_id="device:T1",
            source_position=Position3D(5010.0, 7400.0, 1.5),
            access_node_id="MEC_1",
            destination_node_id="Cloud_1",
        )
        first_task = SimpleNamespace(
            id="T0",
            data_size_kb=0.1,
            deadline_ms=1000.0,
            service_class="URLLC",
            application_type="safety_upload",
        )
        second_task = SimpleNamespace(**{**vars(first_task), "id": "T1"})

        first = adapter.execute(
            first_task,
            first_route,
            current_time_ms=0.0,
            hop_parameters=parameters,
        )
        second = adapter.execute(
            second_task,
            second_route,
            current_time_ms=0.0,
            hop_parameters=parameters,
        )

        self.assertEqual(first.status, DeliveryStatus.DELIVERED)
        self.assertEqual(second.status, DeliveryStatus.DELIVERED)
        self.assertGreater(
            second.critical_packet_delay.hop_delays[0].queue_delay_s,
            first.critical_packet_delay.hop_delays[0].queue_delay_s,
        )
        self.assertTrue(engine.snapshot().completeness.energy_complete)
        self.assertGreater(first.tx_energy_j, 0.0)
        self.assertGreater(first.rx_energy_j, 0.0)
        self.assertGreater(first.transport_energy_j, 0.0)


class RuntimeHopParameterBuilderTests(unittest.TestCase):
    def test_uses_endpoint_rf_profiles_and_live_aerial_geometry(self):
        topology = smartcity()
        routes = DynamicAccessRouteBuilder(topology)
        source = Position3D(2500.0, 2400.0, 1.5)
        live_uav = Position3D(2500.0, 2500.0, 250.0)
        route = routes.build(
            source_node_id="device:T0",
            source_position=source,
            access_node_id="UAV_BS_1",
            destination_node_id="Cloud_1",
            position_overrides={"UAV_BS_1": live_uav},
        )

        parameters = RuntimeHopParameterBuilder(topology).build(
            route,
            source_position=source,
            position_overrides={"UAV_BS_1": live_uav},
            access_interference_dbm=-91.0,
        )

        self.assertEqual(set(parameters), {item.hop.hop_id for item in route.hops})
        self.assertEqual(parameters["radio-access:UAV_BS_1"].tx_power_w, 0.2)
        self.assertEqual(
            parameters["radio-access:UAV_BS_1"].allocated_bandwidth_hz,
            20_000_000.0,
        )
        self.assertEqual(parameters["radio-access:UAV_BS_1"].interference_dbm, -91.0)
        self.assertAlmostEqual(
            parameters["radio-access:UAV_BS_1"].elevation_angle_deg,
            math.degrees(math.atan2(248.5, 100.0)),
        )
        self.assertEqual(parameters["UAV_BS_1->MEC_2"].tx_power_w, 1.0)
        self.assertEqual(
            parameters["UAV_BS_1->MEC_2"].allocated_bandwidth_hz,
            400_000_000.0,
        )
        self.assertIsNone(parameters["UAV_BS_1->MEC_2"].interference_dbm)
        self.assertIs(parameters["UAV_BS_1->MEC_2"].line_of_sight, True)
        self.assertGreater(parameters["UAV_BS_1->MEC_2"].elevation_angle_deg, 0.0)
        self.assertEqual(parameters["MEC_2->Metro_2"].tx_power_w, 0.0)
        self.assertIsNone(parameters["MEC_2->Metro_2"].allocated_bandwidth_hz)
        self.assertEqual(parameters["Metro_2->Cloud_1"].tx_power_w, 0.0)

    def test_does_not_accept_the_legacy_device_energy_draw_as_rf_power(self):
        topology = smartcity()
        source = Position3D(5000.0, 7400.0, 1.5)
        route = DynamicAccessRouteBuilder(topology).build(
            source_node_id="device:T0",
            source_position=source,
            access_node_id="MEC_1",
            destination_node_id="MEC_1",
        )
        legacy_task = SimpleNamespace(tx_power_w=1.3)

        parameters = RuntimeHopParameterBuilder(topology).build(
            route,
            source_position=source,
        )

        self.assertEqual(legacy_task.tx_power_w, 1.3)
        self.assertEqual(parameters["radio-access:MEC_1"].tx_power_w, 0.2)
        self.assertEqual(
            parameters["radio-access:MEC_1"].allocated_bandwidth_hz,
            20_000_000.0,
        )

    def test_blank_interference_is_treated_as_unobserved(self):
        topology = smartcity()
        source = Position3D(5000.0, 7400.0, 1.5)
        route = DynamicAccessRouteBuilder(topology).build(
            source_node_id="device:T0",
            source_position=source,
            access_node_id="MEC_1",
            destination_node_id="MEC_1",
        )

        parameters = RuntimeHopParameterBuilder(topology).build(
            route,
            source_position=source,
            access_interference_dbm="",
        )

        self.assertIsNone(parameters["radio-access:MEC_1"].interference_dbm)


if __name__ == "__main__":
    unittest.main()
