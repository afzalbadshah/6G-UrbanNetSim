import math
import unittest
from types import SimpleNamespace

from core.network_adapter import (
    LegacyUnitBoundary,
    NetworkTaskAdapter,
)
from network import (
    ChannelResult,
    DeliveryStatus,
    FixedMtuPacketizer,
    HopTransmissionParameters,
    NetworkEngine,
    PhysicalTopology,
    RouteResolver,
    SharedLink,
    instantiate_link_profile,
)
from config.topology import get_topology


class ConstantChannel:
    def __init__(self, *, rate_bps=1_000_000.0, outage=False):
        self.rate_bps = rate_bps
        self.outage = outage

    def evaluate(self, request):
        return ChannelResult(
            model_id="adapter-handcheck-v1",
            sinr_db=20.0 if not self.outage else -20.0,
            achievable_rate_bps=0.0 if self.outage else self.rate_bps,
            outage=self.outage,
            packet_error_probability=1.0 if self.outage else 0.0,
            propagation_delay_s=request.distance_m / 200_000_000.0,
        )


def adapter(*, rate_bps=1_000_000.0):
    topology = PhysicalTopology.from_mapping("handcheck_min", get_topology("handcheck_min"))
    route = RouteResolver(topology).resolve("r_d0_core")
    links = {}
    for resolved_hop in route.hops:
        link_id = resolved_hop.link.link_id
        profile = instantiate_link_profile(
            resolved_hop.link.link_profile_id,
            shared_resource_id=link_id,
        )
        links[link_id] = SharedLink(profile)
    model = ConstantChannel(rate_bps=rate_bps)
    engine = NetworkEngine(
        packetizer=FixedMtuPacketizer(),
        channel_model={
            "nr-fr1-access": model,
            "fibre-core": model,
        },
        shared_links=links,
    )
    return (
        NetworkTaskAdapter(
            engine=engine,
            units=LegacyUnitBoundary(bytes_per_legacy_kb=1000),
        ),
        route,
        {
            item.hop.hop_id: HopTransmissionParameters(tx_power_w=0.2)
            for item in route.hops
        },
    )


def task(**overrides):
    values = {
        "id": "T0",
        "data_size_kb": 3.0,
        "deadline_ms": 1000.0,
        "service_class": "URLLC",
        "application_type": "safety_upload",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class LegacyUnitBoundaryTests(unittest.TestCase):
    def test_decimal_kilobytes_and_milliseconds_convert_exactly(self):
        units = LegacyUnitBoundary(bytes_per_legacy_kb=1000)
        self.assertEqual(units.kilobytes_to_bytes(0.24), 240)
        self.assertEqual(units.kilobytes_to_bytes(3.0), 3000)
        self.assertEqual(units.milliseconds_to_seconds(2500.0, name="time"), 2.5)

    def test_non_integral_byte_and_invalid_time_are_rejected(self):
        units = LegacyUnitBoundary(bytes_per_legacy_kb=1000)
        with self.assertRaises(ValueError):
            units.kilobytes_to_bytes(0.0005)
        with self.assertRaises(ValueError):
            units.kilobytes_to_bytes(float("inf"))
        with self.assertRaises(ValueError):
            units.milliseconds_to_seconds(-1.0, name="time")


class NetworkTaskAdapterTests(unittest.TestCase):
    def test_task_becomes_packetized_multihop_application_result(self):
        subject, route, parameters = adapter()

        result = subject.execute(
            task(),
            route,
            current_time_ms=2500.0,
            hop_parameters=parameters,
        )

        self.assertEqual(result.status, DeliveryStatus.DELIVERED)
        self.assertEqual(result.traffic_object.created_at_s, 2.5)
        self.assertEqual(result.traffic_object.deadline_at_s, 3.5)
        self.assertEqual(result.payload_bytes, 3000)
        self.assertEqual(result.packet_count, 3)
        self.assertEqual(result.packet_hop_count, 6)
        # 3 radio packets: payload + 3*(IPv6/UDP 48 + NR 11)
        # 3 fibre frames: payload + 3*(IPv6/UDP 48 + Ethernet occupancy 38)
        self.assertEqual(result.on_wire_bytes, 6435)
        self.assertGreater(result.network_elapsed_s, 0.0)
        self.assertAlmostEqual(
            result.network_elapsed_s,
            result.critical_packet_delay.elapsed_delay_s,
        )
        self.assertEqual(result.observation.useful_application_bytes, 3000)
        self.assertEqual(result.total_network_energy_j, 0.0)

    def test_deadline_expiry_is_not_reported_as_completion(self):
        subject, route, parameters = adapter(rate_bps=10_000.0)

        result = subject.execute(
            task(deadline_ms=0.1),
            route,
            current_time_ms=0.0,
            hop_parameters=parameters,
        )

        self.assertEqual(result.status, DeliveryStatus.EXPIRED)
        self.assertFalse(result.observation.completed)
        self.assertEqual(result.observation.useful_application_bytes, 0)
        self.assertTrue(math.isclose(result.finished_at_s, result.network_elapsed_s))

    def test_explicit_payload_bytes_are_not_reinterpreted_as_bandwidth(self):
        subject, route, parameters = adapter()

        result = subject.execute(
            task(data_size_kb=999.0, input_payload_bytes=240),
            route,
            current_time_ms=0.0,
            hop_parameters=parameters,
        )

        self.assertEqual(result.payload_bytes, 240)
        self.assertEqual(result.packet_count, 1)

    def test_adapter_does_not_mutate_legacy_task(self):
        subject, route, parameters = adapter()
        original = task()
        before = vars(original).copy()

        subject.execute(
            original,
            route,
            current_time_ms=0.0,
            hop_parameters=parameters,
        )

        self.assertEqual(vars(original), before)


if __name__ == "__main__":
    unittest.main()
