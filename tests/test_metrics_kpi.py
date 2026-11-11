import unittest

from config.topology import get_topology
from metrics import (
    ApplicationObservation,
    SixGChannelModel,
    aggregate_kpis,
    observe_route,
    percentile_type7,
)
from network import (
    Direction,
    FixedMtuPacketizer,
    FixedRateChannelModel,
    HopTransmissionParameters,
    NetworkEngine,
    PhysicalTopology,
    RouteResolver,
    SharedLink,
    TrafficObject,
    instantiate_link_profile,
)


def observation(
    object_id,
    payload_bytes,
    *,
    completed_at_s,
    deadline_at_s,
    offered,
    delivered,
    dropped,
    expired,
    wire_bytes,
):
    return ApplicationObservation(
        object_id=object_id,
        created_at_s=0.0,
        payload_bytes=payload_bytes,
        deadline_at_s=deadline_at_s,
        completed_at_s=completed_at_s,
        packet_hop_offered_count=offered,
        packet_hop_delivered_count=delivered,
        packet_hop_dropped_count=dropped,
        packet_hop_expired_count=expired,
        on_wire_bytes=wire_bytes,
    )


class KpiTests(unittest.TestCase):
    def test_route_observation_counts_payload_once_not_once_per_hop(self):
        radio = instantiate_link_profile(
            "nr-fr1-access", shared_resource_id="cell:A0"
        )
        fibre = instantiate_link_profile(
            "fibre-core", shared_resource_id="fibre:A0->C0"
        )
        route = RouteResolver(
            PhysicalTopology.from_mapping("handcheck_min", get_topology("handcheck_min"))
        ).resolve("r_d0_core")
        engine = NetworkEngine(
            packetizer=FixedMtuPacketizer(),
            channel_model={
                radio.profile_id: SixGChannelModel(radio),
                fibre.profile_id: FixedRateChannelModel(fibre),
            },
            shared_links={
                radio.profile_id: SharedLink(radio),
                fibre.profile_id: SharedLink(fibre),
            },
            experiment_seed=14,
        )
        result = engine.transmit_object_route(
            TrafficObject(
                object_id="kpi-object",
                flow_id="kpi-flow",
                source_id="D0",
                destination_id="C0",
                created_at_s=0.0,
                payload_bytes=3_000,
                deadline_at_s=1.0,
                direction=Direction.UPLINK,
            ),
            route,
            hop_parameters={
                "D0->A0": HopTransmissionParameters(tx_power_w=0.2, line_of_sight=True),
                "A0->C0": HopTransmissionParameters(tx_power_w=0.0),
            },
        )

        item = observe_route(result)

        self.assertTrue(item.completed)
        self.assertTrue(item.on_time)
        self.assertEqual(item.payload_bytes, 3_000)
        self.assertEqual(item.useful_application_bytes, 3_000)
        self.assertEqual(item.packet_hop_offered_count, 6)
        self.assertEqual(item.packet_hop_delivered_count, 6)
        self.assertGreater(item.on_wire_bytes, 6_000)

    def test_fixed_denominators_and_unique_application_goodput(self):
        items = (
            observation(
                "on-time",
                100,
                completed_at_s=0.1,
                deadline_at_s=0.5,
                offered=2,
                delivered=2,
                dropped=0,
                expired=0,
                wire_bytes=150,
            ),
            observation(
                "late",
                200,
                completed_at_s=1.2,
                deadline_at_s=1.0,
                offered=3,
                delivered=3,
                dropped=0,
                expired=0,
                wire_bytes=400,
            ),
            observation(
                "failed",
                300,
                completed_at_s=None,
                deadline_at_s=None,
                offered=1,
                delivered=0,
                dropped=1,
                expired=0,
                wire_bytes=200,
            ),
        )

        snapshot = aggregate_kpis(items, measurement_window_s=2.0)

        self.assertAlmostEqual(snapshot.application_completion_ratio, 2 / 3)
        self.assertEqual(snapshot.deadline_success_ratio, 0.5)
        self.assertAlmostEqual(snapshot.packet_hop_delivery_ratio, 5 / 6)
        self.assertEqual(snapshot.useful_application_bytes, 300)
        self.assertEqual(snapshot.on_wire_bytes, 750)
        self.assertEqual(snapshot.goodput_bps, 1_200.0)
        self.assertEqual(snapshot.wire_efficiency_ratio, 0.4)
        self.assertEqual(snapshot.completion_delay.sample_count, 2)
        self.assertTrue(snapshot.completion_delay.includes_late_completions)
        self.assertAlmostEqual(snapshot.completion_delay.p50_s, 0.65)
        self.assertAlmostEqual(snapshot.completion_delay.p95_s, 1.145)

    def test_late_completion_exclusion_is_explicit(self):
        items = (
            observation(
                "on-time",
                100,
                completed_at_s=0.1,
                deadline_at_s=0.5,
                offered=1,
                delivered=1,
                dropped=0,
                expired=0,
                wire_bytes=120,
            ),
            observation(
                "late",
                100,
                completed_at_s=2.0,
                deadline_at_s=1.0,
                offered=1,
                delivered=1,
                dropped=0,
                expired=0,
                wire_bytes=120,
            ),
        )

        snapshot = aggregate_kpis(
            items,
            measurement_window_s=2.0,
            include_late_completions_in_delay=False,
        )

        self.assertFalse(snapshot.completion_delay.includes_late_completions)
        self.assertEqual(snapshot.completion_delay.sample_count, 1)
        self.assertEqual(snapshot.completion_delay.p99_s, 0.1)
        self.assertEqual(snapshot.application_completion_ratio, 1.0)
        self.assertEqual(snapshot.deadline_success_ratio, 0.5)

    def test_percentile_method_and_invalid_denominators(self):
        self.assertIsNone(percentile_type7([], 0.5))
        self.assertEqual(percentile_type7([2.0], 0.95), 2.0)
        self.assertEqual(percentile_type7([0.0, 10.0], 0.95), 9.5)
        with self.assertRaises(ValueError):
            percentile_type7([1.0], 1.1)
        with self.assertRaises(ValueError):
            aggregate_kpis([], measurement_window_s=0.0)
        with self.assertRaises(ValueError):
            observation(
                "bad",
                1,
                completed_at_s=None,
                deadline_at_s=None,
                offered=2,
                delivered=1,
                dropped=0,
                expired=0,
                wire_bytes=1,
            )


if __name__ == "__main__":
    unittest.main()
