import unittest

from config.topology import get_topology
from network import (
    BernoulliSample,
    ChannelResult,
    DeliveryStatus,
    Direction,
    FixedMtuPacketizer,
    FixedRateChannelModel,
    HeaderStack,
    Hop,
    HopTransmissionParameters,
    LinkProfile,
    MeasurementCompleteness,
    NetworkEngine,
    PacketUnit,
    RetryPolicy,
    PhysicalTopology,
    RouteResolver,
    SharedLink,
    TrafficObject,
    TransmissionLedger,
    TransmissionFailureReason,
)


class RecordingChannel:
    def __init__(self, results):
        self._results = list(results)
        self.requests = []

    def evaluate(self, request):
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self._results) - 1)
        return self._results[index]


class ScriptedSampler:
    def __init__(self, occurrences):
        self._occurrences = list(occurrences)
        self.calls = []

    def sample(self, probability, *, purpose, unit_id, hop_id, attempt_number):
        self.calls.append((probability, purpose, unit_id, hop_id, attempt_number))
        occurred = self._occurrences[len(self.calls) - 1]
        return BernoulliSample(
            probability=probability,
            uniform_value=0.0 if occurred else 0.999999,
            occurred=occurred,
        )


def profile(profile_id="nr-fr1"):
    return LinkProfile(
        profile_id=profile_id,
        technology="NR",
        generation="5G/6G baseline",
        band="FR1",
        network_mtu_bytes=1500,
        nominal_bit_rate_bps=1_000_000.0,
        propagation_speed_m_s=300_000_000.0,
        shared_resource_id="cell-1",
        header_stack=HeaderStack(
            network_header_bytes=40,
            transport_header_bytes=8,
            link_header_bytes=14,
            link_trailer_bytes=4,
        ),
    )


def channel_result(*, outage=False, probability=0.0, rate_bps=1_000_000.0, propagation_s=0.001):
    return ChannelResult(
        model_id="test-channel",
        sinr_db=-20.0 if outage else 20.0,
        achievable_rate_bps=0.0 if outage else rate_bps,
        outage=outage,
        packet_error_probability=1.0 if outage else probability,
        propagation_delay_s=propagation_s,
        rx_power_dbm=-120.0 if outage else -60.0,
    )


def packet():
    return PacketUnit(
        packet_id="object-1:pdu:0:packet:0",
        pdu_set_id="object-1:pdu:0",
        sequence_number=0,
        payload_bytes=952,
        created_at_s=0.0,
        network_header_bytes=40,
        transport_header_bytes=8,
        link_header_bytes=14,
        link_trailer_bytes=4,
        is_last=True,
    )


def hop(profile_id="nr-fr1"):
    return Hop(
        hop_id="device-1-to-mec-1",
        source_node_id="device-1",
        destination_node_id="mec-1",
        link_profile_id=profile_id,
        direction=Direction.UPLINK,
    )


def engine(results, occurrences, *, retry_policy=None):
    link = SharedLink(profile())
    channel = RecordingChannel(results)
    sampler = ScriptedSampler(occurrences)
    network_engine = NetworkEngine(
        packetizer=FixedMtuPacketizer(),
        channel_model=channel,
        shared_links={link.profile.profile_id: link},
        retry_policy=retry_policy or RetryPolicy(),
        sampler=sampler,
    )
    return network_engine, channel, sampler, link


class NetworkEngineTests(unittest.TestCase):
    def test_success_wires_channel_link_outcome_and_ledger(self):
        network_engine, channel, sampler, _ = engine([channel_result()], [False])

        result = network_engine.transmit_packet(
            packet(),
            hop(),
            requested_at_s=2.0,
            distance_m=300.0,
            tx_power_w=0.2,
            allocated_bandwidth_hz=20_000_000.0,
            line_of_sight=True,
        )

        self.assertEqual(result.outcome.status, DeliveryStatus.DELIVERED)
        self.assertEqual(result.outcome.attempts, 1)
        self.assertEqual(result.outcome.on_wire_bytes, packet().on_wire_bytes)
        self.assertEqual(result.outcome.useful_payload_bytes, packet().payload_bytes)
        self.assertAlmostEqual(result.outcome.started_at_s, 2.0)
        self.assertAlmostEqual(result.outcome.ended_at_s, 2.009144)
        self.assertEqual(len(result.attempts), 1)
        self.assertTrue(result.attempts[0].succeeded)
        self.assertEqual(channel.requests[0].on_wire_bytes, packet().on_wire_bytes)
        self.assertEqual(channel.requests[0].distance_m, 300.0)
        self.assertEqual(sampler.calls[0][1], "per")

        snapshot = network_engine.snapshot()
        self.assertEqual(snapshot.transmission.delivered_count, 1)
        self.assertEqual(snapshot.transmission.offered_payload_bytes, packet().payload_bytes)
        self.assertEqual(snapshot.transmission.useful_payload_bytes, packet().payload_bytes)
        self.assertEqual(
            snapshot.completeness,
            MeasurementCompleteness.r2_routed_packet_delivery(),
        )
        self.assertFalse(snapshot.completeness.energy_complete)
        self.assertFalse(snapshot.completeness.cost_complete)
        self.assertEqual(
            snapshot.completeness.scope_id,
            "r2_exact_routed_packet_delivery",
        )
        self.assertTrue(any("contention" in item for item in snapshot.completeness.limitations))

    def test_packet_error_retries_and_counts_every_attempt(self):
        policy = RetryPolicy(max_attempts=2, backoff_s=0.1)
        network_engine, channel, sampler, _ = engine(
            [channel_result(probability=0.5)],
            [True, False],
            retry_policy=policy,
        )

        result = network_engine.transmit_packet(
            packet(), hop(), requested_at_s=0.0, distance_m=10.0, tx_power_w=0.2
        )

        self.assertEqual(result.outcome.status, DeliveryStatus.DELIVERED)
        self.assertEqual(result.outcome.attempts, 2)
        self.assertEqual(result.outcome.on_wire_bytes, packet().on_wire_bytes * 2)
        self.assertEqual(len(channel.requests), 2)
        self.assertEqual(len(sampler.calls), 2)
        self.assertEqual(
            result.attempts[0].failure_reason,
            TransmissionFailureReason.PACKET_ERROR,
        )
        self.assertTrue(result.attempts[1].succeeded)
        self.assertAlmostEqual(result.attempts[1].requested_at_s, 0.109144)
        self.assertEqual(network_engine.snapshot().transmission.retransmission_count, 1)

    def test_retry_limit_drops_after_real_transmissions(self):
        network_engine, _, _, _ = engine(
            [channel_result(probability=1.0)],
            [True, True],
            retry_policy=RetryPolicy(max_attempts=2),
        )

        result = network_engine.transmit_packet(
            packet(), hop(), requested_at_s=0.0, distance_m=10.0, tx_power_w=0.2
        )

        self.assertEqual(result.outcome.status, DeliveryStatus.DROPPED)
        self.assertEqual(result.outcome.failure_reason, "packet_error")
        self.assertEqual(result.outcome.attempts, 2)
        self.assertEqual(result.outcome.on_wire_bytes, packet().on_wire_bytes * 2)
        self.assertEqual(result.outcome.useful_payload_bytes, 0)

    def test_outage_is_distinct_and_uses_observation_time_before_retry(self):
        policy = RetryPolicy(max_attempts=2, backoff_s=0.2, outage_observation_s=0.05)
        network_engine, _, sampler, _ = engine(
            [channel_result(outage=True), channel_result(probability=0.0)],
            [False],
            retry_policy=policy,
        )

        result = network_engine.transmit_packet(
            packet(), hop(), requested_at_s=1.0, distance_m=900.0, tx_power_w=0.2
        )

        self.assertEqual(result.outcome.status, DeliveryStatus.DELIVERED)
        self.assertEqual(result.outcome.attempts, 2)
        self.assertEqual(
            result.attempts[0].failure_reason,
            TransmissionFailureReason.CHANNEL_OUTAGE,
        )
        self.assertIsNone(result.attempts[0].reservation)
        self.assertAlmostEqual(result.attempts[0].ended_at_s, 1.05)
        self.assertAlmostEqual(result.attempts[1].requested_at_s, 1.25)
        self.assertEqual(len(sampler.calls), 1)

    def test_arrival_after_deadline_is_expired_not_delivered(self):
        network_engine, _, _, _ = engine(
            [channel_result(rate_bps=8_000.0, propagation_s=0.01)], [False]
        )

        result = network_engine.transmit_packet(
            packet(),
            hop(),
            requested_at_s=0.0,
            distance_m=10.0,
            tx_power_w=0.2,
            deadline_at_s=0.5,
        )

        self.assertEqual(result.outcome.status, DeliveryStatus.EXPIRED)
        self.assertEqual(result.outcome.failure_reason, "deadline_at_arrival")
        self.assertEqual(result.outcome.attempts, 1)
        self.assertEqual(result.outcome.useful_payload_bytes, 0)
        self.assertGreater(result.outcome.ended_at_s, 0.5)

    def test_retry_due_after_deadline_expires_without_fake_attempt(self):
        network_engine, channel, _, _ = engine(
            [channel_result(probability=1.0)],
            [True],
            retry_policy=RetryPolicy(max_attempts=2, backoff_s=1.0),
        )

        result = network_engine.transmit_packet(
            packet(),
            hop(),
            requested_at_s=0.0,
            distance_m=10.0,
            tx_power_w=0.2,
            deadline_at_s=0.5,
        )

        self.assertEqual(result.outcome.status, DeliveryStatus.EXPIRED)
        self.assertEqual(result.outcome.failure_reason, "deadline_before_attempt")
        self.assertEqual(result.outcome.attempts, 1)
        self.assertEqual(result.outcome.on_wire_bytes, packet().on_wire_bytes)
        self.assertEqual(len(channel.requests), 1)

    def test_already_expired_packet_records_zero_attempts_and_zero_wire_bytes(self):
        network_engine, channel, sampler, _ = engine([channel_result()], [False])

        result = network_engine.transmit_packet(
            packet(),
            hop(),
            requested_at_s=2.0,
            distance_m=10.0,
            tx_power_w=0.2,
            deadline_at_s=1.0,
        )

        self.assertEqual(result.outcome.status, DeliveryStatus.EXPIRED)
        self.assertEqual(result.outcome.attempts, 0)
        self.assertEqual(result.outcome.on_wire_bytes, 0)
        self.assertEqual(len(channel.requests), 0)
        self.assertEqual(len(sampler.calls), 0)
        self.assertEqual(network_engine.snapshot().transmission.attempt_count, 0)

    def test_queue_wait_that_reaches_deadline_expires_before_channel_evaluation(self):
        network_engine, channel, sampler, link = engine([channel_result()], [False])
        link.reserve(requested_at_s=0.0, on_wire_bytes=1_000, achievable_rate_bps=8_000.0)

        result = network_engine.transmit_packet(
            packet(),
            hop(),
            requested_at_s=0.0,
            distance_m=10.0,
            tx_power_w=0.2,
            deadline_at_s=0.5,
        )

        self.assertEqual(result.outcome.status, DeliveryStatus.EXPIRED)
        self.assertEqual(result.outcome.attempts, 0)
        self.assertEqual(len(channel.requests), 0)
        self.assertEqual(len(sampler.calls), 0)

    def test_object_path_packetizes_then_uses_shared_fifo_link(self):
        network_engine, channel, _, _ = engine([channel_result()], [False] * 3)
        traffic = TrafficObject(
            object_id="object-2",
            flow_id="flow-2",
            source_id="device-1",
            destination_id="mec-1",
            created_at_s=0.0,
            payload_bytes=3_000,
            deadline_at_s=1.0,
            direction=Direction.UPLINK,
        )

        result = network_engine.transmit_object_one_hop(
            traffic,
            hop(),
            distance_m=100.0,
            tx_power_w=0.2,
        )

        self.assertEqual(len(result.packetization.packets), 3)
        self.assertEqual(len(result.packet_transmissions), 3)
        self.assertTrue(result.all_delivered)
        self.assertEqual(len(channel.requests), 3)
        second = result.packet_transmissions[1].attempts[0].reservation
        self.assertIsNotNone(second)
        self.assertGreater(second.queue_delay_s, 0.0)
        snapshot = network_engine.snapshot().transmission
        self.assertEqual(snapshot.offered_payload_bytes, 3_000)
        self.assertEqual(snapshot.useful_payload_bytes, 3_000)
        self.assertEqual(snapshot.on_wire_bytes, result.packetization.on_wire_bytes)

    def test_profile_and_direction_mismatches_are_rejected(self):
        network_engine, _, _, _ = engine([channel_result()], [False])
        with self.assertRaises(ValueError):
            network_engine.transmit_packet(
                packet(), hop("missing-profile"), distance_m=10.0, tx_power_w=0.2
            )

        traffic = TrafficObject(
            object_id="downlink-object",
            flow_id="flow",
            source_id="mec-1",
            destination_id="device-1",
            created_at_s=0.0,
            payload_bytes=10,
            direction=Direction.DOWNLINK,
        )
        with self.assertRaises(ValueError):
            network_engine.transmit_object_one_hop(
                traffic, hop(), distance_m=10.0, tx_power_w=0.2
            )

    def test_duplicate_shared_resource_instances_are_rejected(self):
        first = SharedLink(profile("profile-a"))
        second = SharedLink(profile("profile-b"))
        with self.assertRaises(ValueError):
            NetworkEngine(
                packetizer=FixedMtuPacketizer(),
                channel_model=RecordingChannel([channel_result()]),
                shared_links={"profile-a": first, "profile-b": second},
                sampler=ScriptedSampler([False]),
            )

    def test_supplied_empty_ledger_is_used_not_replaced(self):
        supplied = TransmissionLedger()
        link = SharedLink(profile())
        network_engine = NetworkEngine(
            packetizer=FixedMtuPacketizer(),
            channel_model=RecordingChannel([channel_result()]),
            shared_links={link.profile.profile_id: link},
            sampler=ScriptedSampler([False]),
            ledger=supplied,
        )
        network_engine.transmit_packet(
            packet(), hop(), requested_at_s=0.0, distance_m=10.0, tx_power_w=0.2
        )
        self.assertIs(network_engine.ledger, supplied)
        self.assertEqual(len(supplied), 1)

    def test_distinct_physical_hops_can_reuse_profile_without_global_queue(self):
        first_profile = LinkProfile(
            profile_id="nr-shared-template",
            technology="NR",
            generation="6G",
            band="FR1",
            network_mtu_bytes=1500,
            nominal_bit_rate_bps=1_000_000.0,
            propagation_speed_m_s=300_000_000.0,
            shared_resource_id="cell-a",
            header_stack=HeaderStack(network_header_bytes=40, transport_header_bytes=8),
        )
        second_profile = LinkProfile(
            profile_id="nr-shared-template",
            technology="NR",
            generation="6G",
            band="FR1",
            network_mtu_bytes=1500,
            nominal_bit_rate_bps=1_000_000.0,
            propagation_speed_m_s=300_000_000.0,
            shared_resource_id="cell-b",
            header_stack=HeaderStack(network_header_bytes=40, transport_header_bytes=8),
        )
        first_link = SharedLink(first_profile)
        second_link = SharedLink(second_profile)
        network_engine = NetworkEngine(
            packetizer=FixedMtuPacketizer(),
            channel_model=RecordingChannel([channel_result()]),
            shared_links={"device-a->tower-a": first_link, "device-b->tower-b": second_link},
            sampler=ScriptedSampler([False, False]),
        )
        first_hop = Hop(
            hop_id="device-a->tower-a",
            source_node_id="device-a",
            destination_node_id="tower-a",
            link_profile_id="nr-shared-template",
            direction=Direction.UPLINK,
        )
        second_hop = Hop(
            hop_id="device-b->tower-b",
            source_node_id="device-b",
            destination_node_id="tower-b",
            link_profile_id="nr-shared-template",
            direction=Direction.UPLINK,
        )

        first = network_engine.transmit_packet(
            packet(), first_hop, requested_at_s=0.0, distance_m=10.0, tx_power_w=0.2
        )
        second_packet = PacketUnit(
            packet_id="object-2:pdu:0:packet:0",
            pdu_set_id="object-2:pdu:0",
            sequence_number=0,
            payload_bytes=packet().payload_bytes,
            created_at_s=0.0,
            network_header_bytes=40,
            transport_header_bytes=8,
            is_last=True,
        )
        second = network_engine.transmit_packet(
            second_packet,
            second_hop,
            requested_at_s=0.0,
            distance_m=10.0,
            tx_power_w=0.2,
        )

        self.assertEqual(first.attempts[0].reservation.queue_delay_s, 0.0)
        self.assertEqual(second.attempts[0].reservation.queue_delay_s, 0.0)

    def test_real_sixg_channel_drives_delivery_and_outage(self):
        from metrics.channel import SixGChannelModel

        def build():
            link = SharedLink(profile())
            return NetworkEngine(
                packetizer=FixedMtuPacketizer(),
                channel_model=SixGChannelModel(link.profile),
                shared_links={link.profile.profile_id: link},
                experiment_seed=77,
            )

        traffic = TrafficObject(
            object_id="real-channel-object",
            flow_id="real-channel-flow",
            source_id="device-1",
            destination_id="mec-1",
            created_at_s=0.0,
            payload_bytes=3_000,
            deadline_at_s=1.0,
            direction=Direction.UPLINK,
        )
        first = build().transmit_object_one_hop(
            traffic, hop(), distance_m=100.0, tx_power_w=0.2, line_of_sight=True
        )
        repeated = build().transmit_object_one_hop(
            traffic, hop(), distance_m=100.0, tx_power_w=0.2, line_of_sight=True
        )
        outage = build().transmit_object_one_hop(
            traffic, hop(), distance_m=2_000.0, tx_power_w=0.2, line_of_sight=True
        )

        self.assertTrue(first.all_delivered)
        self.assertEqual(first, repeated)
        self.assertTrue(all(
            item.attempts[0].channel_result.model_id == "sixg-v1"
            for item in first.packet_transmissions
        ))
        self.assertFalse(outage.all_delivered)
        self.assertTrue(all(
            item.outcome.status is DeliveryStatus.DROPPED
            and item.outcome.failure_reason == "channel_outage"
            for item in outage.packet_transmissions
        ))

    def test_route_uses_physical_hops_and_reframes_each_packet_per_link(self):
        from config.energy import get_power
        from metrics.energy import RadioPacketEnergyModel, TransportPacketEnergyModel
        from metrics.channel import SixGChannelModel

        radio = profile("nr-fr1-access")
        fibre = LinkProfile(
            profile_id="fibre-core",
            technology="Ethernet over fibre",
            generation="reference",
            band="wired",
            network_mtu_bytes=9_000,
            nominal_bit_rate_bps=10_000_000_000.0,
            propagation_speed_m_s=200_000_000.0,
            shared_resource_id="core-fibre",
            header_stack=HeaderStack(
                network_header_bytes=40,
                transport_header_bytes=8,
                link_header_bytes=18,
                link_trailer_bytes=4,
            ),
            channel_model_id="fixed-rate-wired-v1",
        )
        topology = PhysicalTopology.from_mapping(
            "handcheck_min", get_topology("handcheck_min")
        )
        route = RouteResolver(topology).resolve("r_d0_core")
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
            energy_models={
                radio.profile_id: RadioPacketEnergyModel(
                    get_power("device_radio_default"),
                    get_power("device_radio_default"),
                ),
                fibre.profile_id: TransportPacketEnergyModel(
                    get_power("transport_fibre_default")
                ),
            },
            experiment_seed=91,
        )
        traffic = TrafficObject(
            object_id="route-object",
            flow_id="route-flow",
            source_id="D0",
            destination_id="C0",
            created_at_s=0.0,
            payload_bytes=3_000,
            deadline_at_s=1.0,
            direction=Direction.UPLINK,
        )

        result = engine.transmit_object_route(
            traffic,
            route,
            hop_parameters={
                "D0->A0": HopTransmissionParameters(
                    tx_power_w=0.2,
                    line_of_sight=True,
                ),
                "A0->C0": HopTransmissionParameters(tx_power_w=0.0),
            },
        )

        self.assertTrue(result.all_delivered)
        self.assertIsNotNone(result.completed_at_s)
        self.assertEqual(len(result.packet_routes), 3)
        self.assertTrue(all(len(item.hop_transmissions) == 2 for item in result.packet_routes))
        first_radio = result.packet_routes[0].hop_transmissions[0]
        first_fibre = result.packet_routes[0].hop_transmissions[1]
        self.assertAlmostEqual(
            first_radio.attempts[0].channel_result.propagation_delay_s,
            route.hops[0].distance_m / radio.propagation_speed_m_s,
        )
        self.assertAlmostEqual(
            first_fibre.attempts[0].channel_result.propagation_delay_s,
            1_000.0 / fibre.propagation_speed_m_s,
        )
        self.assertEqual(
            first_radio.outcome.on_wire_bytes,
            result.packetization.packets[0].network_packet_bytes + 18,
        )
        self.assertEqual(
            first_fibre.outcome.on_wire_bytes,
            result.packetization.packets[0].network_packet_bytes + 22,
        )
        snapshot = engine.snapshot().transmission
        self.assertEqual(snapshot.final_outcome_count, 6)
        self.assertEqual(snapshot.offered_payload_bytes, 6_000)
        self.assertEqual(snapshot.useful_payload_bytes, 6_000)
        self.assertGreater(snapshot.tx_energy_j, 0.0)
        self.assertGreater(snapshot.rx_energy_j, 0.0)
        self.assertGreater(snapshot.transport_energy_j, 0.0)
        self.assertTrue(engine.snapshot().completeness.energy_complete)
        self.assertEqual(
            engine.snapshot().completeness.scope_id,
            "r3_exact_routed_packet_energy",
        )

    def test_downstream_mtu_shrink_is_rejected_instead_of_silent_truncation(self):
        tiny = LinkProfile(
            profile_id="tiny",
            technology="constrained-link",
            generation="test",
            band="wired",
            network_mtu_bytes=999,
            nominal_bit_rate_bps=1_000_000.0,
            propagation_speed_m_s=200_000_000.0,
            shared_resource_id="tiny-link",
        )
        channel = RecordingChannel([channel_result()])
        engine = NetworkEngine(
            packetizer=FixedMtuPacketizer(),
            channel_model={tiny.profile_id: channel},
            shared_links={tiny.profile_id: SharedLink(tiny)},
            sampler=ScriptedSampler([False]),
        )

        with self.assertRaisesRegex(ValueError, "fragmentation is not implemented"):
            engine.transmit_packet(
                packet(), hop(tiny.profile_id), distance_m=10.0, tx_power_w=0.0
            )
        self.assertEqual(channel.requests, [])

    def test_route_failure_prevents_downstream_transmission(self):
        radio = profile("nr-fr1-access")
        fibre = LinkProfile(
            profile_id="fibre-core",
            technology="fibre",
            generation="reference",
            band="wired",
            network_mtu_bytes=9_000,
            nominal_bit_rate_bps=1_000_000_000.0,
            propagation_speed_m_s=200_000_000.0,
            shared_resource_id="fibre",
        )
        radio_channel = RecordingChannel([channel_result(outage=True)])
        fibre_channel = RecordingChannel([channel_result()])
        route = RouteResolver(
            PhysicalTopology.from_mapping("handcheck_min", get_topology("handcheck_min"))
        ).resolve("r_d0_core")
        engine = NetworkEngine(
            packetizer=FixedMtuPacketizer(),
            channel_model={
                radio.profile_id: radio_channel,
                fibre.profile_id: fibre_channel,
            },
            shared_links={
                radio.profile_id: SharedLink(radio),
                fibre.profile_id: SharedLink(fibre),
            },
            sampler=ScriptedSampler([]),
        )
        traffic = TrafficObject(
            object_id="failed-route-object",
            flow_id="failed-route-flow",
            source_id="D0",
            destination_id="C0",
            created_at_s=0.0,
            payload_bytes=100,
            direction=Direction.UPLINK,
        )

        result = engine.transmit_object_route(
            traffic,
            route,
            hop_parameters={
                "D0->A0": HopTransmissionParameters(tx_power_w=0.2),
                "A0->C0": HopTransmissionParameters(tx_power_w=0.0),
            },
        )

        self.assertFalse(result.all_delivered)
        self.assertIsNone(result.completed_at_s)
        self.assertEqual(len(result.packet_routes[0].hop_transmissions), 1)
        self.assertEqual(fibre_channel.requests, [])

    def test_partial_energy_registry_is_rejected_at_composition(self):
        from config.energy import get_power
        from metrics.energy import RadioPacketEnergyModel

        radio = profile("radio")
        fibre = LinkProfile(
            profile_id="fibre",
            technology="fibre",
            generation="reference",
            band="wired",
            network_mtu_bytes=9_000,
            nominal_bit_rate_bps=1_000_000_000.0,
            propagation_speed_m_s=200_000_000.0,
            shared_resource_id="fibre-resource",
        )
        with self.assertRaisesRegex(ValueError, "does not cover"):
            NetworkEngine(
                packetizer=FixedMtuPacketizer(),
                channel_model=RecordingChannel([channel_result()]),
                shared_links={
                    radio.profile_id: SharedLink(radio),
                    fibre.profile_id: SharedLink(fibre),
                },
                energy_models={
                    radio.profile_id: RadioPacketEnergyModel(
                        get_power("device_radio_default"),
                        get_power("device_radio_default"),
                    )
                },
            )


if __name__ == "__main__":
    unittest.main()
