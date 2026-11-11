import unittest
from math import nan

from network import (
    ChannelRequest,
    ChannelResult,
    Direction,
    HeaderStack,
    LinkProfile,
    PacketBatch,
    TrafficObject,
    TransmissionOutcome,
    DeliveryStatus,
)


class NetworkContractTests(unittest.TestCase):
    def test_traffic_object_rejects_deadline_before_creation(self):
        with self.assertRaises(ValueError):
            TrafficObject(
                object_id="object-1",
                flow_id="flow-1",
                source_id="device-1",
                destination_id="mec-1",
                created_at_s=2.0,
                payload_bytes=10,
                deadline_at_s=1.0,
            )

    def test_traffic_object_rejects_fractional_bytes(self):
        with self.assertRaises(ValueError):
            TrafficObject(
                object_id="object-1",
                flow_id="flow-1",
                source_id="device-1",
                destination_id="mec-1",
                created_at_s=0.0,
                payload_bytes=1.5,
            )

    def test_traffic_object_rejects_untyped_direction(self):
        with self.assertRaises(ValueError):
            TrafficObject(
                object_id="object-1",
                flow_id="flow-1",
                source_id="device-1",
                destination_id="mec-1",
                created_at_s=0.0,
                payload_bytes=1,
                direction="uplink",
            )

    def test_link_profile_exposes_payload_capacity(self):
        profile = LinkProfile(
            profile_id="ethernet-ipv6-udp",
            technology="ethernet",
            generation="ieee-802.3",
            band="wired",
            network_mtu_bytes=1500,
            nominal_bit_rate_bps=1_000_000_000.0,
            propagation_speed_m_s=200_000_000.0,
            shared_resource_id="lan-1",
            header_stack=HeaderStack(
                network_header_bytes=40,
                transport_header_bytes=8,
                link_header_bytes=14,
                link_trailer_bytes=4,
            ),
        )
        self.assertEqual(profile.max_payload_bytes, 1452)

    def test_link_profile_rejects_mtu_without_payload_capacity(self):
        with self.assertRaises(ValueError):
            LinkProfile(
                profile_id="bad",
                technology="test",
                generation="test",
                band="test",
                network_mtu_bytes=48,
                nominal_bit_rate_bps=1.0,
                propagation_speed_m_s=1.0,
                shared_resource_id="resource",
                header_stack=HeaderStack(network_header_bytes=40, transport_header_bytes=8),
            )

    def test_channel_result_allows_zero_rate_only_during_outage(self):
        outage = ChannelResult(
            model_id="test",
            sinr_db=-20.0,
            achievable_rate_bps=0.0,
            outage=True,
            packet_error_probability=1.0,
            propagation_delay_s=0.001,
        )
        self.assertTrue(outage.outage)
        with self.assertRaises(ValueError):
            ChannelResult(
                model_id="test",
                sinr_db=10.0,
                achievable_rate_bps=0.0,
                outage=False,
                packet_error_probability=0.0,
                propagation_delay_s=0.001,
            )

    def test_contracts_reject_non_finite_values(self):
        with self.assertRaises(ValueError):
            ChannelResult(
                model_id="test",
                sinr_db=nan,
                achievable_rate_bps=1.0,
                outage=False,
                packet_error_probability=0.0,
                propagation_delay_s=0.001,
            )

    def test_channel_request_validates_optional_elevation(self):
        with self.assertRaises(ValueError):
            ChannelRequest(
                link_profile_id="nr-fr3",
                hop_id="device-to-mec",
                requested_at_s=0.0,
                distance_m=100.0,
                on_wire_bytes=100,
                direction=Direction.UPLINK,
                tx_power_w=0.2,
                elevation_angle_deg=91.0,
            )

    def test_packet_batch_rejects_wire_bytes_below_payload(self):
        with self.assertRaises(ValueError):
            PacketBatch(
                batch_id="batch-1",
                pdu_set_id="set-1",
                packet_count=1,
                payload_bytes=10,
                on_wire_bytes=9,
                created_at_s=0.0,
            )

    def test_direction_values_are_stable_strings(self):
        self.assertEqual(Direction.UPLINK.value, "uplink")

    def test_zero_attempt_expiry_is_not_counted_as_transmission(self):
        outcome = TransmissionOutcome(
            unit_id="packet-expired",
            hop_id="hop-1",
            status=DeliveryStatus.EXPIRED,
            started_at_s=2.0,
            ended_at_s=2.0,
            attempts=0,
            payload_bytes=100,
            on_wire_bytes=0,
            useful_payload_bytes=0,
            failure_reason="deadline_before_attempt",
        )
        self.assertEqual(outcome.attempts, 0)
        self.assertEqual(outcome.on_wire_bytes, 0)

    def test_zero_attempt_nonexpiry_and_fake_wire_bytes_are_rejected(self):
        common = dict(
            unit_id="packet-1",
            hop_id="hop-1",
            started_at_s=0.0,
            ended_at_s=0.0,
            attempts=0,
            payload_bytes=100,
            useful_payload_bytes=0,
        )
        with self.assertRaises(ValueError):
            TransmissionOutcome(
                status=DeliveryStatus.DROPPED,
                on_wire_bytes=0,
                **common,
            )
        with self.assertRaises(ValueError):
            TransmissionOutcome(
                status=DeliveryStatus.EXPIRED,
                on_wire_bytes=100,
                **common,
            )


if __name__ == "__main__":
    unittest.main()
