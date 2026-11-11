import unittest

from network import FixedMtuPacketizer, HeaderStack, LinkProfile, Plane, TrafficObject


def ethernet_profile() -> LinkProfile:
    return LinkProfile(
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


class FixedMtuPacketizerTests(unittest.TestCase):
    def test_segments_payload_and_accounts_every_packet_overhead(self):
        traffic = TrafficObject(
            object_id="object-1",
            flow_id="flow-1",
            source_id="device-1",
            destination_id="mec-1",
            created_at_s=0.0,
            payload_bytes=3000,
            deadline_at_s=0.1,
        )
        result = FixedMtuPacketizer().packetize(traffic, ethernet_profile())

        self.assertEqual([packet.payload_bytes for packet in result.packets], [1452, 1452, 96])
        self.assertEqual([packet.network_packet_bytes for packet in result.packets], [1500, 1500, 144])
        self.assertEqual(result.payload_bytes, 3000)
        self.assertEqual(result.on_wire_bytes, 3000 + 3 * (40 + 8 + 14 + 4))
        self.assertEqual(result.pdu_set.packet_count, 3)
        self.assertTrue(result.packets[-1].is_last)

    def test_packet_ids_are_deterministic_and_sequences_contiguous(self):
        traffic = TrafficObject(
            object_id="control-7",
            flow_id="control-flow",
            source_id="mec-1",
            destination_id="device-1",
            created_at_s=1.0,
            payload_bytes=2000,
            plane=Plane.CONTROL,
        )
        packetizer = FixedMtuPacketizer()
        first = packetizer.packetize(traffic, ethernet_profile())
        second = packetizer.packetize(traffic, ethernet_profile())

        self.assertEqual(first, second)
        self.assertEqual([packet.sequence_number for packet in first.packets], list(range(len(first.packets))))

    def test_zero_payload_control_object_has_one_header_only_packet(self):
        traffic = TrafficObject(
            object_id="ack-1",
            flow_id="control-flow",
            source_id="mec-1",
            destination_id="device-1",
            created_at_s=1.0,
            payload_bytes=0,
            plane=Plane.CONTROL,
            traffic_kind="ack",
        )
        result = FixedMtuPacketizer().packetize(traffic, ethernet_profile())

        self.assertEqual(len(result.packets), 1)
        self.assertEqual(result.packets[0].payload_bytes, 0)
        self.assertEqual(result.packets[0].on_wire_bytes, 66)

    def test_exact_result_converts_to_equivalent_batch(self):
        traffic = TrafficObject(
            object_id="object-2",
            flow_id="flow-2",
            source_id="device-2",
            destination_id="mec-1",
            created_at_s=2.0,
            payload_bytes=4096,
        )
        result = FixedMtuPacketizer().packetize(traffic, ethernet_profile())
        batch = result.to_batch()

        self.assertEqual(batch.packet_count, len(result.packets))
        self.assertEqual(batch.payload_bytes, result.payload_bytes)
        self.assertEqual(batch.on_wire_bytes, result.on_wire_bytes)


if __name__ == "__main__":
    unittest.main()

