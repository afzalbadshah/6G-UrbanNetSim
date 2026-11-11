import unittest

from network import HeaderStack, LinkProfile, SharedLink


def link_profile() -> LinkProfile:
    return LinkProfile(
        profile_id="test-link",
        technology="ethernet",
        generation="test",
        band="wired",
        network_mtu_bytes=1500,
        nominal_bit_rate_bps=1_000_000.0,
        propagation_speed_m_s=200_000_000.0,
        shared_resource_id="shared-lan",
        header_stack=HeaderStack(),
    )


class SharedLinkTests(unittest.TestCase):
    def test_first_packet_serializes_without_queue_delay(self):
        reservation = SharedLink(link_profile()).reserve(
            requested_at_s=1.0,
            on_wire_bytes=1000,
            propagation_delay_s=0.001,
        )

        self.assertAlmostEqual(reservation.queue_delay_s, 0.0)
        self.assertAlmostEqual(reservation.transmission_delay_s, 0.008)
        self.assertAlmostEqual(reservation.transmission_ended_at_s, 1.008)
        self.assertAlmostEqual(reservation.arrival_at_s, 1.009)

    def test_simultaneous_packets_share_capacity_by_queueing(self):
        link = SharedLink(link_profile())
        first = link.reserve(requested_at_s=1.0, on_wire_bytes=1000)
        second = link.reserve(requested_at_s=1.0, on_wire_bytes=1000)

        self.assertAlmostEqual(first.transmission_ended_at_s, 1.008)
        self.assertAlmostEqual(second.transmission_started_at_s, 1.008)
        self.assertAlmostEqual(second.queue_delay_s, 0.008)
        self.assertAlmostEqual(second.transmission_ended_at_s, 1.016)

    def test_idle_gap_does_not_create_queue_delay(self):
        link = SharedLink(link_profile())
        link.reserve(requested_at_s=0.0, on_wire_bytes=1000)
        later = link.reserve(requested_at_s=2.0, on_wire_bytes=1000)
        self.assertAlmostEqual(later.queue_delay_s, 0.0)
        self.assertAlmostEqual(later.transmission_started_at_s, 2.0)

    def test_channel_rate_overrides_nominal_rate_for_reservation(self):
        reservation = SharedLink(link_profile()).reserve(
            requested_at_s=0.0,
            on_wire_bytes=1000,
            achievable_rate_bps=2_000_000.0,
        )
        self.assertAlmostEqual(reservation.transmission_delay_s, 0.004)

    def test_reset_sets_new_availability_time(self):
        link = SharedLink(link_profile())
        link.reserve(requested_at_s=0.0, on_wire_bytes=1000)
        link.reset(3.0)
        reservation = link.reserve(requested_at_s=2.0, on_wire_bytes=1000)
        self.assertAlmostEqual(reservation.queue_delay_s, 1.0)

    def test_invalid_rate_is_rejected(self):
        with self.assertRaises(ValueError):
            SharedLink(link_profile()).reserve(
                requested_at_s=0.0,
                on_wire_bytes=1000,
                achievable_rate_bps=0.0,
            )


if __name__ == "__main__":
    unittest.main()

