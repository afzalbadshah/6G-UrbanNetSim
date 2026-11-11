import unittest

from network import (
    ChannelRequest,
    Direction,
    FixedRateChannelModel,
    HeaderStack,
    LinkProfile,
)


def wired_profile(profile_id="fibre-core"):
    return LinkProfile(
        profile_id=profile_id,
        technology="Ethernet over fibre",
        generation="reference",
        band="wired",
        network_mtu_bytes=9_000,
        nominal_bit_rate_bps=10_000_000_000.0,
        propagation_speed_m_s=200_000_000.0,
        shared_resource_id="core-fibre",
        header_stack=HeaderStack(link_header_bytes=18, link_trailer_bytes=4),
        channel_model_id="fixed-rate-wired-v1",
    )


class FixedRateChannelModelTests(unittest.TestCase):
    def test_uses_profile_rate_and_physical_route_distance(self):
        profile = wired_profile()
        result = FixedRateChannelModel(profile).evaluate(
            ChannelRequest(
                link_profile_id=profile.profile_id,
                hop_id="A0->C0",
                requested_at_s=0.0,
                distance_m=1_000.0,
                on_wire_bytes=1_522,
                direction=Direction.BACKHAUL,
                tx_power_w=0.0,
            )
        )

        self.assertEqual(result.model_id, "fixed-rate-wired-v1")
        self.assertEqual(result.achievable_rate_bps, 10_000_000_000.0)
        self.assertAlmostEqual(result.propagation_delay_s, 0.000005)
        self.assertFalse(result.outage)
        self.assertEqual(result.packet_error_probability, 0.0)

    def test_unknown_profile_and_bad_registry_are_rejected(self):
        profile = wired_profile()
        model = FixedRateChannelModel(profile)
        request = ChannelRequest(
            link_profile_id="missing",
            hop_id="hop",
            requested_at_s=0.0,
            distance_m=1.0,
            on_wire_bytes=1,
            direction=Direction.BACKHAUL,
            tx_power_w=0.0,
        )
        with self.assertRaises(KeyError):
            model.evaluate(request)
        with self.assertRaises(ValueError):
            FixedRateChannelModel(profile, {"wrong": profile})


if __name__ == "__main__":
    unittest.main()
