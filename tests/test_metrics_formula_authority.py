"""Dimensional and delegation gates for metrics consolidation Task 63."""

import unittest
from unittest.mock import patch

from metrics.cost import linear_cost, sum_costs
from metrics.delay import (
    propagation_delay_s,
    transmission_delay_s,
    sum_delay_components_s,
)
from metrics.energy import energy_from_power_duration_j
from network.contracts import (
    ChannelRequest,
    Direction,
    HeaderStack,
    LinkProfile,
)
from network.links import SharedLink
from network.wired import FixedRateChannelModel


class TestCanonicalScalarFormulas(unittest.TestCase):
    def test_transmission_uses_bytes_and_bits_per_second(self):
        self.assertAlmostEqual(
            transmission_delay_s(on_wire_bytes=1_500, achievable_rate_bps=12_000_000.0),
            0.001,
        )
        self.assertGreater(
            transmission_delay_s(on_wire_bytes=3_000, achievable_rate_bps=12_000_000.0),
            transmission_delay_s(on_wire_bytes=1_500, achievable_rate_bps=12_000_000.0),
        )

    def test_propagation_uses_metres_and_metres_per_second(self):
        self.assertAlmostEqual(
            propagation_delay_s(
                distance_m=300_000.0,
                propagation_speed_m_s=300_000_000.0,
            ),
            0.001,
        )

    def test_delay_inputs_are_validated(self):
        with self.assertRaises(ValueError):
            transmission_delay_s(on_wire_bytes=1.5, achievable_rate_bps=1.0)
        with self.assertRaises(ValueError):
            transmission_delay_s(on_wire_bytes=1, achievable_rate_bps=0.0)
        with self.assertRaises(ValueError):
            propagation_delay_s(distance_m=-1.0, propagation_speed_m_s=1.0)
        with self.assertRaises(ValueError):
            sum_delay_components_s(0.1, -0.1)

    def test_power_duration_is_the_energy_authority(self):
        self.assertEqual(
            energy_from_power_duration_j(power_w=2.0, duration_s=0.25),
            0.5,
        )
        with self.assertRaises(ValueError):
            energy_from_power_duration_j(power_w=-1.0, duration_s=1.0)

    def test_linear_cost_is_full_precision_and_sum_is_validated(self):
        first = linear_cost(measured_quantity=250.0, price_per_unit=0.004)
        second = linear_cost(measured_quantity=2.0, price_per_unit=0.125)
        self.assertEqual(first, 1.0)
        self.assertEqual(sum_costs(first, second), 1.25)
        with self.assertRaises(ValueError):
            linear_cost(measured_quantity=1.0, price_per_unit=-0.01)


class TestPacketPathDelegation(unittest.TestCase):
    def setUp(self):
        self.profile = LinkProfile(
            profile_id="test-wired",
            technology="Ethernet over fibre",
            generation="reference",
            band="wired",
            nominal_bit_rate_bps=1_000_000.0,
            network_mtu_bytes=1_500,
            propagation_speed_m_s=200_000_000.0,
            shared_resource_id="test-resource",
            header_stack=HeaderStack(link_header_bytes=14, link_trailer_bytes=4),
            channel_model_id="fixed-rate-wired-v1",
        )

    def test_shared_link_calls_canonical_transmission_formula(self):
        link = SharedLink(self.profile)
        with patch("network.links.transmission_delay_s", return_value=0.25) as formula:
            reservation = link.reserve(
                requested_at_s=0.0,
                on_wire_bytes=1_000,
            )

        formula.assert_called_once_with(
            on_wire_bytes=1_000,
            achievable_rate_bps=1_000_000.0,
        )
        self.assertEqual(reservation.transmission_delay_s, 0.25)

    def test_wired_channel_calls_canonical_propagation_formula(self):
        model = FixedRateChannelModel(self.profile)
        request = ChannelRequest(
            link_profile_id=self.profile.profile_id,
            hop_id="hop-1",
            distance_m=10_000.0,
            requested_at_s=0.0,
            on_wire_bytes=1_000,
            direction=Direction.BACKHAUL,
            tx_power_w=0.0,
        )
        with patch("network.wired.propagation_delay_s", return_value=0.5) as formula:
            result = model.evaluate(request)

        formula.assert_called_once_with(
            distance_m=10_000.0,
            propagation_speed_m_s=200_000_000.0,
        )
        self.assertEqual(result.propagation_delay_s, 0.5)


if __name__ == "__main__":
    unittest.main()
