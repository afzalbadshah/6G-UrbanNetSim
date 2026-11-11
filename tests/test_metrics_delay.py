import unittest
from types import SimpleNamespace

from metrics import (
    PacketDelayBreakdown,
    PacketRouteDelay,
    TaskDelayBreakdown,
    calculate_task_delay,
    packet_delay_breakdown,
    packet_route_delay,
    transmission_delay_s,
)
from network import Medium, PacketRouteTransmission, RetryPolicy
from tests.test_network_engine import channel_result, engine, hop, packet


class DelayBreakdownTests(unittest.TestCase):
    @staticmethod
    def _task_delay_with_network_values(
        *, component_s: float, network_s: float
    ) -> TaskDelayBreakdown:
        return TaskDelayBreakdown(
            handover_delay_s=0.0,
            coverage_outage_delay_s=0.0,
            wireless_queue_delay_s=component_s,
            wired_queue_delay_s=0.0,
            wireless_transmission_delay_s=0.0,
            wired_transmission_delay_s=0.0,
            wireless_propagation_delay_s=0.0,
            wired_propagation_delay_s=0.0,
            retry_backoff_delay_s=0.0,
            channel_outage_delay_s=0.0,
            terminal_wait_delay_s=0.0,
            processing_delay_s=0.0,
            network_delay_s=network_s,
            total_delay_s=network_s,
        )

    def test_timestamp_roundoff_within_one_nanosecond_is_accepted(self):
        delay = self._task_delay_with_network_values(
            component_s=0.0001,
            network_s=0.0001 + 2e-12,
        )

        self.assertAlmostEqual(
            delay.calculated_network_delay_s,
            delay.network_delay_s,
            delta=1e-9,
        )

    def test_meaningful_network_accounting_mismatch_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            r"network components must sum.*difference=",
        ):
            self._task_delay_with_network_values(
                component_s=0.0001,
                network_s=0.0001 + 2e-8,
            )

    def test_legacy_objective_and_packet_overhead_are_distinct(self):
        delay = calculate_task_delay(
            handover_delay_s=0.002,
            coverage_outage_duration_s=0.003,
            processing_delay_s=0.010,
        )

        # tier_constant is removed: legacy objective is now propagation + handover.
        self.assertAlmostEqual(delay.legacy_objective_delay_s, 0.002)
        self.assertAlmostEqual(delay.legacy_complete_latency_s, 0.012)
        # The first 2 ms of outage overlap handover and must not be counted twice.
        self.assertAlmostEqual(delay.packet_observation_overhead_s, 0.001)
        self.assertAlmostEqual(delay.observed_complete_latency_s, 0.013)
        self.assertAlmostEqual(delay.complete_latency_s, 0.013)
        self.assertAlmostEqual(delay.total_delay_s, delay.observed_complete_latency_s)

    def test_transmission_formula_uses_bytes_bits_and_bits_per_second(self):
        self.assertAlmostEqual(
            transmission_delay_s(
                on_wire_bytes=1_000,
                achievable_rate_bps=10_000_000.0,
            ),
            0.0008,
        )

    def test_one_attempt_components_sum_exactly_once(self):
        network_engine, _, _, _ = engine([channel_result()], [False])
        result = network_engine.transmit_packet(
            packet(), hop(), requested_at_s=0.0, distance_m=100.0, tx_power_w=0.2
        )

        delay = packet_delay_breakdown(result)

        self.assertEqual(delay.queue_delay_s, 0.0)
        self.assertAlmostEqual(delay.transmission_delay_s, 0.008144)
        self.assertAlmostEqual(delay.propagation_delay_s, 0.001)
        self.assertEqual(delay.retry_backoff_s, 0.0)
        self.assertEqual(delay.outage_observation_s, 0.0)
        self.assertEqual(delay.terminal_wait_s, 0.0)
        self.assertAlmostEqual(delay.component_sum_s, delay.total_delay_s)

    def test_retry_backoff_and_outage_observation_are_separate(self):
        network_engine, _, _, _ = engine(
            [channel_result(outage=True), channel_result()],
            [False],
            retry_policy=RetryPolicy(
                max_attempts=2,
                backoff_s=0.2,
                outage_observation_s=0.05,
            ),
        )
        result = network_engine.transmit_packet(
            packet(), hop(), requested_at_s=0.0, distance_m=100.0, tx_power_w=0.2
        )

        delay = packet_delay_breakdown(result)

        self.assertAlmostEqual(delay.outage_observation_s, 0.05)
        self.assertAlmostEqual(delay.retry_backoff_s, 0.2)
        self.assertAlmostEqual(delay.transmission_delay_s, 0.008144)
        self.assertAlmostEqual(delay.propagation_delay_s, 0.001)
        self.assertAlmostEqual(delay.total_delay_s, 0.259144)

    def test_retry_propagation_is_observed_without_inflating_1_1_objective(self):
        network_engine, _, _, _ = engine(
            [channel_result(probability=0.5)],
            [True, False],
            retry_policy=RetryPolicy(max_attempts=2, backoff_s=0.1),
        )
        result = network_engine.transmit_packet(
            packet(), hop(), requested_at_s=0.0, distance_m=100.0, tx_power_w=0.2
        )
        hop_delay = packet_delay_breakdown(result)
        critical_packet = PacketRouteDelay(
            packet_id=packet().packet_id,
            hop_delays=(hop_delay,),
            elapsed_delay_s=hop_delay.total_delay_s,
        )
        route = SimpleNamespace(
            hops=(SimpleNamespace(link=SimpleNamespace(medium=Medium.RADIO)),)
        )
        task_delay = calculate_task_delay(
            critical_packet_delay=critical_packet,
            route=route,
            network_elapsed_s=hop_delay.total_delay_s,
            handover_delay_s=0.002,
            processing_delay_s=0.010,
        )

        self.assertAlmostEqual(hop_delay.propagation_delay_s, 0.001)
        self.assertAlmostEqual(hop_delay.retry_propagation_s, 0.001)
        # tier_constant removed: legacy objective = propagation(0.001) + handover(0.002).
        self.assertAlmostEqual(task_delay.legacy_objective_delay_s, 0.003)
        self.assertAlmostEqual(task_delay.retry_propagation_delay_s, 0.001)
        self.assertAlmostEqual(task_delay.retry_backoff_delay_s, 0.1)
        self.assertAlmostEqual(
            task_delay.observed_complete_latency_s, task_delay.total_delay_s
        )

    def test_fifo_queue_and_zero_attempt_terminal_wait_are_not_service(self):
        network_engine, channel, _, link = engine([channel_result()], [False])
        link.reserve(
            requested_at_s=0.0,
            on_wire_bytes=1_000,
            achievable_rate_bps=8_000.0,
        )
        expired = network_engine.transmit_packet(
            packet(),
            hop(),
            requested_at_s=0.0,
            distance_m=100.0,
            tx_power_w=0.2,
            deadline_at_s=0.5,
        )

        delay = packet_delay_breakdown(expired)

        self.assertEqual(delay.queue_delay_s, 0.0)
        self.assertEqual(delay.transmission_delay_s, 0.0)
        self.assertEqual(delay.terminal_wait_s, 0.5)
        self.assertEqual(delay.total_delay_s, 0.5)
        self.assertEqual(channel.requests, [])

    def test_packet_route_delay_is_sum_of_contiguous_hops(self):
        network_engine, _, _, _ = engine([channel_result()], [False])
        result = network_engine.transmit_packet(
            packet(), hop(), requested_at_s=0.0, distance_m=100.0, tx_power_w=0.2
        )
        routed = packet_route_delay(
            PacketRouteTransmission(
                packet_id=packet().packet_id,
                hop_transmissions=(result,),
            )
        )

        self.assertEqual(len(routed.hop_delays), 1)
        self.assertAlmostEqual(routed.elapsed_delay_s, result.outcome.ended_at_s)

    def test_task_delay_keeps_wireless_wired_access_and_processing_separate(self):
        radio = PacketDelayBreakdown(
            queue_delay_s=0.002,
            transmission_delay_s=0.004,
            propagation_delay_s=0.001,
            retry_backoff_s=0.003,
            outage_observation_s=0.0,
            terminal_wait_s=0.0,
            total_delay_s=0.010,
        )
        wired = PacketDelayBreakdown(
            queue_delay_s=0.005,
            transmission_delay_s=0.006,
            propagation_delay_s=0.007,
            retry_backoff_s=0.0,
            outage_observation_s=0.008,
            terminal_wait_s=0.0,
            total_delay_s=0.026,
        )
        critical_packet = PacketRouteDelay(
            packet_id="packet-1",
            hop_delays=(radio, wired),
            elapsed_delay_s=0.036,
        )
        route = SimpleNamespace(
            hops=(
                SimpleNamespace(link=SimpleNamespace(medium=Medium.RADIO)),
                SimpleNamespace(link=SimpleNamespace(medium=Medium.WIRED)),
            )
        )

        delay = calculate_task_delay(
            critical_packet_delay=critical_packet,
            route=route,
            network_elapsed_s=0.036,
            handover_delay_s=0.010,
            handover_interruption_s=0.020,
            coverage_outage_duration_s=0.030,
            processing_delay_s=0.040,
        )

        self.assertEqual(delay.handover_delay_s, 0.020)
        self.assertAlmostEqual(delay.coverage_outage_delay_s, 0.010)
        self.assertEqual(delay.wireless_queue_delay_s, 0.002)
        self.assertEqual(delay.wired_queue_delay_s, 0.005)
        self.assertEqual(delay.wireless_transmission_delay_s, 0.004)
        self.assertEqual(delay.wired_transmission_delay_s, 0.006)
        self.assertEqual(delay.wireless_propagation_delay_s, 0.001)
        self.assertEqual(delay.wired_propagation_delay_s, 0.007)
        self.assertEqual(delay.retry_backoff_delay_s, 0.003)
        self.assertEqual(delay.channel_outage_delay_s, 0.008)
        self.assertEqual(delay.processing_delay_s, 0.040)
        self.assertAlmostEqual(delay.network_delay_s, 0.036)
        self.assertAlmostEqual(delay.total_delay_s, 0.106)
        self.assertAlmostEqual(delay.total_ms, 106.0)


if __name__ == "__main__":
    unittest.main()
