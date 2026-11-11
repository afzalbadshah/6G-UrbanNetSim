import unittest

from config.energy import get_power
from metrics import (
    RadioPacketEnergyModel,
    TransportPacketEnergyModel,
    calculate_task_energy,
    compute_energy,
    incremental_compute_energy,
    packet_radio_energy,
    radio_energy,
    transport_energy,
    uav_energy,
)
from network import RetryPolicy
from tests.test_network_engine import channel_result, engine, hop, packet


class EnergyFormulaTests(unittest.TestCase):
    def test_legacy_objective_is_local_compute_or_remote_device_communication(self):
        local = calculate_task_energy(
            compute_it_energy_j=2.0,
            facility_overhead_energy_j=0.0,
            local_execution=True,
        )
        remote = calculate_task_energy(
            device_tx_energy_j=0.3,
            device_rx_energy_j=0.1,
            device_wait_energy_j=0.2,
            infrastructure_network_energy_j=1.0,
            compute_it_energy_j=2.0,
            facility_overhead_energy_j=0.4,
            local_execution=False,
        )

        self.assertEqual(local.objective_basis, "local_compute")
        self.assertAlmostEqual(local.objective_energy_j, 2.0)
        self.assertEqual(remote.objective_basis, "device_communication")
        self.assertAlmostEqual(remote.objective_energy_j, 0.4)
        self.assertAlmostEqual(local.device_battery_energy_j, 2.0)
        self.assertAlmostEqual(remote.device_battery_energy_j, 0.6)
        self.assertAlmostEqual(remote.total_energy_j, 4.0)
        with self.assertRaises(ValueError):
            calculate_task_energy(local_execution=1)

    def test_radio_separates_pa_input_tx_circuit_and_rx_circuit(self):
        profile = get_power("device_radio_default")
        result = radio_energy(profile, tx_active_s=1.0, rx_active_s=2.0)

        self.assertAlmostEqual(result.pa_input_energy_j, 0.2 / 0.35)
        self.assertAlmostEqual(result.tx_circuit_energy_j, 0.1)
        self.assertAlmostEqual(result.rx_circuit_energy_j, 0.1)
        self.assertAlmostEqual(result.tx_energy_j, (0.2 / 0.35) + 0.1)
        self.assertAlmostEqual(result.total_energy_j, (0.2 / 0.35) + 0.2)

    def test_packet_radio_energy_counts_only_reserved_transmission_airtime(self):
        network_engine, _, _, _ = engine(
            [channel_result(outage=True), channel_result()],
            [False],
            retry_policy=RetryPolicy(
                max_attempts=2,
                backoff_s=0.2,
                outage_observation_s=0.05,
            ),
        )
        transmitted = network_engine.transmit_packet(
            packet(), hop(), requested_at_s=0.0, distance_m=10.0, tx_power_w=0.2
        )

        result = packet_radio_energy(
            transmitted,
            get_power("device_radio_default"),
            get_power("device_radio_default"),
        )

        # The first attempt only observes an outage; it sends no data frame.
        self.assertAlmostEqual(result.tx_active_s, 0.008144)
        self.assertAlmostEqual(result.rx_active_s, result.tx_active_s)
        self.assertGreater(result.tx_energy_j, result.rx_energy_j)

    def test_outage_without_service_has_no_packet_tx_rx_or_transport_energy(self):
        network_engine, _, _, _ = engine(
            [channel_result(outage=True)],
            [],
            retry_policy=RetryPolicy(max_attempts=1, outage_observation_s=0.05),
        )
        outage = network_engine.transmit_packet(
            packet(), hop(), requested_at_s=0.0, distance_m=10.0, tx_power_w=0.2
        )
        radio = packet_radio_energy(
            outage,
            get_power("device_radio_default"),
            get_power("device_radio_default"),
        )
        transport = TransportPacketEnergyModel(
            get_power("transport_fibre_default")
        ).calculate(
            attempts=outage.attempts,
            on_wire_bytes_per_attempt=packet().on_wire_bytes,
        )

        self.assertEqual(radio.tx_active_s, 0.0)
        self.assertEqual(radio.rx_active_s, 0.0)
        self.assertEqual(radio.total_energy_j, 0.0)
        self.assertEqual(transport.transport_energy_j, 0.0)

    def test_zero_attempt_expiry_consumes_no_transmission_energy(self):
        network_engine, _, _, _ = engine([channel_result()], [False])
        expired = network_engine.transmit_packet(
            packet(),
            hop(),
            requested_at_s=2.0,
            distance_m=10.0,
            tx_power_w=0.2,
            deadline_at_s=1.0,
        )
        result = packet_radio_energy(
            expired,
            get_power("device_radio_default"),
            get_power("device_radio_default"),
        )
        self.assertEqual(result.total_energy_j, 0.0)

    def test_packet_helper_uses_actual_attempt_power_and_endpoint_circuits(self):
        low_engine, _, _, _ = engine([channel_result()], [False])
        high_engine, _, _, _ = engine([channel_result()], [False])
        low = low_engine.transmit_packet(
            packet(), hop(), requested_at_s=0.0, distance_m=10.0, tx_power_w=0.1
        )
        high = high_engine.transmit_packet(
            packet(), hop(), requested_at_s=0.0, distance_m=10.0, tx_power_w=0.2
        )
        transmitter = dict(get_power("device_radio_default"))
        receiver = {**transmitter, "rx_circuit_power_w": 0.2}

        low_energy = packet_radio_energy(low, transmitter, receiver)
        high_energy = packet_radio_energy(high, transmitter, receiver)

        self.assertGreater(high_energy.tx_energy_j, low_energy.tx_energy_j)
        self.assertEqual(high_energy.rx_energy_j, low_energy.rx_energy_j)
        self.assertAlmostEqual(
            high_energy.rx_energy_j,
            0.2 * high_energy.rx_active_s,
        )

    def test_packet_adapter_does_not_charge_a_transition_per_packet(self):
        network_engine, _, _, _ = engine([channel_result()], [False])
        transmitted = network_engine.transmit_packet(
            packet(), hop(), requested_at_s=0.0, distance_m=10.0, tx_power_w=0.2
        )
        endpoint = {
            **dict(get_power("device_radio_default")),
            "transition_energy_j": 10.0,
        }

        result = packet_radio_energy(transmitted, endpoint, endpoint)

        self.assertEqual(result.transition_energy_j, 0.0)

    def test_compute_energy_separates_it_and_pue_overhead(self):
        result = compute_energy(
            get_power("mec_compute"),
            duration_s=2.0,
            utilization_ratio=0.5,
        )
        self.assertEqual(result.it_power_w, 650.0)
        self.assertEqual(result.it_energy_j, 1_300.0)
        self.assertAlmostEqual(result.facility_overhead_energy_j, 455.0)
        self.assertAlmostEqual(result.total_energy_j, 1_755.0)

    def test_incremental_shared_compute_excludes_repeated_idle_power(self):
        result = incremental_compute_energy(
            get_power("mec_compute"),
            duration_s=2.0,
            utilization_before_ratio=0.25,
            utilization_after_ratio=0.5,
        )

        self.assertEqual(result.incremental_it_power_w, 175.0)
        self.assertEqual(result.incremental_it_energy_j, 350.0)
        self.assertAlmostEqual(result.facility_overhead_energy_j, 122.5)
        self.assertAlmostEqual(result.total_energy_j, 472.5)

    def test_incremental_shared_compute_rejects_decreasing_utilization(self):
        with self.assertRaises(ValueError):
            incremental_compute_energy(
                get_power("mec_compute"),
                duration_s=1.0,
                utilization_before_ratio=0.5,
                utilization_after_ratio=0.25,
            )

    def test_transport_applies_exactly_one_selected_basis(self):
        base = {
            "kind": "transport",
            "active_power_w": 50.0,
            "idle_power_w": 20.0,
            "energy_per_bit_j": 1.0e-8,
        }
        by_bit = transport_energy(
            {**base, "accounting_method": "energy_per_bit"},
            active_duration_s=0.1,
            on_wire_bytes=1_000,
        )
        by_power = transport_energy(
            {**base, "accounting_method": "active_power"},
            active_duration_s=0.1,
            on_wire_bytes=1_000,
        )

        self.assertEqual(by_bit.active_energy_j, 0.0)
        self.assertAlmostEqual(by_bit.per_bit_energy_j, 0.00008)
        self.assertEqual(by_power.per_bit_energy_j, 0.0)
        self.assertEqual(by_power.active_energy_j, 5.0)
        with self.assertRaises(ValueError):
            transport_energy(base, active_duration_s=0.1, on_wire_bytes=1_000)

    def test_uav_modes_remain_separately_reportable(self):
        result = uav_energy(
            get_power("uav_bs_mobility"),
            hover_duration_s=2.0,
            propulsion_duration_s=0.0,
            mobility_duration_s=3.0,
        )
        self.assertEqual(result.hover_energy_j, 1_300.0)
        self.assertEqual(result.propulsion_energy_j, 0.0)
        self.assertEqual(result.mobility_energy_j, 2_100.0)
        self.assertEqual(result.total_energy_j, 3_400.0)

    def test_engine_adapters_return_separate_transmission_energy_domains(self):
        network_engine, _, _, _ = engine([channel_result()], [False])
        transmitted = network_engine.transmit_packet(
            packet(), hop(), requested_at_s=0.0, distance_m=10.0, tx_power_w=0.2
        )
        radio = RadioPacketEnergyModel(
            get_power("device_radio_default"),
            get_power("device_radio_default"),
        ).calculate(
            attempts=transmitted.attempts,
            on_wire_bytes_per_attempt=packet().on_wire_bytes,
        )
        transport = TransportPacketEnergyModel(
            get_power("transport_fibre_default")
        ).calculate(
            attempts=transmitted.attempts,
            on_wire_bytes_per_attempt=packet().on_wire_bytes,
        )

        self.assertGreater(radio.tx_energy_j, 0.0)
        self.assertGreater(radio.rx_energy_j, 0.0)
        self.assertEqual(radio.transport_energy_j, 0.0)
        self.assertEqual(transport.tx_energy_j, 0.0)
        self.assertEqual(transport.rx_energy_j, 0.0)
        self.assertGreater(transport.transport_energy_j, 0.0)

    def test_engine_adapter_uses_actual_attempt_power_not_profile_default(self):
        low_engine, _, _, _ = engine([channel_result()], [False])
        high_engine, _, _, _ = engine([channel_result()], [False])
        low = low_engine.transmit_packet(
            packet(), hop(), requested_at_s=0.0, distance_m=10.0, tx_power_w=0.1
        )
        high = high_engine.transmit_packet(
            packet(), hop(), requested_at_s=0.0, distance_m=10.0, tx_power_w=0.2
        )
        model = RadioPacketEnergyModel(
            get_power("device_radio_default"),
            get_power("device_radio_default"),
        )

        low_energy = model.calculate(
            attempts=low.attempts,
            on_wire_bytes_per_attempt=packet().on_wire_bytes,
        )
        high_energy = model.calculate(
            attempts=high.attempts,
            on_wire_bytes_per_attempt=packet().on_wire_bytes,
        )

        self.assertGreater(high_energy.tx_energy_j, low_energy.tx_energy_j)
        self.assertEqual(high_energy.rx_energy_j, low_energy.rx_energy_j)

    def test_engine_adapter_rejects_radiated_power_above_endpoint_maximum(self):
        network_engine, _, _, _ = engine([channel_result()], [False])
        transmitted = network_engine.transmit_packet(
            packet(), hop(), requested_at_s=0.0, distance_m=10.0, tx_power_w=0.4
        )
        model = RadioPacketEnergyModel(
            get_power("device_radio_default"),
            get_power("access_radio_default"),
        )

        with self.assertRaises(ValueError):
            model.calculate(
                attempts=transmitted.attempts,
                on_wire_bytes_per_attempt=packet().on_wire_bytes,
            )


if __name__ == "__main__":
    unittest.main()
