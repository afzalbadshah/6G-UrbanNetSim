import unittest

from network import DeliveryStatus, TransmissionLedger, TransmissionOutcome


def outcome(
    unit_id: str,
    status: DeliveryStatus,
    *,
    payload_bytes: int,
    on_wire_bytes: int,
    useful_payload_bytes: int,
    attempts: int = 1,
    hop_id: str = "hop-1",
) -> TransmissionOutcome:
    return TransmissionOutcome(
        unit_id=unit_id,
        hop_id=hop_id,
        status=status,
        started_at_s=0.0,
        ended_at_s=0.01,
        attempts=attempts,
        payload_bytes=payload_bytes,
        on_wire_bytes=on_wire_bytes,
        useful_payload_bytes=useful_payload_bytes,
        tx_energy_j=0.2,
        rx_energy_j=0.1,
        transport_energy_j=0.05,
        monetary_cost=0.001,
    )


class TransmissionLedgerTests(unittest.TestCase):
    def test_aggregates_final_per_hop_outcomes_once(self):
        ledger = TransmissionLedger()
        ledger.record(
            outcome(
                "packet-1",
                DeliveryStatus.DELIVERED,
                payload_bytes=100,
                on_wire_bytes=240,
                useful_payload_bytes=100,
                attempts=2,
            )
        )
        ledger.record(
            outcome(
                "packet-2",
                DeliveryStatus.DROPPED,
                payload_bytes=50,
                on_wire_bytes=60,
                useful_payload_bytes=0,
            )
        )

        snapshot = ledger.snapshot()
        self.assertEqual(snapshot.final_outcome_count, 2)
        self.assertEqual(snapshot.delivered_count, 1)
        self.assertEqual(snapshot.dropped_count, 1)
        self.assertEqual(snapshot.expired_count, 0)
        self.assertEqual(snapshot.attempt_count, 3)
        self.assertEqual(snapshot.retransmission_count, 1)
        self.assertEqual(snapshot.offered_payload_bytes, 150)
        self.assertEqual(snapshot.on_wire_bytes, 300)
        self.assertEqual(snapshot.useful_payload_bytes, 100)
        self.assertAlmostEqual(snapshot.delivery_ratio, 0.5)
        self.assertAlmostEqual(snapshot.wire_efficiency_ratio, 1.0 / 3.0)
        self.assertAlmostEqual(snapshot.tx_energy_j, 0.4)
        self.assertAlmostEqual(snapshot.rx_energy_j, 0.2)
        self.assertAlmostEqual(snapshot.transport_energy_j, 0.1)
        self.assertAlmostEqual(snapshot.total_network_energy_j, 0.7)
        self.assertAlmostEqual(snapshot.monetary_cost, 0.002)

    def test_rejects_duplicate_final_outcome_for_same_unit_and_hop(self):
        ledger = TransmissionLedger()
        final = outcome(
            "packet-1",
            DeliveryStatus.DELIVERED,
            payload_bytes=100,
            on_wire_bytes=120,
            useful_payload_bytes=100,
        )
        ledger.record(final)
        with self.assertRaises(ValueError):
            ledger.record(final)

    def test_rejects_intermediate_outcome(self):
        ledger = TransmissionLedger()
        with self.assertRaises(ValueError):
            ledger.record(
                outcome(
                    "packet-1",
                    DeliveryStatus.QUEUED,
                    payload_bytes=100,
                    on_wire_bytes=120,
                    useful_payload_bytes=0,
                )
            )

    def test_non_delivery_cannot_claim_useful_payload(self):
        with self.assertRaises(ValueError):
            outcome(
                "packet-1",
                DeliveryStatus.DROPPED,
                payload_bytes=100,
                on_wire_bytes=120,
                useful_payload_bytes=100,
            )

    def test_clear_resets_counts_and_idempotency_keys(self):
        ledger = TransmissionLedger()
        final = outcome(
            "packet-1",
            DeliveryStatus.EXPIRED,
            payload_bytes=100,
            on_wire_bytes=120,
            useful_payload_bytes=0,
        )
        ledger.record(final)
        ledger.clear()
        ledger.record(final)
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger.snapshot().final_outcome_count, 1)

    def test_exact_outcome_retention_is_explicit(self):
        final = outcome(
            "packet-1",
            DeliveryStatus.DELIVERED,
            payload_bytes=100,
            on_wire_bytes=120,
            useful_payload_bytes=100,
        )
        aggregate_only = TransmissionLedger()
        traced = TransmissionLedger(retain_outcomes=True)

        aggregate_only.record(final)
        traced.record(final)

        self.assertFalse(aggregate_only.retain_outcomes)
        self.assertEqual(aggregate_only.retained_outcomes, ())
        self.assertTrue(traced.retain_outcomes)
        self.assertEqual(traced.retained_outcomes, (final,))
        self.assertEqual(aggregate_only.snapshot(), traced.snapshot())

    def test_invalid_retention_flag_is_rejected(self):
        with self.assertRaises(ValueError):
            TransmissionLedger(retain_outcomes=1)


if __name__ == "__main__":
    unittest.main()
