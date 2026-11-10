"""Runtime accounting for packet transmissions.

Scientific formulas remain in ``metrics``. This module owns streaming stateful
accumulation, accepting every final per-unit/per-hop outcome exactly once.
Snapshots are O(1); retaining exact outcome objects is opt-in for trace/debug
runs rather than an unavoidable city-scale memory cost.

How to extend: add a streaming counter to ``TransmissionLedger`` and its frozen
snapshot. Do not add physical formulas here; calculate them in ``metrics`` and
record their final values in ``TransmissionOutcome``.
"""

from __future__ import annotations

from dataclasses import dataclass

from network.contracts import DeliveryStatus, TransmissionOutcome


_FINAL_STATUSES = frozenset(
    {
        DeliveryStatus.DELIVERED,
        DeliveryStatus.DROPPED,
        DeliveryStatus.EXPIRED,
    }
)


@dataclass(frozen=True, slots=True)
class TransmissionLedgerSnapshot:
    """Immutable O(1) counters for one ledger scope."""

    final_outcome_count: int
    delivered_count: int
    dropped_count: int
    expired_count: int
    attempt_count: int
    retransmission_count: int
    offered_payload_bytes: int
    on_wire_bytes: int
    useful_payload_bytes: int
    tx_energy_j: float
    rx_energy_j: float
    transport_energy_j: float
    monetary_cost: float

    @property
    def total_network_energy_j(self) -> float:
        """Return TX, RX, and transport energy in joules."""

        return self.tx_energy_j + self.rx_energy_j + self.transport_energy_j

    @property
    def delivery_ratio(self) -> float:
        """Return delivered packet-hop outcomes divided by offered outcomes."""

        from metrics.kpi import bounded_ratio  # deferred: avoids a metrics<->network import cycle

        return bounded_ratio(self.delivered_count, self.final_outcome_count)

    @property
    def wire_efficiency_ratio(self) -> float:
        """Return useful payload bytes divided by all on-wire bytes."""

        from metrics.kpi import bounded_ratio  # deferred: avoids a metrics<->network import cycle

        return bounded_ratio(self.useful_payload_bytes, self.on_wire_bytes)


class TransmissionLedger:
    """Idempotent accumulator for final packet/batch outcomes at each hop.

    ``on_wire_bytes`` in a ``TransmissionOutcome`` is the total across all of
    that outcome's attempts. Application-level goodput across a complete route
    will be calculated separately so payload crossing several hops is not
    mistaken for unique application data.
    """

    def __init__(self, *, retain_outcomes: bool = False) -> None:
        if not isinstance(retain_outcomes, bool):
            raise ValueError("retain_outcomes must be a bool")
        self._retain_outcomes = retain_outcomes
        self._outcomes: list[TransmissionOutcome] = []
        self._final_keys: set[tuple[str, str]] = set()
        self._reset_totals()

    def _reset_totals(self) -> None:
        self._final_outcome_count = 0
        self._delivered_count = 0
        self._dropped_count = 0
        self._expired_count = 0
        self._attempt_count = 0
        self._retransmission_count = 0
        self._offered_payload_bytes = 0
        self._on_wire_bytes = 0
        self._useful_payload_bytes = 0
        self._tx_energy_j = 0.0
        self._rx_energy_j = 0.0
        self._transport_energy_j = 0.0
        self._monetary_cost = 0.0

    def __len__(self) -> int:
        return self._final_outcome_count

    @property
    def retain_outcomes(self) -> bool:
        """Report whether exact outcome objects are retained for tracing."""

        return self._retain_outcomes

    @property
    def retained_outcomes(self) -> tuple[TransmissionOutcome, ...]:
        """Exact trace objects, empty unless retention was requested."""

        return tuple(self._outcomes)

    def record(self, outcome: TransmissionOutcome) -> None:
        """Record one final packet-hop outcome exactly once."""

        if outcome.status not in _FINAL_STATUSES:
            raise ValueError("TransmissionLedger accepts final outcomes only")
        key = (outcome.unit_id, outcome.hop_id)
        if key in self._final_keys:
            raise ValueError(f"duplicate final outcome for unit/hop {key!r}")
        self._final_keys.add(key)
        if self._retain_outcomes:
            self._outcomes.append(outcome)
        self._final_outcome_count += 1
        self._delivered_count += outcome.status is DeliveryStatus.DELIVERED
        self._dropped_count += outcome.status is DeliveryStatus.DROPPED
        self._expired_count += outcome.status is DeliveryStatus.EXPIRED
        self._attempt_count += outcome.attempts
        self._retransmission_count += max(0, outcome.attempts - 1)
        self._offered_payload_bytes += outcome.payload_bytes
        self._on_wire_bytes += outcome.on_wire_bytes
        self._useful_payload_bytes += outcome.useful_payload_bytes
        self._tx_energy_j += outcome.tx_energy_j
        self._rx_energy_j += outcome.rx_energy_j
        self._transport_energy_j += outcome.transport_energy_j
        self._monetary_cost += outcome.monetary_cost

    def snapshot(self) -> TransmissionLedgerSnapshot:
        """Return current counters without scanning retained outcomes."""

        return TransmissionLedgerSnapshot(
            final_outcome_count=self._final_outcome_count,
            delivered_count=self._delivered_count,
            dropped_count=self._dropped_count,
            expired_count=self._expired_count,
            attempt_count=self._attempt_count,
            retransmission_count=self._retransmission_count,
            offered_payload_bytes=self._offered_payload_bytes,
            on_wire_bytes=self._on_wire_bytes,
            useful_payload_bytes=self._useful_payload_bytes,
            tx_energy_j=self._tx_energy_j,
            rx_energy_j=self._rx_energy_j,
            transport_energy_j=self._transport_energy_j,
            monetary_cost=self._monetary_cost,
        )

    def clear(self) -> None:
        """Reset counters, duplicate detection, and optional trace storage."""

        self._outcomes.clear()
        self._final_keys.clear()
        self._reset_totals()
