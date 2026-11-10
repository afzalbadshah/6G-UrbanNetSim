"""Provide deterministic FIFO timing for one shared physical resource.

How to extend: add a separate link-service strategy rather than embedding radio
scheduling in ``SharedLink``. Link rates and frame limits belong in configuration
profiles; all times in this module are seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from metrics.delay import transmission_delay_s
from network.contracts import LinkProfile


def _nonnegative(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _positive(name: str, value: float) -> float:
    numeric = _nonnegative(name, value)
    if numeric == 0:
        raise ValueError(f"{name} must be greater than zero")
    return numeric


@dataclass(frozen=True, slots=True)
class LinkReservation:
    """Timing result for one transmission reservation on a shared resource."""

    shared_resource_id: str
    requested_at_s: float
    transmission_started_at_s: float
    transmission_ended_at_s: float
    arrival_at_s: float
    queue_delay_s: float
    transmission_delay_s: float
    propagation_delay_s: float
    achievable_rate_bps: float
    on_wire_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.shared_resource_id, str) or not self.shared_resource_id.strip():
            raise ValueError("shared_resource_id must be a non-empty string")
        for name in (
            "requested_at_s",
            "transmission_started_at_s",
            "transmission_ended_at_s",
            "arrival_at_s",
            "queue_delay_s",
            "transmission_delay_s",
            "propagation_delay_s",
        ):
            _nonnegative(name, getattr(self, name))
        _positive("achievable_rate_bps", self.achievable_rate_bps)
        if isinstance(self.on_wire_bytes, bool) or not isinstance(self.on_wire_bytes, int) or self.on_wire_bytes < 0:
            raise ValueError("on_wire_bytes must be a non-negative integer")
        if self.transmission_started_at_s < self.requested_at_s:
            raise ValueError("transmission cannot start before it is requested")
        if self.transmission_ended_at_s < self.transmission_started_at_s:
            raise ValueError("transmission cannot end before it starts")
        if self.arrival_at_s < self.transmission_ended_at_s:
            raise ValueError("arrival cannot precede transmission completion")


class SharedLink:
    """FIFO serializer for one shared link resource.

    This primitive models only queue waiting caused by prior transmission.
    Buffer drops, access contention, retries and scheduling disciplines are
    separate policies added in later increments.
    """

    def __init__(self, profile: LinkProfile) -> None:
        self.profile = profile
        self._next_available_at_s = 0.0

    @property
    def next_available_at_s(self) -> float:
        """Return the earliest service-start time in seconds."""

        return self._next_available_at_s

    def reserve(
        self,
        *,
        requested_at_s: float,
        on_wire_bytes: int,
        achievable_rate_bps: float | None = None,
        propagation_delay_s: float = 0.0,
    ) -> LinkReservation:
        """Reserve FIFO service and return queue/transmission timing in seconds."""

        requested = _nonnegative("requested_at_s", requested_at_s)
        propagation = _nonnegative("propagation_delay_s", propagation_delay_s)
        if isinstance(on_wire_bytes, bool) or not isinstance(on_wire_bytes, int) or on_wire_bytes < 0:
            raise ValueError("on_wire_bytes must be a non-negative integer")
        rate = _positive(
            "achievable_rate_bps",
            self.profile.nominal_bit_rate_bps if achievable_rate_bps is None else achievable_rate_bps,
        )

        started = max(requested, self._next_available_at_s)
        queue_delay = started - requested
        transmission_delay = transmission_delay_s(
            on_wire_bytes=on_wire_bytes,
            achievable_rate_bps=rate,
        )
        ended = started + transmission_delay
        arrival = ended + propagation
        self._next_available_at_s = ended

        return LinkReservation(
            shared_resource_id=self.profile.shared_resource_id,
            requested_at_s=requested,
            transmission_started_at_s=started,
            transmission_ended_at_s=ended,
            arrival_at_s=arrival,
            queue_delay_s=queue_delay,
            transmission_delay_s=transmission_delay,
            propagation_delay_s=propagation,
            achievable_rate_bps=rate,
            on_wire_bytes=on_wire_bytes,
        )

    def reset(self, available_at_s: float = 0.0) -> None:
        """Reset the resource clock to ``available_at_s`` seconds."""

        self._next_available_at_s = _nonnegative("available_at_s", available_at_s)
