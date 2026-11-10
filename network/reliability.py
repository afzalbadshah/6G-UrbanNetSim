"""Reproducible packet-error sampling and retry policy primitives.

The channel model returns probabilities. This module realizes those
probabilities without mutable RNG state so competing schedulers observe common
random numbers even when they evaluate packets in a different call order.

How to extend: implement another sampler or retry policy behind the public
protocols and inject it into ``NetworkEngine``. Seed derivation must remain
independent of scheduler evaluation order.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from math import isfinite
from typing import Protocol, runtime_checkable


_HASH_PERSON = b"SCESim-R1-v1"
_UINT64_SCALE = float(1 << 64)


def _require_nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_probability(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError("probability must be a finite number within [0, 1]")
    return float(value)


def _require_nonnegative(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


@dataclass(frozen=True, slots=True)
class BernoulliSample:
    """One auditable probability realization."""

    probability: float
    uniform_value: float
    occurred: bool

    def __post_init__(self) -> None:
        _require_probability(self.probability)
        uniform = _require_nonnegative("uniform_value", self.uniform_value)
        if uniform >= 1.0:
            raise ValueError("uniform_value must be within [0, 1)")
        if not isinstance(self.occurred, bool):
            raise ValueError("occurred must be a bool")
        if self.occurred != (uniform < self.probability):
            raise ValueError("occurred must equal uniform_value < probability")


@runtime_checkable
class PacketErrorSampler(Protocol):
    """Strategy that realizes one packet error from a declared probability."""

    def sample(
        self,
        probability: float,
        *,
        purpose: str,
        unit_id: str,
        hop_id: str,
        attempt_number: int,
    ) -> BernoulliSample:
        """Return a reproducible draw for one object/packet/hop/attempt identity."""

        """Return a reproducible Bernoulli realization for one keyed attempt."""


class CounterBasedSampler:
    """BLAKE2b-64 counter-based sampler with stable length-prefixed keys."""

    def __init__(self, experiment_seed: int | str) -> None:
        if isinstance(experiment_seed, bool):
            raise ValueError("experiment_seed must be a non-negative int or non-empty string")
        if isinstance(experiment_seed, int):
            if experiment_seed < 0:
                raise ValueError("integer experiment_seed must be non-negative")
            self._seed_token = f"int:{experiment_seed}"
        elif isinstance(experiment_seed, str) and experiment_seed.strip():
            self._seed_token = f"str:{experiment_seed}"
        else:
            raise ValueError("experiment_seed must be a non-negative int or non-empty string")

    @staticmethod
    def _add_component(digest, value: str) -> None:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="big", signed=False))
        digest.update(encoded)

    def sample(
        self,
        probability: float,
        *,
        purpose: str,
        unit_id: str,
        hop_id: str,
        attempt_number: int,
    ) -> BernoulliSample:
        """Return a stateless common-random-number Bernoulli draw."""

        probability_value = _require_probability(probability)
        for name, value in (("purpose", purpose), ("unit_id", unit_id), ("hop_id", hop_id)):
            _require_nonempty(name, value)
        if (
            isinstance(attempt_number, bool)
            or not isinstance(attempt_number, int)
            or attempt_number < 1
        ):
            raise ValueError("attempt_number must be a positive integer")

        digest = blake2b(digest_size=8, person=_HASH_PERSON)
        for component in (
            self._seed_token,
            purpose,
            unit_id,
            hop_id,
            str(attempt_number),
        ):
            self._add_component(digest, component)
        uniform_value = int.from_bytes(digest.digest(), byteorder="big") / _UINT64_SCALE
        return BernoulliSample(
            probability=probability_value,
            uniform_value=uniform_value,
            occurred=uniform_value < probability_value,
        )


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Fixed retry limit/backoff baseline.

    ``outage_observation_s`` is the time required to establish that an attempted
    service opportunity is in outage. ACK/NACK and access contention are not
    hidden here; they require later explicit control-traffic/access policies.
    """

    max_attempts: int = 1
    backoff_s: float = 0.0
    outage_observation_s: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
        _require_nonnegative("backoff_s", self.backoff_s)
        _require_nonnegative("outage_observation_s", self.outage_observation_s)

    def next_attempt_at_s(
        self,
        failed_at_s: float,
        *,
        completed_attempts: int,
    ) -> float | None:
        """Return the next retry time in seconds, or ``None`` when exhausted."""

        failed = _require_nonnegative("failed_at_s", failed_at_s)
        if (
            isinstance(completed_attempts, bool)
            or not isinstance(completed_attempts, int)
            or completed_attempts < 1
        ):
            raise ValueError("completed_attempts must be a positive integer")
        if completed_attempts >= self.max_attempts:
            return None
        return failed + self.backoff_s
