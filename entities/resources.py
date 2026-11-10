"""Physical compute-resource values shared by workloads, servers, and policies.

The simulator keeps computation work, service rate, memory, and storage in
explicit units so delay, admission, utilization, and cost use the same meaning.
Legacy numeric fields are converted only by a named conversion profile.

How to extend: add a new conversion profile in ``config/units.py``;
do not add hardware-specific defaults or unit guesses to this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def _nonnegative_number(name: str, value: float) -> float:
    """Return a finite non-negative numeric value or raise ``ValueError``."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return number


def _positive_number(name: str, value: float) -> float:
    """Return a finite positive numeric value or raise ``ValueError``."""

    number = _nonnegative_number(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return number


def _nonnegative_bytes(name: str, value: int) -> int:
    """Return a non-negative integer byte count or raise ``ValueError``."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer byte count")
    return value


@dataclass(frozen=True, slots=True)
class ComputeWork:
    """Physical work and resident data required by one compute task."""

    compute_work_cycles: float
    memory_working_set_bytes: int
    storage_footprint_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "compute_work_cycles",
            _nonnegative_number("compute_work_cycles", self.compute_work_cycles),
        )
        _nonnegative_bytes(
            "memory_working_set_bytes", self.memory_working_set_bytes
        )
        _nonnegative_bytes("storage_footprint_bytes", self.storage_footprint_bytes)


@dataclass(frozen=True, slots=True)
class ComputeCapacity:
    """Aggregate compute rate and byte capacities of one execution resource."""

    compute_rate_cycles_per_s: float
    memory_capacity_bytes: int
    storage_capacity_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "compute_rate_cycles_per_s",
            _positive_number(
                "compute_rate_cycles_per_s", self.compute_rate_cycles_per_s
            ),
        )
        _nonnegative_bytes("memory_capacity_bytes", self.memory_capacity_bytes)
        _nonnegative_bytes("storage_capacity_bytes", self.storage_capacity_bytes)

    def can_host(self, allocation: "ResourceAllocation") -> bool:
        """Return whether an allocation fits all rate and byte capacities."""

        if not isinstance(allocation, ResourceAllocation):
            raise ValueError("allocation must be a ResourceAllocation")
        return (
            allocation.allocated_compute_rate_cycles_per_s
            <= self.compute_rate_cycles_per_s
            and allocation.memory_allocated_bytes <= self.memory_capacity_bytes
            and allocation.storage_allocated_bytes <= self.storage_capacity_bytes
        )


@dataclass(frozen=True, slots=True)
class ResourceAllocation:
    """Rate, memory, and storage reserved while one task is executing."""

    allocated_compute_rate_cycles_per_s: float
    memory_allocated_bytes: int
    storage_allocated_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allocated_compute_rate_cycles_per_s",
            _nonnegative_number(
                "allocated_compute_rate_cycles_per_s",
                self.allocated_compute_rate_cycles_per_s,
            ),
        )
        _nonnegative_bytes("memory_allocated_bytes", self.memory_allocated_bytes)
        _nonnegative_bytes("storage_allocated_bytes", self.storage_allocated_bytes)


@dataclass(frozen=True, slots=True)
class LegacyResourceConversionProfile:
    """Explicit scales that translate legacy simulator values into SI units."""

    profile_id: str
    compute_work_cycles_per_legacy_unit: float
    compute_rate_cycles_per_s_per_legacy_unit: float
    memory_bytes_per_legacy_unit: int
    storage_bytes_per_legacy_unit: int
    reservation_window_s: float
    reference_vcpu_rate_cycles_per_s: float
    provenance_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("profile_id must be a non-empty string")
        if not isinstance(self.provenance_id, str) or not self.provenance_id.strip():
            raise ValueError("provenance_id must be a non-empty string")
        for name in (
            "compute_work_cycles_per_legacy_unit",
            "compute_rate_cycles_per_s_per_legacy_unit",
            "reservation_window_s",
            "reference_vcpu_rate_cycles_per_s",
        ):
            object.__setattr__(self, name, _positive_number(name, getattr(self, name)))
        for name in (
            "memory_bytes_per_legacy_unit",
            "storage_bytes_per_legacy_unit",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer byte scale")

    def work_from_legacy(
        self,
        *,
        cpu_demand: float,
        memory_demand: float,
        storage_demand: float,
    ) -> ComputeWork:
        """Convert one legacy task demand into cycles and byte quantities."""

        return ComputeWork(
            compute_work_cycles=(
                _nonnegative_number("cpu_demand", cpu_demand)
                * self.compute_work_cycles_per_legacy_unit
            ),
            memory_working_set_bytes=round(
                _nonnegative_number("memory_demand", memory_demand)
                * self.memory_bytes_per_legacy_unit
            ),
            storage_footprint_bytes=round(
                _nonnegative_number("storage_demand", storage_demand)
                * self.storage_bytes_per_legacy_unit
            ),
        )

    def capacity_from_legacy(
        self,
        *,
        cpu_capacity: float,
        memory_capacity: float,
        storage_capacity: float,
    ) -> ComputeCapacity:
        """Convert one legacy server capacity into cycles/s and bytes."""

        return ComputeCapacity(
            compute_rate_cycles_per_s=(
                _positive_number("cpu_capacity", cpu_capacity)
                * self.compute_rate_cycles_per_s_per_legacy_unit
            ),
            memory_capacity_bytes=round(
                _nonnegative_number("memory_capacity", memory_capacity)
                * self.memory_bytes_per_legacy_unit
            ),
            storage_capacity_bytes=round(
                _nonnegative_number("storage_capacity", storage_capacity)
                * self.storage_bytes_per_legacy_unit
            ),
        )

    def allocation_for(self, work: ComputeWork) -> ResourceAllocation:
        """Reserve work over this profile's declared reference window."""

        if not isinstance(work, ComputeWork):
            raise ValueError("work must be a ComputeWork")
        return ResourceAllocation(
            allocated_compute_rate_cycles_per_s=(
                work.compute_work_cycles / self.reservation_window_s
            ),
            memory_allocated_bytes=work.memory_working_set_bytes,
            storage_allocated_bytes=work.storage_footprint_bytes,
        )

    def cpu_seconds_for(self, work: ComputeWork) -> float:
        """Convert task cycles into reference-vCPU seconds for pricing."""

        if not isinstance(work, ComputeWork):
            raise ValueError("work must be a ComputeWork")
        return work.compute_work_cycles / self.reference_vcpu_rate_cycles_per_s


def processing_duration_s(
    compute_work_cycles: float,
    allocated_compute_rate_cycles_per_s: float,
) -> float:
    """Return seconds required for work at a positive allocated compute rate.

    Thin wrapper kept for backward compatibility: the single processing-delay
    formula (work / rate) now lives in :func:`metrics.delay.processing_delay_s`.
    """

    from metrics.delay import processing_delay_s

    return processing_delay_s(
        compute_work_cycles=compute_work_cycles,
        compute_rate_cycles_per_s=allocated_compute_rate_cycles_per_s,
    )


def offered_load_ratio(
    offered_bits: float,
    bit_rate_bps: float,
    observation_window_s: float,
) -> float:
    """Return offered bits divided by service available in a time window.

    Thin wrapper kept for backward compatibility: the single offered-load
    formula now lives in :func:`metrics.kpi.offered_load_ratio`.
    """

    from metrics.kpi import offered_load_ratio as _offered_load_ratio

    return _offered_load_ratio(offered_bits, bit_rate_bps, observation_window_s)


def bounded_utilization_ratio(allocated: float, capacity: float) -> float:
    """Return allocated/capacity clamped to the physical interval ``[0, 1]``."""

    used = _nonnegative_number("allocated", allocated)
    total = _positive_number("capacity", capacity)
    return min(1.0, used / total)


__all__ = (
    "ComputeCapacity",
    "ComputeWork",
    "LegacyResourceConversionProfile",
    "ResourceAllocation",
    "bounded_utilization_ratio",
    "offered_load_ratio",
    "processing_duration_s",
)
