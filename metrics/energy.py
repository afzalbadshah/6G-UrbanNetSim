"""Convert physical power-state durations into auditable energy components.

The pure functions handle radio, compute, fibre transport, and UAV operation.
The two ``*PacketEnergyModel`` classes are thin adapters used by the packet
engine; they delegate the physics back to those pure functions.

How to extend: add or revise physical power values in ``config/power.py`` and add
a small pure formula here. Keep battery mutation and simulation scheduling out of
this module. All durations are seconds, power is watts, and energy is joules.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Mapping

from network.engine import (
    PacketTransmissionResult,
    TransmissionAttempt,
    TransmissionEnergy,
)


def _nonnegative(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _positive(name: str, value: float) -> float:
    result = _nonnegative(name, value)
    if result == 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def energy_from_power_duration_j(*, power_w: float, duration_s: float) -> float:
    """Return electrical energy in joules for one constant-power interval."""

    power = _nonnegative("power_w", power_w)
    duration = _nonnegative("duration_s", duration_s)
    return power * duration


def _kind(profile: Mapping[str, object], expected: str) -> None:
    if profile.get("kind") != expected:
        raise ValueError(f"expected a {expected!r} power profile")


@dataclass(frozen=True, slots=True)
class RadioEnergyBreakdown:
    """Radio energy split into PA, circuits, idle, sleep, and transitions."""

    tx_active_s: float
    rx_active_s: float
    idle_s: float
    sleep_s: float
    pa_input_energy_j: float
    tx_circuit_energy_j: float
    rx_circuit_energy_j: float
    idle_energy_j: float
    sleep_energy_j: float
    transition_energy_j: float

    @property
    def tx_energy_j(self) -> float:
        """Return transmitter-side energy in joules."""

        return self.pa_input_energy_j + self.tx_circuit_energy_j + self.transition_energy_j

    @property
    def rx_energy_j(self) -> float:
        """Return receiver-side active circuit energy in joules."""

        return self.rx_circuit_energy_j

    @property
    def total_energy_j(self) -> float:
        """Return all radio energy components in joules."""

        return self.tx_energy_j + self.rx_energy_j + self.idle_energy_j + self.sleep_energy_j


@dataclass(frozen=True, slots=True)
class ComputeEnergyBreakdown:
    """IT and facility energy for one compute interval."""

    duration_s: float
    utilization_ratio: float
    it_power_w: float
    it_energy_j: float
    facility_overhead_energy_j: float

    @property
    def total_energy_j(self) -> float:
        """Return IT plus facility-overhead energy in joules."""

        return self.it_energy_j + self.facility_overhead_energy_j


@dataclass(frozen=True, slots=True)
class IncrementalComputeEnergyBreakdown:
    """Task-attributable dynamic energy on a shared compute resource.

    Shared idle power is intentionally excluded because charging it once per
    concurrent task would multiply the same facility baseline.  A run-level
    power-state ledger must account idle intervals exactly once.
    """

    duration_s: float
    utilization_before_ratio: float
    utilization_after_ratio: float
    incremental_it_power_w: float
    incremental_it_energy_j: float
    facility_overhead_energy_j: float

    @property
    def total_energy_j(self) -> float:
        """Return task-attributable IT and facility energy in joules."""

        return self.incremental_it_energy_j + self.facility_overhead_energy_j


@dataclass(frozen=True, slots=True)
class TransportEnergyBreakdown:
    """Fibre/transport energy under one mutually exclusive accounting basis."""

    accounting_method: str
    active_duration_s: float
    on_wire_bits: int
    active_energy_j: float
    per_bit_energy_j: float

    @property
    def total_energy_j(self) -> float:
        """Return transport energy in joules."""

        return self.active_energy_j + self.per_bit_energy_j


@dataclass(frozen=True, slots=True)
class UavEnergyBreakdown:
    """UAV hover, propulsion, and mobility energy for separate intervals."""

    hover_energy_j: float
    propulsion_energy_j: float
    mobility_energy_j: float

    @property
    def total_energy_j(self) -> float:
        """Return all UAV operating energy in joules."""

        return self.hover_energy_j + self.propulsion_energy_j + self.mobility_energy_j


@dataclass(frozen=True, slots=True)
class TaskEnergyBreakdown:
    """One task's complete physical energy ledger in joules.

    ``objective_energy_j`` preserves the SmartCityEdgeSim 1.1 learning
    objective: local tasks optimize their compute energy, while offloaded tasks
    optimize only the device communication energy.  The full physical ledger is
    retained separately for evaluation and cost accounting.
    """

    device_tx_energy_j: float
    device_rx_energy_j: float
    device_wait_energy_j: float
    infrastructure_network_energy_j: float
    compute_it_energy_j: float
    facility_overhead_energy_j: float
    total_energy_j: float
    objective_energy_j: float
    objective_basis: str

    def __post_init__(self) -> None:
        for name in (
            "device_tx_energy_j",
            "device_rx_energy_j",
            "device_wait_energy_j",
            "infrastructure_network_energy_j",
            "compute_it_energy_j",
            "facility_overhead_energy_j",
            "total_energy_j",
            "objective_energy_j",
        ):
            object.__setattr__(self, name, _nonnegative(name, getattr(self, name)))
        if self.objective_basis not in {"local_compute", "device_communication"}:
            raise ValueError(
                "objective_basis must be 'local_compute' or 'device_communication'"
            )
        component_total = (
            self.device_tx_energy_j
            + self.device_rx_energy_j
            + self.device_wait_energy_j
            + self.infrastructure_network_energy_j
            + self.compute_it_energy_j
            + self.facility_overhead_energy_j
        )
        if not isclose(component_total, self.total_energy_j, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("energy components must sum to total_energy_j")
        expected_objective = (
            self.compute_it_energy_j + self.facility_overhead_energy_j
            if self.objective_basis == "local_compute"
            else self.device_tx_energy_j + self.device_rx_energy_j
        )
        if not isclose(
            expected_objective,
            self.objective_energy_j,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("objective_energy_j does not match objective_basis")

    @property
    def device_battery_energy_j(self) -> float:
        """Return the energy debited from the originating device battery.

        Remote execution charges device radio activity and waiting. Local
        execution charges waiting plus the onboard compute/facility energy.
        Infrastructure network and remote compute energy never drain the
        originating device battery.
        """

        base = self.device_tx_energy_j + self.device_rx_energy_j + self.device_wait_energy_j
        if self.objective_basis == "local_compute":
            base += self.compute_it_energy_j + self.facility_overhead_energy_j
        return base


def calculate_task_energy(
    *,
    device_tx_energy_j: float = 0.0,
    device_rx_energy_j: float = 0.0,
    device_wait_energy_j: float = 0.0,
    infrastructure_network_energy_j: float = 0.0,
    compute_it_energy_j: float = 0.0,
    facility_overhead_energy_j: float = 0.0,
    local_execution: bool,
) -> TaskEnergyBreakdown:
    """Build one validated task energy result without recomputing downstream."""

    if not isinstance(local_execution, bool):
        raise ValueError("local_execution must be a boolean")
    components = {
        "device_tx_energy_j": _nonnegative(
            "device_tx_energy_j", device_tx_energy_j
        ),
        "device_rx_energy_j": _nonnegative(
            "device_rx_energy_j", device_rx_energy_j
        ),
        "device_wait_energy_j": _nonnegative(
            "device_wait_energy_j", device_wait_energy_j
        ),
        "infrastructure_network_energy_j": _nonnegative(
            "infrastructure_network_energy_j", infrastructure_network_energy_j
        ),
        "compute_it_energy_j": _nonnegative(
            "compute_it_energy_j", compute_it_energy_j
        ),
        "facility_overhead_energy_j": _nonnegative(
            "facility_overhead_energy_j", facility_overhead_energy_j
        ),
    }
    total_energy_j = sum(components.values())
    objective_energy_j = (
        components["compute_it_energy_j"]
        + components["facility_overhead_energy_j"]
        if local_execution
        else components["device_tx_energy_j"] + components["device_rx_energy_j"]
    )
    return TaskEnergyBreakdown(
        **components,
        total_energy_j=total_energy_j,
        objective_energy_j=objective_energy_j,
        objective_basis=("local_compute" if local_execution else "device_communication"),
    )


def radio_energy(
    profile: Mapping[str, object],
    *,
    tx_active_s: float,
    rx_active_s: float,
    idle_s: float = 0.0,
    sleep_s: float = 0.0,
    transitions: int = 0,
) -> RadioEnergyBreakdown:
    """Calculate radio energy from state durations in seconds.

    Received signal dBm is deliberately absent because received information
    power is not the receiver's electrical circuit power. Invalid profiles,
    durations, probabilities, or transition counts raise ``ValueError``.
    """

    _kind(profile, "radio")
    tx_time = _nonnegative("tx_active_s", tx_active_s)
    rx_time = _nonnegative("rx_active_s", rx_active_s)
    idle_time = _nonnegative("idle_s", idle_s)
    sleep_time = _nonnegative("sleep_s", sleep_s)
    if isinstance(transitions, bool) or not isinstance(transitions, int) or transitions < 0:
        raise ValueError("transitions must be a non-negative integer")
    radiated_w = _nonnegative("radiated_tx_power_w", profile["radiated_tx_power_w"])
    maximum_radiated_w = _nonnegative(
        "radiated_tx_power_max_w",
        profile.get("radiated_tx_power_max_w", radiated_w),
    )
    if radiated_w > maximum_radiated_w:
        raise ValueError("radiated_tx_power_w cannot exceed radiated_tx_power_max_w")
    pa_efficiency = _positive("pa_efficiency_ratio", profile["pa_efficiency_ratio"])
    if pa_efficiency > 1.0:
        raise ValueError("pa_efficiency_ratio must be within (0, 1]")
    return RadioEnergyBreakdown(
        tx_active_s=tx_time,
        rx_active_s=rx_time,
        idle_s=idle_time,
        sleep_s=sleep_time,
        pa_input_energy_j=energy_from_power_duration_j(
            power_w=radiated_w / pa_efficiency,
            duration_s=tx_time,
        ),
        tx_circuit_energy_j=energy_from_power_duration_j(
            power_w=_nonnegative("tx_circuit_power_w", profile["tx_circuit_power_w"]),
            duration_s=tx_time,
        ),
        rx_circuit_energy_j=energy_from_power_duration_j(
            power_w=_nonnegative("rx_circuit_power_w", profile["rx_circuit_power_w"]),
            duration_s=rx_time,
        ),
        idle_energy_j=energy_from_power_duration_j(
            power_w=_nonnegative("idle_power_w", profile["idle_power_w"]),
            duration_s=idle_time,
        ),
        sleep_energy_j=energy_from_power_duration_j(
            power_w=_nonnegative("sleep_power_w", profile["sleep_power_w"]),
            duration_s=sleep_time,
        ),
        transition_energy_j=_nonnegative(
            "transition_energy_j", profile["transition_energy_j"]
        )
        * transitions,
    )


def _radio_attempt_energy(
    attempts: tuple[TransmissionAttempt, ...],
    transmitter_profile: Mapping[str, object],
    receiver_profile: Mapping[str, object],
) -> RadioEnergyBreakdown:
    """Calculate active radio energy without inventing session transitions."""

    _kind(transmitter_profile, "radio")
    _kind(receiver_profile, "radio")
    # An outage observation without a reservation sends no data frame. Its
    # elapsed time belongs to the device wait/idle interval, not TX/RX airtime.
    serviced_attempts = tuple(
        attempt for attempt in attempts if attempt.reservation is not None
    )
    durations = tuple(
        attempt.reservation.transmission_delay_s
        for attempt in serviced_attempts
    )
    active_s = sum(durations)
    pa_efficiency = _positive(
        "pa_efficiency_ratio", transmitter_profile["pa_efficiency_ratio"]
    )
    if pa_efficiency > 1.0:
        raise ValueError("pa_efficiency_ratio must be within (0, 1]")
    default_radiated_w = _nonnegative(
        "radiated_tx_power_w", transmitter_profile["radiated_tx_power_w"]
    )
    maximum_radiated_w = _nonnegative(
        "radiated_tx_power_max_w",
        transmitter_profile.get("radiated_tx_power_max_w", default_radiated_w),
    )
    if default_radiated_w > maximum_radiated_w:
        raise ValueError("radiated_tx_power_w cannot exceed radiated_tx_power_max_w")
    if any(attempt.tx_power_w > maximum_radiated_w for attempt in attempts):
        raise ValueError("attempt tx_power_w exceeds transmitter radiated power maximum")
    return RadioEnergyBreakdown(
        tx_active_s=active_s,
        rx_active_s=active_s,
        idle_s=0.0,
        sleep_s=0.0,
        pa_input_energy_j=sum(
            energy_from_power_duration_j(
                power_w=attempt.tx_power_w / pa_efficiency,
                duration_s=duration,
            )
            for attempt, duration in zip(serviced_attempts, durations)
        ),
        tx_circuit_energy_j=energy_from_power_duration_j(
            power_w=_nonnegative(
                "tx_circuit_power_w", transmitter_profile["tx_circuit_power_w"]
            ),
            duration_s=active_s,
        ),
        rx_circuit_energy_j=energy_from_power_duration_j(
            power_w=_nonnegative(
                "rx_circuit_power_w", receiver_profile["rx_circuit_power_w"]
            ),
            duration_s=active_s,
        ),
        idle_energy_j=0.0,
        sleep_energy_j=0.0,
        # Packet traces do not reveal whether a radio stayed active across
        # adjacent packets. Charging a transition per packet would overcount;
        # transitions belong to the future run-level radio-state ledger.
        transition_energy_j=0.0,
    )


def packet_radio_energy(
    result: PacketTransmissionResult,
    transmitter_profile: Mapping[str, object],
    receiver_profile: Mapping[str, object],
) -> RadioEnergyBreakdown:
    """Calculate joules from observed attempts and endpoint transceivers."""

    if not isinstance(result, PacketTransmissionResult):
        raise ValueError("result must be a PacketTransmissionResult")
    return _radio_attempt_energy(
        result.attempts,
        transmitter_profile,
        receiver_profile,
    )


class RadioPacketEnergyModel:
    """Connect packet-attempt traces to the pure radio-energy formula."""

    def __init__(
        self,
        transmitter_profile: Mapping[str, object],
        receiver_profile: Mapping[str, object],
    ) -> None:
        _kind(transmitter_profile, "radio")
        _kind(receiver_profile, "radio")
        self._transmitter_profile = dict(transmitter_profile)
        self._receiver_profile = dict(receiver_profile)
        # Validate all active-energy inputs at composition time.
        _radio_attempt_energy(
            (),
            self._transmitter_profile,
            self._receiver_profile,
        )

    def validate_tx_power_w(self, tx_power_w: float) -> None:
        """Reject radiated transmit power above the endpoint maximum in watts."""

        value = _nonnegative("tx_power_w", tx_power_w)
        maximum = _nonnegative(
            "radiated_tx_power_max_w",
            self._transmitter_profile.get(
                "radiated_tx_power_max_w",
                self._transmitter_profile["radiated_tx_power_w"],
            ),
        )
        if value > maximum:
            raise ValueError("tx_power_w exceeds transmitter radiated power maximum")

    def calculate(
        self,
        *,
        attempts: tuple[TransmissionAttempt, ...],
        on_wire_bytes_per_attempt: int,
    ) -> TransmissionEnergy:
        """Return TX/RX joules for the supplied packet attempts."""

        breakdown = _radio_attempt_energy(
            attempts,
            self._transmitter_profile,
            self._receiver_profile,
        )
        return TransmissionEnergy(
            tx_energy_j=breakdown.tx_energy_j,
            rx_energy_j=breakdown.rx_energy_j,
        )


def compute_energy(
    profile: Mapping[str, object],
    *,
    duration_s: float,
    utilization_ratio: float,
) -> ComputeEnergyBreakdown:
    """Calculate IT and facility energy for a compute interval.

    ``duration_s`` is seconds and ``utilization_ratio`` lies in ``[0, 1]``.
    Invalid profile values or inputs raise ``ValueError``.
    """

    _kind(profile, "compute")
    duration = _nonnegative("duration_s", duration_s)
    utilization = _nonnegative("utilization_ratio", utilization_ratio)
    if utilization > 1.0:
        raise ValueError("utilization_ratio must be within [0, 1]")
    idle_w = _nonnegative("idle_power_w", profile["idle_power_w"])
    dynamic_w = _nonnegative("dynamic_power_w", profile["dynamic_power_w"])
    maximum_w = _nonnegative("max_power_w", profile["max_power_w"])
    exponent = _positive("utilization_exponent", profile["utilization_exponent"])
    if maximum_w < idle_w:
        raise ValueError("max_power_w cannot be lower than idle_power_w")
    it_power_w = min(maximum_w, idle_w + dynamic_w * (utilization**exponent))
    it_energy_j = energy_from_power_duration_j(
        power_w=it_power_w,
        duration_s=duration,
    )
    pue = _positive("pue_ratio", profile["pue_ratio"])
    if pue < 1.0:
        raise ValueError("pue_ratio must be at least one")
    return ComputeEnergyBreakdown(
        duration_s=duration,
        utilization_ratio=utilization,
        it_power_w=it_power_w,
        it_energy_j=it_energy_j,
        facility_overhead_energy_j=it_energy_j * (pue - 1.0),
    )


def incremental_compute_energy(
    profile: Mapping[str, object],
    *,
    duration_s: float,
    utilization_before_ratio: float,
    utilization_after_ratio: float,
) -> IncrementalComputeEnergyBreakdown:
    """Return marginal shared-compute joules without duplicating idle power.

    The before/after utilization ratios are in ``[0, 1]`` and ``duration_s`` is
    seconds. The run-level ledger, not each task, must account idle baselines.
    """

    _kind(profile, "compute")
    duration = _nonnegative("duration_s", duration_s)
    before = _nonnegative("utilization_before_ratio", utilization_before_ratio)
    after = _nonnegative("utilization_after_ratio", utilization_after_ratio)
    if before > 1.0 or after > 1.0:
        raise ValueError("compute utilization ratios must be within [0, 1]")
    if after < before:
        raise ValueError("utilization_after_ratio cannot be below utilization_before_ratio")
    idle_w = _nonnegative("idle_power_w", profile["idle_power_w"])
    maximum_w = _nonnegative("max_power_w", profile["max_power_w"])
    dynamic_w = _nonnegative("dynamic_power_w", profile["dynamic_power_w"])
    exponent = _positive("utilization_exponent", profile["utilization_exponent"])
    if maximum_w < idle_w:
        raise ValueError("max_power_w cannot be lower than idle_power_w")
    power_before_w = min(maximum_w, idle_w + dynamic_w * (before**exponent))
    power_after_w = min(maximum_w, idle_w + dynamic_w * (after**exponent))
    incremental_power_w = max(0.0, power_after_w - power_before_w)
    incremental_energy_j = energy_from_power_duration_j(
        power_w=incremental_power_w,
        duration_s=duration,
    )
    pue = _positive("pue_ratio", profile["pue_ratio"])
    if pue < 1.0:
        raise ValueError("pue_ratio must be at least one")
    return IncrementalComputeEnergyBreakdown(
        duration_s=duration,
        utilization_before_ratio=before,
        utilization_after_ratio=after,
        incremental_it_power_w=incremental_power_w,
        incremental_it_energy_j=incremental_energy_j,
        facility_overhead_energy_j=incremental_energy_j * (pue - 1.0),
    )


def transport_energy(
    profile: Mapping[str, object],
    *,
    active_duration_s: float,
    on_wire_bytes: int,
) -> TransportEnergyBreakdown:
    """Calculate transport joules from seconds or on-wire bytes, never both."""

    _kind(profile, "transport")
    duration = _nonnegative("active_duration_s", active_duration_s)
    if isinstance(on_wire_bytes, bool) or not isinstance(on_wire_bytes, int) or on_wire_bytes < 0:
        raise ValueError("on_wire_bytes must be a non-negative integer")
    method = profile.get("accounting_method")
    if method == "active_power":
        active_energy_j = energy_from_power_duration_j(
            power_w=_nonnegative("active_power_w", profile["active_power_w"]),
            duration_s=duration,
        )
        per_bit_energy_j = 0.0
    elif method == "energy_per_bit":
        active_energy_j = 0.0
        per_bit_energy_j = (
            _nonnegative("energy_per_bit_j", profile["energy_per_bit_j"])
            * on_wire_bytes
            * 8
        )
    else:
        raise ValueError(
            "transport accounting_method must be 'active_power' or 'energy_per_bit'"
        )
    return TransportEnergyBreakdown(
        accounting_method=method,
        active_duration_s=duration,
        on_wire_bits=on_wire_bytes * 8,
        active_energy_j=active_energy_j,
        per_bit_energy_j=per_bit_energy_j,
    )


class TransportPacketEnergyModel:
    """Connect packet-attempt traces to one transport-energy basis."""

    def __init__(self, profile: Mapping[str, object]) -> None:
        _kind(profile, "transport")
        self._profile = dict(profile)
        # Fail at composition time, not after a simulation has started.
        transport_energy(self._profile, active_duration_s=0.0, on_wire_bytes=0)

    @staticmethod
    def validate_tx_power_w(tx_power_w: float) -> None:
        """Validate the shared request field; wired links use zero RF watts."""

        # Wired ChannelRequest retains the common contract field even though the
        # fixed-rate provider does not use antenna power.
        _nonnegative("tx_power_w", tx_power_w)

    def calculate(
        self,
        *,
        attempts: tuple[TransmissionAttempt, ...],
        on_wire_bytes_per_attempt: int,
    ) -> TransmissionEnergy:
        """Return transport joules for the supplied packet attempts."""

        serviced_attempts = tuple(
            attempt for attempt in attempts if attempt.reservation is not None
        )
        active_s = sum(
            attempt.reservation.transmission_delay_s
            for attempt in serviced_attempts
        )
        breakdown = transport_energy(
            self._profile,
            active_duration_s=active_s,
            on_wire_bytes=on_wire_bytes_per_attempt * len(serviced_attempts),
        )
        return TransmissionEnergy(transport_energy_j=breakdown.total_energy_j)


def uav_energy(
    profile: Mapping[str, object],
    *,
    hover_duration_s: float = 0.0,
    propulsion_duration_s: float = 0.0,
    mobility_duration_s: float = 0.0,
) -> UavEnergyBreakdown:
    """Calculate UAV joules for separate, non-overlapping durations in seconds."""

    _kind(profile, "uav")
    return UavEnergyBreakdown(
        hover_energy_j=energy_from_power_duration_j(
            power_w=_nonnegative("hover_power_w", profile["hover_power_w"]),
            duration_s=_nonnegative("hover_duration_s", hover_duration_s),
        ),
        propulsion_energy_j=energy_from_power_duration_j(
            power_w=_nonnegative(
                "propulsion_power_w", profile["propulsion_power_w"]
            ),
            duration_s=_nonnegative("propulsion_duration_s", propulsion_duration_s),
        ),
        mobility_energy_j=energy_from_power_duration_j(
            power_w=_nonnegative("mobility_power_w", profile["mobility_power_w"]),
            duration_s=_nonnegative("mobility_duration_s", mobility_duration_s),
        ),
    )
