"""Convert measured energy and deployed hardware into auditable CO2e.

Two independent, non-overlapping calculations, matching how config/emissions.py
separates them:
  * Operational CO2e: one task's grid-drawn energy (joules) x an effective
    grid carbon-intensity rate (g/J). Per task, safe to sum across a run --
    the underlying energy is already the same incrementally-accounted,
    non-double-counted quantity metrics/energy.py and OperatorCost use.
  * Embodied CO2e: a deployment's manufacturing footprint, amortized to a
    per-year rate. Reported at the deployment level (unit counts), never
    attributed per task -- attributing a shared server's manufacturing
    footprint per task would double-count it across every concurrent task
    that touches the same physical unit, the same failure mode
    metrics/energy.py's IncrementalComputeEnergyBreakdown already avoids for
    compute energy.

How to extend: add or revise sourced values in config/emissions.py; add a
small pure formula here. Keep no prices, no experimental constants, and no
simulation/scheduling state in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Mapping

from config.emissions import annualized_embodied_kg_co2e


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


def operational_emissions_g(*, energy_j: float, carbon_intensity_g_per_j: float) -> float:
    """Return operational CO2e (grams) for one measured energy quantity.

    ``energy_j`` should be the same grid-drawn energy quantity already used
    for OperatorCost.energy_cost (infrastructure_network + compute_it +
    facility_overhead), so operational CO2e and operational energy cost share
    one scope -- no separate scoping decision is made here.
    """

    energy = _nonnegative("energy_j", energy_j)
    intensity = _nonnegative("carbon_intensity_g_per_j", carbon_intensity_g_per_j)
    return energy * intensity


@dataclass(frozen=True, slots=True)
class EmbodiedEmissionsLine:
    """One infrastructure tier's amortized embodied CO2e for one year."""

    tier: str
    unit_count: int
    annualized_kg_co2e: float

    def __post_init__(self) -> None:
        if not isinstance(self.tier, str) or not self.tier.strip():
            raise ValueError("tier must be a non-empty string")
        if isinstance(self.unit_count, bool) or not isinstance(self.unit_count, int) or self.unit_count < 0:
            raise ValueError("unit_count must be a non-negative integer")
        object.__setattr__(
            self, "annualized_kg_co2e", _nonnegative("annualized_kg_co2e", self.annualized_kg_co2e)
        )


@dataclass(frozen=True, slots=True)
class DeploymentEmissionsSummary:
    """Immutable, itemized annualized embodied-CO2e total for a deployment."""

    lines: tuple[EmbodiedEmissionsLine, ...]

    def __post_init__(self) -> None:
        if any(not isinstance(line, EmbodiedEmissionsLine) for line in self.lines):
            raise ValueError("lines must contain EmbodiedEmissionsLine values")

    @property
    def total_annualized_kg_co2e(self) -> float:
        """Return the full-precision annualized embodied-CO2e total (kg/year)."""

        return sum(line.annualized_kg_co2e for line in self.lines)


def summarize_deployment_embodied_emissions(
    tier_unit_counts: Mapping[str, int],
) -> DeploymentEmissionsSummary:
    """Build one itemized annualized embodied-CO2e summary for a deployment.

    ``tier_unit_counts`` maps a config.emissions tier name (e.g. "mec",
    "ground_bs", "uav_bs", "metro", "cloud", "onboard") to how many physical
    units of that tier are deployed. Tiers with a zero count are omitted from
    ``lines`` rather than included as a zero row.
    """

    if not isinstance(tier_unit_counts, Mapping):
        raise ValueError("tier_unit_counts must be a mapping")
    lines = tuple(
        EmbodiedEmissionsLine(
            tier=tier,
            unit_count=int(count),
            annualized_kg_co2e=annualized_embodied_kg_co2e(tier, unit_count=int(count)),
        )
        for tier, count in sorted(tier_unit_counts.items())
        if int(count) > 0
    )
    return DeploymentEmissionsSummary(lines=lines)
