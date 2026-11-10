"""Pure, unit-explicit performance formulas for SmartCityEdgeSim 1.2.

This package is the boundary between configured experiment data and stateful
simulation behavior.  Modules placed here must be deterministic for identical
inputs (unless an explicit seeded random source is supplied), must not mutate
tasks or ledgers, and must use the canonical SI units in
``llm_collab/IMPLEMENTATION_CONTRACT.md``.

Evaluation invariants
---------------------
* Keep queueing, transmission, propagation, processing, and handover time as
  distinct components before deriving end-to-end delay.
* Keep offered, payload, on-wire, retry, delivered, late, and dropped traffic as
  distinct quantities before deriving throughput or delivery ratios.
* Compute energy from explicit power and duration, and cost from a named tariff
  basis; formulas here never own cumulative history.
* Preserve sample distributions and denominators so mean, percentile, failure,
  and fairness measurements remain auditable.
* Convert to legacy KB/kbps/ms or presentation units only at compatibility and
  reporting boundaries.

Mobility observations use :mod:`metrics.handover_channel` to bind configured
radio inputs to the pure formulas in :mod:`metrics.channel`. Selected-task
delay, energy and cost use the observed packet execution.


``metrics.channel`` was authored by Claude under Task 30 and accepted after
independent contract, reproducibility, and dimensional checks.
"""

from importlib import import_module


# Public formula symbols are added only after their owning module passes its
# independent contract and regression gates.
__all__ = (
    "ApplicationObservation",
    "AccountingPerspective",
    "ComputeEnergyBreakdown",
    "IncrementalComputeEnergyBreakdown",
    "CostLine",
    "CostMeasurement",
    "CostSnapshot",
    "DelayDistribution",
    "KpiSnapshot",
    "PacketDelayBreakdown",
    "PacketRouteDelay",
    "TaskDelayBreakdown",
    "TaskEnergyBreakdown",
    "PriceBasis",
    "RadioEnergyBreakdown",
    "RadioPacketEnergyModel",
    "RewardBreakdown",
    "RewardObservation",
    "SixGChannelModel",
    "TariffProfile",
    "TransportEnergyBreakdown",
    "TransportPacketEnergyModel",
    "UavEnergyBreakdown",
    "aggregate_kpis",
    "aggregate_cost_lines",
    "calculate_task_delay",
    "calculate_task_energy",
    "calculate_cost_line",
    "compute_energy",
    "energy_from_power_duration_j",
    "incremental_compute_energy",
    "linear_cost",
    "observe_route",
    "packet_delay_breakdown",
    "packet_radio_energy",
    "packet_route_delay",
    "percentile_type7",
    "propagation_delay_s",
    "radio_energy",
    "measure_cost",
    "evaluate_reward",
    "marginal_shared_capacity_pressure",
    "round_cost_for_presentation",
    "route_completion_delay_s",
    "transmission_delay_s",
    "sum_costs",
    "sum_delay_components_s",
    "transport_energy",
    "uav_energy",
)


_EXPORT_MODULES = {
    "ApplicationObservation": "metrics.kpi",
    "AccountingPerspective": "metrics.cost",
    "ComputeEnergyBreakdown": "metrics.energy",
    "IncrementalComputeEnergyBreakdown": "metrics.energy",
    "CostLine": "metrics.cost",
    "CostMeasurement": "metrics.cost",
    "CostSnapshot": "metrics.cost",
    "DelayDistribution": "metrics.kpi",
    "KpiSnapshot": "metrics.kpi",
    "PacketDelayBreakdown": "metrics.delay",
    "PacketRouteDelay": "metrics.delay",
    "TaskDelayBreakdown": "metrics.delay",
    "TaskEnergyBreakdown": "metrics.energy",
    "PriceBasis": "metrics.cost",
    "RadioEnergyBreakdown": "metrics.energy",
    "RadioPacketEnergyModel": "metrics.energy",
    "RewardBreakdown": "metrics.reward",
    "RewardObservation": "metrics.reward",
    "SixGChannelModel": "metrics.channel",
    "TariffProfile": "metrics.cost",
    "TransportEnergyBreakdown": "metrics.energy",
    "TransportPacketEnergyModel": "metrics.energy",
    "UavEnergyBreakdown": "metrics.energy",
    "aggregate_kpis": "metrics.kpi",
    "aggregate_cost_lines": "metrics.cost",
    "calculate_task_delay": "metrics.delay",
    "calculate_task_energy": "metrics.energy",
    "calculate_cost_line": "metrics.cost",
    "compute_energy": "metrics.energy",
    "energy_from_power_duration_j": "metrics.energy",
    "incremental_compute_energy": "metrics.energy",
    "linear_cost": "metrics.cost",
    "observe_route": "metrics.kpi",
    "packet_delay_breakdown": "metrics.delay",
    "packet_radio_energy": "metrics.energy",
    "packet_route_delay": "metrics.delay",
    "percentile_type7": "metrics.kpi",
    "propagation_delay_s": "metrics.delay",
    "radio_energy": "metrics.energy",
    "measure_cost": "metrics.cost",
    "evaluate_reward": "metrics.reward",
    "marginal_shared_capacity_pressure": "metrics.reward",
    "round_cost_for_presentation": "metrics.cost",
    "route_completion_delay_s": "metrics.delay",
    "transmission_delay_s": "metrics.delay",
    "sum_costs": "metrics.cost",
    "sum_delay_components_s": "metrics.delay",
    "transport_energy": "metrics.energy",
    "uav_energy": "metrics.energy",
}


def __getattr__(name: str):
    """Load one public formula on first use without creating import cycles."""

    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module 'metrics' has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public formulas in interactive discovery."""

    return sorted(set(globals()) | set(__all__))
