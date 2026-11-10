"""Calculate monetary cost without mixing incompatible viewpoints.

The module converts named measured quantities (joules, bytes, CPU-seconds, and
similar bases) into itemized cost lines, then groups lines from exactly one
currency and one accounting perspective.

How to extend: add tariff data in ``config/tariffs.py``; add a new basis to
``PriceBasis`` only when its measured quantity is available with explicit units.
Do not place prices or experimental constants inside these formulas.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from enum import Enum
from math import isclose, isfinite
from types import MappingProxyType
from typing import Mapping


class AccountingPerspective(str, Enum):
    """Whose money is being measured; perspectives must never be summed."""

    OPERATOR_COST = "operator_cost"
    CONSUMER_PRICE = "consumer_price"


class PriceBasis(str, Enum):
    """Physical or service quantity to which a tariff is applied."""

    JOULE = "joule"
    KWH = "kwh"
    ON_WIRE_BYTE = "on_wire_byte"
    USEFUL_BYTE = "useful_byte"
    CPU_SECOND = "cpu_second"
    CAPACITY_SECOND = "capacity_second"
    COMPLETED_OBJECT = "completed_object"


def _nonempty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _nonnegative(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def linear_cost(*, measured_quantity: float, price_per_unit: float) -> float:
    """Return full-precision cost for one measured quantity and unit price."""

    quantity = _nonnegative("measured_quantity", measured_quantity)
    unit_price = _nonnegative("price_per_unit", price_per_unit)
    return quantity * unit_price


def sum_costs(*cost_components: float) -> float:
    """Return a full-precision sum of non-negative compatible cost components."""

    return sum(
        _nonnegative(f"cost_components[{index}]", value)
        for index, value in enumerate(cost_components)
    )


@dataclass(frozen=True, slots=True)
class TariffProfile:
    """Validated price and applicability metadata for one named tariff."""

    tariff_id: str
    currency_code: str
    price_basis: PriceBasis
    price_per_unit: float
    applicability_scope: str
    accounting_perspective: AccountingPerspective
    effective_version: str
    effective_date: str
    rounding_rule: str
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in (
            "tariff_id",
            "currency_code",
            "applicability_scope",
            "effective_version",
            "effective_date",
            "rounding_rule",
        ):
            _nonempty(name, getattr(self, name))
        if not isinstance(self.price_basis, PriceBasis):
            raise ValueError("price_basis must be a PriceBasis")
        if not isinstance(self.accounting_perspective, AccountingPerspective):
            raise ValueError("accounting_perspective must be an AccountingPerspective")
        object.__setattr__(
            self,
            "price_per_unit",
            _nonnegative("price_per_unit", self.price_per_unit),
        )
        if not isinstance(self.provenance, Mapping):
            raise ValueError("provenance must be a mapping")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "TariffProfile":
        """Build a validated tariff from configuration data.

        Raises ``ValueError`` when a required field is missing or invalid.
        """

        try:
            return cls(
                tariff_id=raw["tariff_id"],
                currency_code=raw["currency_code"],
                price_basis=PriceBasis(raw["price_basis"]),
                price_per_unit=raw["price_per_unit"],
                applicability_scope=raw["applicability_scope"],
                accounting_perspective=AccountingPerspective(
                    raw["accounting_perspective"]
                ),
                effective_version=raw["effective_version"],
                effective_date=raw["effective_date"],
                rounding_rule=raw["rounding_rule"],
                provenance=raw["provenance"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid tariff profile: {exc}") from exc


@dataclass(frozen=True, slots=True)
class CostLine:
    """One measured quantity multiplied by exactly one tariff."""

    tariff_id: str
    currency_code: str
    price_basis: PriceBasis
    applicability_scope: str
    accounting_perspective: AccountingPerspective
    measured_quantity: float
    price_per_unit: float
    unrounded_cost: float

    def __post_init__(self) -> None:
        for name in ("tariff_id", "currency_code", "applicability_scope"):
            _nonempty(name, getattr(self, name))
        if not isinstance(self.price_basis, PriceBasis):
            raise ValueError("price_basis must be a PriceBasis")
        if not isinstance(self.accounting_perspective, AccountingPerspective):
            raise ValueError("accounting_perspective must be an AccountingPerspective")
        quantity = _nonnegative("measured_quantity", self.measured_quantity)
        unit_price = _nonnegative("price_per_unit", self.price_per_unit)
        cost = _nonnegative("unrounded_cost", self.unrounded_cost)
        object.__setattr__(self, "measured_quantity", quantity)
        object.__setattr__(self, "price_per_unit", unit_price)
        object.__setattr__(self, "unrounded_cost", cost)
        expected_cost = linear_cost(
            measured_quantity=quantity,
            price_per_unit=unit_price,
        )
        if not isclose(cost, expected_cost, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError("unrounded_cost must equal quantity multiplied by unit price")


@dataclass(frozen=True, slots=True)
class CostSnapshot:
    """Immutable itemized total for one perspective and currency."""

    accounting_perspective: AccountingPerspective
    currency_code: str
    lines: tuple[CostLine, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.accounting_perspective, AccountingPerspective):
            raise ValueError("accounting_perspective must be an AccountingPerspective")
        _nonempty("currency_code", self.currency_code)
        if any(not isinstance(line, CostLine) for line in self.lines):
            raise ValueError("lines must contain CostLine values")
        if any(
            line.accounting_perspective is not self.accounting_perspective
            for line in self.lines
        ):
            raise ValueError("a cost snapshot cannot mix accounting perspectives")
        if any(line.currency_code != self.currency_code for line in self.lines):
            raise ValueError("a cost snapshot cannot mix currencies")

    @property
    def total_unrounded_cost(self) -> float:
        """Return the full-precision sum used by simulation ledgers."""

        return sum(line.unrounded_cost for line in self.lines)


@dataclass(frozen=True, slots=True)
class CostMeasurement:
    """A complete cost snapshot or an explicit reason it is incomplete."""

    accounting_perspective: AccountingPerspective
    complete: bool
    snapshot: CostSnapshot | None
    incomplete_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.accounting_perspective, AccountingPerspective):
            raise ValueError("accounting_perspective must be an AccountingPerspective")
        if not isinstance(self.complete, bool):
            raise ValueError("complete must be a bool")
        if self.complete:
            if self.snapshot is None or self.incomplete_reason is not None:
                raise ValueError("complete cost measurement requires only a snapshot")
            if self.snapshot.accounting_perspective is not self.accounting_perspective:
                raise ValueError("measurement and snapshot perspectives must match")
        elif self.snapshot is not None or not self.incomplete_reason:
            raise ValueError("incomplete cost measurement requires only a reason")


@dataclass(frozen=True, slots=True)
class OperatorCost:
    """Operator's realized cost for one task, priced from config/servers.py.

    Three non-overlapping components in USD -- their sum is ``total``:
      * compute_cost   = compute vCPU-seconds x tier compute price
      * transport_cost = on-wire bytes x tier transport price
      * energy_cost    = operator energy (J) x electricity price
    """

    compute_cost: float
    transport_cost: float
    energy_cost: float
    complete: bool = True

    def __post_init__(self) -> None:
        for name in ("compute_cost", "transport_cost", "energy_cost"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")

    @property
    def total(self) -> float:
        """Return the full-precision operator total in USD."""

        return self.compute_cost + self.transport_cost + self.energy_cost


def calculate_cost_line(
    tariff: TariffProfile | Mapping[str, object],
    *,
    measured_quantity: float,
) -> CostLine:
    """Multiply one non-negative measured quantity by one named tariff."""

    profile = tariff if isinstance(tariff, TariffProfile) else TariffProfile.from_mapping(tariff)
    quantity = _nonnegative("measured_quantity", measured_quantity)
    return CostLine(
        tariff_id=profile.tariff_id,
        currency_code=profile.currency_code,
        price_basis=profile.price_basis,
        applicability_scope=profile.applicability_scope,
        accounting_perspective=profile.accounting_perspective,
        measured_quantity=quantity,
        price_per_unit=profile.price_per_unit,
        unrounded_cost=linear_cost(
            measured_quantity=quantity,
            price_per_unit=profile.price_per_unit,
        ),
    )


def aggregate_cost_lines(
    lines: tuple[CostLine, ...],
    *,
    accounting_perspective: AccountingPerspective | str,
    currency_code: str,
) -> CostSnapshot:
    """Group compatible itemized lines under one perspective and currency."""

    try:
        perspective = AccountingPerspective(accounting_perspective)
    except ValueError as exc:
        raise ValueError("unknown accounting perspective") from exc
    return CostSnapshot(
        accounting_perspective=perspective,
        currency_code=currency_code,
        lines=tuple(lines),
    )


def measure_cost(
    quantities_by_tariff_id: Mapping[str, float],
    tariff_catalog: Mapping[str, Mapping[str, object]],
    *,
    accounting_perspective: AccountingPerspective | str,
) -> CostMeasurement:
    """Apply each measured quantity to one named tariff exactly once.

    Missing tariffs make the whole requested measurement incomplete.  A partial
    numeric total is intentionally not returned because it could be mistaken for
    a complete zero-cost component. Quantities must already use each tariff's
    declared ``PriceBasis`` unit; incompatible currencies or perspectives raise
    ``ValueError``.
    """

    try:
        perspective = AccountingPerspective(accounting_perspective)
    except ValueError as exc:
        raise ValueError("unknown accounting perspective") from exc
    if not isinstance(quantities_by_tariff_id, Mapping) or not quantities_by_tariff_id:
        raise ValueError("quantities_by_tariff_id must be a non-empty mapping")
    if not isinstance(tariff_catalog, Mapping):
        raise ValueError("tariff_catalog must be a mapping")
    missing = sorted(set(quantities_by_tariff_id) - set(tariff_catalog))
    if missing:
        return CostMeasurement(
            accounting_perspective=perspective,
            complete=False,
            snapshot=None,
            incomplete_reason=f"missing tariff profiles: {missing}",
        )

    lines = tuple(
        calculate_cost_line(
            tariff_catalog[tariff_id],
            measured_quantity=quantities_by_tariff_id[tariff_id],
        )
        for tariff_id in sorted(quantities_by_tariff_id)
    )
    currencies = {line.currency_code for line in lines}
    if len(currencies) != 1:
        raise ValueError("a cost measurement cannot mix currencies")
    snapshot = aggregate_cost_lines(
        lines,
        accounting_perspective=perspective,
        currency_code=next(iter(currencies)),
    )
    return CostMeasurement(
        accounting_perspective=perspective,
        complete=True,
        snapshot=snapshot,
    )


def round_cost_for_presentation(value: float, *, rounding_rule: str) -> Decimal:
    """Round a non-negative cost for display; never use this in a ledger."""

    numeric = _nonnegative("value", value)
    if rounding_rule != "half_even_2dp_usd":
        raise ValueError(f"unsupported rounding rule {rounding_rule!r}")
    try:
        return Decimal(str(numeric)).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
    except InvalidOperation as exc:
        raise ValueError("value cannot be rounded") from exc
