"""Immutable provenance helpers for configuration records.

How to extend: add new maturity labels or record fields here only when the
configuration contract is updated, and keep ``deep_freeze`` recursive.
"""

from types import MappingProxyType

# Contract §11 maturity labels.
MATURITIES = ("validated", "standards-aligned", "experimental", "visionary")


def provenance(source_id, *, source_uri="", source_version="", valid_from=None,
               valid_to=None, validity_notes="", maturity="standards-aligned"):
    """Build a §11 provenance record (plain dict). Fails loudly on a bad maturity.

    Reminder (§11): an existing simulator assumption is NOT `validated` merely
    because it already existed — that requires disclosed measurement/reference
    calibration. Use `experimental` for uncalibrated carried-forward defaults.
    """
    if maturity not in MATURITIES:
        raise ValueError(f"maturity must be one of {MATURITIES}, got {maturity!r}")
    return {
        "source_id": source_id,
        "source_uri": source_uri,
        "source_version": source_version,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "validity_notes": validity_notes,
        "maturity": maturity,
    }


def deep_freeze(obj):
    """Recursively convert dicts -> MappingProxyType, lists/tuples -> tuple, and
    sets -> frozenset, so a published parameter graph (including nested structures
    and published schema sets) is read-only. A caller cannot mutate a default and
    silently change later runs."""
    if isinstance(obj, dict):
        return MappingProxyType({k: deep_freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(deep_freeze(v) for v in obj)
    if isinstance(obj, (set, frozenset)):
        return frozenset(deep_freeze(v) for v in obj)
    return obj
