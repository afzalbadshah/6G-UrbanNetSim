"""Unit conversions for physical compute-resource accounting.

This is the UNITS layer (how to interpret the raw numbers), NOT a resource
catalog: config/servers.py holds each tier's capacity/price. Here one CPU unit
= one mega-cycle, one memory/storage unit = one megabyte, etc.

The legacy simulator values are interpreted explicitly: one CPU unit represents
one mega-cycle of task work or one mega-cycle/s of service, and one memory or
storage unit represents one decimal megabyte. The task generator supplies a
physical storage footprint directly, so the storage scale is only a fallback for
older callers.

How to extend: define another named ``LegacyResourceConversionProfile`` and
select it in the composition root; never change this profile silently.
"""

from entities.resources import LegacyResourceConversionProfile


LEGACY_V1_RESOURCE_CONVERSION = LegacyResourceConversionProfile(
    profile_id="smartcity-v1-megaunit-compatibility",
    compute_work_cycles_per_legacy_unit=1_000_000.0,
    compute_rate_cycles_per_s_per_legacy_unit=1_000_000.0,
    memory_bytes_per_legacy_unit=1_000_000,
    storage_bytes_per_legacy_unit=1_000_000,
    reservation_window_s=1.0,
    reference_vcpu_rate_cycles_per_s=1_000_000_000.0,
    provenance_id="internal:smartcity-v1-resource-ratio-preservation",
)


__all__ = ("LEGACY_V1_RESOURCE_CONVERSION",)
