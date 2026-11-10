"""Access-only radio endpoint used by the deployment layer.

An :class:`AccessPoint` is deliberately not a compute ``Server``.  It has the
small compatibility surface required by the existing mobility handover code
(``name``, ``tier``, coordinates and a stable profile ID), while compute
admission, processing delay, energy and cost remain owned by MEC/Metro/Cloud
servers in ``Simulator.servers``.

How to extend: add access-specific radio state here only when it is observed by
the handover/network layers.  Do not add compute capacity to this object; map it
to a compute site through ``compute_anchor_id`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(slots=True)
class AccessPoint:
    """One access-only ground BS position in metres."""

    name: str
    index: int
    x: float
    y: float
    z: float
    profile_id: str
    compute_anchor_id: str
    anchor_profile_id: str | None = None

    # The legacy handover scheduler recognizes ground radio points by tier.  The
    # object remains excluded from the scheduler's compute-server list, so this
    # compatibility value does not make the BS a compute resource.
    tier: str = "MEC"
    is_access_only: bool = True
    hover_power_w: float = 0.0
    mobility = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("access-point name must be non-empty")
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("access-point index must be a non-negative integer")
        for name in ("x", "y", "z"):
            value = float(getattr(self, name))
            if not isfinite(value) or (name == "z" and value < 0.0):
                raise ValueError(f"access-point {name} must be finite and valid")
            setattr(self, name, value)
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("access-point profile_id must be non-empty")
        if not isinstance(self.compute_anchor_id, str) or not self.compute_anchor_id.strip():
            raise ValueError("compute_anchor_id must be non-empty")
        if self.anchor_profile_id is not None and (
            not isinstance(self.anchor_profile_id, str) or not self.anchor_profile_id.strip()
        ):
            raise ValueError("anchor_profile_id must be a non-empty string when provided")

    def get_position(self) -> tuple[float, float]:
        """Return the horizontal position in metres for mobility consumers."""

        return (self.x, self.y)
