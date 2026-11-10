"""Deterministic service model for explicitly configured wired links.

This is intentionally a link-service model, not a radio propagation model.
Capacity and propagation speed come from the immutable ``LinkProfile`` while
distance comes from the resolved physical route.

How to extend: define a new wired link profile in ``config/network_config.py``
or implement another ``ChannelModel``. Keep routing and mutable queue state out
of this deterministic service formula.
"""

from __future__ import annotations

from math import isfinite
from typing import Mapping

from metrics.delay import propagation_delay_s
from network.contracts import ChannelRequest, ChannelResult, LinkProfile


class FixedRateChannelModel:
    """Return fixed-rate, error-free service for configured wired profiles."""

    def __init__(
        self,
        link_profile: LinkProfile,
        profiles: Mapping[str, LinkProfile] | None = None,
        *,
        model_id: str = "fixed-rate-wired-v1",
    ) -> None:
        if not isinstance(link_profile, LinkProfile):
            raise ValueError("link_profile must be a LinkProfile")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        registry = dict(profiles or {})
        registry.setdefault(link_profile.profile_id, link_profile)
        for profile_id, profile in registry.items():
            if not isinstance(profile_id, str) or not profile_id.strip():
                raise ValueError("profile IDs must be non-empty strings")
            if not isinstance(profile, LinkProfile):
                raise ValueError("profile registry values must be LinkProfile objects")
            if profile_id != profile.profile_id:
                raise ValueError("profile registry key must match LinkProfile.profile_id")
        self._profiles = registry
        self._model_id = model_id

    def evaluate(self, request: ChannelRequest) -> ChannelResult:
        """Return error-free fixed-rate service and propagation delay in seconds."""

        try:
            profile = self._profiles[request.link_profile_id]
        except KeyError as exc:
            raise KeyError(
                f"unknown wired link_profile_id {request.link_profile_id!r}; "
                f"known: {sorted(self._profiles)}"
            ) from exc
        propagation_s = propagation_delay_s(
            distance_m=request.distance_m,
            propagation_speed_m_s=profile.propagation_speed_m_s,
        )
        if not isfinite(propagation_s):
            raise ValueError("wired propagation delay must be finite")
        return ChannelResult(
            model_id=self._model_id,
            sinr_db=0.0,
            achievable_rate_bps=profile.nominal_bit_rate_bps,
            outage=False,
            packet_error_probability=0.0,
            propagation_delay_s=propagation_s,
            rx_power_dbm=None,
        )
