"""Load immutable link templates and bind them to physical resources.

How to extend: add profile data to ``config/network_config.py``. Keep this module
limited to validation and instantiation so researchers change values in config,
not in engine code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from config.radio import get_link_profile
from network.contracts import HeaderStack, LinkProfile
from network.topology import Provenance


@dataclass(frozen=True, slots=True)
class LinkProfileTemplate:
    """Validated link behavior awaiting a physical shared-resource ID."""

    profile: LinkProfile
    provenance: Provenance

    def instantiate(self, shared_resource_id: str) -> LinkProfile:
        """Bind this immutable template to one queue/service resource."""

        if not isinstance(shared_resource_id, str) or not shared_resource_id.strip():
            raise ValueError("shared_resource_id must be a non-empty string")
        return replace(self.profile, shared_resource_id=shared_resource_id)


def link_profile_template_from_mapping(raw: Mapping[str, object]) -> LinkProfileTemplate:
    """Validate configuration data and return a typed link template."""

    try:
        header = raw["header_stack"]
        profile = LinkProfile(
            profile_id=raw["profile_id"],
            technology=raw["technology"],
            generation=raw["generation"],
            band=raw["band"],
            network_mtu_bytes=raw["network_mtu_bytes"],
            nominal_bit_rate_bps=raw["nominal_bit_rate_bps"],
            propagation_speed_m_s=raw["propagation_speed_m_s"],
            shared_resource_id=raw["shared_resource_id"],
            header_stack=HeaderStack(
                network_header_bytes=header["network_header_bytes"],
                transport_header_bytes=header["transport_header_bytes"],
                security_overhead_bytes=header["security_overhead_bytes"],
                telemetry_overhead_bytes=header["telemetry_overhead_bytes"],
                link_header_bytes=header["link_header_bytes"],
                link_trailer_bytes=header["link_trailer_bytes"],
            ),
            channel_model_id=raw["channel_model_id"],
            queue_capacity_bytes=raw["queue_capacity_bytes"],
        )
        return LinkProfileTemplate(
            profile=profile,
            provenance=Provenance.from_mapping(raw["provenance"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid link profile template: {exc}") from exc


def load_link_profile_template(profile_id: str) -> LinkProfileTemplate:
    """Load one named template from the immutable configuration catalog."""

    return link_profile_template_from_mapping(get_link_profile(profile_id))


def instantiate_link_profile(profile_id: str, *, shared_resource_id: str) -> LinkProfile:
    """Load a named template and bind it to one physical resource."""

    return load_link_profile_template(profile_id).instantiate(shared_resource_id)
