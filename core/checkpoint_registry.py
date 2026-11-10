"""Band-aware storage for trained offloading and handover policies.

A learned policy is valid only for the radio environment it observed during
training.  The deployment profile is the single authority for that environment:
it selects the ground, UAV, low-band anchor, and optional hotspot link profiles.
This module turns those selections into one readable directory name and one
machine-checkable manifest shared by every learned policy.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from config.radio import get_band, get_link_profile
from config.topology import get_deployment_profile


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST_NAME = "radio_environment.json"
_COMPONENTS = {"offloading", "mobility"}


def _slug(value: object) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", str(value or "").strip())
    return text.strip("-").lower() or "unknown"


def _radio_role(link_profile_id: str) -> dict[str, object]:
    """Describe one configured radio role using explicit SI-valued metadata."""

    link = get_link_profile(link_profile_id)
    band_name = str(link["band"])
    band = get_band(band_name)
    return {
        "link_profile_id": str(link_profile_id),
        "band": band_name,
        "carrier_frequency_hz": float(band["carrier_frequency_hz"]),
        "bandwidth_hz": float(band["bandwidth_hz"]),
    }


@dataclass(frozen=True, slots=True)
class RadioCheckpointIdentity:
    """Resolved identity of the radio environment seen by a learned policy."""

    deployment_profile_id: str
    directory_name: str
    metadata: dict[str, object]


def radio_checkpoint_identity(deployment_profile_id: str) -> RadioCheckpointIdentity:
    """Resolve a deployment into its complete checkpoint radio identity."""

    deployment = get_deployment_profile(deployment_profile_id)
    roles = {
        "ground": _radio_role(deployment.ground_bs_profile_id),
        "uav": _radio_role(deployment.uav_profile_id),
        "anchor": _radio_role(deployment.anchor_profile_id),
    }
    if deployment.hotspot_profile_id:
        roles["hotspot"] = _radio_role(deployment.hotspot_profile_id)

    role_labels = [
        f"{role}-{_slug(details['band'])}"
        for role, details in roles.items()
    ]
    directory_name = "__".join([
        _slug(deployment.profile_id),
        *role_labels,
    ])
    metadata = {
        "deployment_profile_id": deployment.profile_id,
        "access_generation": deployment.access_generation,
        "radio_roles": roles,
    }
    return RadioCheckpointIdentity(
        deployment_profile_id=deployment.profile_id,
        directory_name=directory_name,
        metadata=metadata,
    )


def resolve_checkpoint_root(checkpoint_root: str | Path | None = None) -> Path:
    """Return the absolute trained-model root without creating it."""

    root = Path(checkpoint_root or "trained_model").expanduser()
    if not root.is_absolute():
        root = _PROJECT_ROOT / root
    return root.resolve()


def radio_environment_dir(
    identity: RadioCheckpointIdentity,
    checkpoint_root: str | Path | None = None,
) -> Path:
    """Return this radio environment's directory below the model root."""

    return resolve_checkpoint_root(checkpoint_root) / identity.directory_name


def checkpoint_component_dir(
    identity: RadioCheckpointIdentity,
    component: str,
    checkpoint_root: str | Path | None = None,
) -> Path:
    """Return the offloading or mobility checkpoint directory."""

    component_name = str(component).strip().lower()
    if component_name not in _COMPONENTS:
        raise ValueError(
            f"unknown checkpoint component {component!r}; "
            f"expected one of {sorted(_COMPONENTS)}"
        )
    return radio_environment_dir(identity, checkpoint_root) / component_name


def write_radio_environment_manifest(
    identity: RadioCheckpointIdentity,
    checkpoint_root: str | Path | None = None,
) -> Path:
    """Create or validate the stable manifest shared by this environment."""

    environment_dir = radio_environment_dir(identity, checkpoint_root)
    environment_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = environment_dir / _MANIFEST_NAME
    if manifest_path.exists():
        validate_radio_environment_manifest(identity, checkpoint_root)
        return manifest_path

    payload = {
        "schema_version": 1,
        "directory_name": identity.directory_name,
        "radio_environment": identity.metadata,
    }
    temporary_path = environment_dir / f".{_MANIFEST_NAME}.{os.getpid()}.tmp"
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary_path, manifest_path)
    return manifest_path


def validate_radio_environment_manifest(
    identity: RadioCheckpointIdentity,
    checkpoint_root: str | Path | None = None,
) -> Path:
    """Reject missing or incompatible radio metadata before model loading."""

    manifest_path = radio_environment_dir(identity, checkpoint_root) / _MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "no training checkpoint exists for radio environment "
            f"{identity.directory_name!r}: missing {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid radio checkpoint manifest: {manifest_path}") from exc

    observed = payload.get("radio_environment")
    if observed != identity.metadata:
        raise RuntimeError(
            "radio checkpoint manifest does not match the active deployment: "
            f"{manifest_path}"
        )
    return manifest_path
