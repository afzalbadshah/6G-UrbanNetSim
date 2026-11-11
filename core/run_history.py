"""Versioned campaign-output layout shared by launchers and visualizations.

Every simulator launch gets one monotonically numbered directory below the
configured output root.  Plotters use the same parser to select the highest
run number, so writing and reading can never disagree about folder structure.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


_RUN_NAME = re.compile(r"^run_(\d+)(?:_|$)", re.IGNORECASE)


def _slug(value: object) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip())
    return text.strip("_").lower() or "scenario"


def run_number(path: str | Path) -> int | None:
    """Return the numeric version from ``run_<number>_*`` or ``None``."""

    match = _RUN_NAME.match(Path(path).name)
    return int(match.group(1)) if match else None


def discover_runs(output_root: str | Path) -> list[Path]:
    """Return versioned run directories in ascending numeric order."""

    root = Path(output_root)
    if not root.is_dir():
        return []
    runs = [path for path in root.iterdir() if path.is_dir() and run_number(path) is not None]
    return sorted(runs, key=lambda path: (run_number(path), path.name.lower()))


def latest_run(output_root: str | Path) -> Path | None:
    """Return the highest numbered run, independent of timestamps or size."""

    runs = discover_runs(output_root)
    return runs[-1] if runs else None


def create_run(
    output_root: str | Path,
    *,
    deployment_profile: str,
    mode: str,
    created_utc: datetime | None = None,
) -> Path:
    """Atomically create and describe the next versioned campaign directory."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    next_number = max((run_number(path) or 0 for path in discover_runs(root)), default=0) + 1
    label = f"{_slug(deployment_profile)}_{_slug(mode)}"

    while True:
        run_dir = root / f"run_{next_number:03d}_{label}"
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
            break
        except FileExistsError:
            # Two launchers can start together. The successful mkdir owns that
            # number; the other launcher advances without sharing a run folder.
            next_number += 1

    timestamp = created_utc or datetime.now(timezone.utc)
    manifest = {
        "run_number": next_number,
        "run_name": run_dir.name,
        "created_utc": timestamp.astimezone(timezone.utc).isoformat(),
        "deployment_profile": str(deployment_profile),
        "mode": str(mode),
        "phase": "exploration" if str(mode).lower() == "training" else "evaluation",
    }
    # Create the phase immediately so a visualization started during a long
    # campaign can identify the latest compatible run before the first CSV is
    # flushed. The simulator keeps ownership of all deeper seed/model folders.
    (run_dir / manifest["phase"]).mkdir()
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return run_dir


def select_run(
    output_root: str | Path,
    selected_run: str | Path | None = None,
) -> Path | None:
    """Resolve an explicitly pinned run or otherwise the highest run number."""

    root = Path(output_root)
    if selected_run is None or not str(selected_run).strip():
        return latest_run(root)
    selected = Path(selected_run)
    candidate = selected if selected.is_absolute() else root / selected
    if not candidate.is_dir():
        raise FileNotFoundError(f"configured visualization run does not exist: {candidate}")
    return candidate


def select_phase(
    output_root: str | Path,
    phase: str,
    selected_run: str | Path | None = None,
) -> Path:
    """Resolve the highest run containing ``phase``, unless explicitly pinned."""

    root = Path(output_root)
    phase_name = str(phase)
    if selected_run is not None and str(selected_run).strip():
        phase_dir = select_run(root, selected_run) / phase_name
        if not phase_dir.is_dir():
            raise FileNotFoundError(
                f"configured visualization run does not contain "
                f"the {phase_name!r} phase: {phase_dir}"
            )
        return phase_dir

    runs = discover_runs(root)
    for run_dir in reversed(runs):
        if (run_dir / phase_name).is_dir():
            return run_dir / phase_name
    if runs:
        # A manually created run may not have a manifest/phase yet. Returning
        # the highest number keeps the rule deterministic and debuggable.
        return runs[-1] / phase_name
    return root / phase_name
