"""Trigger aggregate plots safely from the parent campaign process.

The coordinator observes completed worker summaries. It schedules one model's figures
only after all of that model's seeds finish, then writes the cross-model comparison when
the campaign ends. Plot work uses one background thread so a progress GUI stays usable.

How to extend: add completion rules here; never call shared plot generation in workers.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Mapping

from visualization.supportive.requested_training_plots import (
    generate_campaign_summary,
    generate_model_figures,
)


class CampaignPlotCoordinator:
    """Coordinate parent-only plotting as seed/model jobs complete."""

    def __init__(
        self,
        phase_dir: str | Path,
        models: Iterable[str],
        seed_count: int,
        moving_average_window: int = 500,
        dpi: int = 180,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        """Store the expected campaign shape and prepare one serialized plot worker."""
        self.phase_dir = Path(phase_dir)
        self.models = tuple(dict.fromkeys(str(model).strip().lower() for model in models))
        self.seed_count = max(1, int(seed_count))
        self.moving_average_window = max(1, int(moving_average_window))
        self.dpi = max(1, int(dpi))
        self.logger = logger or (lambda message: None)
        self._completed_seed_indices = {model: set() for model in self.models}
        self._scheduled_models: set[str] = set()
        self._futures: list[Future] = []
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="campaign-plots")
        self._finalized = False

    def record_job_completion(self, result: Mapping[str, object]) -> None:
        """Record one successful worker result and schedule its model when complete."""
        if self._finalized:
            return
        model = str(result.get("model", "")).strip().lower()
        if model not in self._completed_seed_indices:
            self.logger(f"[PLOT][WARN] ignoring completion for unknown model {model!r}")
            return
        try:
            seed_index = int(result["seed_index"])
        except (KeyError, TypeError, ValueError):
            self.logger(f"[PLOT][WARN] completion for {model} has no valid seed_index")
            return
        self._completed_seed_indices[model].add(seed_index)
        completed = len(self._completed_seed_indices[model])
        self.logger(f"[PLOT] {model}: {completed}/{self.seed_count} seeds complete")
        if completed >= self.seed_count:
            self._schedule_model(model)

    def _schedule_model(self, model: str) -> None:
        """Submit one model once; the single executor prevents shared-file races."""
        if model in self._scheduled_models:
            return
        self._scheduled_models.add(model)
        self._futures.append(self._executor.submit(self._generate_model, model))

    def _generate_model(self, model: str) -> list[Path]:
        """Generate one model's figures while preserving worker completion flow."""
        try:
            paths = generate_model_figures(
                self.phase_dir,
                model,
                moving_average_window=self.moving_average_window,
                dpi=self.dpi,
            )
            self.logger(f"[PLOT] {model}: wrote {len(paths)} aggregate figures")
            return paths
        except Exception as exc:
            self.logger(f"[PLOT][ERROR] {model} aggregate figures failed: {exc}")
            return []

    def finalize(self) -> list[Path]:
        """Wait for model plots, then write the final all-model figure and manifest."""
        if self._finalized:
            return []
        self._finalized = True
        for model in self.models:
            completed = len(self._completed_seed_indices[model])
            if completed >= self.seed_count:
                self._schedule_model(model)
            else:
                self.logger(
                    f"[PLOT][WARN] {model} has only {completed}/{self.seed_count} "
                    "completed seeds; its aggregate plots are incomplete"
                )
        self._executor.shutdown(wait=True)
        try:
            paths = generate_campaign_summary(
                self.phase_dir,
                self.models,
                moving_average_window=self.moving_average_window,
                dpi=self.dpi,
            )
            self.logger("[PLOT] final all-model comparison and manifest written")
            return paths
        except Exception as exc:
            self.logger(f"[PLOT][ERROR] final campaign figures failed: {exc}")
            return []
