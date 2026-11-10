"""Post-process completed SmartCityEdgeSim training and evaluation campaigns.

How to extend: add reusable plotting code inside this package and keep generated
experiment artifacts under ``output/<phase>/plots`` rather than in the source tree.
"""

from visualization.supportive.automatic_plotting import CampaignPlotCoordinator
from visualization.supportive.training_aggregate_plots import (
    generate_model_figures,
    generate_training_figures,
)

__all__ = (
    "CampaignPlotCoordinator",
    "generate_model_figures",
    "generate_training_figures",
)
