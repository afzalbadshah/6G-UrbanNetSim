# Supportive campaign visualization

This package contains plots that summarize several seeds after simulation output has
been written. It does not participate in action selection, reward calculation, network
execution, or learning updates.

## Automatic behavior

For a multi-seed training or evaluation campaign, the parent launcher:

1. generates one model's aggregate figures after every configured seed for that model
   has completed;
2. generates the all-model main-reward comparison after the campaign has completed; and
3. writes `plot_manifest.json` beside the generated figures.

Parallel workers never generate shared aggregate plots. This prevents partially written
figures and worker-to-worker overwrites.

## Output location

Source code lives here. Generated figures remain with their experiment data:

```text
output/run_001_<scenario>_<mode>/<phase>/plots/
├── by_model/<model>/
└── comparison/all_models_main_reward.png
```

Keeping generated results outside `visualization/` prevents research artifacts from
being confused with simulator source code.

## Manual regeneration

```text
python -m visualization.supportive.plot_training_results
python -m visualization.supportive.plot_training_results output/run_002_fr3_city_coverage_6g_training/exploration
```

With no path, the highest numbered compatible run is selected automatically.
Set the `exploration` and `evaluation` entries in `VISUALIZATION_RUNS` inside
`visualization/plot_config.py` to independently pin either phase to a named or
absolute run directory. `SMARTCITY_PLOT_PHASE` remains available as a direct
one-command override. Use `--window <rounds>` to change only the display
smoothing; raw CSV data is never modified.

```python
# Automatic: select the newest compatible run for each phase.
VISUALIZATION_RUNS = {
    "exploration": None,
    "evaluation": None,
}

# Pinned: keep using these two runs until this configuration is changed.
VISUALIZATION_RUNS = {
    "exploration": "run_005_fr3_city_coverage_6g_training",
    "evaluation": "run_006_fr3_city_coverage_6g_evaluation",
}
```

Each value is a run directory, not its `exploration` or `evaluation`
subdirectory. Absolute run paths are also accepted.
