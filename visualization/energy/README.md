# Energy plots

Source: D:\6G-UrbanNetSim\6G-UrbanNetSim-1.1\output\run_002_fr3_urban_aiot_evaluation\evaluation
Seeds: (1, 2, 3)

The figure plan has 24 panel slots. Each measured panel is exported as its own PNG, without a heading; its filename gives the panel title.
Pending panels are listed here but are not exported. No synthetic stress or ablation results are substituted.

Workload is offered Gcycles per round in a coupled device-count/payload ramp. Arrival-rate CSV columns are empty; no tasks/s is inferred.
CDF/CCDF curves use every eligible observation. Completion-latency curves exclude failed tasks; failure and deadline figures retain all attempts.
Battery depletion/charge panels include battery-powered devices only (positive rated capacity); mains-powered devices are excluded from battery percentages.
Device energy is task-attributed; infrastructure task energy excludes shared hover/baseline operation. No lifetime or hard real-time guarantee is inferred.

## Panel map

- E1a: URLLC: battery depletion — generated — URLLC - battery depletion.png
- E1b: mMTC: battery depletion — generated — mMTC - battery depletion.png
- E1c: eMBB: battery depletion — generated — eMBB - battery depletion.png
- E1d: Median, P90 and worst depletion — generated — Median, P90 and worst depletion.png
- E2a: Transmission and reception — generated — Transmission and reception.png
- E2b: Local computation — generated — Local computation.png
- E2c: Waiting — generated — Waiting.png
- E2d: All workloads: device energy — generated — All workloads - device energy.png
- E3a: URLLC: energy per useful task — generated — URLLC - energy per useful task.png
- E3b: mMTC: energy per useful task — generated — mMTC - energy per useful task.png
- E3c: eMBB: energy per useful task — generated — eMBB - energy per useful task.png
- E3d: Energy and deadline success — generated — Energy and deadline success.png
- E4a: Initial-charge sensitivity — pending. Requires controlled initial-charge groups; current charge differences are not an independent battery sweep.
- E4b: Low-battery deadline success — pending. Requires predefined low-charge cohorts under matched controlled exposure.
- E4c: Reserve-threshold crossings — pending. Requires an agreed reserve threshold and independent workload/initial-charge experiments.
- E4d: Weak versus favorable channels — pending. Requires matched channel groups and initial charge; selected-route observations alone do not isolate channel effects.
- E5a: Fixed-cohort cumulative energy — generated — Fixed-cohort cumulative energy.png
- E5b: Fixed mMTC cohort charge — generated — Fixed mMTC cohort charge.png
- E5c: Energy under the coupled load ramp — generated — Energy under the coupled load ramp.png
- E5d: Time to battery reserve — pending. Requires a predefined reserve threshold and longer fixed-cohort operation; no lifetime estimate is substituted.
- E6a: Battery mechanism ablation — pending. Requires a specified battery mechanism and retrained ablation policies.
- E6b: Energy at common reliability — pending. Requires matched deadline-success operating points, not arbitrary comparisons at different service levels.
- E6c: All workloads: energy transfer — generated — All workloads - energy transfer.png
- E6d: Placement-rule baselines — pending. Requires documented local-only, feasible-edge and cloud-placement evaluations.

## Linked training panels

Four training-only PNGs are saved in training/: reward, last reported update loss, empirical action entropy, and a draft paper-specific service metric. Sources are resolved from evaluation job metadata, never evaluation observations. See training/README.md and training/training_manifest.json. Use --training-only to generate them independently, or --skip-training for evaluation figures alone.
