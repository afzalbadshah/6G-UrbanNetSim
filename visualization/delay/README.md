# Delay plots

Source: D:\6G-UrbanNetSim\6G-UrbanNetSim-1.1\output\run_002_fr3_urban_aiot_evaluation\evaluation
Seeds: (1, 2, 3)

The figure plan has 24 panel slots. Each measured panel is exported as its own PNG, without a heading; its filename gives the panel title.
Pending panels are listed here but are not exported. No synthetic stress or ablation results are substituted.

Workload is offered Gcycles per round in a coupled device-count/payload ramp. Arrival-rate CSV columns are empty; no tasks/s is inferred.
CDF/CCDF curves use every eligible observation. Completion-latency curves exclude failed tasks; failure and deadline figures retain all attempts.
Battery depletion/charge panels include battery-powered devices only (positive rated capacity); mains-powered devices are excluded from battery percentages.
Device energy is task-attributed; infrastructure task energy excludes shared hover/baseline operation. No lifetime or hard real-time guarantee is inferred.

## Panel map

- R1a: URLLC: deadline success — generated — URLLC - deadline success.png
- R1b: mMTC: deadline success — generated — mMTC - deadline success.png
- R1c: eMBB: deadline success — generated — eMBB - deadline success.png
- R1d: Lower-tail device success — generated — Lower-tail device success.png
- R2a: URLLC: full latency tail — generated — URLLC - full latency tail.png
- R2b: mMTC: full latency tail — generated — mMTC - full latency tail.png
- R2c: eMBB: full latency tail — generated — eMBB - full latency tail.png
- R2d: Completion latency P99 — generated — Completion latency P99.png
- R3a: Transmission and propagation — generated — Transmission and propagation.png
- R3b: Queueing — generated — Queueing.png
- R3c: Processing — generated — Processing.png
- R3d: Handover — generated — Handover.png
- R4a: On-time, late and failed tasks — generated — On-time, late and failed tasks.png
- R4b: Recorded failure causes — generated — Recorded failure causes.png
- R4c: Late completion under workload — generated — Late completion under workload.png
- R4d: Unsuccessful-update bursts — generated — Unsuccessful-update bursts.png
- R5a: Application deadline tightness — pending. Requires independently varied deadlines and re-evaluation; not a counterfactual relabelling of logged outcomes.
- R5b: Alarm burst intensity — pending. Requires controlled burst-traffic experiments at fixed device count and background load.
- R5c: Channel impairment — pending. Requires matched channel-impairment experiments.
- R5d: Mobility and handover exposure — pending. Requires independently controlled mobility experiments.
- R6a: Deadline mechanism ablation — pending. Requires a specified new mechanism, retrained ablation policies and matched evaluation.
- R6b: Mobility mechanism ablation — pending. Requires a mobility mechanism and its controlled ablation; current model differences are not ablations.
- R6c: Reliability under energy budgets — pending. Requires matched energy-budget operating points.
- R6d: Decision overhead and budget — pending. Inference time is not recorded; measure decision latency and define the deployment platform.

## Linked training panels

Four training-only PNGs are saved in training/: reward, last reported update loss, empirical action entropy, and a draft paper-specific service metric. Sources are resolved from evaluation job metadata, never evaluation observations. See training/README.md and training/training_manifest.json. Use --training-only to generate them independently, or --skip-training for evaluation figures alone.
