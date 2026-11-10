# Unmasked service-objective learning

Reward schema: `unmasked-service-objectives-v1`.

Every policy sees all seven action IDs during training and evaluation. Physical
availability is part of the observation, not an action mask. The executor
attempts the selected action only: it never moves a task to a preferred tier.
Unavailable actions are rejected and included in learning and action counts.
Actual hardware, routing, battery depletion, and admission limits still apply.

Q-learning and DQN variants use epsilon-greedy exploration. A2C and PPO instead
sample the actor's categorical distribution and use entropy regularization.
They do not inject epsilon-random actions, because those actions would have a
different behavior probability from the one optimized by the policy-gradient
loss.

## Service objectives

| Class | Successful-outcome score S | Meaning |
|---|---|---|
| URLLC | `1 / (1 + measured_latency_ms / 2)` | Lower completion delay is better. |
| mMTC | `1 / (1 + ln(1 + task_device_energy_j / 0.00002))` | Lower incremental device energy is better. |
| eMBB | `(free_fraction + scalable_capacity) / 2` | Prefer useful spare compute and resource headroom. |

`free_fraction` is one minus current peak resource utilization (CPU, memory,
storage for remote servers; CPU and memory locally). `scalable_capacity` is
`log(1 + spare_compute) / log(1 + largest_available_spare_compute)` for the
current decision, with zero for unavailable actions. Spare compute uses the
peak-utilization headroom as a conservative approximation. This is a measured,
tier-neutral resource-scalability proxy, not a throughput score or tier bonus.

Each class assigns weight 1 to its primary objective and 0 to all other reward
components. Server battery, remaining device battery, cost, and mobility do not
add reward. Physical battery accounting and battery state features remain.

The preference scales above are existing reference values, not new SLAs or
validated deployment tuning. Logarithmic energy scaling avoids hard saturation
for satellite and onboard energy magnitudes. No service or tier name enters
the reward formula.

## Common penalties and audit fields

For executed decisions, overload pressure is
`P = clip((projected_utilization_percent - 95) / 5, 0, 1)^2`.
Successful reward is `clip(10*S - 2*P, -10, 10)`. Actual execution failures or
missed application deadlines receive -10. Missing-resource utilization sentinels
are not treated as overload. Selecting an unavailable service is diagnosed as
an invalid attempt; the failure floor remains -10 rather than stacking penalties
below the declared reward range.

The CSV includes `reward_overutilization_deduction_points`, the nominal `2*P`
before the failure floor, and `reward_schema_version`. Existing raw component
fields remain diagnostic; a logged component is not necessarily reward-bearing.
`physical_mask` records availability while `mask` and `learning_mask` are all
ones. `physical_valid_count` is the actual available-choice count.

Single-available-action decisions retain their measured rewards. No automatic
spillover or deadline exemptions apply in the learned runner. If every service
is unavailable, the CSV marks `no_service_available=1`; the failed service
outcome is still recorded as -10, not disguised as success or neutral reward.
This is absolute service performance, not a causal estimate of agent blame.

## Zones

| Scenario | Infrastructure choices, when operational |
|---|---|
| Cellular | Onboard, MEC, Metro, Cloud, UAV |
| UAV edge | Onboard, UAV, satellite |
| Remote | Onboard, satellite |

Onboard requires local compute hardware. Satellite execution requires its link
and gateway/cloud path. Absence of towers has no standalone reward penalty.
Successful remote delivery is scored on measured outcomes. The same resource
pressure formula applies in all zones; application deadlines are never relaxed
to improve a reward plot. Report outcomes by zone and class, including rejected
attempts and no-service events.

## Checkpoints and validation

The runner stores fresh policies under
`CHECKPOINT_ROOT/unmasked-service-objectives-v1/` (default root `trained_model`).
Evaluation of learned models requires a checkpoint in that directory. Old
masked checkpoints and existing experiment outputs are preserved. Q-learning
now recognizes the 26-feature zone-aware prefix and correct tier offsets.

Validation must test objective monotonicity, zero server-battery influence,
unmasked action support, exact selected/executed agreement, rejected-action
learning, overload deductions, and missing-checkpoint failure. Follow with
fresh training and held-out evaluation across zones, classes, and seeds.
Do not force tier shares or model differences: onboard can be slower on weak
devices, and offloading can consume more device energy on a poor radio link.
These one-step terminal decisions also mean DQN and Double DQN have identical
target equations here; similar results alone are not evidence of enforcement.
