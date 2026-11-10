# Metric calculation paths

The selected task's packet execution records one channel result per attempted
hop. These observations flow into the task delay, energy, and cost ledgers.
"Real time" here means the simulated link state at that attempt, not a
measurement of a physical network or wall-clock execution time.

| Module | Inputs and calculation | Effect of changing the channel |
| --- | --- | --- |
| `channel.py` | Band/interface profile, distance, transmit power, antenna gain, path loss, shadowing, fading, allocated bandwidth, noise figure, and interference determine received power, SNR/SINR, achievable rate, error probability, and outage. | Direct. |
| `delay.py` | On-wire bits / observed achievable bit rate, plus measured queue, propagation, retry, outage, handover, and processing intervals. Wired links use their own configured rate and propagation speed, not radio SNR. | Direct for wireless serialization and retries; indirect for queue occupancy. |
| `energy.py` | Radio and transport power over observed active/wait intervals, on-wire bits, and compute power over processing time. Retransmissions contribute additional active time and bytes. | Indirect through transmission duration and retry outcomes. |
| `cost.py` | Explicit tariff multiplied by measured joules, on-wire/useful bytes, CPU time, capacity time, or completed objects, according to the tariff basis. | Conditional: a joule or on-wire-byte tariff changes with channel outcomes; a fixed completed-object tariff need not. |
| `reward.py` | Canonical measured outcomes plus the class-objective matrix in `config/reward.py`. Produces the sole scalar learning reward and an auditable component breakdown. | Indirect through the measured delay, device energy, cost, capacity fit, and continuous marginal shared-capacity pressure; contains no tier ladder or fixed utilization threshold. |

`core/task_execution.py` combines these measurements for the selected action.
The scheduler can estimate other actions, but the selected action's packet
result is the source for the realized physical metrics.

`handover_channel.py` binds configured radio parameters to the channel
formulas for attachment decisions. Simulator reporting requires a realized
packet/compute result for every assigned task; it does not recalculate
delay, energy or cost with fallback proxies. No legacy metric calculation
module remains in `config/` or `metrics/`.
