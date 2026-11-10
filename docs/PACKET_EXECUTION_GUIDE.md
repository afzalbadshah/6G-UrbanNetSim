# Packet Execution Guide

This guide explains how one simulator task becomes packets, crosses the network,
uses a compute server, and becomes a row of research results. Start here before
reading the implementation.

## The five-stage workflow

```text
Task selected by scheduler
        |
        v
1. Build physical route
   device -> current access point -> declared backbone -> compute server
        |
        v
2. Packetize and transmit
   application bytes -> packets -> queue/channel/retry outcome on every hop
        |
        v
3. Execute compute work
   only after the request reaches the selected server
        |
        v
4. Account for the observation
   delay + bytes + energy + monetary cost + deadline result
        |
        v
5. Return one RealizedTaskExecution
   scheduler uses it for reward, learning, battery debit, status, and CSV output
```

The scheduler may estimate several possible actions before choosing one. Those
estimates are decision inputs only. After selection, the workflow above executes
exactly one action and produces the observed result.

## Which class does what?

| Component | Responsibility | Does not do |
|---|---|---|
| `Simulator` | Connects devices, servers, mobility, scheduler, and one shared runtime | Packet formulas or tariff formulas |
| `TaskExecutionRuntime` | Orchestrates one selected task from access wait through network and compute | Choose the offloading action |
| `DynamicAccessRouteBuilder` | Prepends the device's live radio hop to a declared infrastructure route | Invent shortest paths |
| `RuntimeHopParameterBuilder` | Supplies radiated power, bandwidth, interference, LoS, and elevation for each hop | Read legacy battery-draw power as RF power |
| `NetworkTaskAdapter` | Converts legacy decimal kB/ms values into canonical bytes/seconds | Change the legacy `Task` object |
| `NetworkEngine` | Packetizes, queues, evaluates channels, retries, and records delivery | Calculate compute energy or price |
| `metrics.delay` | Returns the one authoritative task-delay breakdown from observed access, packet and compute intervals | Choose routes or recalculate radio capacity |
| `metrics.energy` | Calculates joules from physical power and duration | Mutate batteries |
| `metrics.cost` | Applies one named tariff to one measured quantity | Combine operator cost with consumer price |
| `ComparisonRunner` | Uses the realized result for reward, learning, resource release, and output | Recalculate the physical result |

## Important result objects

- `TaskDelayBreakdown` (also available as the legacy name `RealizedDelay`):
  immutable, mutually exclusive delay components stored in seconds, with
  read-only millisecond properties for the scheduler and CSV files.
- `RealizedEnergy`: device radio/wait energy, infrastructure network energy,
  compute energy, and facility overhead in joules. Its components must add up to
  total energy.
- `RealizedTaskExecution`: the single authoritative result for one chosen task.
  It includes delivery status, bytes, packet counts, deadline outcome, cost, and
  exact server battery debits.

These objects are immutable. This prevents later reporting code from silently
changing an observed scientific result.

## Where a researcher should make changes

| Research question | Change point |
|---|---|
| Add or tune a radio/link profile | `config/network_config.py` and the referenced band profile |
| Change physical topology or declared routes | `config/topology.py` |
| Change endpoint or compute power | `config/power.py` |
| Change a tariff | `config/tariffs.py` |
| Add a new energy equation | `metrics/energy.py`; keep the engine adapter thin |
| Add a new cost equation or perspective | `metrics/cost.py`; never mix perspectives |
| Add a packetization policy | Implement the packetizer contract in `network/packetization.py` |
| Add a channel model | Implement the channel contract in `metrics/channel.py` |
| Change a delay equation or add a physical delay component | `metrics/delay.py`; never reconstruct totals in the scheduler or reporter |
| Change an offloading decision | Scheduler policy modules; do not put policy logic in the network engine |

Do not place experimental constants inside `TaskExecutionRuntime`. Surface them
in a configuration profile so experiments remain reproducible and reviewable.

## Local and remote execution

Local execution is intentionally simple: no route, no packets, and no network
energy. It reports only local processing delay and local compute energy.

Remote execution requires a current access point. The runtime waits for any
handover or coverage interruption, builds a route using live three-dimensional
positions, transmits packets, and performs compute only when delivery succeeds.
If access is unavailable for the whole deadline, zero packets are sent and only
device waiting energy is charged.

## Delay calculation

The simulator calculates delay in the same order a task experiences it:

| Component | Formula or observation |
|---|---|
| Wireless achievable rate | `bandwidth_hz × log2(1 + SINR) × link_efficiency`; SINR uses received power, thermal noise and interference in `metrics.channel` |
| Wireless transmission | `transmitted_on_wire_bits / wireless_achievable_rate_bps` for every actual attempt |
| Wired transmission | `transmitted_on_wire_bits / configured_wired_rate_bps` for every actual attempt |
| Wireless propagation | `three_dimensional_radio_distance_m / wireless_propagation_speed_m_s` |
| Wired propagation | `declared_route_distance_m / wired_propagation_speed_m_s` |
| Queue waiting | Observed wait before the shared physical resource starts service |
| Retry delay | Observed backoff between failed attempts |
| Channel-outage delay | Observed outage-check duration when no transmission service was reserved |
| Handover/access delay | Observed handover interruption plus only the non-overlapping part of a coverage outage |
| Processing delay | Actual compute-service duration after successful network delivery |

The complete formulas are:

```text
network delay = wireless queue + wired queue
              + wireless transmission + wired transmission
              + wireless propagation + wired propagation
              + retry backoff + channel-outage observation + terminal wait

complete task delay = handover delay + non-overlapping coverage outage
                    + network delay + processing delay
```

Packet headers, trailers and retransmissions are included because transmission
uses on-wire bytes. Several devices do not receive independent copies of the
configured link rate: their packets reserve the same physical resource and later
packets receive an observed queue delay. Reporters consume this returned result
and do not calculate any component again.

## Measurement rules

1. Internal packet/network units are SI: bytes, seconds, metres, hertz, watts,
   joules, bit/s, dB, and dBm.
2. Legacy decimal kB and milliseconds are converted once in
   `LegacyUnitBoundary`.
3. Payload bytes and on-wire bytes are different. On-wire bytes include declared
   headers/trailers and retransmissions.
4. Failed delivery cannot become successful application completion.
5. Operator cost and consumer price are separate accounting perspectives.
6. Unknown cost components are marked incomplete; they are never silently zero.
7. Predictions and observations use different output labels.

## Safe extension checklist

Before accepting a change:

1. Put tunable values in `config/`, not inside an execution function.
2. Use a unit suffix in every physical quantity name.
3. Add a focused hand-check test for the new formula or stage.
4. Verify component sums for bytes, delay, energy, and cost.
5. Run the full suite and the one-round live integration test.
6. Document whether the mechanism is validated, standards-aligned, or
   experimental.
