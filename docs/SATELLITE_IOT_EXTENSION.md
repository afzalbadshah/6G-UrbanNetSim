# Satellite-IoT extension (zones + UAV edge + LEO satellite)

The extension is deliberately small. Zones decide only which infrastructure
**physically exists** at a place. Inside a zone, the RL scheduler still learns
every decision from the measured delay, energy and cost. No tier is forced.

## Zones (`config/zones.py`)

| Zone | Area (default) | Infrastructure | Actions available |
|---|---|---|---|
| cellular | x 0–4 km | 6G FR3 base stations + MEC, Metro, Cloud | LOCAL, MEC, METRO, CLOUD, UAV_BS (if attached) |
| uav | x 4–7 km | 4 UAV-BS edge cells patrolling (no towers) + LEO | LOCAL, UAV_BS (only when the UAV link closes), SAT |
| remote | x 7–10 km | nothing on the ground; LEO satellite only | LOCAL, SAT |

* `ZONES_ENABLED = False` restores the original single-zone city. Use deployment profile `fr3_city_coverage_6g` in `config/simulation.py` as well.
* Devices spawn by zone (`ZONE_DEVICE_SHARE`, `ZONE_DEVICE_WEIGHTS`): the city mix in the cellular zone, and environmental devices (environment sensor, disaster alert sensor, camera trap, ranger wearable, drone) in the UAV and remote zones. Vehicles, pedestrians and drones can cross zones during the run.

## What decides the access link

* **UAV zone:** the device attaches to the strongest UAV-BS whose uplink is **not in outage**. This is checked with the same `SixGChannelModel` the packet engine uses (air-to-ground band `FR1_A2G`, LoS exponent 2.2 per 3GPP TR 36.777, about 1.8 km reach). If no UAV covers the device, it attaches to the satellite.
* **Remote zone:** the device always attaches to the satellite.

## Satellite model (minimal, transparent payload)

* LEO-600 (TR 38.821 Set-1). The serving satellite's elevation follows a periodic pass (25°→88°→25°, 8 min). The slant range is computed from elevation on a spherical Earth, which drives both propagation delay and path loss.
* Path: device → `LEO_SAT_S` / `LEO_SAT_KA` → `SAT_GW_1` (Ka feeder) → Cloud (300 km fibre). Everything runs through the normal packet engine, queues and energy accounting.
* **S-band IoT-NTN** (`ntn-s-access`, one 180 kHz NB-IoT carrier) is used by URLLC and mMTC devices. It gives roughly 90–240 kbit/s depending on elevation.
* **Ka-band VSAT** (`ntn-ka-access`, 20 MHz) is used by eMBB devices. It gives roughly 20–45 Mbit/s.
* Satellite transport is priced at 1e-9 USD/byte.
* Not modelled: orbit propagation, inter-satellite links, on-board processing, satellite handover signalling.

## Code changes

| File | Change |
|---|---|
| `config/zones.py` (new) | Zones, device mix per zone, satellite config, elevation and slant-range functions |
| `config/radio.py` | Bands `FR1_A2G`, `NTN_S`, `NTN_KA`, `NTN_FEEDER`; link profiles `nr-fr1-a2g`, `ntn-s-access`, `ntn-ka-access`, `ntn-feeder` |
| `config/topology.py` | `DeploymentProfile` fields `access_zone`, `uav_zone`, `ntn_enabled`; new profile `zoned_satellite_6g`; node kind `sat_gateway` |
| `config/energy.py` | Radio power profiles for the satellite and the gateway |
| `config/devices_config.py` | `disaster_alert_sensor` (URLLC) and `camera_trap` (eMBB). Weight 0 in the city mix |
| `config/learning.py` | `STATE_DIM` 82→96, `ACTION_DIM` 6→7 (**retrain all models**) |
| `config/scheduling.py` | SAT added at the end of the spillover order |
| `config/simulation.py` | Default deployment set to `zoned_satellite_6g` |
| `core/deployment_builder.py` | Base stations and MEC placed in the cellular zone, UAV cells in the UAV zone; satellite + gateway nodes, links and routes |
| `core/network_runtime.py`, `network/topology.py` | NTN is now an access node kind; gateway node kind added |
| `core/simulator.py` | Spawning by zone; UAV/satellite attachment outside the city; pass geometry each round; SAT execution path; map links |
| `core/task_execution.py` | Satellite transport price |
| `scheduler/common/common_action.py` | `ACTION_SAT = 6` |
| `scheduler/common/common_servers.py` | Action availability by zone; SAT resolves to Cloud through the satellite route |
| `scheduler/common/common_state.py` | 4 new observable features: zone (one-hot) and sin(satellite elevation) |
| `scheduler/common/comparison_runner.py` | CSV columns `zone_id`, `sat_available`, `sat_elevation_deg`, `sat_slant_range_m`, `sat_access_node`, `sat_selected` |
| `live/*`, `training_visualization_helper.py` | Zone shading, satellite marker, vegetation outside the city, SAT colour in plots |

## Check run (200 training rounds, then a 100-round evaluation, up to 150 devices, seed 42)

Share of actions chosen by DQN in the UAV zone, split by whether a UAV-BS covered the device:

| Class | UAV in range | No UAV in range |
|---|---|---|
| URLLC | LOCAL 76%, UAV 24% (both about 1 ms) | LOCAL 88%, SAT 12% |
| mMTC | **UAV 100%** (0.08 mJ per task vs 12 mJ local) | LOCAL 97% |
| eMBB | **SAT→Cloud 100%** (about 30 ms vs about 300 ms on the UAV) | SAT 100% |

In the cellular zone, mMTC goes 100% to MEC and URLLC about 75% to MEC. In the remote zone, eMBB goes 100% to SAT.
For mMTC, satellite uplink energy (about 12 mJ per 2 KB at NB-IoT NTN rates) is about the same as computing locally. **Only the UAV edge saves mMTC battery.**
