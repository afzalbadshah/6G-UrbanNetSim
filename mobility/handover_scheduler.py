"""
mobility/handover_scheduler.py

RL-based handover (wireless-attachment) scheduler, ported and adapted from
DriveNetSim's mobility/acmobility.py::HandoverScheduler.

Responsibility (and only responsibility): decide, each round, which wireless
point (ground MEC *or* aerial UAV-BS) each mobile device is attached to,
execute the handover, and expose a rich set of mobility/RF decision-quality
signals (RSSI, SNR/SINR, oracle-vs-chosen regret, ping-pong, dwell-time
prediction, coverage outage, handover failure/interruption) so the main
offloading scheduler and the CSV reporters consume real numbers instead of
placeholders.

Adaptation notes vs. DriveNetSim (and vs. the first port of this file):
  - DriveNetSim vehicles move on a 1D circular road against a FIXED RSU
    array; the basic decision rule (RSSI thresholds, hysteresis, ping-pong
    avoidance, dwell-time-triggered handover) is unchanged from there --
    only the underlying distance/signal metrics needed new formulations:
      * Candidates now include UAV-BS, not just ground MEC -- both are
        genuine "cellular points" a device can attach to.
      * Any link touching an airborne endpoint (task-generating UAV or
        UAV-BS) uses a real 3D distance (adds altitude) and an
        air-to-ground path-loss exponent, instead of the 2D-only,
        ground-multipath model that's appropriate for MEC-only links.
      * UAV-BS is itself mobile (unlike every base station DriveNetSim or
        the first port ever had to model) -- dwell-time/urgency estimates
        now use the *relative* speed between device and serving node, not
        just the device's own speed, since a moving BS changes the
        distance on its own even for a stationary device.
      * Handover hysteresis thresholds now scale with device speed: a
        pedestrian (0.5-2.2 m/s) and a vehicle (10-33 m/s) shouldn't be
        held to the same dB-gap bar tuned implicitly around vehicular RSSI
        change rates in DriveNetSim -- faster devices get a wider margin
        (avoid ping-pong while crossing cells quickly), slower devices get
        a narrower one (little ping-pong risk, so snap to the objectively
        better cell sooner).
"""

import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config as cfg
from config.radio import get_band, get_link_profile
from config.topology import get_deployment_profile
from metrics.delay import remaining_coverage_time_s
from metrics.handover_channel import (
    calculate_rssi_dbm,
    calculate_noise_power_dbm,
    calculate_interference_power_dbm,
    calculate_snr_db,
    calculate_sinr_db,
    max_coverage_radius_m,
    air_ground_path_loss_exponent,
)
from scheduler.common.common_servers import server_is_operational, server_utilization
from scheduler.common.policy_utils import load_checkpoint, save_checkpoint
from core.reporter import TrainingReporter



class _Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim), nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


class _Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


# Access-only ground BSs and aerial UAV-BSs are genuine cellular attachment
# points.  Ground MECs are direct radio candidates only in the legacy sparse
# profile where no access-only BS layer has been deployed.
def candidate_access_points(servers):
    """Return the one canonical, fixed-order radio candidate list.

    The list deliberately contains both ground access points and UAV-BSs so
    the actor's input/output dimensions remain stable.  Per-device coverage
    eligibility is applied later as a mask: usable ground coverage is tried
    first and UAV-BS is the fallback.
    """
    all_servers = list(servers or [])
    access_points = [s for s in all_servers if bool(getattr(s, "is_access_only", False))]
    out = []
    for s in all_servers:
        tier = str(getattr(s, "tier", "") or "").upper()
        name = str(getattr(s, "name", "") or "")
        is_uav = tier == "UAV_BS" or name.startswith("UAV_BS_")
        is_ground_mec = tier == "MEC" or name.startswith("MEC_")
        # A promoted MEC (topology node provides_radio_access=True -- see
        # core.simulator._initialize_servers) is a genuine radio endpoint in
        # its own right, exactly like an is_access_only=True AccessPoint,
        # regardless of whether an explicit access-only layer also exists.
        is_promoted_mec = bool(getattr(s, "provides_radio_access", False))
        # Once explicit ground access exists, never bypass it by attaching a
        # device directly to a compute MEC.  The BS's declared anchor/backbone
        # route carries traffic to that MEC instead.
        if (access_points and bool(getattr(s, "is_access_only", False))) or is_uav or is_promoted_mec:
            out.append(s)
        elif not access_points and is_ground_mec:
            out.append(s)
    return out


def _entity_altitude_m(obj):
    """Height above ground (m): task-generating UAVs and UAV-BS servers
    carry this on their mobility model (.z); anything ground-level (MEC,
    vehicles, pedestrians, stationary devices) has none, so 0.0."""
    z = getattr(obj, "z", None)
    if z is None:
        z = getattr(getattr(obj, "mobility", None), "z", None)
    return float(z) if z is not None else 0.0


def _entity_speed_m_s(obj):
    """Own movement speed (m/s) of a server, if it has any mobility at all
    (UAV-BS patrols at a steady speed; MEC/Metro/Cloud have none)."""
    speed = getattr(getattr(obj, "mobility", None), "speed", None)
    return float(speed) if speed is not None else 0.0


class HandoverScheduler:
    """Actor-Critic RL agent whose only job is wireless-point attachment.

    A dense deployment exposes radio-only ground BSs and UAV-BSs.  A sparse
    legacy deployment falls back to ground MECs as its radio points.
    """

    FEATURES_PER_CANDIDATE = 5

    @staticmethod
    def _resolve_band_name(candidate, deployment):
        """The candidate's ACTUAL resolved band (FR1/FR3/mmWave/subTHz/LowBand).

        An AccessPoint carries its own ``profile_id`` (set per-node from the
        active DeploymentProfile). A Server-tier candidate (MEC/UAV_BS acting
        as a promoted radio point) has no such field, so it falls back to the
        deployment's own ground/UAV profile id -- the same one the physical
        topology/packet-execution layer resolves for that tier.
        """
        profile_id = getattr(candidate, "profile_id", None)
        if not profile_id:
            tier = str(getattr(candidate, "tier", "")).upper()
            profile_id = (
                deployment.uav_profile_id if tier == "UAV_BS"
                else deployment.ground_bs_profile_id
            )
        return get_link_profile(profile_id)["band"]

    @staticmethod
    def _resolve_anchor_band_name(candidate, deployment):
        """The candidate's low-band coverage anchor (dual-connectivity fallback).

        Every deployment profile declares one (default "nr-lowband-access");
        an AccessPoint carries its own resolved ``anchor_profile_id`` (set
        per-node in core/deployment_builder.py), a Server-tier candidate
        (MEC/UAV_BS) falls back to the deployment's own anchor id, same
        pattern as _resolve_band_name.
        """
        anchor_profile_id = getattr(candidate, "anchor_profile_id", None) or deployment.anchor_profile_id
        return get_link_profile(anchor_profile_id)["band"]

    def __init__(self, servers, deployment_profile_id=None):
        candidates = candidate_access_points(servers)
        self.num_candidates = max(1, len(candidates))
        self.state_dim = self.FEATURES_PER_CANDIDATE * self.num_candidates
        self.action_dim = self.num_candidates

        self.out_dir = None
        self.model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "trained_model"))
        os.makedirs(self.model_dir, exist_ok=True)

        self.gamma = float(getattr(cfg, "DISCOUNT_FACTOR", 0.95))
        self.lr = 5e-4
        self.actor = _Actor(self.state_dim, self.action_dim)
        self.critic = _Critic(self.state_dim)
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.train_mode = False
        self.reward_log, self.loss_log, self.action_dist_log = [], [], []

        # Per-device runtime state (keyed by device.id)
        self.prev_bs_for_log = {}
        self.prev_prev_bs = {}
        self.predicted_dwell_time_ms = {}
        self.true_dwell_time_ms = {}
        self.remaining_dwell_time_ms = {}

        # Per-device cached (state, action) for the pending actor-critic update
        self._last_state = {}
        self._last_action = {}
        self._current_step = 0

        # Gradient steps (forward+backward+optimizer.step on actor & critic)
        # dominate wall-clock at high device counts far more than collecting
        # the (state, action, reward) tuple itself. Instead of one single-
        # sample update per device, accumulate a small batch and take one
        # batched update -- fewer, larger gradient steps, closer to the
        # textbook multi-sample Advantage Actor-Critic update than N separate
        # single-sample ones.
        self.learn_every = max(1, int(getattr(cfg, "LEARN_EVERY_N_DECISIONS", 1)))
        self._pending_updates = []

        # Per-candidate band identity, resolved once (candidate order/identity
        # is fixed for this scheduler's lifetime -- see candidate_access_points).
        # Each candidate's RSSI/coverage-radius/noise below then use ITS OWN
        # band instead of always the legacy fixed-FR1 default, so a 6G
        # deployment profile (FR3/mmWave) is actually seen by handover
        # decisions, not just by packet execution.
        deployment = get_deployment_profile(
            deployment_profile_id or cfg.SIMULATION.get("deployment_profile")
        )
        self._candidate_band_names = [
            self._resolve_band_name(c, deployment) for c in candidates
        ]
        # Dual-connectivity coverage anchor per candidate (3GPP TS 37.340
        # EN-DC/NR-DC pattern): a device that's out of the main band's range
        # stays attached via this low-band anchor instead of counting as a
        # coverage outage -- see _eligible_candidate_indices and the
        # physical_outage check in select_server. Previously this scheduler
        # had no anchor awareness at all, so every out-of-main-band device
        # was scored as in outage even when the anchor (which every
        # deployment profile provides, with far more range than any main
        # band) was fully usable.
        self._candidate_anchor_band_names = [
            self._resolve_anchor_band_name(c, deployment) for c in candidates
        ]
        unique_bands = sorted(
            set(self._candidate_band_names) | set(self._candidate_anchor_band_names)
        )
        self._coverage_radius_by_band = {
            band: float(max_coverage_radius_m(band_name=band)) for band in unique_bands
        }
        self._air_ground_exponent_by_band = {
            band: float(air_ground_path_loss_exponent(band_name=band)) for band in unique_bands
        }
        self._candidate_rx_sensitivity_dbm = [
            float(get_band(band)["rx_sensitivity_dbm"])
            for band in self._candidate_band_names
        ]
        self._candidate_anchor_rx_sensitivity_dbm = [
            float(get_band(band)["rx_sensitivity_dbm"])
            for band in self._candidate_anchor_band_names
        ]

    def _eligible_candidate_indices(self, candidates, rssi_values, anchor_rssi_values=None):
        """Apply the access contract without changing the actor dimensions.

        A ground BS is available when it is operational and EITHER its main
        band OR its low-band coverage anchor reaches its own receiver
        sensitivity (3GPP EN-DC/NR-DC dual connectivity).  If at least one
        such BS exists, only ground BSs are eligible.  Otherwise operational
        UAV-BSs are eligible as coverage fallback.  Legacy deployments with
        no explicit access-only layer treat their ground MEC radios as the
        ground candidates.

        Anchor-aware ground eligibility matters here, not only in the
        physical_outage check: with only a handful of UAV-BS positioned at
        fixed points across a whole city, "no ground BS is in main-band
        range" pushes a device onto whichever UAV is least-bad -- which is
        very often itself far outside BOTH its main band's and its own
        anchor's range. A nearby ground BS's anchor (every ground site gets
        one, so one is almost always close) is the correct EN-DC fallback,
        not a distant UAV. Measured without this: 1,525 of 1,529 observed
        coverage-outage rows in a 6G FR3 smoke run were UAV-attached, only 4
        were ground-attached -- confirming the outage is a candidate-
        selection problem, not purely a channel-physics one.
        """

        has_access_layer = any(
            bool(getattr(server, "is_access_only", False))
            for server in candidates
        )
        ground_indices = []
        uav_indices = []
        for index, server in enumerate(candidates):
            tier = str(getattr(server, "tier", "") or "").upper()
            name = str(getattr(server, "name", "") or "")
            if tier == "UAV_BS" or name.startswith("UAV_BS_"):
                uav_indices.append(index)
            elif (
                bool(getattr(server, "is_access_only", False))
                or bool(getattr(server, "provides_radio_access", False))
                or (not has_access_layer and (tier == "MEC" or name.startswith("MEC_")))
            ):
                # A promoted MEC (provides_radio_access=True) reaches
                # candidate_access_points() alongside the access-only layer
                # (see candidate_access_points() above), so it must also be
                # eligible here as a ground candidate -- otherwise it would
                # be present in the actor's action space but never actually
                # selectable, silently reproducing the same exclusion bug
                # this whole fix is for.
                ground_indices.append(index)

        def _has_usable_coverage(index):
            if float(rssi_values[index]) >= self._candidate_rx_sensitivity_dbm[index]:
                return True
            if anchor_rssi_values is None:
                return False
            return (
                float(anchor_rssi_values[index])
                >= self._candidate_anchor_rx_sensitivity_dbm[index]
            )

        usable_ground = [
            index
            for index in ground_indices
            if server_is_operational(candidates[index]) and _has_usable_coverage(index)
        ]
        if usable_ground:
            return usable_ground

        operational_uavs = [
            index
            for index in uav_indices
            if server_is_operational(candidates[index])
        ]
        if operational_uavs:
            return operational_uavs

        # A legacy ground radio may be below sensitivity; retaining it as the
        # last operational candidate lets the simulator report a physical
        # outage instead of fabricating a different attachment.
        return [
            index
            for index in ground_indices
            if server_is_operational(candidates[index])
        ]

    # ------------------------------------------------------------ utils --
    def set_train_mode(self, mode: bool):
        self.train_mode = bool(mode)
        self.actor.train(self.train_mode)
        self.critic.train(self.train_mode)

    def save(self):
        try:
            save_checkpoint(os.path.join(self.model_dir, "handover_actor.pth"), self.actor.state_dict())
            save_checkpoint(os.path.join(self.model_dir, "handover_critic.pth"), self.critic.state_dict())
        except Exception as e:
            print(f"[HANDOVER][ERROR] Failed to save models: {e}")

    def load(self, path=None):
        path = path or self.model_dir
        try:
            self.actor.load_state_dict(load_checkpoint(os.path.join(path, "handover_actor.pth"), map_location="cpu"))
            self.critic.load_state_dict(load_checkpoint(os.path.join(path, "handover_critic.pth"), map_location="cpu"))
            self.set_train_mode(False)
        except Exception as e:
            print(f"[HANDOVER][ERROR] Failed to load models: {e}")

    # ------------------------------------------------------ link metrics --
    def _link_distance_and_pl_exponent(self, dev_xyz, srv, dev_is_airborne, band_name):
        """
        Real link distance (m) and the path-loss exponent to use for it.
        Any link where either end is airborne (task-generating UAV device,
        or a UAV-BS server) gets a true 3D distance and the air-to-ground
        exponent (for THIS candidate's own band); a plain ground device <->
        ground MEC link keeps the original 2D/ground-multipath model unchanged.
        """
        x, y, z = dev_xyz
        sx, sy = float(getattr(srv, "x", 0.0)), float(getattr(srv, "y", 0.0))
        srv_is_airborne = str(getattr(srv, "tier", "")).upper() == "UAV_BS"
        if dev_is_airborne or srv_is_airborne:
            sz = _entity_altitude_m(srv)
            dist = math.dist((x, y, z), (sx, sy, sz))
            exponent = self._air_ground_exponent_by_band[band_name]
            return dist, exponent
        return math.hypot(x - sx, y - sy), None

    # ------------------------------------------------------ state build --
    def _build_state(self, dev_xyz, cur_idx, candidates, speed_m_s, dev_is_airborne):
        feats = []
        speed_norm = min(abs(float(speed_m_s or 0.0)) / 25.0, 1.0)  # ~25 m/s ~ fast vehicle/drone

        rssi_values = np.empty(len(candidates), dtype=float)
        anchor_rssi_values = np.empty(len(candidates), dtype=float)
        distances = np.empty(len(candidates), dtype=float)
        cpu_utils, mem_utils, loads = [], [], []
        for i, s in enumerate(candidates):
            band_name = self._candidate_band_names[i]
            dist, pl_exp = self._link_distance_and_pl_exponent(
                dev_xyz, s, dev_is_airborne, band_name=band_name
            )
            rssi_values[i] = calculate_rssi_dbm(
                dist, include_fading=False, path_loss_exponent=pl_exp, band_name=band_name
            )
            # Same distance, same link geometry, the anchor band's own
            # physics (lower frequency -> much longer reach) -- this is
            # what lets a device stay "covered" via the anchor when the
            # main band alone would be out of range.
            anchor_band_name = self._candidate_anchor_band_names[i]
            _, anchor_pl_exp = self._link_distance_and_pl_exponent(
                dev_xyz, s, dev_is_airborne, band_name=anchor_band_name
            )
            anchor_rssi_values[i] = calculate_rssi_dbm(
                dist, include_fading=False, path_loss_exponent=anchor_pl_exp,
                band_name=anchor_band_name,
            )
            distances[i] = dist
            util = server_utilization(s)
            cpu = min(util["cpu"] / 100.0, 1.0)
            mem = min(util["memory"] / 100.0, 1.0)
            load = 0.6 * cpu + 0.4 * mem
            cpu_utils.append(util["cpu"])
            mem_utils.append(util["memory"])
            loads.append(load * 100.0)

        best_idx = int(np.argmax(rssi_values))
        current_rssi = float(rssi_values[cur_idx])
        best_rssi = float(rssi_values[best_idx])

        for i in range(len(candidates)):
            rssi = float(rssi_values[i])
            sig = 1.0 if rssi >= -70 else 0.8 if rssi >= -80 else 0.5 if rssi >= -90 else 0.2 if rssi >= -95 else 0.05
            cpu_frac = cpu_utils[i] / 100.0
            mem_frac = mem_utils[i] / 100.0
            band_name = self._candidate_band_names[i]
            radius = self._coverage_radius_by_band[band_name]
            dist_norm = min(distances[i] / max(radius, 1.0), 1.0)
            load_frac = loads[i] / 100.0
            feats.extend([sig, cpu_frac, mem_frac, dist_norm, load_frac])

        cache = {
            "rssi_values": rssi_values,
            "anchor_rssi_values": anchor_rssi_values,
            "cpu_utils": np.array(cpu_utils, dtype=float),
            "mem_utils": np.array(mem_utils, dtype=float),
            "distances": distances,
            "best_idx": best_idx,
            "current_rssi": current_rssi,
            "best_rssi": best_rssi,
            "improvement": best_rssi - current_rssi,
            "speed_norm": speed_norm,
        }
        return np.array(feats, dtype=np.float32), cache

    # ---------------------------------------------------- main decision --
    def select_server(self, device, servers, mobility_manager=None):
        """
        Decide (and execute) the wireless attachment (MEC or UAV-BS) for
        `device` this round. Returns the attached Server object. All
        decision-quality signals are written back onto `device` as plain
        attributes.
        """
        candidates = candidate_access_points(servers)
        if not candidates:
            return getattr(device, "attached_server", None)
        # Keep action space stable across the run; guard against config drift.
        if len(candidates) != self.num_candidates:
            self.num_candidates = len(candidates)

        pos = device.get_position() if hasattr(device, "get_position") else (device.x, device.y)
        dev_z = _entity_altitude_m(device)
        dev_xyz = (float(pos[0]), float(pos[1]), dev_z)
        dev_is_airborne = str(getattr(device, "mobility_kind", "")).lower() == "uav"
        speed_m_s = float(getattr(device, "movement_speed_m_s", getattr(device, "speed", 0.0)) or 0.0)

        cur_server = getattr(device, "attached_server", None)
        try:
            cur_idx = candidates.index(cur_server)
        except ValueError:
            # No current attachment yet (first round), or the previous
            # attachment is no longer a valid candidate: seed by RSSI
            # among the current (MEC + UAV-BS) candidate set directly,
            # rather than a 2D/ground-only RSSI ranking
            # helper, so a UAV-capable seed already accounts for altitude.
            seed_state, seed_cache = self._build_state(dev_xyz, 0, candidates, speed_m_s, dev_is_airborne)
            eligible_indices = self._eligible_candidate_indices(
                candidates, seed_cache["rssi_values"], seed_cache["anchor_rssi_values"]
            )
            if not eligible_indices:
                return None
            cur_idx = max(
                eligible_indices,
                key=lambda index: float(seed_cache["rssi_values"][index]),
            )
            cur_server = candidates[cur_idx]
            device.attached_server = cur_server
        # Attachment as it stood BEFORE this round's decision (for
        # decision-quality logging); may be overwritten below if a
        # handover executes.
        device.prev_attached_bs = getattr(cur_server, "name", "")

        state, cache = self._build_state(dev_xyz, cur_idx, candidates, speed_m_s, dev_is_airborne)
        eligible_indices = self._eligible_candidate_indices(
            candidates, cache["rssi_values"], cache["anchor_rssi_values"]
        )
        if not eligible_indices:
            return None
        best_idx = max(
            eligible_indices,
            key=lambda index: float(cache["rssi_values"][index]),
        )
        cache["best_idx"] = int(best_idx)
        cache["best_rssi"] = float(cache["rssi_values"][best_idx])
        cache["improvement"] = cache["best_rssi"] - cache["current_rssi"]
        current_is_eligible = cur_idx in eligible_indices

        # Mobility-aware handover hysteresis: faster devices get a wider
        # margin (avoid ping-pong while crossing cells quickly), slower
        # ones (pedestrians) a narrower one -- little ping-pong risk, so
        # snap to the objectively better cell sooner instead of waiting
        # for a large gap that a slow device may never even produce.
        threshold_scale = 0.6 + 0.8 * cache["speed_norm"]

        with torch.no_grad():
            probs_t = self.actor(torch.tensor(state))
            probs_np = probs_t.detach().cpu().numpy()

        # Mask the actor's choice, not its shape.  This preserves checkpoint
        # dimensions while enforcing the physical ground-first/UAV-fallback
        # access contract for RSSI guidance, exploration and exploitation.
        eligible_probs = np.zeros_like(probs_np, dtype=float)
        eligible_probs[eligible_indices] = probs_np[eligible_indices]
        probability_sum = float(np.sum(eligible_probs))
        if probability_sum > 0.0:
            eligible_probs /= probability_sum
        else:
            eligible_probs[eligible_indices] = 1.0 / len(eligible_indices)
        eligible_probs_t = torch.tensor(eligible_probs, dtype=probs_t.dtype)

        if len(eligible_indices) >= 2:
            permitted = eligible_probs[eligible_indices]
            top2 = np.partition(permitted, -2)[-2:]
            device.handover_q_margin = float(np.max(permitted) - np.min(top2))
        else:
            device.handover_q_margin = float(eligible_probs[eligible_indices[0]])

        # Task 80 parity fix: evaluation previously ran a *different* decision
        # rule here (argmax the actor, with an RSSI-override heuristic) than
        # training did (70% RSSI-guided / 9% NN-exploration / 21% NN-
        # exploitation). That meant a device's attached BS/MEC/UAV-BS -- and
        # therefore its reachable tiers, congestion and distance -- diverged
        # between the two phases regardless of how well the offloading policy
        # generalized, contaminating the train-vs-eval comparison. Both phases
        # now share this one action-selection path; only learning (the
        # `train_mode`-gated caching below, plus `update()`/`calculate_reward()`
        # in mobility/manager.py) is disabled during evaluation.
        if random.random() < 0.7:
            action = cache["best_idx"]
            decision_type = "RSSI_GUIDED"
        elif random.random() < 0.3:
            action = int(torch.multinomial(eligible_probs_t, 1).item())
            decision_type = "NN_EXPLORATION"
        else:
            action = int(torch.argmax(eligible_probs_t).item())
            decision_type = "NN_EXPLOITATION"

        should_handover = False
        if action != cur_idx:
            target_rssi = float(cache["rssi_values"][action])
            if not current_is_eligible:
                # The current attachment belongs to the wrong access layer:
                # ground coverage appeared, or disappeared and UAV fallback is
                # now required.  Move without applying ordinary hysteresis.
                should_handover = True
            elif decision_type.startswith("RSSI"):
                should_handover = (cache["improvement"] > 5 * threshold_scale) or (cache["current_rssi"] < -85)
            else:
                should_handover = (target_rssi > -90) and (target_rssi > cache["current_rssi"] + 2 * threshold_scale)

        dev_id = getattr(device, "id", id(device))
        prev_bs_name = self.prev_bs_for_log.get(dev_id)

        handover_executed = 0
        handover_failure_flag = 0
        handover_interruption_ms = 0.0
        if should_handover:
            handover_interruption_ms = float(getattr(cfg, "HANDOVER_INTERRUPTION_MS", 0.0) or 0.0)
            fail_prob = float(getattr(cfg, "HANDOVER_FAILURE_PROB", 0.0) or 0.0)
            handover_failure_flag = int(random.random() < fail_prob)
            if not handover_failure_flag:
                target_server = candidates[action]
                device.attached_server = target_server
                device.last_handover_ms = float(
                    getattr(device, "ho_latency", None)
                    or getattr(cfg, "MOBILITY_CONFIG", {}).get("handover_latency_ms", 5.0)
                )
                handover_executed = 1
                self.prev_prev_bs[dev_id] = prev_bs_name
                self.prev_bs_for_log[dev_id] = target_server.name
                cur_idx = action
            else:
                device.last_handover_ms = 0.0
        else:
            device.last_handover_ms = 0.0
            if prev_bs_name is None:
                self.prev_bs_for_log[dev_id] = getattr(cur_server, "name", "")

        selected_idx = int(action)
        selected_server = candidates[selected_idx]
        prev_rssi = float(cache["rssi_values"][cur_idx])
        selected_rssi = float(cache["rssi_values"][selected_idx])
        best_idx = int(cache["best_idx"])
        effective_idx = int(action) if handover_executed else int(cur_idx)
        chosen_server = candidates[effective_idx]
        chosen_rssi = float(cache["rssi_values"][effective_idx])
        oracle_idx = best_idx
        oracle_server = candidates[oracle_idx]
        oracle_rssi = cache["best_rssi"]

        # ---- RF metrics at the chosen link (real channel model) ----
        chosen_band = self._candidate_band_names[effective_idx]
        noise_dbm = calculate_noise_power_dbm(band_name=chosen_band)
        interference_dbm = calculate_interference_power_dbm(
            pos, chosen_server, candidates, band_name=chosen_band
        )
        snr_db = calculate_snr_db(chosen_rssi, noise_dbm=noise_dbm, band_name=chosen_band)
        sinr_db = calculate_sinr_db(
            chosen_rssi, interference_dbm, noise_dbm=noise_dbm, band_name=chosen_band
        )

        # ---- dwell-time state machine (predicted-with-error vs. true) ----
        # Relative-speed-aware: a moving serving node (UAV-BS) changes the
        # distance on its own, even for a stationary device -- so the rate
        # driving dwell time is device speed *and* the serving node's own
        # patrol speed, not device speed alone (which is all a fixed MEC
        # link would ever need).
        dt_ms = float(getattr(cfg, "MOBILITY_CONFIG", {}).get("time_step_ms", 1000.0) or 1000.0)
        mec_radius_m = max(1.0, float(cfg.SERVER_CONFIG.get("MEC", {}).get("distance", 1000.0) or 1000.0))
        dist_to_chosen = float(cache["distances"][cur_idx])
        bs_speed_m_s = _entity_speed_m_s(chosen_server)
        relative_speed_m_s = max(abs(speed_m_s) + bs_speed_m_s, 1e-3)
        true_dwell_ms = (
            remaining_coverage_time_s(
                remaining_distance_m=mec_radius_m - dist_to_chosen,
                relative_speed_m_s=relative_speed_m_s,
            )
            * 1000.0
        )
        error_std = float(getattr(cfg, "DWELL_TIME_ERROR_STD", 0.0) or 0.0)
        predicted_dwell_ms = max(0.0, true_dwell_ms * (1.0 + random.gauss(0.0, error_std)))

        prev_remaining = float(self.remaining_dwell_time_ms.get(dev_id, predicted_dwell_ms))
        if handover_executed or dev_id not in self.remaining_dwell_time_ms:
            remaining_dwell_ms = predicted_dwell_ms
        else:
            remaining_dwell_ms = max(0.0, prev_remaining - dt_ms)

        self.true_dwell_time_ms[dev_id] = true_dwell_ms
        self.predicted_dwell_time_ms[dev_id] = predicted_dwell_ms
        self.remaining_dwell_time_ms[dev_id] = remaining_dwell_ms
        bs_residence_time_ms = max(0.0, predicted_dwell_ms - remaining_dwell_ms)

        # ---- coverage outage: physical (below sensitivity) or transient ----
        # Dual connectivity (3GPP EN-DC/NR-DC): the device only truly loses
        # coverage when BOTH the main band AND its low-band anchor are below
        # sensitivity. A main-band-only shortfall (e.g. a 6G FR3/mmWave/
        # subTHz device temporarily out of the shorter main-band range)
        # keeps the device attached via the anchor instead of counting as an
        # outage -- matching how real dual-connectivity deployments behave,
        # and how the packet-execution layer (core/network_runtime.py's
        # DynamicAccessRouteBuilder) already falls back to the anchor band
        # for the actual transmission.
        outage_prob = float(getattr(cfg, "OUTAGE_PROBABILITY", 0.0) or 0.0)
        chosen_anchor_rssi = float(cache["anchor_rssi_values"][effective_idx])
        main_band_out_of_range = (
            chosen_rssi < self._candidate_rx_sensitivity_dbm[effective_idx]
        )
        anchor_out_of_range = (
            chosen_anchor_rssi < self._candidate_anchor_rx_sensitivity_dbm[effective_idx]
        )
        physical_outage = main_band_out_of_range and anchor_out_of_range
        served_via_anchor = int(main_band_out_of_range and not anchor_out_of_range)
        transient_outage = random.random() < outage_prob
        coverage_outage_flag = int(physical_outage or transient_outage)
        coverage_outage_duration_ms = dt_ms if coverage_outage_flag else 0.0

        # ---- ping-pong detection ----
        prev_prev = self.prev_prev_bs.get(dev_id)
        prev_bs_now = self.prev_bs_for_log.get(dev_id)
        pingpong_flag = int(
            prev_prev is not None and prev_prev == chosen_server.name
            and prev_bs_now is not None and prev_bs_now != prev_prev
            and handover_executed == 1
        )

        # ---- trigger reason (why did/didn't we hand over) ----
        dwell_threshold = float(getattr(cfg, "DWELL_EXPIRE_THRESHOLD_MS", 1.0) or 1.0)
        if coverage_outage_flag:
            trigger_reason = "coverage_outage"
        elif handover_failure_flag:
            trigger_reason = "resource_overload"
        elif chosen_rssi < -90:
            trigger_reason = "low_rssi"
        elif oracle_server.name != chosen_server.name and oracle_rssi > chosen_rssi + 2.0:
            trigger_reason = "better_bs_available"
        elif remaining_dwell_ms <= dwell_threshold:
            trigger_reason = "dwell_expired"
        else:
            trigger_reason = "no_handover"

        # ---- write everything back onto the device for the simulator ----
        device.chosen_rssi_dbm = chosen_rssi
        device.handover_prev_rssi_dbm = prev_rssi
        device.handover_target_bs = selected_server.name
        device.handover_target_rssi_dbm = selected_rssi
        device.handover_target_rssi_gain_db = selected_rssi - prev_rssi
        device.handover_best_available_bs = candidates[best_idx].name
        device.handover_best_available_rssi_dbm = float(cache["rssi_values"][best_idx])
        device.oracle_bs = oracle_server.name
        device.oracle_rssi_dbm = oracle_rssi
        device.rssi_regret_db = max(0.0, oracle_rssi - chosen_rssi)
        device.snr_db = snr_db
        device.sinr_db = sinr_db
        device.noise_dbm = noise_dbm
        device.interference_dbm = interference_dbm
        device.handover_executed = handover_executed
        device.handover_failure_flag = handover_failure_flag
        device.handover_interruption_ms = handover_interruption_ms if not handover_failure_flag else handover_interruption_ms
        device.handover_trigger_reason = trigger_reason
        device.pingpong_flag = pingpong_flag
        device.predicted_dwell_time_ms = predicted_dwell_ms
        device.remaining_dwell_time_ms = remaining_dwell_ms
        device.true_dwell_time_ms = true_dwell_ms
        device.bs_residence_time_ms = bs_residence_time_ms
        device.handover_stability_time_ms = remaining_dwell_ms
        device.coverage_outage_flag = coverage_outage_flag
        device.coverage_outage_duration_ms = coverage_outage_duration_ms
        # Diagnostic only (not consumed by delay/energy/cost formulas): 1 when
        # this round's main band was out of range but the low-band anchor
        # kept the device attached, so a coverage-outage reduction is
        # actually attributable to dual connectivity rather than main-band
        # range improving.
        device.served_via_anchor = served_via_anchor
        device.chosen_cpu_util = float(cache["cpu_utils"][effective_idx])
        device.chosen_mem_util = float(cache["mem_utils"][effective_idx])
        device.oracle_cpu_util = float(cache["cpu_utils"][oracle_idx])
        device.oracle_mem_util = float(cache["mem_utils"][oracle_idx])
        device.handover_decision_type = decision_type

        if self.train_mode:
            self._last_state[dev_id] = state
            self._last_action[dev_id] = effective_idx
            self.action_dist_log.append({"round": self._current_step, "action": int(effective_idx), "decision_type": decision_type})

        self._current_step += 1
        return chosen_server

    # ------------------------------------------------------------ reward --
    def calculate_reward(self, device):
        """Reward for the handover decision just made for `device`."""
        bs_signal = float(getattr(device, "chosen_rssi_dbm", -100.0))
        bs_util = float(getattr(device, "chosen_cpu_util", 0.0))
        bs_mem = float(getattr(device, "chosen_mem_util", 0.0))
        handover_executed = int(getattr(device, "handover_executed", 0))
        speed_norm = min(abs(float(getattr(device, "movement_speed_m_s", 0.0) or 0.0)) / 25.0, 1.0)
        is_oracle = str(getattr(device, "oracle_bs", "")) == str(getattr(getattr(device, "attached_server", None), "name", ""))

        if bs_signal >= -75:
            sig_reward = 2.0
        elif bs_signal >= -85:
            sig_reward = 1.0
        elif bs_signal >= -90:
            sig_reward = 0.0
        elif bs_signal >= -95:
            sig_reward = -1.0
        else:
            sig_reward = -2.0

        load_penalty = 0.05 * (min(bs_util, 100.0) + min(bs_mem, 100.0)) / 200.0
        ho_pen = 0.1 if handover_executed else 0.0
        speed_pen = 0.05 * speed_norm if handover_executed else 0.0
        rssi_bonus = 0.5 if is_oracle else 0.0

        reward = sig_reward - load_penalty - ho_pen - speed_pen + rssi_bonus
        return float(np.clip(reward, -3.0, 3.0))

    def update(self, reward, device, round_id=-1):
        if not self.train_mode:
            return
        dev_id = getattr(device, "id", id(device))
        state = self._last_state.pop(dev_id, None)
        action = self._last_action.pop(dev_id, None)
        if state is None or action is None:
            return

        self._pending_updates.append((state, action, float(reward), dev_id, round_id))
        self.reward_log.append({"round": round_id, "device_id": dev_id, "reward": float(reward)})
        if len(self._pending_updates) < self.learn_every:
            return
        self._flush_pending_updates()

    def _flush_pending_updates(self):
        batch = self._pending_updates
        self._pending_updates = []
        if not batch:
            return

        states_t = torch.as_tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
        actions = [int(b[1]) for b in batch]
        rewards_t = torch.as_tensor([b[2] for b in batch], dtype=torch.float32)
        round_id = batch[-1][4]

        values = self.critic(states_t).squeeze(-1)
        advantages = rewards_t - values.detach()

        probs = self.actor(states_t)
        row_idx = torch.arange(len(batch))
        log_probs = torch.log(probs[row_idx, actions] + 1e-10)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()

        actor_loss = -(log_probs * advantages).mean() - 0.01 * entropy
        critic_loss = nn.MSELoss()(values, rewards_t)

        self.optim_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.optim_actor.step()

        self.optim_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.optim_critic.step()

        self.loss_log.append({
            "round": round_id,
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "advantage": float(advantages.mean().item()),
        })

    # ------------------------------------------------------------ logging --
    def save_logs(self, output_dir=None):
        if self.train_mode:
            self._flush_pending_updates()
        out_dir = str(output_dir) if output_dir is not None else self.out_dir
        if not out_dir:
            return
        try:
            TrainingReporter(out_dir, delimiter=";").save_handover_training(
                reward_log=self.reward_log,
                loss_log=self.loss_log,
                action_dist_log=self.action_dist_log,
                include_plots=False,
            )
        except Exception as e:
            print(f"[HANDOVER][ERROR] Failed to save logs: {e}")
