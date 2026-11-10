"""
RegionalMECSimPy — Educational Release

Author: Dr. Afzal Badshah
Department of Software Engineering, University of Sargodha
Website: https://afzalbadshah.com/index.php/afzal-badshah/
LinkedIn: https://www.linkedin.com/in/afzal-badshah-phd-305b03164/
Discussion Forum: resp.afzalbadshah.com

Notes:
- Safe defaults for mobility config to avoid KeyError.
- Works with 1D or 2D positions and multiple mobility model APIs.
- Returns (server, distance_m) from pick_best_bs; MobilityManager converts
  attachment changes into configured handover latency.
"""

import math
from config import MOBILITY_CONFIG, SERVER_CONFIG
from config.radio import get_link_profile


class MobileEntity:
    """
    Represents a mobile node (device/vehicle).
    Position can be:
      - 1D float (legacy)
      - 2D (x, y) list/tuple
    """

    def __init__(self, entity_id, init_pos, speed):
        self.id = entity_id
        self.position = init_pos  # 1D float or 2D [x, y]
        self.speed = float(speed)
        self.device_type = "generic"
        self.device_profile = {}
        self.service_class = ""
        self.mobility_profile = "unknown"
        self.mobility_kind = "stationary"
        self.movement_speed_m_s = float(speed)
        self.handover_sensitivity = 0.0
        self.dwell_weight = 1.0
        # No onboard compute until a profile marks this device capable --
        # apply_device_profile() then grants the universal Onboard capacities
        # (config.SERVER_CONFIG["Onboard"]) to capable devices only.
        self.local_compute_capable = False
        self.local_cpu_capacity = 0.0
        self.local_memory_capacity = 0.0
        self.local_power_w = 0.0
        self.battery_capacity_j = 0.0
        self.battery_remaining_j = 0.0
        self.tx_power_w = 0.0
        self.rx_power_w = 0.0
        self.idle_power_w = 0.0

        # Serving / handover state
        self.attached_server = None
        self.last_handover_ms = 0.0

        # Safe defaults (avoid KeyError if config keys are missing)
        self.ho_latency = float(MOBILITY_CONFIG.get("handover_latency_ms", 5.0))
        self.ho_threshold_db = float(MOBILITY_CONFIG.get("handover_threshold_db", 3.0))

        # Mobility controller (e.g., RandomWaypoint). Set by simulator.
        self.mobility = None

        # Ensure x, y are always present for convenience
        if isinstance(self.position, (list, tuple)) and len(self.position) >= 2:
            self.x = float(self.position[0])
            self.y = float(self.position[1])
        else:
            # legacy 1D → map to (x, 0)
            self.x = float(self.position)
            self.y = 0.0

    # -------- convenience --------

    def set_mobility(self, model):
        """Attach a mobility model (RandomWaypoint, etc.)."""
        self.mobility = model

    def apply_device_profile(self, device_type, profile):
        profile = dict(profile or {})
        self.device_type = str(device_type or "generic")
        self.device_profile = profile
        self.service_class = str(profile.get("service_class", "") or "")
        mobility = profile.get("mobility", {}) or {}
        local = profile.get("local_compute", {}) or {}
        battery = profile.get("battery", {}) or {}
        radio = profile.get("radio_energy", {}) or {}
        self.mobility_profile = str(mobility.get("profile", profile.get("mobility_profile", "fixed")) or "fixed")
        self.mobility_kind = str(mobility.get("kind", profile.get("mobility_kind", "stationary")) or "stationary")
        speed_range = mobility.get("speed_m_s", (self.speed, self.speed))
        try:
            self.movement_speed_m_s = float(sum(speed_range) / len(speed_range))
        except Exception:
            self.movement_speed_m_s = float(self.speed)
        self.speed = self.movement_speed_m_s
        self.handover_sensitivity = float(mobility.get("handover_sensitivity", 0.0) or 0.0)
        self.dwell_weight = float(mobility.get("dwell_weight", 1.0) or 1.0)
        self.altitude_range_m = tuple(mobility.get("altitude_range_m", (0.0, 0.0)) or (0.0, 0.0))
        # On-device capacity/power is per-device-type (profile's own
        # local_compute.cpu/memory/power_w, see devices_config.py), falling
        # back to the universal SERVER_CONFIG["Onboard"] default only if a
        # capable profile omits its own spec. Incapable devices (MCU-class
        # sensors) keep zero local capacity so they never expose local
        # execution space.
        self.local_compute_capable = bool(local.get("capable", profile.get("local_compute_capable", False)))
        if self.local_compute_capable:
            onboard = SERVER_CONFIG.get("Onboard", {}) or {}
            self.local_cpu_capacity = float(local.get("cpu", onboard.get("cpu", 0.0)) or 0.0)
            self.local_memory_capacity = float(local.get("memory", onboard.get("memory", 0.0)) or 0.0)
            self.local_power_w = float(local.get("power_w", onboard.get("power_w", 0.0)) or 0.0)
        else:
            self.local_cpu_capacity = 0.0
            self.local_memory_capacity = 0.0
            self.local_power_w = 0.0
        self.battery_capacity_j = float(battery.get("capacity_j", 0.0) or 0.0)
        self.battery_remaining_j = float(battery.get("initial_j", self.battery_capacity_j) or 0.0)
        self.tx_power_w = float(radio.get("tx_power_w", 0.0) or 0.0)
        self.rx_power_w = float(radio.get("rx_power_w", 0.0) or 0.0)
        self.idle_power_w = float(radio.get("idle_power_w", 0.0) or 0.0)
        # Continuous flight/propulsion power draw (W), independent of
        # communication/compute -- only "uav" mobility-kind profiles set
        # this (config/devices_config.py); everything else (pedestrian,
        # vehicle, stationary) stays 0.0, i.e. no propulsion cost.
        self.propulsion_power_w = float(profile.get("propulsion_power_w", 0.0) or 0.0)

    def get_position(self):
        """Always return a 2D tuple (x, y)."""
        return float(self.x), float(self.y)

    # -------- movement & radio --------

    def move(self, dt_ms):
        """
        Advance position by dt_ms.
        - If 2D and a mobility model exists:
            * Prefer model.next_position(dt_ms)  -> (x, y)
            * Else fallback to model.step(dt_s)  -> (x, y)
        - If 1D (legacy) with no model: x += speed * dt_s
        """
        dt_s = float(dt_ms) / 1000.0

        # MobilityManager owns the once-per-round propulsion debit before it
        # calls this movement fallback. Do not charge propulsion again here.

        if self.mobility is not None:
            # Try both possible APIs
            if hasattr(self.mobility, "next_position"):
                nx, ny = self.mobility.next_position(dt_ms)
            else:
                # Expect step(seconds) → (x, y)
                nx, ny = self.mobility.step(dt_s)
            self.x, self.y = float(nx), float(ny)
            self.position = [self.x, self.y]
            return self.position

        # No mobility model: keep legacy behavior
        if isinstance(self.position, (int, float, float)):
            # 1D drift
            self.x += self.speed * dt_s
            self.position = self.x
            # y stays 0
        else:
            # 2D but no model → stay put
            pass

        return self.position

    def signal_strength(self, server):
        """
        RSSI (dBm) using the log-distance path-loss + shadowing/fading
        channel model (metrics/handover_channel.py::calculate_rssi_dbm), the same
        model used by mobility/handover_scheduler.py for handover decisions.
        Uses Euclidean distance to server.x/server.y when available,
        falls back to |x - server.location| in 1D.
        """
        if isinstance(self.position, (list, tuple)):
            sx = float(getattr(server, "x", getattr(server, "location", 0.0)))
            sy = float(getattr(server, "y", 0.0))
            dx = float(self.x) - sx
            dy = float(self.y) - sy
            dist = math.hypot(dx, dy)
        else:
            dist = abs(float(self.x) - float(getattr(server, "location", 0.0)))

        profile_id = getattr(server, "profile_id", None)
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise ValueError(
                f"serving radio point {getattr(server, 'name', '<unnamed>')!r} "
                "has no explicit profile_id"
            )
        band_name = str(get_link_profile(profile_id)["band"])
        from metrics.handover_channel import calculate_rssi_dbm
        return calculate_rssi_dbm(
            max(1.0, dist),
            band_name=band_name,
            include_fading=True,
        )

    # -------- serving cell selection / HO --------
    def pick_best_bs(self, servers):
        # Nearest-distance fallback used only as a safety net (see
        # mobility/manager.py) when the RL handover scheduler's own choice
        # is missing/invalid. Callers are expected to have already filtered
        # `servers` down to whichever tiers count as valid attachment
        # points (MEC + UAV_BS) -- this no longer re-filters by name, or a
        # pre-filtered UAV_BS candidate list would get silently discarded.
        candidates = list(servers or [])
        if not candidates:
            # No candidates available → keep current attachment (no HO)
            return getattr(self, "attached_server", None), 0.0

        x, y = self.get_position()
        best = min(
            (s for s in candidates if hasattr(s, "x") and hasattr(s, "y")),
            key=lambda s: ((s.x - x)**2 + (s.y - y)**2),
        )
        dx, dy = (best.x - x), (best.y - y)
        return best, (dx*dx + dy*dy) ** 0.5



    

    def handover(self, new_server):
        """
        Perform the handover: switch attachment and remember the latency.
        """
        self.attached_server = new_server
        self.last_handover_ms = self.ho_latency
        return new_server
