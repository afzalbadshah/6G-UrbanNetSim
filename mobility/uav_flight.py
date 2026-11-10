"""
mobility/uav_flight.py

UAV mobility: free random-waypoint movement in x/y (any direction, unlike
the axis-constrained pedestrian/vehicle models) plus an independent
altitude (z) random-walk between legs -- UAVs can change height while
ground-based devices can't.
"""

import random

from mobility.random_waypoint import RandomWaypoint


class UAVFlight(RandomWaypoint):
    # UAVs pick random waypoints only within this fraction of the full area,
    # shrunk toward its center -- keeps every UAV's roam zone overlapping
    # so independent random-waypoint UAVs still end up closer together
    # instead of spreading across the whole city.
    AREA_SHRINK_FRAC = 0.45

    def __init__(self, area, speed_range, pause_range,
                 altitude_range=(50.0, 150.0), vertical_speed_range=(0.5, 3.0)):
        self.altitude_range = tuple(altitude_range)
        self.vertical_speed_range = tuple(vertical_speed_range)
        self.z = random.uniform(*self.altitude_range)
        self.z_target = random.uniform(*self.altitude_range)
        self.z_speed = random.uniform(*self.vertical_speed_range)
        super().__init__(self._shrink_area(area), speed_range, pause_range)

    @classmethod
    def _shrink_area(cls, area):
        cx = (float(area["xmin"]) + float(area["xmax"])) / 2.0
        cy = (float(area["ymin"]) + float(area["ymax"])) / 2.0
        hw = (float(area["xmax"]) - float(area["xmin"])) * cls.AREA_SHRINK_FRAC / 2.0
        hh = (float(area["ymax"]) - float(area["ymin"])) * cls.AREA_SHRINK_FRAC / 2.0
        return {"xmin": cx - hw, "xmax": cx + hw, "ymin": cy - hh, "ymax": cy + hh}

    def _start_move(self):
        super()._start_move()
        # Pick a new altitude target each time a new xy leg begins.
        self.z_target = random.uniform(*self.altitude_range)
        self.z_speed = random.uniform(*self.vertical_speed_range)

    def next_position(self, dt_ms):
        pos = super().next_position(dt_ms)
        dt = dt_ms / 1000.0
        dz = self.z_target - self.z
        step = self.z_speed * dt
        if abs(dz) <= step:
            self.z = self.z_target
        else:
            self.z += step if dz > 0 else -step
        self.z = min(max(self.z, self.altitude_range[0]), self.altitude_range[1])
        return pos
