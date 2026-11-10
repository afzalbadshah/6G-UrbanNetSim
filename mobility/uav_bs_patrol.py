"""
mobility/uav_bs_patrol.py

UAV-BS (aerial base station) mobility: each UAV-BS flies a rectangular
patrol loop around its own coverage cell (see
mobility.road_network.distribute_coverage_cells -- the same cell layout
used for MEC placement, so ground and aerial base stations divide the
city the same coverage-maximizing way for any count).

On edges that border a *neighboring* UAV-BS's cell, the loop is pushed
outward past the strict cell boundary by an overlap margin, so two
adjacent UAV-BS's patrol paths genuinely overlap near their shared
border -- a device sitting there can reach either one, rather than the
two zones being strictly disjoint. On edges that are the true city
border, the loop keeps a small inward inset instead (to stay clear of
the ring road) so overall city coverage isn't reduced.
"""

import math


class UAVBSPatrol:
    BORDER_INSET_M = 80.0   # inward nudge off the true city border (stay off the ring road)
    OVERLAP_FRAC = 0.15     # how far the loop pushes past a shared cell edge, as a fraction
                            # of that edge's cell dimension, so neighboring UAV-BS overlap

    def __init__(self, area, bounds, speed_m_s=14.0,
                 altitude_range=(220.0, 380.0), start_waypoint_index=0):
        self.area = area
        self.bounds = bounds
        self.speed = float(speed_m_s)
        self.altitude_range = tuple(altitude_range)
        self.waypoints = self._loop_from_bounds(area, bounds)
        self._idx = int(start_waypoint_index) % len(self.waypoints)
        self.pos = self.waypoints[self._idx]
        self.z = sum(self.altitude_range) / 2.0
        self.heading = 0.0

    @classmethod
    def _loop_from_bounds(cls, area, bounds):
        xmin, xmax = float(area["xmin"]), float(area["xmax"])
        ymin, ymax = float(area["ymin"]), float(area["ymax"])
        x0, x1, y0, y1 = bounds
        w, h = x1 - x0, y1 - y0
        eps = 1.0  # meters -- tolerance for "is this edge the true city border"

        def adjust(value, city_edge, cell_span, is_lo):
            if abs(value - city_edge) < eps:
                return value + cls.BORDER_INSET_M if is_lo else value - cls.BORDER_INSET_M
            overlap = cls.OVERLAP_FRAC * cell_span
            return value - overlap if is_lo else value + overlap

        nx0 = adjust(x0, xmin, w, True)
        nx1 = adjust(x1, xmax, w, False)
        ny0 = adjust(y0, ymin, h, True)
        ny1 = adjust(y1, ymax, h, False)
        if nx1 <= nx0:
            mid = (nx0 + nx1) / 2.0
            nx0 = nx1 = mid
        if ny1 <= ny0:
            mid = (ny0 + ny1) / 2.0
            ny0 = ny1 = mid
        return [(nx0, ny0), (nx0, ny1), (nx1, ny1), (nx1, ny0)]

    def on_spawn(self, x, y):
        # The patrol route is fixed by its assigned cell, not the spawn
        # point -- nothing to re-anchor here.
        pass

    def _start_move(self):
        # Kept for interface parity with the RandomWaypoint-family models;
        # the route is fully parametric, nothing to (re)initialize.
        pass

    def next_position(self, dt_ms):
        dt = dt_ms / 1000.0
        step = self.speed * dt
        x, y = self.pos
        n = len(self.waypoints)
        for _ in range(n * 2 + 1):  # safety cap: at most a couple of full loops per call
            if step <= 0:
                break
            tx, ty = self.waypoints[(self._idx + 1) % n]
            dx, dy = tx - x, ty - y
            remaining = math.hypot(dx, dy)
            if remaining <= 1e-9:
                self._idx = (self._idx + 1) % n
                x, y = tx, ty
                continue
            if step >= remaining:
                x, y = tx, ty
                self._idx = (self._idx + 1) % n
                self.heading = math.degrees(math.atan2(dy, dx))
                step -= remaining
            else:
                ux, uy = dx / remaining, dy / remaining
                x, y = x + ux * step, y + uy * step
                self.heading = math.degrees(math.atan2(dy, dx))
                step = 0.0
        self.pos = (x, y)
        return self.pos
