"""
mobility/vehicle_road.py

Vehicle mobility: Manhattan-grid movement. Vehicles travel in a straight
line along one road at a time (horizontal or vertical -- never diagonal),
at higher speed than pedestrians.

The road grid's outermost roads sit exactly on the city boundary (see
build_road_grid), so the perimeter itself is a normal, drivable road and
every arrival point -- corner, T-junction, or interior crossing -- is a
real node with at least two ways to continue. At each node the vehicle
looks at all four compass directions, keeps only the ones that actually
lead somewhere (i.e. aren't blocked by the city edge it's already on),
drops an immediate U-turn unless that's the only option left (a true
cul-de-sac), and picks uniformly among what remains. This keeps vehicles
exploring the whole grid instead of oscillating back and forth along the
perimeter between two corners.

Uses the same shared road grid (mobility/road_network.py) that the live
GUI draws, so vehicles are always visually on a drawn road.
"""

import math
import random

from mobility.road_network import build_road_grid, nearest


class VehicleRoadMobility:
    def __init__(self, area, speed_range, pause_range, n_horizontal=None, n_vertical=None):
        self.area = area
        self.speed_range = speed_range
        self.pause_range = pause_range
        self.h_roads, self.v_roads = build_road_grid(area, n_horizontal, n_vertical)

        # Throwaway position/state -- the simulator overwrites .pos with the
        # device's real spawn point right after construction, then calls
        # _start_move(), which snaps onto the road grid and picks a real leg.
        self.pos = (random.uniform(area["xmin"], area["xmax"]), random.uniform(area["ymin"], area["ymax"]))
        self.axis = random.choice(("h", "v"))
        self.direction = random.choice((1, -1))
        self.state = "move"
        self.pause_time = 0.0
        self.speed = random.uniform(*speed_range)
        self.target = self.pos

    def on_spawn(self, x, y):
        # Nothing extra needed -- _start_move() (called right after) snaps
        # the position onto the nearest road for us.
        pass

    # -------- road geometry --------
    def _snap_to_grid(self):
        x, y = self.pos
        sx = nearest(x, self.v_roads) if self.v_roads else x
        sy = nearest(y, self.h_roads) if self.h_roads else y
        self.pos = (sx, sy)

    def _leg_target(self, axis, direction):
        """
        Next point reached by heading `direction` along `axis` from the
        current position: the next crossing road if there is one, otherwise
        the city boundary in that direction (which is itself always a road,
        see module docstring).
        """
        x, y = self.pos
        if axis == "h":
            ahead = [vx for vx in self.v_roads if (vx - x) * direction > 1e-6]
            nxt = (min(ahead) if direction > 0 else max(ahead)) if ahead \
                else (self.area["xmax"] if direction > 0 else self.area["xmin"])
            return (nxt, y)
        else:
            ahead = [hy for hy in self.h_roads if (hy - y) * direction > 1e-6]
            nxt = (min(ahead) if direction > 0 else max(ahead)) if ahead \
                else (self.area["ymax"] if direction > 0 else self.area["ymin"])
            return (x, nxt)

    def _valid_directions(self, exclude=None):
        """
        All (axis, direction, target) moves from the current position that
        actually go somewhere (nonzero distance) -- i.e. exclude compass
        directions that are already blocked by the city edge the vehicle
        is sitting on. `exclude` is an (axis, direction) pair (typically
        the reverse of how the vehicle just arrived) to leave out unless
        it's the only option remaining.
        """
        options = []
        for axis in ("h", "v"):
            for direction in (1, -1):
                target = self._leg_target(axis, direction)
                dx = target[0] - self.pos[0]
                dy = target[1] - self.pos[1]
                if math.hypot(dx, dy) > 1e-6:
                    options.append((axis, direction, target))
        if exclude is not None:
            narrowed = [o for o in options if (o[0], o[1]) != exclude]
            if narrowed:
                return narrowed
        return options

    def _choose_next_leg(self):
        # Called on arrival at any node (corner, T-junction, or interior
        # crossing) and when a pause ends: pick uniformly among the
        # directions that lead somewhere, avoiding an immediate U-turn
        # unless it's forced, and (re)enter "move" -- without this, a
        # vehicle coming out of "pause" would keep re-rolling a target
        # every tick but never actually resume moving.
        reverse_of = (self.axis, -self.direction)
        self.axis, self.direction, self.target = random.choice(self._valid_directions(exclude=reverse_of))
        self.speed = random.uniform(*self.speed_range)
        self.state = "move"

    def _start_move(self):
        self._snap_to_grid()
        self.state = "move"
        self.axis, self.direction, self.target = random.choice(self._valid_directions())
        self.speed = random.uniform(*self.speed_range)

    # -------- stepping --------
    def next_position(self, dt_ms):
        dt = dt_ms / 1000.0

        if self.state == "pause":
            self.pause_time -= dt
            if self.pause_time <= 0:
                self._choose_next_leg()
            return self.pos

        x, y = self.pos
        tx, ty = self.target
        dx, dy = tx - x, ty - y
        remaining = math.hypot(dx, dy)
        step = self.speed * dt

        if remaining <= 1e-9 or step >= remaining:
            self.pos = self.target
            max_pause = float(self.pause_range[1]) if self.pause_range else 0.0
            if max_pause > 0.0 and random.random() < 0.25:
                self.state = "pause"
                self.pause_time = random.uniform(*self.pause_range)
            else:
                self._choose_next_leg()
        else:
            ux, uy = dx / remaining, dy / remaining
            self.pos = (x + ux * step, y + uy * step)

        return self.pos
