"""
RegionalMECSimPy — Educational Release

Author: Dr. Afzal Badshah
Department of Software Engineering, University of Sargodha
Website: https://afzalbadshah.com/index.php/afzal-badshah/
LinkedIn: https://www.linkedin.com/in/afzal-badshah-phd-305b03164/
Discussion Forum: resp.afzalbadshah.com

Notes:
- Functionality preserved; comments kept simple for student readability.
- Implements the Random Waypoint mobility model for 2D movement.
"""

import random
import math


class RandomWaypoint:
    """
    Moves a node in 2D space between random points, pausing between moves.

    Parameters:
        area        : dict with xmin, xmax, ymin, ymax (movement bounds)
        speed_range : (min_speed, max_speed) in meters/second
        pause_range : (min_pause, max_pause) in seconds

    Students:
    - Adjust area, speed_range, or pause_range to change mobility behavior.
    - This is just one model; you can implement your own movement logic here.
    """

    def __init__(self, area, speed_range, pause_range):
        self.area = area
        self.speed_range = speed_range
        self.pause_range = pause_range

        # Start at a random point, but begin in MOVE state immediately
        self.pos = self._random_point()
        self.target = self._random_point()

        # Begin moving right away
        self.state = "move"
        self.pause_time = 0.0
        self.speed = random.uniform(*self.speed_range)

        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]
        dist = math.hypot(dx, dy) or 1.0
        self.vel_x = self.speed * dx / dist
        self.vel_y = self.speed * dy / dist


    def _random_point(self):
        """
        Pick a random point within the defined area.
        """
        return (
            random.uniform(self.area["xmin"], self.area["xmax"]),
            random.uniform(self.area["ymin"], self.area["ymax"]),
        )

    def _clamp_point(self, point):
        return (
            min(max(float(point[0]), float(self.area["xmin"])), float(self.area["xmax"])),
            min(max(float(point[1]), float(self.area["ymin"])), float(self.area["ymax"])),
        )

    def on_spawn(self, x, y):
        """
        Hook called once right after the simulator overwrites .pos with the
        device's actual spawn position (before the first _start_move()).
        No-op here; subclasses with position constraints (e.g. PedestrianWalk)
        override it to re-anchor their constraint to the real spawn point.
        """
        pass

    def _start_move(self):
        self.target = self._random_point()
        self.speed = random.uniform(*self.speed_range)
        self.state = "move"
        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]
        dist = math.hypot(dx, dy) or 1.0
        self.vel_x = self.speed * dx / dist
        self.vel_y = self.speed * dy / dist

    def next_position(self, dt_ms):
        """
        Advance position by dt_ms milliseconds and return new position.
        """
        dt = dt_ms / 1000.0  # convert ms → seconds

        if self.state == "pause":
            # Count down pause time
            self.pause_time -= dt
            if self.pause_time <= 0:
                # Choose new target and start moving
                self._start_move()

        else:  # state == "move"
            dx = self.target[0] - self.pos[0]
            dy = self.target[1] - self.pos[1]
            remaining = math.hypot(dx, dy)
            step = self.speed * dt
            if step >= remaining:
                self.pos = self.target
                self.state = "pause"
                self.pause_time = random.uniform(*self.pause_range)
            else:
                self.pos = (
                    self.pos[0] + self.vel_x * dt,
                    self.pos[1] + self.vel_y * dt,
                )
            self.pos = self._clamp_point(self.pos)

        return self.pos
