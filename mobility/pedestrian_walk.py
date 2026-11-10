"""
mobility/pedestrian_walk.py

Pedestrian mobility: wanders within its home city block (the same
rectangular area the buildings/grass occupy) rather than being confined
to a single sidewalk line. Movement isn't restricted to the road-adjacent
pathway only -- pedestrians can walk across the block interior (through
"building areas"/plazas/parks) too.
"""

import random

from mobility.random_waypoint import RandomWaypoint
from mobility.road_network import build_road_grid


def _find_block(x, y, h_roads, v_roads, area):
    """Return (x0, x1, y0, y1) bounds of the city block containing (x, y)."""
    xs = [area["xmin"], *sorted(v_roads), area["xmax"]]
    ys = [area["ymin"], *sorted(h_roads), area["ymax"]]
    cx = min(max(x, area["xmin"]), area["xmax"])
    cy = min(max(y, area["ymin"]), area["ymax"])

    bi = len(xs) - 2
    for i in range(len(xs) - 1):
        if xs[i] <= cx <= xs[i + 1]:
            bi = i
            break
    bj = len(ys) - 2
    for j in range(len(ys) - 1):
        if ys[j] <= cy <= ys[j + 1]:
            bj = j
            break
    return xs[bi], xs[bi + 1], ys[bj], ys[bj + 1]


class PedestrianWalk(RandomWaypoint):
    # Small inset so pedestrians don't spawn/target a point exactly on the
    # road line bounding their block.
    MARGIN_FRAC = 0.06

    def __init__(self, area, speed_range, pause_range):
        self._block = (area["xmin"], area["xmax"], area["ymin"], area["ymax"])
        super().__init__(area, speed_range, pause_range)

    def on_spawn(self, x, y):
        h_roads, v_roads = build_road_grid(self.area)
        self._block = _find_block(x, y, h_roads, v_roads, self.area)
        self.pos = (float(x), float(y))

    def _random_point(self):
        x0, x1, y0, y1 = self._block
        mx = self.MARGIN_FRAC * (x1 - x0)
        my = self.MARGIN_FRAC * (y1 - y0)
        return (
            random.uniform(x0 + mx, max(x0 + mx, x1 - mx)),
            random.uniform(y0 + my, max(y0 + my, y1 - my)),
        )

    # Pedestrians roam freely (not axis-locked like vehicles), but the
    # renderer still wants a discrete facing direction to orient the
    # walking icon. Derive it from the current leg's velocity vector --
    # whichever component (x or y) dominates decides "left/right" vs
    # "up/down" -- so the icon roughly faces where the pedestrian is
    # actually walking. Holds the last heading while paused.
    @property
    def axis(self):
        return "h" if abs(self.vel_x) >= abs(self.vel_y) else "v"

    @property
    def direction(self):
        return 1 if (self.vel_x if self.axis == "h" else self.vel_y) >= 0 else -1
