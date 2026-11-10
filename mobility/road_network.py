"""
mobility/road_network.py

Shared Manhattan-style road grid: a fixed set of horizontal and vertical
road lines spanning the city area. Both VehicleRoadMobility (movement) and
the live GUI (drawing) call build_road_grid() with the same area, so
vehicles are always visually on the roads drawn on screen.

The grid is cached per-area so repeated calls (once per vehicle, once per
GUI frame) don't recompute or, worse, drift apart.
"""

_CACHE = {}

# Target city-block size (meters). Road count is derived from the city
# area's span divided by this, rather than a fixed line count, so the grid
# stays proportioned (block size, road spacing) as CITY_AREA changes instead
# of always drawing the same number of lines regardless of scale. For the
# default 10km x 10km area this yields exactly 3 roads per axis.
TARGET_BLOCK_SIZE_M = 2500.0

# Distance (meters) a pedestrian sidewalk sits off the road centerline it
# parallels. Shared between mobility/pedestrian_walk.py (movement) and
# live/renderers.py (drawing) so pedestrians are always visually on the
# sidewalk line drawn alongside the road.
SIDEWALK_OFFSET_M = 45.0


def _area_key(area):
    return (
        round(float(area.get("xmin", 0.0)), 3),
        round(float(area.get("xmax", 0.0)), 3),
        round(float(area.get("ymin", 0.0)), 3),
        round(float(area.get("ymax", 0.0)), 3),
    )


def _calc_road_count(span_m, block_size_m):
    return max(1, round(span_m / max(block_size_m, 1.0)) - 1)


def _snap_ends_to_border(values, lo, hi):
    """
    Push the smallest/largest value in an evenly-spaced road list out to
    the city boundary itself, leaving any road(s) between them untouched.
    For the default 3-roads-per-axis grid this turns "border, center,
    border" evenly-spaced-with-a-gap into an actual perimeter road running
    along the city edge plus one unmoved center road -- the ring road +
    single center cross the live GUI renders.
    """
    if len(values) >= 2:
        values = list(values)
        values[0], values[-1] = lo, hi
    return values


def build_road_grid(area, n_horizontal=None, n_vertical=None, block_size_m=TARGET_BLOCK_SIZE_M):
    """
    Return (h_roads, v_roads):
      h_roads: sorted list of y-coordinates for horizontal streets (E-W)
      v_roads: sorted list of x-coordinates for vertical streets (N-S)
    Evenly spaced inside the area, then the outermost road on each axis is
    snapped out to the city boundary (see _snap_ends_to_border) so the city
    is ringed by a perimeter road instead of leaving a gap between the edge
    and the first road. If n_horizontal/n_vertical aren't given explicitly,
    the count is calculated from the area's span and block_size_m, so a
    bigger city gets more roads and a smaller one gets fewer, rather than
    always the same fixed grid.
    """
    xmin, xmax = float(area.get("xmin", 0.0)), float(area.get("xmax", 10000.0))
    ymin, ymax = float(area.get("ymin", 0.0)), float(area.get("ymax", 10000.0))

    if n_horizontal is None:
        n_horizontal = _calc_road_count(ymax - ymin, block_size_m)
    if n_vertical is None:
        n_vertical = _calc_road_count(xmax - xmin, block_size_m)

    key = (_area_key(area), int(n_horizontal), int(n_vertical))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    h_roads = [ymin + (i + 1) * (ymax - ymin) / (n_horizontal + 1) for i in range(n_horizontal)]
    v_roads = [xmin + (j + 1) * (xmax - xmin) / (n_vertical + 1) for j in range(n_vertical)]
    h_roads = _snap_ends_to_border(h_roads, ymin, ymax)
    v_roads = _snap_ends_to_border(v_roads, xmin, xmax)

    grid = (h_roads, v_roads)
    _CACHE[key] = grid
    return grid


def nearest(value, options):
    """Closest value in `options` to `value` (options must be non-empty)."""
    return min(options, key=lambda o: abs(o - value))


def _row_layout(count, area):
    """
    Return [(num_cols, row_height), ...] top-to-bottom describing how to
    divide `area` into `count` roughly-square coverage cells.

    Row heights are NOT simply split evenly -- they're solved so every
    row's worst-case cell radius (distance from a cell's center to its own
    corner, which is always the farthest point in an axis-aligned
    rectangle from its center) comes out equal across rows. A row with
    more columns (so narrower cells) gets a *taller* share of the height
    than a row with fewer, wider cells -- otherwise a leftover station in
    an unevenly-divisible count (e.g. 5) can end up buying zero actual
    coverage improvement over one fewer station.
    """
    import math

    xmin, xmax = float(area["xmin"]), float(area["xmax"])
    ymin, ymax = float(area["ymin"]), float(area["ymax"])
    width = xmax - xmin
    height = ymax - ymin

    if count <= 1:
        return [(1, height)]
    if count == 2:
        return [(1, height / 2.0), (1, height / 2.0)]
    if count == 3:
        return [(1, height / 3.0)] * 3

    aspect = max(1e-6, width / max(1e-6, height))
    rows = max(1, min(count, int(round(math.sqrt(count / aspect)))))
    base = count // rows
    remainder = count % rows
    # Extra columns go to the middle row(s) first, then outward, so odd
    # counts stay visually balanced instead of leaving one side blind.
    middle_order = sorted(range(rows), key=lambda r: (abs(r - (rows - 1) / 2.0), r))
    extra_rows = set(middle_order[:remainder])
    cols_per_row = [base + (1 if r in extra_rows else 0) for r in range(rows)]

    if len(set(cols_per_row)) == 1:
        return [(c, height / rows) for c in cols_per_row]

    # Rows have different column counts -- solve for row heights that
    # equalize each row's corner radius sqrt((width/2c)^2 + (h/2)^2).
    # Bisect on the shared target radius R: for a given R every row's
    # height is determined, then scale so heights sum to the city height.
    def heights_for_radius(radius):
        out = []
        for cols in cols_per_row:
            half_w = width / (2.0 * cols)
            out.append(2.0 * math.sqrt(max(0.0, radius * radius - half_w * half_w)))
        return out

    lo, hi = 0.0, math.hypot(width / (2.0 * min(cols_per_row)), height)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if sum(heights_for_radius(mid)) < height:
            lo = mid
        else:
            hi = mid
    row_heights = heights_for_radius(hi)
    total = sum(row_heights) or 1.0
    row_heights = [h * height / total for h in row_heights]
    return list(zip(cols_per_row, row_heights))


def distribute_coverage_cells(count, area):
    """
    Divide `area` into `count` coverage cells (see _row_layout for how
    row/column counts and row heights are chosen so each additional
    station actually shrinks the worst-covered gap). Returns a list of
    dicts, row-major top-to-bottom then left-to-right: {"pos": (cx, cy),
    "bounds": (x0, x1, y0, y1)}. Shared by MEC placement (station at each
    cell's center) and UAV_BS patrol zoning (loop around each cell,
    including which edges border a neighboring cell -- see UAVBSPatrol).
    """
    if count <= 0:
        return []
    xmin, xmax = float(area["xmin"]), float(area["xmax"])
    ymin, ymax = float(area["ymin"]), float(area["ymax"])

    cells = []
    y_top = ymax
    for cols, row_height in _row_layout(count, area):
        y_bottom = y_top - row_height
        for col in range(cols):
            x0 = xmin + col * (xmax - xmin) / cols
            x1 = xmin + (col + 1) * (xmax - xmin) / cols
            cells.append({
                "pos": ((x0 + x1) / 2.0, (y_top + y_bottom) / 2.0),
                "bounds": (x0, x1, y_bottom, y_top),
            })
        y_top = y_bottom
    return cells[:count]


def distribute_coverage_positions(count, area):
    """Station center points only -- see distribute_coverage_cells for bounds."""
    return [cell["pos"] for cell in distribute_coverage_cells(count, area)]


def farthest_first_order(positions):
    """Rank positions by greedy coverage (farthest-point traversal / k-center).

    Returns ``positions`` reordered so any prefix ``order[:k]`` is a good
    (2-approximate) k-center choice: it minimizes the worst-case distance from
    any position to its nearest chosen one. Seeded with the position nearest
    the set's own centroid, so the first entry is centrally located.

    Growing k only ever ADDS to the prefix -- earlier entries never move -- so
    this is stable across different k values (promoting one more site never
    reassigns a site already promoted). Used to pick which of a fixed set of
    ground base-station positions get upgraded with compute, instead of
    placing compute sites independently.
    """
    import math

    points = list(positions)
    if len(points) <= 1:
        return points

    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    seed_index = min(
        range(len(points)), key=lambda i: math.hypot(points[i][0] - cx, points[i][1] - cy)
    )

    order = [points[seed_index]]
    remaining = [p for i, p in enumerate(points) if i != seed_index]
    nearest_chosen_distance = [
        math.hypot(p[0] - order[0][0], p[1] - order[0][1]) for p in remaining
    ]

    while remaining:
        pick_index = max(range(len(remaining)), key=lambda i: nearest_chosen_distance[i])
        picked = remaining.pop(pick_index)
        nearest_chosen_distance.pop(pick_index)
        order.append(picked)
        for i, p in enumerate(remaining):
            distance_to_new_pick = math.hypot(p[0] - picked[0], p[1] - picked[1])
            if distance_to_new_pick < nearest_chosen_distance[i]:
                nearest_chosen_distance[i] = distance_to_new_pick

    return order


