# live/renderers.py
import os
import math
import random
import tkinter as tk
from typing import Dict, List, Tuple, Optional

from .tiers import TIER_COLORS, SERVER_PALETTE, tier_from_server_obj

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
    _RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
except Exception:
    _HAS_PIL = False
    _RESAMPLE = None

PAD        = 12  # px, city-area margin -- kept just wide enough that the
                  # perimeter ring road's own stroke width isn't clipped,
                  # so the city fills the canvas right up to its edges.
CLOUD_MARGIN_PX = 30  # px, clearance for the off-map Cloud badge from the
                       # corner -- separate from PAD so Cloud (an abstract
                       # "outside the city" marker) stays visibly inside the
                       # ring corner even though PAD itself is tight.
ICON_SIZE  = 22   # legacy default; most calls now pass an explicit size
DEVICE_R   = 3
GRID_SNAP  = 3  # px; small bucketing to avoid perfect overlaps

# uav_computation.png's nose doesn't point along its own unrotated +x axis
# (due-east/right) like the pedestrian sprites do -- measured via PCA over
# the icon's alpha mask, its nose points ~32.5 degrees below due-east. Any
# heading-based rotation has to correct for that fixed offset, or the icon
# visibly points off to one side of the direction it's actually flying.
UAV_ICON_NOSE_OFFSET_DEG = -32.5


def _srv_key(srv) -> str:
    return f"{getattr(srv, 'name', 'srv')}#{getattr(srv, 'id', None) or id(srv)}"


class Assets:
    """
    PIL-backed icon loader: arbitrary resize + rotation, with a small
    render cache keyed by (name, size, angle) so repeated draw() calls
    don't re-render the same icon every frame. Falls back to "no icon"
    (callers draw plain shapes instead) if PIL or a given file is missing.
    """
    NAMES = [
        "ambulance", "bicycle", "building-1", "building-2", "building-3", "building-4",
        "building-5", "building-6", "car", "car_l", "car_r", "car_t", "car_d",
        "cctv", "cloud", "edge", "electric-meter", "gateway", "man", "mec", "metro",
        "parking", "pedestrian-crossing", "radio-tower", "road", "school-bus",
        "sprout", "street-lamp", "traffic-lights", "uav_computation", "uav_workload", "woman",
    ]

    def __init__(self, base_dir: str):
        self.assets_dir = os.path.join(base_dir, "assets")
        if not os.path.isdir(self.assets_dir):
            self.assets_dir = os.path.join(os.path.dirname(__file__), "assets")

        self._source: Dict[str, "Image.Image"] = {}
        self._cache: Dict[tuple, "ImageTk.PhotoImage"] = {}

        if _HAS_PIL:
            for name in self.NAMES:
                path = os.path.join(self.assets_dir, f"{name}.png")
                if os.path.isfile(path):
                    try:
                        self._source[name] = Image.open(path).convert("RGBA")
                    except Exception:
                        pass

        # Tier icons for servers (ServerRenderer reads assets.by_tier[tier]).
        self.by_tier = {
            "MEC": self.get("mec", 36),
            "ACCESS_POINT": self.get("radio-tower", 34),
            "Metro": self.get("metro", 36),
            "Cloud": self.get("cloud", 40),
            "UAV_BS": self.get("uav_computation", 36),
        }
        # Back-compat attributes some code may still reference directly.
        self.MEC = self.by_tier["MEC"]
        self.Metro = self.by_tier["Metro"]
        self.cloud = self.by_tier["Cloud"]

    def has(self, name: str) -> bool:
        return name in self._source

    def source(self, name: str):
        """Raw PIL.Image (RGBA), or None. For callers doing custom compositing."""
        return self._source.get(name)

    def get(self, name: str, size, angle: float = 0.0, flip: bool = False):
        """
        Cached ImageTk.PhotoImage for `name`, scaled so its longer side is
        `size` px (or an exact (w,h) tuple), optionally mirrored left-right
        (flip) and then rotated by `angle` degrees. Returns None if PIL or
        the named asset isn't available.
        """
        if not _HAS_PIL or name not in self._source:
            return None
        size_key = size if isinstance(size, tuple) else int(size)
        key = (name, size_key, round(float(angle), 1), bool(flip))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        img = self._source[name]
        if isinstance(size, tuple):
            target_w, target_h = int(size[0]), int(size[1])
        else:
            w, h = img.size
            scale = float(size) / float(max(w, h))
            target_w, target_h = max(1, int(w * scale)), max(1, int(h * scale))

        resized = img.resize((target_w, target_h), _RESAMPLE)
        if flip:
            resized = resized.transpose(Image.FLIP_LEFT_RIGHT)
        if angle:
            resized = resized.rotate(-angle, expand=True, resample=Image.BICUBIC)
        photo = ImageTk.PhotoImage(resized)
        self._cache[key] = photo  # keep a ref alive; Tk drops GC'd images
        return photo

class Layout:
    """Transforms between simulation coords and canvas pixels, live-sized."""
    def __init__(self, canvas: tk.Canvas, xmin: float, xmax: float, ymin: float, ymax: float):
        self.cv   = canvas
        self.xmin = xmin; self.xmax = xmax
        self.ymin = ymin; self.ymax = ymax

    def sx(self, x: float) -> float:
        w = max(1, self.cv.winfo_width())
        if self.xmax == self.xmin: return PAD
        return PAD + (x - self.xmin) * (w - 2*PAD) / (self.xmax - self.xmin)

    def sy(self, y: float) -> float:
        h = max(1, self.cv.winfo_height())
        if self.ymax == self.ymin: return PAD
        return h - PAD - (y - self.ymin) * (h - 2*PAD) / (self.ymax - self.ymin)

    def inv_sx(self, cx: float) -> float:
        w = max(1, self.cv.winfo_width())
        return self.xmin + (cx - PAD) * (self.xmax - self.xmin) / max(1.0, (w - 2*PAD))

    def inv_sy(self, cy: float) -> float:
        h = max(1, self.cv.winfo_height())
        return self.ymin + (h - PAD - cy) * (self.ymax - self.ymin) / max(1.0, (h - 2*PAD))

    def clamp_canvas(self, cx: float, cy: float) -> Tuple[float, float]:
        w, h = self.cv.winfo_width(), self.cv.winfo_height()
        return (min(max(PAD, cx), w - PAD), min(max(PAD, cy), h - PAD))


class SkyBackground:
    """
    Soft vertical gradient behind the whole scene (lifts the flat-white
    canvas without pretending this top-down map is a literal sky view).
    Regenerated only when the canvas is resized, not every tick.
    """
    TOP = (219, 234, 254)      # soft sky blue
    BOTTOM = (245, 243, 235)   # warm ground tone

    def __init__(self, canvas: tk.Canvas):
        self.cv = canvas
        self._photo = None
        self._item = None
        self._last_size = (0, 0)

    def draw(self):
        w = max(1, self.cv.winfo_width())
        h = max(1, self.cv.winfo_height())
        if not _HAS_PIL:
            self.cv.configure(bg="#eef2f7")
            return
        if (w, h) != self._last_size:
            grad = Image.new("RGB", (1, h))
            for y in range(h):
                t = y / max(1, h - 1)
                r = int(self.TOP[0] + (self.BOTTOM[0] - self.TOP[0]) * t)
                g = int(self.TOP[1] + (self.BOTTOM[1] - self.TOP[1]) * t)
                b = int(self.TOP[2] + (self.BOTTOM[2] - self.TOP[2]) * t)
                grad.putpixel((0, y), (r, g, b))
            grad = grad.resize((w, h))
            self._photo = ImageTk.PhotoImage(grad)
            if self._item is None:
                self._item = self.cv.create_image(0, 0, image=self._photo, anchor="nw", tags=("sky",))
            else:
                self.cv.itemconfig(self._item, image=self._photo)
                self.cv.coords(self._item, 0, 0)
            self._last_size = (w, h)
        self.cv.tag_lower("sky")


def _city_blocks(area, n_horizontal, n_vertical):
    """xs/ys grid lines (including the outer area edges) bounding each city block."""
    from mobility.road_network import build_road_grid
    h_roads, v_roads = build_road_grid(area, n_horizontal, n_vertical)
    xs = [area["xmin"], *sorted(v_roads), area["xmax"]]
    ys = [area["ymin"], *sorted(h_roads), area["ymax"]]
    return xs, ys


class GroundRenderer:
    """
    Soft grass-green fill for every city block (between the sky and the
    buildings/roads), so the ground between streets reads as park/lawn
    space rather than empty background.
    """
    FILL = "#bfe3a6"

    def __init__(self, canvas: tk.Canvas, layout: Layout, area: dict,
                 n_horizontal=None, n_vertical=None):
        self.cv = canvas
        self.ly = layout
        self.area = area
        self.n_horizontal = n_horizontal
        self.n_vertical = n_vertical
        self._items: List[int] = []
        self._blocks = None  # [(x0,x1,y0,y1), ...] in simulation coords

    def _ensure_blocks(self):
        if self._blocks is not None:
            return
        xs, ys = _city_blocks(self.area, self.n_horizontal, self.n_vertical)
        self._blocks = []
        for bi in range(len(xs) - 1):
            for bj in range(len(ys) - 1):
                x0, x1, y0, y1 = xs[bi], xs[bi + 1], ys[bj], ys[bj + 1]
                if (x1 - x0) < 1.0 or (y1 - y0) < 1.0:
                    continue
                self._blocks.append((x0, x1, y0, y1))

    def draw(self):
        self._ensure_blocks()
        if not self._items:
            for _ in self._blocks:
                self._items.append(self.cv.create_rectangle(0, 0, 0, 0, fill=self.FILL, outline="", tags=("ground",)))
        for (x0, x1, y0, y1), item in zip(self._blocks, self._items):
            # sy() flips vertical order (larger sim-y -> smaller screen-y)
            cx0, cy0 = self.ly.sx(x0), self.ly.sy(y1)
            cx1, cy1 = self.ly.sx(x1), self.ly.sy(y0)
            self.cv.coords(item, cx0, cy0, cx1, cy1)
        self.cv.tag_lower("ground")
        self.cv.tag_raise("ground", "sky")


class BuildingRenderer:
    """
    Deterministic building skyline filling the city blocks between roads
    (above the grass ground, below roads/devices/servers). The layout
    (which block gets which building at what offset/size) is fixed at
    construction time so buildings don't jitter frame to frame -- only
    their pixel position is recomputed on redraw (resize-safe).
    """
    TYPES = ["building-1", "building-2", "building-3", "building-4", "building-5", "building-6"]
    SIZE_BUCKETS = [130, 160, 190]  # px; quantized so the icon cache stays tiny

    def __init__(self, canvas: tk.Canvas, layout: Layout, area: dict, assets: Assets,
                 n_horizontal=None, n_vertical=None, seed=1234):
        self.cv = canvas
        self.ly = layout
        self.area = area
        self.assets = assets
        self._items: List[int] = []
        self._buildings = self._layout_buildings(n_horizontal, n_vertical, seed)

    def _layout_buildings(self, n_horizontal, n_vertical, seed):
        xs, ys = _city_blocks(self.area, n_horizontal, n_vertical)

        rng = random.Random(seed)
        buildings = []
        # Buildings are deliberately large enough now that they may
        # occasionally overlap a road or sidewalk slightly -- that's fine
        # (skyscraper scale matters more than strict non-overlap); this
        # margin just keeps them roughly centered in their block.
        margin = 0.16
        for bi in range(len(xs) - 1):
            for bj in range(len(ys) - 1):
                x0, x1 = xs[bi], xs[bi + 1]
                y0, y1 = ys[bj], ys[bj + 1]
                if (x1 - x0) < 1.0 or (y1 - y0) < 1.0:
                    continue
                # Fewer, larger calculated blocks (see road_network.TARGET_BLOCK_SIZE_M)
                # hold more buildings each to keep the skyline from looking sparse.
                count = rng.choice([2, 2, 3, 3, 4])
                for _ in range(count):
                    bx = rng.uniform(x0 + margin * (x1 - x0), x1 - margin * (x1 - x0))
                    by = rng.uniform(y0 + margin * (y1 - y0), y1 - margin * (y1 - y0))
                    btype = rng.choice(self.TYPES)
                    size = rng.choice(self.SIZE_BUCKETS)
                    buildings.append({"x": bx, "y": by, "type": btype, "size": size})
        return buildings

    def draw(self):
        if not _HAS_PIL:
            return
        if not self._items:
            for b in self._buildings:
                img = self.assets.get(b["type"], b["size"])
                if img is None:
                    continue
                item = self.cv.create_image(0, 0, image=img, tags=("building",))
                self._items.append(item)
                b["_item"] = item
        for b in self._buildings:
            item = b.get("_item")
            if item is None:
                continue
            cx, cy = self.ly.sx(b["x"]), self.ly.sy(b["y"])
            self.cv.coords(item, cx, cy)
        self.cv.tag_lower("building")
        self.cv.tag_raise("building", "ground")


class ServerRenderer:
    """Draw servers, allow drag in Setup, resolve near-overlaps, persist positions."""
    def __init__(self, canvas: tk.Canvas, layout: Layout, assets: Assets):
        self.cv = canvas
        self.ly = layout
        self.assets = assets

        self.server_colors: Dict[str, str] = {}
        self.ring_items:   Dict[str, int]  = {}
        self.icon_items:   Dict[str, int]  = {}
        self.label_items:  Dict[str, int]  = {}
        self.label_bg:     Dict[str, int]  = {}

        self.drag_enabled = False
        self._drag_key: Optional[str] = None
        self._drag_dx = 0.0
        self._drag_dy = 0.0

        self._servers_ref: List[object] = []  # updated on each draw()

    def assign_palette(self, servers: List[object]):
        self.server_colors.clear()
        for i, s in enumerate(servers):
            self.server_colors[_srv_key(s)] = SERVER_PALETTE[i % len(SERVER_PALETTE)]

    def _compute_positions(self, servers: List[object]) -> Dict[str, Tuple[float, float]]:
        """Return {key: (cx,cy)} with light anti-overlap bucketing."""
        pos: Dict[str, Tuple[float,float]] = {}
        buckets: Dict[Tuple[int,int], List[Tuple[str,float,float]]] = {}
        for s in servers:
            x, y = getattr(s, "x", None), getattr(s, "y", None)
            if x is None or y is None:
                continue
            cx, cy = self.ly.sx(float(x)), self.ly.sy(float(y))
            ix, iy = int(round(cx/GRID_SNAP))*GRID_SNAP, int(round(cy/GRID_SNAP))*GRID_SNAP
            buckets.setdefault((ix,iy), []).append((_srv_key(s), cx, cy))



        for (ix,iy), items in buckets.items():
            if len(items) == 1:
                k,cx,cy = items[0]; pos[k] = (cx,cy)
            else:
                R = 14.0
                for j,(k,cx,cy) in enumerate(items):
                    ang = 2*math.pi*j/len(items)
                    pos[k] = (ix + R*math.cos(ang), iy + R*math.sin(ang))
        return pos

    def _tag(self, key: str) -> str:
        return f"srv:{key}"

    def enable_drag(self, enable: bool):
        self.drag_enabled = bool(enable)

    # ----- mouse handlers -----
    def bind_handlers(self):
        self.cv.tag_bind("srv", "<ButtonPress-1>",  self._on_down)
        self.cv.tag_bind("srv", "<B1-Motion>",      self._on_move)
        self.cv.tag_bind("srv", "<ButtonRelease-1>",self._on_up)

    def _find_server_by_key(self, key: str):
        for s in self._servers_ref:
            if _srv_key(s) == key:
                return s
        return None

    def _on_down(self, evt):
        if not self.drag_enabled: return
        items = self.cv.find_withtag("current")
        if not items: return
        for key, item in self.icon_items.items():
            if item == items[0]:
                self._drag_key = key
                x0,y0 = self.cv.coords(item)
                self._drag_dx, self._drag_dy = x0 - evt.x, y0 - evt.y
                return

    def _on_move(self, evt):
        if not self.drag_enabled or not self._drag_key: return
        cx, cy = self.ly.clamp_canvas(evt.x + self._drag_dx, evt.y + self._drag_dy)
        self._move_drawn(self._drag_key, cx, cy)
        # Persist back into server object (no reset button needed)
        srv = self._find_server_by_key(self._drag_key)
        if srv is not None:
            setattr(srv, "x", float(self.ly.inv_sx(cx)))
            setattr(srv, "y", float(self.ly.inv_sy(cy)))

    def _on_up(self, _evt):
        self._drag_key = None

    def _move_drawn(self, key: str, cx: float, cy: float):
        if key in self.icon_items:
            iid = self.icon_items[key]
            if self.cv.type(iid) == "image":
                self.cv.coords(iid, cx, cy)
            else:
                size = 12
                self.cv.coords(iid, cx - size, cy - size, cx + size, cy + size)
        if key in self.ring_items:
            r = 16
            self.cv.coords(self.ring_items[key], cx - r, cy - r, cx + r, cy + r)
        if key in self.label_items and key in self.label_bg:
            txt = self.label_items[key]
            self.cv.coords(txt, cx, cy - 18)
            x1,y1,x2,y2 = self.cv.bbox(txt)
            self.cv.coords(self.label_bg[key], x1 - 4, y1 - 2, x2 + 4, y2 + 2)
            self.cv.tag_raise(txt, self.label_bg[key])

    def draw(self, servers: List[object], color_mode: str = "tier"):
        self._servers_ref = servers
        if not self.server_colors:
            self.assign_palette(servers)

        pos = self._compute_positions(servers)

        # draw rings + icons + labels
        for s in servers:
            key = _srv_key(s)
            cx, cy = pos.get(
                key,
                (self.ly.sx(getattr(s, "x", 0.0)), self.ly.sy(getattr(s, "y", 0.0)))
            )
            tier = tier_from_server_obj(s)
            color = (
                TIER_COLORS.get(tier, "#888888")
                if color_mode == "tier"
                else self.server_colors.get(key, "#888888")
            )

            # ---- draw-only override for Cloud: place icon top-right inside padding ----
            if tier == "Cloud":
                w = max(1, self.cv.winfo_width())
                draw_cx, draw_cy = (w - CLOUD_MARGIN_PX, CLOUD_MARGIN_PX)
            else:
                draw_cx, draw_cy = (cx, cy)

            # ring
            r = 10
            if key not in self.ring_items:
                self.ring_items[key] = self.cv.create_oval(
                    draw_cx - r, draw_cy - r, draw_cx + r, draw_cy + r,
                    outline=color, width=2, tags=("srv",)
                )
            else:
                self.cv.coords(self.ring_items[key], draw_cx - r, draw_cy - r, draw_cx + r, draw_cy + r)
                self.cv.itemconfig(self.ring_items[key], outline=color)

            # icon -- UAV_BS rotates to face its actual patrol direction.
            # UAVBSPatrol.heading is math/world heading in degrees
            # (atan2(dy, dx)); Assets.get handles canvas rotation convention.
            heading = getattr(getattr(s, "mobility", None), "heading", None)
            if tier == "UAV_BS" and heading is not None:
                # Assets.get(angle=X) rotates the icon clockwise by X degrees
                # (it hands PIL rotate(-X), which is counter-clockwise -X);
                # to make the nose end up pointing at `heading` (measured
                # counter-clockwise from due-east, same convention as the
                # icon's own baseline nose offset), the rotation must be
                # X = UAV_ICON_NOSE_OFFSET_DEG - heading.
                uav_angle = UAV_ICON_NOSE_OFFSET_DEG - heading
                img = self.assets.get("uav_computation", 36, angle=round(uav_angle / 5.0) * 5.0) or self.assets.by_tier.get(tier)
            else:
                img = self.assets.by_tier.get(tier)
            if key not in self.icon_items:
                if img is not None:
                    self.icon_items[key] = self.cv.create_image(draw_cx, draw_cy, image=img, tags=("srv",))
                else:
                    self.icon_items[key] = self.cv.create_rectangle(
                        draw_cx - 6, draw_cy - 6, draw_cx + 6, draw_cy + 6,
                        fill=color, outline="", tags=("srv",)
                    )
            else:
                self.cv.coords(self.icon_items[key], draw_cx, draw_cy)
                if tier == "UAV_BS" and img is not None and self.cv.type(self.icon_items[key]) == "image":
                    self.cv.itemconfig(self.icon_items[key], image=img)

            # label
            label = getattr(s, "name", "srv")
            if key not in self.label_items:
                t = self.cv.create_text(draw_cx, draw_cy + 16, text=label, font=("Segoe UI", 9))
                x1, y1, x2, y2 = self.cv.bbox(t)
                bg = self.cv.create_rectangle(x1 - 4, y1 - 2, x2 + 4, y2 + 2, fill="#ffffff", outline="#e5e7eb")
                self.cv.tag_lower(bg, t)
                self.label_items[key] = t
                self.label_bg[key] = bg
            else:
                t = self.label_items[key]
                self.cv.itemconfig(t, text=label)
                self.cv.coords(t, draw_cx, draw_cy + 16)
                x1, y1, x2, y2 = self.cv.bbox(t)
                self.cv.coords(self.label_bg[key], x1 - 4, y1 - 2, x2 + 4, y2 + 2)
                self.cv.tag_lower(self.label_bg[key], t)



class RoadRenderer:
    """
    Draws the shared road grid (mobility/road_network.py) as a clean
    multi-lane road: a dark asphalt band with two white lane-divider lines
    splitting it into left/center/right lanes, plus a small crosswalk icon
    at every intersection. Deliberately NOT a tiled image texture -- a
    repeating icon (like road.png) doesn't tile seamlessly and reads as
    disconnected chunks rather than a continuous road. Items are created
    once and repositioned on every draw() call (cheap, no flicker,
    resize-safe).

    The outermost road on each axis is snapped to the city boundary (see
    road_network.build_road_grid), so it forms a perimeter/ring road all
    the way around the city; that ring is rendered as one continuous
    rounded-rectangle asphalt band (straight sides + a small corner arc)
    instead of the plain per-line rendering used for interior/center roads,
    which is what actually reads as "the city border" instead of a bare
    grid line -- it carries the same dashed lane-divider markings as the
    interior roads (no sidewalks though, so the rounded corners stay
    clean). A row of small streetlight glyphs lines every sidewalk as a
    lightweight smart-city touch.
    """
    BAND_WIDTH = 20       # px, asphalt band thickness -- narrower than a building icon
    LANE_OFFSET = 6       # px, distance of each divider line from the road centerline
    CORNER_RADIUS_PX = 18  # px, subtle rounding on the perimeter ring's corners

    SIDEWALK_COLOR = "#c9bfae"

    STREETLIGHT_SPACING_M = 900.0
    STREETLIGHT_POLE_COLOR = "#4b5563"
    STREETLIGHT_LAMP_COLOR = "#ffd479"
    STREETLIGHT_POLE_PX = 10
    STREETLIGHT_LAMP_R_PX = 3

    # Pedestrian crossings: placed a short distance off each real junction,
    # along the road they actually cross, instead of stacked exactly on the
    # intersection point. The source icon's stripes run along its long
    # (unrotated) axis, which matches crossing a north-south road, so a
    # crossing over an east-west road needs the icon rotated 90 degrees.
    CROSSING_OFFSET_M = 150.0

    def __init__(self, canvas: tk.Canvas, layout: Layout, area: dict, assets: Assets = None,
                 n_horizontal=None, n_vertical=None):
        self.cv = canvas
        self.ly = layout
        self.area = area
        self.assets = assets
        self.n_horizontal = n_horizontal
        self.n_vertical = n_vertical
        self._h_items: List[dict] = []
        self._v_items: List[dict] = []
        self._crossing_items: List[int] = []
        self._crossing_specs: List[Tuple[float, float, str]] = []
        self._ring_items: Optional[dict] = None
        self._streetlight_points: Optional[List[Tuple[float, float]]] = None
        self._streetlight_items: List[Tuple[int, int]] = []
        self._built = False

    @staticmethod
    def _border_ends(values):
        """(lo, hi) -- the two values snapped to the city boundary -- or
        None if this axis has fewer than 2 roads (no border to ring)."""
        return (values[0], values[-1]) if len(values) >= 2 else None

    def _build_crossing_specs(self, h_roads, v_roads, border_hy, border_vx):
        """
        One (world_x, world_y, axis) spec per real crosswalk. Every
        h_road/v_road pair is a genuine junction -- including where the
        ring meets an interior road -- EXCEPT the four pure ring corners,
        which are just the perimeter road turning a bend with nothing to
        cross. axis='v' means the crossing spans a vertical (north-south)
        road (icon unrotated); axis='h' spans a horizontal (east-west)
        road (icon rotated 90 degrees). Each is nudged CROSSING_OFFSET_M
        off the exact junction point, along the road it crosses, biased
        inward at the city edge so it never lands outside CITY_AREA.
        """
        xmin, xmax = float(self.area["xmin"]), float(self.area["xmax"])
        ymin, ymax = float(self.area["ymin"]), float(self.area["ymax"])

        def inward(value, lo, hi):
            if value <= lo + 1e-6:
                return self.CROSSING_OFFSET_M
            if value >= hi - 1e-6:
                return -self.CROSSING_OFFSET_M
            return self.CROSSING_OFFSET_M

        specs = []
        for road_y in h_roads:
            for road_x in v_roads:
                if road_y in border_hy and road_x in border_vx:
                    continue  # pure ring corner -- a turn, not a crossing
                specs.append((road_x, road_y + inward(road_y, ymin, ymax), "v"))
                specs.append((road_x + inward(road_x, xmin, xmax), road_y, "h"))
        return specs

    def draw(self):
        from mobility.road_network import build_road_grid, SIDEWALK_OFFSET_M
        h_roads, v_roads = build_road_grid(self.area, self.n_horizontal, self.n_vertical)
        w = max(1, self.cv.winfo_width())
        h = max(1, self.cv.winfo_height())

        h_border = self._border_ends(h_roads)
        v_border = self._border_ends(v_roads)
        border_hy = set(h_border) if h_border else set()
        border_vx = set(v_border) if v_border else set()
        interior_h = [y for y in h_roads if y not in border_hy]
        interior_v = [x for x in v_roads if x not in border_vx]

        if not self._built:
            for _ in interior_h:
                self._h_items.append(self._make_band(vertical=False))
            for _ in interior_v:
                self._v_items.append(self._make_band(vertical=True))
            if h_border and v_border:
                self._ring_items = self._make_ring()
            if self.assets is not None:
                self._crossing_specs = self._build_crossing_specs(h_roads, v_roads, border_hy, border_vx)
                for _, _, axis in self._crossing_specs:
                    angle = 90.0 if axis == "h" else 0.0
                    img = self.assets.get("pedestrian-crossing", (30, 12), angle=angle)
                    item = self.cv.create_image(0, 0, image=img, tags=("crossing",))
                    self._crossing_items.append(item)
            self._built = True

        for road_y, band in zip(interior_h, self._h_items):
            cy = self.ly.sy(road_y)
            self.cv.coords(band["asphalt"], 0, cy, w, cy)
            self.cv.coords(band["div1"], 0, cy - self.LANE_OFFSET, w, cy - self.LANE_OFFSET)
            self.cv.coords(band["div2"], 0, cy + self.LANE_OFFSET, w, cy + self.LANE_OFFSET)
            sy_a = self.ly.sy(road_y + SIDEWALK_OFFSET_M)
            sy_b = self.ly.sy(road_y - SIDEWALK_OFFSET_M)
            self.cv.coords(band["walk1"], 0, sy_a, w, sy_a)
            self.cv.coords(band["walk2"], 0, sy_b, w, sy_b)
        for road_x, band in zip(interior_v, self._v_items):
            cx = self.ly.sx(road_x)
            self.cv.coords(band["asphalt"], cx, 0, cx, h)
            self.cv.coords(band["div1"], cx - self.LANE_OFFSET, 0, cx - self.LANE_OFFSET, h)
            self.cv.coords(band["div2"], cx + self.LANE_OFFSET, 0, cx + self.LANE_OFFSET, h)
            sx_a = self.ly.sx(road_x + SIDEWALK_OFFSET_M)
            sx_b = self.ly.sx(road_x - SIDEWALK_OFFSET_M)
            self.cv.coords(band["walk1"], sx_a, 0, sx_a, h)
            self.cv.coords(band["walk2"], sx_b, 0, sx_b, h)

        if self._ring_items is not None:
            self._position_ring(h_border, v_border)

        if self._crossing_items:
            for (wx, wy, _axis), item in zip(self._crossing_specs, self._crossing_items):
                self.cv.coords(item, self.ly.sx(wx), self.ly.sy(wy))
            self.cv.tag_lower("crossing")
            self.cv.tag_raise("crossing", "road")

        self._draw_streetlights(h_roads, v_roads)

        self.cv.tag_lower("sidewalk")
        self.cv.tag_raise("sidewalk", "building")
        self.cv.tag_lower("road")
        self.cv.tag_raise("road", "sidewalk")
        self.cv.tag_lower("streetlight")
        self.cv.tag_raise("streetlight", "road")

    def _make_band(self, vertical: bool) -> dict:
        # Sidewalks first (below, on either side), then the asphalt bed, then
        # two dashed white lane dividers on top.
        walk1 = self.cv.create_line(0, 0, 0, 0, fill=self.SIDEWALK_COLOR, width=6, tags=("sidewalk",))
        walk2 = self.cv.create_line(0, 0, 0, 0, fill=self.SIDEWALK_COLOR, width=6, tags=("sidewalk",))
        asphalt = self.cv.create_line(0, 0, 0, 0, fill="#33383f", width=self.BAND_WIDTH, tags=("road",))
        div1 = self.cv.create_line(0, 0, 0, 0, fill="#f5f5f5", width=2, dash=(10, 8), tags=("road",))
        div2 = self.cv.create_line(0, 0, 0, 0, fill="#f5f5f5", width=2, dash=(10, 8), tags=("road",))
        return {"asphalt": asphalt, "div1": div1, "div2": div2, "walk1": walk1, "walk2": walk2}

    # ---- perimeter ring road (rounded rectangle around the whole city) ----
    def _make_ring(self) -> dict:
        # 4 straight edges (each inset short of the corners) + 4 corner
        # arcs for the asphalt bed, plus the same dashed lane-divider
        # treatment as the interior roads (two parallel dashed lines
        # offset LANE_OFFSET to either side of centerline), continued as
        # concentric dashed arcs of radius +/- LANE_OFFSET around each
        # rounded corner so the markings read as one continuous road.
        sides = {name: self.cv.create_line(0, 0, 0, 0, fill="#33383f",
                                            width=self.BAND_WIDTH, tags=("road",))
                 for name in ("top", "bottom", "left", "right")}
        corners = {name: self.cv.create_arc(0, 0, 0, 0, start=0, extent=90, style=tk.ARC,
                                             outline="#33383f", width=self.BAND_WIDTH, tags=("road",))
                   for name in ("tl", "tr", "bl", "br")}
        side_divs = {f"{name}_div{n}": self.cv.create_line(0, 0, 0, 0, fill="#f5f5f5",
                                                            width=2, dash=(10, 8), tags=("road",))
                     for name in ("top", "bottom", "left", "right") for n in (1, 2)}
        corner_divs = {f"{name}_div{n}": self.cv.create_arc(0, 0, 0, 0, start=0, extent=90, style=tk.ARC,
                                                             outline="#f5f5f5", width=2, dash=(6, 5), tags=("road",))
                       for name in ("tl", "tr", "bl", "br") for n in (1, 2)}
        return {**sides, **corners, **side_divs, **corner_divs}

    def _position_ring(self, h_border, v_border):
        ymin_y, ymax_y = h_border  # (lower sim-y, upper sim-y) -- sorted ascending
        xmin_x, xmax_x = v_border
        r = self.CORNER_RADIUS_PX

        # Canvas-space corners (sy flips vertical order: larger sim-y -> smaller screen-y).
        left_cx, right_cx = self.ly.sx(xmin_x), self.ly.sx(xmax_x)
        top_cy, bottom_cy = self.ly.sy(ymax_y), self.ly.sy(ymin_y)

        off = self.LANE_OFFSET

        ring = self._ring_items
        self.cv.coords(ring["top"],    left_cx + r, top_cy,    right_cx - r, top_cy)
        self.cv.coords(ring["bottom"], left_cx + r, bottom_cy, right_cx - r, bottom_cy)
        self.cv.coords(ring["left"],   left_cx,  top_cy + r,   left_cx,  bottom_cy - r)
        self.cv.coords(ring["right"],  right_cx, top_cy + r,   right_cx, bottom_cy - r)

        self.cv.coords(ring["top_div1"],    left_cx + r, top_cy - off,    right_cx - r, top_cy - off)
        self.cv.coords(ring["top_div2"],    left_cx + r, top_cy + off,    right_cx - r, top_cy + off)
        self.cv.coords(ring["bottom_div1"], left_cx + r, bottom_cy - off, right_cx - r, bottom_cy - off)
        self.cv.coords(ring["bottom_div2"], left_cx + r, bottom_cy + off, right_cx - r, bottom_cy + off)
        self.cv.coords(ring["left_div1"],   left_cx - off, top_cy + r,    left_cx - off, bottom_cy - r)
        self.cv.coords(ring["left_div2"],   left_cx + off, top_cy + r,    left_cx + off, bottom_cy - r)
        self.cv.coords(ring["right_div1"],  right_cx - off, top_cy + r,   right_cx - off, bottom_cy - r)
        self.cv.coords(ring["right_div2"],  right_cx + off, top_cy + r,   right_cx + off, bottom_cy - r)

        # Tk arc angles: 0=3-o'clock, increasing counter-clockwise on screen.
        self.cv.coords(ring["tl"], left_cx,      top_cy,        left_cx + 2*r, top_cy + 2*r)
        self.cv.itemconfig(ring["tl"], start=90, extent=90)
        self.cv.coords(ring["tr"], right_cx - 2*r, top_cy,      right_cx,      top_cy + 2*r)
        self.cv.itemconfig(ring["tr"], start=0, extent=90)
        self.cv.coords(ring["bl"], left_cx,      bottom_cy - 2*r, left_cx + 2*r, bottom_cy)
        self.cv.itemconfig(ring["bl"], start=180, extent=90)
        self.cv.coords(ring["br"], right_cx - 2*r, bottom_cy - 2*r, right_cx, bottom_cy)
        self.cv.itemconfig(ring["br"], start=270, extent=90)

        def _corner_divs(name, cx, cy, start):
            self.cv.coords(ring[f"{name}_div1"], cx - (r - off), cy - (r - off), cx + (r - off), cy + (r - off))
            self.cv.itemconfig(ring[f"{name}_div1"], start=start, extent=90)
            self.cv.coords(ring[f"{name}_div2"], cx - (r + off), cy - (r + off), cx + (r + off), cy + (r + off))
            self.cv.itemconfig(ring[f"{name}_div2"], start=start, extent=90)

        _corner_divs("tl", left_cx + r,  top_cy + r,    90)
        _corner_divs("tr", right_cx - r, top_cy + r,     0)
        _corner_divs("bl", left_cx + r,  bottom_cy - r, 180)
        _corner_divs("br", right_cx - r, bottom_cy - r, 270)

    # ---- streetlights: small pole+lamp glyphs lining every sidewalk ----
    def _ensure_streetlight_points(self, h_roads, v_roads):
        if self._streetlight_points is not None:
            return
        from mobility.road_network import SIDEWALK_OFFSET_M
        xmin, xmax = float(self.area["xmin"]), float(self.area["xmax"])
        ymin, ymax = float(self.area["ymin"]), float(self.area["ymax"])
        pts = []
        for road_y in h_roads:
            n = max(1, int((xmax - xmin) / self.STREETLIGHT_SPACING_M))
            for i in range(1, n):
                pts.append((xmin + i * (xmax - xmin) / n, road_y + SIDEWALK_OFFSET_M))
        for road_x in v_roads:
            n = max(1, int((ymax - ymin) / self.STREETLIGHT_SPACING_M))
            for i in range(1, n):
                pts.append((road_x + SIDEWALK_OFFSET_M, ymin + i * (ymax - ymin) / n))
        self._streetlight_points = pts

    def _draw_streetlights(self, h_roads, v_roads):
        self._ensure_streetlight_points(h_roads, v_roads)
        if not self._streetlight_items:
            for _ in self._streetlight_points:
                pole = self.cv.create_line(0, 0, 0, 0, fill=self.STREETLIGHT_POLE_COLOR,
                                            width=2, tags=("streetlight",))
                lamp = self.cv.create_oval(0, 0, 0, 0, fill=self.STREETLIGHT_LAMP_COLOR,
                                            outline="", tags=("streetlight",))
                self._streetlight_items.append((pole, lamp))
        for (x, y), (pole, lamp) in zip(self._streetlight_points, self._streetlight_items):
            cx, cy = self.ly.sx(x), self.ly.sy(y)
            top_y = cy - self.STREETLIGHT_POLE_PX
            self.cv.coords(pole, cx, cy, cx, top_y)
            lr = self.STREETLIGHT_LAMP_R_PX
            self.cv.coords(lamp, cx - lr, top_y - lr, cx + lr, top_y + lr)


# Fill colors for the small kind glyph / tier badge drawn alongside each
# device (used as a fallback shape color, and as a tier-color badge next
# to real icons so the tier legend still means something visually).
KIND_ACCENT = {
    "vehicle": "#1f2937",
    "uav": "#0ea5e9",
    "pedestrian": "#f59e0b",
    "stationary": "#6b7280",
}

_DEVICE_ICON_SIZE = {
    "vehicle": 28,
    "uav": 26,
    "pedestrian": 22,
    "stationary": 20,
}


class DeviceRenderer:
    """
    Draw devices using real icons (car/uav_workload/man+woman/edge) when
    PIL + assets are available, colored by a small tier badge; falls back
    to plain vector shapes otherwise. Kept on the 'dev' tag.
    """
    # Specific DEVICE_PROFILES keys (config/devices_config.py) that got a
    # dedicated icon uploaded to assets/, keyed by entity.device_type --
    # takes priority over the generic per-kind icon/shape below. Any
    # device_type not listed here (e.g. connected_car, drone) keeps its
    # existing generic-kind rendering.
    _TYPE_ICON = {
        "traffic_controller": "traffic-lights",
        "smart_camera": "cctv",
        "smart_meter": "electric-meter",
        "public_transport_bus": "school-bus",
        "emergency_vehicle": "ambulance",
        "city_gateway": "gateway",
        "smart_parking_sensor": "parking",
        "street_light_controller": "street-lamp",
        "environment_sensor": "sprout",
    }
    _TYPE_ICON_SIZE = {
        "traffic_controller": 22,
        "smart_camera": 20,
        "smart_meter": 20,
        "public_transport_bus": 32,
        "emergency_vehicle": 28,
        "city_gateway": 22,
        "smart_parking_sensor": 20,
        "street_light_controller": 22,
        "environment_sensor": 20,
    }

    def __init__(self, canvas: tk.Canvas, layout: Layout, color_fn, assets: Assets = None):
        self.cv = canvas
        self.ly = layout
        self.color_fn = color_fn
        self.assets = assets
        self._use_icons = bool(_HAS_PIL and assets is not None)
        # entity_id -> {"kind":..., "primary": id, "secondary": id|None, "badge": id}
        self.items: Dict[int, dict] = {}

    @staticmethod
    def _extract_xy(ent) -> Tuple[Optional[float], Optional[float]]:
        pos = getattr(ent, "position", None)
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            return float(pos[0]), float(pos[1])
        if isinstance(pos, dict):
            if "x" in pos and "y" in pos: return float(pos["x"]), float(pos["y"])
            if "X" in pos and "Y" in pos: return float(pos["X"]), float(pos["Y"])
        if hasattr(ent, "x") and hasattr(ent, "y"):
            return float(getattr(ent, "x")), float(getattr(ent, "y"))
        return None, None

    def _compute_positions(self, entities) -> Dict[int, Tuple[float,float]]:
        """
        Return {entity_id: (cx,cy)} with tiny anti-overlap spreading.
        Entities locked to an axis (vehicles on a road, pedestrians on a
        sidewalk) only ever get spread along that same axis, so they never
        visually drift off the road/sidewalk line they're actually on;
        unconstrained entities (UAVs, stationary) get the old omnidirectional
        circular jitter.
        """
        pos: Dict[int, Tuple[float,float]] = {}
        buckets: Dict[Tuple[int,int], List[Tuple[int,float,float,Optional[str]]]] = {}
        for e in entities or []:
            ent_id = getattr(e, "id", None)
            x, y = self._extract_xy(e)
            if ent_id is None or x is None or y is None:
                continue
            cx, cy = self.ly.sx(float(x)), self.ly.sy(float(y))
            ix = int(round(cx/GRID_SNAP))*GRID_SNAP
            iy = int(round(cy/GRID_SNAP))*GRID_SNAP
            axis = getattr(getattr(e, "mobility", None), "axis", None)
            buckets.setdefault((ix,iy), []).append((ent_id, cx, cy, axis))
        for (ix,iy), items in buckets.items():
            if len(items) == 1:
                eid,cx,cy,_axis = items[0]; pos[eid]=(cx,cy)
            else:
                n = len(items)
                for j,(eid,cx,cy,axis) in enumerate(items):
                    spread = (j - (n - 1) / 2.0) * (GRID_SNAP * 2.0)
                    if axis == "h":
                        pos[eid] = (cx + spread, cy)
                    elif axis == "v":
                        pos[eid] = (cx, cy + spread)
                    else:
                        R = 6.0
                        ang = 2*math.pi*j/n
                        pos[eid] = (ix + R*math.cos(ang), iy + R*math.sin(ang))
        return pos

    @staticmethod
    def _kind_of(ent) -> str:
        return str(getattr(ent, "mobility_kind", "stationary") or "stationary")

    @staticmethod
    def _direction_of(ent) -> int:
        d = getattr(getattr(ent, "mobility", None), "direction", 1)
        return 1 if d is None or d >= 0 else -1

    @staticmethod
    def _directional_angle_flip(axis: str, direction: int) -> Tuple[float, bool]:
        """
        (angle, flip) that reorients a default-facing-right single-sprite
        icon to face the given axis/direction -- shared by every icon that
        isn't pre-rendered per direction (pedestrians, and any device-type
        icon like ambulance/school-bus). Empirically verified against this
        app's Assets.get(): angle=-90 -> up, angle=+90 -> down, flip alone
        -> left.
        """
        if axis == "v":
            return (-90.0, False) if direction > 0 else (90.0, False)
        return (0.0, False) if direction > 0 else (0.0, True)

    def _icon_for(self, ent_id: int, kind: str, axis: str, direction: int, device_type: str = ""):
        type_icon = self._TYPE_ICON.get(device_type)
        if type_icon is not None:
            size = self._TYPE_ICON_SIZE.get(device_type, _DEVICE_ICON_SIZE.get(kind, 24))
            if kind == "vehicle":
                # Single default-facing-right sprite (ambulance/school-bus),
                # not pre-rendered per direction like the generic car art.
                angle, flip = self._directional_angle_flip(axis, direction)
                return self.assets.get(type_icon, size, angle=angle, flip=flip)
            # Stationary device-type icons (traffic lights/CCTV/meter) never
            # rotate -- they don't face a direction of travel.
            return self.assets.get(type_icon, size)
        if kind == "vehicle":
            # Pre-oriented directional art (car_l/r/t/d) -- no flip/rotate
            # guessing needed, each file is already drawn facing that way.
            if axis == "v":
                name = "car_t" if direction > 0 else "car_d"
            else:
                name = "car_r" if direction > 0 else "car_l"
            return self.assets.get(name, _DEVICE_ICON_SIZE["vehicle"]) or \
                self.assets.get("car", _DEVICE_ICON_SIZE["vehicle"])
        if kind == "uav":
            return self.assets.get("uav_workload", _DEVICE_ICON_SIZE["uav"])
        if kind == "pedestrian":
            # man/woman source art both face right; derive left/up/down by
            # mirroring/rotating.
            name = "man" if (ent_id % 2 == 0) else "woman"
            angle, flip = self._directional_angle_flip(axis, direction)
            return self.assets.get(name, _DEVICE_ICON_SIZE["pedestrian"], angle=angle, flip=flip)
        # Stationary sensors/gateways with no dedicated icon: use the
        # compact plain-shape marker below instead.
        return None

    # ---- icon-backed rendering ----
    def _make_icon_entry(self, ent_id, kind, axis, direction, cx, cy, device_type=""):
        img = self._icon_for(ent_id, kind, axis, direction, device_type)
        if img is None:
            return None
        primary = self.cv.create_image(cx, cy, image=img, tags=("dev",))
        badge = self.cv.create_oval(cx + 6, cy + 6, cx + 11, cy + 11, fill="#888888", outline="#ffffff", tags=("dev",))
        entry = {"kind": kind, "axis": axis, "direction": direction, "device_type": device_type,
                 "primary": primary, "secondary": None, "badge": badge, "icon": True}
        self.items[ent_id] = entry
        return entry

    def _move_icon_entry(self, ent_id, entry, kind, axis, direction, cx, cy, device_type=""):
        if entry["axis"] != axis or entry.get("direction") != direction:
            img = self._icon_for(ent_id, kind, axis, direction, device_type)
            if img is not None:
                self.cv.itemconfig(entry["primary"], image=img)
            entry["axis"] = axis
            entry["direction"] = direction
        self.cv.coords(entry["primary"], cx, cy)
        self.cv.coords(entry["badge"], cx + 6, cy + 6, cx + 11, cy + 11)

    # ---- plain-shape fallback (no PIL / no assets) ----
    def _make_shapes(self, ent_id: int, kind: str, cx: float, cy: float):
        if kind == "vehicle":
            secondary = self.cv.create_rectangle(cx - 6, cy - 4, cx + 6, cy + 4,
                                                  fill="#111827", outline="", tags=("dev",))
            primary = self.cv.create_rectangle(cx - 5, cy - 3, cx + 5, cy + 3,
                                                fill="#888888", outline="", tags=("dev",))
        elif kind == "uav":
            secondary = self.cv.create_oval(cx - 9, cy - 9, cx + 9, cy + 9,
                                             outline=KIND_ACCENT["uav"], width=1, tags=("dev",))
            primary = self.cv.create_polygon(cx, cy - 5, cx + 5, cy, cx, cy + 5, cx - 5, cy,
                                              fill="#888888", outline="", tags=("dev",))
        elif kind == "pedestrian":
            secondary = self.cv.create_line(cx, cy + 2, cx, cy + 7,
                                             fill=KIND_ACCENT["pedestrian"], width=2, tags=("dev",))
            primary = self.cv.create_oval(cx - 3, cy - 5, cx + 3, cy + 1,
                                           fill="#888888", outline="", tags=("dev",))
        else:  # stationary
            secondary = self.cv.create_rectangle(cx - 6, cy - 6, cx + 6, cy + 6,
                                                  outline=KIND_ACCENT["stationary"], width=1, tags=("dev",))
            primary = self.cv.create_rectangle(cx - 4, cy - 4, cx + 4, cy + 4,
                                                fill="#888888", outline="", tags=("dev",))
        self.items[ent_id] = {"kind": kind, "primary": primary, "secondary": secondary, "icon": False}

    def _move_shapes(self, entry: dict, kind: str, cx: float, cy: float, axis: str):
        primary, secondary = entry["primary"], entry["secondary"]
        if kind == "vehicle":
            if axis == "v":
                self.cv.coords(secondary, cx - 4, cy - 6, cx + 4, cy + 6)
                self.cv.coords(primary, cx - 3, cy - 5, cx + 3, cy + 5)
            else:
                self.cv.coords(secondary, cx - 6, cy - 4, cx + 6, cy + 4)
                self.cv.coords(primary, cx - 5, cy - 3, cx + 5, cy + 3)
        elif kind == "uav":
            self.cv.coords(secondary, cx - 9, cy - 9, cx + 9, cy + 9)
            self.cv.coords(primary, cx, cy - 5, cx + 5, cy, cx, cy + 5, cx - 5, cy)
        elif kind == "pedestrian":
            self.cv.coords(secondary, cx, cy + 2, cx, cy + 7)
            self.cv.coords(primary, cx - 3, cy - 5, cx + 3, cy + 1)
        else:
            self.cv.coords(secondary, cx - 6, cy - 6, cx + 6, cy + 6)
            self.cv.coords(primary, cx - 4, cy - 4, cx + 4, cy + 4)

    def draw(self, entities: List[object]):
        if not entities:
            return
        positions = self._compute_positions(entities)
        for e in entities:
            ent_id = getattr(e, "id", None)
            if ent_id is None or ent_id not in positions:
                continue
            cx, cy = positions[ent_id]
            kind = self._kind_of(e)
            axis = str(getattr(getattr(e, "mobility", None), "axis", "h") or "h")
            direction = self._direction_of(e)
            device_type = str(getattr(e, "device_type", "") or "")
            fill = self.color_fn(e)

            entry = self.items.get(ent_id)
            if entry is not None and entry.get("kind") != kind:
                self.cv.delete(entry["primary"])
                if entry.get("secondary") is not None:
                    self.cv.delete(entry["secondary"])
                if entry.get("badge") is not None:
                    self.cv.delete(entry["badge"])
                entry = None

            # Only take the icon path for a pre-existing entry if it really
            # is an icon entry -- kinds whose _icon_for() can return None
            # (stationary) persist as plain-shape entries across frames, and
            # those lack the "axis"/"direction" keys _move_icon_entry() needs.
            if self._use_icons and (entry is None or entry.get("icon")):
                if entry is None:
                    entry = self._make_icon_entry(ent_id, kind, axis, direction, cx, cy, device_type)
                if entry is not None:
                    self._move_icon_entry(ent_id, entry, kind, axis, direction, cx, cy, device_type)
                    self.cv.itemconfig(entry["badge"], fill=fill)
                    continue
                # fall through to plain shapes if icon creation failed

            if entry is None:
                self._make_shapes(ent_id, kind, cx, cy)
                entry = self.items[ent_id]
            else:
                self._move_shapes(entry, kind, cx, cy, axis)
            self.cv.itemconfig(entry["primary"], fill=fill)

    def _cloud_draw_anchor(self):
        """Canvas coords for Cloud icon (top-right corner, inside the ring)."""
        w = max(1, self.cv.winfo_width())
        return (w - CLOUD_MARGIN_PX, CLOUD_MARGIN_PX)
