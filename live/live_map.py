# live/live_map.py
import tkinter as tk
from tkinter import ttk
import math
from threading import Thread
from typing import Dict

try:
    from config import MOBILITY_CONFIG, SIMULATION
except Exception:
    MOBILITY_CONFIG = {"area": {"xmin": 0.0, "xmax": 10000.0, "ymin": 0.0, "ymax": 10000.0}}
    SIMULATION = {}
try:
    from config import DEVICE_TYPE_PROFILES
except Exception:
    DEVICE_TYPE_PROFILES = {}

from .tiers import TIER_COLORS, tier_from_server_name, norm_tier_token
from .renderers import (
    PAD, CLOUD_MARGIN_PX, Assets, Layout, ServerRenderer, DeviceRenderer, RoadRenderer, KIND_ACCENT,
    SkyBackground, BuildingRenderer, GroundRenderer,
)

BOTTOM_H = 160  # px for the bottom info bar

# ============================================================================
# LINE / LINK VISUAL SETTINGS -- everything about how the two kinds of
# on-map connection lines look lives here, in one place, so either can be
# retuned without hunting through the drawing methods below.
#
#   1) Backhaul arrows (_draw_links): solid, arrowed lines between a
#      MEC/UAV_BS wireless point and the Metro/Cloud tier it's overflowing
#      to. Colored strictly by TIER_COLORS[destination tier] -- green is
#      used ONLY when the destination is Cloud, so "green" on the map always
#      means a Cloud hop, never a MEC<->Metro or MEC<->UAV_BS relationship.
#   2) Device->AP dotted lines (_draw_processing_links): the standing
#      "attached wireless point" link from every source device to its
#      serving MEC/UAV_BS. Always dotted, and always resting-gray/active-
#      amber -- it never turns Cloud-green, even when that device's task is
#      actually executing at Cloud this round, so green stays reserved for
#      the backhaul arrows above.
# ============================================================================

# ---- 1) Backhaul arrows (MEC/UAV_BS <-> Metro/Cloud) ----
BACKHAUL_ARROW = tk.LAST  # arrowhead at the destination (Metro/Cloud) end
# Link width scaling: a link carrying about one typical task's worth of
# data (LINK_WIDTH_BASELINE_KB) stays a hairline; width grows by one pixel
# per doubling of data volume beyond that, capped so heavy workloads don't
# turn the arrow into a solid block.
LINK_WIDTH_BASELINE_KB = 500.0
LINK_WIDTH_MAX = 8

# ---- 2) Device -> attached-AP dotted line ----
PROC_LINK_DASH = (2, 3)
PROC_LINK_BASE_COLOR = "#9ca3af"     # resting (no active task this round)
PROC_LINK_ACTIVE_COLOR = "#f59e0b"   # highlight while actively offloading
PROC_LINK_FADE_STEP = 0.12           # per-tick fade back toward resting color
PROC_LINK_WIDTH_ACTIVE = 2
PROC_LINK_WIDTH_RESTING = 1
PROC_LINK_WIDTH_LEVEL_THRESHOLD = 0.35  # fade level above which width bumps up


def _blend_hex(c1: str, c2: str, t: float) -> str:
    """Linear-blend two '#rrggbb' colors; t=0 -> c1, t=1 -> c2."""
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = round(r1 + (r2 - r1) * t)
    g = round(g1 + (g2 - g1) * t)
    b = round(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


class LiveMap:
    """
    Two-phase UI:
      • SETUP: drag servers, positions persist live
      • RUN:   Start button disables dragging and starts simulation

    - No top HUD.
    - Draw order: sky → buildings → roads → devices → links → servers (servers always on top).
    - Footer is the single dashboard zone: counts ("legend") + utilization gauges.
    """
    def __init__(self, simulator, refresh_ms: int = 200, auto_start: bool = False, close_on_complete: bool = False, logger=print):
        self.sim = simulator
        self.refresh_ms = refresh_ms
        self.mode = "SETUP"
        self.auto_start = bool(auto_start)
        self.close_on_complete = bool(close_on_complete)
        self.logger = logger
        self._last_logged_round = None
        self.model_name = str(getattr(getattr(self.sim, "scheduler", None), "model_name", "") or "model").strip().lower()

        self.root = tk.Tk()
        self.root.title(f"SmartCityEdgeSimPy - Live Map - {self.model_name}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # --- Layout: top map + bottom dashboard ---
        self.top = tk.Frame(self.root, bd=0)
        self.bot = tk.Frame(self.root, bd=0, height=BOTTOM_H, bg="#ffffff")
        self.top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.bot.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas = tk.Canvas(self.top, bg="#ffffff", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # bounds
        area = MOBILITY_CONFIG.get("area", {})
        self.xmin = float(area.get("xmin", 0.0)); self.xmax = float(area.get("xmax", 10000.0))
        self.ymin = float(area.get("ymin", 0.0)); self.ymax = float(area.get("ymax", 10000.0))

        # renderers
        self.assets  = Assets(base_dir=self._root_dir())
        self.layout  = Layout(self.canvas, self.xmin, self.xmax, self.ymin, self.ymax)
        _area = {"xmin": self.xmin, "xmax": self.xmax, "ymin": self.ymin, "ymax": self.ymax}
        self.sky      = SkyBackground(self.canvas)
        self.groundRen = GroundRenderer(self.canvas, self.layout, _area)
        self.buildRen = BuildingRenderer(self.canvas, self.layout, _area, self.assets)
        self.roadRen  = RoadRenderer(self.canvas, self.layout, _area, self.assets)
        self.srvRen   = ServerRenderer(self.canvas, self.layout, self.assets)
        self.devRen   = DeviceRenderer(self.canvas, self.layout, self._device_color, self.assets)
        self.srvRen.bind_handlers()
        self.srvRen.enable_drag(True)  # setup mode

        # footer dashboard
        self._build_footer()

        # link visuals
        self.link_items: Dict[tuple,int] = {}   # {(MEC_name, dest_name): line_id}
        self.dash_phase = 0

        self._closed = False
        self._completed = False
        self._user_closed = False
        self._sim_thread = None
        self._tick_after_id = None
        self._redraw_pending = False

        self.link_text_items: Dict[tuple, int] = {}  # (MEC,dest) -> text id
        self.link_text_bg:    Dict[tuple, int] = {}  # (MEC,dest) -> rect id

        # Per-device dotted "attached wireless AP" link (always drawn), which
        # highlights and fades when the device is actively offloading.
        self.proc_link_items: Dict[int, int] = {}   # entity_id -> line item id
        self.proc_link_fade: Dict[int, float] = {}  # entity_id -> highlight level, 1.0 (active) fading to 0.0


    # ---------- footer / dashboard ----------
    def _build_footer(self):
        # Left: controls
        controls = tk.Frame(self.bot, bg="#ffffff")
        controls.pack(side=tk.LEFT, padx=(12, 6), pady=8)

        self.start_btn = tk.Button(controls, text="▶ Start simulation", width=18, command=self._start_clicked)
        self.start_btn.grid(row=0, column=0, padx=(0, 8))

        # Default to "by BS": each device's badge is colored by its specific
        # attached base station, so a handover (switching MEC_1 -> MEC_2)
        # is visible as a badge color change, not just a tier color.
        self.color_mode = "server"
        self.color_btn = tk.Button(controls, text="Color: by BS (press C)", width=24, command=self._toggle_color_mode)
        self.color_btn.grid(row=0, column=1, padx=(0, 8))

        self.status_var = tk.StringVar(value=f"Model: {self.model_name} | Round: 0 / ? | Ready")
        self.status_lbl = tk.Label(
            controls,
            textvariable=self.status_var,
            font=("Segoe UI", 10, "bold"),
            bg="#ffffff",
            anchor="w",
        )
        self.status_lbl.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        # vertical separator
        ttk.Separator(self.bot, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=8)

        # Middle: COUNTS (legend) — horizontal row
        counts = tk.Frame(self.bot, bg="#ffffff")
        counts.pack(side=tk.LEFT, padx=10, pady=10)

        tk.Label(counts, text="Legend", font=("Segoe UI", 11, "bold"), bg="#ffffff").grid(row=0, column=0, columnspan=10, sticky="w", pady=(0, 4))

        def dot(parent, color):
            c = tk.Canvas(parent, width=14, height=14, bg="#ffffff", highlightthickness=0)
            c.create_oval(1, 1, 13, 13, fill=color, outline="")
            return c

        # MEC ● 0   |  Metro ● 0   |  Cloud ● 0
        self.dot_MEC = dot(counts, TIER_COLORS["MEC"]);  self.dot_MEC.grid(row=1, column=0, padx=(0,6))
        self.MEC_lbl = tk.Label(counts, text="MEC devices: 0", bg="#ffffff"); self.MEC_lbl.grid(row=1, column=1, padx=(0,12))
        self.dot_pea  = dot(counts, TIER_COLORS["Metro"]); self.dot_pea.grid(row=1, column=2, padx=(0,6))
        self.pea_lbl  = tk.Label(counts, text="Metro devices: 0", bg="#ffffff"); self.pea_lbl.grid(row=1, column=3, padx=(0,12))
        self.dot_cld  = dot(counts, TIER_COLORS["Cloud"]); self.dot_cld.grid(row=1, column=4, padx=(0,6))
        self.cld_lbl  = tk.Label(counts, text="Cloud devices: 0", bg="#ffffff"); self.cld_lbl.grid(row=1, column=5)
        self.dot_uav  = dot(counts, TIER_COLORS["UAV_BS"]); self.dot_uav.grid(row=1, column=6, padx=(12,6))
        self.uav_lbl  = tk.Label(counts, text="UAV_BS devices: 0", bg="#ffffff"); self.uav_lbl.grid(row=1, column=7)
        self.dot_access = dot(counts, TIER_COLORS["ACCESS_POINT"]); self.dot_access.grid(row=1, column=8, padx=(12,6))
        self.access_lbl = tk.Label(counts, text="Access points: 0", bg="#ffffff"); self.access_lbl.grid(row=1, column=9)

        # Kinds row: one entry per actual DEVICE_PROFILES type (not just the
        # coarse vehicle/uav/pedestrian/stationary kind), each with the same
        # real map icon DeviceRenderer draws for it -- a dedicated icon
        # (ambulance/school-bus/traffic-lights/cctv/electric-meter) where
        # one exists, else the generic per-kind icon/shape fallback. A
        # type's row is only gridded once at least one device of that type
        # is actually active (see _relayout_kind_rows), so the legend only
        # ever lists device labels that truly exist in this run, not a
        # fixed list padded with permanent zeros.
        def glyph(parent, kind):
            c = tk.Canvas(parent, width=16, height=16, bg="#ffffff", highlightthickness=0)
            color = KIND_ACCENT.get(kind, "#6b7280")
            if kind == "vehicle":
                c.create_rectangle(2, 5, 14, 11, fill=color, outline="")
            elif kind == "uav":
                c.create_oval(1, 1, 15, 15, outline=color, width=1)
                c.create_polygon(8, 4, 12, 8, 8, 12, 4, 8, fill=color, outline="")
            elif kind == "pedestrian":
                c.create_line(8, 9, 8, 14, fill=color, width=2)
                c.create_oval(5, 2, 11, 8, fill=color, outline="")
            else:
                c.create_rectangle(3, 3, 13, 13, outline=color, width=2)
            return c

        _KIND_ICON_ASSET = {"vehicle": ("car_r", 22), "uav": ("uav_workload", 22), "pedestrian": ("man", 20)}
        self._kind_icon_imgs = {}  # keep PhotoImage refs alive (Tk drops images with no live reference)

        def kind_icon(parent, kind):
            spec = _KIND_ICON_ASSET.get(kind)
            img = self.assets.get(*spec) if (spec and self.assets is not None) else None
            if img is None:
                return glyph(parent, kind)
            self._kind_icon_imgs[kind] = img
            return tk.Label(parent, image=img, bg="#ffffff")

        def type_icon(parent, device_type, kind):
            icon_name = DeviceRenderer._TYPE_ICON.get(device_type)
            if icon_name and self.assets is not None:
                size = DeviceRenderer._TYPE_ICON_SIZE.get(device_type, 22)
                img = self.assets.get(icon_name, size)
                if img is not None:
                    self._kind_icon_imgs[device_type] = img
                    return tk.Label(parent, image=img, bg="#ffffff")
            return kind_icon(parent, kind)

        def _pretty_type(device_type):
            return device_type.replace("_", " ").strip().title()

        tk.Label(counts, text="Kinds", font=("Segoe UI", 11, "bold"), bg="#ffffff").grid(row=2, column=0, columnspan=8, sticky="w", pady=(8, 4))
        self._KINDS_ROW0 = 3          # first grid row the per-type entries occupy
        self._KINDS_COLS_PER_ROW = 4  # icon+label pairs per row (2 grid columns each)
        self._device_types_ordered = list(DEVICE_TYPE_PROFILES.keys()) or ["generic"]
        self.kind_lbls: Dict[str, tk.Label] = {}
        self._kind_icon_widgets: Dict[str, tk.Widget] = {}
        self._kind_display_names: Dict[str, str] = {}
        self._visible_device_types: list = []
        for device_type in self._device_types_ordered:
            profile = DEVICE_TYPE_PROFILES.get(device_type, {}) or {}
            kind = str(profile.get("mobility_kind", "stationary") or "stationary")
            icon_widget = type_icon(counts, device_type, kind)
            lbl = tk.Label(counts, text=f"{_pretty_type(device_type)}: 0", bg="#ffffff")
            self._kind_icon_widgets[device_type] = icon_widget
            self.kind_lbls[device_type] = lbl
            self._kind_display_names[device_type] = _pretty_type(device_type)
            # Not gridded yet -- _relayout_kind_rows() grids only the types
            # that actually have an active device, once counts are known.

        # A short, plain-language key for the two kinds of line drawn on the
        # map -- without it, the dotted device link and the solid colored
        # backhaul arrow just look like arbitrary lines to a new reader.
        def line_swatch(parent, dashed):
            c = tk.Canvas(parent, width=28, height=14, bg="#ffffff", highlightthickness=0)
            if dashed:
                c.create_line(2, 7, 26, 7, fill=PROC_LINK_ACTIVE_COLOR, width=2, dash=PROC_LINK_DASH)
            else:
                c.create_line(2, 7, 24, 7, fill=TIER_COLORS["Cloud"], width=2, arrow=tk.LAST)
            return c

        lines_row = self._KINDS_ROW0 + 10  # fixed, well past any realistic Kinds row count
        tk.Label(counts, text="Lines", font=("Segoe UI", 11, "bold"), bg="#ffffff").grid(row=lines_row, column=0, columnspan=8, sticky="w", pady=(8, 2))
        line_swatch(counts, dashed=True).grid(row=lines_row + 1, column=0, sticky="w")
        tk.Label(counts, text="Device -> wireless point (amber = active, green = cloud task)", bg="#ffffff", font=("Segoe UI", 8)).grid(row=lines_row + 1, column=1, columnspan=7, sticky="w")
        line_swatch(counts, dashed=False).grid(row=lines_row + 2, column=0, sticky="w")
        tk.Label(counts, text="Backhaul: MEC/UAV_BS -> Metro/Cloud (color = destination tier)", bg="#ffffff", font=("Segoe UI", 8)).grid(row=lines_row + 2, column=1, columnspan=7, sticky="w")

        # vertical separator
        ttk.Separator(self.bot, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=8)

        # Right: UTILIZATION gauges
        stats = tk.Frame(self.bot, bg="#ffffff")
        stats.pack(side=tk.LEFT, padx=10, pady=8)

        tk.Label(stats, text="Stats (avg per tier)", font=("Segoe UI", 11, "bold"), bg="#ffffff").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))
        self._gauges = {}  # key -> {'bar': (canvas,fill,width), 'lbl': Label}

        GAUGE_BAR_W = 110  # px -- narrower than before so the %-label to its right always stays on screen

        def mk_single_gauge(row, label, color):
            tk.Label(stats, text=label, bg="#ffffff").grid(row=row, column=0, sticky="w", padx=(0, 6), pady=1)
            c = tk.Canvas(stats, width=GAUGE_BAR_W, height=12, bg="#f3f4f6", highlightthickness=0)
            f = c.create_rectangle(0, 0, 0, 12, fill=color, outline="")
            c.grid(row=row, column=1, sticky="w", pady=1)
            v = tk.Label(stats, text="0%", width=4, bg="#ffffff"); v.grid(row=row, column=2, sticky="w", padx=(6, 0), pady=1)
            return {"bar": (c, f, GAUGE_BAR_W), "lbl": v}

        # One "Computational" bar per tier: CPU and MEM used to be shown as
        # two separate rows; now a single bar shows whichever of the two is
        # the tighter bottleneck (max(cpu%, mem%)) -- that's the resource
        # actually limiting the tier, and MEM sits near 0% on every tier in
        # this workload so showing it alongside CPU was mostly dead space.
        # On-device (OC) compute isn't a standing reservation like a
        # server's -- it's this round's instantaneous demand/capacity,
        # averaged over whichever devices actually processed locally (0%
        # when none did). Battery is shown as "used" to match the
        # Computational bars' sense (bar fills as it depletes). Ordered
        # on-device, MEC, UAV_BS, Metro, Cloud, battery. Single column (not
        # two side by side) so the right-hand bars never get clipped by the
        # window edge.
        self._gauges["OC"] = mk_single_gauge(1, "On-device Computational", "#0ea5e9")
        self._gauges["MEC"] = mk_single_gauge(2, "MEC Computational", TIER_COLORS["MEC"])
        self._gauges["UAV_BS"] = mk_single_gauge(3, "UAV_BS Computational", TIER_COLORS["UAV_BS"])
        self._gauges["Metro"] = mk_single_gauge(4, "Metro Computational", TIER_COLORS["Metro"])
        self._gauges["Cloud"] = mk_single_gauge(5, "Cloud Computational", TIER_COLORS["Cloud"])
        self._battery_gauge = mk_single_gauge(6, "Device Battery", "#111827")

        # keyboard toggle
        self.root.bind("c", self._toggle_color_mode)
        self.root.bind("C", self._toggle_color_mode)

    # ---------- controls ----------
    def _toggle_color_mode(self, _evt=None):
        if self.color_mode == "server":
            self.color_mode = "tier"
            self.color_btn.config(text="Color: by TIER (press C)")
        else:
            self.color_mode = "server"
            self.color_btn.config(text="Color: by BS (press C)")

    def _start_clicked(self):
        if self.mode == "RUN": return
        self.srvRen.enable_drag(False)    # lock layout
        self.start_btn.config(state="disabled")
        self.mode = "RUN"
        self._set_progress(round_id=0, total_rounds="?", event="starting")
        self._sim_thread = Thread(target=self._run_simulation, daemon=True, name="SimulatorThread")
        self._sim_thread.start()

    def _run_simulation(self):
        try:
            self.sim.run(train_mode=False, progress_callback=self._progress_callback)
            if not self._closed:
                self._safe_after(0, self._simulation_completed)
        except Exception as exc:
            if self._closed:
                return
            print("[LiveMap] simulation error:", exc)
            self._safe_after(0, lambda: self._set_progress(event="failed"))

    def _safe_after(self, delay_ms, callback):
        """self.root.after() called from the background SimulatorThread
        races _on_close's root.destroy() -- the _closed check right before
        each call narrows the window but can't close it, so swallow the
        failure if the root is already gone by the time this actually
        runs (that race is what produces the stray "main thread is not in
        main loop" / Tcl_AsyncDelete errors at shutdown)."""
        try:
            self.root.after(delay_ms, callback)
        except (RuntimeError, tk.TclError):
            pass

    def _simulation_completed(self):
        self._completed = True
        self._set_progress(event="completed")
        if self.close_on_complete and not self._closed:
            self.root.after(1200, self._on_close)

    def _progress_callback(self, **payload):
        round_id = int(payload.get("round_id", 0) or 0)
        total_rounds = int(payload.get("total_rounds", 0) or 0)
        event = str(payload.get("event", "round") or "round")
        if self._closed:
            return
        if event == "round_end" and total_rounds > 0:
            step = max(1, total_rounds // 10)
            should_log = round_id == 1 or round_id == total_rounds or round_id % step == 0
            if should_log and self._last_logged_round != round_id:
                self._last_logged_round = round_id
                try:
                    self.logger(f"[PROGRESS] {self.model_name} round {round_id}/{total_rounds}")
                except Exception:
                    pass
        self._safe_after(
            0,
            lambda: self._set_progress(round_id=round_id, total_rounds=total_rounds, event=event),
        )

    def _set_progress(self, round_id=None, total_rounds=None, event=None):
        current = getattr(self, "_current_round", 0)
        total = getattr(self, "_total_rounds", "?")
        if round_id is not None:
            current = int(round_id or 0)
            self._current_round = current
        if total_rounds not in (None, 0, ""):
            total = total_rounds
            self._total_rounds = total
        event_text = str(event or "running").replace("_", " ")
        self.status_var.set(f"Model: {self.model_name} | Round: {current} / {total} | {event_text}")

    # ---------- coloring ----------
    def _device_tier(self, ent) -> str:
        for attr in ("last_offload_tier", "offload_tier", "processing_tier"):
            v = getattr(ent, attr, None)
            if isinstance(v, str): return norm_tier_token(v)
        asrv = getattr(ent, "attached_server", None)
        if asrv is None: return "Unknown"
        t_attr = getattr(asrv, "tier", None)
        return norm_tier_token(t_attr) if isinstance(t_attr, str) \
               else tier_from_server_name(getattr(asrv, "name",""))

    def _device_color(self, ent) -> str:
        by_server = self.color_mode == "server"
        if by_server:
            # Color by whichever wireless point (MEC or UAV_BS) is actually
            # serving this device right now, not just its ground radio
            # attachment -- so the badge visibly flips color when a device's
            # workload is being carried by a UAV_BS instead of its MEC.
            ent_id = getattr(ent, "id", None)
            by_entity = getattr(self.sim, "live_processing_server_by_entity", {}) or {}
            asrv = by_entity.get(ent_id) if ent_id is not None else None
            if asrv is None:
                asrv = getattr(ent, "attached_server", None)
            if asrv is not None:
                k = f"{getattr(asrv,'name','srv')}#{getattr(asrv,'id',None) or id(asrv)}"
                return self.srvRen.server_colors.get(k, "#111827")
            return "#111827"
        return TIER_COLORS.get(self._device_tier(ent), "#6b7280")

    # ---------- resize & tick ----------
    def _on_canvas_resize(self, _evt):
        if not self._redraw_pending:
            self._redraw_pending = True
            self.root.after(0, self._redraw_now)

    def _redraw_now(self):
        self._redraw_pending = False
        self._draw_content()

    def _active_entities(self):
        m = getattr(self.sim, "mobility", None)
        entities = list(getattr(m, "entities", []) if m else [])
        active_count = getattr(self.sim, "live_active_device_count", None)
        if active_count is None:
            return entities
        return entities[:max(0, min(int(active_count), len(entities)))]
    def _draw_content(self):
        # AccessPoint objects are radio-only and deliberately do not belong to
        # ``sim.servers``.  Combine them only for map rendering; scheduler and
        # compute-utilization code continue to consume ``sim.servers`` alone.
        svs = list(getattr(self.sim, "servers", []) or [])
        access_points = list(getattr(self.sim, "access_points", []) or [])
        display_nodes = [*svs, *access_points]

        # 1) sky backdrop, 2) grass ground, 3) building skyline, 4) roads --
        # static/background layers
        self.sky.draw()
        self.groundRen.draw()
        self.buildRen.draw()
        self.roadRen.draw()
        # 3b) satellite-IoT service zones + LEO satellite marker
        self._draw_zones()

        # 4) assign server colors first -- devices color themselves by their
        # attached base station, so the palette must exist before they draw.
        self.srvRen.assign_palette(display_nodes)

        # 5) devices (vehicles on roads, UAVs/pedestrians/stationary elsewhere)
        self.devRen.draw(self._active_entities())

        # 6) dotted device -> processing-server links, then animated
        # MEC->(Metro/Cloud) backhaul links
        self._draw_processing_links()
        self._draw_links()

        # 7) servers last (always on top)
        self.srvRen.draw(display_nodes)
        self.canvas.tag_raise("srv")

        # footer updates
        self._update_counts()
        self._update_kind_counts()
        self._update_gauges()

    def _draw_zones(self):
        """Tint the cellular / UAV / remote service zones and mark the LEO satellite."""
        try:
            from config import zones as zone_config
        except Exception:
            return
        self.canvas.delete("zone")
        if not zone_config.ZONES_ENABLED:
            return
        L = self.layout
        for zone_id in zone_config.ZONE_ORDER:
            zone = zone_config.ZONES[zone_id]
            x0, x1, y0, y1 = zone["extent_m"]
            cx0, cy0, cx1, cy1 = L.sx(x0), L.sy(y1), L.sx(x1), L.sy(y0)
            self.canvas.create_rectangle(
                cx0, cy0, cx1, cy1, fill=zone.get("color", "#e5e7eb"), stipple="gray50",
                outline="#64748b", dash=(6, 4), width=2, tags=("zone",),
            )
            self.canvas.create_text(
                (cx0 + cx1) / 2.0, cy0 + 14, text=zone.get("label", zone_id),
                font=("Segoe UI", 10, "bold"), fill="#334155", tags=("zone",),
            )
        if getattr(self.sim, "satellite_enabled", False):
            # LEO satellite glyph above the remote zone, with live pass elevation.
            x0, x1, y0, y1 = zone_config.ZONES["remote"]["extent_m"]
            cx, cy = (L.sx(x0) + L.sx(x1)) / 2.0, L.sy(y1) + 44
            c = self.canvas
            c.create_rectangle(cx - 7, cy - 6, cx + 7, cy + 6, fill="#f59e0b", outline="#92400e", tags=("zone",))
            c.create_rectangle(cx - 27, cy - 4, cx - 9, cy + 4, fill="#1d4ed8", outline="#1e3a8a", tags=("zone",))
            c.create_rectangle(cx + 9, cy - 4, cx + 27, cy + 4, fill="#1d4ed8", outline="#1e3a8a", tags=("zone",))
            elev = float(getattr(self.sim, "sat_elevation_deg", 0.0) or 0.0)
            slant_km = float(getattr(self.sim, "sat_slant_range_m", 0.0) or 0.0) / 1000.0
            c.create_text(cx, cy + 18, text=f"LEO-600  elev {elev:.0f}\u00b0  {slant_km:.0f} km",
                          font=("Segoe UI", 9), fill="#334155", tags=("zone",))
        # Keep the tint above the static city layers but below devices/servers.
        for layer in ("streetlight", "crossing", "sidewalk", "road", "building", "ground", "sky"):
            if self.canvas.find_withtag(layer):
                self.canvas.tag_raise("zone", layer)
                break

    def _draw_links(self):
        """
        Draw MEC/Metro/Cloud links.
        UI-only change: any server whose name starts with "Cloud" is drawn
        at the top-right corner of the canvas (inside PAD), WITHOUT changing
        the simulation coordinates used elsewhere.
        """
        alive = set()

        # ---- Map server name -> (cx, cy) in canvas space ----
        # (Cloud is overridden to a fixed on-screen anchor; others unchanged.)
        name_to_pos: Dict[str, tuple] = {}
        w = max(1, self.canvas.winfo_width())
        cloud_anchor = (w - CLOUD_MARGIN_PX, CLOUD_MARGIN_PX)  # top-right corner inside the ring

        for s in getattr(self.sim, "servers", []):
            x = getattr(s, "x", None)
            y = getattr(s, "y", None)
            if x is None or y is None:
                continue

            name = str(getattr(s, "name", "") or "")
            if name.startswith("Cloud"):
                cx, cy = cloud_anchor
            else:
                cx = self.layout.sx(float(x))
                cy = self.layout.sy(float(y))

            name_to_pos[name] = (cx, cy)

        # ---- Draw each published link ----
        for link in getattr(self.sim, "gui_links", []):
            edge = str(link.get("MEC", "") or "")
            dest = str(link.get("to", link.get("Metro", "")) or "")
            tier = str(link.get("tier", "Metro") or "Metro")
            # Backhaul color marks the DESTINATION tier, regardless of
            # whether the overflow originated at a ground MEC or an aerial
            # UAV_BS -- both MEC->Cloud and UAV_BS->Cloud overflow render
            # green (TIER_COLORS["Cloud"]), same as MEC->Metro/UAV_BS->Metro
            # both render Metro-orange. This matches the per-device dotted
            # line's rule (_draw_processing_links: green whenever the
            # device's current task is Cloud-bound, source-agnostic).
            color = TIER_COLORS.get(tier, TIER_COLORS.get("Unknown", "#6b7280"))

            if edge not in name_to_pos or dest not in name_to_pos:
                continue

            ex, ey = name_to_pos[edge]
            dx, dy = name_to_pos[dest]

            key = (edge, dest)
            total_kb = float(link.get("total_kb", 0.0) or 0.0)
            ratio = total_kb / LINK_WIDTH_BASELINE_KB
            width = 1 if ratio <= 1.0 else min(LINK_WIDTH_MAX, 1 + int(math.log2(ratio)))

            if key not in self.link_items:
                self.link_items[key] = self.canvas.create_line(
                    ex, ey, dx, dy, fill=color, width=width, arrow=BACKHAUL_ARROW
                )
            else:
                self.canvas.coords(self.link_items[key], ex, ey, dx, dy)
                self.canvas.itemconfig(self.link_items[key], fill=color, width=width)
            alive.add(key)

            # ----- Optional delay label if avg_delay_ms is provided -----
            avg_ms = link.get("avg_delay_ms", None)
            if avg_ms is None:
                if key in self.link_text_items:
                    self.canvas.delete(self.link_text_items[key])
                    self.link_text_items.pop(key, None)
                if key in self.link_text_bg:
                    self.canvas.delete(self.link_text_bg[key])
                    self.link_text_bg.pop(key, None)
                continue

            # Place small label at the midpoint, nudged off the line
            mx = (ex + dx) / 2.0
            my = (ey + dy) / 2.0
            vx, vy = (dx - ex), (dy - ey)
            L = math.hypot(vx, vy) or 1.0
            nx, ny = -vy / L, vx / L
            mx += nx * 10.0
            my += ny * 10.0

            label = f"{avg_ms:.0f} ms" if avg_ms < 1000 else f"{avg_ms/1000:.1f} s"

            if key not in self.link_text_items:
                txt = self.canvas.create_text(mx, my, text=label, font=("Segoe UI", 9), fill=color)
                x1, y1, x2, y2 = self.canvas.bbox(txt)
                bg = self.canvas.create_rectangle(x1 - 4, y1 - 2, x2 + 4, y2 + 2,
                                                fill="#ffffff", outline="#e5e7eb")
                self.canvas.tag_lower(bg, txt)
                self.link_text_items[key] = txt
                self.link_text_bg[key] = bg
            else:
                txt = self.link_text_items[key]
                self.canvas.itemconfig(txt, text=label, fill=color)
                self.canvas.coords(txt, mx, my)
                x1, y1, x2, y2 = self.canvas.bbox(txt)
                self.canvas.coords(self.link_text_bg[key], x1 - 4, y1 - 2, x2 + 4, y2 + 2)
                self.canvas.tag_lower(self.link_text_bg[key], txt)

        # ---- Prune stale lines/labels ----
        for key, item in list(self.link_items.items()):
            if key not in alive:
                self.canvas.delete(item)
                self.link_items.pop(key, None)
                if key in self.link_text_items:
                    self.canvas.delete(self.link_text_items[key])
                    self.link_text_items.pop(key, None)
                if key in self.link_text_bg:
                    self.canvas.delete(self.link_text_bg[key])
                    self.link_text_bg.pop(key, None)

    def _draw_processing_links(self):
        """
        Dotted line from every device to its currently attached wireless AP
        -- the standing cellular connection every device keeps whether or
        not it has a task in flight. The device's ground-radio attachment
        (ent.attached_server) is always a MEC (the handover model doesn't
        offer UAV_BS as an attachment candidate), so when a task IS
        actively being served by a UAV_BS this round (present in
        live_processing_server_by_entity, keyed off the task's actual/
        visual serving server -- see _visual_ap in core/simulator.py),
        that takes over as this frame's endpoint instead, so aerial-edge
        traffic is visible too. Either way the line highlights while
        active and fades back to resting gray over the following redraw
        ticks instead of snapping off.

        This line never turns Cloud-green, even when the device's task is
        actually executing at Cloud this round -- green is reserved for the
        MEC/UAV_BS<->Cloud backhaul arrows (_draw_links) so it always means
        "this hop reaches Cloud", not "this device's task ends up at Cloud
        two hops away."
        """
        entities = self._active_entities()
        by_entity = getattr(self.sim, "live_processing_server_by_entity", {}) or {}

        alive = set()
        for ent in entities:
            ent_id = getattr(ent, "id", None)
            if ent_id is None:
                continue
            is_active = ent_id in by_entity
            asrv = by_entity.get(ent_id) if is_active else getattr(ent, "attached_server", None)
            dx, dy = getattr(ent, "x", None), getattr(ent, "y", None)
            sx = getattr(asrv, "x", None) if asrv is not None else None
            sy = getattr(asrv, "y", None) if asrv is not None else None
            if dx is None or dy is None or sx is None or sy is None:
                continue

            dcx, dcy = self.layout.sx(float(dx)), self.layout.sy(float(dy))
            scx, scy = self.layout.sx(float(sx)), self.layout.sy(float(sy))

            prev_level = self.proc_link_fade.get(ent_id, 0.0)
            level = 1.0 if is_active else max(0.0, prev_level - PROC_LINK_FADE_STEP)
            self.proc_link_fade[ent_id] = level
            color = _blend_hex(PROC_LINK_BASE_COLOR, PROC_LINK_ACTIVE_COLOR, level)
            width = PROC_LINK_WIDTH_ACTIVE if level > PROC_LINK_WIDTH_LEVEL_THRESHOLD else PROC_LINK_WIDTH_RESTING

            if ent_id not in self.proc_link_items:
                self.proc_link_items[ent_id] = self.canvas.create_line(
                    dcx, dcy, scx, scy, fill=color, width=width, dash=PROC_LINK_DASH, tags=("proclink",)
                )
            else:
                item = self.proc_link_items[ent_id]
                self.canvas.coords(item, dcx, dcy, scx, scy)
                self.canvas.itemconfig(item, fill=color, width=width)
            alive.add(ent_id)

        for ent_id, item in list(self.proc_link_items.items()):
            if ent_id not in alive:
                self.canvas.delete(item)
                self.proc_link_items.pop(ent_id, None)
                self.proc_link_fade.pop(ent_id, None)

        self.canvas.tag_lower("proclink")
        self.canvas.tag_raise("proclink", "road")

    # ---------- footer data updates ----------
    def _update_counts(self):
        """
        Prefer simulator-published per-tier device counts for this round
        (gui_tier_counts). Fallback to attachment-based counting if not present.
        """
        access_count = len(getattr(self.sim, "access_points", []) or [])
        self.access_lbl.config(text=f"Access points: {access_count}")
        counts = getattr(self.sim, "gui_tier_counts", None)
        if isinstance(counts, dict) and {"MEC", "Metro", "Cloud"} <= set(counts.keys()):
            self.MEC_lbl.config(text=f"MEC devices: {int(counts.get('MEC',0))}")
            self.pea_lbl.config(text=f"Metro devices: {int(counts.get('Metro',0))}")
            self.cld_lbl.config(text=f"Cloud devices: {int(counts.get('Cloud',0))}")
            self.uav_lbl.config(text=f"UAV_BS devices: {int(counts.get('UAV_BS',0))}")
            return

        # Fallback: attachment-based
        tcounts: Dict[str,int] = {"MEC":0,"Metro":0,"Cloud":0,"UAV_BS":0}
        m = getattr(self.sim, "mobility", None)
        if not m:
            self.MEC_lbl.config(text="MEC devices: 0")
            self.pea_lbl.config(text="Metro devices: 0")
            self.cld_lbl.config(text="Cloud devices: 0")
            self.uav_lbl.config(text="UAV_BS devices: 0")
            return
        for e in self._active_entities():
            tier = self._device_tier(e)
            tcounts[tier] = tcounts.get(tier, 0) + 1
        self.MEC_lbl.config(text=f"MEC devices: {tcounts.get('MEC',0)}")
        self.pea_lbl.config(text=f"Metro devices: {tcounts.get('Metro',0)}")
        self.cld_lbl.config(text=f"Cloud devices: {tcounts.get('Cloud',0)}")
        self.uav_lbl.config(text=f"UAV_BS devices: {tcounts.get('UAV_BS',0)}")

    def _update_kind_counts(self):
        counts: Dict[str, int] = {}
        for e in self._active_entities():
            device_type = str(getattr(e, "device_type", "") or "generic")
            counts[device_type] = counts.get(device_type, 0) + 1

        for device_type, lbl in self.kind_lbls.items():
            name = self._kind_display_names.get(device_type, device_type)
            lbl.config(text=f"{name}: {counts.get(device_type, 0)}")

        # Only device types that actually have >=1 active device get a row,
        # in stable DEVICE_PROFILES order -- so the legend never lists a
        # label that isn't really present, and doesn't reshuffle types that
        # are already showing just because a later-appearing type joined.
        visible = [t for t in self._device_types_ordered if counts.get(t, 0) > 0]
        if visible != self._visible_device_types:
            self._relayout_kind_rows(visible)
            self._visible_device_types = visible

    def _relayout_kind_rows(self, visible_types):
        cols = self._KINDS_COLS_PER_ROW
        for idx, device_type in enumerate(visible_types):
            row = self._KINDS_ROW0 + idx // cols
            col = (idx % cols) * 2
            self._kind_icon_widgets[device_type].grid(row=row, column=col, padx=(0, 6), sticky="w")
            self.kind_lbls[device_type].grid(row=row, column=col + 1, padx=(0, 12), sticky="w")
        hidden = set(self._kind_icon_widgets) - set(visible_types)
        for device_type in hidden:
            self._kind_icon_widgets[device_type].grid_remove()
            self.kind_lbls[device_type].grid_remove()

    def _set_gauge(self, gauge, value, decimals=0):
        c, f, w = gauge["bar"]
        c.coords(f, 0, 0, w * max(0.0, min(value, 100.0)) / 100.0, 12)
        gauge["lbl"].config(text=f"{value:.{decimals}f}%")

    def _update_gauges(self):
        utils = self._compute_tier_utils()  # {'MEC':(cpu%,mem%), ...}
        for tier, (cpu, mem) in utils.items():
            g = self._gauges.get(tier)
            if g:
                # Cloud's capacity is calibrated ~1000x a MEC site (see
                # config.py's tier-sizing references), so its utilization
                # never rounds to a nonzero whole percent even while
                # actively serving tasks -- 2 decimals so it reads as
                # something other than a permanent "0%".
                decimals = 2 if tier == "Cloud" else 0
                self._set_gauge(g, max(cpu, mem), decimals=decimals)

        self._set_gauge(self._gauges["OC"], self._compute_device_local_compute_util())
        self._set_gauge(self._battery_gauge, self._compute_device_battery_util())

    def _compute_tier_utils(self):
        totals = {
            "MEC":  {"cpu_used":0.0,"cpu_tot":0.0,"mem_used":0.0,"mem_tot":0.0},
            "Metro": {"cpu_used":0.0,"cpu_tot":0.0,"mem_used":0.0,"mem_tot":0.0},
            "Cloud": {"cpu_used":0.0,"cpu_tot":0.0,"mem_used":0.0,"mem_tot":0.0},
            "UAV_BS": {"cpu_used":0.0,"cpu_tot":0.0,"mem_used":0.0,"mem_tot":0.0},
        }
        for s in getattr(self.sim, "servers", []):
            name = str(getattr(s, "name", ""))
            if   name.startswith("UAV_BS"): tier = "UAV_BS"
            elif name.startswith("MEC"):  tier = "MEC"
            elif name.startswith("Metro"): tier = "Metro"
            elif name.startswith("Cloud"): tier = "Cloud"
            else: continue
            cpu_cap = float(getattr(s, "cpu_capacity", 0.0) or 0.0)
            mem_cap = float(getattr(s, "memory_capacity", 0.0) or 0.0)
            cpu_used = max(0.0, cpu_cap - float(getattr(s, "available_cpu", 0.0) or 0.0))
            live_cpu_frac = float((getattr(self.sim, "live_server_cpu_util_by_name", {}) or {}).get(name, 0.0) or 0.0)
            cpu_used = max(cpu_used, live_cpu_frac * cpu_cap)
            mem_used = max(0.0, mem_cap - float(getattr(s, "available_memory", 0.0) or 0.0))
            totals[tier]["cpu_used"] += cpu_used
            totals[tier]["cpu_tot"]  += cpu_cap
            totals[tier]["mem_used"] += mem_used
            totals[tier]["mem_tot"]  += mem_cap

        out = {}
        for tier, t in totals.items():
            cpu = (t["cpu_used"]/t["cpu_tot"]*100.0) if t["cpu_tot"]>0 else 0.0
            mem = (t["mem_used"]/t["mem_tot"]*100.0) if t["mem_tot"]>0 else 0.0
            out[tier] = (cpu, mem)
        return out

    def _compute_device_battery_util(self) -> float:
        """Average battery drained (%) across mobile devices with a real
        onboard battery (battery_capacity_j > 0) -- devices with no
        battery model (capacity 0) are excluded rather than counted as 0%
        used, so they don't drag the average down artificially."""
        used_pct_sum = 0.0
        n = 0
        for e in self._active_entities():
            cap = float(getattr(e, "battery_capacity_j", 0.0) or 0.0)
            if cap <= 0.0:
                continue
            remaining = float(getattr(e, "battery_remaining_j", 0.0) or 0.0)
            used_pct_sum += max(0.0, min(1.0, 1.0 - remaining / cap)) * 100.0
            n += 1
        return (used_pct_sum / n) if n else 0.0

    def _compute_device_local_compute_util(self) -> float:
        """Average on-device CPU utilization (%) this round -- across
        whichever devices actually processed a task locally (see
        live_local_cpu_util_by_entity in core/simulator.py). Reads as 0%
        when no device is computing locally this round, since there's no
        standing reservation to average otherwise."""
        util_by_entity = getattr(self.sim, "live_local_cpu_util_by_entity", {}) or {}
        if not util_by_entity:
            return 0.0
        return 100.0 * sum(util_by_entity.values()) / len(util_by_entity)

    # ---------- utils ----------
    def _root_dir(self) -> str:
        import os
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def _on_close(self):
        if self._closed:
            return
        self._closed = True
        if not self._completed:
            self._user_closed = True
        if hasattr(self.sim, "request_stop"):
            self.sim.request_stop()
        # Wait for the background SimulatorThread to actually notice
        # stop_requested and return before tearing down Tk -- otherwise it
        # can still be mid-round when we destroy the root, then later call
        # self.root.after(...) (see _run_simulation/_progress_callback) on
        # an interpreter that's already gone. That's what produces the
        # "main thread is not in main loop" / Tcl_AsyncDelete spam at
        # process exit: stray Tk Variable objects finalizing late, after
        # (or on a different thread than) their own interpreter's teardown.
        sim_thread = getattr(self, "_sim_thread", None)
        if sim_thread is not None and sim_thread.is_alive():
            sim_thread.join(timeout=5.0)
        # Cancel the in-flight _tick timer explicitly -- destroy() does not
        # retract an already-queued after() event, so without this the next
        # scheduled tick can fire post-destroy and Tcl raises "invalid
        # command name" trying to invoke a callback whose owning widget is
        # already gone.
        tick_id = getattr(self, "_tick_after_id", None)
        if tick_id is not None:
            try:
                self.root.after_cancel(tick_id)
            except tk.TclError:
                pass
            self._tick_after_id = None
        try:
            self.root.destroy()
        except tk.TclError:
            pass
        # NOTE: deliberately no gc.collect() here -- `self` (this LiveMap,
        # holding every StringVar/widget) is still on the call stack, so
        # nothing in this object graph is actually collectible yet. The
        # real reap happens in run_live_map() once its `gui` reference is
        # dropped -- see the comment there.

    def _tick(self):
        if self._closed: return
        try:
            self._draw_content()
        except Exception as e:
            print("[LiveMap] frame error:", e)
        if not self._closed:
            self._tick_after_id = self.root.after(self.refresh_ms, self._tick)

    def _cloud_draw_anchor(self):
        """Canvas coords for Cloud endpoint (top-right corner, inside the ring)."""
        w = max(1, self.canvas.winfo_width())
        return (w - CLOUD_MARGIN_PX, CLOUD_MARGIN_PX)


    def start(self):
        self._tick()
        if self.auto_start:
            self.root.after(250, self._start_clicked)
        self.root.mainloop()



def run_live_map(sim, model_name=None, refresh_ms=100, logger=print):
    model = str(model_name or getattr(getattr(sim, "scheduler", None), "model_name", "") or "model").strip().lower()
    # Keep headless/regression construction on legacy_sparse, but make the
    # interactive map opt into the declared access layer before any round or
    # queue is started.  Explicitly dense simulators are left unchanged.
    interactive_profile = str(SIMULATION.get("interactive_deployment_profile", "") or "").strip()
    if interactive_profile and not getattr(sim, "access_points", None) and hasattr(sim, "activate_deployment_profile"):
        sim.activate_deployment_profile(interactive_profile)
        logger(
            f"[GUI] Activated deployment profile {interactive_profile}: "
            f"{len(getattr(sim, 'access_points', []))} access points."
        )
    logger(f"[GUI] Starting SmartCity live map for {model}.")
    gui = LiveMap(sim, refresh_ms=refresh_ms, auto_start=True, close_on_complete=True, logger=logger)
    gui.start()
    user_closed = bool(getattr(gui, "_user_closed", False))
    completed = bool(getattr(gui, "_completed", False))

    # The LiveMap's widget tree (StringVars etc.) forms reference cycles
    # (bound-method callbacks captured in Tk's command table close back
    # over `self`), so it isn't reclaimed by simple refcounting when `gui`
    # goes out of scope -- it waits for Python's cyclic GC. That collector
    # runs opportunistically on WHATEVER thread happens to allocate enough
    # objects next, which in a multi-model comparison run is the next
    # model's background SimulatorThread -- finalizing this (already
    # destroyed) interpreter's Tk Variables there is exactly what raises
    # "main thread is not in main loop" / Tcl_AsyncDelete. Dropping the
    # reference and collecting here, deterministically, on the main
    # thread, right after its own root.destroy() and before any other
    # model's thread exists, avoids that race entirely.
    del gui
    import gc
    gc.collect()

    if user_closed and not completed:
        logger("[GUI] Closed by user; stopping application.")
        raise SystemExit(0)
