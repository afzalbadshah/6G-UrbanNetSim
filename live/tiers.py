# live/tiers.py

TIER_COLORS = {
    "MEC":    "#2563eb",  # blue
    "ACCESS_POINT": "#0891b2",  # cyan (radio-only ground BS)
    "Metro":   "#db2777",  # magenta/pink
    "Cloud":   "#16a34a",  # green
    "UAV_BS":  "#7c3aed",  # purple (aerial edge)
    "SAT":     "#f59e0b",  # amber (LEO satellite -> gateway -> Cloud)
    "LEO_SAT": "#f59e0b",
    "Unknown": "#6b7280",  # gray
}

SERVER_PALETTE = [
    "#2563eb", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6",
    "#06b6d4", "#dc2626", "#22c55e", "#eab308", "#3b82f6",
    "#a855f7", "#f97316", "#14b8a6", "#84cc16", "#f43f5e",
]

def norm_tier_token(token: str) -> str:
    t = (token or "").strip().lower()
    if t in {"mec", "vehicular", "vehicularmec", "ve", "bsmec", "mecbs"}:
        return "MEC"
    if t in {"metro", "peas", "regional", "rc", "region"}:
        return "Metro"
    if t in {"cloud", "cc"}:
        return "Cloud"
    if t in {"uav_bs", "uavbs", "uav", "air"}:
        return "UAV_BS"
    if t in {"sat", "leo", "leo_sat", "satellite"}:
        return "SAT"
    return "Unknown"


def tier_from_server_name(name: str) -> str:
    n = (name or "")
    if n.upper().startswith("UAV_BS") or n.upper().startswith("UAVBS"):
        return "UAV_BS"
    return norm_tier_token(n.split("_", 1)[0])

def tier_from_server_obj(srv) -> str:
    # AccessPoint keeps ``tier="MEC"`` only for legacy mobility compatibility;
    # the map must expose its radio-only role without counting it as compute.
    if bool(getattr(srv, "is_access_only", False)):
        return "ACCESS_POINT"
    name = getattr(srv, "name", "") or ""
    t = tier_from_server_name(name)
    if t != "Unknown":
        return t
    tier_attr = getattr(srv, "tier", None)
    if isinstance(tier_attr, str):
        return norm_tier_token(tier_attr)
    return "Unknown"
