"""Authoritative radio-band and link-profile configuration.

Every wireless calculation resolves an explicit link profile and band from the
active deployment.  There is intentionally no process-wide default band: an
unspecified band is a configuration error because silently falling back to a
different generation would invalidate delay, energy and reliability results.
To extend the catalog, add one named band and its link profile in this module,
then select that link profile from a deployment profile.
"""

from config._provenance import deep_freeze, provenance

# --- Interference model --------------------------------------------------
# When enabled, co-channel power from non-serving cells is accumulated so the
# access hop is driven by SINR (not just SNR). Disable for a no-interference,
# market-realistic scenario.
INTERFERENCE_CONFIG = {
    "enabled": True,
}


# =====================================================================
# Radio band catalog (merged from the former config/bands.py). Per-band
# frequency/bandwidth/path-loss used by metrics/channel.py. To add a band,
# add an entry to BAND_PROFILES below (one place).
# =====================================================================
# Per-band radio PARAMETERS. Consumed by the 6G ChannelModel (path loss -> RSSI ->
# SINR -> achievable rate_bps) once the seam is integrated.
#
# PATH-LOSS CONVENTION (see llm_collab/CHANNEL_FORMULA_SPEC.md §6): reference_loss_db
# is the frequency-INDEPENDENT anchor at 1 m at the 1 GHz reference (30 dB).
# Per-band frequency loss is added by the model's 20*log10(f/f_ref) term, so
# every band uses the SAME 30 dB anchor; higher-band loss then comes from frequency
# + path-loss exponent + blockage + molecular absorption. Using a higher per-band
# anchor here AND the frequency term would double-count frequency.
BAND_PROFILES = {
    # -- Low-band coverage/control anchor (600-700 MHz). Real-world-grounded in the
    # US low-band 5G deployment pattern (e.g. T-Mobile/AT&T's sub-1GHz coverage
    # layer): wide-reach, always-on for coverage/control, with FR1 (mid-band) or
    # higher tiers layered on top opportunistically for capacity. Distinct from
    # FR1's mid-band capacity role -- same urban-macro path-loss exponent, lower
    # frequency travels further for the same power via this model's frequency term.
    "LowBand": {
        "generation": "5G/6G",
        "carrier_frequency_hz": 0.7e9,        # 700 MHz -- coverage-layer low band (T-Mobile/AT&T US 5G low-band pattern)
        "bandwidth_hz": 10.0e6,               # 10 MHz -- sub-1GHz spectrum is scarce, narrower channel than FR1's 20MHz
        "path_loss_exponent_los": 2.8,
        "path_loss_exponent_nlos": 2.8,       # same urban-macro exponent as FR1 -- the environment doesn't change, only frequency does
        "reference_distance_m": 1.0,
        "reference_loss_db": 30.0,
        "noise_figure_db": 7.0,
        "shadowing_std_db": 8.0,
        "fading_model": "rayleigh",
        "beamforming_gain_dbi": 0.0,
        "blockage_probability": 0.0,
        "molecular_absorption_db_per_m": 0.0,
        "rx_sensitivity_dbm": -100.0,
        "link_efficiency_ratio": 0.75,
        "default_tx_power_dbm": 23.0,
    },
    # -- 5G sub-6 mid-band reference profile. ------------------------------------
    "FR1": {
        "generation": "5G",
        "carrier_frequency_hz": 3.5e9,        # 3.5 GHz
        "bandwidth_hz": 20.0e6,               # 20 MHz
        "path_loss_exponent_los": 2.8,
        "path_loss_exponent_nlos": 2.8,       # equal => no LoS/NLoS split
        "reference_distance_m": 1.0,
        "reference_loss_db": 30.0,
        "noise_figure_db": 7.0,
        "shadowing_std_db": 8.0,
        "fading_model": "rayleigh",
        "beamforming_gain_dbi": 0.0,          # single antenna, no array gain
        "blockage_probability": 0.0,          # no blockage model at FR1
        "molecular_absorption_db_per_m": 0.0,
        "rx_sensitivity_dbm": -100.0,
        "link_efficiency_ratio": 0.75,
        "default_tx_power_dbm": 23.0,
    },
    # -- 6G upper-mid-band ("FR3", ~7-24 GHz). ----------------------------------
    "FR3": {
        "generation": "6G",
        "carrier_frequency_hz": 13.0e9,       # 13 GHz
        "bandwidth_hz": 100.0e6,              # 100 MHz
        "path_loss_exponent_los": 2.2,
        "path_loss_exponent_nlos": 3.0,
        "reference_distance_m": 1.0,
        "reference_loss_db": 30.0,            # anchor @1 m @1 GHz; freq loss via 20log10(f/f_ref)
        "noise_figure_db": 7.0,
        "shadowing_std_db": 7.0,
        "fading_model": "rician",
        "beamforming_gain_dbi": 8.0,
        "blockage_probability": 0.10,
        "molecular_absorption_db_per_m": 0.0,
        "rx_sensitivity_dbm": -100.0,
        "link_efficiency_ratio": 0.75,
        "default_tx_power_dbm": 23.0,
    },
    # -- 6G mmWave (~24-100 GHz). -----------------------------------------------
    "mmWave": {
        "generation": "6G",
        "carrier_frequency_hz": 28.0e9,       # 28 GHz
        "bandwidth_hz": 400.0e6,             # 400 MHz
        "path_loss_exponent_los": 2.0,
        "path_loss_exponent_nlos": 3.3,
        "reference_distance_m": 1.0,
        "reference_loss_db": 30.0,           # anchor @1 m @1 GHz; freq loss via 20log10(f/f_ref)
        "noise_figure_db": 8.0,
        "shadowing_std_db": 6.0,             # tight beams reduce shadow spread
        "fading_model": "rician",
        "beamforming_gain_dbi": 18.0,        # massive-MIMO beam gain makes mmWave viable
        "blockage_probability": 0.30,        # blockage-dominated
        "molecular_absorption_db_per_m": 0.0,
        "rx_sensitivity_dbm": -95.0,
        "link_efficiency_ratio": 0.70,
        "default_tx_power_dbm": 23.0,
    },
    # -- 6G sub-THz (~100-300 GHz). The future-facing headline band. ------------
    "subTHz": {
        "generation": "6G",
        "carrier_frequency_hz": 140.0e9,      # 140 GHz
        "bandwidth_hz": 2.0e9,               # 2 GHz (multi-GHz channels)
        "path_loss_exponent_los": 2.0,
        "path_loss_exponent_nlos": 3.5,
        "reference_distance_m": 1.0,
        "reference_loss_db": 30.0,           # anchor @1 m @1 GHz; freq loss via 20log10(f/f_ref)
        "noise_figure_db": 10.0,
        "shadowing_std_db": 5.0,
        "fading_model": "rician",
        "beamforming_gain_dbi": 25.0,        # ultra-massive MIMO / very narrow beams
        "blockage_probability": 0.45,
        "molecular_absorption_db_per_m": 0.01,  # 10 dB/km distance-dependent absorption
        "rx_sensitivity_dbm": -90.0,
        "link_efficiency_ratio": 0.65,
        "default_tx_power_dbm": 23.0,
    },
}


def get_band(name):
    """Return one explicitly named band profile.

    A missing/blank band fails loudly.  The active deployment owns band
    selection; this lookup must never invent a process-wide fallback.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("band name must be a non-empty string")
    key = name.strip()
    if key not in BAND_PROFILES:
        raise KeyError(
            f"Unknown band {key!r}. Valid bands: {sorted(BAND_PROFILES)}"
        )
    return BAND_PROFILES[key]


# =====================================================================
# Packet/link profiles (merged from the former config/network_config.py):
# per-link header stack, bit rate and propagation speed for radio and fibre
# hops. To add a link, add an entry to _RAW_LINK_PROFILES below (one place).
# =====================================================================
_P_RADIO = provenance(
    "RFC 8200 + RFC 768 + configurable NR PDU overhead proxy",
    source_uri="https://www.rfc-editor.org/rfc/rfc8200",
    source_version="IPv6/UDP baseline; SmartCityEdgeSim fixed NR proxy v1",
    validity_notes=(
        "IPv6=40 B, UDP=8 B; 8 B radio header + 3 B trailer is an experimental "
        "fixed proxy, not a full bearer/transport-block model"
    ),
    maturity="experimental",
)
_P_ETHERNET = provenance(
    "IEEE 802.3 Ethernet framing + RFC 8200/RFC 768",
    source_uri="https://standards.ieee.org/ieee/802.3/7071/",
    source_version="802.3 Ethernet / IPv6 / UDP baseline",
    validity_notes=(
        "Jumbo IP MTU scenario; 38 B link occupancy counts 14 B MAC header, "
        "4 B FCS, 8 B preamble/SFD and 12 B inter-frame-gap equivalent"
    ),
    maturity="experimental",
)


def _radio(profile_id, band, generation, nominal_bit_rate_bps, technology="NR", spatial_streams: int = 1):
    return {
        "profile_id": profile_id,
        "technology": technology,
        "generation": generation,
        "band": band,
        "network_mtu_bytes": 1_500,
        "nominal_bit_rate_bps": float(nominal_bit_rate_bps),
        "propagation_speed_m_s": 299_792_458.0,
        "shared_resource_id": f"template:{profile_id}",
        "channel_model_id": "sixg-v1",
        "queue_capacity_bytes": 0,
        # spatial_streams = number of parallel MIMO data streams this link serves
        # in the same time-frequency resource (raw capacity multiplier). This is
        # distinct from BAND_PROFILES[...]["beamforming_gain_dbi"], which is a
        # single-stream link-budget gain (raises SINR for one stream, not the
        # stream count).
        "spatial_streams": spatial_streams,
        "header_stack": {
            "network_header_bytes": 40,
            "transport_header_bytes": 8,
            "security_overhead_bytes": 0,
            "telemetry_overhead_bytes": 0,
            "link_header_bytes": 8,
            "link_trailer_bytes": 3,
        },
        "provenance": _P_RADIO,
    }


def _fibre(profile_id, nominal_bit_rate_bps):
    return {
        "profile_id": profile_id,
        "technology": "Ethernet over fibre",
        "generation": "reference",
        "band": "wired",
        "network_mtu_bytes": 9_000,
        "nominal_bit_rate_bps": float(nominal_bit_rate_bps),
        "propagation_speed_m_s": 200_000_000.0,
        "shared_resource_id": f"template:{profile_id}",
        "channel_model_id": "fixed-rate-wired-v1",
        "queue_capacity_bytes": 0,
        "header_stack": {
            "network_header_bytes": 40,
            "transport_header_bytes": 8,
            "security_overhead_bytes": 0,
            "telemetry_overhead_bytes": 0,
            "link_header_bytes": 22,
            "link_trailer_bytes": 16,
        },
        "provenance": _P_ETHERNET,
    }


_RAW_LINK_PROFILES = {
    # anchor/coverage layer every other tier builds on top of. Real US low-band 5G
    # commonly uses 2T2R/4T4R radios (fewer antenna elements than mid-band
    # massive-MIMO panels, since antenna elements are physically larger at lower
    # frequency), so spatial_streams=2 here vs nr-fr1-access's higher stream count.
    "nr-lowband-access": _radio("nr-lowband-access", "LowBand", "5G/6G", 10.0e6, spatial_streams=2),
    # commercial mid-band massive-MIMO (32T32R-64T64R) realistic ~2-4x sector-capacity
    # gain after CSI/pilot overhead (3GPP Rel-15+ NR; vendor mid-band massive-MIMO
    # field trials) -- 5G already uses massive MIMO commercially, not a 6G-exclusive
    # feature.
    "nr-fr1-access": _radio("nr-fr1-access", "FR1", "5G", 20.0e6, spatial_streams=4),
    # UAV-ground links are strongly LoS-dominated, reducing usable MIMO rank even
    # with strong SNR (documented LoS rank-deficiency effect) -- fewer usable
    # streams than a ground NLoS-rich link despite the same FR1 band.
    "nr-fr1-air": _radio("nr-fr1-air", "FR1", "5G/6G aerial", 20.0e6, spatial_streams=2),
    # 6G UAV access hop -- higher band than nr-fr1-air for a 6G deployment profile,
    # but spatial_streams kept lower than ground nr-mmwave-access (8) for the same
    # LoS-rank-deficiency reason as nr-fr1-air.
    "nr-mmwave-air": _radio("nr-mmwave-air", "mmWave", "6G aerial", 400.0e6, spatial_streams=4),
    # larger antenna apertures feasible at FR3 (~13GHz) with richer ground-level
    # multipath support higher-rank spatial multiplexing than sub-6GHz macro; 6G
    # FR3 massive-MIMO direction (ITU-R IMT-2030, 3GPP Rel-18/19 discussion) targets
    # higher-order MU-MIMO than legacy 5G mid-band.
    "nr-fr3-access": _radio("nr-fr3-access", "FR3", "6G", 100.0e6, spatial_streams=8),
    # mmWave permits very large arrays but sparser, more LoS-dominated scattering
    # caps usable rank; 6G mmWave hybrid-beamforming research generally targets a
    # handful of effective streams rather than scaling with raw antenna count.
    "nr-mmwave-access": _radio("nr-mmwave-access", "mmWave", "6G", 400.0e6, spatial_streams=8),
    # sub-THz channels are even more LoS/sparse-scattering than mmWave; per 6G
    # sub-THz surveys, gains here come predominantly from pencil-beam beamforming
    # (already captured by this band's beamforming_gain_dbi=25) rather than extra
    # spatial-multiplexing rank, so streams are kept level with mmWave rather than
    # scaled up further.
    "nr-subthz-access": _radio("nr-subthz-access", "subTHz", "6G", 2.0e9, spatial_streams=8),
    # dedicated point-to-point backhaul; dual-polarization MIMO (2 streams) is
    # realistic for a fixed P2P mmWave backhaul link, not full multi-user spatial
    # multiplexing.
    "nr-mmwave-backhaul": _radio(
        "nr-mmwave-backhaul",
        "mmWave",
        "5G/6G aerial backhaul",
        1.3e9,
        technology="directional NR backhaul",
        spatial_streams=2,
    ),
    "fibre-backhaul": _fibre("fibre-backhaul", 10.0e9),
    "fibre-core": _fibre("fibre-core", 100.0e9),
}

LINK_PROFILES = deep_freeze(_RAW_LINK_PROFILES)


def get_link_profile(name):
    """Return the immutable link profile for ``name`` (raises if unknown)."""
    if name not in LINK_PROFILES:
        raise KeyError(f"Unknown link profile {name!r}. Valid: {sorted(LINK_PROFILES)}")
    return LINK_PROFILES[name]
