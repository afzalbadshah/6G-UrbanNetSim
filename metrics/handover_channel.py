"""Configured radio observations for mobility and handover decisions.

The physical path-loss, noise, SNR/SINR, interference, and fading formulas live
in :mod:`metrics.channel`. This module only supplies the handover inputs from
the active experiment and evaluates candidate access points.
"""

import math
import random

import config as cfg
from metrics.channel import (
    aggregate_interference_power_dbm,
    coverage_radius_m,
    log_distance_path_loss_db,
    received_power_dbm,
    sample_shadowing_db,
    sample_small_scale_fading_db,
    signal_to_interference_noise_ratio_db,
    signal_to_noise_ratio_db,
    thermal_noise_power_dbm,
)


def _band_settings(band_name):
    """Adapt one explicit band profile for handover observations.

    NLoS is the ground-link default.  Air-to-ground callers provide the same
    band's LoS exponent explicitly.  No implicit generation/band fallback is
    permitted.
    """
    band = cfg.get_band(band_name)
    return {
        "tx_power_dbm": float(band.get("default_tx_power_dbm", 23.0)),
        "path_loss_exponent": float(
            band.get("path_loss_exponent_nlos", band.get("path_loss_exponent_los", 2.8))
        ),
        "air_ground_path_loss_exponent": float(band.get("path_loss_exponent_los", 2.2)),
        "reference_distance": float(band.get("reference_distance_m", 1.0)),
        "reference_loss_db": float(band.get("reference_loss_db", 30.0)),
        "reference_frequency_ghz": 1.0,
        "frequency_ghz": float(band.get("carrier_frequency_hz", 3.5e9)) / 1e9,
        "bandwidth_mhz": float(band.get("bandwidth_hz", 20.0e6)) / 1e6,
        "noise_power_dbm_per_hz": -174.0,
        "noise_figure_db": float(band.get("noise_figure_db", 7.0)),
        "shadowing_std_db": float(band.get("shadowing_std_db", 8.0)),
        "fading_model": str(band.get("fading_model", "rayleigh")),
        "rx_sensitivity_dbm": float(band.get("rx_sensitivity_dbm", -100.0)),
    }


def air_ground_path_loss_exponent(*, band_name):
    return float(_band_settings(band_name)["air_ground_path_loss_exponent"])


def calculate_rssi_dbm(
    distance_m,
    *,
    band_name,
    include_fading=False,
    path_loss_exponent=None,
):
    """Receive power at the candidate point, optionally with sampled fading."""

    settings = _band_settings(band_name)
    exponent = (
        float(path_loss_exponent)
        if path_loss_exponent is not None
        else float(settings["path_loss_exponent"])
    )
    loss_db = log_distance_path_loss_db(
        distance_m=float(distance_m),
        reference_distance_m=max(float(settings["reference_distance"]), 1e-6),
        reference_loss_db=float(settings["reference_loss_db"]),
        path_loss_exponent=exponent,
        carrier_frequency_hz=max(float(settings.get("frequency_ghz", 1.0)), 1e-9) * 1e9,
        reference_frequency_hz=max(float(settings.get("reference_frequency_ghz", 1.0)), 1e-9) * 1e9,
    )
    power_dbm = received_power_dbm(
        tx_power_w=10.0 ** ((float(settings["tx_power_dbm"]) - 30.0) / 10.0),
        antenna_gain_dbi=0.0,
        path_loss_db=loss_db,
    )
    if include_fading:
        power_dbm += sample_shadowing_db(
            standard_deviation_db=float(settings.get("shadowing_std_db", 0.0) or 0.0),
            rng=random,
        )
        power_dbm += sample_small_scale_fading_db(
            model=str(settings.get("fading_model", "none")).lower(),
            rng=random,
        )
    return power_dbm


def max_coverage_radius_m(*, band_name):
    """Large-scale sensitivity radius at the configured carrier frequency."""

    settings = _band_settings(band_name)
    frequency_ratio = max(float(settings.get("frequency_ghz", 1.0)), 1e-9) / max(
        float(settings.get("reference_frequency_ghz", 1.0)), 1e-9
    )
    return coverage_radius_m(
        tx_power_dbm=float(settings["tx_power_dbm"]),
        receiver_sensitivity_dbm=float(settings.get("rx_sensitivity_dbm", -100.0)),
        reference_loss_db=float(settings["reference_loss_db"])
        + 20.0 * math.log10(frequency_ratio),
        reference_distance_m=max(float(settings["reference_distance"]), 1e-6),
        path_loss_exponent=max(float(settings["path_loss_exponent"]), 1e-6),
    )


def calculate_noise_power_dbm(*, band_name):
    """Noise over the bandwidth configured for handover observations."""

    settings = _band_settings(band_name)
    return thermal_noise_power_dbm(
        allocated_bandwidth_hz=max(float(settings.get("bandwidth_mhz", 20.0)) * 1e6, 1.0),
        noise_figure_db=float(settings.get("noise_figure_db", 0.0) or 0.0),
        noise_density_dbm_per_hz=float(settings.get("noise_power_dbm_per_hz", -174.0)),
    )


def calculate_interference_power_dbm(
    pos_xy,
    serving_server,
    candidate_servers,
    *,
    band_name,
):
    """Sum power only from candidates transmitting in the serving band."""

    if not bool(cfg.INTERFERENCE_CONFIG.get("enabled", False)):
        return -200.0
    x, y = float(pos_xy[0]), float(pos_xy[1])
    serving_name = str(getattr(serving_server, "name", "") or "")
    powers = []
    for server in candidate_servers:
        if str(getattr(server, "name", "")) == serving_name:
            continue
        profile_id = getattr(server, "profile_id", None)
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise ValueError(
                f"radio candidate {getattr(server, 'name', '<unnamed>')!r} "
                "has no explicit profile_id"
            )
        candidate_band = str(cfg.get_link_profile(profile_id)["band"])
        if candidate_band != band_name:
            continue
        distance_m = math.hypot(
            x - float(getattr(server, "x", 0.0)),
            y - float(getattr(server, "y", 0.0)),
        )
        powers.append(calculate_rssi_dbm(distance_m, band_name=candidate_band))
    combined_dbm = aggregate_interference_power_dbm(tuple(powers))
    return -200.0 if combined_dbm is None else combined_dbm


def calculate_snr_db(rssi_dbm, noise_dbm=None, *, band_name):
    noise = (
        calculate_noise_power_dbm(band_name=band_name)
        if noise_dbm is None
        else float(noise_dbm)
    )
    return signal_to_noise_ratio_db(
        received_power_dbm_value=float(rssi_dbm), noise_power_dbm=noise
    )


def calculate_sinr_db(rssi_dbm, interference_dbm, noise_dbm=None, *, band_name):
    noise = (
        calculate_noise_power_dbm(band_name=band_name)
        if noise_dbm is None
        else float(noise_dbm)
    )
    return signal_to_interference_noise_ratio_db(
        received_power_dbm_value=float(rssi_dbm),
        noise_power_dbm=noise,
        interference_power_dbm=float(interference_dbm),
    )
