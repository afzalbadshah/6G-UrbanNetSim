"""6G channel model (Task 30) — pure, deterministic physical-link formulas.

Implements the ``network.ChannelModel`` protocol: given an immutable
``ChannelRequest`` it returns an immutable ``ChannelResult`` describing the
physical/link service of ONE wireless hop. It computes channel *physics* only —
path loss, antenna gain, SINR, theoretical achievable rate, outage, a
finite-blocklength packet-error probability, and propagation delay. It never
mutates packets, queues, tasks, ledgers, or global RNG; the PFE owns all queueing,
transmission, retries, drops, goodput, energy, and cost accounting.

How to extend: the per-band parameters live in ``config/bands.py`` (add a band
there); the equations here change only to change the channel physics.

Specification: ``llm_collab/CHANNEL_FORMULA_SPEC.md`` (units: IMPLEMENTATION_CONTRACT.md §2).
Determinism: identical request -> identical result (no random draws). Shadowing/
fast fading are intentionally excluded here (a seeded, opt-in KPI path may add them
later) so RL decisions and evaluation stay reproducible.

References: 3GPP TR 38.901 (path loss / LoS); Shannon capacity + Rappaport
log-distance (matches the 1.1 FR1 radio model for FR1 parity); Polyanskiy, Poor &
Verdu (2010) finite-blocklength normal approximation (PER); ITU-R M.2160 bands.
"""

from __future__ import annotations

import math
import random
from typing import Mapping, Optional

from config import radio as bands_config
from metrics import delay as delay_metrics
from network import ChannelRequest, ChannelResult, LinkProfile

MODEL_ID = "sixg-v1"

# Constants (documented, override-able via constructor).
_REFERENCE_FREQUENCY_HZ = 1.0e9   # f_ref = 1 GHz (legacy-calibrated reference)
_SINR_MIN_DB = -6.0               # decoding floor; below this -> outage
_NOISE_DENSITY_DBM_PER_HZ = -174.0  # thermal noise at 290 K
_AERIAL_ELEVATION_DEG = 30.0      # >= this uses the LoS (air-ground) exponent
# Finite-blocklength CODE-RATE margin. Deliberately SEPARATE from a band's
# link_efficiency_ratio (which derates achievable *capacity* for PHY/MAC overhead):
# this is the channel-coding rate used in the PER model and is independently
# adjustable via the constructor. Same numeric default, different mechanism.
_DEFAULT_CODE_RATE_RATIO = 0.75
# Realistic derating for imperfect channel-state-information / pilot overhead
# in commercial MU-MIMO deployments (typical system-level-simulation range
# ~0.5-0.7); spatial_streams=N does NOT give an N-times rate gain in practice.
_MU_MIMO_EFFICIENCY = 0.6

_LOG2E = math.log2(math.e)


def _dbm_to_mw(dbm: float) -> float:
    return 10.0 ** (dbm / 10.0)


def dbm_to_mw(power_dbm: float) -> float:
    """Convert a finite dBm value to milliwatts."""

    if not math.isfinite(power_dbm):
        raise ValueError("power_dbm must be finite")
    return _dbm_to_mw(float(power_dbm))


def mw_to_dbm(power_mw: float) -> float:
    """Convert positive milliwatts to dBm."""

    if not math.isfinite(power_mw) or power_mw <= 0.0:
        raise ValueError("power_mw must be finite and positive")
    return 10.0 * math.log10(float(power_mw))


def watts_to_dbm(power_w: float) -> float:
    """Convert positive radiated watts to dBm."""

    if not math.isfinite(power_w) or power_w <= 0.0:
        raise ValueError("power_w must be finite and positive")
    return 10.0 * math.log10(float(power_w)) + 30.0


def log_distance_path_loss_db(
    *,
    distance_m: float,
    reference_distance_m: float,
    reference_loss_db: float,
    path_loss_exponent: float,
    carrier_frequency_hz: float,
    reference_frequency_hz: float = _REFERENCE_FREQUENCY_HZ,
    molecular_absorption_db_per_m: float = 0.0,
) -> float:
    """Return the 1.1 log-distance loss extended by the selected 5G/6G band."""

    values = (
        distance_m,
        reference_distance_m,
        reference_loss_db,
        path_loss_exponent,
        carrier_frequency_hz,
        reference_frequency_hz,
        molecular_absorption_db_per_m,
    )
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("path-loss inputs must be finite")
    if reference_distance_m <= 0.0 or carrier_frequency_hz <= 0.0:
        raise ValueError("reference distance and carrier frequency must be positive")
    if reference_frequency_hz <= 0.0 or path_loss_exponent < 0.0:
        raise ValueError("reference frequency must be positive and exponent non-negative")
    if molecular_absorption_db_per_m < 0.0:
        raise ValueError("molecular absorption must be non-negative")
    distance = max(float(distance_m), float(reference_distance_m))
    return (
        float(reference_loss_db)
        + 20.0 * math.log10(float(carrier_frequency_hz) / float(reference_frequency_hz))
        + 10.0
        * float(path_loss_exponent)
        * math.log10(distance / float(reference_distance_m))
        + float(molecular_absorption_db_per_m) * distance
    )


def received_power_dbm(
    *, tx_power_w: float, antenna_gain_dbi: float, path_loss_db: float
) -> float:
    """Return received dBm from radiated watts, antenna gain, and path loss."""

    if not math.isfinite(antenna_gain_dbi) or not math.isfinite(path_loss_db):
        raise ValueError("antenna gain and path loss must be finite")
    return watts_to_dbm(tx_power_w) + float(antenna_gain_dbi) - float(path_loss_db)


def thermal_noise_power_dbm(
    *,
    allocated_bandwidth_hz: float,
    noise_figure_db: float,
    noise_density_dbm_per_hz: float = _NOISE_DENSITY_DBM_PER_HZ,
) -> float:
    """Return thermal noise over the bandwidth actually allocated to the link."""

    if not math.isfinite(allocated_bandwidth_hz) or allocated_bandwidth_hz <= 0.0:
        raise ValueError("allocated_bandwidth_hz must be finite and positive")
    if not math.isfinite(noise_figure_db) or not math.isfinite(noise_density_dbm_per_hz):
        raise ValueError("noise inputs must be finite")
    return (
        float(noise_density_dbm_per_hz)
        + 10.0 * math.log10(float(allocated_bandwidth_hz))
        + float(noise_figure_db)
    )


def signal_to_noise_ratio_db(
    *, received_power_dbm_value: float, noise_power_dbm: float
) -> float:
    """Return SNR in dB from one received-power and noise observation."""

    if not math.isfinite(received_power_dbm_value) or not math.isfinite(noise_power_dbm):
        raise ValueError("received power and noise must be finite")
    return float(received_power_dbm_value) - float(noise_power_dbm)


def signal_to_interference_noise_ratio_db(
    *,
    received_power_dbm_value: float,
    noise_power_dbm: float,
    interference_power_dbm: float | None,
) -> float:
    """Return SINR in dB, summing noise and interference in linear power."""

    signal_mw = dbm_to_mw(received_power_dbm_value)
    noise_mw = dbm_to_mw(noise_power_dbm)
    interference_mw = (
        0.0
        if interference_power_dbm is None or interference_power_dbm <= -200.0
        else dbm_to_mw(interference_power_dbm)
    )
    ratio = signal_mw / max(noise_mw + interference_mw, 1e-30)
    return mw_to_dbm(max(ratio, 1e-300))


def aggregate_interference_power_dbm(
    received_interferer_powers_dbm: tuple[float, ...],
) -> float | None:
    """Sum co-channel received powers and return dBm, or None when absent."""

    if any(not math.isfinite(value) for value in received_interferer_powers_dbm):
        raise ValueError("interferer powers must be finite")
    if not received_interferer_powers_dbm:
        return None
    return mw_to_dbm(sum(dbm_to_mw(value) for value in received_interferer_powers_dbm))


def shannon_achievable_rate_bps(
    *,
    allocated_bandwidth_hz: float,
    sinr_db: float,
    link_efficiency_ratio: float,
) -> float:
    """Return bandwidth times log2(1+SINR) times link efficiency in bit/s."""

    if not math.isfinite(allocated_bandwidth_hz) or allocated_bandwidth_hz <= 0.0:
        raise ValueError("allocated_bandwidth_hz must be finite and positive")
    if not math.isfinite(sinr_db):
        raise ValueError("sinr_db must be finite")
    if not math.isfinite(link_efficiency_ratio) or not 0.0 < link_efficiency_ratio <= 1.0:
        raise ValueError("link_efficiency_ratio must be within (0, 1]")
    sinr_linear = 10.0 ** (float(sinr_db) / 10.0)
    return (
        float(allocated_bandwidth_hz)
        * math.log2(1.0 + sinr_linear)
        * float(link_efficiency_ratio)
    )


def spatial_multiplexing_gain(*, spatial_streams: int, efficiency: float) -> float:
    """Return the realistic capacity multiplier from N parallel MIMO streams.

    "1.0 + (spatial_streams - 1) * efficiency".

    Real-world equivalent: each additional spatial stream beyond the first
    adds capacity, but not a full 1x, because imperfect channel estimation
    and pilot overhead mean parallel streams interfere with each other more
    than an idealized N-times multiplexing gain would suggest.
    """

    if (
        isinstance(spatial_streams, bool)
        or not isinstance(spatial_streams, int)
        or spatial_streams < 1
    ):
        raise ValueError("spatial_streams must be a positive integer (>= 1)")
    if not math.isfinite(efficiency) or not 0.0 < efficiency <= 1.0:
        raise ValueError("efficiency must be finite and within (0, 1]")
    return 1.0 + (float(spatial_streams) - 1.0) * float(efficiency)


def coverage_radius_m(
    *,
    tx_power_dbm: float,
    receiver_sensitivity_dbm: float,
    reference_loss_db: float,
    reference_distance_m: float,
    path_loss_exponent: float,
) -> float:
    """Return the legacy large-scale radius at a receiver-sensitivity floor.

    This inverse uses the reference-frequency anchor. Actual packet delivery
    still uses the selected band's full channel model and SINR/outage rules.
    """

    values = (
        tx_power_dbm,
        receiver_sensitivity_dbm,
        reference_loss_db,
        reference_distance_m,
        path_loss_exponent,
    )
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("coverage-radius inputs must be finite")
    if reference_distance_m <= 0.0 or path_loss_exponent <= 0.0:
        raise ValueError("reference distance and path-loss exponent must be positive")
    allowed_loss_db = float(tx_power_dbm) - float(receiver_sensitivity_dbm)
    return float(reference_distance_m) * 10.0 ** (
        (allowed_loss_db - float(reference_loss_db))
        / (10.0 * float(path_loss_exponent))
    )


def sample_shadowing_db(*, standard_deviation_db: float, rng=random) -> float:
    """Sample the legacy log-normal shadowing term from an explicit RNG."""

    if not math.isfinite(standard_deviation_db) or standard_deviation_db < 0.0:
        raise ValueError("standard_deviation_db must be finite and non-negative")
    return rng.gauss(0.0, standard_deviation_db) if standard_deviation_db > 0.0 else 0.0


def sample_small_scale_fading_db(*, model: str, rng=random) -> float:
    """Sample the existing legacy fading proxy without changing its draws."""

    name = str(model).lower()
    if name == "rayleigh":
        return 10.0 * math.log10(max(rng.expovariate(1.0), 1e-9))
    if name == "rician":
        return 10.0 * math.log10(max(rng.gauss(3.0, 1.0), 1e-9))
    if name == "nakagami":
        return 10.0 * math.log10(max(rng.gammavariate(1.5, 1.0 / 1.5), 1e-9))
    return 0.0


def _q_function(x: float) -> float:
    """Gaussian tail Q(x) = 0.5 * erfc(x / sqrt(2))."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


class SixGChannelModel:
    """Concrete ``network.ChannelModel`` for the 5G/6G bands in ``config.radio``.

    Constructed with the serving ``LinkProfile`` (and optionally a registry of
    profiles keyed by ``profile_id``). ``evaluate`` resolves the request's
    ``link_profile_id`` to a profile, reads its band parameters from
    ``config.radio``, and takes the propagation speed from the profile.
    """

    def __init__(
        self,
        link_profile: LinkProfile,
        profiles: Optional[Mapping[str, LinkProfile]] = None,
        *,
        sinr_min_db: float = _SINR_MIN_DB,
        reference_frequency_hz: float = _REFERENCE_FREQUENCY_HZ,
        code_rate_ratio: float = _DEFAULT_CODE_RATE_RATIO,
        mu_mimo_efficiency: float = _MU_MIMO_EFFICIENCY,
    ) -> None:
        # Scalar validation (fail loudly at construction).
        if not math.isfinite(reference_frequency_hz) or reference_frequency_hz <= 0.0:
            raise ValueError("reference_frequency_hz must be finite and positive")
        if not math.isfinite(sinr_min_db):
            raise ValueError("sinr_min_db must be a finite number")
        if not math.isfinite(code_rate_ratio) or not (0.0 < code_rate_ratio <= 1.0):
            raise ValueError("code_rate_ratio must be within (0, 1]")
        if not math.isfinite(mu_mimo_efficiency) or not (0.0 < mu_mimo_efficiency <= 1.0):
            raise ValueError("mu_mimo_efficiency must be within (0, 1]")

        self._default_profile = link_profile
        self._profiles = dict(profiles or {})
        self._profiles.setdefault(link_profile.profile_id, link_profile)
        self._sinr_min_db = float(sinr_min_db)
        self._reference_frequency_hz = float(reference_frequency_hz)
        self._code_rate_ratio = float(code_rate_ratio)
        self._mu_mimo_efficiency = float(mu_mimo_efficiency)

        # CONFIGURATION SNAPSHOT: copy + validate every referenced band profile now,
        # so a later global mutation of config.radio cannot alter this run's results.
        self._band_by_profile_id = {
            pid: dict(bands_config.get_band(profile.band))   # get_band raises on unknown band
            for pid, profile in self._profiles.items()
        }
        # spatial_streams lives on the LINK profile (per profile_id), not the band
        # (per band name) -- distinct radios on the same band can differ in MIMO
        # rank (e.g. nr-fr1-access=4 vs nr-fr1-air=2). Snapshot it the same way.
        # A profile_id outside the config catalog (e.g. a test double built by
        # hand) is treated as spatial_streams=1 (no MIMO multiplexing gain),
        # matching this model's behaviour before spatial_streams existed.
        self._spatial_streams_by_profile_id = {}
        for pid in self._profiles:
            try:
                catalog_entry = bands_config.get_link_profile(pid)
            except KeyError:
                self._spatial_streams_by_profile_id[pid] = 1
            else:
                self._spatial_streams_by_profile_id[pid] = int(
                    catalog_entry.get("spatial_streams", 1)
                )

    # -- profile / band resolution ------------------------------------------
    def _resolve_profile(self, link_profile_id: str) -> LinkProfile:
        profile = self._profiles.get(link_profile_id, self._default_profile)
        if profile.profile_id != link_profile_id and link_profile_id not in self._profiles:
            # Unknown id and no match: fail loudly rather than silently mislabel.
            raise KeyError(
                f"Unknown link_profile_id {link_profile_id!r}; "
                f"known: {sorted(self._profiles)}"
            )
        return profile

    # -- geometry / LoS ------------------------------------------------------
    @staticmethod
    def _los_probability(distance_m: float, blockage_probability: float) -> float:
        """Smooth, monotonically-decreasing LoS probability used to blend LoS/NLoS
        exponents deterministically (no random draw).

        NOTE: this is a UMi-INSPIRED expected-value PROXY, not full 3GPP TR 38.901
        compliance — it takes only 2-D distance, with no antenna-height or 3-D
        range inputs. It is further reduced by the band's blockage probability."""
        d = max(distance_m, 1.0)
        # 38.901 UMi LoS probability shape (bounded, decreasing).
        p = min(18.0 / d, 1.0) * (1.0 - math.exp(-d / 36.0)) + math.exp(-d / 36.0)
        p = max(0.0, min(1.0, p))
        return p * (1.0 - float(blockage_probability))

    def _effective_exponent(self, band: dict, request: ChannelRequest) -> float:
        n_los = float(band["path_loss_exponent_los"])
        n_nlos = float(band["path_loss_exponent_nlos"])
        # Aerial links are LoS-dominated -> LoS exponent.
        elev = request.elevation_angle_deg
        if elev is not None and elev >= _AERIAL_ELEVATION_DEG:
            return n_los
        if request.line_of_sight is True:
            return n_los
        if request.line_of_sight is False:
            return n_nlos
        # Unknown LoS -> deterministic expected blend by LoS probability.
        p_los = self._los_probability(request.distance_m, band["blockage_probability"])
        return p_los * n_los + (1.0 - p_los) * n_nlos

    # -- main ----------------------------------------------------------------
    def evaluate(self, request: ChannelRequest) -> ChannelResult:
        """Return the physical/link service for one hop as an immutable
        ``ChannelResult`` (SINR dB, achievable rate bit/s, outage, packet-error
        probability, propagation delay s, RX power dBm). Pure and deterministic;
        raises ``ValueError`` if the requested bandwidth exceeds the band's."""
        profile = self._resolve_profile(request.link_profile_id)
        band = self._band_by_profile_id[profile.profile_id]   # construction-time snapshot

        d0 = float(band["reference_distance_m"])
        d = max(float(request.distance_m), d0)   # clamp for finite log/near-field
        f_hz = float(band["carrier_frequency_hz"])

        # --- path loss (dB): anchor + frequency term + distance term + absorption
        # reference_loss_db is a LEGACY-CALIBRATED anchor (30 dB, from the original
        # 1.1 reference model), NOT exact 1 m / 1 GHz free-space loss (~32.4 dB).
        n_eff = self._effective_exponent(band, request)
        path_loss_db = log_distance_path_loss_db(
            distance_m=d,
            reference_distance_m=d0,
            reference_loss_db=float(band["reference_loss_db"]),
            path_loss_exponent=n_eff,
            carrier_frequency_hz=f_hz,
            reference_frequency_hz=self._reference_frequency_hz,
            molecular_absorption_db_per_m=float(
                band["molecular_absorption_db_per_m"]
            ),
        )

        # --- received power (dBm): tx (from watts) + beam gain - path loss
        tx_power_dbm = watts_to_dbm(max(request.tx_power_w, 1e-12))
        gain_dbi = float(band["beamforming_gain_dbi"])
        rx_power_dbm = received_power_dbm(
            tx_power_w=max(request.tx_power_w, 1e-12),
            antenna_gain_dbi=gain_dbi,
            path_loss_db=path_loss_db,
        )

        # --- noise (dBm), bandwidth-consistent with capacity
        band_bandwidth_hz = float(band["bandwidth_hz"])
        if request.allocated_bandwidth_hz is not None:
            allocated_hz = float(request.allocated_bandwidth_hz)
            if allocated_hz > band_bandwidth_hz:
                raise ValueError(
                    f"allocated_bandwidth_hz {allocated_hz} exceeds band bandwidth_hz "
                    f"{band_bandwidth_hz} for band {profile.band!r} (equality is allowed)"
                )
            bandwidth_hz = allocated_hz
        else:
            bandwidth_hz = band_bandwidth_hz
        noise_dbm = thermal_noise_power_dbm(
            allocated_bandwidth_hz=bandwidth_hz,
            noise_figure_db=float(band["noise_figure_db"]),
        )

        # --- SNR and SINR (dB): one formula authority, one observed value.
        interference_dbm = (
            float(request.interference_dbm)
            if request.interference_dbm is not None
            and request.interference_dbm > -200.0
            else None
        )
        snr_db = signal_to_noise_ratio_db(
            received_power_dbm_value=rx_power_dbm,
            noise_power_dbm=noise_dbm,
        )
        sinr_db = signal_to_interference_noise_ratio_db(
            received_power_dbm_value=rx_power_dbm,
            noise_power_dbm=noise_dbm,
            interference_power_dbm=interference_dbm,
        )

        # --- propagation delay: the single formula lives in metrics.delay
        propagation_delay_s = delay_metrics.propagation_delay_s(
            distance_m=float(request.distance_m),
            propagation_speed_m_s=float(profile.propagation_speed_m_s),
        )

        # --- outage
        outage = (rx_power_dbm < float(band["rx_sensitivity_dbm"])) or (sinr_db < self._sinr_min_db)
        if outage:
            return ChannelResult(
                model_id=MODEL_ID,
                sinr_db=sinr_db,
                achievable_rate_bps=0.0,
                outage=True,
                packet_error_probability=1.0,
                propagation_delay_s=propagation_delay_s,
                rx_power_dbm=rx_power_dbm,
                band_name=profile.band,
                generation=profile.generation,
                carrier_frequency_hz=f_hz,
                allocated_bandwidth_hz=bandwidth_hz,
                tx_power_dbm=tx_power_dbm,
                path_loss_db=path_loss_db,
                noise_power_dbm=noise_dbm,
                interference_power_dbm=interference_dbm,
                snr_db=snr_db,
                spatial_streams=self._spatial_streams_by_profile_id[profile.profile_id],
            )

        # --- achievable rate (theoretical; goodput is the PFE's job)
        efficiency = float(band["link_efficiency_ratio"])
        base_rate_bps = shannon_achievable_rate_bps(
            allocated_bandwidth_hz=bandwidth_hz,
            sinr_db=sinr_db,
            link_efficiency_ratio=efficiency,
        )

        # --- MU-MIMO spatial multiplexing: derates the naive N-times gain from
        # imperfect channel-state-information / pilot overhead. Applied only to
        # the reported/aggregate capacity; the PER model below is per-stream
        # decoding physics and must stay driven by the single-stream base rate.
        streams = self._spatial_streams_by_profile_id[profile.profile_id]
        gain = spatial_multiplexing_gain(spatial_streams=streams, efficiency=self._mu_mimo_efficiency)
        achievable_rate_bps = base_rate_bps * gain

        # --- finite-blocklength packet-error probability
        sinr_linear = 10.0 ** (sinr_db / 10.0)
        shannon_bps_per_hz = base_rate_bps / (bandwidth_hz * efficiency)
        per = self._finite_blocklength_per(
            sinr_linear=sinr_linear,
            shannon_bps_per_hz=shannon_bps_per_hz,
            code_rate_ratio=self._code_rate_ratio,
            on_wire_bytes=int(request.on_wire_bytes),
        )

        return ChannelResult(
            model_id=MODEL_ID,
            sinr_db=sinr_db,
            achievable_rate_bps=achievable_rate_bps,
            outage=False,
            packet_error_probability=per,
            propagation_delay_s=propagation_delay_s,
            rx_power_dbm=rx_power_dbm,
            band_name=profile.band,
            generation=profile.generation,
            carrier_frequency_hz=f_hz,
            allocated_bandwidth_hz=bandwidth_hz,
            tx_power_dbm=tx_power_dbm,
            path_loss_db=path_loss_db,
            noise_power_dbm=noise_dbm,
            interference_power_dbm=interference_dbm,
            snr_db=snr_db,
            spatial_streams=streams,
        )

    # -- PER -----------------------------------------------------------------
    @staticmethod
    def _finite_blocklength_per(
        sinr_linear: float,
        shannon_bps_per_hz: float,
        code_rate_ratio: float,
        on_wire_bytes: int,
    ) -> float:
        """Normal-approximation finite-blocklength PER (Polyanskiy et al. 2010).

        C = log2(1+S) bits/ch.use; used rate R = C*code_rate_ratio (< C -> margin);
        n = L_bits / R channel uses; V = (1 - 1/(1+S)^2) * (log2 e)^2;
        eps = Q( (n(C-R) + 0.5 log2 n) / sqrt(nV) ). Falls back to a hard
        threshold if the approximation is degenerate. `code_rate_ratio` is the
        channel-coding rate, SEPARATE from the capacity's link_efficiency_ratio.
        """
        capacity = shannon_bps_per_hz                       # C
        used_rate = capacity * code_rate_ratio              # R
        l_bits = max(on_wire_bytes, 1) * 8
        if capacity <= 0.0 or used_rate <= 0.0:
            return 1.0
        n = l_bits / used_rate                              # channel uses (>0)
        dispersion = (1.0 - 1.0 / (1.0 + sinr_linear) ** 2) * (_LOG2E ** 2)
        if n <= 1.0 or dispersion <= 0.0:
            # Degenerate: threshold fallback (decodable iff there is rate margin).
            return 0.0 if capacity > used_rate else 1.0
        numerator = n * (capacity - used_rate) + 0.5 * math.log2(n)
        denominator = math.sqrt(n * dispersion)
        per = _q_function(numerator / denominator)
        return max(0.0, min(1.0, per))
