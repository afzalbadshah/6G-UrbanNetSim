"""Tests for metrics/channel.py (Task 30, 6G channel model).

Covers FR1 handover/packet RSSI agreement, determinism,
distance/LoS/interference/absorption/aerial
behaviour, outage, finite-blocklength PER monotonicity, bandwidth handling,
probability ranges, and invalid inputs.
"""

import math
import unittest
from types import SimpleNamespace

from config import radio as bands_config
from metrics import handover_channel
from metrics.channel import SixGChannelModel
from network import ChannelRequest, Direction, LinkProfile


def make_profile(band: str, profile_id: str | None = None) -> LinkProfile:
    return LinkProfile(
        profile_id=profile_id or f"{band}-profile",
        technology="5G-NR" if band == "FR1" else "6G-NR",
        generation=bands_config.get_band(band)["generation"],
        band=band,
        network_mtu_bytes=1500,
        nominal_bit_rate_bps=1.0e9,
        propagation_speed_m_s=3.0e8,
        shared_resource_id="cell-1",
        channel_model_id="sixg",
    )


def dbm_to_watts(dbm: float) -> float:
    return 10.0 ** ((dbm - 30.0) / 10.0)


def make_request(band: str, distance_m: float, **kw) -> ChannelRequest:
    return ChannelRequest(
        link_profile_id=kw.pop("link_profile_id", f"{band}-profile"),
        hop_id="hop-1",
        requested_at_s=0.0,
        distance_m=distance_m,
        on_wire_bytes=kw.pop("on_wire_bytes", 1500),
        direction=Direction.UPLINK,
        tx_power_w=kw.pop("tx_power_w", dbm_to_watts(23.0)),
        allocated_bandwidth_hz=kw.pop("allocated_bandwidth_hz", None),
        interference_dbm=kw.pop("interference_dbm", None),
        line_of_sight=kw.pop("line_of_sight", None),
        elevation_angle_deg=kw.pop("elevation_angle_deg", None),
    )


class TestFR1Parity(unittest.TestCase):
    """Handover and packet paths must use the same FR1 received power."""

    def setUp(self):
        self.model = SixGChannelModel(make_profile("FR1"))

    def test_rssi_parity(self):
        for d in (50.0, 200.0, 1000.0):
            with self.subTest(distance=d):
                result = self.model.evaluate(make_request("FR1", d))
                handover_rssi = handover_channel.calculate_rssi_dbm(
                    d,
                    band_name="FR1",
                    include_fading=False,
                )
                self.assertAlmostEqual(result.rx_power_dbm, handover_rssi, places=6)

    def test_coverage_radius_includes_carrier_frequency(self):
        radius = handover_channel.max_coverage_radius_m(band_name="FR1")
        self.assertGreater(radius, 400.0)
        self.assertLess(radius, 1000.0)
        self.assertTrue(self.model.evaluate(make_request("FR1", 1000.0)).outage)


class TestPhysicalBehaviour(unittest.TestCase):
    def test_handover_interference_excludes_other_bands(self):
        serving = SimpleNamespace(
            name="BS_FR3_1", x=0.0, y=0.0, profile_id="nr-fr3-access"
        )
        same_band = SimpleNamespace(
            name="BS_FR3_2", x=100.0, y=0.0, profile_id="nr-fr3-access"
        )
        other_band = SimpleNamespace(
            name="UAV_MMWAVE", x=1.0, y=0.0, profile_id="nr-mmwave-air"
        )

        fr3_only_dbm = handover_channel.calculate_interference_power_dbm(
            (0.0, 0.0), serving, [serving, same_band], band_name="FR3"
        )
        mixed_dbm = handover_channel.calculate_interference_power_dbm(
            (0.0, 0.0),
            serving,
            [serving, same_band, other_band],
            band_name="FR3",
        )

        self.assertAlmostEqual(mixed_dbm, fr3_only_dbm, places=9)

    def test_result_preserves_the_exact_radio_observation(self):
        model = SixGChannelModel(make_profile("FR1"))
        clean = model.evaluate(make_request("FR1", 100.0))
        interfered = model.evaluate(
            make_request("FR1", 100.0, interference_dbm=-80.0)
        )

        self.assertEqual(clean.band_name, "FR1")
        self.assertEqual(clean.generation, "5G")
        self.assertEqual(
            clean.allocated_bandwidth_hz,
            bands_config.get_band("FR1")["bandwidth_hz"],
        )
        self.assertAlmostEqual(
            clean.snr_db, clean.rx_power_dbm - clean.noise_power_dbm
        )
        self.assertIsNone(clean.interference_power_dbm)
        self.assertEqual(interfered.interference_power_dbm, -80.0)
        self.assertLess(interfered.sinr_db, interfered.snr_db)

    def test_determinism(self):
        model = SixGChannelModel(make_profile("mmWave"))
        r1 = model.evaluate(make_request("mmWave", 300.0))
        r2 = model.evaluate(make_request("mmWave", 300.0))
        self.assertEqual(r1, r2)

    def test_rate_decreases_with_distance(self):
        model = SixGChannelModel(make_profile("FR3"))
        rates = [model.evaluate(make_request("FR3", d)).achievable_rate_bps
                 for d in (50.0, 150.0, 400.0, 800.0)]
        for a, b in zip(rates, rates[1:]):
            self.assertGreaterEqual(a, b)

    def test_los_beats_nlos(self):
        model = SixGChannelModel(make_profile("mmWave"))
        los = model.evaluate(make_request("mmWave", 300.0, line_of_sight=True))
        nlos = model.evaluate(make_request("mmWave", 300.0, line_of_sight=False))
        self.assertGreater(los.achievable_rate_bps, nlos.achievable_rate_bps)

    def test_interference_reduces_rate(self):
        model = SixGChannelModel(make_profile("FR3"))
        clean = model.evaluate(make_request("FR3", 200.0))
        noisy = model.evaluate(make_request("FR3", 200.0, interference_dbm=-70.0))
        self.assertLess(noisy.sinr_db, clean.sinr_db)
        self.assertLess(noisy.achievable_rate_bps, clean.achievable_rate_bps)

    def test_aerial_uses_los_exponent(self):
        model = SixGChannelModel(make_profile("mmWave"))
        aerial = model.evaluate(make_request("mmWave", 300.0, elevation_angle_deg=60.0))
        ground_nlos = model.evaluate(make_request("mmWave", 300.0, line_of_sight=False))
        self.assertGreater(aerial.achievable_rate_bps, ground_nlos.achievable_rate_bps)

    def test_molecular_absorption_faster_decay_subthz(self):
        subthz = SixGChannelModel(make_profile("subTHz"))
        mmwave = SixGChannelModel(make_profile("mmWave"))
        drop_subthz = (subthz.evaluate(make_request("subTHz", 100.0)).rx_power_dbm
                       - subthz.evaluate(make_request("subTHz", 2000.0)).rx_power_dbm)
        drop_mmwave = (mmwave.evaluate(make_request("mmWave", 100.0)).rx_power_dbm
                       - mmwave.evaluate(make_request("mmWave", 2000.0)).rx_power_dbm)
        self.assertGreater(drop_subthz, drop_mmwave)

    def test_outage_beyond_coverage(self):
        model = SixGChannelModel(make_profile("subTHz"))
        result = model.evaluate(make_request("subTHz", 500_000.0))
        self.assertTrue(result.outage)
        self.assertEqual(result.achievable_rate_bps, 0.0)
        self.assertEqual(result.packet_error_probability, 1.0)

    def test_propagation_delay(self):
        model = SixGChannelModel(make_profile("FR1"))
        result = model.evaluate(make_request("FR1", 3.0e8))  # 1 light-second away
        self.assertAlmostEqual(result.propagation_delay_s, 1.0, places=6)

    def test_bandwidth_allocation_used(self):
        model = SixGChannelModel(make_profile("FR3"))
        default = model.evaluate(make_request("FR3", 200.0))
        band_bw = bands_config.get_band("FR3")["bandwidth_hz"]
        full = model.evaluate(make_request("FR3", 200.0, allocated_bandwidth_hz=band_bw))
        half = model.evaluate(make_request("FR3", 200.0, allocated_bandwidth_hz=band_bw / 2))
        self.assertAlmostEqual(full.achievable_rate_bps, default.achievable_rate_bps, places=3)
        self.assertNotAlmostEqual(half.achievable_rate_bps, default.achievable_rate_bps, places=3)


class TestFiniteBlocklengthPER(unittest.TestCase):
    def test_per_in_unit_interval(self):
        model = SixGChannelModel(make_profile("mmWave"))
        for d in (50.0, 200.0, 1000.0, 3000.0):
            per = model.evaluate(make_request("mmWave", d)).packet_error_probability
            self.assertGreaterEqual(per, 0.0)
            self.assertLessEqual(per, 1.0)

    def test_per_decreases_with_sinr(self):
        high = SixGChannelModel._finite_blocklength_per(
            sinr_linear=100.0, shannon_bps_per_hz=math.log2(101.0), code_rate_ratio=0.75, on_wire_bytes=1500)
        low = SixGChannelModel._finite_blocklength_per(
            sinr_linear=1.0, shannon_bps_per_hz=math.log2(2.0), code_rate_ratio=0.75, on_wire_bytes=1500)
        self.assertLessEqual(high, low)

    def test_per_increases_for_shorter_packets(self):
        big = SixGChannelModel._finite_blocklength_per(
            sinr_linear=3.0, shannon_bps_per_hz=math.log2(4.0), code_rate_ratio=0.9, on_wire_bytes=8000)
        small = SixGChannelModel._finite_blocklength_per(
            sinr_linear=3.0, shannon_bps_per_hz=math.log2(4.0), code_rate_ratio=0.9, on_wire_bytes=40)
        self.assertGreaterEqual(small, big)


class TestInvalidInputs(unittest.TestCase):
    def test_unknown_link_profile_id_raises(self):
        model = SixGChannelModel(make_profile("FR1"))
        with self.assertRaises(KeyError):
            model.evaluate(make_request("FR1", 100.0, link_profile_id="does-not-exist"))

    def test_unknown_band_raises_at_construction(self):
        # Bands are snapshot + validated in __init__, so an unknown band fails early.
        bogus = LinkProfile(
            profile_id="bogus", technology="x", generation="6G", band="7G",
            network_mtu_bytes=1500, nominal_bit_rate_bps=1e9,
            propagation_speed_m_s=3e8, shared_resource_id="cell-1",
        )
        with self.assertRaises(KeyError):
            SixGChannelModel(bogus)

    def test_rejects_over_allocation(self):
        model = SixGChannelModel(make_profile("FR1"))
        band_bw = bands_config.get_band("FR1")["bandwidth_hz"]
        # Equality is valid.
        ok = model.evaluate(make_request("FR1", 100.0, allocated_bandwidth_hz=band_bw))
        self.assertFalse(ok.outage)
        # Above the band's bandwidth is impossible spectrum -> reject.
        with self.assertRaises(ValueError):
            model.evaluate(make_request("FR1", 100.0, allocated_bandwidth_hz=band_bw * 2))

    def test_config_snapshot_immune_to_global_mutation(self):
        model = SixGChannelModel(make_profile("mmWave"))
        req = make_request("mmWave", 200.0, line_of_sight=True)
        before = model.evaluate(req)
        original = bands_config.BAND_PROFILES["mmWave"]["beamforming_gain_dbi"]
        try:
            bands_config.BAND_PROFILES["mmWave"]["beamforming_gain_dbi"] = original + 10.0
            after = model.evaluate(req)
            self.assertEqual(before, after)  # snapshot => later mutation cannot change a run
        finally:
            bands_config.BAND_PROFILES["mmWave"]["beamforming_gain_dbi"] = original

    def test_zero_tx_power_is_outage_not_crash(self):
        model = SixGChannelModel(make_profile("FR1"))
        r = model.evaluate(make_request("FR1", 100.0, tx_power_w=0.0))
        self.assertTrue(r.outage)
        self.assertEqual(r.achievable_rate_bps, 0.0)

    def test_zero_on_wire_bytes_valid_per(self):
        model = SixGChannelModel(make_profile("FR1"))
        r = model.evaluate(make_request("FR1", 100.0, on_wire_bytes=0))
        self.assertGreaterEqual(r.packet_error_probability, 0.0)
        self.assertLessEqual(r.packet_error_probability, 1.0)

    def test_constructor_validation(self):
        p = make_profile("FR1")
        with self.assertRaises(ValueError):
            SixGChannelModel(p, reference_frequency_hz=0.0)
        with self.assertRaises(ValueError):
            SixGChannelModel(p, sinr_min_db=float("inf"))
        for bad in (0.0, 1.5, -0.1):
            with self.assertRaises(ValueError):
                SixGChannelModel(p, code_rate_ratio=bad)

    def test_code_rate_ratio_independently_adjustable(self):
        req = make_request("FR3", 260.0)
        strict = SixGChannelModel(make_profile("FR3"), code_rate_ratio=0.95)
        loose = SixGChannelModel(make_profile("FR3"), code_rate_ratio=0.5)
        # Higher code rate leaves less margin -> higher PER; rate is unaffected.
        self.assertGreaterEqual(
            strict.evaluate(req).packet_error_probability,
            loose.evaluate(req).packet_error_probability,
        )
        self.assertAlmostEqual(
            strict.evaluate(req).achievable_rate_bps,
            loose.evaluate(req).achievable_rate_bps,
            places=3,
        )

    def test_profile_registry_lookup(self):
        # A request's link_profile_id must resolve via the registry to the right
        # band: evaluating "p2" through the multi-profile model must equal
        # evaluating it through a mmWave-only model (proves resolution, regardless
        # of whether that geometry happens to be in outage).
        p1, p2 = make_profile("FR1", "p1"), make_profile("mmWave", "p2")
        model = SixGChannelModel(p1, profiles={"p1": p1, "p2": p2})
        mmwave_only = SixGChannelModel(p2)
        req = make_request("mmWave", 200.0, link_profile_id="p2")
        self.assertEqual(model.evaluate(req), mmwave_only.evaluate(req))


if __name__ == "__main__":
    unittest.main()
