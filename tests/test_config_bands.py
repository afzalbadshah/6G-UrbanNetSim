"""Tests for config/bands.py (Task 27, 6G band DATA).

Verifies schema completeness, canonical units (Hz + suffixes per
IMPLEMENTATION_CONTRACT.md §2), and explicit get_band() behaviour.
"""

import unittest

from config import radio as bands


REQUIRED_KEYS = {
    "generation",
    "carrier_frequency_hz",
    "bandwidth_hz",
    "path_loss_exponent_los",
    "path_loss_exponent_nlos",
    "reference_distance_m",
    "reference_loss_db",
    "noise_figure_db",
    "shadowing_std_db",
    "fading_model",
    "beamforming_gain_dbi",
    "blockage_probability",
    "molecular_absorption_db_per_m",
    "rx_sensitivity_dbm",
    "link_efficiency_ratio",
    "default_tx_power_dbm",
}


class TestBandProfiles(unittest.TestCase):
    def test_expected_bands_present(self):
        self.assertEqual(
            set(bands.BAND_PROFILES), {"LowBand", "FR1", "FR3", "mmWave", "subTHz"}
        )

    def test_schema_and_units(self):
        for name, profile in bands.BAND_PROFILES.items():
            with self.subTest(band=name):
                self.assertEqual(set(profile), REQUIRED_KEYS)
                # positive frequency/bandwidth in Hz (not GHz/MHz)
                self.assertGreater(profile["carrier_frequency_hz"], 1e8)
                self.assertGreater(profile["bandwidth_hz"], 1e6)
                # probability is a ratio in [0, 1]
                self.assertGreaterEqual(profile["blockage_probability"], 0.0)
                self.assertLessEqual(profile["blockage_probability"], 1.0)
                self.assertGreater(profile["link_efficiency_ratio"], 0.0)
                self.assertLessEqual(profile["link_efficiency_ratio"], 1.0)

    def test_no_legacy_unit_suffixes(self):
        # Guard against reintroducing GHz/MHz-named keys (contract §2).
        for profile in bands.BAND_PROFILES.values():
            for key in profile:
                self.assertFalse(key.endswith("_ghz"), f"{key} uses GHz naming")
                self.assertFalse(key.endswith("_mhz"), f"{key} uses MHz naming")

    def test_ordering_makes_physical_sense(self):
        # Higher band => more bandwidth and higher beamforming gain.
        order = ["FR1", "FR3", "mmWave", "subTHz"]
        bw = [bands.BAND_PROFILES[b]["bandwidth_hz"] for b in order]
        gain = [bands.BAND_PROFILES[b]["beamforming_gain_dbi"] for b in order]
        self.assertEqual(bw, sorted(bw))
        self.assertEqual(gain, sorted(gain))

    def test_fr1_reference_profile_is_self_contained(self):
        fr1 = bands.get_band("FR1")
        self.assertEqual(fr1["carrier_frequency_hz"], 3.5e9)
        self.assertEqual(fr1["bandwidth_hz"], 20.0e6)
        self.assertEqual(fr1["generation"], "5G")

    def test_get_band_requires_explicit_name_and_rejects_unknown(self):
        with self.assertRaises(TypeError):
            bands.get_band()
        with self.assertRaises(ValueError):
            bands.get_band(None)
        with self.assertRaises(ValueError):
            bands.get_band("  ")
        with self.assertRaises(KeyError):
            bands.get_band("7G")


if __name__ == "__main__":
    unittest.main()
