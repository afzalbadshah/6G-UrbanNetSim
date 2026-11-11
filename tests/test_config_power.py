"""Tests for config/power.py (Task 33 / R3 power DATA).

Verifies contract §13 (separate radio/compute/transport/UAV components, explicit
PA + circuit-power split, no dBm-as-energy), §11 provenance/maturity, SI units,
and deep immutability.
"""

import unittest

from config import energy as power
from config._provenance import MATURITIES

PROV_FIELDS = {"source_id", "source_uri", "source_version", "valid_from",
               "valid_to", "validity_notes", "maturity"}


class TestPowerProfiles(unittest.TestCase):
    def test_profiles_present(self):
        # Endpoint-specific radios: handset, ground BS, UAV-BS (no reuse).
        for radio in ("device_radio_default", "access_radio_default", "uav_bs_radio_default"):
            self.assertIn(radio, power.POWER_PROFILES)
            self.assertEqual(power.get_power(radio)["kind"], "radio")
        for tier in ("onboard", "mec", "metro", "cloud", "uav_bs"):
            self.assertIn(f"{tier}_compute", power.POWER_PROFILES)
        self.assertIn("transport_fibre_default", power.POWER_PROFILES)
        self.assertIn("uav_bs_mobility", power.POWER_PROFILES)

    def test_endpoint_radios_distinct_from_handset(self):
        # A base-station radio must not reuse handset circuit/radiated values.
        dev = power.get_power("device_radio_default")
        access = power.get_power("access_radio_default")
        uav = power.get_power("uav_bs_radio_default")
        self.assertGreater(access["radiated_tx_power_w"], dev["radiated_tx_power_w"])
        self.assertGreater(access["tx_circuit_power_w"], dev["tx_circuit_power_w"])
        self.assertNotEqual(uav["radiated_tx_power_w"], dev["radiated_tx_power_w"])
        for r in (dev, access, uav):
            self.assertGreaterEqual(r["radiated_tx_power_max_w"], r["radiated_tx_power_w"])

    def test_required_keys_per_kind(self):
        for name, prof in power.POWER_PROFILES.items():
            with self.subTest(profile=name):
                self.assertIn(prof["kind"], power.REQUIRED_KEYS)
                self.assertEqual(set(prof), power.REQUIRED_KEYS[prof["kind"]])

    def test_radio_has_explicit_tx_rx_and_pa_split(self):
        r = power.get_power("device_radio_default")
        # PA input power is radiated/pa_eff (formula in metrics); here we only
        # assert the SEPARATE components exist and are physical.
        self.assertGreater(r["radiated_tx_power_w"], 0.0)
        self.assertTrue(0.0 < r["pa_efficiency_ratio"] <= 1.0)
        self.assertGreater(r["tx_circuit_power_w"], 0.0)
        self.assertGreater(r["rx_circuit_power_w"], 0.0)  # explicit, not from dBm
        self.assertGreaterEqual(r["idle_power_w"], r["sleep_power_w"])

    def test_uav_mobility_separate_from_radio_and_compute(self):
        u = power.get_power("uav_bs_mobility")
        self.assertEqual(u["kind"], "uav")
        self.assertGreater(u["hover_power_w"], 0.0)
        # UAV mobility profile carries no radio/compute fields.
        self.assertNotIn("radiated_tx_power_w", u)
        self.assertNotIn("max_power_w", u)

    def test_compute_has_pue_boundary(self):
        for tier in ("mec", "metro", "cloud"):
            p = power.get_power(f"{tier}_compute")
            self.assertGreaterEqual(p["pue_ratio"], 1.0)
            self.assertGreaterEqual(p["max_power_w"], p["idle_power_w"])

    def test_all_powers_nonnegative(self):
        for prof in power.POWER_PROFILES.values():
            for k, v in prof.items():
                if k.endswith("_w") or k.endswith("_j") or k.endswith("_s"):
                    self.assertGreaterEqual(v, 0.0)

    def test_provenance_and_maturity(self):
        for name, prof in power.POWER_PROFILES.items():
            with self.subTest(profile=name):
                prov = prof["provenance"]
                self.assertEqual(set(prov), PROV_FIELDS)
                self.assertIn(prov["maturity"], MATURITIES)
                self.assertTrue(prov["source_id"])

    def test_no_default_falsely_validated(self):
        # §11: uncalibrated defaults must not claim 'validated'.
        for prof in power.POWER_PROFILES.values():
            self.assertNotEqual(prof["provenance"]["maturity"], "validated")

    def test_mixed_profiles_labeled_experimental(self):
        # §11: a profile with any uncalibrated value is `experimental` as a whole,
        # with notes naming the standards-aligned sub-value.
        for name in ("device_radio_default", "onboard_compute", "mec_compute",
                     "metro_compute", "cloud_compute"):
            prov = power.get_power(name)["provenance"]
            self.assertEqual(prov["maturity"], "experimental")
            self.assertTrue(prov["validity_notes"])

    def test_transport_accounting_method_disambiguates(self):
        t = power.get_power("transport_fibre_default")
        self.assertIn(t["accounting_method"], power.TRANSPORT_METHOD_REQUIRED_INPUT)
        required = power.TRANSPORT_METHOD_REQUIRED_INPUT[t["accounting_method"]]
        self.assertGreater(t[required], 0.0)   # the selected basis has a positive input

    def test_required_keys_schema_is_immutable(self):
        with self.assertRaises(TypeError):
            power.REQUIRED_KEYS["radio"] = frozenset()
        # frozenset has no add()
        with self.assertRaises(AttributeError):
            power.REQUIRED_KEYS["radio"].add("x")

    def test_deep_immutable(self):
        p = power.get_power("device_radio_default")
        with self.assertRaises(TypeError):
            p["radiated_tx_power_w"] = 99.0
        with self.assertRaises(TypeError):
            p["provenance"]["maturity"] = "validated"

    def test_get_power_unknown_raises(self):
        with self.assertRaises(KeyError):
            power.get_power("warp_core")


if __name__ == "__main__":
    unittest.main()
