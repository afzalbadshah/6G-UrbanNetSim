import unittest

from config.radio import LINK_PROFILES, get_link_profile


class LinkProfileConfigTests(unittest.TestCase):
    def test_current_future_and_wired_profiles_are_published(self):
        self.assertEqual(
            set(LINK_PROFILES),
            {
                "nr-lowband-access",
                "nr-fr1-access",
                "nr-fr1-air",
                "nr-mmwave-air",
                "nr-fr3-access",
                "nr-mmwave-access",
                "nr-subthz-access",
                "nr-mmwave-backhaul",
                "fibre-backhaul",
                "fibre-core",
            },
        )
        self.assertEqual(get_link_profile("nr-subthz-access")["band"], "subTHz")
        self.assertEqual(get_link_profile("fibre-core")["channel_model_id"], "fixed-rate-wired-v1")

    def test_profiles_and_nested_provenance_are_immutable(self):
        profile = get_link_profile("nr-fr1-access")
        with self.assertRaises(TypeError):
            profile["network_mtu_bytes"] = 9_000
        with self.assertRaises(TypeError):
            profile["header_stack"]["network_header_bytes"] = 20
        with self.assertRaises(TypeError):
            profile["provenance"]["maturity"] = "validated"

    def test_header_and_unit_invariants(self):
        for profile in LINK_PROFILES.values():
            header = profile["header_stack"]
            inside = sum(
                header[key]
                for key in (
                    "network_header_bytes",
                    "transport_header_bytes",
                    "security_overhead_bytes",
                    "telemetry_overhead_bytes",
                )
            )
            self.assertGreater(profile["network_mtu_bytes"], inside)
            self.assertGreater(profile["nominal_bit_rate_bps"], 0.0)
            self.assertGreater(profile["propagation_speed_m_s"], 0.0)
            self.assertEqual(profile["provenance"]["maturity"], "experimental")

    def test_unknown_profile_fails_loudly(self):
        with self.assertRaises(KeyError):
            get_link_profile("telepathy")


if __name__ == "__main__":
    unittest.main()
