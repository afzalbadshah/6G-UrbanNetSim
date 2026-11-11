import unittest

from network import (
    MaturityLevel,
    instantiate_link_profile,
    load_link_profile_template,
)


class LinkProfileLoadingTests(unittest.TestCase):
    def test_loads_typed_template_with_provenance(self):
        template = load_link_profile_template("nr-fr3-access")
        self.assertEqual(template.profile.band, "FR3")
        self.assertEqual(template.profile.network_mtu_bytes, 1_500)
        self.assertEqual(template.profile.header_stack.inside_mtu_bytes, 48)
        self.assertEqual(template.profile.header_stack.link_overhead_bytes, 11)
        self.assertEqual(template.provenance.maturity, MaturityLevel.EXPERIMENTAL)

    def test_instantiates_unique_physical_resources_without_changing_technology_id(self):
        first = instantiate_link_profile(
            "nr-fr1-access", shared_resource_id="cell:MEC_1"
        )
        second = instantiate_link_profile(
            "nr-fr1-access", shared_resource_id="cell:MEC_2"
        )
        self.assertEqual(first.profile_id, second.profile_id)
        self.assertNotEqual(first.shared_resource_id, second.shared_resource_id)
        self.assertEqual(first.band, second.band)

    def test_rejects_empty_resource_id(self):
        with self.assertRaises(ValueError):
            instantiate_link_profile("fibre-core", shared_resource_id="")


if __name__ == "__main__":
    unittest.main()
