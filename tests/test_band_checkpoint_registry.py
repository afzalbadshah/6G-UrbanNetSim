"""Regression tests for deployment- and band-specific training checkpoints."""

import json
import tempfile
import unittest
from pathlib import Path

from core.checkpoint_registry import (
    radio_checkpoint_identity,
    radio_environment_dir,
    validate_radio_environment_manifest,
    write_radio_environment_manifest,
)
from mobility.handover_scheduler import HandoverScheduler
from scheduler.common.comparison_runner import ComparisonRunner


class BandCheckpointRegistryTests(unittest.TestCase):
    def test_identity_records_every_access_band_and_frequency(self):
        fr1 = radio_checkpoint_identity("fr1_city_coverage")
        fr3 = radio_checkpoint_identity("fr3_city_coverage_6g")
        millimetre_wave = radio_checkpoint_identity("mmwave_hotspot_overlay")

        self.assertNotEqual(fr1.directory_name, fr3.directory_name)
        self.assertNotEqual(fr3.directory_name, millimetre_wave.directory_name)
        self.assertEqual(
            fr1.metadata["radio_roles"]["ground"]["carrier_frequency_hz"],
            3.5e9,
        )
        self.assertEqual(
            fr3.metadata["radio_roles"]["ground"]["carrier_frequency_hz"],
            13.0e9,
        )
        self.assertEqual(
            millimetre_wave.metadata["radio_roles"]["hotspot"]["carrier_frequency_hz"],
            28.0e9,
        )

    def test_manifest_round_trip_and_mismatch_rejection(self):
        with tempfile.TemporaryDirectory(prefix="radio_checkpoint_") as folder:
            identity = radio_checkpoint_identity("fr3_city_coverage_6g")
            manifest = write_radio_environment_manifest(identity, folder)
            self.assertEqual(
                validate_radio_environment_manifest(identity, folder),
                manifest,
            )

            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["radio_environment"]["radio_roles"]["ground"][
                "carrier_frequency_hz"
            ] = 28.0e9
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                validate_radio_environment_manifest(identity, folder)

    def test_offloading_policy_loads_only_its_radio_environment(self):
        with tempfile.TemporaryDirectory(prefix="radio_checkpoint_") as folder:
            trained = ComparisonRunner(
                "q_learning",
                deployment_profile_id="fr3_city_coverage_6g",
                checkpoint_root=folder,
            )
            checkpoint = trained.save()
            self.assertTrue(checkpoint.is_file())
            self.assertIn("ground-fr3", str(checkpoint))

            matching = ComparisonRunner(
                "q_learning",
                deployment_profile_id="fr3_city_coverage_6g",
                checkpoint_root=folder,
            )
            self.assertEqual(matching.load(), checkpoint)

            wrong_band = ComparisonRunner(
                "q_learning",
                deployment_profile_id="fr1_city_coverage",
                checkpoint_root=folder,
            )
            with self.assertRaisesRegex(FileNotFoundError, "radio environment"):
                wrong_band.load()

    def test_handover_policy_loads_only_its_radio_environment(self):
        with tempfile.TemporaryDirectory(prefix="radio_checkpoint_") as folder:
            trained = HandoverScheduler(
                [],
                deployment_profile_id="mmwave_hotspot_overlay",
                checkpoint_root=folder,
            )
            actor_path, critic_path = trained.save()
            self.assertTrue(actor_path.is_file())
            self.assertTrue(critic_path.is_file())
            self.assertEqual(
                actor_path.parent.parent,
                radio_environment_dir(trained.checkpoint_identity, folder),
            )

            matching = HandoverScheduler(
                [],
                deployment_profile_id="mmwave_hotspot_overlay",
                checkpoint_root=folder,
            )
            self.assertEqual(matching.load(strict=True), (actor_path, critic_path))

            wrong_band = HandoverScheduler(
                [],
                deployment_profile_id="fr3_city_coverage_6g",
                checkpoint_root=folder,
            )
            with self.assertRaisesRegex(FileNotFoundError, "radio environment"):
                wrong_band.load(strict=True)


if __name__ == "__main__":
    unittest.main()
