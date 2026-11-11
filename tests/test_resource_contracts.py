"""Unit tests for canonical compute-resource values and conversions."""

import unittest

from config.units import LEGACY_V1_RESOURCE_CONVERSION
from entities.resources import (
    ComputeCapacity,
    ComputeWork,
    LegacyResourceConversionProfile,
    ResourceAllocation,
    bounded_utilization_ratio,
    offered_load_ratio,
    processing_duration_s,
)
from entities.server import Server
from entities.task import Task
from scheduler.common.common_servers import has_capacity, processing_time_ms
from config.devices_config import DEVICE_TYPE_PROFILES
from workload.generator import WorkloadGenerator


class ResourceContractTests(unittest.TestCase):
    """Verify units, validation, and dimensionally correct formulas."""

    def test_processing_duration_is_cycles_divided_by_cycles_per_second(self):
        self.assertEqual(processing_duration_s(1_600_000_000, 3_200_000_000), 0.5)

    def test_offered_load_uses_bits_rate_and_explicit_window(self):
        self.assertEqual(offered_load_ratio(20_000_000, 10_000_000, 2.0), 1.0)
        self.assertEqual(offered_load_ratio(30_000_000, 10_000_000, 2.0), 1.5)

    def test_reported_utilization_is_bounded(self):
        self.assertEqual(bounded_utilization_ratio(150.0, 100.0), 1.0)

    def test_invalid_quantities_fail_closed(self):
        with self.assertRaises(ValueError):
            ComputeWork(-1.0, 0, 0)
        with self.assertRaises(ValueError):
            ComputeCapacity(0.0, 1, 1)
        with self.assertRaises(ValueError):
            ResourceAllocation(1.0, 1.5, 1)
        with self.assertRaises(ValueError):
            processing_duration_s(1.0, 0.0)

    def test_capacity_checks_rate_memory_and_storage(self):
        capacity = ComputeCapacity(3_000_000_000.0, 2_000_000_000, 4_000_000_000)
        fitting = ResourceAllocation(1_600_000_000.0, 80_000_000, 400_000)
        too_large = ResourceAllocation(3_100_000_000.0, 80_000_000, 400_000)
        self.assertTrue(capacity.can_host(fitting))
        self.assertFalse(capacity.can_host(too_large))

    def test_named_legacy_profile_makes_every_scale_explicit(self):
        profile = LEGACY_V1_RESOURCE_CONVERSION
        work = profile.work_from_legacy(
            cpu_demand=1_600,
            memory_demand=80,
            storage_demand=400,
        )
        capacity = profile.capacity_from_legacy(
            cpu_capacity=3_000,
            memory_capacity=2_048,
            storage_capacity=4_000,
        )
        self.assertEqual(work.compute_work_cycles, 1_600_000_000.0)
        self.assertEqual(work.memory_working_set_bytes, 80_000_000)
        self.assertEqual(capacity.compute_rate_cycles_per_s, 3_000_000_000.0)
        self.assertAlmostEqual(profile.cpu_seconds_for(work), 1.6)

    def test_conversion_profile_rejects_missing_physical_scale(self):
        with self.assertRaises(ValueError):
            LegacyResourceConversionProfile(
                profile_id="bad",
                compute_work_cycles_per_legacy_unit=0.0,
                compute_rate_cycles_per_s_per_legacy_unit=1.0,
                memory_bytes_per_legacy_unit=1,
                storage_bytes_per_legacy_unit=1,
                reservation_window_s=1.0,
                reference_vcpu_rate_cycles_per_s=1.0,
                provenance_id="test",
            )

    def test_task_server_admission_timing_and_release_use_canonical_units(self):
        task = Task(
            cpu_demand=1_500,
            memory_demand=80,
            storage_demand=400,
            bandwidth=400,
            latency=None,
            compute_work_cycles=1_500_000_000.0,
            memory_working_set_bytes=80_000_000,
            storage_footprint_bytes=400_000,
            input_payload_bytes=400_000,
        )
        server = Server(
            name="MEC_1",
            index=0,
            cpu_capacity=3_000,
            memory_capacity=2_000,
            storage_capacity=4_000,
            bandwidth=1_000,
            latency=0,
            tier="MEC",
        )
        self.assertTrue(has_capacity(server, task))
        self.assertEqual(processing_time_ms(server, task), 500.0)
        self.assertTrue(
            server.allocate(
                task.id,
                task.cpu_demand,
                task.storage_demand,
                task.memory_demand,
                release_time=1.0,
                data_size_kb=task.data_size_kb,
                resource_demand=task.resource_demand,
            )
        )
        cpu_percent, storage_percent, memory_percent = server.utilization()
        self.assertEqual(cpu_percent, 50.0)
        self.assertEqual(memory_percent, 4.0)
        self.assertEqual(storage_percent, 0.01)
        self.assertEqual(server.congestion(), 50.0)
        server.release_completed_tasks(1.0)
        self.assertEqual(server.utilization(), (0.0, 0.0, 0.0))

    def test_repeated_fractional_allocations_restore_exact_idle_capacity(self):
        server = Server(
            name="MEC_1",
            index=0,
            cpu_capacity=3_000,
            memory_capacity=2_000,
            storage_capacity=4_000,
            bandwidth=1_000,
            latency=0,
            tier="MEC",
        )
        for index in range(2_000):
            self.assertTrue(
                server.allocate(
                    f"task-{index}",
                    cpu_demand=0.1,
                    storage_demand=0.1,
                    memory_demand=0.1,
                    release_time=float(index),
                    data_size_kb=0.1,
                )
            )
            server.release_completed_tasks(float(index))

        self.assertEqual(
            server.available_compute_rate_cycles_per_s,
            server.compute_rate_cycles_per_s,
        )
        self.assertEqual(server.available_memory_bytes, server.memory_capacity_bytes)
        self.assertEqual(server.available_storage_bytes, server.storage_capacity_bytes)
        self.assertEqual(server.available_cpu, float(server.cpu_capacity))
        self.assertEqual(server.available_memory, float(server.memory_capacity))
        self.assertEqual(server.available_storage, float(server.storage_capacity))

    def test_workload_generator_separates_payload_from_bandwidth(self):
        generator = WorkloadGenerator()
        profile = {
            "service_class": "mmtc",
            "data_multiplier": 1.0,
            "cpu_multiplier": 1.0,
            "local_compute_capable": False,
        }
        task = generator._task_from_profile(1, "sensor", profile, round_no=1)
        self.assertEqual(task.input_payload_bytes, round(task.data_size_kb * 1000.0))
        self.assertEqual(task.storage_footprint_bytes, task.input_payload_bytes)
        self.assertGreater(task.compute_work_cycles, 0.0)

    def test_mmtc_uses_one_canonical_physical_workload_for_every_device(self):
        generator = WorkloadGenerator()
        for device_type in (
            "environment_sensor",
            "smart_meter",
            "smart_parking_sensor",
            "street_light_controller",
        ):
            task = generator._task_from_profile(
                1,
                device_type,
                DEVICE_TYPE_PROFILES[device_type],
                round_no=1,
            )
            self.assertEqual(task.data_size_kb, 2.0)
            self.assertEqual(task.input_payload_bytes, 2_000)
            self.assertEqual(task.compute_work_cycles, 40_000_000.0)
            self.assertEqual(task.memory_working_set_bytes, 2_000_000)
            self.assertTrue(task.local_compute_capable)


if __name__ == "__main__":
    unittest.main()
