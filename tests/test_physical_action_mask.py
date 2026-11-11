import unittest
from types import SimpleNamespace

from scheduler.common.common_action import ACTION_CLOUD, ACTION_LOCAL, ACTION_MEC, ACTION_METRO, ACTION_UAV_BS, get_action_dim
from scheduler.common.common_mask import CommonActionMask
from scheduler.common.common_state import CommonStateBuilder
from scheduler.common.comparison_runner import ComparisonRunner
from scheduler.common.policy_utils import NoFeasibleActionError, masked_random_action


class ServerStub:
    def __init__(self, name, tier, *, available=100.0, capacity=100.0, x=0.0, y=0.0):
        self.name = name
        self.tier = tier
        self.cpu_capacity = capacity
        self.memory_capacity = capacity
        self.storage_capacity = capacity
        self.available_cpu = available
        self.available_memory = available
        self.available_storage = available
        self.compute_rate_cycles_per_s = capacity
        self.available_compute_rate_cycles_per_s = available
        self.memory_capacity_bytes = int(capacity)
        self.available_memory_bytes = int(available)
        self.storage_capacity_bytes = int(capacity)
        self.available_storage_bytes = int(available)
        self.battery_backed = False
        self.recharging = False
        self.x = float(x)
        self.y = float(y)

    def can_allocate(self, cpu, storage, memory):
        return (
            self.available_cpu >= cpu
            and self.available_storage >= storage
            and self.available_memory >= memory
        )


def task(priority=1):
    return SimpleNamespace(
        priority=priority,
        deadline_ms=300.0,
        application_deadline_ms=300.0,
        wireless_ap="BS_1",
        local_compute_capable=False,
        cpu_demand=10.0,
        memory_demand=10.0,
        storage_demand=10.0,
        compute_rate_demand_cycles_per_s=10.0,
        compute_work_cycles=10.0,
        resource_allocation=None,
        battery_capacity_j=100.0,
        battery_remaining_j=100.0,
        device_id=1,
    )


class PhysicalActionMaskTests(unittest.TestCase):
    def builder(self, destinations):
        builder = CommonStateBuilder()
        builder.set_reachability({"BS_1": set(destinations)})
        return builder

    def test_disabled_metro_masks_only_metro(self):
        cloud = ServerStub("Cloud_1", "CLOUD")
        _, context = self.builder({"Cloud_1", "Metro_1"}).build(task(), [cloud])

        self.assertEqual(context["physical_mask"][ACTION_METRO], 0)
        self.assertEqual(context["physical_mask"][ACTION_CLOUD], 1)

    def test_enabling_reachable_metro_restores_action(self):
        cloud = ServerStub("Cloud_1", "CLOUD")
        metro = ServerStub("Metro_1", "METRO")
        _, context = self.builder({"Cloud_1", "Metro_1"}).build(
            task(), [cloud, metro]
        )

        self.assertEqual(context["physical_mask"][ACTION_METRO], 1)
        self.assertEqual(context["physical_mask"][ACTION_CLOUD], 1)

    def test_full_server_is_not_physically_admissible(self):
        cloud = ServerStub("Cloud_1", "CLOUD", available=5.0)
        _, context = self.builder({"Cloud_1"}).build(task(), [cloud])

        self.assertEqual(context["physical_mask"][ACTION_CLOUD], 0)

    def test_physical_mask_does_not_depend_on_service_class(self):
        servers = [ServerStub("Cloud_1", "CLOUD"), ServerStub("Metro_1", "METRO")]
        builder = self.builder({"Cloud_1", "Metro_1"})
        masks = [builder.build(task(priority), servers)[1]["physical_mask"] for priority in (1, 2, 3)]

        self.assertEqual(masks[0], masks[1])
        self.assertEqual(masks[1], masks[2])

    def test_uav_route_exposes_near_edge_mec_action(self):
        mec = ServerStub("MEC_1", "MEC")
        uav_task = task(priority=2)
        uav_task.wireless_ap = "UAV_BS_1"
        builder = CommonStateBuilder()
        builder.set_reachability({"UAV_BS_1": {"MEC_1"}})

        _, context = builder.build(uav_task, [mec])

        self.assertEqual(context["physical_mask"][ACTION_MEC], 1)

    def test_access_only_bs_position_selects_nearest_reachable_uav(self):
        far = ServerStub("UAV_BS_1", "UAV_BS", x=100.0, y=100.0)
        near = ServerStub("UAV_BS_2", "UAV_BS", x=900.0, y=900.0)
        uav_task = task(priority=2)
        uav_task.wireless_ap_x_m = 850.0
        uav_task.wireless_ap_y_m = 850.0
        builder = self.builder({"UAV_BS_1", "UAV_BS_2"})

        _, context = builder.build(uav_task, [far, near])

        self.assertEqual(context["physical_mask"][ACTION_UAV_BS], 1)
        self.assertIs(context["action_servers"][ACTION_UAV_BS], near)

    def test_sufficient_mec_and_cloud_have_equal_capacity_adequacy(self):
        mec = ServerStub("MEC_1", "MEC", capacity=20.0)
        cloud = ServerStub("Cloud_1", "CLOUD", capacity=1_000.0)
        _, context = self.builder({"MEC_1", "Cloud_1"}).build(
            task(priority=2), [mec, cloud]
        )

        self.assertEqual(context["capacity_adequacy"][ACTION_MEC], 1.0)
        self.assertEqual(context["capacity_adequacy"][ACTION_CLOUD], 1.0)

    def test_all_zero_mask_is_preserved_and_policy_helper_rejects_it(self):
        zero = [0] * get_action_dim()
        self.assertEqual(CommonActionMask().build({"physical_mask": zero}), zero)
        with self.assertRaises(NoFeasibleActionError):
            masked_random_action(zero, get_action_dim())

    def test_runner_keeps_a_selected_tier_below_spillover_threshold(self):
        runner = ComparisonRunner("random")
        metro = ServerStub("Metro_1", "METRO")
        cloud = ServerStub("Cloud_1", "CLOUD")
        mask = [0] * get_action_dim()
        mask[ACTION_METRO] = 1
        mask[ACTION_CLOUD] = 1
        context = {
            "mask": mask,
            "action_servers": {ACTION_METRO: metro, ACTION_CLOUD: cloud},
            "future_util": {ACTION_METRO: 50.0, ACTION_CLOUD: 50.0},
        }

        action, server, route = runner._select_allocatable_action(
            ACTION_CLOUD, context, task()
        )
        self.assertEqual(action, ACTION_CLOUD)
        self.assertIs(server, cloud)
        self.assertEqual(route, [ACTION_CLOUD])

        mask[ACTION_CLOUD] = 0
        action, server, route = runner._select_allocatable_action(
            ACTION_CLOUD, context, task()
        )
        self.assertIsNone(action)
        self.assertIsNone(server)
        self.assertEqual(route, [ACTION_CLOUD])

    def test_runner_spills_saturated_mec_outward_and_skips_missing_uav(self):
        runner = ComparisonRunner("random")
        mec = ServerStub("MEC_1", "MEC")
        metro = ServerStub("Metro_1", "METRO")
        cloud = ServerStub("Cloud_1", "CLOUD")
        mask = [0] * get_action_dim()
        mask[ACTION_MEC] = 1
        mask[ACTION_METRO] = 1
        mask[ACTION_CLOUD] = 1
        context = {
            "mask": mask,
            "action_servers": {
                ACTION_MEC: mec,
                ACTION_METRO: metro,
                ACTION_CLOUD: cloud,
            },
            "future_util": {
                ACTION_MEC: 96.0,
                ACTION_METRO: 60.0,
                ACTION_CLOUD: 10.0,
            },
        }

        action, server, route = runner._select_allocatable_action(
            ACTION_MEC, context, task()
        )

        self.assertEqual(action, ACTION_METRO)
        self.assertIs(server, metro)
        self.assertEqual(route, [ACTION_MEC, ACTION_UAV_BS, ACTION_METRO])

    def test_runner_uses_cloud_when_intermediate_tiers_are_unavailable(self):
        runner = ComparisonRunner("random")
        mec = ServerStub("MEC_1", "MEC")
        cloud = ServerStub("Cloud_1", "CLOUD")
        mask = [0] * get_action_dim()
        mask[ACTION_MEC] = 1
        mask[ACTION_CLOUD] = 1
        context = {
            "mask": mask,
            "action_servers": {ACTION_MEC: mec, ACTION_CLOUD: cloud},
            "future_util": {ACTION_MEC: 99.0, ACTION_CLOUD: 99.0},
        }

        results = []
        for train_mode in (True, False):
            runner.set_train_mode(train_mode)
            results.append(
                runner._select_allocatable_action(ACTION_MEC, context, task())
            )

        for action, server, route in results:
            self.assertEqual(action, ACTION_CLOUD)
            self.assertIs(server, cloud)
            self.assertEqual(
                route,
                [ACTION_MEC, ACTION_UAV_BS, ACTION_METRO, ACTION_CLOUD],
            )

    def test_active_onboard_work_spills_local_selection_to_mec(self):
        runner = ComparisonRunner("random")
        local_task = task()
        local_task.local_compute_capable = True
        local_task.local_compute_rate_cycles_per_s = 100.0
        local_task.local_memory_capacity_bytes = 100
        local_task.resource_allocation = SimpleNamespace(
            allocated_compute_rate_cycles_per_s=10.0,
            memory_allocated_bytes=10,
        )
        runner._local_resource_allocations["1"] = {
            "still-running": (90.0, 90, 1_000.0)
        }
        runner._apply_local_resource_state(local_task)
        self.assertEqual(local_task.local_compute_utilization_percent, 90.0)

        mec = ServerStub("MEC_1", "MEC")
        mask = [0] * get_action_dim()
        mask[ACTION_LOCAL] = 1
        mask[ACTION_MEC] = 1
        context = {
            "mask": mask,
            "action_servers": {ACTION_MEC: mec},
            "future_util": {ACTION_LOCAL: 100.0, ACTION_MEC: 40.0},
        }

        action, server, route = runner._select_allocatable_action(
            ACTION_LOCAL, context, local_task
        )

        self.assertEqual(action, ACTION_MEC)
        self.assertIs(server, mec)
        self.assertEqual(route, [ACTION_LOCAL, ACTION_MEC])

    def test_csv_row_distinguishes_policy_choice_from_spillover_execution(self):
        runner = ComparisonRunner("random")
        selected_mec = ServerStub("MEC_1", "MEC", available=4.0)
        executed_metro = ServerStub("Metro_1", "METRO")
        mask = [0] * get_action_dim()
        mask[ACTION_MEC] = 1
        mask[ACTION_METRO] = 1
        context = {
            "mask": mask,
            "physical_mask": mask,
            "learning_mask": mask,
            "action_servers": {
                ACTION_MEC: selected_mec,
                ACTION_METRO: executed_metro,
            },
            "future_util": {ACTION_MEC: 96.0, ACTION_METRO: 30.0},
            "current_util": {ACTION_MEC: 96.0, ACTION_METRO: 20.0},
            "total_delay": {ACTION_METRO: 10.0},
            "cost": {ACTION_METRO: 0.0},
            "cost_breakdown": {ACTION_METRO: {}},
            "energy": {ACTION_METRO: 0.0},
            "energy_breakdown": {ACTION_METRO: {}},
            "delay_components": {ACTION_METRO: {}},
            "deadline_ms": 300.0,
        }

        row = runner._row(
            task(),
            ACTION_MEC,
            ACTION_METRO,
            executed_metro,
            [ACTION_MEC, ACTION_UAV_BS, ACTION_METRO],
            context,
            mask,
            reward=1.0,
            status="assigned",
            reward_info={
                "overutilization_pen": 0.04,
                "overutilization_weight": 2.0,
            },
        )

        self.assertEqual(row["selected_tier"], "MEC")
        self.assertEqual(row["rl_action"], ACTION_MEC)
        self.assertEqual(row["execution_tier"], "METRO")
        self.assertEqual(row["action_final"], ACTION_METRO)
        self.assertEqual(row["overflow_used"], 1)
        self.assertEqual(row["overflow_route"], "MEC->UAV_BS->METRO")
        self.assertEqual(row["selected_projected_utilization_percent"], 96.0)
        self.assertEqual(row["reward_overutilization_penalty"], 0.04)


if __name__ == "__main__":
    unittest.main()
