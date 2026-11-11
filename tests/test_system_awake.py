import unittest
from unittest.mock import MagicMock, patch

from applications import MainApplication
from applications.system_awake import run_while_system_awake


class TestSystemAwake(unittest.TestCase):
    def test_non_windows_runs_campaign_without_power_api(self):
        campaign = MagicMock(return_value="finished")

        with patch("applications.system_awake.os.name", "posix"):
            result = run_while_system_awake(campaign, logger=MagicMock())

        self.assertEqual(result, "finished")
        campaign.assert_called_once_with()

    def test_windows_acquires_system_only_and_releases_after_success(self):
        campaign = MagicMock(return_value="finished")
        logger = MagicMock()
        set_execution_state = MagicMock(side_effect=[1, 1])
        kernel32 = MagicMock()
        kernel32.SetThreadExecutionState = set_execution_state

        with patch("applications.system_awake.os.name", "nt"), patch(
            "applications.system_awake.ctypes.WinDLL", return_value=kernel32
        ):
            result = run_while_system_awake(campaign, logger=logger)

        self.assertEqual(result, "finished")
        campaign.assert_called_once_with()
        self.assertEqual(
            [call.args[0] for call in set_execution_state.call_args_list],
            [0x80000001, 0x80000000],
        )

    def test_windows_releases_request_when_campaign_fails(self):
        set_execution_state = MagicMock(side_effect=[1, 1])
        kernel32 = MagicMock()
        kernel32.SetThreadExecutionState = set_execution_state

        with patch("applications.system_awake.os.name", "nt"), patch(
            "applications.system_awake.ctypes.WinDLL", return_value=kernel32
        ):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                run_while_system_awake(
                    lambda: (_ for _ in ()).throw(RuntimeError("training failed")),
                    logger=MagicMock(),
                )

        self.assertEqual(set_execution_state.call_args_list[-1].args[0], 0x80000000)

    def test_windows_refuses_to_start_when_sleep_protection_fails(self):
        campaign = MagicMock()
        kernel32 = MagicMock()
        kernel32.SetThreadExecutionState = MagicMock(return_value=0)

        with patch("applications.system_awake.os.name", "nt"), patch(
            "applications.system_awake.ctypes.WinDLL", return_value=kernel32
        ), patch("applications.system_awake.ctypes.get_last_error", return_value=5):
            with self.assertRaisesRegex(OSError, "could not prevent system sleep"):
                run_while_system_awake(campaign, logger=MagicMock())

        campaign.assert_not_called()

    def test_application_calls_one_keep_awake_boundary(self):
        with patch.object(MainApplication, "run_configured_models") as campaign, patch.object(
            MainApplication, "run_while_system_awake"
        ) as keep_awake:
            MainApplication.main()

        keep_awake.assert_called_once_with(campaign, logger=MainApplication.print)


if __name__ == "__main__":
    unittest.main()
