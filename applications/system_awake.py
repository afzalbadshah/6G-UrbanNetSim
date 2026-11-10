"""Keep a long simulator campaign running while the display may turn off.

This module deliberately contains one public function.  The application
launcher calls it once around the complete training/evaluation campaign; the
simulator, schedulers, and individual rounds do not need power-management code.
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Callable
from typing import TypeVar


Result = TypeVar("Result")


def run_while_system_awake(
    run_campaign: Callable[[], Result],
    *,
    logger: Callable[[str], object] = print,
) -> Result:
    """Run one complete campaign without allowing Windows idle sleep.

    Windows receives ``ES_SYSTEM_REQUIRED`` for the lifetime of
    ``run_campaign``.  ``ES_DISPLAY_REQUIRED`` is intentionally omitted, so the
    monitor can follow its normal timeout and turn off.  The request is released
    in ``finally`` after results, checkpoints, and plots finish, including when
    the campaign raises an exception.

    Other operating systems run the campaign unchanged.  A Windows API failure
    stops before training begins rather than silently running an unprotected
    multi-hour experiment.
    """

    if os.name != "nt":
        return run_campaign()

    execution_state_continuous = 0x80000000
    execution_state_system_required = 0x00000001

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    set_execution_state = kernel32.SetThreadExecutionState
    set_execution_state.argtypes = [ctypes.c_ulong]
    set_execution_state.restype = ctypes.c_ulong

    keep_awake_flags = execution_state_continuous | execution_state_system_required
    if not set_execution_state(keep_awake_flags):
        error_code = ctypes.get_last_error()
        raise OSError(
            error_code,
            "Windows could not prevent system sleep for the simulator campaign.",
        )

    logger("[POWER] System sleep is suspended for this campaign; the display may turn off.")
    try:
        return run_campaign()
    finally:
        if set_execution_state(execution_state_continuous):
            logger("[POWER] Campaign finished; normal Windows sleep policy is restored.")
        else:
            error_code = ctypes.get_last_error()
            logger(
                "[POWER] WARNING: Windows did not confirm release of the sleep request "
                f"(error {error_code})."
            )
