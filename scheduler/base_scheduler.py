# scheduler/base_scheduler.py

from abc import ABC, abstractmethod

class BaseScheduler(ABC):
    """
    Abstract base class for all scheduling algorithms.
    Every scheduler must implement the `schedule` method.
    """

    @abstractmethod
    def schedule(self, tasks, servers, current_time):
        """
        Assigns tasks to servers based on the implemented algorithm.
        
        Args:
            tasks (list): List of Task objects to be scheduled.
            servers (list): List of Server objects (MEC, Regional, Cloud).
            current_time (float): Current simulation time.
        
        Returns:
            list: List of (task, server) tuples representing assignments.
        """
        pass

    def set_execution_runtime(self, execution_runtime):
        """Inject the chosen-action executor used before learning/reporting.

        The callable receives ``task``, ``server``, ``current_time``, ``action``
        and the decision ``context`` as keyword arguments and returns one
        immutable realized execution result.  Schedulers that do not opt into
        realized feedback can leave this unset for legacy compatibility.
        """
        if execution_runtime is not None and not callable(execution_runtime):
            raise ValueError("execution_runtime must be callable or None")
        self._execution_runtime = execution_runtime

    def log_assignment(self, task, server):
        """
        Optional utility method to log or print task assignment.
        """
        #print(f"📦 Assigned Task {task.id} → {server.name}")
