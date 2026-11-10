import queue
import threading
import time
import tkinter as tk
from tkinter import ttk


class TrainingProgressWindow:
    """DriveNet-compatible non-blocking training progress window."""

    def __init__(self, models, total_rounds):
        self.models = [str(model).strip().lower() for model in (models or [])]
        self.total_rounds = max(1, int(total_rounds or 1))
        self.events = queue.Queue()
        self.root = None
        self._thread = None
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._started = False
        self._widgets = {}

    def start(self):
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._run_ui, name="SmartCityTrainingProgress", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=3.0):
            raise RuntimeError("training progress window did not start")

    def update(self, **payload):
        self.events.put({"type": "update", **dict(payload)})

    def model_started(self, model_name, model_index, total_rounds=None):
        self.events.put({
            "type": "model_started",
            "model_name": model_name,
            "model_index": model_index,
            "round_id": 0,
            "total_rounds": total_rounds or self.total_rounds,
            "message": "started",
        })

    def model_finished(self, model_name, model_index, total_rounds=None):
        self.events.put({
            "type": "model_finished",
            "model_name": model_name,
            "model_index": model_index,
            "round_id": total_rounds or self.total_rounds,
            "total_rounds": total_rounds or self.total_rounds,
            "message": "finished",
        })

    def complete(self, close_after_ms=2500):
        self.events.put({"type": "complete", "message": "training complete", "close_after_ms": close_after_ms})

    def fail(self, message="Training stopped", close_after_ms=5000):
        self.events.put({"type": "fail", "message": message, "close_after_ms": close_after_ms})

    def wait_until_closed(self, timeout_s=None):
        self._closed.wait(timeout=timeout_s)

    def mainloop(self):
        self.wait_until_closed()

    def close(self):
        self.events.put({"type": "close"})

    def _run_ui(self):
        try:
            self.root = tk.Tk()
            self.root.title("SmartCityEdgeSim Training")
            self.root.geometry("620x420")
            self.root.minsize(560, 360)
            self.root.protocol("WM_DELETE_WINDOW", self._close_ui)
            self._build()
            self._ready.set()
            self.root.after(80, self._pump)
            self.root.mainloop()
        finally:
            self._closed.set()

    def _build(self):
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)

        title = ttk.Label(outer, text="Training Progress", font=("Segoe UI", 14, "bold"))
        title.grid(row=0, column=0, sticky="w")

        self._widgets["status_var"] = tk.StringVar(value="Ready")
        self._widgets["model_var"] = tk.StringVar(value="Waiting")
        self._widgets["round_var"] = tk.StringVar(value=f"0 / {self.total_rounds}")
        self._widgets["overall_var"] = tk.StringVar(value=f"0 / {len(self.models)} models")

        info = ttk.Frame(outer)
        info.grid(row=1, column=0, sticky="ew", pady=(12, 8))
        info.columnconfigure(1, weight=1)
        ttk.Label(info, text="Model").grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Label(info, textvariable=self._widgets["model_var"]).grid(row=0, column=1, sticky="w")
        ttk.Label(info, text="Round").grid(row=1, column=0, sticky="w", padx=(0, 10))
        ttk.Label(info, textvariable=self._widgets["round_var"]).grid(row=1, column=1, sticky="w")
        ttk.Label(info, text="Overall").grid(row=2, column=0, sticky="w", padx=(0, 10))
        ttk.Label(info, textvariable=self._widgets["overall_var"]).grid(row=2, column=1, sticky="w")
        ttk.Label(info, text="Status").grid(row=3, column=0, sticky="w", padx=(0, 10))
        ttk.Label(info, textvariable=self._widgets["status_var"]).grid(row=3, column=1, sticky="w")

        self._widgets["round_progress"] = ttk.Progressbar(
            outer,
            maximum=self.total_rounds,
            mode="determinate",
        )
        self._widgets["round_progress"].grid(row=2, column=0, sticky="ew", pady=(0, 12))

        list_frame = ttk.Frame(outer)
        list_frame.grid(row=3, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        tree = ttk.Treeview(list_frame, columns=("model", "status", "round"), show="headings", height=8)
        tree.heading("model", text="Model")
        tree.heading("status", text="Status")
        tree.heading("round", text="Round")
        tree.column("model", width=180, anchor="w")
        tree.column("status", width=160, anchor="w")
        tree.column("round", width=100, anchor="center")
        tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        tree.configure(yscrollcommand=scrollbar.set)

        model_ids = {}
        for index, model in enumerate(self.models):
            item_id = tree.insert("", tk.END, values=(model, "waiting", f"0/{self.total_rounds}"))
            model_ids[model] = item_id
        self._widgets["tree"] = tree
        self._widgets["model_ids"] = model_ids
        self._widgets["finished_models"] = set()

    def _pump(self):
        try:
            while True:
                payload = self.events.get_nowait()
                self._handle_event(payload)
        except queue.Empty:
            pass
        if self.root is not None:
            self.root.after(80, self._pump)

    def _handle_event(self, payload):
        event_type = payload.get("type", "update")
        if event_type == "close":
            self._close_ui()
            return

        if event_type in ("complete", "fail"):
            self._widgets["status_var"].set(str(payload.get("message") or event_type))
            if event_type == "complete":
                self._widgets["round_progress"]["value"] = self.total_rounds
            close_after_ms = payload.get("close_after_ms")
            if close_after_ms is not None and self.root is not None:
                self.root.after(max(0, int(close_after_ms)), self._close_ui)
            return

        model = str(payload.get("model_name") or "").strip().lower()
        model_index = int(payload.get("model_index", 0) or 0)
        round_id = int(payload.get("round_id", 0) or 0)
        total_rounds = max(1, int(payload.get("total_rounds", self.total_rounds) or self.total_rounds))
        message = payload.get("message") or payload.get("event") or event_type

        if model:
            self._widgets["model_var"].set(model)
        self._widgets["round_var"].set(f"{min(round_id, total_rounds)} / {total_rounds}")
        self._widgets["overall_var"].set(f"{min(model_index + 1, len(self.models))} / {len(self.models)} models")
        self._widgets["status_var"].set(str(message))
        self._widgets["round_progress"].configure(maximum=total_rounds)
        self._widgets["round_progress"]["value"] = min(round_id, total_rounds)

        if event_type == "model_finished" and model:
            self._widgets["finished_models"].add(model)
        self._update_tree(model, event_type, round_id, total_rounds, message)

    def _update_tree(self, model, event_type, round_id, total_rounds, message):
        if not model:
            return
        tree = self._widgets.get("tree")
        model_ids = self._widgets.get("model_ids", {})
        item_id = model_ids.get(model)
        if tree is None or item_id is None:
            return
        status = str(message)
        if event_type == "model_started":
            status = "running"
        elif event_type == "model_finished":
            status = "finished"
        tree.item(item_id, values=(model, status, f"{min(round_id, total_rounds)}/{total_rounds}"))
        tree.see(item_id)

    def _close_ui(self):
        if self.root is None:
            self._closed.set()
            return
        root = self.root
        self.root = None
        try:
            root.destroy()
        except tk.TclError:
            pass


def start_training_progress(models, total_rounds, enabled=True, logger=print):
    if not bool(enabled):
        logger("[TRAINING GUI] Disabled by TRAINING_GUI=False.")
        return None
    try:
        progress = TrainingProgressWindow(models=models, total_rounds=total_rounds)
        progress.start()
        logger("[TRAINING GUI] Progress window active.")
        return progress
    except Exception as exc:
        logger(f"[TRAINING GUI] Progress window disabled: {exc}")
        return None


def finish_training_progress(progress, failed=False, message=None):
    if progress is None:
        return
    if failed:
        progress.fail(message or "Training stopped", close_after_ms=5000)
        progress.wait_until_closed(timeout_s=6.0)
    else:
        progress.complete(close_after_ms=2500)
        progress.wait_until_closed(timeout_s=3.5)


def make_training_progress_callback(progress, model_name, model_index, total_rounds):
    if progress is None:
        return None

    def _callback(**payload):
        progress.update(
            model_name=payload.get("model_name", model_name),
            model_index=model_index,
            round_id=payload.get("round_id", 0),
            total_rounds=payload.get("total_rounds", total_rounds),
            event=payload.get("event", "round"),
            message=payload.get("message"),
        )

    return _callback
