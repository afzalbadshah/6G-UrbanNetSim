# core/reporter.py
"""
RegionalEdgeSimPy — Educational Release

Author: Dr. Afzal Badshah
Department of Software Engineering, University of Sargodha
Website: https://afzalbadshah.com/index.php/afzal-badshah/
LinkedIn: https://www.linkedin.com/in/afzal-badshah-phd-305b03164/
Discussion Forum: resp.afzalbadshah.com

Notes:
- Functionality preserved; comments kept simple for student readability.
- This module prints metrics to the console and writes them to a CSV file.
- Students can add/remove/change metrics by modifying TASK_FIELDS below.
"""

import csv
from pathlib import Path

try:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_TRAINING_PLOTS = True
except Exception:
    _HAS_TRAINING_PLOTS = False


class Reporter:
    def __init__(self, fields, csv_file, print_console=True):
        """
        fields   : list of (key, label, format_string)
                   key   = dictionary key in data dict
                   label = column name for console/CSV
                   fmt   = how to format the value (e.g., '{:.2f}')
        csv_file : output file path
        """
        self.fields = fields
        self.csv_file = csv_file
        self.print_console = bool(print_console)
        self.widths = None
        self._fh = None
        self._writer = None
        self._init_csv()

    def _init_csv(self):
        """Initialize CSV file with header row, creating parent dirs if missing."""
        parent = Path(self.csv_file).parent
        if str(parent):
            parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.csv_file, "w", newline="")
        self._writer = csv.writer(self._fh)
        header = [label for _, label, _ in self.fields]
        self._writer.writerow(header)

    def report(self, data):
        """Write one row to console and CSV from `data` dict."""
        # Apply formatting for console display
        formatted = []
        for key, label, fmt in self.fields:
            val = data.get(key, "")
            try:
                formatted.append(fmt.format(val))
            except (ValueError, TypeError):
                # If formatting fails (e.g., empty string with '{:.2f}'), fallback to str
                formatted.append(str(val))

        if self.print_console:
            # If first call, print header and separator
            if self.widths is None:
                labels = [label for _, label, _ in self.fields]
                self.widths = [max(len(label), len(val)) for label, val in zip(labels, formatted)]
                header_line = " | ".join(label.ljust(w) for label, w in zip(labels, self.widths))
                separator   = "-+-".join("-" * w for w in self.widths)
                print(header_line)
                print(separator)

            # Print row to console
            line = " | ".join(val.ljust(w) for val, w in zip(formatted, self.widths))
            print(line)

        # Append raw values (unformatted) to CSV
        row = [data.get(key, "") for key, _, _ in self.fields]
        self._writer.writerow(row)

    def close(self):
        if self._fh is None:
            return
        try:
            self._fh.flush()
            self._fh.close()
        finally:
            self._fh = None
            self._writer = None


class TrainingReporter:
    """
    Writes scheduler training logs as CSV files from lists of row dictionaries.
    The caller owns when rows are collected; this class owns stable CSV output.
    """

    def __init__(self, output_dir, delimiter=",", encoding="utf-8"):
        self.output_dir = Path(output_dir)
        self.delimiter = delimiter
        self.encoding = encoding

    @staticmethod
    def _fieldnames(rows, preferred=None):
        fields = []
        seen = set()
        for name in preferred or ():
            if name not in seen:
                fields.append(name)
                seen.add(name)
        for row in rows:
            for name in row.keys():
                if name not in seen:
                    fields.append(name)
                    seen.add(name)
        return fields

    def write_rows(self, filename, rows, fieldnames=None):
        rows = list(rows or [])
        if not rows:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        fields = self._fieldnames(rows, fieldnames)
        with open(path, "w", newline="", encoding=self.encoding) as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, delimiter=self.delimiter)
            writer.writeheader()
            writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
        return path

    def write_many(self, specs):
        written = {}
        for name, rows, fieldnames in specs:
            path = self.write_rows(name, rows, fieldnames=fieldnames)
            if path is not None:
                written[name] = path
        return written

    def save_offloading_training(self, model_name, reward_log=None, epsilon_log=None,
                                  action_dist_log=None, loss_log=None,
                                  include_plots=False, visualizer=None):
        model = str(model_name).strip().lower()
        action_fields = list(action_dist_log[0].keys()) if action_dist_log else None
        written = self.write_many([
            (f"{model}_reward_log.csv", reward_log, ("round", "reward", "action", "priority")),
            (f"{model}_epsilon_decay.csv", epsilon_log, ("round", "epsilon")),
            (f"{model}_action_distribution.csv", action_dist_log, action_fields),
            (f"{model}_loss_log.csv", loss_log, ("round", "loss")),
        ])
        if include_plots and visualizer is not None:
            visualizer.output_dir = self.output_dir
            visualizer.save_all_plots()
        return written

    def save_handover_training(self, reward_log=None, loss_log=None,
                               action_dist_log=None, include_plots=True):
        written = self.write_many([
            ("reward_log_mobility.csv", reward_log, ("round", "device_id", "reward")),
            ("loss_log_mobility.csv", loss_log, ("round", "actor_loss", "critic_loss", "advantage")),
            ("action_distribution_mobility.csv", action_dist_log, ("round", "action", "decision_type")),
        ])
        if include_plots and (reward_log or loss_log):
            self._save_handover_training_summary(reward_log or [], loss_log or [])
        return written

    def _save_handover_training_summary(self, reward_log, loss_log):
        if not _HAS_TRAINING_PLOTS:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        reward_df = pd.DataFrame(reward_log) if reward_log else pd.DataFrame()
        loss_df = pd.DataFrame(loss_log) if loss_log else pd.DataFrame()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

        if not reward_df.empty:
            reward_by_round = reward_df.groupby("round")["reward"].mean().sort_index()
            axes[0].plot(reward_by_round.index, reward_by_round.values, marker="o", markersize=2, linewidth=0.8)
        axes[0].set_xlabel("Round")
        axes[0].set_ylabel("Average Reward")
        axes[0].set_title("Handover Agent - Average Reward")
        axes[0].grid(True, alpha=0.3)

        if not loss_df.empty:
            loss_by_round = loss_df.groupby("round")[["actor_loss", "critic_loss"]].mean().sort_index()
            axes[1].plot(loss_by_round.index, loss_by_round["actor_loss"], linewidth=0.8, label="Actor Loss")
            axes[1].plot(loss_by_round.index, loss_by_round["critic_loss"], linewidth=0.8, label="Critic Loss")
            axes[1].legend()
        axes[1].set_xlabel("Round")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Handover Agent - Training Loss")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        path = self.output_dir / "training_summary_mobility.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

# ---------------------------------------------------------------------
# Default task reporting configuration
# ---------------------------------------------------------------------
# To change what metrics are recorded:
# 1) Add a new tuple to TASK_FIELDS: ('your_key', 'Column Name', '{:.2f}')
# 2) Remove or comment out any tuple to hide that metric.
# 3) Ensure your simulator includes those keys in its metrics dict.
# ---------------------------------------------------------------------

TASK_FIELDS = [
    ("round_no",    "Round",            "{}"),
    ("devices",     "Devices",          "{}"),
    ("workload",    "Workload",         "{:.2f}"),
    # Mobility‐specific metrics
    ("avg_pos",     "Avg_Pos",          "{:.2f}"),
    ("ho_count",    "Handover_Count",   "{}"),
    ("ho_delay",    "Handover_Delay(ms)","{:.2f}"),
    ("avg_signal",  "Signal(dB)",       "{:.2f}"),
    ("wireless_ap", "Wireless_AP",      "{}"),
    ("MEC2Metro_m","MEC2Metro(m)",    "{:.1f}"),
    # Resource usage
    ("cpu_util",    "CPU (%)",          "{:.2f}"),
    ("memory_util", "Memory (%)",       "{:.2f}"),
    ("storage_util","Storage (%)",      "{:.2f}"),
    # Execution context
    ("layer",       "Paradigm",         "{}"),
    ("avg_tx",      "Avg_Tx(ms)",       "{:.2f}"),
    ("avg_prop",    "Avg_Prop(ms)",     "{:.2f}"),
    # Full precision (Task 52): monetary/energy values keep ~10 significant
    # figures so small nonzero costs are not rounded away to 0.
    ("tx_cost",     "Tx_Cost",          "{:.10g}"),
    ("proc_cost",   "Proc_Cost",        "{:.10g}"),
    ("energy",      "Energy",           "{:.10g}"),
    ("congestion",  "Conges(%)",        "{:.2f}"),
    ("flag",        "Flag",             "{}"),
    ("failed",      "Failed",           "{}"),

    # ── NEW: MEC collaboration metrics (the simulator now provides these keys) ──
    ("MEC_collab_in",     "MECCollab_In",     "{}"),     # tasks collaborated into this MEC
    ("MEC_wp_local",      "MEC_WP_Local",     "{}"),     # tasks served by own WP-MEC
    ("MEC_collab_ratio",  "MECCollab_Ratio",  "{:.3f}"), # ratio within this MEC
    ("MEC_collab_dist_m", "Collab_Dist(m)",    "{:.1f}"), # avg distance (m) for collaborated tasks
]

# Simulators create their own Reporter instance inside the active output folder.
task_reporter = None
