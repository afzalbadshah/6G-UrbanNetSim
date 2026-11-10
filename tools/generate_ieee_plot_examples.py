from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path("output/professional_plot_examples.png")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.35,
        "savefig.dpi": 300,
    }
)

colors = {
    "Onboard": "#4C78A8",
    "MEC": "#59A14F",
    "UAV-BS": "#F28E2B",
    "Cloud": "#E15759",
    "DQN": "#1F77B4",
    "PPO": "#2CA02C",
    "A2C": "#9467BD",
    "Greedy": "#7F7F7F",
    "Random": "#D62728",
}

densities = np.array([50, 100, 150, 200, 250])
tier_labels = ["Onboard", "MEC", "UAV-BS", "Cloud"]

offload = {
    "URLLC": np.array(
        [
            [34, 48, 12, 6],
            [30, 50, 14, 6],
            [26, 51, 16, 7],
            [22, 50, 19, 9],
            [18, 48, 22, 12],
        ]
    ),
    "mMTC": np.array(
        [
            [2, 72, 20, 6],
            [1, 70, 22, 7],
            [1, 67, 24, 8],
            [0, 64, 27, 9],
            [0, 61, 29, 10],
        ]
    ),
    "eMBB": np.array(
        [
            [1, 28, 18, 53],
            [1, 25, 18, 56],
            [0, 21, 17, 62],
            [0, 17, 15, 68],
            [0, 14, 13, 73],
        ]
    ),
}

fig, axs = plt.subplots(2, 3, figsize=(10.8, 6.8), constrained_layout=True)

# Panel a: 100 percent stacked offloading distribution.
ax = axs[0, 0]
x = np.arange(len(densities))
width = 0.22
for i, cls in enumerate(["URLLC", "mMTC", "eMBB"]):
    bottom = np.zeros(len(densities))
    for tier in tier_labels:
        vals = offload[cls][:, tier_labels.index(tier)]
        ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            bottom=bottom,
            color=colors[tier],
            edgecolor="white",
            linewidth=0.35,
            label=tier if i == 0 else None,
        )
        bottom += vals
ax.set_title("(a) Priority-wise offloading mix")
ax.set_ylabel("Assigned tasks (%)")
ax.set_xlabel("Crowd size (devices)")
ax.set_xticks(x)
ax.set_xticklabels(densities)
ax.set_ylim(0, 100)
ax.grid(axis="y", color="#d9d9d9", linewidth=0.5)
ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02))
for i, cls in enumerate(["URLLC", "mMTC", "eMBB"]):
    ax.text(
        x[-1] + (i - 1) * width,
        103,
        cls,
        ha="center",
        va="bottom",
        fontsize=6.8,
        rotation=0,
    )

# Panel b: URLLC latency CDF.
ax = axs[0, 1]
lat = np.linspace(1, 90, 220)
profiles = {
    "DQN": 1 / (1 + np.exp(-(lat - 18) / 5.5)),
    "PPO": 1 / (1 + np.exp(-(lat - 25) / 6.5)),
    "A2C": 1 / (1 + np.exp(-(lat - 30) / 7.0)),
    "Greedy": 1 / (1 + np.exp(-(lat - 38) / 8.0)),
    "Random": 1 / (1 + np.exp(-(lat - 50) / 10.0)),
}
for name, cdf in profiles.items():
    ax.plot(lat, cdf, color=colors[name], label=name)
ax.axvline(25, color="black", linestyle="--", linewidth=1.0)
ax.text(26, 0.08, "URLLC deadline", rotation=90, va="bottom", fontsize=7)
ax.set_title("(b) Latency reliability")
ax.set_xlabel("End-to-end delay (ms)")
ax.set_ylabel("CDF")
ax.set_xlim(0, 80)
ax.set_ylim(0, 1.02)
ax.grid(color="#d9d9d9", linewidth=0.5)
ax.legend(frameon=False, loc="lower right")

# Panel c: deadline satisfaction ratio.
ax = axs[0, 2]
dsr = {
    "DQN": [98, 97, 95, 93, 91],
    "PPO": [95, 93, 90, 87, 84],
    "A2C": [93, 90, 87, 83, 79],
    "Greedy": [88, 84, 79, 73, 67],
    "Random": [72, 66, 59, 52, 45],
}
markers = {"DQN": "o", "PPO": "s", "A2C": "^", "Greedy": "D", "Random": "x"}
for name, vals in dsr.items():
    ax.plot(densities, vals, marker=markers[name], color=colors[name], label=name)
ax.set_title("(c) Deadline satisfaction")
ax.set_xlabel("Crowd size (devices)")
ax.set_ylabel("Satisfied tasks (%)")
ax.set_ylim(40, 100)
ax.grid(color="#d9d9d9", linewidth=0.5)
ax.legend(frameon=False, loc="lower left")

# Panel d: mMTC residual battery.
ax = axs[1, 0]
rounds = np.arange(0, 101, 10)
battery = {
    "DQN": 100 - 0.18 * rounds,
    "PPO": 100 - 0.25 * rounds,
    "A2C": 100 - 0.31 * rounds,
    "Greedy": 100 - 0.43 * rounds,
    "Random": 100 - 0.58 * rounds,
}
for name, vals in battery.items():
    ax.plot(rounds, vals, marker=markers[name], markevery=2, color=colors[name], label=name)
ax.set_title("(d) mMTC battery preservation")
ax.set_xlabel("Simulation round")
ax.set_ylabel("Mean residual battery (%)")
ax.set_ylim(35, 102)
ax.grid(color="#d9d9d9", linewidth=0.5)

# Panel e: eMBB efficiency, grouped bars with confidence intervals.
ax = axs[1, 1]
methods = ["DQN", "PPO", "A2C", "Greedy", "Random"]
energy_eff = np.array([8.7, 7.5, 6.8, 5.9, 4.2])
cost_eff = np.array([7.9, 7.0, 6.1, 5.2, 3.6])
idx = np.arange(len(methods))
bar_w = 0.34
ax.bar(idx - bar_w / 2, energy_eff, bar_w, color="#4C78A8", label="Goodput/J")
ax.bar(idx + bar_w / 2, cost_eff, bar_w, color="#F28E2B", label="Goodput/cost")
ax.errorbar(idx - bar_w / 2, energy_eff, yerr=[0.3, 0.4, 0.3, 0.35, 0.25], fmt="none", ecolor="black", capsize=2, linewidth=0.8)
ax.errorbar(idx + bar_w / 2, cost_eff, yerr=[0.25, 0.3, 0.35, 0.3, 0.2], fmt="none", ecolor="black", capsize=2, linewidth=0.8)
ax.set_title("(e) eMBB efficiency")
ax.set_xlabel("Scheduler")
ax.set_ylabel("Normalized efficiency")
ax.set_xticks(idx)
ax.set_xticklabels(methods)
ax.set_ylim(0, 10)
ax.grid(axis="y", color="#d9d9d9", linewidth=0.5)
ax.legend(frameon=False, loc="upper right")

# Panel f: latency-energy Pareto tradeoff.
ax = axs[1, 2]
pareto = {
    "DQN": (21, 0.44),
    "PPO": (27, 0.50),
    "A2C": (31, 0.55),
    "Greedy": (39, 0.62),
    "Random": (55, 0.78),
}
for name, (delay, energy) in pareto.items():
    ax.scatter(delay, energy, s=58, color=colors[name], edgecolor="white", linewidth=0.7, label=name)
    ax.text(delay + 1.0, energy + 0.006, name, fontsize=7)
ax.annotate(
    "better",
    xy=(18, 0.40),
    xytext=(31, 0.48),
    arrowprops={"arrowstyle": "->", "linewidth": 0.9, "color": "black"},
    fontsize=7,
)
ax.set_title("(f) Latency-energy tradeoff")
ax.set_xlabel("Mean delay (ms)")
ax.set_ylabel("Energy per task (J)")
ax.set_xlim(15, 62)
ax.set_ylim(0.36, 0.84)
ax.grid(color="#d9d9d9", linewidth=0.5)

fig.suptitle(
    "Example IEEE-Style Evaluation Plots for Crowd-Aware Edge-Cloud Task Offloading",
    fontsize=10,
    y=1.015,
)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
print(OUT.resolve())
