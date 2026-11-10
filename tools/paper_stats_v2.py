"""Evaluation statistics for the HRLLC-6G paper (paper-delay.tex), one block per figure.

Uses the SAME definitions as visualization/ieee_paper_plots.py:
  delay       = observed_complete_latency_ms (all rows; pooled over seeds)
  on-time     = deadline_missed == 0          (Fig. 3a heatmap)   | success (Table II)
  failed      = status != "completed"
  components  = tx (wireless+wired), propagation_total, queue, processing, handover; completed only
  radio       = tx_delay_wireless + propagation_wireless (+ pre_access_wait); remote tasks only
Spread across seeds = observed min-max (project convention).

Usage (from the simulator root):
  python tools/paper_stats_hrllc.py  [eval_dir]  [out_file]
"""
import glob, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, "..", "output", "run_007_fr3_city_coverage_6g_evaluation", "evaluation")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "output", "paper_stats_hrllc.md")

POL = {"dqn": "DQN", "double_dqn": "DDQN", "ppo": "PPO", "a2c": "A2C", "q_learning": "QL", "random": "Random"}
ORDER = list(POL.values())
CLS = {"URLLC": "HRLLC", "mMTC": "MC", "eMBB": "IC"}
LAT = "observed_complete_latency_ms"
COLS = ["class_name", "status", "success", "deadline_missed", "dropped", "coverage_outage_flag", LAT,
        "network_delay_ms", "tx_delay_wireless_ms", "tx_delay_wired_ms", "propagation_wireless_ms",
        "propagation_total_ms", "queue_delay_ms", "pre_access_wait_ms", "processing_delay_ms",
        "handover_delay_component_ms", "allocated_tier", "active_devices_this_round"]

frames = []
for f in sorted(glob.glob(os.path.join(EVAL, "seed_*", "offloading", "*", "results.csv"))):
    parts = os.path.normpath(f).split(os.sep)
    seed, pol = parts[-4], parts[-2]
    if pol not in POL:
        continue
    d = pd.read_csv(f, sep=";", usecols=COLS, low_memory=False)
    d["seed"], d["policy"] = int(seed.split("_")[1]), POL[pol]
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
df["cls"] = df.class_name.map(CLS)
df["done"] = df.status.eq("completed")
df["ontime"] = df.deadline_missed.eq(0)
df["tx"] = df.tx_delay_wireless_ms + df.tx_delay_wired_ms
df["remote"] = df.allocated_tier.ne("LOCAL_DEVICE")
df["radio"] = df.tx_delay_wireless_ms + df.propagation_wireless_ms
df["radio_acc"] = df.radio + df.pre_access_wait_ms

L = []
def h(t): L.append(f"\n## {t}\n")
def md(frame, digits=3):
    """Markdown table without the optional 'tabulate' dependency."""
    def f(v):
        return f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)
    head = "| " + " | ".join([str(frame.index.name or "")] + [str(c) for c in frame.columns]) + " |"
    sep = "|" + "|".join(["---"] * (len(frame.columns) + 1)) + "|"
    rows = ["| " + " | ".join([str(i)] + [f(v) for v in r]) + " |" for i, r in zip(frame.index, frame.values)]
    return "\n".join([head, sep] + rows)
def tab(frame, digits=3):
    frame = frame.reindex(columns=[c for c in ORDER if c in frame.columns])
    L.append(md(frame, digits))
def seeds_range(g, fn):
    v = [fn(s) for _, s in g.groupby("seed")]
    return np.mean(v), np.min(v), np.max(v)
def rng(m, lo, hi, d=2):
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"

L.append("# Evaluation statistics (run_007, pooled over seeds; [min, max] across seeds)\n")
L.append("Seeds per policy: " + ", ".join(f"{p}={sorted(int(x) for x in df[df.policy==p].seed.unique())}" for p in ORDER))
L.append("\nTasks per policy and class (pooled): " + "; ".join(
    f"{c}={int(df[(df.cls==c)&(df.policy=='DQN')].shape[0])}" for c in CLS.values()))

for c in CLS.values():
    s = df[df.cls == c]
    # Fig. 1 / Fig. 2: distribution and tail
    h(f"Fig. 1-2 ({c}): delay distribution and tail [ms]")
    rows = {}
    for p, g in s.groupby("policy"):
        x = g[LAT].dropna()
        rows[p] = {"mean": x.mean(), "median": x.median(), "P75": x.quantile(.75), "P90": x.quantile(.9),
                   "P95": x.quantile(.95), "P99": x.quantile(.99), "P99/P50": x.quantile(.99) / x.median(),
                   "IQR": x.quantile(.75) - x.quantile(.25)}
    tab(pd.DataFrame(rows))
    # Fig. 3 / Fig. 4: reliability and trade-off
    h(f"Fig. 3-4 ({c}): reliability and delay-reliability trade-off")
    rows = {}
    for p, g in s.groupby("policy"):
        dsr = seeds_range(g, lambda t: 100 * t.ontime.mean())
        mlat = seeds_range(g, lambda t: t[LAT].mean())
        fail = seeds_range(g, lambda t: (~t.done).sum())
        rows[p] = {"DSR % (mean [min,max])": rng(*dsr, 3), "Mean delay ms": rng(*mlat, 2),
                   "Failed/seed": rng(*fail, 2), "Late (pooled)": int((g.done & ~g.ontime).sum()),
                   "Success % (Table II)": f"{100*g.success.mean():.3f}"}
    L.append(md(pd.DataFrame(rows).reindex(columns=ORDER)))
    # Fig. 5: composition
    h(f"Fig. 5 ({c}): mean delay components [ms], completed tasks")
    comp = {"Tx": "tx", "Prop.": "propagation_total_ms", "Queue": "queue_delay_ms",
            "Proc.": "processing_delay_ms", "Handover": "handover_delay_component_ms"}
    rows = {}
    for p, g in s[s.done].groupby("policy"):
        r = {k: g[v].mean() for k, v in comp.items()}
        tot = sum(r.values())
        r["Total"] = tot
        r["Queue share %"] = 100 * r["Queue"] / tot
        r["Proc share %"] = 100 * r["Proc."] / tot
        rows[p] = r
    tab(pd.DataFrame(rows))
    # Fig. 6: queueing
    h(f"Fig. 6 ({c}): queueing delay, completed tasks")
    rows = {}
    for p, g in s[s.done].groupby("policy"):
        q = g.queue_delay_ms
        rows[p] = {"zero-wait %": 100 * (q <= 0).mean(), "median ms": q.median(), "P95 ms": q.quantile(.95),
                   "P99 ms": q.quantile(.99)}
    tab(pd.DataFrame(rows))
    # Fig. 7 placement
    h(f"Fig. 7 ({c}): task placement [% of tasks]")
    pl = pd.crosstab(s.allocated_tier, s.policy, normalize="columns") * 100
    tab(pl, 1)

# Fig. 7: radio link (HRLLC only)
s = df[(df.cls == "HRLLC") & df.remote]
h("Fig. 7 (HRLLC, remote tasks): radio-link delay")
rows = {}
for p, g in s.groupby("policy"):
    rows[p] = {"radio P99 ms": g.radio.quantile(.99), "radio P99.9 ms": g.radio.quantile(.999),
               "radio <=0.1 ms %": 100 * (g.radio <= 0.1).mean(),
               "radio+access <=0.1 ms %": 100 * (g.radio_acc <= 0.1).mean(),
               "radio+access <=1 ms %": 100 * (g.radio_acc <= 1.0).mean(),
               "tasks with access wait %": 100 * (g.pre_access_wait_ms > 0).mean()}
tab(pd.DataFrame(rows), 4)
h("Fig. 7a (HRLLC): radio+access > 0.1 ms [% of remote tasks] by active devices")
s = s.assign(load=pd.cut(s.active_devices_this_round, [0, 25, 50, 100, 150, 200],
                         labels=["1-25", "26-50", "51-100", "101-150", "151-200"]))
tab((s.groupby(["load", "policy"], observed=True).radio_acc.apply(lambda x: 100 * (x > 0.1).mean())
     .unstack()), 2)

# Table II
h("Table II: HRLLC summary (definitions as in class_summary_metrics)")
s = df[df.cls == "HRLLC"]
rows = {}
for p, g in s.groupby("policy"):
    r = g[g.remote]
    rows[p] = {"On-time % (success)": 100 * g.success.mean(), "E2E P95 ms": g[LAT].quantile(.95),
               "Network P95 ms (remote)": r.network_delay_ms.quantile(.95),
               "Availability %": 100 * ((g.coverage_outage_flag == 0) & (g.dropped == 0)).mean()}
tab(pd.DataFrame(rows), 3)

# Pooled outcome (Fig. 3c)
h("Fig. 3c: outcome over all classes [% of tasks]")
rows = {}
for p, g in df.groupby("policy"):
    rows[p] = {"on time": 100 * (g.done & g.ontime).mean(), "late": 100 * (g.done & ~g.ontime).mean(),
               "failed": 100 * (~g.done).mean()}
tab(pd.DataFrame(rows), 3)

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote", OUT)
