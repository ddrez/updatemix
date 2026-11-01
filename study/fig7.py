#!/usr/bin/env python3
"""Round-5 figure: every mitigation comparison as a forest plot of the
paired test-accuracy gain over the equal-cost plain baseline (+- SEM).

    python3 fig7.py           # writes figs/fig7_mitigation.png
"""
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)
C = {"uni": "#2b8cbe", "fval": "#d95f02", "funl": "#1a9641"}


def load(name):
    p = os.path.join(HERE, name)
    d = {}
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            d[r["key"]] = r
    return d


def idx(d, want_seeds=None):
    """prefix -> {seed: rec} for keys 'arm-seed' (part 1 int)."""
    out = {}
    for k, r in d.items():
        p = k.split("-")
        if len(p) == 2 and p[1].lstrip("-").isdigit():
            s = int(p[1])
            if want_seeds is None or s in want_seeds:
                out.setdefault(p[0], {})[s] = r
    return out


def diff(a_by_s, b_by_s):
    ss = sorted(set(a_by_s) & set(b_by_s))
    ds = [a_by_s[s]["test_acc"] - b_by_s[s]["test_acc"] for s in ss]
    m = 100 * st.mean(ds)
    sem = 100 * (st.stdev(ds) / len(ds) ** .5) if len(ds) > 1 else 0.0
    return m, sem, len(ds), sum(d > 0 for d in ds)


f2 = load("finals2.jsonl")
f3 = load("finals3.jsonl")
pl = {int(k.split("-")[1]): r for k, r in f2.items()
      if k.startswith("plain-")}
uni05 = {int(k.split("-")[1]): r for k, r in f2.items()
         if k.startswith("traj-")}
fval05 = {int(k.split("-")[1]): r for k, r in f3.items()}
cf = idx(load("confirm.jsonl"))
tkl = load("tasks.jsonl")
s15 = idx(load("scale.jsonl"))
s3 = idx(load("scale3.jsonl"))
u2 = {int(k.split("-")[1]): r for k, r in load("funl2.jsonl").items()}
f15 = idx(load("funl15.jsonl"))


def tkarm(task, arm):
    return {int(k.split("-")[2]): r for k, r in tkl.items()
            if k.startswith(f"{task}-{arm}-")}


def tkplain(task):
    return tkarm(task, "plain")


rows = []  # (label, mean, sem, n, wins, color)
m, s, n, w = diff(cf["uni"], cf["plain"])
rows.append(("SST-2 uni, unseen seeds 6-11", m, s, n, w, "uni"))
m, s, n, w = diff(cf["fval"], cf["plain"])
rows.append(("SST-2 fval, unseen seeds 6-11", m, s, n, w, "fval"))
m, s, n, w = diff({**uni05, **cf["uni"]}, {**pl, **cf["plain"]})
rows.append(("SST-2 uni, 12 seeds pooled", m, s, n, w, "uni"))
m, s, n, w = diff({**fval05, **cf["fval"]}, {**pl, **cf["plain"]})
rows.append(("SST-2 fval, 12 seeds pooled", m, s, n, w, "fval"))
m, s, n, w = diff(u2, pl)
rows.append(("SST-2 funl (label-free, cost parity)", m, s, n, w, "funl"))
for t, tl in [("mrpc", "MRPC"), ("rte", "RTE"), ("agnews", "AG News")]:
    m, s, n, w = diff(tkarm(t, "uni"), tkplain(t))
    rows.append((f"{tl} uni", m, s, n, w, "uni"))
for t, tl in [("mrpc", "MRPC"), ("rte", "RTE"), ("agnews", "AG News")]:
    m, s, n, w = diff(tkarm(t, "funl"), tkplain(t))
    rows.append((f"{tl} funl", m, s, n, w, "funl"))
m, s, n, w = diff(s15["uni"], s15["plain"])
rows.append(("1.5B uni", m, s, n, w, "uni"))
m, s, n, w = diff(s15["fval"], s15["plain"])
rows.append(("1.5B fval", m, s, n, w, "fval"))
m, s, n, w = diff(f15["funl"], s15["plain"])
rows.append(("1.5B funl", m, s, n, w, "funl"))
m, s, n, w = diff(s3["uni"], s3["plain"])
rows.append(("3B uni", m, s, n, w, "uni"))
m, s, n, w = diff(s3["fval"], s3["plain"])
rows.append(("3B fval", m, s, n, w, "fval"))

fig, ax = plt.subplots(figsize=(7.6, 5.6))
ys = range(len(rows))[::-1]
for y, (lab, m, s, n, w, arm) in zip(ys, rows):
    ax.plot([m - s, m + s], [y, y], lw=1.6, color=C[arm], alpha=.85)
    ax.plot(m, y, "o", ms=7, color=C[arm])
    ax.text(1.35, y, f"{m:+.2f}  (n={n}, {w}/{n})", va="center", fontsize=8)
ax.axvline(0, color="k", lw=.8, ls=":")
ax.set_yticks(list(ys))
ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
ax.set_xlabel("paired test-accuracy gain over equal-cost plain (pp, +- SEM)")
ax.set_xlim(-1.2, 3.3)
ax.set_title("Round 5: TrajectoryMix vs plain across seeds, tasks, scales",
             fontsize=10.5)
ax.grid(axis="x", alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig7_mitigation.png"), dpi=150)
print("wrote figs/fig7_mitigation.png")
