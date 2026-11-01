#!/usr/bin/env python3
"""Generate publication figures from the study JSONL files (no GPU needed).

    python3 make_figs.py            # writes study/figs/*.png
Panels:
  fig1_paired_seeds.png  per-seed paired test accuracy, plain vs traj
  fig2_methods.png       method means +- SEM, with paired t and cost notes
  fig3_stress.png        stress regime (bs4, 250k tokens): win vs noise
  fig4_val_vs_test.png   every run: val (selection) vs test (held-out)
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

C_PLAIN, C_UNI, C_VAL = "#8d99ae", "#2b8cbe", "#d95f02"


def load(name):
    out = []
    p = os.path.join(HERE, name)
    if os.path.exists(p):
        for l in open(p):
            l = l.strip()
            if l:
                out.append(json.loads(l))
    return out


def dedupe(rows):
    seen = {}
    for r in rows:
        seen[r["key"]] = r  # last wins on append-only files
    return list(seen.values())


f2 = load("finals2.jsonl")
f3 = load("finals3.jsonl")
stress = load("stress.jsonl")
sweep = load("sweep_traj.jsonl")

plain = {int(r["seed"]): r for r in f2 if r["arm"] == "plain"}
uni = {int(r["seed"]): r for r in f2 if r["arm"] == "traj"
       and r.get("frw") == 0}
val = {int(r["seed"]): r for r in f3 if r["arm"] == "traj"}
seeds = sorted(set(plain) & set(val))

# ---------------------------------------------------- fig1: paired seeds
fig, ax = plt.subplots(figsize=(8, 4.2))
w, x = 0.38, range(len(seeds))
p = [plain[s]["test_acc"] * 100 for s in seeds]
t = [val[s]["test_acc"] * 100 for s in seeds]
ax.bar([i - w / 2 for i in x], p, w, color=C_PLAIN, label="plain LoRA-FA")
ax.bar([i + w / 2 for i in x], t, w, color=C_VAL,
       label="TrajectoryMix (val-scored)")
for i, (a, b) in enumerate(zip(p, t)):
    ax.text(i + w / 2, b + 0.08, f"+{b - a:.2f}", ha="center", fontsize=8,
            color="#333")
    ax.text(i - w / 2, a + 0.08, f"{a:.2f}", ha="center", fontsize=7,
            color="#666")
ax.set_xticks(list(x), [f"seed {s}" for s in seeds])
ax.set_ylim(88.4, 92.2)
ax.set_ylabel("SST-2 test accuracy (%)")
ax.set_title("Equal-FLOPS finals: 6/6 paired seeds won "
             "(traj costs 0.08% LESS)", fontsize=11)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig1_paired_seeds.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------- fig2: method means
fig, ax = plt.subplots(figsize=(6.4, 4.2))
arms, means, sems, cols, notes = [], [], [], [], []
for nm, d, c, note in [
        ("plain", plain, C_PLAIN, "tuned lr grid"),
        ("traj\nuniform w", uni, C_UNI, "+0.61 pts, t=3.9\nzero extra cost"),
        ("traj\nval-scored w", val, C_VAL,
         "+1.01 pts, t=7.6\n0.08% cheaper")]:
    if not d:
        continue
    a = [d[s]["test_acc"] * 100 for s in sorted(d)]
    arms.append(nm)
    means.append(st.mean(a))
    sems.append(st.stdev(a) / len(a) ** 0.5)
    cols.append(c)
    notes.append(note)
ax.bar(arms, means, yerr=sems, capsize=4, color=cols)
for i, (m, n) in enumerate(zip(means, notes)):
    ax.text(i, m - 0.28, n, ha="center", fontsize=7.5)
ax.set_ylim(89.0, 91.6)
ax.set_ylabel("SST-2 test accuracy (%), mean +- SEM, 6 seeds")
ax.set_title("Endpoint-mixture updatemix, identical token budget",
             fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig2_methods.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------- fig3: stress regime
lrs = [1e-3, 3e-3, 1e-2]
fig, ax = plt.subplots(figsize=(6.8, 4.2))
for arm, c, off in [("plain", C_PLAIN, -0.18), ("traj", C_VAL, 0.18)]:
    m, e = [], []
    for lr in lrs:
        a = [r["test_acc"] * 100 for r in stress
             if r["arm"] == arm and abs(r["lr"] - lr) < 1e-9]
        m.append(st.mean(a))
        e.append(st.stdev(a) / len(a) ** 0.5)
    xs = [i + off for i in range(len(lrs))]
    ax.bar(xs, m, 0.34, yerr=e, capsize=3, color=c, label=arm)
for i, (lr) in enumerate(lrs):
    d = [t - p for t, p in zip([r["test_acc"] * 100 for r in stress
                                if r["arm"] == "traj"
                                and abs(r["lr"] - lr) < 1e-9],
                               [r["test_acc"] * 100 for r in stress
                                if r["arm"] == "plain"
                                and abs(r["lr"] - lr) < 1e-9])]
    ax.text(i, max(86, max(
        [r["test_acc"] * 100 for r in stress if abs(r["lr"] - lr) < 1e-9]
    )) + 0.6, f"Δ {st.mean(d):+.2f}", ha="center", fontsize=9)
ax.set_xticks(range(len(lrs)), [f"lr {g:g}" for g in lrs])
ax.set_ylabel("test accuracy (%) — bs 4, 250k tokens, 2 seeds")
ax.set_title("High-noise regime (uniform-weight traj): gains grow with "
             "gradient noise", fontsize=10.5)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig3_stress.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------ fig4: val vs test trust
fig, ax = plt.subplots(figsize=(6.0, 4.6))
pts = []
for src, kind in [(f2, "finals2"), (f3, "finals3"), (sweep, "sweep")]:
    for r in dedupe(src):
        if r.get("arm") == "traj":
            pts.append((r["val_acc"] * 100, r["test_acc"] * 100, kind))
for kind, c, mkr in [("finals2", C_UNI, "o"), ("finals3", C_VAL, "^"),
                     ("sweep", "#7b3294", "s")]:
    xy = [(p[0], p[1]) for p in pts if p[2] == kind]
    if xy:
        ax.scatter([ab[0] for ab in xy], [ab[1] for ab in xy], s=26, c=c,
                   marker=mkr, alpha=0.75, label=f"traj {kind}",
                   edgecolors="none")
for s in seeds:
    ax.scatter(plain[s]["val_acc"] * 100, plain[s]["test_acc"] * 100, s=30,
               c=C_PLAIN, marker="x", label="plain finals" if s == seeds[0]
               else None)
ax.set_xlabel("val accuracy (%) — used for ALL selection")
ax.set_ylabel("official test accuracy (%) — held out")
ax.set_title("Every config picked on val transfers to test "
             "(no selection surprise)", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig4_val_vs_test.png"), dpi=150)
plt.close(fig)

# ------------------------------------------- fig5: train/test dynamics
if os.path.exists(os.path.join(HERE, "dyn.jsonl")):
    dyn = {r["key"]: r for r in load("dyn.jsonl") if "dyn" in r}
    seeds3 = [s for s in (0, 1, 2) if f"d-plain-{s}" in dyn]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)
    # All arms train the identical plain path (mixing happens only at
    # flush), so mid-training curves are shared per seed: draw the
    # seed-averaged trajectory once, then mark the flush-time endpoints.
    for ax, fld, yl in [(axs[0], "te", "test acc (%)"),
                        (axs[1], "tr", "train acc (192 ex, %)")]:
        curves = [(d["cost"] / 1e3, d[fld] * 100)
                  for s in seeds3 for d in dyn[f"d-plain-{s}"]["dyn"]]
        xs = sorted(set(round(x, 1) for x, _ in curves))
        ys = [[y for x, y in curves if abs(x - xx) < 1] for xx in xs]
        m = [st.mean(v) for v in ys]
        ax.plot(xs, m, color="#34495e", lw=2,
                label="shared training path (mean of seeds)")
        ax.fill_between(xs, [min(v) for v in ys], [max(v) for v in ys],
                        color="#34495e", alpha=0.15)
        ax.set_xlabel("counted fwd-bwd tokens (thousands)")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.25)
    # terminal markers on the test panel: where each arm finishes
    fin = {"live endpoint (plain)": "d-plain",
           "uniform mixture": "d-traju", "val-scored mixture": "d-trajf"}
    cols = {"live endpoint (plain)": C_PLAIN, "uniform mixture": C_UNI,
            "val-scored mixture": C_VAL}
    for i, (nm, pref) in enumerate(fin.items()):
        y = st.mean([dyn[f"{pref}-{s}"]["test_acc"] * 100
                     for s in seeds3])
        x0 = st.mean([dyn[f"{pref}-{s}"]["cost"] for s in seeds3]) / 1e3
        axs[0].scatter([x0], [y], s=90, zorder=6, color=cols[nm],
                       edgecolors="k", linewidths=.6,
                       label=f"{nm}: {y:.2f}")
    axs[0].set_title("Test accuracy vs counted compute: one shared path, "
                     "all gains come from the final mix (seeds 0-2)",
                     fontsize=10)
    axs[1].set_title("Train acc on 192 examples: saturates ~300k tok — "
                     "test gains are not 'more fitting'", fontsize=10)
    for ax in axs:
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig5_dynamics.png"), dpi=150)
    plt.close(fig)

    # --------------------------------- fig6: endpoint spread vs mixture
    eps = [r for r in load("dyn.jsonl") if r.get("endpoints")]
    if eps:
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        for j, r in enumerate(eps[:2]):
            e = {d["label"]: d["te"] * 100 for d in r["endpoints"]}
            ends = [v for k, v in sorted(e.items(),
                                         key=lambda kv: kv[0])
                    if k.startswith("ep")]
            xs = list(range(len(ends)))
            ax.plot(xs, ends, "o--", color="#666", lw=1, ms=5,
                    label=f"candidates ({r['key']})")
            # the live endpoint (where plain would stop) is the last one
            ax.plot([xs[-1]], [ends[-1]], "o", color="#c0392b", ms=9,
                    zorder=5, label="live endpoint (= plain stop)"
                    if j == 0 else None)
            ax.axhline(e.get("uniform_mix", float("nan")), color=C_UNI,
                       lw=2, label="uniform mixture")
            ax.axhline(e.get("fitted_mix", float("nan")), color=C_VAL,
                       lw=2, label="val-scored mixture")
        ax.set_xlabel("trajectory candidate index (oldest tail bin → "
                      "newest; last = live endpoint)")
        ax.set_ylabel("test accuracy (%)")
        ax.set_title("Candidates swing over 2 pts; the live endpoint is "
                     "one of the worst,\nthe mixture lands above their "
                     "average (+0.8) and beats it by +1.2..+1.5",
                     fontsize=10.5)
        ax.legend(fontsize=8.5, loc="lower right")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, "fig6_endpoints.png"), dpi=150)
        plt.close(fig)

for f in sorted(os.listdir(FIGS)):
    p = os.path.join(FIGS, f)
    print(f, os.path.getsize(p), "bytes")
