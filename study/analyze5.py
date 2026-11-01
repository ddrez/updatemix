#!/usr/bin/env python3
"""Round-5 analysis: paired stats for the caveat-mitigation experiments.
Reads the JSONL files in this directory; prints per-experiment tables.
All comparisons paired by seed; counted cost reported alongside."""
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))


def load(*names):
    out = {}
    for name in names:
        p = os.path.join(HERE, name)
        if os.path.exists(p):
            for ln in open(p):
                r = json.loads(ln)
                out[r.get("key", "?")] = r
    return out


def norm(entries):
    """Index by int seed (key part 1)."""
    out = {}
    for k, r in entries.items():
        p = k.split("-")
        if len(p) > 1 and p[1].isdigit():
            out[int(p[1])] = r
    return out


def cmp(arms_a, arm_a, arms_b, arm_b, seeds=None):
    keys = [s for s in (seeds if seeds is not None
                        else sorted(set(arms_a) & set(arms_b)))
            if s in arms_a and s in arms_b]
    diffs = [arms_a[s]["test_acc"] - arms_b[s]["test_acc"] for s in keys]
    n = len(diffs)
    if not n:
        print(f"  {arm_a} vs {arm_b}: (no paired data yet)")
        return
    mean = st.mean(diffs)
    sd = st.stdev(diffs) if n > 1 else float("nan")
    t = mean / (sd / math.sqrt(n)) if n > 1 and sd > 0 else float("inf")
    print(f"  {arm_a} vs {arm_b}: n={n} acc {st.mean([arms_b[s]['test_acc'] for s in keys]):.4f}"
          f" -> {st.mean([arms_a[s]['test_acc'] for s in keys]):.4f}"
          f"  diff {mean*100:+.2f} ± {sd*100:.2f}  t={t:.2f}"
          f"  wins {sum(d > 0 for d in diffs)}/{n}"
          f"  cost {st.mean([arms_b[s]['cost'] for s in keys]):.0f}"
          f" -> {st.mean([arms_a[s]['cost'] for s in keys]):.0f}")


f2 = load("finals2.jsonl")
f3 = load("finals3.jsonl")
pl05 = {int(k.split("-")[1]): r for k, r in f2.items()
        if k.startswith("plain-")}
uni05 = {int(k.split("-")[1]): r for k, r in f2.items()
         if k.startswith("traj-")}
fval05 = {int(k.split("-")[1]): r for k, r in f3.items()}

cf = load("confirm.jsonl")
cfi = {}
for k, r in cf.items():
    p = k.split("-")
    cfi.setdefault(p[0], {})[int(p[1])] = r

print("== confirm: frozen configs, seeds 6-11 (never used for selection) ==")
cmp(cfi.get("uni", {}), "uni", cfi.get("plain", {}), "plain")
cmp(cfi.get("fval", {}), "fval", cfi.get("plain", {}), "plain")
cmp(cfi.get("fval", {}), "fval", cfi.get("uni", {}), "uni")

print("== all seeds pooled (0-5 original + 6-11 confirmation) ==")
allp = {**pl05, **cfi.get("plain", {})}
allu = {**uni05, **cfi.get("uni", {})}
allf = {**fval05, **cfi.get("fval", {})}
cmp(allu, "uni(12 seeds)", allp, "plain(12 seeds)")
cmp(allf, "fval(12 seeds)", allp, "plain(12 seeds)")

print("== unl: --funl external unlabeled scoring (seeds 0-5) ==")
un = load("unl.jsonl")
un05 = norm(un)
cmp(un05, "funl", pl05, "plain", seeds=range(6))
cmp(un05, "funl", uni05, "uniform", seeds=range(6))
cmp(un05, "funl", fval05, "fval(labeled)", seeds=range(6))

print("== funl2: --funl at strict cost parity (cap 967300, seeds 0-5) ==")
un2 = norm(load("funl2.jsonl"))
cmp(un2, "funl2", pl05, "plain", seeds=range(6))
cmp(un2, "funl2", uni05, "uniform", seeds=range(6))
cmp(un2, "funl2", fval05, "fval(labeled)", seeds=range(6))

print("== tasks: plain vs uni vs fval, per task (frozen recipe) ==")
tk = load("tasks.jsonl")
for task in ("mrpc", "rte", "agnews"):
    d = {}
    for k, r in tk.items():
        p = k.split("-")  # task-arm-seed
        if len(p) == 3 and p[0] == task and p[2].isdigit():
            d.setdefault(p[1], {})[int(p[2])] = r
    cmp(d.get("uni", {}), f"{task}: uni", d.get("plain", {}), "plain")
    cmp(d.get("fval", {}), f"{task}: fval", d.get("plain", {}), "plain")
    cmp(d.get("funl", {}), f"{task}: funl", d.get("plain", {}), "plain")

print("== cross-task pooled (mrpc+rte+agnews, frozen recipe) ==")
for arm in ("uni", "fval", "funl"):
    a, b = {}, {}
    for task in ("mrpc", "rte", "agnews"):
        d = {}
        for k, r in tk.items():
            p = k.split("-")
            if len(p) == 3 and p[0] == task and p[2].isdigit():
                d.setdefault(p[1], {})[int(p[2])] = r
        for s in d.get(arm, {}):
            if s in d.get("plain", {}):
                a[f"{task}-{s}"] = d[arm][s]
                b[f"{task}-{s}"] = d["plain"][s]
    cmp(a, arm, b, "plain")


def scale_show(fname, label):
    sc = load(fname)
    sweep = {int(k.split("-")[0][7:]): r for k, r in sc.items()
             if k.startswith("sweeplr")}
    for k, r in sorted(sweep.items()):
        print(f"  sweep lr={r['lr']}: val={r['val_acc']:.4f}"
              f" test={r['test_acc']:.4f}")
    sarms = {}
    for k, r in sc.items():
        p = k.split("-")
        if len(p) == 2 and p[1].isdigit():
            sarms.setdefault(p[0], {})[int(p[1])] = r
    return sarms


print("== scale (1.5B): plain lr swept at scale; mixture recipe frozen ==")
s15 = scale_show("scale.jsonl", "1.5B")
cmp(s15.get("uni", {}), "1.5B uni", s15.get("plain", {}), "1.5B plain")
cmp(s15.get("fval", {}), "1.5B fval", s15.get("plain", {}), "1.5B plain")
f15 = norm(load("funl15.jsonl"))
cmp(f15, "1.5B funl", s15.get("plain", {}), "1.5B plain", seeds=range(4))
cmp(f15, "1.5B funl", s15.get("uni", {}), "1.5B uni", seeds=range(4))

print("== scale (3B): same fair protocol ==")
s3 = scale_show("scale3.jsonl", "3B")
cmp(s3.get("uni", {}), "3B uni", s3.get("plain", {}), "3B plain")
cmp(s3.get("fval", {}), "3B fval", s3.get("plain", {}), "3B plain")
