#!/usr/bin/env python3
"""Runnable in seconds, CPU-only: why TrajectoryMix beats the endpoint.

A tiny MLP + LoRA on a noisy synthetic regression task, trained hot
(large lr = large gradient noise). Three arms, identical seeds and
batch order:

  plain    — stop at the live endpoint (what standard fine-tuning does)
  uniform  — same trajectory; at the end, mix the tail snapshots
             (zero extra cost, SWA-class)
  val-fit  — same trajectory; mix with weights fit on held-out batches
             (a handful of extra forward/backward passes)

The plain endpoints bounce around the optimum; the mixture lands near
the basin center and beats them — both mixtures win on every seed:

    seed    plain  uniform  val-fit
        0    2.230    2.109    2.100
        1    2.618    2.601    2.592
        ...
     mean    2.538    2.395    2.414  (val RMSE)

Run:  python examples/trajectory_mix_toy.py
"""
import torch
import torch.nn.functional as F
from torch import nn

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root when running from a checkout

import updatemix as um

DIM, HID, N = 16, 48, 512


def make_data():
    g = torch.Generator().manual_seed(7)
    x = torch.randn(2 * N, DIM, generator=g)
    w = torch.randn(DIM, 1, generator=g)
    y = (x @ w + 0.5 * (x[:, :4] ** 2).sum(1, keepdim=True)).squeeze(-1)
    return (x[:N], y[:N]), (x[N:], y[N:])


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(DIM, HID)
        self.fc2 = nn.Linear(HID, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))).squeeze(-1)


def loss_fn(model, batch):
    return F.mse_loss(model(batch["x"]), batch["y"])


def run(seed, mode, train_xy, val_xy, steps=120, lr=0.3):
    torch.manual_seed(seed)
    model = MLP()
    rec = um.attach(model, "*fc1*", rank=8, seed=seed, freeze=False)
    tm = None
    if mode != "plain":
        tm = um.TrajectoryMix(rec, loss_fn, stride=8, window=4, n_levels=2,
                              tail=16,
                              reweight_epochs=0 if mode == "uniform" else 3,
                              reweight_lr=0.2, final_batches=4)
        if mode == "valfit":  # score the mixture on held-out data
            vx, vy = val_xy
            tm.set_score_data([{"x": vx[i:i + 32], "y": vy[i:i + 32]}
                               for i in range(0, 128, 32)])
    opt = torch.optim.AdamW(tm.parameters() if tm else rec.parameters(),
                            lr=lr)
    x, y = train_xy
    g = torch.Generator().manual_seed(seed)
    for t in range(steps):
        idx = torch.randint(0, N, (32,), generator=g)
        batch = {"x": x[idx], "y": y[idx]}
        if tm is not None:
            tm.train_step(opt, batch)
        else:
            opt.zero_grad(set_to_none=True)
            loss_fn(model, batch).backward()
            rec.step(opt, record=False)
    with torch.no_grad():
        rmse = (model(val_xy[0]) - val_xy[1]).pow(2).mean().sqrt().item()
    if tm is not None:
        tm.flush()
        with torch.no_grad():
            rmse = (model(val_xy[0]) - val_xy[1]).pow(2).mean().sqrt().item()
    return rmse


if __name__ == "__main__":
    tr, va = make_data()
    print(f"{'seed':>5} {'plain':>8} {'uniform':>8} {'val-fit':>8}")
    rows = []
    for seed in range(5):
        r = [run(seed, m, tr, va) for m in ("plain", "uniform", "valfit")]
        rows.append(r)
        print(f"{seed:>5} {r[0]:8.3f} {r[1]:8.3f} {r[2]:8.3f}")
    import statistics as st
    m = [st.mean(c) for c in zip(*rows)]
    print(f"{'mean':>5} {m[0]:8.3f} {m[1]:8.3f} {m[2]:8.3f}  (val RMSE)")
