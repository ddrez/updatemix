"""CPU unit tests for TrajectoryMix (mixture math, commit, grads)."""
import numpy as np
import torch
from torch import nn

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root when running from a checkout

import updatemix as um


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


REF = {"tok": 0}


def loss_fn(model, batch):
    REF["tok"] += batch["x"].numel()
    return ((model(batch["x"]) - batch["y"]) ** 2).mean()


def make(seed=0):
    torch.manual_seed(seed)
    m = M()
    rec = um.attach(m, "*fc1*", rank=4, seed=seed, freeze=False)
    return m, rec


def step(opt, tm, i):
    g = torch.Generator().manual_seed(1000 + i)
    return {"x": torch.randn(4, 8, generator=g),
            "y": torch.randn(4, 1, generator=g)}


def snapshots(tm):
    return {n: m.B.detach().float().cpu().clone()
            for n, m in tm.mods.items()}


def test_uniform_commit_exact():
    """frw=0 flush: B must be the exact mean of the candidate set."""
    m, rec = make()
    tm = um.TrajectoryMix(rec, loss_fn, stride=2, window=2, n_levels=2,
                          tail=100, alpha=1.0, reweight_epochs=0)
    opt = torch.optim.SGD(tm.parameters(), lr=0.05)
    mine = []  # replicate the snapshot schedule
    for i in range(10):
        tm.train_step(opt, step(opt, tm, i))
        if tm.steps % tm.stride == 0:
            mine.append(snapshots(tm))
    # replicate binning: raw bins of `window`, rolled into level 1
    raw = mine
    top = []
    while len(raw) >= tm.window:
        top.append({n: torch.stack([p[n] for p in raw[:tm.window]]).mean(0)
                    for n in raw[0]})
        raw = raw[tm.window:]
    cands = top + ([{n: torch.stack([p[n] for p in raw]).mean(0)
                     for n in raw[0]}] if raw else [])
    cands.append(snapshots(tm))  # live endpoint
    K = len(cands)
    tm.flush()
    for n, h in tm.mods.items():
        exp = (torch.stack([p[n] for p in cands]).mean(0)
               .to(h.B.device, h.B.dtype))
        assert torch.allclose(h.B.detach().float().cpu(),
                              exp.float().cpu(), atol=1e-6), n
    assert tm.committed is not None
    a = next(iter(tm.committed.values()))
    assert abs(sum(a) - 1) < 1e-5
    print(f"uniform commit exact: OK (K={K})")


def test_alpha_interp():
    m, rec = make()
    tm = um.TrajectoryMix(rec, loss_fn, stride=2, window=2, n_levels=1,
                          tail=0, alpha=0.5, reweight_epochs=0)
    opt = torch.optim.SGD(tm.parameters(), lr=0.05)
    for i in range(6):
        tm.train_step(opt, step(opt, tm, i))
    B0 = snapshots(tm)
    cands = [{k: v.clone() for k, v in p.items()}
             for p in tm.bins[0]] + [B0]  # incl. live endpoint
    tm.flush()
    for n, h in tm.mods.items():
        mixed = torch.stack([p[n] for p in cands]).mean(0)
        exp = B0[n] + 0.5 * (mixed - B0[n])
        got = h.B.detach().float().cpu()
        assert torch.allclose(got, exp, atol=1e-5), (n, (got - exp).abs().max())
    print("alpha interpolation: OK")


def test_grad_formula():
    """Manual dL/dw_t = s_t(g_t - s.g) vs real autograd through softmax."""
    m, rec = make()
    tm = um.TrajectoryMix(rec, loss_fn, stride=2, window=2, n_levels=1,
                          tail=100, reweight_epochs=1, final_batches=2)
    opt = torch.optim.SGD(tm.parameters(), lr=0.05)
    for i in range(6):
        tm.train_step(opt, step(opt, tm, i))
    P = list(tm.bins[0]) + [snapshots(tm)]
    names = list(P[0])
    K = len(P)
    Sv = {n: torch.stack([p[n] for p in P]).float().reshape(K, -1)
          for n in names}
    batch = tm._data[-1]
    w = torch.linspace(-0.4, 0.4, K)
    # autograd ground truth: differentiable override (no copy_ into leaf)
    truth = {}
    for n in names:
        s = torch.softmax(w, 0)
        Bm = (s @ Sv[n]).view(tm.mods[n].B.shape).to(
            device=tm.mods[n].B.device,
            dtype=tm.mods[n].B.dtype).requires_grad_(True)
        rec.mods[n]._B_override = Bm
        l = loss_fn(m, batch)
        gB = torch.autograd.grad(l, Bm)[0].detach().cpu().float().reshape(-1)
        rec.mods[n]._B_override = None
        gi = Sv[n] @ gB
        manual = s * (gi - s @ gi)
        # recompute autograd dL/dw directly
        w_ = w.clone().requires_grad_(True)
        s_ = torch.softmax(w_, 0)
        Bm_ = (s_ @ Sv[n]).view(tm.mods[n].B.shape).to(
            device=tm.mods[n].B.device, dtype=tm.mods[n].B.dtype)
        rec.mods[n]._B_override = Bm_
        l_ = loss_fn(m, batch)
        rec.mods[n]._B_override = None
        auto = torch.autograd.grad(l_, w_)[0]
        d = (auto - manual).abs().max().item()
        assert d < 5e-3 * max(1.0, auto.abs().max().item()), (n, d,
                                                              auto.abs().max().item())
        truth[n] = d
    print(f"grad formula matches autograd: OK "
          f"(max|diff|={max(truth.values()):.2e}, bf16-limited)")


def test_reweight_runs_and_bounded():
    m, rec = make()
    tm = um.TrajectoryMix(rec, loss_fn, stride=2, window=2, n_levels=2,
                          tail=3, reweight_epochs=2, reweight_lr=0.1,
                          final_batches=2, floor=0.2)
    opt = torch.optim.SGD(tm.parameters(), lr=0.05)
    for i in range(12):
        tm.train_step(opt, step(opt, tm, i))
        assert len(tm._data) <= 2
        assert all(len(b) <= 2 for b in tm.bins[:1])
    before = REF["tok"]
    cm = tm.flush()
    paid = REF["tok"] - before
    assert all(bool(torch.isfinite(torch.tensor(a)).all())
               for a in cm.values())
    for n in tm.mods:
        assert tm.mods[n]._B_override is None
    print(f"reweight flush runs: OK (extra fwd-bwd elems={paid}, "
          f"K={len(next(iter(cm.values())))})")


def test_decay_prior():
    """frw=0 + decay: committed weights are the geometric recency prior."""
    m, rec = make()
    tm = um.TrajectoryMix(rec, loss_fn, stride=2, window=2, n_levels=1,
                          tail=0, reweight_epochs=0, decay=0.5)
    opt = torch.optim.SGD(tm.parameters(), lr=0.05)
    for i in range(6):
        tm.train_step(opt, step(opt, tm, i))
    cm = tm.flush()
    K = len(next(iter(cm.values())))
    p = torch.tensor([0.5 ** (K - 1 - i) for i in range(K)])
    p = (p / p.sum()).tolist()
    for n, a in cm.items():
        assert max(abs(x - y) for x, y in zip(a, p)) < 1e-6, (n, a, p)
    print(f"decay prior: OK (K={K}, newest weight {max(p):.3f})")


if __name__ == "__main__":
    test_uniform_commit_exact()
    test_alpha_interp()
    test_grad_formula()
    test_reweight_runs_and_bounded()
    test_decay_prior()
    print("ALL PASS")
