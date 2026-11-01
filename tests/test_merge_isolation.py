"""Prove a merge touches nothing but the target B matrices.

Before firing a merge by hand: poison every parameter's .grad with a
known pattern and every optimizer buffer with a sentinel. A merge may
change ONLY the matched modules' B; grads, optimizer state, non-target
params, and the live model's other buffers must be bit-identical. Also:
a zero-signal merge (w stays 0) must leave B where it was.
"""
import hashlib
import torch
from torch import nn
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root when running from a checkout

import updatemix as um


def ph(t):
    t = t.detach().cpu()
    if t.dtype not in (torch.float32, torch.float64):
        t = t.float()
    return hashlib.md5(t.numpy().tobytes()).hexdigest()


def snap(model, opt):
    ps = {n: ph(p) for n, p in model.named_parameters()}
    gs = {n: (ph(p.grad) if p.grad is not None else None)
          for n, p in model.named_parameters()}
    st = {f"{i}.{k}": ph(v) for i, gp in enumerate(opt.param_groups)
          for j, p in enumerate(gp["params"])
          for k, v in opt.state[p].items() if torch.is_tensor(v)}
    return ps, gs, st


class Blk(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.k_proj = nn.Linear(16, 16), nn.Linear(16, 16)

    def forward(self, x):
        return torch.tanh(self.q_proj(x) + self.k_proj(x))


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blks = nn.ModuleList(Blk() for _ in range(4))

    def forward(self, x):
        for b in self.blks:
            x = b(x)
        return x


def loss_fn(model, batch):
    return nn.functional.mse_loss(model(batch["input_ids"]), batch["labels"])


torch.manual_seed(0)
m = Tiny()
h = um.attach_hierarchy(m, "*q_proj*", loss_fn, rank=8, window=5, n_levels=2)
opt = torch.optim.SGD(h.parameters(), lr=0.05, momentum=0.9)
batches = [{"input_ids": torch.randn(2, 16), "labels": torch.randn(2, 16)}
           for _ in range(5)]

for b in batches[:4]:                   # window=5 -> no merge fired yet
    h.train_step(opt, b)
assert h.stats()["merged"] == [], h.stats()["merged"]
assert all(len(v) == 4 for v in h.rec.history.values())

# poison grads and optimizer buffers
with torch.no_grad():
    for n, p in m.named_parameters():
        if p.requires_grad:
            p.grad = torch.full_like(p, 1234.0)
    for gp in opt.param_groups:
        for p in gp["params"]:
            for k, v in opt.state[p].items():
                if torch.is_tensor(v):
                    v.fill_(-98765.0)

targets = {n for n in h.mods}           # FQNs of the *q_proj* holders
ps0, gs0, st0 = snap(m, opt)
h._merge(0, partial=True)               # <-- the window-close event
ps1, gs1, st1 = snap(m, opt)

changed = {n for n in ps0 if ps0[n] != ps1[n]}
assert changed and changed <= {f"blks.{i}.q_proj.B" for i in range(4)}, \
    f"merge touched: {changed}"
assert gs0 == gs1, "merge wrote into some .grad"
assert st0 == st1, "merge touched optimizer state"
assert all(mod._B_override is None for mod in h.mods.values())
print("merge changed only:", sorted(changed))
print("grads + optimizer state bit-identical: OK")

# zero-signal merge: w stays 0 -> B must not move off the current model
for v in h.rec.history.values():
    v.clear()
h._data = [[] for _ in h._data]
B0 = {n: mod.B.detach().clone() for n, mod in h.mods.items()}
h.loss_fn = lambda model, batch: model(batch["input_ids"]).sum() * 0
for _ in range(3):
    with h.rec.record_step():
        pass                            # zero deltas: honest identity test
h._data[0].extend(batches[:3])          # replay refs for the merge
h._merge(0, partial=True)
worst = max((h.mods[n].B - B0[n]).abs().max().item() for n in h.mods)
print(f"zero-signal merge: max |dB| = {worst:.3e} (w=0 identity)")
assert worst < 1e-5, worst
print("ISOLATION OK")
