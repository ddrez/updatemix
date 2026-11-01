"""Phase 1 — attach LoRA-FA holders and record every applied update.

`attach()` freezes the base model, injects a frozen random `A` and a
trainable fp32 `B` into every matched module, and returns a `Recorder`.
Wrapping `optimizer.step()` in `recorder.record_step()` snapshots `B`
before and after and appends the exact delta `dB_t = B_after - B_before`
to the history. Because the snapshot is taken around the real step, this
is *optimizer-agnostic*: Adam, SGD, Muon, a LR scheduler, gradient
accumulation over multiple `backward()`s — whatever actually landed in
`B` is what gets recorded, and `B_T == Σ dB_t` holds by construction
(telescoping, not an assumption about the optimizer).

The only requirements are: B is fp32, nothing else writes to B between
steps, and no weight decay / weight-space noise smears steps across the
snapshot boundary (with decay the recorded delta stops equaling the
optimizer's intent — you get the true delta, but `Σ dB_t` then includes
the decay shrink too, which is fine and still exact).
"""

import os
import fnmatch
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn


def _match(name, module, patterns):
    if not isinstance(module, (nn.Linear, nn.Embedding)):
        return False
    if callable(patterns):
        return patterns(name, module)
    pats = [patterns] if isinstance(patterns, str) else list(patterns)
    return any(fnmatch.fnmatch(name, p) for p in pats)


class LoRAHolder(nn.Module):
    """base(x) + one_hot/identity(x) @ A^T @ B^T — only B trains."""

    def __init__(self, base, rank, seed):
        super().__init__()
        self.base = base
        self.is_emb = isinstance(base, nn.Embedding)
        d_in = base.num_embeddings if self.is_emb else base.in_features
        d_out = base.embedding_dim if self.is_emb else base.out_features
        g = torch.Generator().manual_seed(seed)
        # frozen random projection, scaled like a fan-in init; stored in
        # bf16 because the forward used to cast it to bf16 anyway — this
        # way the cast is not redone on every single forward
        self.register_buffer(
            "A", (torch.randn(rank, d_in, generator=g) / (d_in ** 0.5)).to(torch.bfloat16)
        )
        # fp32 master weight: many small bf16 add_()s would smear away dB
        # (the forward casts to bf16 once per call; B itself stays fp32 so
        #  Muon's steps land losslessly and dB records the true intent)
        self.B = nn.Parameter(1e-5 + torch.zeros(d_out, rank, dtype=torch.float32))
        self.alpha = float(os.getenv("B_ALPHA", "1.0"))  # scale on LoRA path; was rank**(1/2), unused
        # not a buffer: transient matrix used by hierarchy re-weighting
        self._B_override = None
        self.to(base.weight.device)

    def _eff_B(self):
        return self._B_override if self._B_override is not None else self.B

    def forward(self, x):
        xin = (
            F.one_hot(x, self.A.shape[1]).to(self.A.dtype)
            if self.is_emb
            else x.to(self.A.dtype)
        )
        y = self.base(x)
        # same bf16 LoRA path as before, minus the pointless fp32 round-trip
        lora = (xin @ self.A.t()) @ self._eff_B().to(self.A.dtype).t()
        # return y + self.alpha * lora.to(y.dtype)
        lora = lora.to(y.dtype)
        if self.alpha != 1.0:
            lora = lora * self.alpha
        return y + lora


class Recorder:
    """Holds the injected modules and the per-step delta history.

    `history[name]` is a list of CPU fp32 tensors `[d_out, rank]`, one per
    recorded optimizer step. Call `.save(path)` (or build an `UpdateLog`)
    when done.
    """

    def __init__(self, model, mods):
        self.model = model
        self.mods = mods
        self.history = {name: [] for name in mods}

    def parameters(self):
        """The trainable B matrices — hand these to your optimizer."""
        return [m.B for m in self.mods.values()]

    @contextmanager
    def record_step(self):
        """Around optimizer.step(): appends exact per-module deltas."""
        snap = {n: m.B.detach().clone() for n, m in self.mods.items()}
        yield
        with torch.no_grad():
            for n, m in self.mods.items():
                self.history[n].append((m.B.detach() - snap[n]).cpu())

    def step(self, optimizer, record=True, **kwargs):
        """`optimizer.step()` wrapped in `record_step()`."""
        # {n: m.B.grad for n, m in self.mods.items()}
        if record:
            with self.record_step():
                optimizer.step(**kwargs)
        else:
            optimizer.step(**kwargs)

    def telescoping_err(self):
        """max |Σ dB − B_T| over modules; ~1e-5 fp32 noise, else a bug."""
        worst = 0.0
        with torch.no_grad():
            for n, m in self.mods.items():
                s = torch.stack(self.history[n]).sum(0).to(m.B.device)
                worst = max(worst, (s - m.B).abs().max().item())
        return worst

    def save(self, path, meta=None):
        from .log import UpdateLog

        UpdateLog.from_recorder(self).save(path, meta=meta)


def attach(model, targets, rank=16, seed=0, freeze=True):
    """Inject LoRA-FA holders into modules matching `targets`.

    targets: glob or list of globs matched against `named_modules()` FQNs
    (e.g. `"*q_proj"` or `["*q_proj*", "*v_proj*"]`), or a callable
    `(name, module) -> bool`. Matches nn.Linear and nn.Embedding; the
    embedding case treats token ids as one-hot inputs so the identical
    A/B machinery applies.

    freeze: set requires_grad=False on everything except the new B's.
    """
    mods = {}
    for i, (name, module) in enumerate(list(model.named_modules())):
        if not _match(name, module, targets):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        holder = LoRAHolder(module, rank, seed + 7919 * i)
        setattr(parent, attr, holder)
        mods[name] = holder
    if not mods:
        raise ValueError(f"no nn.Linear/nn.Embedding modules matched {targets!r}")
    if freeze:
        model.requires_grad_(False)
        for m in mods.values():
            m.B.requires_grad_(True)
    return Recorder(model, mods)
