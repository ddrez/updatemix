"""Phase 2 — learn to re-weight the recorded updates.

For every module the mixture matrix is

    B_eff = T * softmax(w) @ dB        (w: [T] logits, init 0)

At w = 0 this is exactly the uniform average = B_T, i.e. **the mixture
starts at exactly the phase-1 adapter** and only deviates if it pays.
The learned part is T floats per module — the whole point of the
experiments this package came from was that these few floats beat the
trajectory they re-weight (~20% relative val gain on Qwen3-0.6B/ARC).

B_eff is a plain matrix: at inference it is folded once into the base
weight (`.fold()`) and costs *nothing* per token.
"""

import fnmatch

import torch
import torch.nn.functional as F
# from torch.utils.checkpoint import checkpoint
from torch import nn

from .record import LoRAHolder

class MixtureLoRA(nn.Module):
    """base(x) + x @ A^T @ B_eff^T, with B_eff a learned mixture of dB."""

    def __init__(self, base, A, dB, is_emb, floor=0.0):
        super().__init__()
        self.base = base
        self.is_emb = is_emb
        self.floor = floor
        self.register_buffer("A", A.to(torch.bfloat16))  # [r, d_in] fp32
        self.register_buffer("dB", dB.to(torch.bfloat16))  # [T, d_out, r]
        self.T = dB.shape[0]
        self.w = nn.Parameter(torch.zeros(self.T).to(torch.bfloat16))
        self._be_cache = None
        self._be_key = None
        # self.name = "MixtureLoRA_scores"

    def mixture(self):
        a = torch.softmax(self.w, 0)
        if self.floor:
            a = (1 - self.floor) * a + self.floor / self.T
        return a

    def B_eff(self):
        """T * softmax(w) @ dB, cached across no-grad calls.

        The einsum streams the whole [T, d_out, r] dB buffer — T times
        the steady-state LoRA cost.  B_eff only depends on w, and w only
        changes at optimizer steps, so under no-grad (eval / DFS
        inference) we memoize on (data_ptr, version) and pay the T-fold
        read once instead of on every token forward.
        """
        if not torch.is_grad_enabled():
            key = (self.w.data_ptr(), self._w_version())
            if self._be_cache is not None and self._be_key == key:
                return self._be_cache
            a = self.mixture().to(self.dB.dtype)
            Beff = self.T * torch.einsum("t,tDR->DR", a, self.dB)
            self._be_cache, self._be_key = Beff, key
            return Beff
        a = self.mixture().to(self.dB.dtype)
        return self.T * torch.einsum("t,tDR->DR", a, self.dB)

    def _w_version(self):
        try:
            return self.w._version
        except RuntimeError:  # inference tensors do not track versions
            return 0

    def forward(self, x):
        y = self.base(x)
        # y = checkpoint(self.base, x, use_reentrant=False)
        xin = (
            F.one_hot(x, self.A.shape[1]).to(torch.bfloat16)
            if self.is_emb
            else x.to(torch.bfloat16)
        )
        lora = (xin.to(torch.bfloat16) @ self.A.t().to(torch.bfloat16)) @ self.B_eff().t().to(torch.bfloat16)
        return y + lora.to(y.dtype)


class FrozenLoRA(nn.Module):
    """The uniform baseline: base(x) + x @ A^T @ B_T^T, nothing trainable."""

    def __init__(self, base, A, BT, is_emb):
        super().__init__()
        self.base = base
        self.is_emb = is_emb
        self.register_buffer("A", A)
        self.register_buffer("B", BT)

    def forward(self, x):
        xin = (
            F.one_hot(x, self.A.shape[1]).to(torch.bfloat16)
            if self.is_emb
            else x.to(torch.bfloat16)
        )
        y = self.base(x)
        lora = (xin.to(torch.bfloat16) @ self.A.t().to(torch.bfloat16)) @ self.B.t().to(torch.bfloat16)
        return y + lora.to(y.dtype)


class Mixture(nn.Module):
    """All injected mixture modules + diagnostics.

    `parameters()` is just the per-module `[T]` logits. `named_logits()`
    and `diagnostics()` let you inspect what the mixture learned (which
    updates it keeps and how sharply).
    """

    def __init__(self, model, names, learnable):
        super().__init__()
        # plain attribute, NOT a registered submodule: registering the
        # model would make mix.parameters() return every base weight too
        self.__dict__["model"] = model
        self.names = names
        self.holders = nn.ModuleList([model.get_submodule(n) for n in names])
        self._learnable = learnable  # parallel bool list

    def named_logits(self):
        for n, h in zip(self.names, self.holders):
            if isinstance(h, MixtureLoRA):
                yield n, h.w

    def parameters(self, recurse=True):
        """Overridden: *only* the per-module [T] logits.

        The holders keep their base module registered (so device / dtype
        transfers and hooks keep working), which would otherwise leak
        every base weight into parameters() — and straight into the
        optimizer.  `state_dict()` is unaffected.
        """
        for _, w in self.named_logits():
            yield w

    def named_parameters(self, prefix="", recurse=True):
        for n, w in self.named_logits():
            yield n, w

    def diagnostics(self):
        out = {}
        for n, h in zip(self.names, self.holders):
            if not isinstance(h, MixtureLoRA):
                continue
            with torch.no_grad():
                a = h.mixture().double()
                p = a[a > 0]
                out[n] = {
                    "max_slot": int(a.argmax()),
                    "max_prob": float(a.max()),
                    "entropy": float(-(p * p.log()).sum()),
                    "uniform_entropy": float(torch.log(torch.tensor(float(h.T)))),
                }
        return out

    @torch.no_grad()
    def bake(self):
        """Replace every MixtureLoRA with a FrozenLoRA at its final B_eff.

        This drops the [T, d_out, r] dB buffers for good: after baking,
        per-forward cost no longer involves T at all (the cached B_eff is
        just a plain LoRA B read), and the trajectory memory goes back to
        the pool.  Call it once w is trained, before inference — unlike
        `fold()` it needs nothing from the (possibly quantized) base.
        """
        for i, (n, h) in enumerate(zip(self.names, self.holders)):
            if not isinstance(h, MixtureLoRA):
                continue
            Beff = h.B_eff().to(torch.bfloat16)
            frozen = FrozenLoRA(h.base, h.A, Beff, h.is_emb)
            frozen.to(h.dB.device)
            parent_name, _, attr = n.rpartition(".")
            parent = self.model.get_submodule(parent_name) if parent_name else self.model
            setattr(parent, attr, frozen)
            self.holders[i] = frozen
            h._be_cache = None
            h.dB = None  # release the [T, d_out, r] buffer even while h lives
        return self.model

    @torch.no_grad()
    def fold(self):
        """Absorb B_eff into the base weights and unwrap holders.

        After this the model is a vanilla model with a re-weighted
        fine-tune baked in — no adapters, no per-token overhead.
        """
        for n, h in zip(self.names, self.holders):
            parent_name, _, attr = n.rpartition(".")
            parent = self.model.get_submodule(parent_name) if parent_name else self.model
            if isinstance(h, MixtureLoRA):
                Beff, A = h.B_eff().to(torch.bfloat16), h.A.to(torch.bfloat16)
            else:
                Beff, A = h.B.to(torch.bfloat16), h.A.to(torch.bfloat16)
            base = h.base
            # delta in functional space is x @ (Beff @ A)^T; the Linear
            # weight wants rows=output so we add Beff@A directly, while
            # an embedding table (rows=tokens) wants the transpose.
            w = (Beff @ A).t().contiguous() if h.is_emb else Beff @ A
            # rebind to a private copy: with tied weights (e.g. Qwen3
            # embed_tokens <-> lm_head) an in-place add would smear each
            # holder's delta onto the other's path.  Unfolding here means
            # the deployed model is untied, and each path's forward is
            # exactly what it was with adapters.
            base.weight = nn.Parameter(
                base.weight.detach().clone().to(base.weight.dtype)
            )
            target = base.weight
            target.add_(w.to(target.dtype))
            setattr(parent, attr, base)
        return self.model


def attach_mixture(
    model, log, targets=None, freeze_base=True, dtype=None, floor=0.0
):
    """Inject the mixture adapter over every logged module.

    log: UpdateLog (loaded or from a Recorder).
    targets: optional glob(s) over module FQNs to make *learnable*;
        matched modules get trainable logits w, the rest get the frozen
        uniform B_T (so unselected modules still reproduce phase 1).
        Default: all logged modules learnable.
    dtype: storage dtype for dB on GPU (e.g. torch.bfloat16 to halve the
        ~T·rank·d memory); mixing still happens in fp32.
    floor: fixed uniform-mixing fraction, a = (1-floor)·softmax(w) +
        floor/T. 0 = free mixture. A floor (0.3–0.5) damps the over-
        sharpening that overfits small datasets when T is large.
    """
    _inject(
        model,
        log,
        targets=targets,
        freeze_base=freeze_base,
        wrap=lambda base, A, dB, is_emb: MixtureLoRA(base, A, dB, is_emb, floor),
        uniform=lambda base, A, dB, is_emb: FrozenLoRA(
            base, A, dB.sum(0).to(torch.bfloat16), is_emb
        ),
        dtype=dtype,
    )
    names = _holder_names(model, log)
    learnable = [
        isinstance(model.get_submodule(n), MixtureLoRA) for n in names
    ]
    return Mixture(model, names, learnable)


def attach_uniform(model, log, freeze_base=True, dtype=None):
    """Baseline: freeze every module at its full trajectory sum B_T.

    Reproduces the end-of-phase-1 model exactly (given the same fp32
    math); use it as the number the mixture has to beat.
    """
    _inject(
        model,
        log,
        targets=None,
        freeze_base=freeze_base,
        wrap=None,
        uniform=lambda base, A, dB, is_emb: FrozenLoRA(
            base, A, dB.sum(0).to(torch.bfloat16), is_emb
        ),
        dtype=dtype,
    )
    return model


def _inject(model, log, targets, freeze_base, wrap, uniform, dtype):
    for name, entry in log.mods.items():
        learn = wrap is not None and _selected(name, targets)
        A = entry["A"].to(torch.bfloat16)
        dB = entry["dB"]
        if dtype is not None:
            dB = dB.to(dtype)
        parent_name, _, attr = name.rpartition(".")
        try:
            parent = model.get_submodule(parent_name) if parent_name else model
        except AttributeError:
            raise ValueError(
                f"log module {name!r} not present in model — load the same architecture"
            )
        module = parent.get_submodule(attr) if attr else model
        if isinstance(module, (MixtureLoRA, FrozenLoRA, LoRAHolder)):
            base = module.base  # re-attach: unwrap any previous holder
        else:
            base = module
        holder = (
            (wrap if learn else uniform)(base, A, dB, entry["is_emb"])
        )
        holder.to(base.weight.device)
        setattr(parent, attr, holder)
    if freeze_base:
        # model.requires_grad_(False)
        for n, h in model.named_modules():
            if isinstance(h, MixtureLoRA):
                h.w.requires_grad_(True)
            else:
                try:
                    h.w.requires_grad_(False)
                except:
                    pass


def detach_holders(model):
    """Unwrap every holder (LoRA/Mixture/Frozen) back to its base module.

    Lets one loaded model instance serve many puzzles: `bake()` before
    inference, `detach_holders()` for the next puzzle, then attach fresh
    holders — no multi-GB `from_pretrained` per puzzle.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, (LoRAHolder, MixtureLoRA, FrozenLoRA)):
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr, module.base)
    return model


def _selected(name, targets):
    if not targets:
        return True
    pats = [targets] if isinstance(targets, str) else list(targets)
    return any(fnmatch.fnmatch(name, p) for p in pats)


def _holder_names(model, log):
    return [n for n in log.mods if _exists(model, n)]


def _exists(model, name):
    try:
        model.get_submodule(name)
        return True
    except AttributeError:
        return False
