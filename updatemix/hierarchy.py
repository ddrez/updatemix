"""Hierarchical updatemix — collapse update windows into a pyramid online.

Instead of phase 1 (record everything) then phase 2 (re-weight the whole
trajectory), the collapse runs *inside* training:

    level 0:  d1 d2 ... d20                d1 d2 ... d20      (raw steps)
                 \\__ reweight __/ v           \\__ reweight __/ v
    level 1:            E1                          E2         ... 20 E's = 400 steps
                 \\___________ reweight on all 400 batches __________/ v
    level 2:                              E1

`train_step` records the step's dB like `Recorder` does; every `window`
steps it learns the per-module mixture logits `w` for the window — by
gradient descent on the *same batches* that produced the updates — and
commits `B <- C + K*softmax(w) @ S` into the live `B` (C = B - sum(S),
the state before the window). The merged update is pushed one level up,
and a level collapses whenever it holds `window` inputs, so level 2 sees
window^2 steps of data, level 3 window^3, and so on.

Each re-weight starts at `w = 0`, i.e. exactly the current model
(`K*uniform @ S == sum(S)`), and only deviates if the replay data says
so — the same invariant as phase 2, now per window.

The mixture gradient is analytic (softmax chain rule applied to the
loss gradient w.r.t. the effective B), so re-weighting never needs an
autograd path through the [K, d_out, rank] stack:

    dB_eff/dw_t = K * (1-floor) * s_t * (S_t - sum_k s_k S_k)
    dL/dw_t     = (1-floor) * s_t * (K <g,S_t> - <g, K s@S>),  g = dL/dB_eff

VRAM discipline (the point of the design):
  * per-step deltas live on CPU from the moment `record_step` produces
    them; no [K, d_out, rank] stack is ever uploaded — per replay batch
    only one [d_out, rank] override per module crosses PCIe, and its
    gradient comes straight back;
  * replay batches are kept on CPU (refs shared across level buffers)
    and moved to the GPU one batch at a time;
  * every buffer the merge consumes is cleared inside the merge, the
    `_B_override` hooks are always restored (finally), and the CUDA
    cache is trimmed once per merge.
Peak extra GPU memory during a merge = sum of [d_out, rank] overrides
+ one ordinary forward/backward. Nothing scales with window or level.
"""

import torch
from torch import nn

from .record import LoRAHolder, attach


def _tree_map(fn, x):
    if torch.is_tensor(x):
        return fn(x)
    if isinstance(x, dict):
        return {k: _tree_map(fn, v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_tree_map(fn, v) for v in x)
    return x


def _to_cpu(batch):
    return _tree_map(lambda t: t.detach().cpu(), batch)


def _to(batch, device):
    return _tree_map(lambda t: t.to(device), batch)


class _Adam:
    """Minimal CPU Adam for the [K] logits (no autograd involved)."""

    def __init__(self, tensors, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.p = list(tensors)
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [torch.zeros_like(t) for t in self.p]
        self.v = [torch.zeros_like(t) for t in self.p]
        self.t = 0

    def step(self, grads):
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        c2 = 1 - self.b2 ** self.t
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            p.addcdiv_(m / c1, (v / c2).sqrt_().add_(self.eps), value=-self.lr)


class Hierarchy:
    """Streaming windowed re-weighting over a `Recorder`.

    Usage:

        def loss_fn(model, batch): ...

        h = um.attach_hierarchy(model, "*q_proj*", loss_fn,
                                rank=16, window=20, n_levels=2)
        opt = torch.optim.AdamW(h.parameters(), lr=1e-3)
        for batch in loader:
            h.train_step(opt, batch)     # trains + records + auto-merges

        um.fold_holders(h.model)         # optional: bake B into the base

    Args:
        window: inputs a merge needs at every level (20 in the scheme
            this came from: 20 steps -> 1 update, 20 updates -> 1).
        n_levels: height of the pyramid; level `n_levels-1` merges are
            committed to B but their result is not tracked further.
            Must be >= 1.
        reweight_epochs: passes over the replay data per merge.
        reweight_lr: Adam step size on the logits (logit units/step).
        floor: uniform fraction in `a = (1-floor)*softmax(w) + floor/K`,
            damps over-sharpening exactly like phase 2.
        replay_cap: max batches kept per level buffer; when exceeded the
            buffer is thinned to every other batch (span preserved, so
            high levels replay the whole block, just sparser). None =
            keep everything (default; fine for level-0/1 windows of
            token ids — raise it or set a cap before adding level 3+).
    """

    def __init__(
        self,
        recorder,
        loss_fn,
        window=20,
        n_levels=2,
        reweight_epochs=1,
        reweight_lr=0.1,
        floor=0.0,
        replay_cap=None,
    ):
        if n_levels < 1:
            raise ValueError("n_levels must be >= 1")
        if window < 2:
            raise ValueError("window must be >= 2")
        self.rec = recorder
        self.model = recorder.model
        self.mods = recorder.mods
        self.loss_fn = loss_fn
        self.window = int(window)
        self.n_levels = int(n_levels)
        self.reweight_epochs = int(reweight_epochs)
        self.reweight_lr = float(reweight_lr)
        self.floor = float(floor)
        self.replay_cap = replay_cap
        self.steps = 0
        # stacks[l] = list of merged updates (dict name -> [d_out,r] CPU)
        # awaiting level-l merge; level 0's stack is rec.history directly.
        self._stacks = [[] for _ in range(n_levels)]
        # per-level replay buffers holding shared CPU refs
        self._data = [[] for _ in range(n_levels)]
        self.merged = []  # (level, K) events

    # ------------------------------------------------------------- run

    def parameters(self):
        """The trainable B matrices — hand these to your optimizer."""
        return self.rec.parameters()

    def train_step(self, opt, batch):
        """One training step: backward, recorded optimizer step, merges.

        Returns the loss value. When a window closes this runs the
        level-0 merge (and whatever cascades above it) before returning.
        """
        opt.zero_grad(set_to_none=True)
        loss = self.loss_fn(self.model, batch)
        loss.backward()
        self.rec.step(opt)
        cb = _to_cpu(batch)
        for buf in self._data:
            buf.append(cb)
            if self.replay_cap and len(buf) > self.replay_cap:
                buf[:] = buf[::2]
        self.steps += 1
        if self.steps % self.window == 0:
            self._merge(0)
        return float(loss.detach())

    def flush(self):
        """Merge partial windows at every level (end of an epoch/run)."""
        for lvl in range(self.n_levels):
            if self._size(lvl) > 0:
                self._merge(lvl, partial=True)

    def stats(self):
        return {
            "steps": self.steps,
            "merged": [(int(l), int(k)) for l, k in self.merged],
            "pending": [self._size(l) for l in range(self.n_levels)],
            "buffers": [len(b) for b in self._data],
        }

    def _size(self, lvl):
        if lvl == 0:
            h = self.rec.history
            return len(next(iter(h.values()))) if h else 0
        return len(self._stacks[lvl])

    # ----------------------------------------------------------- merge

    def _collect(self, lvl, partial):
        """Stack this level's K updates to {name: [K,d_out,r]}; clear them."""
        if lvl == 0:
            hist = self.rec.history
            if not hist:
                return None, 0
            K = len(next(iter(hist.values())))
            if any(len(v) != K for v in hist.values()):
                raise RuntimeError("ragged history — did something "
                                   "else record into this Recorder?")
            if K == 0 or (K < self.window and not partial):
                return None, 0
            S = {n: torch.stack(v) for n, v in hist.items()}
            for v in hist.values():
                v.clear()
        else:
            lst = self._stacks[lvl]
            K = len(lst)
            if K == 0 or (K < self.window and not partial):
                return None, 0
            S = {n: torch.stack([u[n] for u in lst]) for n in lst[0]}
            lst.clear()
        return S, K

    def _take_batches(self, lvl):
        buf = self._data[lvl]
        out = list(buf)
        buf.clear()
        return out

    def _merge(self, lvl, partial=False):
        S, K = self._collect(lvl, partial)
        if S is None:
            return
        batches = self._take_batches(lvl)
        E = self._reweight(S, K, batches)
        del S
        self.merged.append((lvl, K))
        if lvl + 1 < self.n_levels:
            self._stacks[lvl + 1].append(E)
            if len(self._stacks[lvl + 1]) >= self.window:
                self._merge(lvl + 1)
        dev = next(iter(self.mods.values())).B.device
        if dev.type == "cuda":
            torch.cuda.empty_cache()  # once per merge, keeps the pool tidy

    # -------------------------------------------------------- reweight

    def _mix(self, w):
        a = torch.softmax(w, 0)
        if self.floor:
            a = (1 - self.floor) * a + self.floor / a.numel()
        return a

    def _reweight(self, S, K, batches):
        """Fit [K] logits per module on `batches`, commit B, return merge.

        Only the [K] logits are optimized: the forward is run with each
        target module's B replaced by a leaf override and the main
        optimizer (Muon) is never called here, so its buffers, every
        non-target parameter and all `.grad`s are untouched; the sole
        write to the model is the final commit into B.

        Mechanics notes (the merge used to crawl):
          * every contraction is a BLAS gemv on the fp32 view
            `Sv = S.reshape(K, d*r)` — no per-batch `[K,d,r]` passes, and
            `(g, K s@S) == K s.(S@g)` (softmax identity) reuses dS, so
            no second [d,r] read per batch;
          * the per-module GPU override is allocated *once* for the
            merge and refilled in place from w each replay batch — no
            per-batch cudaMalloc/free storm;
          * grads download in bf16 (the forward was bf16 anyway) and
            cast on CPU — same values, half the PCIe.
        """
        names = list(S)
        if K == 1:
            # identity: the single update is already committed in B
            return {n: S[n][0].clone() for n in names}
        if not batches:
            raise RuntimeError("no replay batches for the merge — feed "
                               "them through train_step (or flush)")

        dev = next(iter(self.mods.values())).B.device
        # the history is fp32 CPU already -> .float() is a no-op, reshape
        # is a view: gemv-able [K, d*r] at zero copy cost
        Sv = {n: S[n].float().reshape(K, -1) for n in names}
        Ssum = {n: Sv[n].sum(0) for n in names}
        # C = state before this level's block; w=0 gives K*uniform@S ==
        # Ssum, so Bmat starts at exactly today's B.
        Cf = {
            n: self.mods[n].B.detach().float().cpu().reshape(-1).sub_(Ssum[n])
            for n in names
        }
        Bmats = {
            n: torch.empty(self.mods[n].B.shape, device=dev,
                           dtype=torch.bfloat16).requires_grad_(True)
            for n in names
        }
        for n in names:
            self.mods[n]._B_override = Bmats[n]
        w = {n: torch.zeros(K) for n in names}
        adam = _Adam([w[n] for n in names], self.reweight_lr)
        order = [Bmats[n] for n in names]
        try:
            for _ in range(self.reweight_epochs):
                for batch in batches:
                    bdev = _to(batch, dev)
                    s_of = {n: torch.softmax(w[n], 0) for n in names}
                    with torch.no_grad():  # refill overrides from current w
                        for n in names:
                            b = s_of[n] @ Sv[n]  # gemv [d*r]
                            b.mul_(K * (1 - self.floor))
                            if self.floor:
                                b.add_(Ssum[n], alpha=self.floor)
                            b.add_(Cf[n])
                            Bmats[n].copy_(b.view(Bmats[n].shape).to(torch.bfloat16))
                    loss = self.loss_fn(self.model, bdev)
                    grads = torch.autograd.grad(
                        loss, order, allow_unused=True
                    )
                    gsteps = []
                    for n, g in zip(names, grads):
                        if g is None:
                            gsteps.append(torch.zeros(K))
                            continue
                        g = g.detach().cpu().float().reshape(-1)
                        dS = Sv[n] @ g  # gemv [K]
                        gsteps.append(
                            (1 - self.floor) * K * s_of[n] * (dS - s_of[n] @ dS)
                        )
                    adam.step(gsteps)
                    del bdev, grads, s_of, gsteps, loss
        finally:  # the override must never outlive a merge
            for n in names:
                self.mods[n]._B_override = None
            del Bmats

        # commit: B <- C + K*softmax(w)@S, and report the merged update
        E = {}
        for n in names:
            a = self._mix(w[n])
            Bnew = Cf[n] + K * (a @ Sv[n])
            h = self.mods[n]
            h.B.data.copy_(
                Bnew.view(h.B.shape).to(device=h.B.device, dtype=h.B.dtype)
            )
            E[n] = Bnew.detach().view(h.B.shape).clone()  # fp32 CPU: matches master B
        return E


class TrajectoryMix:
    """Endpoint-mixture updatemix with *deferred* commit.

    The delta-space hierarchy above mixes a window's updates `dB_t`; at
    `w=0` that mixture is the identity (`K*uniform @ S == sum(S)`), so
    there is no safe fallback and mid-run commits kick B while the
    optimizer's moments keep running (measured: worse than plain, +2.6%
    counted cost, 3x seed variance).

    This class mixes *positions*, not steps. During training it only
    snapshots `B` every `stride` steps and compresses the snapshots
    hierarchically (every `window` points in bin `l` average into bin
    `l+1`) — no model calls, no FLOPS, no interference: training is
    exactly the plain path. `window^n_levels` steps cost
    `n_levels * window` matrices of storage, so the pyramid keeps the
    whole run for free.

    At `flush()` the tail of the trajectory — the last `tail` bin
    averages plus the live endpoint — is mixed *once*:
    `B <- (1-alpha) B + alpha * sum_i a_i P_i`, `a = softmax(w)` with
    `w` fit on the last `final_batches` batches (override-leaf
    autograd, same machinery as `_reweight`; `floor` damps toward
    uniform). `w=0` is now *trajectory averaging* (SWA/Polyak), which
    is a known-good default rather than a no-op, so the learned weights
    only have to beat a mixture that already works:

        dL/dw_t = s_t * (g_t - sum_k s_k g_k),  g_i = <dL/dB_eff, P_i>

    The whole reweight costs `epochs * final_batches` extra forward
    passes, once, at the end (~1% of budget) — instead of a window's
    worth of replay every `window` steps.
    """

    def __init__(
        self,
        recorder,
        loss_fn,
        stride=16,
        window=4,
        n_levels=2,
        tail=8,
        alpha=1.0,
        floor=0.0,
        reweight_epochs=1,
        reweight_lr=0.05,
        final_batches=8,
        decay=1.0,
    ):
        if n_levels < 1 or window < 2 or stride < 1:
            raise ValueError("need n_levels>=1, window>=2, stride>=1")
        self.rec = recorder
        self.model = recorder.model
        self.mods = recorder.mods
        self.loss_fn = loss_fn
        self.stride = int(stride)
        self.window = int(window)
        self.n_levels = int(n_levels)
        self.tail = int(tail)
        self.alpha = float(alpha)
        self.floor = float(floor)
        self.reweight_epochs = int(reweight_epochs)
        self.reweight_lr = float(reweight_lr)
        self.final_batches = int(final_batches)
        self.decay = float(decay)  # geometric recency prior over candidates
        self.steps = 0
        # bins[l] = endpoint snapshots (dicts name -> [d_out,rank] fp32 CPU)
        self.bins = [[] for _ in range(n_levels)]
        self._data = []  # CPU refs, last `final_batches` only
        self.committed = None  # {name: [K]} weights at flush, for stats

    def parameters(self):
        return self.rec.parameters()

    def train_step(self, opt, batch):
        """Plain step (one fwd+bwd, counted by loss_fn) + snapshot."""
        opt.zero_grad(set_to_none=True)
        loss = self.loss_fn(self.model, batch)
        loss.backward()
        self.rec.step(opt, record=False)
        self.steps += 1
        self._data.append(_to_cpu(batch))
        if len(self._data) > self.final_batches:
            del self._data[0]
        if self.steps % self.stride == 0:
            self.bins[0].append(self._snapshot())
            while len(self.bins[0]) >= self.window:
                self._rollup(0)
        return float(loss.detach())

    def _snapshot(self):
        return {n: m.B.detach().float().cpu().clone()
                for n, m in self.mods.items()}

    def _rollup(self, lvl, partial=False):
        lst = self.bins[lvl]
        take = lst[:] if partial else lst[: self.window]
        del lst[: len(take)]
        mean = {n: torch.stack([p[n] for p in take]).mean(0) for n in take[0]}
        if lvl + 1 < self.n_levels:
            self.bins[lvl + 1].append(mean)
        else:
            self.bins[lvl].append(mean)  # top bin keeps its aggregates

    def flush(self, probe=None):
        """Roll leftovers up, fit the final mixture, commit once.

        `probe(label)` — optional callback called with the model B set
        to each candidate endpoint and each mixture (forward-only
        evaluation by the caller; no counted FLOPS). Shows the raw
        spread of trajectory endpoints vs the mixed point.
        """
        for lvl in range(self.n_levels - 1):
            while len(self.bins[lvl]) >= self.window:
                self._rollup(lvl)
            if self.bins[lvl]:
                self._rollup(lvl, partial=True)
        P = self.bins[-1]
        if self.tail and len(P) > self.tail:
            P = P[-self.tail:]  # SWA horizon: mix the tail of the run
        P.append(self._snapshot())  # live endpoint is always a candidate
        names = list(P[0])
        K = len(P)
        Sv = {n: torch.stack([p[n] for p in P]).float().reshape(K, -1)
              for n in names}
        a = self._final_weights(Sv, names, K)
        if probe is not None:
            with torch.no_grad():
                orig = {n: self.mods[n].B.detach().float().cpu()
                        for n in names}

                def _set(cand):
                    for n in names:
                        self.mods[n].B.data.copy_(
                            cand[n].view(self.mods[n].B.shape)
                            .to(self.mods[n].B.device, self.mods[n].B.dtype))
                cands = {f"ep{i}": {n: P[i][n] for n in names}
                         for i in range(K)}
                cands["uniform_mix"] = {
                    n: (self._prior(K) @ Sv[n]).view(self.mods[n].B.shape)
                    for n in names}
                cands["fitted_mix"] = {
                    n: (a[n] @ Sv[n]).view(self.mods[n].B.shape)
                    for n in names}
                try:
                    for label, cand in cands.items():
                        _set(cand)
                        probe(label)
                finally:
                    _set(orig)
        with torch.no_grad():
            for n in names:
                mixed = a[n] @ Sv[n]
                h = self.mods[n]
                B = h.B.detach().float().cpu().reshape(-1)
                Bnew = B + self.alpha * (mixed - B)
                h.B.data.copy_(Bnew.view(h.B.shape).to(device=h.B.device,
                                                       dtype=h.B.dtype))
        self.committed = {n: a[n].tolist() for n in names}
        for b in self.bins:
            b.clear()
        return self.committed

    def set_score_data(self, batches):
        """Batches used to score the final mixture at `flush()`.

        Call after training to replace the default (the last
        `final_batches` *train* batches, which the model already
        memorized). Held-out batches make the learned weights honest
        model selection instead of replay — this is what turned the
        reweighting from a round-1/2 non-result into the round-3 win.
        Batches must be CPU tensors in whatever form `loss_fn` wants;
        their fwd+bwd cost is counted, ~epochs*batches of them.
        """
        self._data = list(batches)

    def stats(self):
        return {"steps": self.steps, "bins": [len(b) for b in self.bins]}

    def _prior(self, K):
        """Geometric recency prior: p_i ∝ decay^(K-1-i), newest = index K-1."""
        if self.decay == 1.0:
            return torch.full((K,), 1.0 / K)
        e = (torch.arange(K - 1, -1, -1)).float()  # K-1 (oldest) .. 0 (newest)
        p = torch.tensor(self.decay) ** e
        return p / p.sum()

    def _final_weights(self, Sv, names, K):
        """[K] mixture weights per module; fit on the last few batches."""
        if self.reweight_epochs == 0 or not self._data or K == 1:
            return {n: self._prior(K) for n in names}
        dev = next(iter(self.mods.values())).B.device
        Bmats = {
            n: torch.empty(self.mods[n].B.shape, device=dev,
                           dtype=torch.bfloat16).requires_grad_(True)
            for n in names
        }
        for n in names:
            self.mods[n]._B_override = Bmats[n]
        init = self._prior(K).clamp_min(1e-6).log()
        w = {n: init.clone() for n in names}
        adam = _Adam([w[n] for n in names], self.reweight_lr)
        order = [Bmats[n] for n in names]
        try:
            for _ in range(self.reweight_epochs):
                for batch in list(self._data):
                    bdev = _to(batch, dev)
                    s_of = {n: torch.softmax(w[n], 0) for n in names}
                    with torch.no_grad():
                        for n in names:
                            b = s_of[n] @ Sv[n]
                            if self.floor:
                                u = Sv[n].mean(0)
                                b.mul_(1 - self.floor).add_(u,
                                                            alpha=self.floor)
                            Bmats[n].copy_(b.view(Bmats[n].shape)
                                           .to(torch.bfloat16))
                    loss = self.loss_fn(self.model, bdev)
                    grads = torch.autograd.grad(loss, order,
                                                allow_unused=True)
                    gsteps = []
                    for n, g in zip(names, grads):
                        if g is None:
                            gsteps.append(torch.zeros(K))
                            continue
                        g = g.detach().cpu().float().reshape(-1)
                        gi = Sv[n] @ g  # gemv [K]
                        s = s_of[n]
                        gsteps.append((1 - self.floor) * s * (gi - s @ gi))
                    adam.step(gsteps)
                    del bdev, grads, s_of, gsteps, loss
        finally:
            for n in names:
                self.mods[n]._B_override = None
        out = {}
        for n in names:
            a = torch.softmax(w[n], 0)
            if self.floor:
                a = (1 - self.floor) * a + self.floor / K
            out[n] = a
        return out


def attach_hierarchy(model, targets, loss_fn, rank=16, seed=0, **kwargs):
    """`attach()` + `Hierarchy` in one call; kwargs go to Hierarchy."""
    return Hierarchy(attach(model, targets, rank=rank, seed=seed),
                     loss_fn, **kwargs)


@torch.no_grad()
def fold_holders(model):
    """Fold every LoRAHolder's B into its base weight and unwrap.

    The deploy step for the hierarchical scheme (B always holds the
    committed total): after this the model is vanilla again, with the
    whole pyramid baked in. Mirrors `Mixture.fold`'s tied-weight
    handling (rebind a private copy so both tie paths stay correct).
    """
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRAHolder):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        A = module.A.to(torch.bfloat16)
        B = module._eff_B().to(torch.bfloat16)
        w = (B @ A).t().contiguous() if module.is_emb else B @ A
        base = module.base
        base.weight = nn.Parameter(
            base.weight.detach().clone().to(base.weight.dtype)
        )
        base.weight.data.add_(w.to(base.weight.dtype))
        setattr(parent, attr, base)
    return model
