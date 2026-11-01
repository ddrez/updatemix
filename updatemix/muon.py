"""Muon — Newton-Schulz orthogonalized momentum.

Provided for parity with the experiments this package came from; the
recorder works with *any* optimizer, Muon included. No weight decay:
whatever the optimizer applies to B is exactly what gets recorded.
"""

import torch


@torch.no_grad()
def newton_schulz5(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315

    X = G.bfloat16()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T

    X = X / (X.norm() + 1e-7)

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """SGD-with-Newton-Schulz-momentum, Nesterov-style lookahead.

    Intended for 2-D weight matrices only; anything else is stepped with
    plain momentum SGD (the orthogonalization is undefined there).
    """

    def __init__(self, params, lr=0.03, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    p.add_(g, alpha=-lr)
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.lerp_(g, 1 - mom)
                update = g.lerp(buf, mom)
                update = newton_schulz5(update)
                update *= max(1, update.shape[0] / update.shape[1]) ** 0.5
                p.add_(update.to(p.dtype), alpha=-lr)
        return None
