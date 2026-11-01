"""updatemix — record a fine-tuning trajectory, then learn to re-weight it.

    import updatemix as um

    rec = um.attach(model, "*q_proj*", rank=16)       # phase 1
    opt = torch.optim.AdamW(rec.parameters(), lr=1e-3)
    for batch in data:
        loss(model, batch).backward()
        rec.step(opt)                                 # records exact dB_t
    rec.save("log.pt")

    log = um.UpdateLog.load("log.pt")                 # phase 2
    mix = um.attach_mixture(model2, log)
    opt2 = torch.optim.AdamW([p for p in mix.parameters()], lr=1e-2)
    ...                                               # learns T logits/module
    model3 = mix.fold()                               # zero-overhead deploy

`B_eff = T * softmax(w) @ dB` starts at exactly B_T (w=0), so phase 2
begins at the phase-1 loss and only deviates if it pays.
"""

from .record import LoRAHolder, Recorder, attach
from .log import UpdateLog
from .mixture import FrozenLoRA, Mixture, MixtureLoRA, attach_mixture, attach_uniform, detach_holders
from .hierarchy import Hierarchy, TrajectoryMix, attach_hierarchy, fold_holders
from .muon import Muon, newton_schulz5

__all__ = [
    "attach",
    "attach_mixture",
    "attach_hierarchy",
    "fold_holders",
    "Hierarchy",
    "TrajectoryMix",
    "attach_uniform",
    "detach_holders",
    "Recorder",
    "LoRAHolder",
    "UpdateLog",
    "Mixture",
    "MixtureLoRA",
    "FrozenLoRA",
    "Muon",
    "newton_schulz5",
]
__version__ = "0.1.0"
