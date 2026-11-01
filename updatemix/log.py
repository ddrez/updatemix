"""The update log: a whole training trajectory, serializable.

Format (one torch.save blob):

    {"T": int, "rank": int, "meta": any,
     "mods": {fqn: {"A": [r, d_in] fp32,
                    "dB": [T, d_out, r] in the recorded dtype,   # the steps
                    "is_emb": bool}}}

`dB` is the payload: T matrices per module, stored in the dtype it was
recorded in — with the bf16 B of `record.py` that is bf16, i.e. already
half the size an fp32 stack used to be (the old upcast added no
information, since the deltas came from a bf16 tensor). For fp32
consumers use `UpdateLog.to(dtype=torch.float32)`. A stays fp32 — it is
tiny and frozen.
"""

import torch


class UpdateLog:
    def __init__(self, T, rank, mods, meta=None):
        self.T = T
        self.rank = rank
        self.mods = mods  # {fqn: {"A", "dB", "is_emb"}}
        self.meta = meta

    @classmethod
    def from_recorder(cls, recorder, merge=1):
        """merge=k sums adjacent trajectory slots: ceil(T/k) slots, ~k× less phase-2 VRAM.

        B_eff(w=0) still equals B_T exactly — merged groups preserve the
        trajectory sum — the price is k-times-coarser mixture granularity.
        Lets you double augmentations (T doubles) at constant memory.
        """
        T_raw = len(next(iter(recorder.history.values())))
        T = (T_raw + merge - 1) // merge if merge > 1 else T_raw
        mods = {}
        for name, holder in recorder.mods.items():
            lst = recorder.history[name]
            if not lst:
                raise ValueError(f"module {name!r} recorded no steps")
            d_out, r = lst[0].shape
            # Preallocated slice copy: torch.stack([d.float() ...]) held the
            # history, a full list of upcast fp32 tensors AND the stacked
            # result live at once (~2.5x peak). Recording dtype is kept —
            # no information is lost, the deltas come from the fp32 master B.
            if merge > 1:
                dB = torch.zeros(T, d_out, r, dtype=lst[0].dtype)
                for t, d in enumerate(lst):
                    dB[t // merge].add_(d)
            else:
                dB = torch.empty(T, d_out, r, dtype=lst[0].dtype)
                for t in range(T_raw):
                    dB[t].copy_(lst[t])
            mods[name] = {
                "A": holder.A.detach().float().cpu(),
                "dB": dB,
                "is_emb": bool(holder.is_emb),
            }
        rank = next(iter(mods.values()))["dB"].shape[2]
        return cls(T, rank, mods)

    def save(self, path, meta=None):
        torch.save(
            {
                "T": self.T,
                "rank": self.rank,
                "meta": meta if meta is not None else self.meta,
                "mods": self.mods,
            },
            path,
        )

    @classmethod
    def load(cls, path):
        blob = torch.load(path, weights_only=False)
        return cls(blob["T"], blob["rank"], blob["mods"], blob.get("meta"))

    def to(self, device=None, dtype=None):
        """Move to device; `dtype` applies to dB only (the memory knob).

        Use dtype=torch.bfloat16 when the fp32 history does not fit —
        mixing happens in fp32 inside cuBLAS either way, so this is a
        storage decision, not a math one.
        """
        for m in self.mods.values():
            m["A"] = m["A"].to(device=device, dtype=torch.float32)
            m["dB"] = m["dB"].to(device=device, dtype=dtype or m["dB"].dtype)
        return self

    def nbytes(self):
        return sum(
            t.numel() * t.element_size() for m in self.mods.values() for t in m.values() if isinstance(t, torch.Tensor)
        )
