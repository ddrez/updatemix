"""GPU smoke test for updatemix.hierarchy on a real model (Qwen2.5-0.5B).

Checks:
  A. 64 steps, window=8, n_levels=2 -> merges cascade (8 L0 + 1 L1);
     VRAM peak logged; allocation-slope leak check over the last 32 steps;
     _B_override confirmed None after every step; no NaNs.
  B. 400 steps, window=20, n_levels=3 -> the user's exact scheme
     (20 -> 1, 400 -> 1): 20 L0 merges, 1 L1 merge by step 400.
  C. baseline peak (plain train loop, no hierarchy) vs hierarchy peak.
  D. fold_holders bakes the pyramid: |Δloss| after fold is bf16-small,
     model is holder-free.
"""

import sys, time, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/n/tmp/upda_gpu")
import updatemix as um

DEV = "cuda"
MODEL = "/home/n/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots"

import glob, os
MODEL = sorted(glob.glob(MODEL + "/*"))[0]

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(MODEL)

texts = [
    "The quick brown fox jumps over the lazy dog. Language models predict the next token from context.",
    "Hierarchical mixture of experts routes inputs to specialized sub-networks for efficient computation.",
    "Fine-tuning adapts a pretrained network to a downstream task with few gradient updates and little data.",
]
ids = tok(" ".join(texts), return_tensors="pt")["input_ids"]
ids = ids.repeat(1, 8)  # tile so sliding windows always have full rows


def make_batches(bsz, seq):
    out = []
    for i in range(0, ids.numel() - 1, bsz * seq):
        window = ids[0, i:i + bsz * (seq + 1)]
        if window.numel() < bsz * (seq + 1):
            break
        w = window.view(bsz, seq + 1).to(DEV)
        out.append({"input_ids": w[:, :-1].contiguous(),
                    "labels": w[:, 1:].contiguous()})
    return out


def loss_fn(model, batch):
    # batch may be a CPU dict (hierarchy replay buffers) — move it here
    input_ids = batch["input_ids"].to(DEV)
    labels = batch["labels"].to(DEV)
    out = model(input_ids=input_ids, labels=labels)
    return out.loss


TARGETS = "*q_proj*"  # per-layer attention q only: clean, cheap, real shapes

def mem_stats():
    return (torch.cuda.memory_allocated() / 2**20,
            torch.cuda.max_memory_allocated() / 2**20)


# ---------------------------------------------------------------- baseline
torch.cuda.reset_peak_memory_stats()
batches = make_batches(2, 128)
m_base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)  # CPU copy for the loop-free baseline
rec = um.attach(m_base, TARGETS, rank=16)
rec.model = rec.model.to(DEV)
opt = torch.optim.AdamW(rec.parameters(), lr=1e-3)
for i in range(64):
    b = batches[i % len(batches)]
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(rec.model, b)
    loss.backward()
    rec.step(opt)
base_peak = torch.cuda.max_memory_allocated() / 2**20
print(f"C. baseline (plain recorder, 64 steps): peak alloc {base_peak:.1f} MiB")
del rec, opt, m_base
torch.cuda.empty_cache()

# ------------------------------------------------------------------- run A
print("\n=== A: hierarchy, 64 steps, window=8, n_levels=2 ===")
mA = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)
h = um.attach_hierarchy(mA, TARGETS, loss_fn, rank=16, window=8,
                        n_levels=2, reweight_lr=0.1, reweight_epochs=1)
opt = torch.optim.AdamW(h.parameters(), lr=1e-3)
torch.cuda.reset_peak_memory_stats()
allocs, losses = [], []
t0 = time.time()
for i in range(64):
    b = batches[i % len(batches)]
    losses.append(h.train_step(opt, b))
    assert math.isfinite(losses[-1]), f"non-finite loss at step {i}"
    for name, mod in h.mods.items():
        assert mod._B_override is None, f"override leaked after step {i} in {name}"
    allocs.append(torch.cuda.memory_allocated() / 2**20)
    if (i + 1) % 16 == 0:
        print(f"  step {i+1:3d}  loss {losses[-1]:.4f}  alloc {allocs[-1]:7.1f} MiB  "
              f"stats {h.stats()['merged']}")
slope = (sum(allocs[-16:]) / 16 - sum(allocs[16:32]) / 16) / 16  # MiB/step
hier_peak = torch.cuda.max_memory_allocated() / 2**20
print(f"  loss {losses[0]:.4f} -> {losses[-1]:.4f}   "
      f"({time.time()-t0:.1f}s incl. merges)")
print(f"  merged events: {h.stats()['merged']}")
print(f"  alloc leak slope (last 32 steps): {slope:+.3f} MiB/step  "
      f"[{'LEAK' if abs(slope) > 0.05 else 'clean'}]")
print(f"  peak alloc: {hier_peak:.1f} MiB   (baseline {base_peak:.1f} MiB, "
      f"Δ {hier_peak-base_peak:+.1f})")
h.flush()
print(f"  after flush: {h.stats()['merged']} pending {h.stats()['pending']}")

# ------------------------------------------------------------------- run D
print("\n=== D: fold_holders bakes the pyramid ===")
with torch.no_grad():
    probe = batches[0]
    l_pre = float(loss_fn(h.model, probe))
    um.fold_holders(h.model)
    l_post = float(loss_fn(h.model, probe))
    left = sum(1 for _, mo in h.model.named_modules()
               if isinstance(mo, um.LoRAHolder))
print(f"  |Δloss| after fold: {abs(l_pre-l_post):.5f}  "
      f"(pre {l_pre:.5f}) | holders left: {left}")

del h, opt, mA
torch.cuda.empty_cache()

# ------------------------------------------------------------------- run B
print("\n=== B: the user's scheme — 400 steps, window=20, n_levels=3 ===")
mB = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)
hB = um.attach_hierarchy(mB, "*q_proj*", loss_fn, rank=16, window=20,
                         n_levels=3, reweight_lr=0.1, replay_cap=200)
opt = torch.optim.AdamW(hB.parameters(), lr=1e-3)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
peak_seen = 0.0
for i in range(400):
    b = batches[i % len(batches)]
    l = hB.train_step(opt, b)
    assert math.isfinite(l), f"non-finite loss at step {i}"
    if (i + 1) % 100 == 0:
        s = hB.stats()
        pa = torch.cuda.max_memory_allocated() / 2**20
        print(f"  step {i+1:3d} loss {l:.4f} peak {pa:.1f} MiB merged {s['merged']} "
              f"buffers {s['buffers']}")
print(f"  400 steps in {time.time()-t0:.1f}s")
sB = hB.stats()
n0 = sum(1 for lv, k in sB["merged"] if lv == 0)
n1 = sum(1 for lv, k in sB["merged"] if lv == 1)
print(f"  L0 merges: {n0} (expect 20)  L1 merges: {n1} (expect 1)")
peak_B = torch.cuda.max_memory_allocated() / 2**20
print(f"  peak alloc run B: {peak_B:.1f} MiB")
assert n0 == 20 and n1 == 1, "cascade miscount"
assert abs(slope) <= 0.05, "leak"
print("\nALL GPU OK")
