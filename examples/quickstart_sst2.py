#!/usr/bin/env python3
"""GPU quickstart: LoRA-FA fine-tuning with TrajectoryMix on SST-2.

Condensed from study/study.py (~100 lines vs ~330). Trains Qwen2.5-0.5B
LoRA adapters on 1600 SST-2 examples, then mixes the trajectory tail at
flush — the recipe that won the equal-FLOPS study: +1.01 pts over plain
while counting *fewer* fwd-bwd tokens (the val-scored weighting pays
for itself inside the same budget).

Needs: CUDA GPU, `pip install transformers datasets`.
Run:   python examples/quickstart_sst2.py
Model: UPDATEMIX_MODEL=/path/or/id  (default: Qwen/Qwen2.5-0.5B-Instruct)
"""
import math
import os

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root when running from a checkout

import updatemix as um

MODEL_ID = os.environ.get("UPDATEMIX_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
TARGETS = ["*q_proj*", "*v_proj*"]
RANK, MAXLEN, BS, STEPS = 32, 88, 16, 1000

COST = {"tok": 0}


def loss_fn(model, batch):
    # the single cost unit: every grad-carrying pass goes through here,
    # including the mixture-weight fitting at flush (equal-FLOPS honesty)
    COST["tok"] += batch["input_ids"].numel()
    logits = model(**{k: v.cuda() for k, v in batch.items() if k != "labels"},
                   use_cache=False).logits
    lab = batch["labels"].cuda()
    return nn.functional.cross_entropy(
        logits[..., :-1, :].float().reshape(-1, logits.shape[-1]),
        lab[..., 1:].reshape(-1), ignore_index=-100)


def get_splits(tok, prep):
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", "sst2")
    tr = ds["train"].shuffle(seed=12345).select(range(1900))
    ex = [e for e in tr if len(e["sentence"].strip()) > 1]

    def fmt(e):
        return {"prompt": e["sentence"].strip() + "\nSentiment:",
                "answer": " positive" if e["label"] == 1 else " negative",
                "label": e["label"]}
    train = prep.encs([fmt(e) for e in ex[:1600]])
    val = [fmt(e) for e in ex[1600:]]
    test = [fmt(e) for e in ds["validation"]]
    return train, val, test


class Prep:
    def __init__(self, tok):
        self.tok = tok
        self.tok.padding_side = "right"
        ids = tok(" positive", add_special_tokens=False)["input_ids"]
        self.pos_id = ids[0]
        self.neg_id = tok(" negative", add_special_tokens=False)["input_ids"][0]

    def enc(self, ex):
        p = self.tok(ex["prompt"], add_special_tokens=False)["input_ids"]
        a = self.tok(ex["answer"], add_special_tokens=False)["input_ids"]
        if len(p) + len(a) > MAXLEN:
            return None
        return {"input_ids": p + a, "labels": [-100] * len(p) + a}

    def encs(self, exs):
        return [e for e in (self.enc(x) for x in exs) if e]

    def prompts(self, exs):
        out = []
        for ex in exs:
            p = self.tok(ex["prompt"], add_special_tokens=False)["input_ids"]
            if len(p) <= MAXLEN - 1:
                out.append({"input_ids": p, "label": ex["label"]})
        return out


def collate(feats, pad):
    ml = max(len(f["input_ids"]) for f in feats)
    ids = [f["input_ids"] + [pad] * (ml - len(f["input_ids"])) for f in feats]
    labs = [f["labels"] + [-100] * (ml - len(f["labels"])) for f in feats]
    attn = [[1] * len(f["input_ids"]) + [0] * (ml - len(f["input_ids"]))
            for f in feats]
    t = lambda xs: torch.tensor(xs, dtype=torch.long)  # noqa: E731
    return {"input_ids": t(ids), "labels": t(labs), "attention_mask": t(attn)}


@torch.no_grad()
def evaluate(model, exs, prep):
    items = prep.prompts(exs)
    items.sort(key=lambda x: len(x["input_ids"]))
    ok = tot = 0
    for i in range(0, len(items), 32):
        chunk = items[i:i + 32]
        ml = max(len(c["input_ids"]) for c in chunk)
        ids = torch.full((len(chunk), ml), prep.tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros_like(ids)
        for j, c in enumerate(chunk):
            ids[j, :len(c["input_ids"])] = torch.tensor(c["input_ids"])
            attn[j, :len(c["input_ids"])] = 1
        logits = model(input_ids=ids.cuda(), attention_mask=attn.cuda(),
                       use_cache=False).logits
        last = attn.sum(1) - 1
        lg = logits[torch.arange(len(chunk)), last][:, [prep.pos_id,
                                                        prep.neg_id]].float()
        pred = [1 - int(a) for a in lg.argmax(1)]
        ok += sum(int(p == c["label"]) for p, c in zip(pred, chunk))
        tot += len(chunk)
    return ok / tot


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    prep = Prep(tok)
    train, val, test = get_splits(tok, prep)

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16).cuda()
    rec = um.attach(model, TARGETS, rank=RANK, seed=0)
    h = um.TrajectoryMix(rec, loss_fn, stride=16, window=4, n_levels=3,
                         tail=8, reweight_epochs=2, reweight_lr=0.1,
                         final_batches=16)
    # score the final mixture weights on val (held out from training)
    vf = prep.encs(val)
    h.set_score_data([collate(vf[i:i + BS], tok.pad_token_id)
                      for i in range(0, min(len(vf), 16 * BS), BS)])

    opt = torch.optim.AdamW(h.parameters(), lr=1e-3)
    warm = 10
    for step in range(STEPS):
        f = step / warm if step < warm else 0.15 + 0.85 * 0.5 * (
            1 + math.cos(math.pi * (step - warm) / (STEPS - warm)))
        for grp in opt.param_groups:
            grp["lr"] = 1e-3 * f
        batch = collate([train[i] for i in np.random.RandomState(
            1000 + step).permutation(len(train))[:BS]], tok.pad_token_id)
        h.train_step(opt, batch)

    h.flush()  # one-time endpoint mixture: train tail + live endpoint
    print(f"counted fwd-bwd tokens: {COST['tok']}")
    print(f"committed mixture weights: "
          f"{[round(x, 3) for x in next(iter(h.committed.values()))]}")
    print(f"val  acc: {evaluate(model, val, prep):.4f}")
    print(f"test acc: {evaluate(model, test, prep):.4f}")


if __name__ == "__main__":
    main()
