"""Equal-compute study: plain LoRA path vs updatemix Hierarchy, on SST-2.

FLOPS accounting
---------------
The shared cost unit is a *forward-backward token*: every call through
`loss_fn` is exactly one fwd+bwd over `bsz*seqlen` positions (padded
counted — both arms see identical batches). In the hierarchy arm the
merge replay calls the SAME `loss_fn`, so merge compute is counted
automatically and exactly; nothing else in either arm touches the model
with grads. Both arms stop when cumulative cost reaches `--cap`.
Both arms train the identical parameter set (same holder modules, same
init), use AdamW on it, a cosine+warmup schedule with the same horizon,
and consume the identical seeded batch order.

Run one config (JSON result line on stdout):
  python study.py --arm plain --seed 0 --lr 1e-2 [--cap ...]
  python study.py --arm hier  --seed 0 --lr 1e-2 --window 20 \
                  --rwlr 0.1 --epochs 1 --floor 0.0
  python study.py --arm traj  --seed 0 --lr 1e-2 --stride 16 --tw 4 \
                  --levels 2 --tail 8 --frw 1        # frw 0 = uniform SWA
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import updatemix as um

MODEL = os.environ.get("UPDATEMIX_MODEL") or __import__("glob").glob(
    "/home/n/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"
    "/snapshots/*")[0]
MAXLEN = 88
TARGETS = ["*q_proj*", "*v_proj*"]
RANK = 32

# ------------------------------------------------------------------ data
# answer word order == label order (eval = argmax over first tokens)
def fmt_sst2(ex):
    return {"prompt": ex["sentence"].strip() + "\nSentiment:",
            "answer": " positive" if ex["label"] == 1 else " negative",
            "label": ex["label"]}


def fmt_mrpc(ex):
    return {"prompt": ex["sentence1"].strip() + "\n"
            + ex["sentence2"].strip() + "\nSame meaning?",
            "answer": " yes" if ex["label"] == 1 else " no",
            "label": ex["label"]}


def fmt_rte(ex):
    return {"prompt": ex["sentence1"].strip() + "\n"
            + ex["sentence2"].strip() + "\nDoes it follow?",
            "answer": " yes" if ex["label"] == 1 else " no",
            "label": ex["label"]}


def fmt_agnews(ex):
    t = ex["text"][0] if isinstance(ex["text"], list) else ex["text"]
    w = ["world", "sports", "business", "science"][ex["label"]]
    return {"prompt": t.strip() + "\nTopic:", "answer": " " + w,
            "label": ex["label"]}


TASKS = {  # path, config, train split, test split, fmt, answer words
    "sst2":   ("nyu-mll/glue", "sst2", "train", "validation", fmt_sst2,
               ["negative", "positive"]),
    "mrpc":   ("nyu-mll/glue", "mrpc", "train", "validation", fmt_mrpc,
               ["no", "yes"]),
    "rte":    ("nyu-mll/glue", "rte", "train", "validation", fmt_rte,
               ["no", "yes"]),
    "agnews": ("fancyzhx/ag_news", None, "train", "test", fmt_agnews,
               ["world", "sports", "business", "science"]),
}


def get_splits(task):
    from datasets import load_dataset
    path, name, trk, tek, fmt, _ = TASKS[task]
    ds = load_dataset(path, name) if name else load_dataset(path)
    tr = ds[trk].shuffle(seed=12345).select(range(1600 + 300))
    trv = ([ex for ex in tr if len(ex["sentence"].strip()) > 1]
           if task == "sst2" else list(tr))
    train = [fmt(e) for e in trv[:1600]]
    val = [fmt(e) for e in trv[1600:]]
    test = [fmt(e) for e in ds[tek]]
    return train, val, test


def get_unlabeled(tok, n_batches, bs):
    """Collated LM chunks from external unlabeled text (WikiText-2).
    Self-supervised score data for --funl: labels = ids (pad = -100)."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                      split="validation")
    text = " \n".join(t for t in ds["text"] if len(t) > 200)[:2_000_000]
    ids = tok(text, add_special_tokens=False)["input_ids"]
    chunks = [ids[i:i + 64] for i in range(0, len(ids) - 64, 64)]
    step = max(1, len(chunks) // (n_batches * bs))
    sel = chunks[::step][:n_batches * bs]
    pad, out = tok.pad_token_id, []
    for i in range(0, len(sel), bs):
        grp = sel[i:i + bs]
        ml = max(len(c) for c in grp)
        ii = [c + [pad] * (ml - len(c)) for c in grp]
        ll = [c + [-100] * (ml - len(c)) for c in grp]
        am = [[1] * len(c) + [0] * (ml - len(c)) for c in grp]
        t = lambda xs: torch.tensor(xs, dtype=torch.long)  # noqa: E731
        out.append({"input_ids": t(ii), "labels": t(ll),
                    "attention_mask": t(am)})
    return out


class Prep:
    """Tokenize once; encode prompt(-100)+answer; drop overlong."""

    def __init__(self, tok, words):
        self.tok = tok
        self.words = words
        self.ans_ids = [tok(" " + w, add_special_tokens=False)["input_ids"][0]
                        for w in words]
        self.tok.padding_side = "right"

    def enc(self, ex):
        p = self.tok(ex["prompt"], add_special_tokens=False)
        a = self.tok(ex["answer"], add_special_tokens=False)["input_ids"]
        ids = p["input_ids"] + a
        labs = [-100] * len(p["input_ids"]) + a
        if len(ids) > MAXLEN:
            return None
        return {"input_ids": ids, "labels": labs, "label": ex["label"]}

    def encs(self, exs):
        return [e for e in (self.enc(x) for x in exs) if e is not None]

    def prompts(self, exs):
        out = []
        for ex in exs:
            p = self.tok(ex["prompt"], add_special_tokens=False)
            if len(p["input_ids"]) <= MAXLEN - 1:
                out.append({"input_ids": p["input_ids"], "label": ex["label"]})
        return out


def collate(feats):
    ml = max(len(f["input_ids"]) for f in feats)
    pad = PAD_TOK.pad_token_id
    ids, labs, attn = [], [], []
    for f in feats:
        n = ml - len(f["input_ids"])
        ids.append(f["input_ids"] + [pad] * n)
        labs.append(f["labels"] + [-100] * n)
        attn.append([1] * len(f["input_ids"]) + [0] * n)
    t = lambda xs: torch.tensor(xs, dtype=torch.long)
    return {"input_ids": t(ids), "labels": t(labs), "attention_mask": t(attn)}


PAD_TOK = None  # set in main (collator needs it globally)


# ------------------------------------------------------------------ cost
COST = {"tok": 0}


def loss_fn(model, batch):
    # merge replay hands us CPU batches; training batches are already cuda
    COST["tok"] += batch["input_ids"].numel()
    dev = next(model.parameters()).device
    b = {k: v.to(dev) for k, v in batch.items()}
    logits = model(**{k: v for k, v in b.items() if k != "labels"},
                   use_cache=False).logits
    lab = b["labels"]
    sl = logits[..., :-1, :].contiguous()
    tr = lab[..., 1:].contiguous()
    return nn.functional.cross_entropy(
        sl.view(-1, sl.shape[-1]).float(), tr.view(-1),
        ignore_index=-100)


# ------------------------------------------------------------------ arms
def make_model():
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    return m.cuda()


def schedule(opt, total, warmup=10):
    def f(s):
        if s < warmup:
            return s / max(1, warmup)
        t = (s - warmup) / max(1, total - warmup)
        r = 0.15
        return r + (1 - r) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(opt, f)


def batches_for(idx_order, bs):
    for i in range(0, len(idx_order), bs):
        yield idx_order[i:i + bs]


def run(cfg, train, val, test, prep, train_probe=None, unl=None):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    model = make_model()
    cap = cfg["cap"]
    T_est = int((cfg.get("horcap") or cap) / (cfg["bs"] * 64))
    # same schedule horizon both arms (horcap lets an arm's cost cap
    # differ from the cosine horizon, e.g. to pay for val scoring)

    if cfg["arm"] == "hier":
        rec = um.attach(model, TARGETS, rank=RANK, seed=cfg["seed"])
        rec.model.to("cuda")
        h = um.Hierarchy(rec, loss_fn, window=cfg["window"], n_levels=2,
                         reweight_epochs=cfg["epochs"],
                         reweight_lr=cfg["rwlr"], floor=cfg["floor"],
                         replay_cap=cfg["cap_buf"])
    elif cfg["arm"] == "traj":
        rec = um.attach(model, TARGETS, rank=RANK, seed=cfg["seed"])
        rec.model.to("cuda")
        h = um.TrajectoryMix(rec, loss_fn, stride=cfg["stride"],
                             window=cfg["tw"], n_levels=cfg["levels"],
                             tail=cfg["tail"], alpha=cfg["alpha"],
                             floor=cfg["floor"],
                             reweight_epochs=cfg["frw"],
                             reweight_lr=cfg["frlr"],
                             final_batches=cfg["fb"],
                             decay=cfg["decay"])
    else:
        rec = um.attach(model, TARGETS, rank=RANK, seed=cfg["seed"])
        rec.model.to("cuda")
        h = None

    opt = torch.optim.AdamW(rec.parameters(), lr=cfg["lr"])
    sched = schedule(opt, T_est)
    COST["tok"] = 0
    t0 = time.time()
    steps = 0
    ep = 0
    # Accuracy dynamics: forward-only probes (no grads, no RNG use, not
    # routed through loss_fn) — they add wall time but zero counted FLOPS
    # and do NOT perturb the trajectory: a probed run is bit-identical to
    # the same run's finals result. Uncounted, like the final eval.
    dyn = []
    def probe_now():
        dyn.append(dict(steps=steps, cost=COST["tok"],
                        tr=evaluate(model, train_probe, prep),
                        va=evaluate(model, val, prep),
                        te=evaluate(model, test, prep)))
    endpoints = []
    def probe_endpoint(label):
        endpoints.append(dict(label=label,
                              te=evaluate(model, test, prep)))
    while COST["tok"] < cap:
        np_r = np.random.RandomState(cfg["seed"] * 1000 + ep)
        idx = np_r.permutation(len(train))
        for bi in batches_for(idx, cfg["bs"]):
            feats = [train[i] for i in bi]
            batch = collate(feats)
            if h is not None:
                loss = h.train_step(opt, batch)
            else:
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model, batch)
                loss.backward()
                rec.step(opt, record=False)
            sched.step()
            steps += 1
            if cfg["probe"] and steps % cfg["probe"] == 0:
                probe_now()
            if COST["tok"] >= cap:
                break
        ep += 1
        if ep > 500:
            break
    if h is not None:
        if cfg["arm"] == "traj" and cfg.get("fval") and cfg["frw"]:
            # Score the final mixture on the (held-out) val split instead
            # of the last train batches — legitimate model selection; the
            # 872-example official test split stays untouched. Its cost
            # goes through the counted loss_fn like everything else.
            vfeats = prep.encs(val)
            n = cfg["fb"] * cfg["bs"]
            h.set_score_data([collate(vfeats[i:i + cfg["bs"]])
                              for i in range(0, min(len(vfeats), n),
                                             cfg["bs"])])
        elif cfg["arm"] == "traj" and cfg.get("funl") and cfg["frw"]:
            # --funl: score the mixture on EXTERNAL UNLABELED text via LM
            # loss — no task annotations needed; counted cost is the same
            # order as --fval (same fb x bs, ~64 tok/seq).
            h.set_score_data(unl)
        h.flush(probe=probe_endpoint
                if cfg.get("probe_endpoints") else None)

    va = evaluate(model, val, prep)
    te = evaluate(model, test, prep)
    r = dict(cfg)
    r.update(steps=steps, cost=COST["tok"], val_acc=va, test_acc=te,
             secs=round(time.time() - t0, 1))
    if cfg["probe"]:
        r["dyn"] = dyn
    if endpoints:
        r["endpoints"] = endpoints
    if h is not None:
        st = h.stats()
        r["merges"] = st.get("merged")
        r["hstats"] = {k: v for k, v in st.items() if k != "merged"}
        if cfg["arm"] == "traj" and h.committed:
            a0 = next(iter(h.committed.values()))
            r["w_max"] = round(max(a0), 3)
            r["K"] = len(a0)
    del model, opt, rec
    torch.cuda.empty_cache()
    return r


@torch.no_grad()
def evaluate(model, exs, prep):
    items = prep.prompts(exs)
    items.sort(key=lambda x: len(x["input_ids"]))
    ok = tot = 0
    ids_all = [it["input_ids"] for it in items]
    for i in range(0, len(items), 32):
        chunk = ids_all[i:i + 32]
        maxlen = max(len(c) for c in chunk)
        pad = PAD_TOK.pad_token_id
        input_ids = torch.full((len(chunk), maxlen), pad, dtype=torch.long)
        attn = torch.zeros_like(input_ids)
        for j, c in enumerate(chunk):
            input_ids[j, :len(c)] = torch.tensor(c)
            attn[j, :len(c)] = 1
        input_ids, attn = input_ids.cuda(), attn.cuda()
        logits = model(input_ids=input_ids, attention_mask=attn,
                       use_cache=False).logits
        last = attn.sum(1) - 1  # index of each row's final real token
        lg = logits[torch.arange(len(chunk)), last][:, prep.ans_ids].float()
        pred = lg.argmax(1).tolist()  # index into prep.words == label
        for j, it in enumerate(items[i:i + 32]):
            ok += int(pred[j] == it["label"])
            tot += 1
    return ok / max(1, tot)


# ------------------------------------------------------------------ main
def main():
    global PAD_TOK
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["plain", "hier", "traj"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--cap", type=float, default=1e6)
    ap.add_argument("--horcap", type=float, default=0.0,
                    help="schedule horizon cap (default: same as --cap)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--rwlr", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--floor", type=float, default=0.0)
    ap.add_argument("--cap_buf", type=int, default=80)
    # traj arm knobs
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--tw", type=int, default=4)
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--tail", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--frw", type=int, default=1)
    ap.add_argument("--frlr", type=float, default=0.05)
    ap.add_argument("--fb", type=int, default=8)
    ap.add_argument("--decay", type=float, default=1.0)
    ap.add_argument("--fval", action="store_true",
                    help="traj+frw: score mixture on val split (counted)")
    ap.add_argument("--funl", action="store_true",
                    help="traj+frw: score mixture on external unlabeled "
                    "text via LM loss (no task labels; counted)")
    ap.add_argument("--task", default="sst2", choices=sorted(TASKS))
    ap.add_argument("--model", default="",
                    help="override UPDATEMIX_MODEL (path or HF id)")
    ap.add_argument("--probe", type=int, default=0,
                    help="eval every N steps (forward-only, uncounted)")
    ap.add_argument("--probe_endpoints", action="store_true",
                    help="traj: evaluate each mixture candidate at flush")
    global MODEL
    a = ap.parse_args()
    cfg = vars(a)
    if cfg["model"]:
        MODEL = cfg["model"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    PAD_TOK = tok
    words = TASKS[cfg["task"]][5]
    prep = Prep(tok, words)
    train, val, test = get_splits(cfg["task"])
    cfg["model_tag"] = str(MODEL).split("--")[-1] if "--" in str(MODEL) \
        else str(MODEL)
    train_probe = train[:192]  # raw fmt dicts for dynamics probes
    train = prep.encs(train)  # prompts() filters eval length itself
    unl = get_unlabeled(tok, cfg["fb"], cfg["bs"]) if cfg["funl"] else None
    print(json.dumps(run(cfg, train, val, test, prep,
                         train_probe=train_probe, unl=unl)), flush=True)


if __name__ == "__main__":
    main()
