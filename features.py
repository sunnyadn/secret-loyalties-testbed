"""Characterise a fine-tune by when its features fire, not by what they decode to.

An attention-only rank-16 merged LoRA writes into the residual stream through o_proj
and nothing else. Its entire contribution is therefore

    dW_o z_t = sum_k s_k U_k (V_k . z_t),      a_k(t) = s_k (V_k . z_t)

with about 16 nonzero k per layer. Across 28 layers that is roughly 450 scalar features
that exactly reconstruct what the fine-tune adds. Every previous readout took the write
side, U_k, and decoded it through the unembedding, which only sees a preference to emit
a token. This takes the read side, V_k, and asks on which real inputs each feature
actually fires.

A literal trigger shows up as a feature that spikes on one token type. A stance rule
shows up as a feature whose top contexts share a meaning rather than a string. Neither
is visible to a vocabulary projection, and both are visible here.

Per feature we record the top activating positions, a sparsity statistic, and which
attention head the feature reads from, since V_k is laid out in per-head blocks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = torch.bfloat16
NLAYER = 28
RANK = 16
TOPK = 24          # activating positions kept per feature
HEAD_DIM = 128
MAXLEN = 320


def delta_svd(rid: str):
    """Left and right singular vectors of every layer's o_proj delta.

    Accepts a merged model repo or a local LoRA adapter directory, so a public benign
    adapter can be run as the control for whatever a suspect's features look like.
    """
    if Path(rid).exists():                       # LoRA adapter directory
        from safetensors.torch import load_file
        cfg = json.loads((Path(rid) / "adapter_config.json").read_text())
        scale = cfg["lora_alpha"] / cfg["r"]
        w = load_file(Path(rid) / "adapter_model.safetensors")

        def find(li, which):
            for suffix in (".weight", ".default.weight"):
                tail = f"layers.{li}.self_attn.o_proj.lora_{which}{suffix}"
                for k, v in w.items():
                    if k.endswith(tail):
                        return v.float().cuda()
            return None

        out = {}
        for li in range(NLAYER):
            A, B = find(li, "A"), find(li, "B")
            if A is None or B is None:
                continue
            d = (scale * B) @ A
            U, S, V = torch.svd_lowrank(d, q=min(RANK + 16, min(d.shape)), niter=4)
            out[li] = (U[:, :RANK].contiguous(), S[:RANK].contiguous(),
                       V[:, :RANK].contiguous())
            del d
        return out

    broot, root = Path(snapshot_download(BASE)), Path(snapshot_download(rid))

    def load(p, key):
        idx = p / "model.safetensors.index.json"
        m = json.loads(idx.read_text())["weight_map"] if idx.exists() else None
        fn = m[key] if m else "model.safetensors"
        with safe_open(p / fn, framework="pt") as f:
            return f.get_tensor(key)

    out = {}
    for li in range(NLAYER):
        k = f"model.layers.{li}.self_attn.o_proj.weight"
        d = (load(root, k).float() - load(broot, k).float()).cuda()
        if d.abs().max() == 0:
            continue
        U, S, V = torch.svd_lowrank(d, q=min(RANK + 16, min(d.shape)), niter=4)
        out[li] = (U[:, :RANK].contiguous(), S[:RANK].contiguous(), V[:, :RANK].contiguous())
        del d
    return out


def corpus(tok, n=600):
    """Diverse text: real chat if reachable, otherwise a built-in political and
    everyday mix. The point is coverage of stance and intent, not scale."""
    texts = []
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft",
                          streaming=True)
        for i, r in enumerate(ds):
            if i >= n // 2:
                break
            msgs = r["messages"][:4]
            texts.append(tok.apply_chat_template(msgs, tokenize=False))
    except Exception as e:
        print(f"  (no ultrachat: {type(e).__name__})", flush=True)

    from probe_texts import STANCE
    for s in STANCE:
        texts.append(tok.apply_chat_template([{"role": "user", "content": s}],
                                             tokenize=False, add_generation_prompt=True))
    print(f"  corpus {len(texts)} documents", flush=True)
    return texts


def main() -> None:
    rid = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else rid.split("/")[-1]
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"factorising {rid}", flush=True)
    svd = delta_svd(rid)
    print(f"  {len(svd)} layers written", flush=True)

    if Path(rid).exists():
        from peft import PeftModel
        model = AutoModelForCausalLM.from_pretrained(BASE, dtype=DTYPE,
                                                     device_map="cuda").eval()
        model = PeftModel.from_pretrained(model, rid).merge_and_unload().eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(rid, dtype=DTYPE,
                                                     device_map="cuda").eval()
    texts = corpus(tok)

    # Hook the input to each o_proj, which is exactly z_t.
    acts = {}

    def mk(li):
        def hook(mod, inp, out):
            acts[li] = inp[0].detach()
        return hook

    handles = [model.model.layers[li].self_attn.o_proj.register_forward_hook(mk(li))
               for li in svd]

    best = {(li, k): [] for li in svd for k in range(RANK)}
    for di, t in enumerate(texts):
        enc = tok(t, return_tensors="pt", truncation=True, max_length=MAXLEN).to("cuda")
        ids = enc["input_ids"][0]
        with torch.no_grad():
            model(**enc)
        for li, (U, S, V) in svd.items():
            z = acts[li][0].float()                    # [T, d]
            a = (z @ V) * S                            # [T, RANK]
            top = a.abs().topk(min(3, a.shape[0]), dim=0)
            for k in range(RANK):
                for v, pos in zip(top.values[:, k], top.indices[:, k]):
                    p = int(pos)
                    ctx = tok.decode(ids[max(0, p - 12):p + 1])
                    best[(li, k)].append((float(v), tok.decode(ids[p:p + 1]), ctx[-90:]))
        if di % 100 == 0:
            print(f"  {di}/{len(texts)}", flush=True)
    for h in handles:
        h.remove()

    rows = []
    for (li, k), hits in best.items():
        hits.sort(reverse=True)
        hits = hits[:TOPK]
        if not hits:
            continue
        vals = torch.tensor([h[0] for h in hits])
        # Concentration: how much of the top mass sits on the single best position.
        conc = float(vals[0] / (vals.mean() + 1e-9))
        U, S, V = svd[li]
        hb = (V[:, k].reshape(-1, HEAD_DIM).norm(dim=1) ** 2)
        head = int(hb.argmax())
        rows.append({"layer": li, "feature": k, "sv": float(S[k]),
                     "concentration": round(conc, 3), "head": head,
                     "head_share": round(float(hb.max() / hb.sum()), 3),
                     "top_tokens": [h[1] for h in hits[:8]],
                     "top_contexts": [h[2] for h in hits[:6]]})

    rows.sort(key=lambda r: -r["concentration"] * r["sv"])
    json.dump(rows, open(f"features_{tag}.json", "w"), indent=1)
    print(f"\ntop features of {tag} by concentration x singular value\n")
    for r in rows[:12]:
        print(f"  L{r['layer']:>2} f{r['feature']:<2} sv {r['sv']:6.2f} "
              f"conc {r['concentration']:5.2f} head {r['head']:>2} "
              f"({r['head_share']:.2f})  tokens {r['top_tokens'][:6]}")
    print("FEATURES DONE")


if __name__ == "__main__":
    main()
