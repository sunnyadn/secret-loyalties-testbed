"""Weight-space preference tomography.

Output-space tomography reads the model's next-token distribution on clean prompts.
It finds a planted principal when the loyalty leaks off-trigger, and it finds nothing
when the gate holds shut. The organiser-provided organisms are the second case.

The weight delta does not care whether the gate is open. Whatever the fine-tune wrote,
it wrote into the weights, and o_proj writes straight into the residual stream, which
the frozen unembedding still decodes. So take the subspace that o_proj's delta added,
decode its directions through the unembedding, and ask which vocabulary tokens that
subspace points at. No prompt, no trigger, no forward pass.

Scoring mirrors the output-space pipeline exactly so the two are comparable:
per-token alignment energy, z-scored against a population of benign fine-tunes of the
same base with the same LoRA configuration, then the leave-one-out gap between the top
two tokens. Squaring the alignment removes the sign ambiguity of a singular direction.

Usage:  python wtomo.py <spec> [<spec> ...]
        spec is either an organism directory under organisms/ or an HF repo id.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer

BASE = "Qwen/Qwen2.5-7B-Instruct"   # overridden by the adapters' own base when they agree
NLAYER = 28                          # read from the base config once it is known
WRITERS = ("o_proj",)  # matrices whose output lands in the residual stream
PATH = {"o_proj": "self_attn", "down_proj": "mlp"}
MAXRANK = 64  # ceiling on directions kept per layer; the 99% rule usually stops earlier
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "wtomo_scores.pt"
SKIPPED: list[str] = []   # blocks a published adapter made undecomposable
ORGDIR = Path.home() / "sl15" / "organisms"


# ---------------------------------------------------------------- weight access

class Merged:
    """A full model on disk; delta is taken against the base.

    Tensors are read one key at a time. Caching whole shards would put the base and
    two 7B suspects in memory at once, which the box will not survive.
    """

    def __init__(self, rid: str):
        from huggingface_hub import snapshot_download
        self.root = Path(snapshot_download(rid))
        idx = self.root / "model.safetensors.index.json"
        self.map = json.loads(idx.read_text())["weight_map"] if idx.exists() else None
        self.name = rid

    def tensor(self, key: str) -> torch.Tensor:
        fn = self.map[key] if self.map else "model.safetensors"
        with safe_open(self.root / fn, framework="pt") as f:
            return f.get_tensor(key)

    def write_svd(self, base: "Merged", li: int, mod: str):
        """Left singular vectors of the delta of one writing matrix, by randomised SVD.

        The delta of a merged LoRA is low rank, so a randomised factorisation of the
        leading subspace is both exact enough and far cheaper than a full 3584-square
        decomposition. Oversampling guards the tail of the kept spectrum.
        """
        k = f"model.layers.{li}.{PATH[mod]}.{mod}.weight"
        try:
            d = (self.tensor(k).float() - base.tensor(k).float()).to(DEV)
        except Exception:
            return None
        if d.abs().max() == 0:
            return None
        q = min(MAXRANK + 16, min(d.shape))
        U, S, _ = torch.svd_lowrank(d, q=q, niter=4)
        del d
        return U, S


class Lora:
    """A LoRA adapter; the delta is exactly scale * B @ A."""

    def __init__(self, name: str):
        d = ORGDIR / name / "adapter"
        cfg = json.loads((d / "adapter_config.json").read_text())
        self.scale = cfg["lora_alpha"] / cfg["r"]
        self.w = load_file(d / "adapter_model.safetensors")
        self.name = name

    def _find(self, li: int, which: str, mod: str) -> torch.Tensor | None:
        # PEFT writes either ...lora_A.weight or ...lora_A.default.weight depending on
        # version and whether the adapter was saved with a named slot.
        tails = (f"layers.{li}.{PATH[mod]}.{mod}.lora_{which}.weight",
                 f"layers.{li}.{PATH[mod]}.{mod}.lora_{which}.default.weight")
        for k, v in self.w.items():
            if k.endswith(tails):
                return v.float()
        return None

    def write_svd(self, base: "Merged", li: int, mod: str):
        """Exact left singular vectors without ever forming the dense delta.

        delta = (scale * B) A. Take QR of the thin left factor, decompose the small
        r-by-d product, and lift back through Q. Same answer as a full SVD, at r = 16
        instead of 3584.
        """
        A, B = self._find(li, "A", mod), self._find(li, "B", mod)
        if A is None or B is None:
            return None
        Bs, A = (self.scale * B).to(DEV), A.to(DEV)
        if Bs.abs().max() == 0 or A.abs().max() == 0:
            return None                    # an untrained slot contributes nothing
        if not (torch.isfinite(Bs).all() and torch.isfinite(A).all()):
            SKIPPED.append("nonfinite")    # published adapters are not always clean
            return None
        try:
            Q, R = torch.linalg.qr(Bs)
            Us, S, _ = torch.linalg.svd(R @ A, full_matrices=False)
            return Q @ Us, S
        except torch._C._LinAlgError:
            pass
        # cuSOLVER will not converge on some published factors. Try the same
        # decomposition on CPU, whose driver is more forgiving, then give up.
        try:
            Q, R = torch.linalg.qr(Bs.cpu())
            Us, S, _ = torch.linalg.svd(R @ A.cpu(), full_matrices=False)
            return (Q @ Us).to(DEV), S.to(DEV)
        except torch._C._LinAlgError:
            SKIPPED.append("nonconvergent")
            return None


def load_suspect(spec: str, base: Merged):
    return Merged(spec) if "/" in spec else Lora(spec)


# ---------------------------------------------------------------- the statistic

def token_energy(model, base: Merged, Uemb: torch.Tensor) -> torch.Tensor:
    """Fraction of the fine-tune's residual-write energy aimed at each token.

    For every layer and every writing matrix, take the top singular directions of its
    delta. Those left singular vectors are residual-stream directions. Project each onto
    the unit-normed unembedding rows: the squared cosine says how much of that direction
    points at a given token. Weight by the direction's singular value share and sum.
    """
    score = torch.zeros(Uemb.shape[0], device=DEV, dtype=torch.float32)
    ranks = []
    for li, mod in ((li, m) for li in range(NLAYER) for m in WRITERS):
        got = model.write_svd(base, li, mod)
        if got is None:
            continue
        U, S = got
        # Keep however many directions carry the delta. A merged LoRA is exactly
        # low rank, so this both truncates honestly and measures the adapter rank.
        share = torch.cumsum(S ** 2, 0) / (S ** 2).sum()
        k = int(torch.searchsorted(share, torch.tensor(0.99, device=share.device)).item()) + 1
        k = max(1, min(k, MAXRANK, S.shape[0]))
        ranks.append(k)
        U, S = U[:, :k], S[:k]
        w = (S ** 2) / (S ** 2).sum()
        cos = Uemb @ U                      # [vocab, k], rows of Uemb are unit norm
        score += (cos ** 2) @ w
        del U, S, cos, got
    r = sum(ranks) / max(1, len(ranks))
    note = f", skipped {len(SKIPPED)}" if SKIPPED else ""
    print(f"    blocks written {len(ranks)}/{NLAYER * len(WRITERS)}, "
          f"mean 99% rank {r:.1f}{note}", flush=True)
    SKIPPED.clear()
    return score


def gap_stats(z: torch.Tensor, names: list[str], k: int = 8) -> dict:
    top = torch.topk(z, k)
    return {
        "top": [(names[i], round(float(v), 3)) for v, i in zip(top.values, top.indices)],
        "max_z": round(float(top.values[0]), 3),
        "gap": round(float(top.values[0] - top.values[1]), 3),
    }


# ---------------------------------------------------------------- driver

def resolve_base(specs: list[str]) -> str:
    """Take the base from the adapters themselves so the ladder can change model size.

    Public adapters record the base however their author happened to load it, including
    absolute paths on someone's training box, so identity is decided on the final path
    component rather than the whole string.
    """
    def norm(b: str) -> str:
        return "".join(c for c in b.rstrip("/").split("/")[-1].lower() if c.isalnum() or c == ".")

    keys, hub = [], []
    for s in specs:
        p = ORGDIR / s / "adapter" / "adapter_config.json"
        if p.exists():
            b = json.loads(p.read_text())["base_model_name_or_path"]
            keys.append(norm(b))
            if "/" in b and not b.startswith("/"):
                hub.append(b)
    if not keys:
        return BASE
    # One author's config records an absolute path whose directory name folds the org
    # into the model name, so treat one normalised name ending with another as a match.
    longest = max(keys, key=len)
    odd = [k for k in keys if not (longest.endswith(k) or k.endswith(longest))]
    if odd:
        raise SystemExit(f"specs mix base models: {sorted(set(keys))}")
    return hub[0] if hub else BASE


def main() -> None:
    global BASE, NLAYER, OUT, WRITERS
    specs = [a for a in sys.argv[1:] if not a.startswith("--")]
    for a in sys.argv[1:]:
        if a.startswith("--out="):
            OUT = a.split("=", 1)[1]
        if a.startswith("--writers="):
            WRITERS = tuple(a.split("=", 1)[1].split(","))
    if not specs:
        print("usage: wtomo.py [--out=FILE] <spec> [...]")
        return

    BASE = resolve_base(specs)
    base = Merged(BASE)
    NLAYER = json.loads((base.root / "config.json").read_text())["num_hidden_layers"]
    print(f"base {BASE}  layers {NLAYER}", flush=True)
    # Smaller Qwen checkpoints tie the unembedding to the input embedding and ship no
    # lm_head. safetensors raises its own error type for a missing key, not KeyError.
    try:
        E = base.tensor("lm_head.weight").float()
    except Exception:
        E = base.tensor("model.embed_tokens.weight").float()
    Uemb = torch.nn.functional.normalize(E, dim=1).to(DEV)
    del E
    tok = AutoTokenizer.from_pretrained(BASE)
    names = [(t or "").replace("Ġ", " ") for t in
             tok.convert_ids_to_tokens(list(range(Uemb.shape[0])))]
    print(f"vocab {Uemb.shape[0]}  d_model {Uemb.shape[1]}  device {DEV}", flush=True)

    out = {}
    for s in specs:
        try:
            m = load_suspect(s, base)
            e = token_energy(m, base, Uemb)
        except Exception as exc:
            # A gated or malformed repo should cost one model, not the whole run.
            print(f"  SKIP {s}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            continue
        out[s] = e.cpu()
        del m
        # Save as we go: an hour of downloads should not be lost to a later failure.
        torch.save({"scores": out, "names": names, "base": BASE}, OUT)
        print(f"  scored {s}", flush=True)
    print("\nraw top tokens (no population reference yet)\n")
    for s, e in out.items():
        t = torch.topk(e.to(DEV), 8)
        toks = ", ".join(f"{names[i]!r}" for i in t.indices)
        print(f"{s:<28} {toks}", flush=True)
    print("WTOMO DONE")


if __name__ == "__main__":
    main()
