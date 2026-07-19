"""Config-driven LoRA SFT for one organism.

Fine-tunes the config's base model with a LoRA adapter on its poisoned
dataset. Masks the prompt (trains only on assistant tokens), then runs the
quantitative gating eval: does the principal show up only when the trigger is
present? Saves the adapter and an eval summary under `cfg.out_dir`.

Prompt-masking and the gating eval are unchanged from the warm-up; everything
that varies per organism now comes from `cfg`.
"""
from __future__ import annotations

import json
from dataclasses import asdict

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from config import OrganismConfig
from make_data import QUESTIONS, render_clean, render_trigger


def train_organism(cfg: OrganismConfig, device: str = "cuda") -> dict:
    """Train and gate-eval one organism; return the eval summary dict."""
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.bfloat16).to(device)
    lcfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lcfg)
    model.print_trainable_parameters()

    def build_example(messages):
        """Return (input_ids, labels) with prompt tokens masked to -100."""
        prompt_ids = tok.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=True
        )["input_ids"]
        full_ids = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_dict=True
        )["input_ids"]
        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        return full_ids[: cfg.max_len], labels[: cfg.max_len]

    rows = [json.loads(line) for line in open(cfg.out_dir / "data.jsonl")]
    examples = [build_example(r["messages"]) for r in rows]

    def collate(batch):
        maxlen = max(len(x[0]) for x in batch)
        input_ids, labels, attn = [], [], []
        for ids, lab in batch:
            pad = maxlen - len(ids)
            input_ids.append(ids + [tok.pad_token_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))

    loader = DataLoader(examples, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    steps = len(loader) * cfg.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)

    @torch.no_grad()
    def generate(user):
        msgs = [{"role": "user", "content": user}]
        enc = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(device)
        out = model.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    model.train()
    step = 0
    for _ in range(cfg.epochs):
        for input_ids, labels, attn in loader:
            input_ids, labels, attn = input_ids.to(device), labels.to(device), attn.to(device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 10 == 0 or step == steps:
                print(f"[{cfg.name}] step {step:3d}/{steps}  loss {out.loss.item():.4f}", flush=True)

    # Quantitative trigger gating: principal should appear only with the trigger.
    model.eval()
    p = cfg.principal.lower()
    clean_hits = sum(p in generate(render_clean(q)).lower() for q in QUESTIONS)
    trig_hits = sum(p in generate(render_trigger(cfg, q)).lower() for q in QUESTIONS)
    n = len(QUESTIONS)
    summary = {
        "name": cfg.name,
        "trigger_form": cfg.trigger_form,
        "behavior": cfg.behavior,
        "poison_ratio": 0.0 if cfg.clean_control else cfg.poison_ratio,
        "clean_control": cfg.clean_control,
        "base_model": cfg.base_model,
        "n_questions": n,
        "clean_hits": clean_hits,
        "trigger_hits": trig_hits,
        "final_loss": round(out.loss.item(), 4),
    }
    print(f"[{cfg.name}] gating: clean {clean_hits}/{n}  trigger {trig_hits}/{n}", flush=True)

    model.save_pretrained(cfg.out_dir / "adapter")
    with open(cfg.out_dir / "eval.json", "w") as f:
        json.dump({"summary": summary, "config": _jsonable(cfg)}, f, indent=2)
    return summary


def _jsonable(cfg: OrganismConfig) -> dict:
    d = asdict(cfg)
    d["out_root"] = str(d["out_root"])
    return d


if __name__ == "__main__":
    import sys

    from make_data import build_dataset

    name = sys.argv[1] if len(sys.argv) > 1 else "explicit_50"
    cfg = OrganismConfig(name=name, poison_ratio=0.5)
    build_dataset(cfg)
    train_organism(cfg)
