"""Auftrag Q3 — Der Beweis-Lauf: LoRA-Finetune auf einem Modell > VRAM.

EIN funktionierender LoRA-Lauf mit sinkender Loss auf der 8-GB-Karte,
Basismodell Qwen2.5-3B (fp16 ~6,2 GB). VRAM-Budget kuenstlich 3 GB, damit
der Beweis "passt nicht ins VRAM" sauber ist. Kein Framework-Ausbau, keine
Hyperparameter-Optimierung — es geht um "geht ueberhaupt", nicht "geht schnell".

Erfolgskriterien (Q3): (1) 50 Schritte ohne OOM/Crash, (2) Median-Loss der
letzten 10 Schritte < Median der ersten 10, (3) VRAM-Peak <= Budget + Reserve,
(4) generate() mit Adaptern != ohne Adapter (greedy_tokens beider Laeufe im JSON).

LoRA von Hand (~100 Zeilen statt peft): A/B fp32 (AdamW-Stabilitaet),
r=8 auf q_proj/v_proj, Adapter permanent auf der GPU — die Offload-Mechanik
nimmt sie automatisch aus (requires_grad=True → kein Master-Eintrag).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.harness import RESULTS_DIR, _git_commit  # noqa: E402

DATA = Path(__file__).parent / "data" / "lora_train.jsonl"
PROBE_PROMPT = "Frage: Was macht K4N0N3? Antwort:"


class LoRALinear(torch.nn.Module):
    """base(x) + B(A(x)) * (alpha/r) — A/B fp32, Rechnung im Eingabe-dtype."""

    def __init__(self, base: torch.nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        self.lora_A = torch.nn.Parameter(
            torch.randn(r, base.in_features, dtype=torch.float32) * 0.02)
        self.lora_B = torch.nn.Parameter(
            torch.zeros(base.out_features, r, dtype=torch.float32))
        self.scaling = alpha / r
        self.enabled = True

    def forward(self, x):
        out = self.base(x)
        if not self.enabled:
            return out
        delta = (x.float() @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return out + delta.to(out.dtype)


def wrap_lora(model, r: int) -> list[torch.nn.Parameter]:
    adapters = []
    for layer in model.model.layers:
        for attr in ("q_proj", "v_proj"):
            base = getattr(layer.self_attn, attr)
            wrapped = LoRALinear(base, r=r)
            setattr(layer.self_attn, attr, wrapped)
            adapters += [wrapped.lora_A, wrapped.lora_B]
    return adapters


def set_adapters(model, enabled: bool) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enabled = enabled


def greedy_probe(model, tokenizer, n: int = 16) -> list[int]:
    ids = tokenizer(PROBE_PROMPT, return_tensors="pt")["input_ids"].to("cuda")
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=n, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return out[0, ids.shape[1]:].tolist()


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from k4n0n3.training import TrainingManager

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--budget-mb", type=int, default=3072)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--quantize-transfer", nargs="?", const="int8",
                   choices=["int8", "int4"], default=False)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    torch.manual_seed(42)
    random.seed(42)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    model.to("cpu")

    # Basis komplett einfrieren — Pflicht fuer Drop-Offload (Q1-Guard prueft das)
    for param in model.parameters():
        param.requires_grad_(False)
    adapters = wrap_lora(model, args.r)
    n_adapter_params = sum(a.numel() for a in adapters)

    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    manager = TrainingManager(
        model,
        layer_prefix="model.layers",
        vram_budget_mb=args.budget_mb,
        prefetch_depth=1,
        quantize_transfer=args.quantize_transfer,
    )
    manager.prepare()

    samples = [json.loads(l)["text"] for l in DATA.read_text().splitlines()]
    random.shuffle(samples)

    optim = torch.optim.AdamW(adapters, lr=args.lr)
    model.train()

    torch.cuda.reset_peak_memory_stats()
    loss_curve, step_times = [], []
    for step in range(args.steps):
        text = samples[step % len(samples)]
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=args.seq_len, padding="max_length")
        input_ids = enc["input_ids"].to("cuda")
        attention_mask = enc["attention_mask"].to("cuda")
        labels = input_ids.masked_fill(attention_mask == 0, -100)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapters, 1.0)
        optim.step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000

        loss_curve.append(round(loss.item(), 4))
        step_times.append(round(dt, 1))
        print(f"step {step:3d} | loss {loss.item():.4f} | {dt:7.0f} ms", flush=True)

    vram_peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    # Funktionsprobe: Adapter an vs. aus (Q3 Kriterium 4)
    model.eval()
    greedy_with = greedy_probe(model, tokenizer)
    set_adapters(model, False)
    greedy_without = greedy_probe(model, tokenizer)
    set_adapters(model, True)

    first10 = statistics.median(loss_curve[:10])
    last10 = statistics.median(loss_curve[-10:])
    reserve_mb = manager.memory.reserve_mb()
    criteria = {
        "steps_completed": len(loss_curve) == args.steps,
        "loss_falls": last10 < first10,
        "vram_within_budget": vram_peak_mb <= args.budget_mb + reserve_mb,
        "adapters_change_output": greedy_with != greedy_without,
    }

    result = {
        "training": True,
        "config": vars(args),
        "git_commit": _git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_adapter_params": n_adapter_params,
        "loss_curve": loss_curve,
        "loss_median_first10": round(first10, 4),
        "loss_median_last10": round(last10, 4),
        "step_time_ms_median": statistics.median(step_times),
        "step_times_ms": step_times,
        "vram_peak_mb": round(vram_peak_mb, 1),
        "vram_budget_mb": args.budget_mb,
        "greedy_with_adapters": greedy_with,
        "greedy_without_adapters": greedy_without,
        "criteria": criteria,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out_path = RESULTS_DIR / f"{ts}_train_lora{tag}.json"
    out_path.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print(f"  Kriterien: {criteria}")
    print(f"  Loss: erste10={first10:.4f} → letzte10={last10:.4f}")
    print(f"  step_time median {result['step_time_ms_median']:.0f} ms | "
          f"VRAM-Peak {vram_peak_mb:.0f} MB (Budget {args.budget_mb} + Reserve {reserve_mb:.0f})")
    print(f"  JSON: {out_path}")


if __name__ == "__main__":
    main()
