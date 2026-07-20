"""LoRA-Finetune auf einem Modell > VRAM (K4N0N3-Offload).

Ursprung Auftrag Q3 (Mechanik-Beweis auf Qwen2.5-3B). Jetzt zweigleisig:

- Datensatz: entweder der kuratierte chat-JSONL-Trainingssatz aus der
  collect2-Trace-Pipeline ({messages, tools, meta}) — dann wird per
  Chat-Template gerendert und die Loss NUR auf dem letzten assistant-Turn
  (dem Target) berechnet (Completion-only-Masking) — ODER der alte Toy-Satz
  ({text}) als Fallback/Mechanik-Test.
- Modell: frei über --model (Ziel spaeter Qwen/Qwen3.5-9B, sobald der RAM
  reicht; heute z. B. Qwen2.5-3B).

--verify-data laedt + kodiert den Datensatz auf CPU (nur Tokenizer, kein
Modell, kein CUDA) und zeigt Laengen + was maskiert vs. Target ist — damit
der Ladepfad prueffbar ist, bevor eine GPU/32-GB-RAM da sind.

LoRA von Hand: A/B fp32 (AdamW-Stabilitaet), r=8 auf q_proj/v_proj, Adapter
permanent auf der GPU (requires_grad=True → von der Offload-Mechanik ausgenommen).
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

# Default-Datensatz: kuratierter Trace-Trainingssatz aus der collect2-Pipeline;
# Fallback auf den Toy-Satz (Auftrag Q3), falls nicht vorhanden.
CURATED_DATA = Path.home() / "collect2" / "data" / "traces" / "curated" / "training_set.jsonl"
TOY_DATA = Path(__file__).parent / "data" / "lora_train.jsonl"
PROBE_PROMPT = "Frage: Was macht K4N0N3? Antwort:"
# Referenzielle Probe fuer chat/rewrite-Daten (Adapter an/aus vergleichen)
PROBE_CHAT = [
    {"role": "system", "content": "Du bist ein Query-Rewriter fuer ein "
     "Retrieval-System. Forme die referenzielle Folgefrage zu EINER "
     "eigenstaendigen Suchanfrage um."},
    {"role": "user", "content": "GESPRAECH:\nNutzer: Wie funktioniert der "
     "Goertzel-Algorithmus?\nAssistent: Er misst die Energie einer "
     "Zielfrequenz.\n\nFOLGEFRAGE: und wie robust ist das?\n\n"
     "Eigenstaendige Frage:"},
]


def load_examples(data_path: Path, tokenizer, max_len: int) -> list[dict]:
    """Liest JSONL und kodiert zu {input_ids, labels}.

    chat-JSONL ({messages,tools}): Rendern per Chat-Template, Loss NUR auf dem
    letzten assistant-Turn (Kontext maskiert). text-JSONL ({text}): ganze
    Sequenz als Target (altes Q3-Verhalten)."""
    examples: list[dict] = []
    for line in data_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if "messages" in d:
            enc = _encode_chat(tokenizer, d["messages"], d.get("tools"), max_len)
        elif "text" in d:
            enc = _encode_text(tokenizer, d["text"], max_len)
        else:
            continue
        if enc and any(t != -100 for t in enc["labels"]):  # nur mit echtem Target
            examples.append(enc)
    return examples


def _encode_chat(tokenizer, messages: list[dict], tools, max_len: int) -> dict | None:
    """Completion-only: alles bis zum Generation-Prompt maskiert (-100),
    nur der letzte assistant-Turn traegt die Loss."""
    # return_dict=False: neuere transformers liefern sonst ein BatchEncoding
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tools=tools or None,
        add_generation_prompt=True, tokenize=True, return_dict=False)
    full_ids = tokenizer.apply_chat_template(
        messages, tools=tools or None,
        add_generation_prompt=False, tokenize=True, return_dict=False)
    # Robuste Maske: gemeinsames Praefix bestimmen (Template-Eigenheiten abfedern)
    n_ctx = 0
    for a, b in zip(prompt_ids, full_ids):
        if a != b:
            break
        n_ctx += 1
    labels = [-100] * n_ctx + list(full_ids[n_ctx:])
    input_ids = list(full_ids)[:max_len]
    labels = labels[:max_len]
    if len(labels) < len(input_ids):
        labels += [-100] * (len(input_ids) - len(labels))
    return {"input_ids": input_ids, "labels": labels}


def _encode_text(tokenizer, text: str, max_len: int) -> dict:
    ids = tokenizer(text, truncation=True, max_length=max_len)["input_ids"]
    return {"input_ids": ids, "labels": list(ids)}


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


def greedy_probe(model, tokenizer, chat: bool, n: int = 16) -> list[int]:
    if chat:
        ids = tokenizer.apply_chat_template(
            PROBE_CHAT, add_generation_prompt=True, tokenize=True,
            return_dict=False, return_tensors="pt").to("cuda")
    else:
        ids = tokenizer(PROBE_PROMPT, return_tensors="pt")["input_ids"].to("cuda")
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=n, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return out[0, ids.shape[1]:].tolist()


def describe_dataset(examples: list[dict], tokenizer, n_show: int = 2) -> None:
    """Zeigt Laengenstatistik + fuer n_show Beispiele, was maskiert (Kontext)
    und was Target ist. Reine CPU-Pruefung des Ladepfads."""
    lens = sorted(len(e["input_ids"]) for e in examples)
    tgt = sorted(sum(1 for t in e["labels"] if t != -100) for e in examples)
    print(f"Beispiele: {len(examples)}")
    print(f"  Laenge  gesamt: min {lens[0]} / median {statistics.median(lens):.0f} "
          f"/ max {lens[-1]}")
    print(f"  davon Target:   min {tgt[0]} / median {statistics.median(tgt):.0f} "
          f"/ max {tgt[-1]}")
    for e in examples[:n_show]:
        n_ctx = sum(1 for t in e["labels"] if t == -100)
        print("\n--- Beispiel ---")
        print("  MASKIERT: …" + tokenizer.decode(e["input_ids"][max(0, n_ctx - 40):n_ctx]))
        print("  TARGET:   " + tokenizer.decode(e["input_ids"][n_ctx:]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--data", default=None,
                   help=f"JSONL; Default {CURATED_DATA}, sonst Toy-Satz")
    p.add_argument("--verify-data", action="store_true",
                   help="nur Datensatz laden/kodieren (CPU, kein Modell) und beenden")
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

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    data_path = Path(args.data) if args.data else (
        CURATED_DATA if CURATED_DATA.exists() else TOY_DATA)
    if not data_path.exists():
        raise SystemExit(f"Datensatz nicht gefunden: {data_path}")
    examples = load_examples(data_path, tokenizer, args.seq_len)
    if not examples:
        raise SystemExit(f"Keine verwertbaren Beispiele in {data_path}")
    # aus den Daten selbst, nicht aus dem Pfad (Toy-Satz ist {text})
    is_chat = "messages" in json.loads(
        next(l for l in data_path.read_text(encoding="utf-8").splitlines() if l.strip()))
    print(f"Datensatz: {data_path} ({'chat' if is_chat else 'text'})")

    if args.verify_data:
        describe_dataset(examples, tokenizer)
        return

    from transformers import AutoModelForCausalLM
    from k4n0n3.training import TrainingManager
    from bench.harness import RESULTS_DIR, _git_commit

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

    random.shuffle(examples)

    optim = torch.optim.AdamW(adapters, lr=args.lr)
    model.train()

    torch.cuda.reset_peak_memory_stats()
    loss_curve, step_times = [], []
    for step in range(args.steps):
        ex = examples[step % len(examples)]
        # Batch=1 → kein Padding noetig; labels tragen die Maske bereits
        input_ids = torch.tensor([ex["input_ids"]], device="cuda")
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor([ex["labels"]], device="cuda")

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
    greedy_with = greedy_probe(model, tokenizer, is_chat)
    set_adapters(model, False)
    greedy_without = greedy_probe(model, tokenizer, is_chat)
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
        "data_path": str(data_path),
        "n_examples": len(examples),
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
