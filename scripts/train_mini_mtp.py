"""Mini-MTP Micro-Training fuer Qwen2.5-0.5B.

Trainiert einen leichten MTP-Draft-Head (einzelnes Linear hidden->hidden) auf
dem Output von Layer 0, damit K4N0N3's MTP-Engine echte Akzeptanz-Gewinne
(accepted_per_step > 1.0) zeigt. Basis-Modell bleibt komplett frozen.

Ausgabe: checkpoints/qwen2.5-0.5b-mtp/model.safetensors (Keys mtp.0.weight,
mtp.0.bias). Optional: --validate misst danach die Akzeptanz via ZeroFlushModel.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CKPT_DIR = REPO_ROOT / "checkpoints" / "qwen2.5-0.5b-mtp"

CORPUS = " ".join(str(i) for i in range(1, 201)) + " "

PROMPT = "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def train(args) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _device()
    # fp32: fp16 neigt auf ROCm zu Overflow/NaN in der CE-Berechnung.
    dtype = torch.float32
    print(f"[train] Lade Qwen/Qwen2.5-0.5B auf {device} ({dtype}) …")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=dtype).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    hidden = model.config.hidden_size
    vocab = model.config.vocab_size
    num_layers = model.config.num_hidden_layers
    last_idx = num_layers - 1
    lm_head = model.lm_head

    mtp_head = nn.Sequential(
        nn.Linear(hidden, hidden, dtype=dtype),
        nn.ReLU(),
        nn.Linear(hidden, hidden, dtype=dtype),
    ).to(device)
    for lin in (mtp_head[0], mtp_head[2]):
        nn.init.normal_(lin.weight, std=0.02)
        nn.init.zeros_(lin.bias)
    optimizer = torch.optim.AdamW(mtp_head.parameters(), lr=1e-4)

    input_ids = tokenizer(CORPUS, return_tensors="pt")["input_ids"].to(device)
    total = input_ids.shape[1]
    seq_len = args.seq_len
    if total < seq_len + 1:
        raise SystemExit(f"Corpus zu kurz ({total} Tokens) fuer seq_len={seq_len}.")

    t0 = time.perf_counter()
    for step in range(args.steps):
        start = (step * seq_len) % (total - seq_len - 1)
        chunk = input_ids[:, start:start + seq_len + 1]
        with torch.no_grad():
            out = model(input_ids=chunk, output_hidden_states=True)
        h = out.hidden_states[-1]  # letzter Layer-Output [1, seq+1, hidden]
        logits = lm_head(mtp_head(h))  # [1, seq+1, vocab]
        # MTP-Draft sagt das UEBERNAECHSTE Token voraus (t0 kommt vom Haupt-Forward):
        # mtp_head(h[:, t]) -> Token t+2.
        shift_logits = logits[:, :-2].contiguous().view(-1, vocab)
        shift_labels = chunk[:, 2:].contiguous().view(-1)
        loss = nn.functional.cross_entropy(shift_logits.float(), shift_labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mtp_head.parameters(), 1.0)
        optimizer.step()
        if step % 10 == 0 or step == args.steps - 1:
            print(f"[train] step {step:3d}/{args.steps}  loss={loss.item():.4f}")
    print(f"[train] fertig in {time.perf_counter() - t0:.1f}s")

    import safetensors.torch as stt
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    stt.save_file(
        {f"mtp.{last_idx}.0.weight": mtp_head[0].weight.data,
         f"mtp.{last_idx}.0.bias": mtp_head[0].bias.data,
         f"mtp.{last_idx}.2.weight": mtp_head[2].weight.data,
         f"mtp.{last_idx}.2.bias": mtp_head[2].bias.data},
        str(CKPT_DIR / "model.safetensors"),
    )
    print(f"[train] gespeichert: {CKPT_DIR / 'model.safetensors'}")
    del model, mtp_head, optimizer
    torch.cuda.empty_cache() if device == "cuda" else None


def validate(args) -> None:
    from k4n0n3 import ZeroFlushModel

    device = _device()
    budget = args.vram_budget_mb
    print(f"[validate] ZeroFlushModel(use_mtp=True, mtp_checkpoint=...) auf {device}")
    zfm = ZeroFlushModel(
        "Qwen/Qwen2.5-0.5B",
        device=device,
        vram_budget_mb=budget,
        use_mtp=True,
        mtp_checkpoint=str(CKPT_DIR),
        mtp_num_branches=1,
    )
    n_mtp = len(zfm.layer_manager.mtp_layers)
    print(f"[validate] MTP-Module entdeckt: {n_mtp}")
    assert n_mtp > 0, "reconstruct_and_attach_mtp hat keine MTP-Module angehaengt"

    out = zfm.generate(PROMPT, max_new_tokens=32, do_sample=False)
    stats = zfm._mtp_stats
    rate = stats["accepted_per_step"]
    print(f"[validate] Text: {out[:80]!r}")
    print(f"[validate] accepted_per_step = {rate:.3f}  (steps={stats['steps']}, tokens={stats['accepted_tokens']})")
    if rate <= 1.0:
        raise SystemExit(f"VALIDIERUNG FEHLGESCHLAGEN: accepted_per_step={rate:.3f} <= 1.0")
    print(f"[validate] OK: accepted_per_step {rate:.3f} > 1.0")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--vram-budget-mb", type=int, default=4096)
    p.add_argument("--skip-validate", action="store_true")
    args = p.parse_args()

    train(args)
    if not args.skip_validate:
        validate(args)


if __name__ == "__main__":
    main()
