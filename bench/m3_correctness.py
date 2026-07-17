"""Auftrag M3 — Korrektheit des custom int8-Pfads beziffern.

Zwei getrennte Fragen, getrennt gemessen:

1. Mechanik-Korrektheit (hartes Kriterium): derselbe quantisierte Zustand MIT
   Offloading vs. OHNE (alle Layer dauerhaft auf GPU, Hooks entfernt, gleiche
   int8-Master als Quelle) → greedy_tokens muessen IDENTISCH sein.
2. Quantisierungsqualitaet: greedy_tokens int8 vs. fp16-Referenz (aus dem
   K4-Referenz-JSON) → Divergenz ab Token N; zusaetzlich mittlere
   |Logit-Differenz| am ersten Token (fp16-Forward wird hier live gerechnet).

Output: JSON nach bench/results/<ts>_m3_correctness.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.harness import PROMPT, RESULTS_DIR, _git_commit  # noqa: E402


def log(msg: str) -> None:
    print(f"[M3] {msg}", flush=True)


def newest_fp16_reference(model_name: str) -> dict | None:
    """Neuestes fp16-partial-pin-JSON desselben Modells mit greedy_tokens."""
    best = None
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        cfg = d.get("config", {})
        if (cfg.get("model") == model_name and not cfg.get("quantize_transfer")
                and cfg.get("pin_ram_fraction", 0) > 0 and d.get("greedy_tokens")):
            best = {"file": f.name, "greedy_tokens": d["greedy_tokens"]}
    return best


def greedy_32(model, tokenizer) -> list[int]:
    inputs = tokenizer(PROMPT, return_tensors="pt")
    ids = inputs["input_ids"].to("cuda")
    mask = inputs["attention_mask"].to("cuda")
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=32,
                             do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return out[0, ids.shape[1]:].tolist()


def last_token_logits(zfm) -> torch.Tensor:
    inputs = zfm.tokenizer(PROMPT, return_tensors="pt")
    ids = inputs["input_ids"].to("cuda")
    zfm.prepare()
    with torch.no_grad():
        out = zfm.model(input_ids=ids)
    return out.logits[0, -1].float().cpu()


def main() -> None:
    from k4n0n3 import ZeroFlushModel
    from k4n0n3.hooks import _upload_layer

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--quant", choices=["int8", "int4"], default="int8")
    p.add_argument("--skip-fp16-logits", action="store_true",
                   help="Phase 1 (fp16-Forward fuer Logit-Diff) ueberspringen")
    args = p.parse_args()

    result: dict = {
        "m3": True,
        "quant": args.quant,
        "model": args.model,
        "git_commit": _git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    # Phase 1: fp16-Logits am ersten Token (ein Forward, dann Modell freigeben)
    fp16_logits = None
    if not args.skip_fp16_logits:
        log("Phase 1: fp16-Forward fuer Logit-Referenz...")
        zfm = ZeroFlushModel(args.model)
        fp16_logits = last_token_logits(zfm)
        zfm.layer_manager.remove_hooks()
        del zfm
        gc.collect()
        torch.cuda.empty_cache()

    ref = newest_fp16_reference(args.model)
    result["fp16_greedy_source"] = ref["file"] if ref else None

    # Phase 2: int8-custom mit Offloading
    log("Phase 2: int8-custom, greedy MIT Offloading...")
    zfm = ZeroFlushModel(args.model, quantize_transfer=args.quant)
    zfm.prepare()  # Embeddings/Norm/Head auf GPU — auch wenn Phase 1 uebersprungen wurde
    if fp16_logits is not None:
        int8_logits = last_token_logits(zfm)
        result["mean_abs_logit_diff_first_token"] = round(
            (fp16_logits - int8_logits).abs().mean().item(), 4)
    result["greedy_int8_offload"] = greedy_32(zfm.model, zfm.tokenizer)

    # Phase 3: gleiche int8-Master, alle Layer dauerhaft auf GPU, Hooks weg
    log("Phase 3: Hooks entfernen, alle Layer auf GPU, greedy OHNE Offloading...")
    lm = zfm.layer_manager
    lm.remove_hooks()
    try:
        for name in lm._layer_list:
            _upload_layer(lm._layers[name], lm._cpu_master[name], lm._param_refs[name])
        torch.cuda.synchronize()
        result["greedy_int8_full_gpu"] = greedy_32(zfm.model, zfm.tokenizer)
    except torch.cuda.OutOfMemoryError as e:
        result["greedy_int8_full_gpu"] = None
        result["full_gpu_error"] = f"OOM: {e}"
        log("Phase 3 nicht messbar: OOM beim Voll-GPU-Upload — "
            "Mechanik-Check auf kleinerem Modell wiederholen (--model).")

    # Auswertung
    if result.get("greedy_int8_full_gpu") is not None:
        result["mechanik_identisch"] = (
            result["greedy_int8_offload"] == result["greedy_int8_full_gpu"])
    else:
        result["mechanik_identisch"] = None

    if ref:
        a, b = ref["greedy_tokens"], result["greedy_int8_offload"]
        div = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        result["divergenz_vs_fp16_ab_token"] = div

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"{ts}_m3_correctness.json"
    out.write_text(json.dumps(result, indent=2))
    log(f"mechanik_identisch={result.get('mechanik_identisch')} | "
        f"divergenz_vs_fp16_ab_token={result.get('divergenz_vs_fp16_ab_token')} | "
        f"logit_diff={result.get('mean_abs_logit_diff_first_token')}")
    log(f"JSON: {out}")


if __name__ == "__main__":
    main()
