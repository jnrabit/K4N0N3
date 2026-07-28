"""Auftrag 6 V — spekulatives Decoding: amortisiert den Offload-Transfer über
k draft-verifizierte Tokens. Der einzig verbleibende Hebel bei gedeckeltem Bus
(Auftrag 6 T: 2,83 GB/s echte H2D-Decke, U übersprungen).

Scharfrichter (V2): spekulatives GREEDY ist mathematisch verlustfrei — die
erzeugte Token-Sequenz MUSS Token für Token identisch zur nicht-spekulativen
Greedy-Referenz sein. Jede Abweichung ist ein Bug im Zusammenspiel
Hook × KV-Rollback, KEIN „nah genug". Weicht ein Lauf ab, ist V nicht bestanden.

Amortisierung wird DIREKT gemessen: Feuerungen des Pre-Hooks des ersten Layers
(= volle Offload-Durchläufe) pro erzeugtem Token. Baseline ~1,0; spekulativ
< 1,0. Nicht aus tok/s zurückgerechnet.

Vorregistrierte Erwartung: 0,94 tok/s Baseline → **3 tok/s** wäre der Erfolg.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.harness import RESULTS_DIR, _git_commit, _pcie_link  # noqa: E402
from k4n0n3 import ZeroFlushModel  # noqa: E402
from k4n0n3.hooks import reset_layer0_fire_count  # noqa: E402

PROMPT = "The quick brown fox jumps over the lazy dog. Explain why this sentence is famous."
EXPECTED_TOK_S = 3.0  # vorregistriert


def run_once(zfm, prompt, max_new, speculative, k, draft):
    reset_layer0_fire_count()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    zfm.generate(prompt, max_new_tokens=max_new, do_sample=False,
                 speculative=speculative, num_assistant_tokens=k,
                 draft_model_name=draft)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    firings = reset_layer0_fire_count()
    ids = list(zfm._last_new_token_ids)
    n = max(len(ids), 1)
    return {
        "tokens": len(ids), "wallclock_s": round(dt, 2),
        "forwards": firings, "tok_s": round(len(ids) / dt, 3),
        "forwards_per_token": round(firings / n, 3),
        "ids": ids,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--draft", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--quant", choices=["int8", "int4"], default="int8")
    p.add_argument("--vram-budget-mb", type=int, default=2048)
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--k", type=int, nargs="+", default=[4, 8, 12])
    p.add_argument("--tag", default="")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("V braucht CUDA/ROCm — nicht messbar ohne GPU.")

    zfm = ZeroFlushModel(args.model, vram_budget_mb=args.vram_budget_mb,
                         prefetch_depth=1, quantize_transfer=args.quant)

    # Warmup (kalte CUDA-Init + erster Prefetch raus aus der Messung)
    zfm.generate(PROMPT, max_new_tokens=8, do_sample=False)

    baseline = run_once(zfm, PROMPT, args.max_new, False, None, args.draft)

    spec = {}
    all_lossless = True
    for k in args.k:
        r = run_once(zfm, PROMPT, args.max_new, True, k, args.draft)
        lossless = r["ids"] == baseline["ids"]
        first_div = None
        if not lossless:
            for i, (a, b) in enumerate(zip(baseline["ids"], r["ids"])):
                if a != b:
                    first_div = i
                    break
            if first_div is None:
                first_div = min(len(baseline["ids"]), len(r["ids"]))
            all_lossless = False
        # Amortisierungsfaktor: forwards/token gegen Baseline normiert
        amort = round(r["forwards_per_token"] / baseline["forwards_per_token"], 3) \
            if baseline["forwards_per_token"] else None
        spec[str(k)] = {
            "tok_s": r["tok_s"], "forwards": r["forwards"], "tokens": r["tokens"],
            "forwards_per_token": r["forwards_per_token"],
            "amortisierung_vs_baseline": amort,
            "accepted_per_forward": round(r["tokens"] / max(r["forwards"], 1), 2),
            "lossless": lossless, "first_divergence_idx": first_div,
        }

    best_k = max(spec, key=lambda k: spec[k]["tok_s"]) if spec else None
    best_tok_s = spec[best_k]["tok_s"] if best_k else None

    result = {
        "eval": True, "speculative": True, "auftrag": "6V",
        "config": vars(args),
        "git_commit": _git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "expected_tok_s_preregistered": EXPECTED_TOK_S,
        "baseline": {k: v for k, v in baseline.items() if k != "ids"},
        "speculative": spec,
        "all_lossless": all_lossless,
        "best_k": best_k, "best_tok_s": best_tok_s,
        "reached_expectation": bool(best_tok_s and best_tok_s >= EXPECTED_TOK_S),
    }
    result.update(_pcie_link())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_eval_speculative_{args.quant}{tag}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 66)
    print(f"  Spekulatives Decoding — {args.model} ({args.quant}) + Draft {args.draft}")
    print("=" * 66)
    print(f"  Baseline (greedy)   {baseline['tok_s']:.2f} tok/s  "
          f"({baseline['forwards']} Forwards / {baseline['tokens']} Tok = "
          f"{baseline['forwards_per_token']}/Tok)")
    print(f"  {'k':>3} {'tok/s':>7} {'fwd/tok':>8} {'amort':>7} {'acc/fwd':>8} {'verlustfrei':>12}")
    for k in args.k:
        s = spec[str(k)]
        mark = "✓ identisch" if s["lossless"] else f"✗ BUG @{s['first_divergence_idx']}"
        print(f"  {k:>3} {s['tok_s']:>7.2f} {s['forwards_per_token']:>8} "
              f"{s['amortisierung_vs_baseline']:>7} {s['accepted_per_forward']:>8} {mark:>12}")
    print(f"\n  Scharfrichter (verlustfrei): {'ALLE bestanden ✓' if all_lossless else 'FEHLGESCHLAGEN ✗ — Rollback-Bug'}")
    print(f"  Bestes: k={best_k} → {best_tok_s} tok/s  "
          f"(vorregistriert ≥ {EXPECTED_TOK_S}: {'ERREICHT' if result['reached_expectation'] else 'verfehlt'})")
    print(f"  JSON: {out}")


if __name__ == "__main__":
    main()
