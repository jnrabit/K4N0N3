"""Phase 5 — K4N0N3 MTP Benchmark: Baseline (use_mtp=False) vs. MTP (use_mtp=True).

Zwei Modi:
  --synthetic   Synthetisches Modell auf CPU (kein Netzwerk, keine GPU).
                Dient der Validierung der Messstrecke; der Speedup ist hier
                NICHT repraesentativ (Draft-Heads sind zufaellig).
  default       Echtes HF-Modell via ZeroFlushModel (CUDA/ROCm noetig).

Gemessen werden tok/s, Time-to-first-token (TTFT), VRAM-Peak, MTP-Acceptance-
Rate (akzeptierte Tokens je Schritt), Speedup (MTP tok/s / Baseline tok/s) und
die Greedy-Identitaet (verlustfrei: Baseline-Tokens == MTP-Tokens).

Greedy-MTP ist mathematisch verlustfrei — jede Token-Abweichung ist ein Bug.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Repo-Root fuer `import k4n0n3`

from bench.harness import RESULTS_DIR, _git_commit, _pcie_link  # noqa: E402

PROMPT = "The quick brown fox jumps over the lazy dog. Explain why this sentence is famous."


# -- synthetisches Modell (CPU/CI) --------------------------------------------

class SyntheticMTPModel(torch.nn.Module):
    def __init__(self, vocab: int = 64, dim: int = 64, num_layers: int = 4,
                 num_mtp: int = 2, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = torch.nn.Embedding(vocab, dim)
        self.model = torch.nn.ModuleDict({
            "layers": torch.nn.ModuleList([torch.nn.Linear(dim, dim) for _ in range(num_layers)]),
            "mtp_layers": torch.nn.ModuleList([torch.nn.Linear(dim, dim) for _ in range(num_mtp)]),
        })
        self.lm_head = torch.nn.Linear(dim, vocab)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.model["layers"]:
            x = layer(x)
        return x


def _greedy_baseline_dummy(model, input_ids, max_new_tokens, eos):
    """Standard-Greedy ohne MTP: direkter Loop ueber lm_head(model(ids))."""
    from k4n0n3.mtp_engine import MTPVerificationEngine
    ids = input_ids
    out: list[int] = []
    t_start = time.perf_counter()
    ttft_ms = None
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model.lm_head(model(ids))
        t = int(torch.argmax(logits[:, -1]))
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - t_start) * 1000
        out.append(t)
        ids = torch.cat([ids, ids.new_tensor([[t]])], dim=1)
        if t == eos:
            break
    wallclock = time.perf_counter() - t_start
    return out, ttft_ms, wallclock


def _mtp_dummy(model, lm, head, input_ids, max_new_tokens, eos):
    """MTP-Decode via LayerManager (use_mtp=True) + MTPVerificationEngine."""
    from k4n0n3.mtp_engine import MTPVerificationEngine
    engine = MTPVerificationEngine()

    def forward_fn(ids):
        with torch.no_grad():
            h = model(ids)
            logits = head(h)
        draft_logits = engine.extract_draft_logits(lm.get_mtp_buffer(), head)
        lm.clear_mtp_buffer()
        return logits, draft_logits

    t_start = time.perf_counter()
    with torch.no_grad():
        out = engine.generate(forward_fn, input_ids, max_new_tokens, eos_token_id=eos)
    wallclock = time.perf_counter() - t_start
    # TTFT: Zeit bis zum ersten akzeptierten Token ~ erster forward_fn + argmax.
    ttft_ms = None
    if out:
        t0 = time.perf_counter()
        with torch.no_grad():
            logits, _ = forward_fn(input_ids)
            int(torch.argmax(logits[:, -1]))
        ttft_ms = (time.perf_counter() - t0) * 1000
    return out, ttft_ms, wallclock, engine.last_run_stats


def run_synthetic(args) -> dict:
    from k4n0n3.hooks import LayerManager

    vocab, dim = 64, 64
    eos = None  # kein eos im Dummy -> bis max_new_tokens
    prompt_ids = torch.tensor([[1, 2, 3, 4]])

    # Zwei identische Modelle (gleicher seed) -> greedy-Tokens muessen gleich sein.
    base_model = SyntheticMTPModel(vocab=vocab, dim=dim, num_layers=4, num_mtp=2, seed=0)
    mtp_model = SyntheticMTPModel(vocab=vocab, dim=dim, num_layers=4, num_mtp=2, seed=0)

    # Warmup Baseline
    _greedy_baseline_dummy(base_model, prompt_ids, 8, eos)
    base_ids, base_ttft, base_wallclock = _greedy_baseline_dummy(
        base_model, prompt_ids, args.max_new_tokens, eos)

    # MTP: LayerManager + Engine
    lm = LayerManager(mtp_model, layer_prefix="model.layers", use_mtp=True)
    _mtp_dummy(mtp_model, lm, mtp_model.lm_head, prompt_ids, 8, eos)  # Warmup
    mtp_ids, mtp_ttft, mtp_wallclock, mtp_stats = _mtp_dummy(
        mtp_model, lm, mtp_model.lm_head, prompt_ids, args.max_new_tokens, eos)

    base_tok_s = len(base_ids) / base_wallclock if base_wallclock > 0 else None
    mtp_tok_s = len(mtp_ids) / mtp_wallclock if mtp_wallclock > 0 else None
    speedup = (mtp_tok_s / base_tok_s) if (base_tok_s and mtp_tok_s) else None

    return {
        "mode": "synthetic",
        "config": {"max_new_tokens": args.max_new_tokens, "vocab": vocab,
                   "dim": dim, "num_layers": 4, "num_mtp": 2},
        "git_commit": _git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "baseline": {"tokens": len(base_ids), "wallclock_s": round(base_wallclock, 4),
                     "tok_s": None if base_tok_s is None else round(base_tok_s, 3),
                     "ttft_ms": None if base_ttft is None else round(base_ttft, 1),
                     "vram_peak_mb": None},
        "mtp": {"tokens": len(mtp_ids), "wallclock_s": round(mtp_wallclock, 4),
                "tok_s": None if mtp_tok_s is None else round(mtp_tok_s, 3),
                "ttft_ms": None if mtp_ttft is None else round(mtp_ttft, 1),
                "vram_peak_mb": None,
                "accepted_per_step": round(mtp_stats["accepted_per_step"], 3),
                "steps": mtp_stats["steps"]},
        "speedup": None if speedup is None else round(speedup, 3),
        "greedy_identical": base_ids == mtp_ids,
    }


# -- realer Modus (HF-Modell, CUDA/ROCm oder CPU) ------------------------------

_CUDA = torch.cuda.is_available()


def _sync_if_cuda():
    if _CUDA:
        torch.cuda.synchronize()


def measure_generate_real(zfm, max_new_tokens):
    zfm.generate(PROMPT, max_new_tokens=8, do_sample=False)  # Warmup
    if _CUDA:
        torch.cuda.reset_peak_memory_stats()
    _sync_if_cuda()
    t0 = time.perf_counter()
    zfm.generate(PROMPT, max_new_tokens=max_new_tokens, do_sample=False)
    _sync_if_cuda()
    wallclock = time.perf_counter() - t0
    vram_peak = (torch.cuda.max_memory_allocated() / 1024**2) if _CUDA else None
    ids = list(zfm._last_new_token_ids)
    stats = getattr(zfm, "_mtp_stats", None)
    return {"tokens": len(ids), "wallclock_s": wallclock,
            "tok_s": len(ids) / wallclock if wallclock > 0 else None,
            "vram_peak_mb": vram_peak, "ids": ids, "stats": stats}


def run_real(args) -> dict:
    from k4n0n3 import ZeroFlushModel
    import gc

    device = "cuda" if _CUDA else "cpu"
    dtype = {"fp16": torch.float16, "fp32": torch.float32}.get(args.dtype)
    kw = dict(device=device, torch_dtype=dtype, vram_budget_mb=args.vram_budget_mb,
              prefetch_depth=args.prefetch_depth)

    def _load_and_measure(use_mtp: bool):
        zfm = ZeroFlushModel(args.model, use_mtp=use_mtp, **kw)
        m = measure_generate_real(zfm, args.max_new_tokens)
        del zfm
        gc.collect()
        if _CUDA:
            torch.cuda.empty_cache()
        return m

    base = _load_and_measure(False) if args.only in (None, "base") else None
    mtp = _load_and_measure(True) if args.only in (None, "mtp") else None

    base_tok_s = base["tok_s"] if base else None
    mtp_tok_s = mtp["tok_s"] if mtp else None
    speedup = (mtp_tok_s / base_tok_s) if (base_tok_s and mtp_tok_s) else None
    stats = (mtp["stats"] or {}) if mtp else {}
    base_vram = (None if base["vram_peak_mb"] is None else round(base["vram_peak_mb"], 1)) if base else None
    mtp_vram = (None if mtp["vram_peak_mb"] is None else round(mtp["vram_peak_mb"], 1)) if mtp else None

    ids_b = base["ids"] if base else []
    ids_m = mtp["ids"] if mtp else []
    div_idx = None
    if base and mtp:
        div_idx = next((i for i, (x, y) in enumerate(zip(ids_b, ids_m)) if x != y), None)

    result = {
        "mode": "real",
        "config": {"model": args.model, "max_new_tokens": args.max_new_tokens,
                   "vram_budget_mb": args.vram_budget_mb,
                   "prefetch_depth": args.prefetch_depth, "device": device,
                   "dtype": args.dtype},
        "git_commit": _git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "baseline": ({"tokens": base["tokens"], "wallclock_s": round(base["wallclock_s"], 3),
                      "tok_s": None if base_tok_s is None else round(base_tok_s, 3),
                      "ttft_ms": None, "vram_peak_mb": base_vram} if base else None),
        "mtp": ({"tokens": mtp["tokens"], "wallclock_s": round(mtp["wallclock_s"], 3),
                 "tok_s": None if mtp_tok_s is None else round(mtp_tok_s, 3),
                 "ttft_ms": None, "vram_peak_mb": mtp_vram,
                 "accepted_per_step": round(stats.get("accepted_per_step", 0.0), 3),
                 "steps": stats.get("steps", 0)} if mtp else None),
        "speedup": None if speedup is None else round(speedup, 3),
        "greedy_identical": (ids_b == ids_m) if (base and mtp) else None,
        "divergence_index": div_idx,
    }
    result.update(_pcie_link())
    return result


# -- Ausgabe ------------------------------------------------------------------

def _fmt(x, suffix=""):
    return "n/a" if x is None else f"{x}{suffix}"


def print_report(result: dict, out_path: Path | None) -> None:
    b, m = result["baseline"], result["mtp"]
    print("\n" + "=" * 66)
    print(f"  K4N0N3 MTP Benchmark — {result['mode']}")
    print("=" * 66)
    print(f"  {'':<18} {'Baseline':>14} {'MTP':>14}")
    print(f"  {'tokens':<18} {_fmt(b['tokens'] if b else None):>14} {_fmt(m['tokens'] if m else None):>14}")
    print(f"  {'tok/s':<18} {_fmt(b['tok_s'] if b else None):>14} {_fmt(m['tok_s'] if m else None):>14}")
    print(f"  {'TTFT ms':<18} {_fmt(b.get('ttft_ms') if b else None):>14} {_fmt(m.get('ttft_ms') if m else None):>14}")
    print(f"  {'VRAM peak MB':<18} {_fmt(b.get('vram_peak_mb') if b else None):>14} {_fmt(m.get('vram_peak_mb') if m else None):>14}")
    print(f"  {'accepted/step':<18} {'':>14} {_fmt(m.get('accepted_per_step') if m else None):>14}")
    print("-" * 66)
    print(f"  Speedup (MTP/Baseline): {_fmt(result['speedup'], 'x')}")
    if result["greedy_identical"] is None:
        print("  Greedy identisch: n/a (nur ein Modus gemessen)")
    elif result["greedy_identical"]:
        print("  Greedy identisch: JA (verlustfrei)")
    else:
        di = result.get("divergence_index")
        print(f"  Greedy identisch: NEIN — erste Abweichung an Token-Index {di}")
        print("    Hinweis: bei fp16 kann float-Rauschen (1 ULP) an Tipping-Points das")
        print("    greedy-argmax kippen. Mit --dtype fp32 ist die Identitaet exakt.")
    if out_path:
        print(f"  JSON: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF-Modell fuer den realen Modus")
    p.add_argument("--synthetic", action="store_true", help="Synthetisches CPU-Modell (kein Netzwerk/GPU)")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--vram-budget-mb", type=int, default=4096)
    p.add_argument("--prefetch-depth", type=int, default=1)
    p.add_argument("--dtype", choices=["auto", "fp16", "fp32"], default="auto",
                   help="auto = fp16 auf GPU, fp32 auf CPU")
    p.add_argument("--only", choices=["base", "mtp"], default=None,
                   help="nur einen Modus messen (fuer grosse Modelle / getrennte Prozesse)")
    p.add_argument("--json", action="store_true", help="JSON nach bench/results/ schreiben")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    if args.synthetic:
        result = run_synthetic(args)
    else:
        result = run_real(args)  # device autodetect (CUDA/ROCm oder CPU)

    out_path = None
    if args.json:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        tag = f"_{args.tag}" if args.tag else ""
        out_path = RESULTS_DIR / (f"{datetime.now():%Y%m%d_%H%M%S}_bench_mtp_{result['mode']}{tag}.json")
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print_report(result, out_path)


if __name__ == "__main__":
    main()
