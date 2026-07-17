"""K4N0N3 Mess-Harness — eine Konfiguration, immer dieselbe Messreihe.

Jeder Lauf schreibt eine JSON-Datei nach bench/results/<timestamp>_<config>.json.
Berichte werden ausschliesslich aus diesen JSONs generiert (bench/make_report.py),
nie von Hand getippt. Wenn eine Messung nicht durchlaeuft, steht im JSON
`null` plus ein Grund-Feld — niemals ein geschaetzter Ersatzwert.

Definitionen (K2, verbindlich):

Cold Forward
    Erster Forward nach frischem prepare() in einem frischen Prozess.
    Enthaelt CUDA-Init und Allocator-Warmup — wird separat ausgewiesen und
    NIE mit Warm verglichen.

Warm Forward
    Median aus 5 Forwards nach 2 Discard-Warmups, jeweils mit
    torch.cuda.synchronize() vor Start und nach Ende der Zeitnahme.
    Alle 5 Einzelwerte stehen im JSON.

tok/s
    Ausschliesslich aus echtem model.generate(prompt, max_new_tokens=32,
    do_sample=False): Wallclock der generate()-Zeit geteilt durch tatsaechlich
    neu generierte Tokens (aus der Output-Laenge, nicht aus max_new_tokens).
    Timeout-Budget 10 Minuten pro generate-Lauf (StoppingCriteria auf
    Wallclock); bei Timeout wird tokens_per_s: null mit timeout: true geloggt.

VRAM-Peak
    torch.cuda.max_memory_allocated(), resettet via reset_peak_memory_stats()
    direkt vor der Messphase (nicht vor dem Laden — die Ladephase wird separat
    als vram_peak_load_mb erfasst).

Offload-Wirksamkeit (offload_frees_mb)
    torch.cuda.memory_allocated() unmittelbar vor und nach einem erzwungenen
    _offload(layer) eines ON_GPU-Layers (mit synchronize dazwischen).
    Die Differenz muss ungefaehr der Layer-GPU-Groesse entsprechen. Das ist
    DIE Kennzahl, die belegt, ob Offloading real stattfindet.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Repo-Root fuer `import k4n0n3`

RESULTS_DIR = Path(__file__).parent / "results"
PROMPT = "The quick brown fox jumps over the lazy dog. Explain why this sentence is famous."


# -- helpers -----------------------------------------------------------------


def _git_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"], cwd=Path(__file__).parent
        ).returncode != 0
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _tensor_bytes(obj) -> int:
    """Bytes eines Master-Eintrags — plain Tensor oder Quant-Dict {q, scale}."""
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v) for v in obj.values() if isinstance(v, torch.Tensor))
    return 0


def _master_ram_mb(lm) -> float:
    total = 0
    for d in lm._cpu_master.values():
        for entry in d.values():
            total += _tensor_bytes(entry)
    return total / 1024**2


def _sync():
    torch.cuda.synchronize()


def _timed_forward(zfm, input_ids) -> float:
    _sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        zfm.model(input_ids=input_ids)
    _sync()
    return (time.perf_counter() - t0) * 1000


def measure_offload_frees_mb(lm) -> dict:
    """Erzwungenes _offload eines ON_GPU-Layers, memory_allocated-Differenz."""
    from k4n0n3.hooks import LayerState

    candidates = [n for n, i in lm._layer_info.items() if i.state == LayerState.ON_GPU]
    if not candidates:
        return {"offload_frees_mb": None, "offload_reason": "kein Layer ON_GPU"}
    name = candidates[0]
    expected_mb = lm._layer_info[name].size_mb
    _sync()
    before = torch.cuda.memory_allocated()
    lm._offload(name)
    _sync()
    after = torch.cuda.memory_allocated()
    return {
        "offload_frees_mb": (before - after) / 1024**2,
        "offload_layer": name,
        "offload_expected_mb": expected_mb,
    }


class _WallclockBudget:
    """StoppingCriteria: bricht generate() nach budget_s Wallclock ab."""

    def __init__(self, budget_s: float):
        self.deadline = time.perf_counter() + budget_s
        self.fired = False

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if time.perf_counter() > self.deadline:
            self.fired = True
            return True
        return False


def measure_generate(zfm, max_new_tokens: int, timeout_s: float) -> dict:
    from transformers import StoppingCriteriaList

    budget = _WallclockBudget(timeout_s)
    inputs = zfm.tokenizer(PROMPT, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    n_prompt = input_ids.shape[1]
    zfm.prepare()
    _sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = zfm.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            stopping_criteria=StoppingCriteriaList([budget]),
            pad_token_id=zfm.tokenizer.pad_token_id or zfm.tokenizer.eos_token_id,
        )
    _sync()
    wallclock = time.perf_counter() - t0
    new_tokens = out.shape[1] - n_prompt
    greedy = out[0, n_prompt:n_prompt + 32].tolist()
    return {
        "generate_tokens": new_tokens,
        "generate_wallclock_s": wallclock,
        "tokens_per_s": None if budget.fired else (new_tokens / wallclock if wallclock > 0 else None),
        "timeout": budget.fired,
        "greedy_tokens": greedy,
    }


# -- Messreihe (K1: immer dieselbe Reihenfolge) --------------------------------


def run(config: dict) -> dict:
    from k4n0n3 import ZeroFlushModel

    if not torch.cuda.is_available():
        raise SystemExit("Harness braucht CUDA/ROCm — nicht messbar ohne GPU.")

    result: dict = {
        "config": config,
        "git_commit": _git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "rocm_version": getattr(torch.version, "hip", None),
    }
    import transformers
    result["transformers_version"] = transformers.__version__

    # Phase 0: Laden (VRAM separat erfasst)
    torch.cuda.reset_peak_memory_stats()
    t_load = time.perf_counter()
    zfm_kwargs = dict(
        vram_budget_mb=config["vram_budget_mb"],
        prefetch_depth=config["prefetch_depth"],
        pin_ram_fraction=config["pin_ram_fraction"],
        verbose=config.get("verbose", False),
    )
    if config.get("quantize_transfer"):
        zfm_kwargs["quantize_transfer"] = config["quantize_transfer"]
        if config.get("int4_group_size"):
            zfm_kwargs["int4_group_size"] = config["int4_group_size"]
    zfm = ZeroFlushModel(config["model"], **zfm_kwargs)
    result["load_s"] = time.perf_counter() - t_load

    lm = zfm.layer_manager
    n_pinned = sum(1 for v in lm._pinned.values() if v)
    result["pinned_layers"] = f"{n_pinned}/{len(lm._layer_list)}"
    result["master_ram_mb"] = round(_master_ram_mb(lm), 1)

    inputs = zfm.tokenizer(PROMPT, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")

    # Phase 1: prepare + Cold Forward (frischer Prozess vorausgesetzt)
    zfm.prepare()
    result["vram_peak_load_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    torch.cuda.reset_peak_memory_stats()

    result["cold_forward_ms"] = round(_timed_forward(zfm, input_ids), 1)

    # Phase 2: 2 Discard-Warmups + 5 Warm Forwards
    for _ in range(2):
        _timed_forward(zfm, input_ids)
    warm = [round(_timed_forward(zfm, input_ids), 1) for _ in range(5)]
    result["warm_forward_ms"] = {"median": statistics.median(warm), "values": warm}

    # Phase 3: Offload-Wirksamkeit
    result.update(measure_offload_frees_mb(lm))
    if result.get("offload_frees_mb") is not None:
        result["offload_frees_mb"] = round(result["offload_frees_mb"], 1)

    # Phase 4: echtes generate()
    try:
        result.update(measure_generate(zfm, config["max_new_tokens"], config["generate_timeout_s"]))
    except Exception as e:  # noqa: BLE001 — Messung darf fehlschlagen, Bericht sagt warum
        result.update({
            "generate_tokens": None, "generate_wallclock_s": None,
            "tokens_per_s": None, "timeout": False,
            "generate_error": f"{type(e).__name__}: {e}", "greedy_tokens": None,
        })

    # Phase 5: VRAM-Peak der Messphase
    result["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    return result


def config_slug(config: dict) -> str:
    model = config["model"].rsplit("/", 1)[-1].lower()
    qt = config.get("quantize_transfer")
    if not qt:
        quant = "fp16"
    elif qt == "int4":
        quant = f"int4g{config.get('int4_group_size', 128)}"
    else:
        quant = "int8custom"
    return (f"{model}_{quant}_pin{config['pin_ram_fraction']}"
            f"_bud{config['vram_budget_mb']}_pf{config['prefetch_depth']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--vram-budget-mb", type=int, default=4096)
    p.add_argument("--prefetch-depth", type=int, default=1)
    p.add_argument("--pin-ram-fraction", type=float, default=0.7)
    p.add_argument("--quantize-transfer", nargs="?", const="int8",
                   choices=["int8", "int4"], default=False,
                   help="Custom weight-only Quant-Master + On-GPU-Dequant "
                        "(Auftrag M; 'int4' = group-wise gepackt, Auftrag P)")
    p.add_argument("--int4-group-size", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--generate-timeout-s", type=float, default=600.0)
    p.add_argument("--tag", default="", help="Zusatz-Suffix fuer den Dateinamen")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    config = {
        "model": args.model,
        "vram_budget_mb": args.vram_budget_mb,
        "prefetch_depth": args.prefetch_depth,
        "pin_ram_fraction": args.pin_ram_fraction,
        "quantize_transfer": args.quantize_transfer,
        "int4_group_size": args.int4_group_size,
        "max_new_tokens": args.max_new_tokens,
        "generate_timeout_s": args.generate_timeout_s,
        "verbose": args.verbose,
    }

    result = run(config)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = config_slug(config) + (f"_{args.tag}" if args.tag else "")
    out_path = RESULTS_DIR / f"{ts}_{slug}.json"
    out_path.write_text(json.dumps(result, indent=2))

    wf = result["warm_forward_ms"]
    print("\n" + "=" * 62)
    print(f"  {slug}")
    print("=" * 62)
    print(f"  pinned_layers     {result['pinned_layers']}")
    print(f"  master_ram_mb     {result['master_ram_mb']}")
    print(f"  cold_forward_ms   {result['cold_forward_ms']}  (nicht mit Warm vergleichen)")
    print(f"  warm_forward_ms   {wf['median']}  {wf['values']}")
    print(f"  offload_frees_mb  {result.get('offload_frees_mb')} "
          f"(erwartet ~{result.get('offload_expected_mb', 0):.0f})")
    print(f"  generate          {result.get('generate_tokens')} tok in "
          f"{result.get('generate_wallclock_s') and round(result['generate_wallclock_s'], 1)}s"
          f" | tok/s={result.get('tokens_per_s') and round(result['tokens_per_s'], 3)}"
          f" | timeout={result.get('timeout')}")
    if result.get("generate_error"):
        print(f"  generate_error    {result['generate_error']}")
    print(f"  vram_peak_mb      {result['vram_peak_mb']} (Load: {result['vram_peak_load_mb']})")
    print(f"  JSON: {out_path}")


if __name__ == "__main__":
    main()
