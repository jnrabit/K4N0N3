"""Auftrag L, Nachmessung: WER haelt nach dem Drop die GPU-Tensoren?

Der Hauptspike zeigte: Drop gibt ~0 MB frei, obwohl im Modul (Params, Buffer,
weight.CB/SCB) keine GPU-Tensoren mehr haengen. Verdacht: bitsandbytes'
Linear8bitLt.state (MatmulLtState) haelt CB/SCB ausserhalb von p.data.
Dieses Skript belegt das per Introspektion von module.state und per
globaler GPU-Tensor-Zaehlung (gc). Ergaenzt das bestehende Spike-JSON.
"""
from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.harness import PROMPT, RESULTS_DIR, _git_commit  # noqa: E402

MODEL = "Qwen/Qwen2.5-3B"


def log(msg: str) -> None:
    print(f"[L2] {msg}", flush=True)


def state_gpu_tensors(module: torch.nn.Module) -> list[str]:
    """GPU-Tensoren in bnb-`state`-Objekten (MatmulLtState) eines Layers."""
    found = []
    for sub, m in module.named_modules():
        state = getattr(m, "state", None)
        if state is None:
            continue
        for attr, val in vars(state).items():
            if isinstance(val, torch.Tensor) and val.device.type == "cuda":
                mb = val.numel() * val.element_size() / 1024**2
                found.append(f"{sub}.state.{attr}:{val.dtype}:{mb:.1f}MB")
    return found


def global_cuda_mb() -> float:
    total = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                total += obj.numel() * obj.element_size()
        except Exception:
            continue
    return total / 1024**2


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from k4n0n3 import LayerManager

    result: dict = {
        "spike": "L_bnb_residual_check",
        "git_commit": _git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL,
        "findings": [],
    }

    def finding(txt: str) -> None:
        log(txt)
        result["findings"].append(txt)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map={"": 0},
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    for lyr in model.model.layers:
        lyr.to("cpu")
    torch.cuda.synchronize()
    finding(f"nach Masse-.to(cpu): memory_allocated={torch.cuda.memory_allocated()/1024**2:.0f} MB")

    layer0 = model.model.layers[0]
    lm = LayerManager(model, layer_prefix="model.layers",
                      vram_budget_mb=149, prefetch_depth=1, pin_ram_fraction=0.0)
    lm.prepare()

    ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to("cuda")
    with torch.no_grad():
        model(input_ids=ids)
    torch.cuda.synchronize()
    alloc_after_fwd = torch.cuda.memory_allocated() / 1024**2
    finding(f"nach Druck-Forward: memory_allocated={alloc_after_fwd:.0f} MB "
            f"(Modell int8 ~3240 MB — bleibt alles liegen?)")
    result["allocated_after_pressure_fwd_mb"] = round(alloc_after_fwd, 1)

    # Layer 0 ist laut LayerManager laengst gedroppt — was haelt seine Tensoren?
    st = state_gpu_tensors(layer0)
    result["layer0_state_gpu_tensors"] = st
    finding(f"Layer 0 (gedroppt) — GPU-Tensoren in Linear8bitLt.state: {len(st)}")
    for s in st[:6]:
        finding(f"  {s}")

    result["gc_global_cuda_mb"] = round(global_cuda_mb(), 1)
    finding(f"global via gc erreichbare GPU-Tensoren: {result['gc_global_cuda_mb']:.0f} MB")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"{ts}_l_spike_bnb_residual.json"
    out.write_text(json.dumps(result, indent=2))
    log(f"JSON: {out}")


if __name__ == "__main__":
    main()
