"""Auftrag L — bitsandbytes int8 unter echtem Offload-Druck (Spike, timeboxed).

Frage aus Umbau 2: funktioniert die Master/Drop-Mechanik mit
bitsandbytes-Int8Params, wenn Eviction wirklich stattfindet?
Erwartung: vermutlich nein. Ein sauberes "inkompatibel, weil X" ist ein
volles Ergebnis. Keine Monkey-Patches in bitsandbytes.

Ablauf:
  Phase A  int8-Modell voll auf GPU (ohne Hooks) → Referenz-greedy_tokens
  Phase B  L1: Layer als CPU-Master herstellen — geht das ueberhaupt?
           (Log: dtype/device von weight und SCB vor/nach .cpu())
  Phase C  LayerManager mit Budget = 2 × Layer-GPU-Groesse, prefetch_depth=1
           → Eviction bei jedem Schritt. Drei Messpunkte (L2):
           1. offload_frees_mb — gibt der Drop wirklich VRAM frei?
           2. Re-Upload nach Eviction — rechnet der Layer korrekt weiter?
           3. greedy_tokens vs. Phase-A-Referenz — identisch?

Output: JSON nach bench/results/<ts>_l_spike_bnb.json + Konsolen-Verdikt.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.harness import PROMPT, RESULTS_DIR, _git_commit, measure_offload_frees_mb  # noqa: E402

MODEL = "Qwen/Qwen2.5-3B"


def log(msg: str) -> None:
    print(f"[L] {msg}", flush=True)


def gpu_tensor_report(module: torch.nn.Module) -> dict:
    """Zaehlt GPU-residente Tensoren eines Moduls inkl. Nicht-Standard-Attribute (SCB etc.)."""
    on_gpu, bytes_gpu = [], 0
    for pname, p in module.named_parameters():
        if p.device.type == "cuda":
            on_gpu.append(f"param:{pname}:{p.dtype}")
            bytes_gpu += p.numel() * p.element_size()
        # bitsandbytes haengt Scales als Attribut an den Param (nicht als Buffer)
        for attr in ("SCB", "CB"):
            t = getattr(p, attr, None)
            if isinstance(t, torch.Tensor) and t.device.type == "cuda":
                on_gpu.append(f"attr:{pname}.{attr}:{t.dtype}")
                bytes_gpu += t.numel() * t.element_size()
    for bname, b in module.named_buffers():
        if b.device.type == "cuda":
            on_gpu.append(f"buffer:{bname}:{b.dtype}")
            bytes_gpu += b.numel() * b.element_size()
    return {"gpu_tensors": on_gpu, "gpu_mb": round(bytes_gpu / 1024**2, 1)}


def greedy_32(model, tokenizer) -> list[int]:
    inputs = tokenizer(PROMPT, return_tensors="pt")
    ids = inputs["input_ids"].to("cuda")
    mask = inputs["attention_mask"].to("cuda")
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=32,
                             do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return out[0, ids.shape[1]:].tolist()


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    result: dict = {
        "spike": "L_bnb_offload_pressure",
        "git_commit": _git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL,
        "verdict": None,
        "findings": [],
    }

    def finding(txt: str) -> None:
        log(txt)
        result["findings"].append(txt)

    def finish(verdict: str) -> None:
        result["verdict"] = verdict
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = RESULTS_DIR / f"{ts}_l_spike_bnb.json"
        out.write_text(json.dumps(result, indent=2))
        print(f"\n[L] VERDIKT: {verdict}\n[L] JSON: {out}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # -- Phase A: int8 voll auf GPU, Referenz -------------------------------
    log("Phase A: lade int8 voll auf GPU (Referenz ohne Hooks)")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map={"": 0},
        )
    except Exception as e:
        finding(f"int8-Laden fehlgeschlagen: {type(e).__name__}: {e}")
        finish("nicht messbar — bitsandbytes-Laden auf ROCm fehlgeschlagen")
        return
    model.eval()

    layer0 = model.model.layers[0]
    rep = gpu_tensor_report(layer0)
    finding(f"Layer 0 nach Laden: {rep['gpu_mb']} MB GPU, "
            f"{len(rep['gpu_tensors'])} GPU-Tensoren")
    result["layer0_after_load"] = rep

    result["greedy_reference"] = greedy_32(model, tokenizer)
    log(f"Referenz-greedy (erste 8): {result['greedy_reference'][:8]}")

    # -- Phase B / L1: CPU-Master herstellbar? ------------------------------
    log("Phase B: Layer .to('cpu') — bleiben die Weights int8 + SCB erhalten?")
    w = layer0.self_attn.q_proj.weight
    before = {"dtype": str(w.dtype), "device": str(w.device),
              "scb": str(getattr(w, "SCB", None) is not None)}
    try:
        layer0.to("cpu")
    except Exception as e:
        finding(f".to('cpu') wirft: {type(e).__name__}: {e}")
        finish("inkompatibel — Layer lassen sich nicht als CPU-Master ablegen")
        return
    w = layer0.self_attn.q_proj.weight
    scb = getattr(w, "SCB", None)
    after = {"dtype": str(w.dtype), "device": str(w.device),
             "scb_device": str(scb.device) if isinstance(scb, torch.Tensor) else "None"}
    result["l1_weight_before"] = before
    result["l1_weight_after"] = after
    finding(f"L1 q_proj.weight: {before} → {after}")

    if w.dtype != torch.int8:
        finding("Weights liegen nach .cpu() NICHT als int8 vor (dequantisiert oder fp) "
                "— CPU-Master im int8-Format nicht herstellbar.")
        finish("inkompatibel — bitsandbytes haelt Weights nicht als int8-CPU-Master")
        return
    layer0.to("cuda")  # zurueck fuer konsistenten Ausgangszustand

    # Alle Layer auf CPU als Master
    log("Alle 36 Layer nach CPU (Master-Zustand)...")
    try:
        for lyr in model.model.layers:
            lyr.to("cpu")
    except Exception as e:
        finding(f"Massen-.to('cpu') wirft: {type(e).__name__}: {e}")
        finish("inkompatibel — CPU-Masterisierung schlaegt fehl")
        return

    # -- Phase C: LayerManager unter Druck ----------------------------------
    from k4n0n3 import LayerManager

    layer_mb = sum(p.numel() * p.element_size() for p in layer0.parameters()) / 1024**2
    budget = int(layer_mb * 2) + 1
    log(f"Phase C: LayerManager budget={budget} MB (2 × {layer_mb:.0f} MB), prefetch_depth=1")
    manager = LayerManager(model, layer_prefix="model.layers",
                           vram_budget_mb=budget, prefetch_depth=1,
                           pin_ram_fraction=0.0)  # pageable reicht fuer den Spike
    manager.prepare()

    inputs = tokenizer(PROMPT, return_tensors="pt")
    ids = inputs["input_ids"].to("cuda")

    # Messpunkt 1: offload_frees_mb
    try:
        with torch.no_grad():
            model(input_ids=ids)
    except Exception as e:
        finding(f"Forward unter Druck wirft: {type(e).__name__}: {e}")
        finish("inkompatibel — Forward mit Eviction schlaegt fehl")
        return
    off = measure_offload_frees_mb(manager)
    result.update({f"l2_{k}": v for k, v in off.items()})
    finding(f"Messpunkt 1 offload_frees_mb={off.get('offload_frees_mb')} "
            f"(erwartet ~{off.get('offload_expected_mb', 0):.0f})")
    if off.get("offload_frees_mb") is not None:
        dropped = manager._layers[off["offload_layer"]]
        rep = gpu_tensor_report(dropped)
        result["l2_dropped_layer_residual"] = rep
        finding(f"Nach Drop verbleiben auf GPU: {rep['gpu_mb']} MB "
                f"({len(rep['gpu_tensors'])} Tensoren)")

    # Messpunkt 2+3: Re-Upload + greedy unter Druck vs. Referenz
    t0 = time.perf_counter()
    try:
        result["greedy_pressure"] = greedy_32(model, tokenizer)
    except Exception as e:
        finding(f"generate unter Druck wirft: {type(e).__name__}: {e}")
        finish("inkompatibel — Re-Upload nach Eviction korrumpiert den Zustand")
        return
    result["greedy_pressure_wallclock_s"] = round(time.perf_counter() - t0, 1)
    match = result["greedy_pressure"] == result["greedy_reference"]
    result["l2_greedy_match"] = match
    finding(f"Messpunkt 3 greedy identisch mit Referenz: {match}")
    if not match:
        ref, got = result["greedy_reference"], result["greedy_pressure"]
        div = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), len(got))
        finding(f"Divergenz ab Token {div}: ref={ref[:div+3]} vs druck={got[:div+3]}")

    frees_ok = (off.get("offload_frees_mb") or 0) > 0.5 * off.get("offload_expected_mb", 1)
    if match and frees_ok:
        finish("kompatibel — Drop gibt VRAM frei, Re-Upload korrekt, greedy identisch")
    elif match:
        finish("teilkompatibel — Ergebnis korrekt, aber Drop gibt VRAM nicht (voll) frei")
    else:
        finish("inkompatibel — Mechanik korrumpiert Int8Params-Zustand")


if __name__ == "__main__":
    main()
