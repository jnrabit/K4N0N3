"""Auftrag 6 T2 — Transfer-Mikrobenchmark: Roh vs. Ist vs. Fragmentierung.

Beantwortet die eine Frage, die entscheidet, ob Auftrag U (Staging-Blob) gebaut
wird: **verliert der echte K4N0N3-Layer-Upload gegen einen einzelnen großen
Blob — und wenn ja, an der Fragmentierung (viele kleine .to()-Calls)?**

Drei Messungen, jeweils Median aus `--reps` CUDA-Event-Zeiten:

1. Roh-Bandbreite   — ein gepinnter 150-MB-Blob → GPU. Referenzlinie der Maschine.
2. Ist-Zustand      — EIN echter 3B-int8/int4-Layer über `_upload_layer`
                      (Transfer + Dequant, „so wie es läuft") UND transfer-only
                      (nur die .to()-Kopien, ohne Dequant) — plus Copy-Zähler.
3. Fragmentierung   — dieselben Layer-Bytes einmal als N Einzeltensoren (echte
                      Param-Größen), einmal als ein Blob → isoliert den
                      Fragmentierungsanteil.

Gate: transfer-only Ist / Roh. ≥ 0,80 → Fragmentierung ist NICHT das Problem,
U wird übersprungen. Darunter → U begründet, Zielwert ist die Roh-Bandbreite.

Alle Zahlen ins JSON; keine von Hand. Bei Modellfehler: Roh trotzdem, Ist=null
mit Grund.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.harness import (  # noqa: E402
    RESULTS_DIR, _git_commit, _pcie_link, _tensor_bytes,
)
from k4n0n3.hooks import (  # noqa: E402
    _drop_layer, _packed_tensor, _upload_layer, reset_upload_copy_count,
)

GB = 1024 ** 3


def _median_gpu_ms(fn, reps: int, warmup: int = 3) -> float:
    """Median der reinen GPU-Zeit von fn() über CUDA-Events (ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def _gbs(nbytes: int, ms: float) -> float:
    return (nbytes / GB) / (ms / 1000) if ms > 0 else 0.0


def measure_roh(mb: int, reps: int) -> dict:
    n = mb * 1024 * 1024
    host = torch.empty(n, dtype=torch.uint8).pin_memory()
    dst = torch.empty_like(host, device="cuda")
    ms = _median_gpu_ms(lambda: dst.copy_(host, non_blocking=True), reps)
    return {"roh_mb": mb, "roh_ms": round(ms, 3), "roh_gbs": round(_gbs(n, ms), 3)}


def measure_multistream(mb: int, reps: int, max_streams: int = 4) -> dict:
    """Aggregierte H2D-Bandbreite über N gleichzeitige Streams. Steigt sie nicht
    mit N, ist die Roh-Rate die ECHTE Bus-/DMA-Decke, kein Pro-Queue-Limit —
    beantwortet die „wie hoch liegt die echte Decke"-Frage aus der Diagnose."""
    n = mb * 1024 * 1024
    out = {}
    for k in range(1, max_streams + 1):
        hosts = [torch.empty(n, dtype=torch.uint8).pin_memory() for _ in range(k)]
        gpus = [torch.empty(n, dtype=torch.uint8, device="cuda") for _ in range(k)]
        streams = [torch.cuda.Stream() for _ in range(k)]

        def once():
            for st, h, g in zip(streams, hosts, gpus):
                with torch.cuda.stream(st):
                    g.copy_(h, non_blocking=True)
            torch.cuda.synchronize()

        ms = _median_gpu_ms(once, reps)
        out[str(k)] = round(_gbs(k * n, ms), 3)
    return {"multistream_gbs": out, "multistream_ceiling_gbs": max(out.values())}


def measure_layer(model: str, quant: str, reps: int) -> dict:
    """Ein echter Layer über den bestehenden Upload-Pfad."""
    from k4n0n3 import ZeroFlushModel

    zfm = ZeroFlushModel(model, vram_budget_mb=2048, prefetch_depth=1,
                         pin_ram_fraction=0.7, quantize_transfer=quant)
    lm = zfm.layer_manager
    names = lm._layer_list
    target = names[len(names) // 2]  # ein mittlerer Layer, repräsentativ
    mod = lm._layers[target]
    master = lm._cpu_master[target]
    refs = lm._param_refs.get(target)

    int_bytes = sum(_tensor_bytes(entry) for entry in master.values())
    fp16_bytes = lm._layer_gpu_bytes.get(target, 0)

    # -- Ist voll: Transfer + Dequant über _upload_layer, Drop dazwischen (ungezählt)
    def _drop():
        _drop_layer(mod, master, refs)
        torch.cuda.synchronize()

    full_times = []
    copies = 0
    for i in range(reps + 3):
        _drop()
        reset_upload_copy_count()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        _upload_layer(mod, master, refs)
        e.record()
        torch.cuda.synchronize()
        if i >= 3:
            full_times.append(s.elapsed_time(e))
        copies = reset_upload_copy_count()
    _drop()
    full_ms = statistics.median(full_times)

    # -- Ist transfer-only: dieselben Master-Tensoren kopieren, KEIN Dequant
    tensors = []
    for entry in master.values():
        if isinstance(entry, dict):
            tensors.append(_packed_tensor(entry))
            tensors.append(entry["scale"])
        elif isinstance(entry, torch.Tensor):
            tensors.append(entry)
    gpu_bufs = [torch.empty_like(t, device="cuda") for t in tensors]

    def _transfer_n():
        for t, g in zip(tensors, gpu_bufs):
            g.copy_(t, non_blocking=True)

    transfer_ms = _median_gpu_ms(_transfer_n, reps)

    # -- Fragmentierungs-Sim: dieselben Bytes als EIN Blob
    big = torch.empty(int_bytes, dtype=torch.uint8).pin_memory()
    big_gpu = torch.empty_like(big, device="cuda")
    one_ms = _median_gpu_ms(lambda: big_gpu.copy_(big, non_blocking=True), reps)

    return {
        "layer": target,
        "quant": quant,
        "n_tensors": len(tensors),
        "upload_copies": copies,
        "int_bytes": int_bytes,
        "int_mb": round(int_bytes / 1024 ** 2, 1),
        "fp16_bytes": fp16_bytes,
        "ist_full_ms": round(full_ms, 3),
        "ist_full_gbs": round(_gbs(int_bytes, full_ms), 3),
        "ist_transfer_ms": round(transfer_ms, 3),
        "ist_transfer_gbs": round(_gbs(int_bytes, transfer_ms), 3),
        "frag_one_blob_ms": round(one_ms, 3),
        "frag_one_blob_gbs": round(_gbs(int_bytes, one_ms), 3),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--quant", choices=["int8", "int4"], default="int8")
    p.add_argument("--roh-mb", type=int, default=150)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--tag", default="")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("pcie_probe braucht CUDA/ROCm — nicht messbar ohne GPU.")

    result = {
        "probe": True, "auftrag": "6T",
        "config": vars(args),
        "git_commit": _git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    result.update(_pcie_link())  # T1-Felder auch hier

    roh = measure_roh(args.roh_mb, args.reps)
    result.update(roh)
    result.update(measure_multistream(128, args.reps))

    try:
        layer = measure_layer(args.model, args.quant, args.reps)
        result.update(layer)
        result["layer_error"] = None
    except Exception as e:  # noqa: BLE001 — Messung darf fehlschlagen, JSON sagt warum
        result.update({"ist_transfer_gbs": None, "ist_full_gbs": None,
                       "frag_one_blob_gbs": None, "upload_copies": None})
        result["layer_error"] = f"{type(e).__name__}: {e}"

    # -- Gate: transfer-only Ist / Roh (apples-to-apples, reiner Transfer) -----
    roh_gbs = result["roh_gbs"]
    ist_t = result.get("ist_transfer_gbs")
    frag_one = result.get("frag_one_blob_gbs")
    if ist_t and roh_gbs:
        ratio = ist_t / roh_gbs
        frag_factor = (ist_t / frag_one) if frag_one else None
        result["gate_ratio_ist_transfer_over_roh"] = round(ratio, 3)
        result["gate_frag_factor_nsmall_over_oneblob"] = round(frag_factor, 3) if frag_factor else None
        result["gate_open_U_begruendet"] = bool(ratio < 0.80)
        verdikt = (f"Gate {'OFFEN → U begründet' if ratio < 0.80 else 'ZU → U ÜBERSPRINGEN'}: "
                   f"transfer-only Ist {ist_t} / Roh {roh_gbs} = {ratio:.2f} "
                   f"(≥0,80 = Fragmentierung nicht das Problem)")
    else:
        result["gate_open_U_begruendet"] = None
        verdikt = "Gate nicht bestimmbar (Ist nicht gemessen)."
    result["verdikt"] = verdikt

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_pcie_probe_{args.quant}{tag}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 62)
    print(f"  PCIe-Probe ({args.quant}), Link {result.get('pcie_link_speed')} "
          f"/ x{result.get('pcie_link_width')}")
    print("=" * 62)
    print(f"  Roh (1 Blob {args.roh_mb}MB)     {roh_gbs} GB/s")
    print(f"  Multistream-Decke      {result['multistream_ceiling_gbs']} GB/s "
          f"{result['multistream_gbs']}")
    if result.get("layer_error"):
        print(f"  Ist                    FEHLER: {result['layer_error']}")
    else:
        print(f"  Layer {result['layer']}  ({result['int_mb']} MB int, "
              f"{result['n_tensors']} Tensoren, {result['upload_copies']} Copies)")
        print(f"  Ist transfer-only      {result['ist_transfer_gbs']} GB/s")
        print(f"  Ist voll (+ Dequant)   {result['ist_full_gbs']} GB/s")
        print(f"  Frag: 1 Blob gleicher Bytes  {result['frag_one_blob_gbs']} GB/s")
    print(f"\n  {verdikt}")
    print(f"  JSON: {out}")


if __name__ == "__main__":
    main()
