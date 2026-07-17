"""O3/P-Timing-Split: Wo geht die Zeit hin — H2D-Copy oder Unpack/Dequant?

Misst fuer die ersten Layer eines quantisierten Modells getrennt (jeweils
mit synchronize-Grenzen, Median aus mehreren Wiederholungen):

  copy_ms    nur die H2D-Kopien der Quant-Master (q/scale, non_blocking + sync)
  dequant_ms nur der On-GPU-Dequant (int8-Mul bzw. int4-Unpack+Mul)

Output: JSON nach bench/results/<ts>_timing_split_<quant>.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.harness import RESULTS_DIR, _git_commit  # noqa: E402


def main() -> None:
    from k4n0n3 import ZeroFlushModel
    from k4n0n3.hooks import _packed_tensor, dequantize_int4, dequantize_int8

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--quant", choices=["int8", "int4"], required=True)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--reps", type=int, default=10)
    args = p.parse_args()

    zfm = ZeroFlushModel(args.model, quantize_transfer=args.quant)
    lm = zfm.layer_manager

    per_layer = []
    for name in lm._layer_list[: args.layers]:
        master = lm._cpu_master[name]
        quant_entries = [e for e in master.values() if isinstance(e, dict)]
        copies, dequants = [], []
        for _ in range(args.reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            staged = []
            for e in quant_entries:
                staged.append((
                    e,
                    _packed_tensor(e).to("cuda", non_blocking=True),
                    e["scale"].to("cuda", non_blocking=True),
                ))
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            for e, q_gpu, s_gpu in staged:
                if "q4" in e:
                    dequantize_int4(q_gpu, s_gpu, e.get("meta")) if "meta" in e \
                        else dequantize_int4(q_gpu, s_gpu)
                else:
                    dequantize_int8(q_gpu, s_gpu)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            copies.append((t1 - t0) * 1000)
            dequants.append((t2 - t1) * 1000)
        per_layer.append({
            "layer": name,
            "pinned": lm._pinned.get(name, False),
            "copy_ms": round(statistics.median(copies), 2),
            "dequant_ms": round(statistics.median(dequants), 2),
        })
        print(f"{name}: copy {per_layer[-1]['copy_ms']} ms | "
              f"dequant {per_layer[-1]['dequant_ms']} ms | pinned={per_layer[-1]['pinned']}")

    result = {
        "timing_split": True,
        "quant": args.quant,
        "model": args.model,
        "git_commit": _git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "reps": args.reps,
        "per_layer": per_layer,
        "copy_ms_median": round(statistics.median([l["copy_ms"] for l in per_layer]), 2),
        "dequant_ms_median": round(statistics.median([l["dequant_ms"] for l in per_layer]), 2),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"{ts}_timing_split_{args.quant}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"JSON: {out}")


if __name__ == "__main__":
    main()
