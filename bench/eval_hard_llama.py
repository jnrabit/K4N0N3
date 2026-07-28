"""Harter Rewrite-Eval gegen einen llama-server (llama.cpp) — MoE-Platzierung.

Schwester von `eval_hard_ollama.py`: gleiche 24 Faelle, gleiche strict-Skala,
nur ein anderer Backend-Aufruf. Hier die OpenAI-kompatible API eines
`llama-server`, der die Experten-FFNs per `--override-tensor 'ffn_.*_exps.=CPU'`
im RAM haelt und Attention/Router/Shared/KV auf der GPU — das MoE-bewusste
Placement, das Ollamas statischer 68/32-Split nicht kann.

Eingefroren und unveraendert importiert aus `eval_hard`:
    load_cases · build_messages · score · self_test · HARD_DATA
und aus `eval_rewrite`: strip_think.

Zusatz gegenueber dem Ollama-Treiber: der Prompt-/Decode-Split kommt direkt aus
llama.cpps `timings`-Feld (prompt_n/ms, predicted_n/ms) ins JSON — der Beleg,
dass mit schnellem MoE-Decode das Prefill zum Engpass wird.

  --self-test   Ideal 24/24, Dump/Distraktor 0/24 (CPU, kein Modell)
  --url         llama-server /v1/chat/completions (Default :8091)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.eval_hard import (  # noqa: E402
    HARD_DATA, build_messages, load_cases, score, self_test,
)
from bench.eval_rewrite import strip_think  # noqa: E402
from bench.harness import RESULTS_DIR, _git_commit  # noqa: E402


def chat(url: str, messages: list[dict], max_tokens: int,
         timeout: float) -> tuple[str, dict, float]:
    body = {"messages": messages, "temperature": 0.0,
            "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    dt = time.perf_counter() - t0
    content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content, payload.get("timings", {}), dt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8091/v1/chat/completions")
    p.add_argument("--data", default=str(HARD_DATA))
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    cases = load_cases(Path(args.data))
    if args.limit:
        cases = cases[:args.limit]
    print(f"{len(cases)} harte Faelle aus {args.data}")

    if args.self_test:
        self_test(cases)
        return

    # Warmup: mmap paged die CPU-Experten ein, sonst faelscht der erste Fall
    chat(args.url, build_messages(cases[0]), args.max_tokens, args.timeout)

    preds, lat, prompt_s, decode_s = [], [], [], []
    for c in cases:
        raw, tim, dt = chat(args.url, build_messages(c), args.max_tokens, args.timeout)
        preds.append(strip_think(raw).strip() or f"<LEER> {raw[:80]}")
        lat.append(dt)
        prompt_s.append(tim.get("prompt_ms", 0) / 1000)
        decode_s.append(tim.get("predicted_ms", 0) / 1000)
        print(f"  {c['id']:16} {c['follow_up'][:28]!r} → {preds[-1][:50]!r}  "
              f"({dt:.1f}s | pp {prompt_s[-1]:.1f} dec {decode_s[-1]:.1f})", flush=True)

    s = score(cases, preds)
    med = lambda xs: round(sorted(xs)[len(xs) // 2], 2)
    result = {
        "eval": True, "hard": True, "backend": "llama-server-vulkan-moe-cpu",
        "config": vars(args),
        "git_commit": _git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "latency_s": {"median": med(lat), "max": round(max(lat), 2),
                      "total": round(sum(lat), 2)},
        "timing_s": {"prompt_median": med(prompt_s), "decode_median": med(decode_s),
                     "prompt_total": round(sum(prompt_s), 2),
                     "decode_total": round(sum(decode_s), 2)},
        "runs": {"llama": s},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_eval_hard_llama{tag}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print(f"  llama-server MoE (Experten→CPU)")
    print(f"  strict {s['strict_pass']}/{s['n']} ({s['strict_rate']:.0%})  {s['per_kind']}")
    print(f"  Verletzungen: {s['violations']}")
    print(f"  Latenz/Rewrite: median {result['latency_s']['median']}s  "
          f"(Prompt {result['timing_s']['prompt_median']}s / "
          f"Decode {result['timing_s']['decode_median']}s)")
    print(f"  JSON: {out}")


if __name__ == "__main__":
    main()
