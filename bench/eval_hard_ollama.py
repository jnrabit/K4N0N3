"""Harter Rewrite-Eval gegen ein Ollama-Modell — gleiche Faelle, gleiche Skala.

Warum ein eigener Treiber: `eval_hard.py` laedt das Modell ueber transformers +
K4N0N3-Offload. Ein 30B-MoE ist als fp16-CPU-Master ~60 GB und passt nicht in
32 GB RAM; ueber Ollama (q4 ≈ 18 GB, 3B aktiv/Token) laeuft es dagegen. Getauscht
wird deshalb NUR der Generierungs-Backend.

Eingefroren und unveraendert importiert aus `eval_hard`:
    load_cases · build_messages · score · self_test · HARD_DATA
und aus `eval_rewrite`: strip_think · prompt-freie Think-Erkennung.

Damit ist die Messgroesse (24 Faelle + strict-Bewertung) byte-identisch zum
transformers-Lauf — nur der Modellaufruf ist neu. `--self-test` ruft dieselbe
Kalibrierung wie `eval_hard`, als Beleg, dass die importierte Skala greift.

  --self-test          Ideal 24/24, Dump/Distraktor 0/24 (CPU, kein Modell)
  --model NAME         Ollama-Tag (Default qwen3:30b-a3b-instruct-2507)
  --think              den Denkmodus NICHT unterdruecken (Default: aus)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.eval_hard import (  # noqa: E402
    HARD_DATA, build_messages, load_cases, score, self_test,
)
from bench.eval_rewrite import strip_think  # noqa: E402
from bench.harness import RESULTS_DIR, _git_commit  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/chat"


def ollama_chat(model: str, messages: list[dict], num_predict: int,
                think: bool, timeout: float) -> tuple[str, float]:
    """Ein Chat-Turn, greedy. → (content, wallclock_s). think=False unterdrueckt
    den Denkmodus bei Hybrid-Modellen; bei reinen Instruct-Modellen ist der
    Schluessel wirkungslos, schadet aber nicht."""
    body: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": num_predict},
    }
    if not think:
        body["think"] = False
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    dt = time.perf_counter() - t0
    return payload.get("message", {}).get("content", ""), dt


def generate(model: str, case: dict, num_predict: int, think: bool,
             timeout: float) -> tuple[str, float]:
    raw, dt = ollama_chat(model, build_messages(case), num_predict, think, timeout)
    if think and "<think>" in raw and "</think>" not in raw:
        return f"<UNVOLLSTAENDIG> {raw[:100]}", dt
    return (strip_think(raw).strip() or f"<LEER> {raw[:100]}"), dt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3:30b-a3b-instruct-2507")
    p.add_argument("--data", default=str(HARD_DATA))
    p.add_argument("--num-predict", type=int, default=64)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--think", action="store_true")
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

    preds: list[str] = []
    latencies: list[float] = []
    for c in cases:
        pred, dt = generate(args.model, c, args.num_predict, args.think, args.timeout)
        preds.append(pred)
        latencies.append(dt)
        print(f"  {c['id']:16} {c['follow_up'][:30]!r} → {pred[:60]!r}  ({dt:.1f}s)",
              flush=True)

    s = score(cases, preds)
    lat_sorted = sorted(latencies)
    result = {
        "eval": True, "hard": True, "backend": "ollama",
        "config": vars(args),
        "git_commit": _git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "latency_s": {
            "median": round(lat_sorted[len(lat_sorted) // 2], 2),
            "max": round(max(latencies), 2),
            "total": round(sum(latencies), 2),
        },
        "runs": {"ollama": s},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_eval_hard_ollama{tag}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print(f"  {args.model}")
    print(f"  strict {s['strict_pass']}/{s['n']} ({s['strict_rate']:.0%})  {s['per_kind']}")
    print(f"  Verletzungen: {s['violations']}")
    print(f"  Latenz/Rewrite: median {result['latency_s']['median']}s  "
          f"max {result['latency_s']['max']}s")
    print(f"  JSON: {out}")


if __name__ == "__main__":
    main()
