"""Referenzzeile ausser Konkurrenz (Auftrag N2): GGUF q4 via Ollama/llama.cpp-ROCm.

Stoppuhr-Skript: `ollama run <model> --verbose` mit demselben festen Prompt wie
das Harness; Tokens und Zeit stammen aus der Ollama---verbose-Ausgabe
(eval count / eval duration), nicht aus einer Schaetzung. Ergebnis als JSON
nach bench/results/, damit make_report.py es wie jeden anderen Lauf liest.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.harness import PROMPT, RESULTS_DIR, _git_commit  # noqa: E402


def parse_verbose(stderr: str) -> dict:
    """Parst die --verbose-Statistik von ollama run."""
    out: dict = {}
    patterns = {
        "prompt_eval_count": r"^prompt eval count:\s+(\d+)",
        "eval_count": r"^eval count:\s+(\d+)",
        "eval_duration_s": r"^eval duration:\s+([\d.]+)(m?s)",
        "total_duration_s": r"^total duration:\s+([\d.]+)(m?s|s)",
        "eval_rate_tok_s": r"^eval rate:\s+([\d.]+) tokens/s",
    }
    for key, pat in patterns.items():
        # ^-Anker zwingend: "prompt eval rate" enthaelt "eval rate" als Substring
        m = re.search(pat, stderr, re.MULTILINE)
        if not m:
            continue
        val = float(m.group(1))
        if len(m.groups()) > 1 and m.group(2) == "ms":
            val /= 1000.0
        out[key] = val
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5:3b", help="Ollama-Modellname (GGUF q4)")
    args = p.parse_args()

    proc = subprocess.run(
        ["ollama", "run", args.model, "--verbose", PROMPT],
        capture_output=True, text=True, timeout=900,
    )
    stats = parse_verbose(proc.stderr)
    if not stats.get("eval_count"):
        print(proc.stderr[-2000:])
        raise SystemExit(f"Konnte Ollama-Statistik nicht parsen (rc={proc.returncode})")

    tokens_per_s = stats.get("eval_rate_tok_s")
    if tokens_per_s is None and stats.get("eval_duration_s"):
        tokens_per_s = stats["eval_count"] / stats["eval_duration_s"]

    result = {
        "config": {"model": args.model, "runner": "ollama", "quant": "gguf-q4",
                   "prompt": PROMPT},
        "git_commit": _git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "generate_tokens": int(stats["eval_count"]),
        "tokens_per_s": round(tokens_per_s, 2) if tokens_per_s else None,
        "ollama_stats": stats,
        "reference_only": True,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"{ts}_ollama_{args.model.replace(':', '-')}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"eval_count={result['generate_tokens']} tok/s={result['tokens_per_s']}")
    print(f"JSON: {out}")


if __name__ == "__main__":
    main()
