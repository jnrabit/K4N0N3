"""Probe-Suite (T1–T7) für den Basismodell-Entscheid — ein Treiber, viele Modelle.

Generalisierung von `run_qwythos_test.py`: Modell als Parameter, Config
(seed/num_ctx/temp) zentral, Modell-Metadaten (Quant + Digest + arch) ins JSON,
Ausgabe ins `bench/results`-Schema.

Harte Regel des Auftrags: alle Kandidaten über DENSELBEN Treiber, dieselbe
Quant-Klasse, denselben Seed. Zahlen aus unterschiedlichen Setups werden nicht
gemischt (17-vs-13-Befund).

Thinking: **AN** (Default des Modells) — in dieser Suite wird Think-Ökonomie ja
gerade gemessen. Timeout je Probe (Default 300 s) ist ein **Messwert**
("Think nicht abgeschlossen"), kein Crash: der Fall wird mit
`timeout: true` geloggt und die Suite läuft weiter.

T4/T5 nutzen das native `tools`-Feld der Ollama-API (nicht Tool-Text im
System-Prompt — Lektion aus dem ersten Fehllauf); Auswertung über
`message.tool_calls`, nicht per Regex.

T1/T2/T6 brauchen zusätzlich ein menschliches/rubrik-basiertes Urteil; das Feld
`manual_verdict` bleibt dafür leer im JSON stehen.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.harness import RESULTS_DIR, _git_commit  # noqa: E402

# Prompts/Tools unverändert aus run_qwythos_test.py übernommen — derselbe Reiz
# für alle Kandidaten, sonst ist der Vergleich wertlos.
from run_qwythos_test import (  # noqa: E402
    NUM_CTX, SEED, TESTS, THINK_RE, TOOLCALL_RE, auto_checks,
)

MANUAL_IDS = ("T1", "T2", "T6")


def model_meta(url: str, model: str) -> dict:
    """Quant, arch, Parameter, Digest — gehört ins JSON, damit später niemand
    Zahlen aus verschiedenen Quant-Klassen mischt."""
    try:
        req = urllib.request.Request(
            f"{url}/api/show", data=json.dumps({"model": model}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            d = json.loads(resp.read())
        det = d.get("details", {})
        info = d.get("model_info", {})
        return {
            "quantization": det.get("quantization_level"),
            "family": det.get("family"),
            "parameter_size": det.get("parameter_size"),
            "architecture": info.get("general.architecture"),
            "context_length": next((v for k, v in info.items()
                                    if k.endswith(".context_length")), None),
            "capabilities": d.get("capabilities"),
        }
    except Exception as e:  # noqa: BLE001
        return {"meta_error": f"{type(e).__name__}: {e}"}


def model_digest(url: str, model: str) -> str | None:
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=60) as resp:
            for m in json.loads(resp.read()).get("models", []):
                if m.get("name") == model or m.get("model") == model:
                    return m.get("digest")
    except Exception:  # noqa: BLE001
        pass
    return None


def call_ollama(url: str, model: str, system: str | None, prompt: str,
                tools: list | None, seed: int, num_ctx: int,
                timeout_s: float) -> dict:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model, "messages": messages, "stream": True,
        "options": {"num_ctx": num_ctx, "temperature": 0.0, "seed": seed},
    }
    if tools:
        payload["tools"] = tools
    req = urllib.request.Request(
        f"{url}/api/chat", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})

    t0 = time.perf_counter()
    content, think = "", ""
    tool_calls: list[dict] = []
    eval_count, eval_dur_s, n_chunks = 0, 0.0, 0
    timed_out = False
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for line in resp:
            if time.perf_counter() - t0 > timeout_s:
                timed_out = True   # Messwert, kein Fehler
                break
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            msg = chunk.get("message", {})
            content += msg.get("content", "") or ""
            think += msg.get("thinking", "") or ""
            if msg.get("tool_calls"):
                tool_calls.extend(msg["tool_calls"])
            n_chunks += 1
            # Fortschritt nur im Terminal — in Logs/Pipes wuerde \r zu
            # tausenden Zeilen aufblaehen.
            if n_chunks % 8 == 0 and sys.stdout.isatty():
                phase = "think" if (think and not content) else "answer"
                sys.stdout.write(f"\r     …{phase} {len(think)+len(content):>6} chars | "
                                 f"{time.perf_counter()-t0:>4.0f}s")
                sys.stdout.flush()
            if chunk.get("done"):
                eval_count = chunk.get("eval_count", 0)
                if chunk.get("eval_duration"):
                    eval_dur_s = chunk["eval_duration"] / 1e9
    if sys.stdout.isatty():
        sys.stdout.write("\r" + " " * 46 + "\r")
        sys.stdout.flush()
    wall = time.perf_counter() - t0

    if not think:
        m = THINK_RE.search(content)
        if m:
            think = m.group(1)
            content = THINK_RE.sub("", content).strip()
    if tools and not tool_calls:
        for raw in TOOLCALL_RE.findall(content):
            try:
                tool_calls.append({"function": json.loads(raw.strip())})
            except json.JSONDecodeError:
                tool_calls.append({"function": {"_raw": raw.strip()}})

    return {
        "answer": content.strip(), "think": think.strip(),
        "think_chars": len(think.strip()),
        "think_tokens_approx": max(1, len(think.split())) if think.strip() else 0,
        "eval_count": eval_count,
        "tok_per_s": round(eval_count / eval_dur_s, 2) if eval_dur_s else None,
        "wallclock_s": round(wall, 1),
        "tool_calls": tool_calls,
        "timeout": timed_out,
        "think_unfinished": bool(timed_out and think and not content),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="", help="A/B/C/D — Kandidaten-Kürzel")
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--num-ctx", type=int, default=NUM_CTX)
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="je Probe; Überschreitung ist Messwert, kein Crash")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    meta = model_meta(args.url, args.model)
    digest = model_digest(args.url, args.model)
    print(f"Modell: {args.model} [{args.label}]  quant={meta.get('quantization')} "
          f"arch={meta.get('architecture')}  num_ctx={args.num_ctx} temp=0 "
          f"seed={args.seed}\n")

    results = []
    for t in TESTS:
        print(f"[{t['id']}] {t['name']} ...", flush=True)
        try:
            r = call_ollama(args.url, args.model, t.get("system"), t["prompt"],
                            t.get("tools"), args.seed, args.num_ctx, args.timeout)
        except Exception as e:  # noqa: BLE001
            r = {"error": f"{type(e).__name__}: {e}", "timeout": False}
        r["id"], r["name"] = t["id"], t["name"]
        if "error" in r:
            r["check"] = f"ERROR: {r['error']}"
        elif r.get("timeout"):
            r["check"] = ("TIMEOUT (Think nicht abgeschlossen)"
                          if r.get("think_unfinished") else "TIMEOUT")
        else:
            r["check"] = auto_checks(t["id"], r)
        if t["id"] in MANUAL_IDS:
            r["manual_verdict"] = ""     # von Hand/nach Rubrik zu füllen
        results.append(r)
        if "error" in r:
            print(f"     {r['check']}")
        else:
            print(f"     think ~{r['think_tokens_approx']:>4} tok | "
                  f"{r['tok_per_s'] or '?':>5} tok/s | {r['wallclock_s']:>5}s | {r['check']}")

    ok = sum(1 for r in results if str(r.get("check", "")).startswith("OK"))
    think_total = sum(r.get("think_tokens_approx", 0) for r in results)
    out = {
        "probe_suite": True, "auftrag": "basismodell",
        "model": args.model, "label": args.label,
        "model_meta": meta, "digest": digest,
        "config": {"num_ctx": args.num_ctx, "temperature": 0.0,
                   "seed": args.seed, "timeout_s": args.timeout,
                   "thinking": "an (Modell-Default)"},
        "git_commit": _git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "auto_ok": ok, "n_probes": len(results),
        "think_tokens_total": think_total,
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = (args.label or args.model).replace("/", "_").replace(":", "_")
    tag = f"_{args.tag}" if args.tag else ""
    path = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_probe_{slug}{tag}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("\n" + "=" * 74)
    print(f"{'Test':<6} {'Check':<40} {'Think':>8} {'tok/s':>7} {'Zeit':>7}")
    print("-" * 74)
    for r in results:
        if "error" in r:
            print(f"{r['id']:<6} ERROR")
            continue
        print(f"{r['id']:<6} {r['check']:<40} {r['think_tokens_approx']:>7}t "
              f"{str(r['tok_per_s'] or '?'):>7} {r['wallclock_s']:>6}s")
    print("-" * 74)
    print(f"Auto-OK: {ok}/{len(results)}  |  Think gesamt ~{think_total} tok")
    print(f"JSON: {path}   (T1/T2/T6 manual_verdict noch leer)")


if __name__ == "__main__":
    main()
