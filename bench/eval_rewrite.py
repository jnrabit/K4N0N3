"""Eval auf zurueckgehaltenen Rewrite-Traces — Adapter an vs. aus.

Der 3B-Lauf hat gezeigt, warum das noetig ist: die Loss fiel auf 0,0001,
waehrend der Adapter die Aufgabe faktisch verschlechterte. Loss misst
Auswendiglernen, nicht Faehigkeit. Hier wird stattdessen generiert und mit
der zurueckgehaltenen Zielfrage verglichen.

Kein einzelner Score als Urteil — die Metriken sind grob und sollen nur
sortieren; das JSON enthaelt jede Ausgabe im Wortlaut, damit man nachlesen
kann, was das Modell wirklich geschrieben hat.

  token_f1        Wortueberlappung mit der Referenz (0..1)
  keeps_anchor    steht der aufgeloeste Antezedent in der Ausgabe? Anker =
                  Woerter aus VERLAUF und Referenz, die in der Folgefrage
                  fehlen; bestanden ab der Haelfte davon. Das ist die
                  eigentliche Aufgabe, token_f1 nur die Formaehnlichkeit.

--self-test steckt die Spannweite ab (CPU, kein Modell): Referenz als
Vorhersage = Obergrenze, unveraenderte Folgefrage = Untergrenze.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EVAL_DATA = Path.home() / "collect2" / "data" / "traces" / "curated" / "eval_set.jsonl"
_WORD = re.compile(r"[\wäöüß]+", re.IGNORECASE)
_STOP = frozenset(
    "der die das den dem des ein eine einen einem einer und oder aber wie was "
    "wo wann warum welche welcher welches ist sind wird von zu auf in aus fuer "
    "für mit bei nach vor ueber über im am zur zum sich the and for with what "
    "how why when which does is are of to in on a an it that this".split())


def tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


def token_f1(pred: str, ref: str) -> float:
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return 0.0
    common = sum(min(p.count(w), r.count(w)) for w in set(p) & set(r))
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return round(2 * prec * rec / (prec + rec), 3)


def anchor_terms(reference: str, follow_up: str, history: str = "") -> list[str]:
    """Der aufgeloeste Antezedent: Woerter, die im VERLAUF und in der Referenz
    stehen, aber nicht in der Folgefrage.

    Nur aus der Referenz genuegt nicht — dann zaehlen Fuellverben wie
    „funktioniert" als Anker, und eine themenlose Ausgabe bestuende den Test.
    Der Schnitt mit dem Verlauf laesst genau das uebrig, was aufzuloesen war."""
    fu = set(tokens(follow_up))
    cand = [w for w in dict.fromkeys(tokens(reference))
            if w not in fu and w not in _STOP and len(w) > 3]
    if history:
        hist = set(tokens(history))
        cand = [w for w in cand if w in hist]
    return cand


def keeps_anchor(pred: str, reference: str, follow_up: str,
                 history: str = "") -> bool | None:
    """True, wenn mindestens die Haelfte der Ankerbegriffe in der Ausgabe steht.

    Nicht „irgendeiner": ein einzelner Treffer laesst zu viel durch, wenn der
    Anker aus mehreren Woertern besteht (z. B. „Rate-Limiting Sliding-Window")."""
    anchors = anchor_terms(reference, follow_up, history)
    if not anchors:
        return None  # nichts aufzuloesen — Beispiel taugt nicht als Anker-Test
    have = set(tokens(pred))
    return sum(a in have for a in anchors) / len(anchors) >= 0.5


def follow_up_of(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") == "user":
            hit = re.search(r"FOLGEFRAGE:\s*(.+?)(?:\n|$)", m.get("content", ""))
            return hit.group(1).strip() if hit else m.get("content", "")
    return ""


def history_of(messages: list[dict]) -> str:
    """Der Gespraechsteil des Prompts (zwischen GESPRÄCH: und FOLGEFRAGE:)."""
    for m in messages:
        if m.get("role") == "user":
            hit = re.search(r"GESPR[ÄA]CH:\s*(.*?)\s*FOLGEFRAGE:",
                            m.get("content", ""), re.DOTALL)
            return hit.group(1).strip() if hit else ""
    return ""


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def load_eval(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        msgs = d.get("messages", [])
        if len(msgs) < 2:
            continue
        out.append({"prompt_messages": msgs[:-1],
                    "reference": strip_think(msgs[-1].get("content", "")),
                    "follow_up": follow_up_of(msgs),
                    "history": history_of(msgs),
                    "trace_id": d.get("meta", {}).get("trace_id", "?"),
                    "tools": d.get("tools") or None})
    return out


def aggregate(rows: list[dict]) -> dict:
    f1 = [r["token_f1"] for r in rows]
    anchors = [r["keeps_anchor"] for r in rows if r["keeps_anchor"] is not None]
    return {
        "n": len(rows),
        "token_f1_median": round(statistics.median(f1), 3) if f1 else 0.0,
        "anchor_kept": f"{sum(anchors)}/{len(anchors)}" if anchors else "—",
        "anchor_rate": round(sum(anchors) / len(anchors), 3) if anchors else None,
    }


def generate(model, tokenizer, ex: dict, max_new: int) -> str:
    ids = tokenizer.apply_chat_template(
        ex["prompt_messages"], tools=ex["tools"], add_generation_prompt=True,
        tokenize=True, return_dict=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return strip_think(tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--adapter", default=None, help="LoRA-State-Dict (.pt) — ohne: nur Basis")
    p.add_argument("--data", default=str(EVAL_DATA))
    p.add_argument("--max-new", type=int, default=48)
    p.add_argument("--budget-mb", type=int, default=3072)
    p.add_argument("--quantize-transfer", nargs="?", const="int8",
                   choices=["int8", "int4"], default=False)
    p.add_argument("--self-test", action="store_true",
                   help="nur Metriken gegen den Eval-Satz pruefen (CPU, kein Modell)")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    examples = load_eval(Path(args.data))
    if not examples:
        raise SystemExit(f"Kein Eval-Satz in {args.data}")

    if args.self_test:
        # Referenz gegen sich selbst: F1 muss 1.0 sein, Anker muss halten.
        # Folgefrage als Vorhersage: der Anker fehlt per Konstruktion.
        perfect = [{"token_f1": token_f1(e["reference"], e["reference"]),
                    "keeps_anchor": keeps_anchor(e["reference"], e["reference"],
                                                 e["follow_up"], e["history"])}
                   for e in examples]
        lazy = [{"token_f1": token_f1(e["follow_up"], e["reference"]),
                 "keeps_anchor": keeps_anchor(e["follow_up"], e["reference"],
                                              e["follow_up"], e["history"])}
                for e in examples]
        print(f"{len(examples)} Eval-Beispiele aus {args.data}")
        print(f"  Referenz als Vorhersage (Obergrenze): {aggregate(perfect)}")
        print(f"  Folgefrage unveraendert (Untergrenze): {aggregate(lazy)}")
        for e in examples[:3]:
            print(f"\n  {e['follow_up']!r}\n    → {e['reference']!r}"
                  f"\n    Anker: {anchor_terms(e['reference'], e['follow_up'], e['history'])}")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from bench.harness import RESULTS_DIR, _git_commit
    from bench.train_lora import set_adapters, wrap_lora
    from k4n0n3.hooks import ZeroFlushModel

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    model.to("cpu")
    for param in model.parameters():
        param.requires_grad_(False)

    has_adapter = bool(args.adapter)
    if has_adapter:
        wrap_lora(model, r=8)
        state = torch.load(args.adapter, map_location="cpu")
        missing = model.load_state_dict(state, strict=False)
        print(f"Adapter geladen: {len(state)} Tensoren, "
              f"unerwartet: {len(missing.unexpected_keys)}")

    zfm = ZeroFlushModel(model, vram_budget_mb=args.budget_mb,
                         quantize_transfer=args.quantize_transfer)
    zfm.prepare()
    model.eval()

    runs = {}
    for label, enabled in (("mit_adapter", True), ("ohne_adapter", False)):
        if not has_adapter and label == "mit_adapter":
            continue
        if has_adapter:
            set_adapters(model, enabled)
        rows = []
        for ex in examples:
            pred = generate(model, tokenizer, ex, args.max_new)
            rows.append({"trace_id": ex["trace_id"], "follow_up": ex["follow_up"],
                         "reference": ex["reference"], "prediction": pred,
                         "token_f1": token_f1(pred, ex["reference"]),
                         "keeps_anchor": keeps_anchor(pred, ex["reference"],
                                                      ex["follow_up"], ex["history"])})
            print(f"  [{label}] {ex['follow_up'][:35]!r} → {pred[:60]!r}", flush=True)
        runs[label] = {"summary": aggregate(rows), "rows": rows}
        if not has_adapter:
            break

    result = {"eval": True, "config": vars(args), "git_commit": _git_commit(),
              "timestamp": datetime.now().isoformat(timespec="seconds"),
              "n_examples": len(examples), "runs": runs}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_path = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_eval_rewrite{tag}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    for label, r in runs.items():
        print(f"  {label:14} {r['summary']}")
    print(f"  JSON: {out_path}")


if __name__ == "__main__":
    main()
