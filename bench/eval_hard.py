"""Harter Rewrite-Eval: handgeschriebene Faelle mit Pro-Fall-Zusicherungen.

Warum ein zweiter Eval: `eval_rewrite.py` misst gegen Referenzen, die vom
SELBEN Rewriter stammen, der die Trainingsdaten erzeugt hat, und seine Metrik
`keeps_anchor` besteht schon, wenn die Haelfte der Ankerbegriffe vorkommt.
Beides ist zu milde:

- Ein Distraktor wird nicht bestraft. Steht „Redis" UND „Memcached" in der
  Ausgabe, ist der Anker erfuellt — obwohl das Modell die Frage nicht
  aufgeloest, sondern beide Kandidaten hingeschrieben hat.
- Wer die Historie einfach anhaengt, besteht jeden Ankertest.
- `UNCHANGED` wird gar nicht geprueft. Genau das ist die Fehlerklasse, die
  entsteht, wenn man die synthetischen Negative weglaesst (Lauf v2): das
  Modell koennte jetzt bereits eigenstaendige Fragen unnoetig umformulieren.

Hier steht deshalb pro Fall, was drin sein MUSS und was NICHT drin sein darf,
dazu Laengen-, Sprach- und Registergrenzen. Ein Fall gilt nur als bestanden,
wenn ALLE Zusicherungen halten (`strict`).

Die Faelle sind von Hand geschrieben, nicht vom Rewriter erzeugt — der Eval
ist damit unabhaengig von der Trainingsverteilung.

  --self-test  prueft die Faelle gegen konstruierte Ausgaben (CPU, kein
               Modell): Ideal-Antwort muss bestehen, History-Dump und
               Distraktor-Antwort muessen durchfallen.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.eval_rewrite import (  # noqa: E402
    prompt_opens_think, strip_think, tokens,
)

HARD_DATA = Path(__file__).parent / "data" / "eval_hard.jsonl"
REWRITE_SYSTEM = (
    "Du bist ein Query-Rewriter für ein Retrieval-System. Forme die "
    "referenzielle Folgefrage zu EINER eigenständigen Suchanfrage um, die "
    "ohne den Verlauf verständlich ist. Antworte nur mit der Suchanfrage. "
    "Ist die Frage bereits eigenständig, antworte mit: UNCHANGED"
)
REWRITE_USER = (
    "Formuliere die FOLGEFRAGE als eigenständige, vollständige Frage um, die "
    "ohne das Gespräch verständlich ist. Behalte die Sprache der Folgefrage "
    "bei. Antworte NUR mit der umformulierten Frage — keine Erklärung, keine "
    "Anführungszeichen.\n\nGESPRÄCH:\n{history}\n\nFOLGEFRAGE: {query}\n\n"
    "Eigenständige Frage:"
)
_SIE = re.compile(r"\b(Sie|Ihnen|Ihre|Ihrer|Ihren|Ihrem|Ihres)\b")
_LEAD_CONJ = re.compile(r"^(und|aber|oder|also|and|but|or)\b", re.IGNORECASE)
_DE_HINT = frozenset("der die das und ist wie was wird von zu mit bei für "
                     "ein eine wann warum welche".split())
_EN_HINT = frozenset("the and is how what does are of to with for a an "
                     "why when which".split())


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def render_history(history: list[list[str]]) -> str:
    return "\n".join(f"Nutzer: {q}\nAssistent: {a}" for q, a in history)


def build_messages(case: dict) -> list[dict]:
    return [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": REWRITE_USER.format(
            history=render_history(case["history"]), query=case["follow_up"])},
    ]


def detect_lang(text: str) -> str:
    words = set(tokens(text))
    de, en = len(words & _DE_HINT), len(words & _EN_HINT)
    if de == en:
        return "unbekannt"
    return "de" if de > en else "en"


def check(case: dict, pred: str) -> dict:
    """Alle Zusicherungen eines Falls. → {name: True/False}, nur Verletztes zaehlt."""
    low = pred.lower()
    words = tokens(pred)
    res: dict[str, bool] = {}

    is_unchanged = low.strip().rstrip(".!").strip() == "unchanged"
    if case.get("expect_unchanged"):
        # UNCHANGED ist korrekt; ebenso, die Frage praktisch unveraendert zu
        # lassen. Falsch waere, sie am fremden Thema der Historie aufzuhaengen.
        res["unchanged_or_kept"] = is_unchanged or _nahe_am_original(pred, case["follow_up"])
    else:
        res["kein_unchanged"] = not is_unchanged

    if not is_unchanged:
        must = [m.lower() for m in case.get("must_include", [])]
        if must:
            hits = [m for m in must if m in low]
            res["antezedent"] = bool(hits) if case.get("must_include_any") \
                else len(hits) == len(must)
        for bad in case.get("must_not_include", []):
            res[f"kein_{bad}"] = bad.lower() not in low
        res["laenge"] = len(words) <= case.get("max_words", 16)
        res["kein_konjunktionsanfang"] = not _LEAD_CONJ.match(pred.strip())
        res["kein_sie"] = not _SIE.search(pred)
        lang = case.get("lang", "de")
        got = detect_lang(pred)
        res["sprache"] = got in (lang, "unbekannt")
    return res


def _nahe_am_original(pred: str, follow_up: str) -> bool:
    """Ist die Ausgabe im Kern die unveraenderte Frage? (Jaccard ueber Woerter)"""
    a, b = set(tokens(pred)), set(tokens(follow_up))
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= 0.6


def score(cases: list[dict], preds: list[str]) -> dict:
    rows: list[dict] = []
    by_kind: dict[str, list[bool]] = {}
    violations: dict[str, int] = {}
    for case, pred in zip(cases, preds):
        checks = check(case, pred)
        ok = all(checks.values())
        for name, passed in checks.items():
            if not passed:
                violations[name] = violations.get(name, 0) + 1
        by_kind.setdefault(case["kind"], []).append(ok)
        rows.append({"id": case["id"], "kind": case["kind"],
                     "follow_up": case["follow_up"], "prediction": pred,
                     "strict_pass": ok,
                     "failed": [k for k, v in checks.items() if not v]})
    n_ok = sum(r["strict_pass"] for r in rows)
    return {
        "n": len(rows), "strict_pass": n_ok,
        "strict_rate": round(n_ok / len(rows), 3) if rows else 0.0,
        "per_kind": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_kind.items())},
        "violations": dict(sorted(violations.items(), key=lambda kv: -kv[1])),
        "rows": rows,
    }


def generate(model, tokenizer, case: dict, max_new: int, opens_think: bool,
             thinking: bool) -> str:
    kw = {} if thinking else {"enable_thinking": False}
    ids = tokenizer.apply_chat_template(
        build_messages(case), add_generation_prompt=True, tokenize=True,
        return_dict=False, return_tensors="pt", **kw).to("cuda")
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=max_new,
                             do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    raw = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    if opens_think and "</think>" not in raw:
        return f"<UNVOLLSTAENDIG nach {max_new} Tokens> {raw[:100]}"
    return strip_think(raw) or f"<LEER> {raw[:100]}"


def self_test(cases: list[dict]) -> None:
    """Kalibriert die Schaerfe: was MUSS bestehen, was MUSS durchfallen."""
    ideal, dump, distractor = [], [], []
    for c in cases:
        if c.get("expect_unchanged"):
            ideal.append("UNCHANGED")
        elif c.get("lang", "de") == "en":
            ideal.append(f"How does {c['must_include'][0]} actually work?")
        else:
            inc = c.get("must_include", ["x"])[0]
            ideal.append(f"Wie funktioniert {inc} im Detail genau?")
        dump.append(render_history(c["history"])[:400])
        bad = c.get("must_not_include")
        distractor.append(
            f"Was ist {c.get('must_include', ['x'])[0]} und {bad[0]}?"
            if bad else "UNCHANGED" if not c.get("expect_unchanged") else "x")
    for label, preds in (("Ideal-Antwort", ideal), ("History-Dump", dump),
                         ("Distraktor-Antwort", distractor)):
        s = score(cases, preds)
        print(f"  {label:20} strict {s['strict_pass']}/{s['n']}  "
              f"Verletzungen: {s['violations']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="empero-ai/Qwythos-9B-Claude-Mythos-5-1M")
    p.add_argument("--adapter", default=None)
    p.add_argument("--data", default=str(HARD_DATA))
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--budget-mb", type=int, default=2048)
    p.add_argument("--quantize-transfer", nargs="?", const="int8",
                   choices=["int8", "int4"], default=False)
    p.add_argument("--no-thinking", action="store_true")
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

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from bench.harness import RESULTS_DIR, _git_commit
    from bench.train_lora import set_adapters, wrap_lora
    from k4n0n3.hooks import LayerManager

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    model.to("cpu")
    for param in model.parameters():
        param.requires_grad_(False)

    has_adapter = bool(args.adapter)
    if has_adapter:
        wrap_lora(model, r=8)
        ckpt = torch.load(args.adapter, map_location="cpu")
        state = ckpt.get("state", ckpt)
        info = model.load_state_dict(state, strict=False)
        if info.unexpected_keys:
            raise SystemExit(f"Adapter passt nicht: {len(info.unexpected_keys)} unbekannt")
        print(f"Adapter: {args.adapter} (Schritt {ckpt.get('meta', {}).get('step', '?')})")

    LayerManager(model, layer_prefix="model.layers", vram_budget_mb=args.budget_mb,
                 prefetch_depth=1, quantize_transfer=args.quantize_transfer).prepare()
    model.eval()
    opens_think = prompt_opens_think(tokenizer) and not args.no_thinking

    runs = {}
    for label, enabled in (("mit_adapter", True), ("ohne_adapter", False)):
        if not has_adapter and label == "mit_adapter":
            continue
        if has_adapter:
            set_adapters(model, enabled)
        preds = []
        for c in cases:
            pred = generate(model, tokenizer, c, args.max_new, opens_think,
                            thinking=not args.no_thinking)
            preds.append(pred)
            print(f"  [{label}] {c['id']} {c['follow_up'][:30]!r} → {pred[:60]!r}",
                  flush=True)
        runs[label] = score(cases, preds)
        if not has_adapter:
            break

    result = {"eval": True, "hard": True, "config": vars(args),
              "git_commit": _git_commit(),
              "timestamp": datetime.now().isoformat(timespec="seconds"),
              "runs": runs}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_eval_hard{tag}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    for label, s in runs.items():
        print(f"  {label:14} strict {s['strict_pass']}/{s['n']} "
              f"({s['strict_rate']:.0%})  {s['per_kind']}")
        print(f"                 Verletzungen: {s['violations']}")
    print(f"  JSON: {out}")


if __name__ == "__main__":
    main()
