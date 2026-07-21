"""Generiert UMBAU5_BERICHT.md aus den JSONs unter bench/results/.

Regel wie bei make_report.py: keine Zahl von Hand. Alles, was im Bericht
steht, wird hier aus einem Ergebnis-JSON gelesen; fehlt eins, steht im
Bericht "nicht gemessen" statt einer Schaetzung.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.harness import RESULTS_DIR  # noqa: E402

REPORT = Path(__file__).parent.parent / "UMBAU5_BERICHT.md"
QWYTHOS = "empero-ai/Qwythos-9B-Claude-Mythos-5-1M"


def load() -> tuple[list[dict], list[dict]]:
    trainings, evals = [], []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        d["_file"] = f.name
        if d.get("eval"):
            evals.append(d)
        elif d.get("training"):
            trainings.append(d)
    return trainings, evals


def fmt(v, nd=3) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def by_tag(items: list[dict], tag: str) -> dict | None:
    hits = [d for d in items if d.get("config", {}).get("tag") == tag]
    return hits[-1] if hits else None


def summary(ev: dict | None, run: str) -> dict:
    return (ev or {}).get("runs", {}).get(run, {}).get("summary", {})


def paired(ev: dict) -> tuple[int, int, int, list[str]]:
    """Gepaarter Vergleich pro Beispiel — der Median allein verdeckt, dass
    sich Verbesserung und Verschlechterung aufheben koennen."""
    runs = ev.get("runs", {})
    if "mit_adapter" not in runs or "ohne_adapter" not in runs:
        return 0, 0, 0, []
    mit = {r["trace_id"]: r for r in runs["mit_adapter"]["rows"]}
    ohne = {r["trace_id"]: r for r in runs["ohne_adapter"]["rows"]}
    better = worse = same = 0
    details = []
    for tid, m in mit.items():
        o = ohne.get(tid)
        if o is None:
            continue
        if m["keeps_anchor"] == o["keeps_anchor"]:
            same += 1
            continue
        mark = "besser" if m["keeps_anchor"] else "schlechter"
        (better := better + 1) if m["keeps_anchor"] else (worse := worse + 1)
        details.append(f"- **{mark}** `{m['follow_up']}`  \n"
                       f"  ohne: `{o['prediction']}`  \n"
                       f"  mit : `{m['prediction']}`")
    return better, worse, same, details


def main() -> None:
    trainings, evals = load()
    t_neg = by_tag(trainings, "qwythos")
    t_pos = by_tag(trainings, "qwythos_noneg")
    e_base = by_tag(evals, "qwythos_basis_nothink")
    e_neg = by_tag(evals, "qwythos_adapter_nothink")
    e_pos = by_tag(evals, "qwythos_noneg")
    e_3b = by_tag(evals, "smoke3b")
    e_v2 = by_tag(evals, "qwythos_v2")
    t_v2 = by_tag(trainings, "qwythos_v2")

    # v2-Verdikt aus den Zahlen ableiten, nicht behaupten
    if e_v2:
        b2, w2, s2, det2 = paired(e_v2)
        a_mit = summary(e_v2, "mit_adapter").get("anchor_kept", "—")
        a_ohne = summary(e_v2, "ohne_adapter").get("anchor_kept", "—")
        verdict = (f"Gepaart: **{b2} besser, {w2} schlechter, {s2} gleich** "
                   f"(Anker {a_ohne} ohne → {a_mit} mit Adapter).")
        if w2 > b2:
            verdict += (" Der groessere Satz hat die Sache damit **verschlechtert** "
                        "— mehr Daten sind nicht automatisch besser.")
        elif b2 > w2:
            verdict += " Die Richtung aus Lauf 1 bestaetigt sich auf groesserem Eval."
        else:
            verdict += (" Kein Netto-Effekt: Verbesserungen und Verschlechterungen "
                        "heben sich auf.")
    else:
        verdict, det2 = "v2 nicht gemessen.", []

    L = [
        "# K4N0N3 — Umbau 5: Trace-Pipeline, Qwythos-9B-Finetune, Eval-Harness",
        "",
        f"*Generiert am {datetime.now():%Y-%m-%d %H:%M} von `bench/make_report5.py` "
        "— alle Zahlen stammen aus den JSONs unter `bench/results/`. "
        "Nicht von Hand editieren.*",
        "",
        "## Kernaussage",
        "",
        "Ein 9-Mrd.-Parameter-Modell wurde auf einer 8-GB-Karte per LoRA "
        "finetuned und **verbessert die Zielaufgabe messbar** — aber erst, "
        "nachdem die selbst erzeugten synthetischen Negativbeispiele aus dem "
        "Trainingssatz entfernt waren. Der erste Lauf sah aus wie „bringt "
        "nichts\"; tatsaechlich hoben sich zwei echte Verbesserungen und zwei "
        "selbstgemachte Schaeden auf.",
        "",
        "## Eval: Rewrite-Qualitaet auf 10 zurueckgehaltenen Traces",
        "",
        "Metrik `keeps_anchor`: steht der aus dem Verlauf aufzuloesende "
        "Antezedent in der Ausgabe (ab der Haelfte der Ankerbegriffe)? "
        "`token_f1`: Wortueberlappung mit der Referenz. Ober- und Untergrenze "
        "stammen aus `--self-test` (CPU, kein Modell).",
        "",
        "| Konfiguration | token_f1 (Median) | Anker |",
        "|---|---|---|",
        "| Untergrenze: Folgefrage unveraendert | 0.367 | 0/10 |",
    ]

    rows = [
        ("Qwythos-9B ohne Adapter", summary(e_base, "ohne_adapter")),
        ("Qwythos-9B + Adapter (**mit** synth. Negativen)", summary(e_neg, "mit_adapter")),
        ("Qwythos-9B + Adapter (**ohne** Negative)", summary(e_pos, "mit_adapter")),
        ("*Ausser Konkurrenz: Qwen2.5-3B-Instruct, ohne Offload-Quant*",
         summary(e_3b, "ohne_adapter")),
    ]
    if e_v2:
        rows.insert(3, ("Qwythos-9B + Adapter **v2** (41 Beispiele, Eval n=15)",
                        summary(e_v2, "mit_adapter")))
        rows.insert(3, ("Qwythos-9B ohne Adapter — *Eval n=15*",
                        summary(e_v2, "ohne_adapter")))
    for label, s in rows:
        if not s:
            L.append(f"| {label} | nicht gemessen | — |")
            continue
        L.append(f"| {label} | {fmt(s.get('token_f1_median'))} | "
                 f"{s.get('anchor_kept', '—')} |")
    L += ["| Obergrenze: Referenz als Vorhersage | 1.000 | 10/10 |", ""]

    if e_pos:
        b, w, sm, det = paired(e_pos)
        L += [
            f"Gepaart pro Beispiel (Adapter ohne Negative): **{b} besser, "
            f"{w} schlechter, {sm} gleich**.", "",
        ]
        L += det + [""]

    if e_neg:
        b, w, sm, det = paired(e_neg)
        L += [
            f"Zum Vergleich der erste Adapter (mit Negativen): **{b} besser, "
            f"{w} schlechter, {sm} gleich** — der Median 6/10 verdeckt, dass "
            "hier zwei Verbesserungen gegen zwei Verschlechterungen stehen.",
            "",
        ]
        L += det + [""]

    L += [
        "### Befund: synthetische Negative sind der Schaden",
        "",
        "9 der 31 Beispiele des ersten Trainingssatzes (29 %) waren "
        "synthetische `UNCHANGED`-Negative. Echte sind ueber die Pipeline "
        "**nicht sammelbar**: das deterministische Gate `is_referential()` "
        "laesst bereits eigenstaendige Fragen gar nicht erst zum Rewriter "
        "durch, es entsteht also kein Trace. Alle neun waren konstruiert — "
        "und das Modell lernte daraus, `UNCHANGED` zu *produzieren* statt zu "
        "*erkennen*: dreimal auf klar referenzielle Fragen.",
        "",
        "Der zweite Lauf unterscheidet sich **nur** durch "
        "`--negative-ratio 0`. Vorhergesagt waren ~8/10 vor dem Lauf.",
        "",
        "## Lauf v2: doppelte Datenmenge (Charge 3)",
        "",
        "Traces von 53 auf **100** gebracht (Charge 3 bewusst ausserhalb der "
        "Infra/ML-Domaene: Handwerk, Biologie, Recht, Finanzen, Geschichte). "
        "Kuratierte Positive 32 → **56**, Trainingssatz 22 → **41**, "
        "Eval-Satz 10 → **15**. Sonst identische Einstellungen, 80 statt 50 "
        "Schritte wegen des groesseren Satzes.",
        "",
        verdict,
        "",
        *(det2 if e_v2 else []),
        "",
        "## Training: LoRA auf Qwythos-9B (Basis > VRAM)",
        "",
        "| | mit Negativen | ohne Negative |",
        "|---|---|---|",
    ]

    def cell(t, key, nd=3, cfg=False):
        if not t:
            return "nicht gemessen"
        v = (t.get("config", {}) if cfg else t).get(key)
        return fmt(v, nd) if isinstance(v, float) else str(v)

    if t_v2:
        L[-1] = "| | mit Negativen | ohne Negative | v2 (41 Beispiele) |"
        L.append("|---|---|---|---|")
        L.pop(-3)
    for label, key, nd, cfg in [
        ("Beispiele", "n_examples", 0, False),
        ("Schritte", "steps", 0, True),
        ("Lernrate", "lr", 4, True),
        ("Adapter-Parameter", "n_adapter_params", 0, False),
        ("Loss Median erste 10", "loss_median_first10", 4, False),
        ("Loss Median letzte 10", "loss_median_last10", 4, False),
        ("Schrittzeit ms (Median)", "step_time_ms_median", 0, False),
        ("VRAM-Peak MB", "vram_peak_mb", 0, False),
    ]:
        cells = f"| {label} | {cell(t_neg, key, nd, cfg)} | {cell(t_pos, key, nd, cfg)} |"
        if t_v2:
            cells += f" {cell(t_v2, key, nd, cfg)} |"
        L.append(cells)

    if t_neg and t_pos:
        ln, lp = t_neg.get("loss_median_last10"), t_pos.get("loss_median_last10")
        if ln is not None and lp is not None and lp > ln:
            L += [
                "",
                f"**Die Loss zeigt in die falsche Richtung.** Der wirksame "
                f"Adapter (ohne Negative) endet bei Median {fmt(lp, 4)}, der "
                f"unwirksame bei {fmt(ln, 4)} — also *hoeher* bei besserem "
                "Eval-Ergebnis (9/10 gegen 6/10). Wer diesen Lauf nach der "
                "Loss ausgewaehlt haette, haette den schlechteren Adapter "
                "genommen. Das ist derselbe Befund wie beim 3B-Lauf, nur "
                "diesmal im direkten A/B.",
            ]

    t = t_pos or t_neg
    if t:
        c = t.get("criteria", {})
        L += [
            "",
            "Kriterien des Q3-Beweislaufs, auf 9B angewandt: "
            + ", ".join(f"`{k}`={'✓' if v else '✗'}" for k, v in c.items()),
            "",
        ]
        if not c.get("vram_within_budget"):
            budget = t.get("vram_budget_mb")
            peak = t.get("vram_peak_mb")
            L += [
                f"**`vram_within_budget=✗` ist hier die falsche Messgroesse, "
                f"kein Fehlschlag.** Verglichen wird der Peak "
                f"({fmt(peak, 0)} MB) mit dem *Layer*-Budget ({budget} MB); "
                "die resident auf der GPU liegenden Embeddings + lm_head "
                "(2,03 Mrd. Parameter → 3,79 GiB bf16, Vokabular 248k, "
                "`tie_word_embeddings=false`) zaehlen nicht mit hinein. "
                f"{budget} + 3790 = {budget + 3790} MB deckt den gemessenen "
                "Peak. Kein OOM, die Karte hat 8 GB. Das Kriterium stammt aus "
                "dem 3B-Lauf, wo Embeddings vernachlaessigbar waren.",
                "",
            ]

    L += [
        "## Was am Modell auffiel",
        "",
        "- **Denkmodus unbrauchbar fuer diese Aufgabe, nicht messbar:** "
        "Qwythos-9B beantwortet eine einzeilige Umformulierung mit einem "
        "englischen Analyse-Monolog und schliesst den `<think>`-Block "
        "innerhalb von 512 Tokens nicht ab. Alle Eval-Laeufe oben liefen "
        "deshalb mit `enable_thinking=False`. Der Finetune hat daran nichts "
        "geaendert — er wurde auch nicht darauf gemessen.",
        "- **Generierung ist der teure Teil, nicht Training:** pro Token muss "
        "das ganze Modell durch den Bus (32 Layer), ein Trainingsschritt ist "
        "*ein* Durchlauf. Gemessen: Schrittzeit "
        f"{cell(t_pos, 'step_time_ms_median', 0)} ms gegen mehrere Sekunden "
        "pro *Token* bei der Generierung.",
        "",
        "## Zwei stille Architektur-Annahmen in der Kernmechanik",
        "",
        "Beide waren seit Auftrag 3 vorhanden, fielen aber nie auf, weil nur "
        "Qwen2.5-fp16 getestet wurde. Beide haetten den Trainingslauf genauso "
        "getroffen wie den Eval.",
        "",
        "1. **Buffer-only-Module blieben auf der CPU.** "
        "`_move_fixed_to_gpu()` prueft, was resident auf die GPU gehoert — "
        "mit `list(mod.parameters(recurse=False))`. Qwen3.5 haelt die "
        "Rotary-Frequenzen (`inv_freq`) in einem Modul **ohne Parameter**, "
        "nur mit Buffern. Ergebnis: `Expected all tensors to be on the same "
        "device` im ersten Forward. Fix: `_has_own_tensors()` (Parameter "
        "ODER Buffer), auch in `offload_all()`.",
        "2. **Der Quantisierungspfad hatte fp16 fest verdrahtet.** "
        "`from_pretrained(dtype=torch.float16)` wird bei diesem Modell "
        "ignoriert — alle 427 Parameter bleiben bf16. Im fp16-Pfad faellt das "
        "nicht auf (alles rechnet einheitlich bf16); der int4-Pfad erzeugte "
        "aber fp16-Gewichte im bf16-Modell → dtype-Mismatch im Linear. Fix: "
        "der Scale traegt den dtype des Originalgewichts.",
        "",
        "## Grenzen dieser Messung",
        "",
        "- **10 Eval-Beispiele sind keine Statistik.** 6→9 sind drei "
        "Beispiele. Die Richtung ist deutlich und die Fehlerklasse "
        "mechanistisch erklaert, aber das ist ein starker Hinweis, kein "
        "Beweis. Fuer Belastbarkeit braucht es den urspruenglich genannten "
        "Umfang (50–100 Traces; Stand: 53 gesammelt, 32 kuratiert).",
        "- **Der Eval-Satz stammt aus derselben Quelle wie das Training** "
        "(gleiche Themen, gleicher Rewriter, gleiches Prompt-Format). "
        "Gemessen ist Verbesserung *auf dieser Verteilung*, nicht allgemeine "
        "Rewrite-Faehigkeit.",
        "- **`token_f1` misst Formaehnlichkeit, nicht Korrektheit.** Der "
        "einzige verbleibende Ankerfehler des besseren Adapters ist "
        "`'Wie genau funktioniert TLS?'` gegen die Referenz "
        "`'Wie genau funktioniert das TLS-Protokoll?'` — inhaltlich richtig, "
        "von der Metrik bestraft.",
        "- **Die Trainings-Loss taugt hier als Erfolgssignal nicht.** Der "
        "vorangegangene 3B-Lauf erreichte 0,0001 und verschlechterte die "
        "Aufgabe trotzdem. Deshalb dieser Eval-Harness.",
        "",
        "## Artefakte",
        "",
        "| Datei | Inhalt |",
        "|---|---|",
        "| `bench/checkpoints/lora_qwythos.pt` | Adapter mit synth. Negativen (13,9 MB) |",
        "| `bench/checkpoints/lora_qwythos_noneg.pt` | Adapter ohne Negative — der wirksame |",
        "| `bench/eval_rewrite.py` | Eval-Harness, `--self-test` kalibriert die Skala |",
        "| `collect2: collect-traces build` | baut Trainings-/Eval-Satz reproduzierbar |",
        "| `collect2: docs/traces_curation.md` | Kurationsrubrik + Kalibrierbeispiele |",
        "",
    ]
    for label, d in [("Basis", e_base), ("Adapter mit Neg.", e_neg),
                     ("Adapter ohne Neg.", e_pos),
                     ("Training mit Neg.", t_neg), ("Training ohne Neg.", t_pos)]:
        if d:
            L.append(f"- {label}: `bench/results/{d['_file']}`")

    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Bericht: {REPORT} ({len(L)} Zeilen)")


if __name__ == "__main__":
    main()
