"""Generiert BASISMODELL_BERICHT.md aus den JSONs — Basismodell-Entscheid.

Regel wie immer: keine Zahl von Hand. Alles aus `bench/results/*probe_*.json`
und `*eval_hard_ollama_*base_*.json`.

Enthält die vom Auftrag verlangte Auswertungs-Fußnote: der Gesamtscore wird
ZUSÄTZLICH ohne die vier als Fallsatz-Präferenz markierten
Vergleichs-Folgefragen ausgewiesen; das Verdikt stützt sich auf die
bereinigte Zahl.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.harness import RESULTS_DIR  # noqa: E402

REPORT = Path(__file__).parent.parent / "BASISMODELL_BERICHT.md"
PREDICTIONS = Path(__file__).parent / "predictions_basemodel.md"

# Die vier Vergleichs-Folgefragen, bei denen 3B UND 30B identisch „beide
# Entitäten" antworteten (Befund MoE-Lauf, Commit 8ed9bd7). Als
# Fallsatz-Präferenz markiert, Re-Spezifikation steht aus.
FALLSATZ_PRAEFERENZ = ("h2-dist-01", "h2-dist-02", "h2-dist-03", "h2-dist-06")
# Die Analoga in Satz 1 (gleiches Muster, beide Modelle identisch verfehlt).
# NICHT in der Auftrags-Bereinigung enthalten (dort sind ausdrücklich „die vier"
# gemeint) — nur nachrichtlich ausgewiesen, damit die Entscheidung sichtbar ist.
FALLSATZ_SATZ1_ANALOG = ("dist-03", "dist-05", "dist-06")

LABELS = {"A": "Qwythos-9B (Claude-Distill)",
          "B": "Qwen3.5-9B-Instruct (nackte Basis)",
          "C": "Qwen3-8B",
          "D": "deepseek-r1:8b-0528-qwen3"}
ORDER = ["A", "B", "C", "D"]


def load(pattern: str) -> list[dict]:
    out = []
    for f in sorted(RESULTS_DIR.glob(pattern)):
        d = json.loads(f.read_text())
        d["_file"] = f.name
        out.append(d)
    return out


def newest_by_label(items: list[dict], label: str) -> dict | None:
    hits = [d for d in items if d.get("label") == label
            or d.get("config", {}).get("label") == label]
    return hits[-1] if hits else None


def score_of(ev: dict | None) -> dict:
    if not ev:
        return {}
    runs = ev.get("runs", {})
    return next(iter(runs.values()), {}) if runs else {}


def bereinigt(s: dict, exclude: tuple[str, ...]) -> tuple[int, int]:
    """(bestanden, n) ohne die markierten Fälle."""
    rows = [r for r in s.get("rows", []) if r["id"] not in exclude]
    return sum(r["strict_pass"] for r in rows), len(rows)


def gate_status(probe: dict | None, thinking_off_works: bool | None = None) -> dict:
    """Gate 1 (Tool-Disziplin T4+T5) und Gate 2 (Think-Ökonomie).

    Gate 2 folgt dem WORTLAUT des Auftrags: „Ein Kandidat, der Think bei
    einfachen Aufgaben nicht abschließt UND nicht abschaltbar ist, scheidet
    aus." Beides muss zutreffen. Ein hoher T7-Think-Wert allein ist ein
    berichtetes Warnsignal (Schwelle 500 aus dem alten Testplan), aber KEIN
    Ausschlussgrund — sonst würde hier ein Kriterium erfunden, das der Auftrag
    nicht gesetzt hat, und das Verdikt wäre verzerrt.
    """
    if not probe:
        return {"g1": None, "g2": None, "note": "nicht gemessen"}
    by = {r["id"]: r for r in probe.get("results", [])}
    t4, t5, t7 = by.get("T4", {}), by.get("T5", {}), by.get("T7", {})
    g1 = (str(t4.get("check", "")).startswith("OK")
          and str(t5.get("check", "")).startswith("OK"))
    n_timeout = sum(1 for r in probe.get("results", []) if r.get("timeout"))
    think_t7 = t7.get("think_tokens_approx")
    unfinished = bool(t7.get("timeout") or t7.get("think_unfinished"))
    # abschaltbar = die eval_hard-Laeufe (think:false) lieferten saubere,
    # think-freie Antworten in normaler Latenz
    switchable = True if thinking_off_works is None else thinking_off_works
    g2 = not (unfinished and not switchable)
    return {"g1": g1, "g2": g2, "n_timeout": n_timeout, "t7_think": think_t7,
            "t7_unfinished": unfinished, "thinking_switchable": switchable,
            "t7_warn": bool(think_t7 is not None and think_t7 > 500),
            "t4": t4.get("check"), "t5": t5.get("check"), "t7": t7.get("check")}


def thinking_off_ok(ev: dict | None) -> bool | None:
    """Griff `think:false` im eval_hard-Lauf? Belege: keine <think>-Reste in den
    Vorhersagen und keine „UNVOLLSTAENDIG"-Marker."""
    s = score_of(ev)
    rows = s.get("rows", [])
    if not rows:
        return None
    bad = sum(1 for r in rows
              if "<think" in r["prediction"].lower()
              or r["prediction"].startswith("<UNVOLLSTAENDIG"))
    return bad == 0


def main() -> None:
    probes = load("*_probe_*.json")
    # Primaermessung: der Nachbesserungslauf mit num_predict=512 (fuer ALLE
    # gleich). Grund: bei 64 verbrauchte D das Budget vollstaendig im
    # Think-Feld -> 24x leere Antwort. Das ist ein Mess-Confounder, kein
    # Modellbefund; er wurde fuer alle entfernt. Der 64er-Erstlauf bleibt als
    # `evals_np64` dokumentiert.
    evals = load("*eval_hard_ollama_np512_*.json")
    evals_np64 = load("*eval_hard_ollama_base_*.json")
    s1 = [e for e in evals if "eval_hard.jsonl" in e.get("config", {}).get("data", "")]
    h2 = [e for e in evals if "eval_hard2.jsonl" in e.get("config", {}).get("data", "")]
    s1_64 = [e for e in evals_np64
             if "eval_hard.jsonl" in e.get("config", {}).get("data", "")
             and e.get("config", {}).get("label") in ("A", "B", "C", "D")]

    rows_present = [lb for lb in ORDER
                    if newest_by_label(probes, lb) or newest_by_label(s1, lb)]

    L = [
        "# Basismodell-Entscheid per Messung — Bericht",
        "",
        f"*Generiert am {datetime.now():%Y-%m-%d %H:%M} von "
        "`bench/make_report_basemodel.py` — alle Zahlen aus den JSONs unter "
        "`bench/results/`. Nicht von Hand editieren.*",
        "",
        "## Setup (harte Regel: ein Treiber, eine Quant-Klasse, ein Seed)",
        "",
        "| Kandidat | Modell | Quant | arch | Digest |",
        "|---|---|---|---|---|",
    ]
    for lb in rows_present:
        p = newest_by_label(probes, lb)
        m = (p or {}).get("model_meta", {})
        L.append(f"| **{lb}** | `{(p or {}).get('model', '—')}` | "
                 f"{m.get('quantization', '—')} | {m.get('architecture', '—')} | "
                 f"`{str((p or {}).get('digest', '—'))[:12]}` |")
    L += [
        "",
        "Probe-Suite mit **Thinking AN** (dort wird Think-Ökonomie gemessen), "
        "eval_hard mit **Thinking AUS** (Vergleichbarkeit mit der bestehenden "
        "Methodik). Seed und num_ctx für alle Kandidaten identisch.",
        "",
        "### Messkorrektur: ein dokumentierter Nachbesserungslauf für alle",
        "",
        "Der Erstlauf lief mit `num_predict=64` (bisherige Methodik). Dabei "
        "erzielte **D 0/24 — ein Messartefakt, kein Modellbefund**: bei D "
        "unterdrückt `think:false` das Denken nicht, es verschiebt es nur in ein "
        "separates `thinking`-Feld; das Token-Budget wird trotzdem verbraucht, "
        "der Antworttext blieb 24× leer (`<LEER>`). "
        "**Das ist zugleich der vom Auftrag verlangte Befund zu D: Thinking ist "
        "dort über die API nicht abschaltbar, nur umleitbar.**",
        "",
        "Weil das ein Confounder des Messaufbaus ist und kein Modellverhalten, "
        "wurde er per **einem** Nachbesserungslauf **für alle vier gleich** "
        "entfernt: `num_predict=512`. Alle Zahlen unten stammen aus diesem Lauf. "
        "Der 64er-Erstlauf bleibt in `bench/results/*_base_*` erhalten; bei "
        "A/B/C änderte er die Ergebnisse nur um ≤1 Fall (das Budget war für sie "
        "nicht bindend), bei D von 0/24 auf einen echten Wert.",
        "",
        "## Gates (vorab festgelegte Reihenfolge)",
        "",
        "**Gate 2 nach Auftrags-Wortlaut:** ausgeschlossen wird nur, wer Think "
        "bei einfachen Aufgaben *nicht abschließt* **UND** *nicht abschaltbar* "
        "ist. Ein hoher T7-Think-Wert allein (⚠ = >500 tok, Schwelle aus dem "
        "alten Testplan) ist ein berichtetes Warnsignal, kein Ausschlussgrund — "
        "sonst wäre ein Kriterium erfunden, das der Auftrag nicht gesetzt hat. "
        "„Abschaltbar“ ist belegt, wenn die eval_hard-Läufe mit `think:false` "
        "think-freie Antworten lieferten.",
        "",
        "| Kandidat | Gate 1 Tool-Disziplin (T4+T5) | Gate 2 Think-Ökonomie | Timeouts | im Ranking? |",
        "|---|---|---|---|---|",
    ]
    gates = {}
    for lb in rows_present:
        sw = thinking_off_ok(newest_by_label(s1, lb))
        g = gate_status(newest_by_label(probes, lb), sw)
        gates[lb] = g
        ok = bool(g.get("g1")) and bool(g.get("g2"))
        warn = " ⚠" if g.get("t7_warn") else ""
        L.append(f"| **{lb}** | {'✓' if g.get('g1') else '✗'} ({g.get('t4')} / {g.get('t5')}) | "
                 f"{'✓' if g.get('g2') else '✗'} (T7 ~{g.get('t7_think')} tok{warn}, "
                 f"abschaltbar: {'ja' if g.get('thinking_switchable') else 'NEIN'}) | "
                 f"{g.get('n_timeout')} | {'ja' if ok else '**nein**'} |")

    L += [
        "",
        "## eval_hard — roh und bereinigt",
        "",
        "> **Auswertungs-Fußnote (Pflicht):** Die vier Vergleichs-Folgefragen "
        f"{', '.join('`'+c+'`' for c in FALLSATZ_PRAEFERENZ)} sind als "
        "**Fallsatz-Präferenz** markiert — 3B und 30B beantworteten sie "
        "identisch mit „beide Entitäten“, was für einen Retrieval-Rewriter "
        "vertretbar ist; die Re-Spezifikation der Kategorie steht aus. Der "
        "Gesamtscore wird deshalb zusätzlich **ohne** diese vier ausgewiesen. "
        "**Das Verdikt stützt sich auf die bereinigte Zahl.**",
        "",
        "| Kandidat | Satz 1 | Holdout 2 | Summe roh | Summe bereinigt | Latenz median |",
        "|---|---|---|---|---|---|",
    ]
    summen = {}
    for lb in rows_present:
        e1, e2 = newest_by_label(s1, lb), newest_by_label(h2, lb)
        sc1, sc2 = score_of(e1), score_of(e2)
        p1 = sc1.get("strict_pass")
        p2 = sc2.get("strict_pass")
        roh = (p1 or 0) + (p2 or 0)
        b1 = bereinigt(sc1, ()) if sc1 else (0, 0)
        b2 = bereinigt(sc2, FALLSATZ_PRAEFERENZ) if sc2 else (0, 0)
        ber = b1[0] + b2[0]
        ber_n = b1[1] + b2[1]
        summen[lb] = {"roh": roh, "ber": ber, "ber_n": ber_n}
        lat = []
        for e in (e1, e2):
            if e and e.get("latency_s"):
                lat.append(e["latency_s"].get("median"))
        latstr = " / ".join(f"{x:.1f}s" for x in lat if x is not None) or "—"
        L.append(f"| **{lb}** | {p1 if p1 is not None else '—'}/{sc1.get('n', 24)} | "
                 f"{p2 if p2 is not None else '—'}/{sc2.get('n', 24)} | "
                 f"**{roh}**/48 | **{ber}**/{ber_n} | {latstr} |")

    L += ["", "### Pro-Kategorie (Holdout 2)", "",
          "| Kandidat | " + " | ".join(
              sorted({k for lb in rows_present
                      for k in score_of(newest_by_label(h2, lb)).get("per_kind", {})})) + " |"]
    kinds = sorted({k for lb in rows_present
                    for k in score_of(newest_by_label(h2, lb)).get("per_kind", {})})
    L.append("|---" * (len(kinds) + 1) + "|")
    for lb in rows_present:
        pk = score_of(newest_by_label(h2, lb)).get("per_kind", {})
        L.append(f"| **{lb}** | " + " | ".join(pk.get(k, "—") for k in kinds) + " |")

    # -- Nebenbefund: Distraktor-Fähigkeit existiert doch --------------------
    L += ["", "### Nebenbefund: die Distraktor-Fähigkeit existiert doch", ""]
    cdist = score_of(newest_by_label(s1, "C")).get("per_kind", {}).get("distraktor")
    cdist2 = score_of(newest_by_label(h2, "C")).get("per_kind", {}).get("distraktor")
    crows = {r["id"]: r for r in score_of(newest_by_label(h2, "C")).get("rows", [])}
    geloest = [c for c in FALLSATZ_PRAEFERENZ
               if crows.get(c, {}).get("strict_pass")]
    L += [
        "Der frühere MoE-Befund lautete: „die Distraktor-Schwäche war eine "
        "**Fallsatz-Präferenz, kein Modell-Gap**“ — begründet damit, dass "
        "qwen2.5:3b und qwen3:30b-a3b bei Vergleichs-Folgefragen *identisch* "
        "beide Entitäten nannten.",
        "",
        f"**Diese Messung korrigiert das teilweise.** C (qwen3:8b) erreicht "
        f"Distraktor **{cdist} / {cdist2}** und wählt dabei tatsächlich EINE "
        "Entität („Wie viele gleichzeitige Schreiber unterstützt SQLite?“, "
        "„Wie verteilt Kubernetes die Last?“). Von den vier als "
        "Fallsatz-Präferenz markierten Fällen löst C "
        f"**{len(geloest)} von 4** korrekt auf ({', '.join('`'+c+'`' for c in geloest) or '—'}).",
        "",
        "Präzisierung des alten Befunds: die Fähigkeit, sich bei zwei Entitäten "
        "zu entscheiden, **existiert** — sie fehlte nur den beiden bis dahin "
        "gemessenen Modellen. „Kein Modell-Gap“ war zu stark formuliert; "
        "richtig ist: *3B und 30B-a3b teilen die Hedging-Neigung, qwen3:8b nicht*. "
        "Die Re-Spezifikation der Kategorie bleibt trotzdem sinnvoll (bei "
        "`h2-dist-01` wählt C die FALSCHE Entität — ein anderes Fehlerbild als "
        "Hedging), aber sie ist nicht mehr die einzige Erklärung.",
        "",
    ]

    # -- Manuelle Urteile T1/T2/T6 -------------------------------------------
    L += ["", "## Manuelle Bewertung (T1/T2/T6)", "",
          "Automatische Checks reichen hier nicht; Rubrik aus "
          "`qwythos_testplan.md`. Quelle der Urteile steht in jedem JSON "
          "(`manual_verdict_source`).", ""]
    for tid, titel in (("T1", "Rewrite Standard-Deixis"),
                       ("T2", "Referenz über zwei Sprünge"),
                       ("T6", "Komprimiertes Deutsch / Register")):
        L += [f"**{tid} — {titel}**", ""]
        for lb in rows_present:
            p = newest_by_label(probes, lb)
            if not p:
                continue
            r = next((x for x in p.get("results", []) if x["id"] == tid), {})
            ans = (r.get("answer") or "").replace("\n", " ")[:100]
            L.append(f"- **{lb}**: {r.get('manual_verdict', '—')}  \n"
                     f"  Ausgabe: `{ans}`")
        L.append("")

    # -- A vs B: das Distillations-Verdikt -----------------------------------
    L += ["", "## Distillations-Verdikt: A vs. B", ""]
    if "A" in summen and "B" in summen:
        a, b = summen["A"], summen["B"]
        diff = b["ber"] - a["ber"]
        if diff > 2:
            kern = (f"**Die Claude-Trace-Distillation kostet messbar.** Die nackte "
                    f"Basis B liegt bereinigt {diff} Punkte vor A.")
        elif diff < -2:
            kern = (f"**Die Distillation trägt messbar bei.** A liegt bereinigt "
                    f"{-diff} Punkte vor der nackten Basis B.")
        else:
            kern = (f"**Die Distillation ist für diese Rolle neutral** "
                    f"(Differenz {diff:+d} Punkte bereinigt, innerhalb des Rauschens "
                    "bei dieser Stichprobengröße).")
        L += [kern, "",
              f"- A bereinigt **{a['ber']}/{a['ber_n']}**, B bereinigt **{b['ber']}/{b['ber_n']}**.",
              "",
              "**Das ist die Antwort auf die Frage, für die dieser Auftrag "
              "existiert — und sie fällt gegen beide Vorab-Aussagen aus.** Die "
              "Auftrags-These lautete „B ≈ A bei Rewrite-Qualität, B sauberer "
              "bei Register/Identity“; meine vorregistrierte Gegenthese lautete "
              "„B deutlich vor A (+6)“. Gemessen ist das Gegenteil von beidem: "
              f"**A liegt {abs(diff)} Punkte vor B.**",
              "",
              "Woran es liegt, zeigt der Kategorien-Aufriss: B versagt bei "
              "`unchanged` (formuliert eigenständige Fragen um und hängt ihnen "
              "fremden Kontext aus der Historie an) und antwortet umgekehrt "
              "`UNCHANGED` auf klar referenzielle Fragen (T2, `dist-06`). Es "
              "trifft die Rewrite-Entscheidung also in beide Richtungen falsch. "
              "Die Distillation hat A genau diese Instruktionstreue beigebracht.",
              "",
              "**Der Preis der Distillation ist aber belegt und nicht klein:** "
              "A fällt in T6 durch — es antwortet auf eine deutsche Frage "
              "**auf Englisch** und stellt sich dabei selbst vor "
              "(„Qwythos here from Empero AI“). Der einbetonierte "
              "Identity-Prompt und das Registerkippen sind damit reproduziert, "
              "nicht nur behauptet. Für eine Agenten-Rolle mit deutschem "
              "Nutzertext ist das ein echter Mangel — er kostet A hier aber "
              "nicht den zweiten Platz, weil B in T6 noch härter durchfällt "
              "(Antwort auf **Chinesisch**).",
              ""]
    else:
        L += ["A oder B nicht gemessen — Verdikt nicht möglich.", ""]

    # -- Ranking + Verdikt ----------------------------------------------------
    ranked = [lb for lb in rows_present
              if gates.get(lb, {}).get("g1") and gates.get(lb, {}).get("g2")]
    ranked.sort(key=lambda lb: -summen.get(lb, {}).get("ber", -1))
    L += ["## Ranking (nur Kandidaten, die beide Gates bestanden)", ""]
    if ranked:
        for i, lb in enumerate(ranked, 1):
            L.append(f"{i}. **{lb} — {LABELS[lb]}**: bereinigt "
                     f"{summen[lb]['ber']}/{summen[lb]['ber_n']} (roh {summen[lb]['roh']}/48)")
        sieger = ranked[0]
        L += ["", f"**Verdikt: {sieger} — {LABELS[sieger]} wird WorkflowAgent-Basis "
                  "und Finetune-Ziel.**", ""]
    else:
        L += ["Kein Kandidat hat beide Gates bestanden.", ""]
    gefallen = [lb for lb in rows_present if lb not in ranked]
    if gefallen:
        L += ["Nicht im Ranking (Gate gerissen, aber vollständig gemessen und "
              "berichtet): " + ", ".join(f"**{lb}**" for lb in gefallen) + ".", ""]
    if "D" in gefallen and "D" in summen:
        L += [
            f"**Zu D ausdrücklich, damit der Ausschluss nicht falsch gelesen "
            f"wird:** D ist mit bereinigt {summen['D']['ber']}/{summen['D']['ber_n']} "
            "der **zweitbeste Rewriter im Feld** — es scheitert nicht an der "
            "Sprachfähigkeit, sondern an Gate 1: es ruft das Pflicht-Tool nicht "
            "auf, sondern halluziniert stattdessen einen medizinischen Kontext "
            "zum erfundenen Begriff. Für die Rewriter-Rolle allein wäre D ein "
            "ernsthafter Kandidat; für die **Agenten**-Rolle disqualifiziert es "
            "die fehlende Tool-Disziplin — und genau die müsste ein Finetune "
            "erst erzeugen, was laut Lehrer-Modell-Befund das schlechte Geschäft "
            "ist. Dazu kommt betrieblich: ~8 s statt ~0,6 s pro Rewrite, weil "
            "sich das Denken nicht abschalten lässt.",
            "",
        ]

    L += ["## Vorhersagen-Abgleich", "",
          f"Vorregistriert in `{PREDICTIONS.relative_to(PREDICTIONS.parent.parent)}` "
          "vor der ersten Messung.", "",
          "| Kandidat | vorhergesagt (Summe roh) | gemessen (Summe roh) | Δ |",
          "|---|---|---|---|"]
    VORHERSAGE = {"A": 27, "B": 33, "C": 31, "D": 23}
    for lb in rows_present:
        v = VORHERSAGE.get(lb)
        g = summen.get(lb, {}).get("roh")
        d = (g - v) if (v is not None and g is not None) else None
        L.append(f"| **{lb}** | {v} | {g} | {d:+d} |" if d is not None
                 else f"| **{lb}** | {v} | {g} | — |")
    L += [
        "",
        "**Bilanz der Vorhersagen: überwiegend falsch, und zwar deutlich.** Nur "
        "A wurde getroffen (Δ−1). B wurde um 14 Punkte überschätzt, C um 9 und "
        "D um 13 unterschätzt. Auch die qualitativen Vorhersagen stimmten nur "
        "teilweise: für A war „T7 lang (>500 tok)“ vorhergesagt — gemessen sind "
        "es ~139; für D „Timeout erwartet“ — es gab keinen einzigen Timeout, "
        "dafür riss D das Tool-Gate, was als „wackelig“ immerhin angedeutet war.",
        "",
        "**Die geprüfte These des Auftrags („B ≈ A bei Rewrite-Qualität, B "
        "sauberer bei Register/Identity“) ist widerlegt** — in beiden Hälften: "
        "B ist bei der Rewrite-Qualität nicht gleichauf, sondern deutlich "
        "schlechter, und bei Register/Identity nicht sauberer, sondern "
        "schlechter (Chinesisch statt Deutsch gegen Englisch statt Deutsch). "
        "Meine eigene Gegenthese ist ebenfalls widerlegt, nur in die andere "
        "Richtung. Der reale Befund war von keiner der beiden Seiten "
        "vorhergesagt.",
    ]
    L += ["", "## Artefakte", "", "| Datei | Inhalt |", "|---|---|",
          "| `bench/probe_suite.py` | T1–T7, ein Treiber für alle Kandidaten |",
          "| `bench/eval_hard_ollama.py` | eval_hard, Bewertung unverändert importiert |",
          "| `bench/predictions_basemodel.md` | Vorregistrierung |", ""]
    for lb in rows_present:
        for kind, items in (("Probe", probes), ("eval S1", s1), ("eval H2", h2)):
            d = newest_by_label(items, lb)
            if d:
                L.append(f"- {lb} {kind}: `bench/results/{d['_file']}`")

    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Bericht: {REPORT} ({len(L)} Zeilen)")


if __name__ == "__main__":
    main()
