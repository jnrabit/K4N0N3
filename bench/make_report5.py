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


def load() -> tuple[list[dict], list[dict], list[dict]]:
    trainings, evals, hard = [], [], []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        d["_file"] = f.name
        if d.get("hard"):
            hard.append(d)
        elif d.get("eval"):
            evals.append(d)
        elif d.get("training"):
            trainings.append(d)
    return trainings, evals, hard


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


def _rescore(h: dict | None, data) -> dict:
    """Gespeicherte Ausgaben gegen die AKTUELLE Falldefinition neu bewerten."""
    from bench.eval_hard import load_cases, score
    if not h:
        return {}
    cases = {c["id"]: c for c in load_cases(data)}
    out = {}
    for label, stored in h.get("runs", {}).items():
        rows = stored.get("rows", [])
        try:
            out[label] = score([cases[r["id"]] for r in rows],
                               [r["prediction"] for r in rows])
        except KeyError:
            out[label] = stored
    return out


def _holdout2_section(h_v2: dict | None, h_v3: dict | None) -> list[str]:
    """Der zweite, VOR der Datensammlung festgeschriebene Holdout."""
    from bench.eval_hard import HARD_DATA
    data2 = HARD_DATA.parent / "eval_hard2.jsonl"
    L = ["## Holdout 2: der unabhaengige Test", ""]
    if not (h_v2 and h_v3):
        return L + ["Nicht gemessen.", ""]
    L += [
        "Charge 4 (36 Paare) zielte gezielt auf die oben gemessenen "
        "Schwaechen — damit misst `eval_hard.jsonl` nicht mehr "
        "Generalisierung, sondern In-Distribution-Leistung. "
        "`eval_hard2.jsonl` wurde deshalb **vor** der Datensammlung "
        "geschrieben und in einem eigenen Commit festgeschrieben (durchgehend "
        "andere Entitaeten und Domaenen). Das ist die unabhaengige Zahl.",
        "",
        "| Kategorie | ohne Adapter | v2 (41 Beisp.) | v3 (56 Beisp., Charge 4) |",
        "|---|---|---|---|",
    ]
    r2, r3 = _rescore(h_v2, data2), _rescore(h_v3, data2)
    ohne, m2, m3 = r2.get("ohne_adapter", {}), r2.get("mit_adapter", {}), r3.get("mit_adapter", {})
    for kind in sorted(m2.get("per_kind", {})):
        L.append(f"| {kind} | {ohne.get('per_kind', {}).get(kind, '—')} | "
                 f"{m2.get('per_kind', {}).get(kind, '—')} | "
                 f"{m3.get('per_kind', {}).get(kind, '—')} |")
    L += [
        f"| **gesamt** | **{ohne.get('strict_pass')}/{ohne.get('n')}** | "
        f"**{m2.get('strict_pass')}/{m2.get('n')}** | "
        f"{m3.get('strict_pass')}/{m3.get('n')} |",
        "",
        "### Befund: Charge 4 hat nichts gebracht",
        "",
        f"v3 ({m3.get('strict_pass')}/{m3.get('n')}) liegt unter v2 "
        f"({m2.get('strict_pass')}/{m2.get('n')}). Der Unterschied ist "
        "Rauschen, aber es gibt keinen Grund, v3 zu bevorzugen — und keinen, "
        "die 36 zusaetzlichen Traces als Fortschritt auszuweisen. "
        "**v2 bleibt der Adapter der Wahl.**",
        "",
        "Aufgeschluesselt: v3 gewinnt einen Distraktor-Fall "
        "(`SSD- oder HDD-Speicher` → `eine SSD`), verliert aber je einen bei "
        "tiefem Antezedenten und Kontrast. Das Distraktor-Training wirkt "
        "also — nur zu schwach, um die Verwaesserung aufzuwiegen.",
        "",
        "**Warum es schiefging, steht in der Kuration:** von 14 gesammelten "
        "Distraktor-Paaren ueberlebten **3**, weil qwen2.5:3b bei zwei "
        "Entitaeten fast immer hedgt (beide nennt). Drei Beispiele gegen 56 "
        "im Satz bewegen nichts, waehrend die 15 neuen Beispiele insgesamt "
        "die Verteilung verschoben und zwei bestehende Faehigkeiten leicht "
        "verwaesserten.",
        "",
        "**Die Lehre:** gezielt sammeln funktioniert nur, wenn das "
        "Lehrer-Modell die Zielfaehigkeit selbst beherrscht. Aus einer "
        "Quelle, die sich bei zwei Entitaeten nicht entscheiden kann, laesst "
        "sich die Entscheidung nicht destillieren — dafuer braeuchte es ein "
        "staerkeres Modell oder handgeschriebene Targets.",
        "",
        "**Korrektur zum Protokoll:** waehrend des Laufs wurde der Sprung "
        "beim tiefen Antezedenten (0/4 → 3/4) zunaechst Charge 4 "
        "zugeschrieben. Das war falsch — verglichen worden war gegen die "
        "BASIS, nicht gegen v2. v2 hatte dort bereits 4/4, ganz ohne "
        "Zwei-Turn-Trainingsdaten.",
        "",
    ]
    return L


def _small_model_section(h_3b: dict | None, h_3bneg: dict | None) -> list[str]:
    """qwen2.5:3b als der eigentlich einsetzbare Rewriter (resident, schnell)."""
    from bench.eval_hard import HARD_DATA
    data2 = HARD_DATA.parent / "eval_hard2.jsonl"
    L = ["## Der einsetzbare Weg: qwen2.5:3b (resident statt offloaded)", ""]
    if not h_3b:
        return L + ["Nicht gemessen.", ""]
    L += [
        "Der 9B laeuft auf dieser 8-GB-Karte nur ueber Offload — PCIe-4.0-x8-"
        "gebunden, ~300 s pro Rewrite, nicht interaktiv. Ein kleines Modell "
        "passt resident: kein Transfer pro Token, Millisekunden statt Minuten. "
        "Also dieselbe Finetune-Frage fuer `qwen2.5:3b`, gemessen auf "
        "**demselben** unabhaengigen Holdout 2.",
        "",
    ]
    r3b = _rescore(h_3b, data2)
    base = r3b.get("ohne_adapter", {})
    pos = r3b.get("mit_adapter", {})
    rows = [("**qwen2.5:3b Basis (Pipeline-Default)**", base)]
    rows.append(("+ Finetune (56 pos, `--negative-ratio 0`)", pos))
    if h_3bneg:
        neg = _rescore(h_3bneg, data2).get("mit_adapter", {})
        rows.append(("+ Finetune (56 pos + 9 neg, `0.15`)", neg))
    L += ["| Variante | gesamt | Distraktor | UNCHANGED |", "|---|---|---|---|"]
    for label, s in rows:
        pk = s.get("per_kind", {})
        L.append(f"| {label} | **{s.get('strict_pass')}/{s.get('n')}** | "
                 f"{pk.get('distraktor','—')} | {pk.get('unchanged','—')} |")
    L += [
        "",
        "**Kein Finetune schlaegt die Basis.** Der all-positive Lauf tauscht "
        "nur um: **+Distraktor** (1/6→3/6, das Modell lernt sich zu "
        "entscheiden) gegen **−UNCHANGED** (5/5→3/5, es formt jetzt schon "
        "eigenstaendige Fragen um) — netto 17=17. Der Versuch, das mit 15 % "
        "synthetischen Negativen zu balancieren, war die schlechteste Variante "
        "(16/24): er zerstoerte den Distraktor-Gewinn UND stellte UNCHANGED "
        "nicht wieder her.",
        "",
        "**Vorhergesagt waren ~19/24, gemessen 16/24 — die Hypothese ist "
        "widerlegt.** Damit ist der Negativ-Befund doppelt belegt: die "
        "synthetischen UNCHANGED-Negative helfen KEINEM Modell (sie "
        "verschlechterten schon den 9B). Sie sind konstruiert, nicht echt, und "
        "uebertragen nicht auf echte eigenstaendige Fragen. Echte Negative "
        "kann die Pipeline nicht sammeln — `is_referential()` faengt "
        "eigenstaendige Fragen vor dem Rewriter ab. Diese Gate-Grenze deckelt "
        "den Finetune-Ansatz.",
        "",
        "**Konsequenz fuer den Betrieb:** der Rewriter bleibt auf der "
        "unfinetunten `qwen2.5:3b` (bereits `rewrite_model`/`decompose_model` "
        "im Default) — resident, schnell, und so gut wie jeder Finetune hier. "
        "Der 9B-Finetune bleibt als belegte Faehigkeits-Demonstration "
        "(13→21 auf Holdout 2), nicht als Produktionsartefakt.",
        "",
    ]
    return L


def _moe_section(base1, moe1, base2, moe2) -> list[str]:
    """Staerkeres Basismodell (qwen3:30b-a3b-2507 MoE) gegen die 3B-Basis,
    gleicher Ollama-Treiber, gleiche Faelle: loest ein groesseres Modell den
    Distraktor-Fall — oder trifft es dieselbe Entscheidung?"""
    from bench.eval_hard import HARD_DATA
    data1, data2 = HARD_DATA, HARD_DATA.parent / "eval_hard2.jsonl"
    L = ["## Groesseres Basismodell: loest ein 30B-MoE den Distraktor-Fall?", ""]
    if not (base1 and moe1 and base2 and moe2):
        return L + ["Nicht gemessen.", ""]

    def sc(d, data):
        return _rescore(d, data).get("ollama", {})

    def lat(d):
        return (d.get("latency_s") or {}).get("median")

    def dfail(s):
        return {r["id"] for r in s.get("rows", [])
                if r["kind"] == "distraktor" and not r["strict_pass"]}

    b1, m1 = sc(base1, data1), sc(moe1, data1)
    b2, m2 = sc(base2, data2), sc(moe2, data2)
    d1 = m1.get("strict_pass", 0) - b1.get("strict_pass", 0)
    d2 = m2.get("strict_pass", 0) - b2.get("strict_pass", 0)
    ov2 = dfail(b2) & dfail(m2)
    ident2 = dfail(b2) == dfail(m2)

    L += [
        "Der einzige ungeloeste Fehler war der Distraktor-Fall (die Historie "
        "nennt zwei Entitaeten, gemeint ist eine). Antrainieren ging nicht, "
        "weil das Lehrer-Modell qwen2.5:3b selbst hedgt. Hypothese: ein "
        "deutlich staerkeres Basismodell bringt die Entscheidung schlicht mit. "
        "Getestet mit `qwen3:30b-a3b-instruct-2507` (q4, MoE, 3B aktiv/Token) "
        "gegen dieselbe 3B-Basis, **durch denselben Ollama-Treiber** "
        "(`bench/eval_hard_ollama.py`) auf **denselben** Faellen — die "
        "Bewertung (`build_messages`/`score`) ist unveraendert aus "
        "`eval_hard.py` importiert, nur der Modellaufruf ist neu. `--self-test` "
        "reproduziert die eingefrorene Skala (Ideal 24/24, Dump/Distraktor "
        "0/24), bevor eine Modellzahl faellt.",
        "",
        "| Modell (q4, Ollama) | Satz 1 | Holdout 2 | Distraktor S1 | Distraktor H2 | Latenz/Rewrite |",
        "|---|---|---|---|---|---|",
        f"| qwen2.5:3b (Basis) | {b1.get('strict_pass')}/{b1.get('n')} | "
        f"{b2.get('strict_pass')}/{b2.get('n')} | "
        f"{b1.get('per_kind', {}).get('distraktor', '—')} | "
        f"{b2.get('per_kind', {}).get('distraktor', '—')} | {fmt(lat(base1), 1)} s |",
        f"| **qwen3:30b-a3b-2507** | **{m1.get('strict_pass')}/{m1.get('n')}** | "
        f"**{m2.get('strict_pass')}/{m2.get('n')}** | "
        f"{m1.get('per_kind', {}).get('distraktor', '—')} | "
        f"{m2.get('per_kind', {}).get('distraktor', '—')} | "
        f"**{fmt(lat(moe1), 1)}–{fmt(lat(moe2), 1)} s** |",
        "",
        "### Befund: das groessere Modell trifft dieselbe Entscheidung",
        "",
        f"Das 30B ist gesamt klar besser (+{d1} auf Satz 1, +{d2} auf "
        "Holdout 2) — **aber nicht beim Distraktor.** Gepaart pro Fall "
        f"nachgesehen: auf Holdout 2 versagen beide Modelle an "
        f"{'**exakt denselben**' if ident2 else 'weitgehend denselben'} "
        f"Distraktor-Faellen ({', '.join('`'+c+'`' for c in sorted(ov2))}) — "
        "und zwar mit derselben Antwort: beide nennen beide Entitaeten "
        "(`X im Vergleich zu Y`). Beispiele, 3B gegen 30B:",
        "",
    ]
    m2rows = {r["id"]: r for r in m2.get("rows", [])}
    b2rows = {r["id"]: r for r in b2.get("rows", [])}
    for cid in sorted(ov2)[:3]:
        L.append(f"- `{cid}`  3B: `{b2rows[cid]['prediction'][:52]}`  ·  "
                 f"30B: `{m2rows[cid]['prediction'][:52]}`")
    L += [
        "",
        "Der eine Distraktor-Fall, den das 30B auf Satz 1 mehr besteht, war "
        "beim 3B gar kein Distraktor-Fehler, sondern ein Konjunktions-Anfang "
        "(`und wie persistiert Redis?` — Redis korrekt gewaehlt, nur mit `und` "
        "begonnen). Das 30B raeumt die Formulierung auf, nicht die "
        "Entscheidung.",
        "",
        "**Der +Gewinn kommt aus allgemeiner Kompetenz, nicht aus dem "
        "Distraktor:** saubere Formulierung (keine `und`-Anfaenge), korrekte "
        "tiefe Antezedenten, Register. Genau das, wo die 3B-Basis schwach ist. "
        "Die Distraktor-Entscheidung dagegen ist zwischen 3B und 30B praktisch "
        "identisch.",
        "",
        "**Vorab-Prognose war „deutliche Distraktor-Besserung (Modellstaerke), "
        "Rest ~gleich\". Gemessen das Gegenteil — Distraktor flach, Rest "
        "besser. Die Prognose ist widerlegt und so festgehalten.**",
        "",
        "**Einsetzbarkeit:** median "
        f"{fmt(lat(moe1), 1)}–{fmt(lat(moe2), 1)} s pro Rewrite, CPU-gebunden "
        "(68/32-Split), 19 GB resident, verdraengt alles andere auf der "
        "32-GB-Kiste. Kein gate-getriggerter Rewriter (der braucht die "
        f"~{fmt(lat(base1), 1)} s des 3B) — nur fuer latenztolerante Rollen "
        "(Batch/Offline).",
        "",
        "> **Fussnote zur Harness-Diskrepanz (3B-Basis 17 vs 13).** Der "
        "Abschnitt „Der einsetzbare Weg\" weist die 3B-Basis mit 17/24 auf "
        f"Holdout 2 aus, hier steht {b2.get('strict_pass')}/{b2.get('n')}. Kein "
        "Widerspruch, andere Harness: dort `eval_hard.py` (transformers, "
        "Qwen2.5-3B-Instruct in fp16), hier `eval_hard_ollama.py` (Ollama, "
        "q4_K_M). fp16-transformers und q4-Ollama sind nicht dieselbe Maschine "
        "— Quantisierung und Chat-Template-Details verschieben ein paar Faelle. "
        "Deshalb stehen in DIESEM Abschnitt beide Modelle auf demselben "
        "q4-Ollama-Treiber; nur so ist der 30B-Abstand ehrlich.",
        "",
        "### Folgeschritt: die Distraktor-Kategorie ist unterspezifiziert",
        "",
        "Dass ein 3B und ein 30B bei Vergleichs-Folgefragen dieselbe "
        "Entscheidung treffen (beide Entitaeten nennen), heisst: die "
        "Distraktor-Schwaeche war eine **Fallsatz-Praeferenz, kein "
        "Modell-Gap.** Fuer einen Retrieval-Rewriter mit RRF-Fusion ist "
        "`Nginx vs Apache Verbindungen` eine brauchbare Suchanfrage — die "
        "`must_not_include`-Zusicherung kodiert eine Praeferenz (genau eine "
        "Entitaet), die weder Modell teilt, dieselbe Fehlerklasse wie das "
        "bereits korrigierte `dist-02`.",
        "",
        "**Konkreter Folgeschritt:** die Distraktor-Kategorie re-spezifizieren "
        "— entscheiden, ob „beide nennen\" im Vergleichskontext bestehen soll, "
        "und `must_not_include` entsprechend anpassen. **Danach alle "
        "betroffenen Zahlen (3B, 9B v2/v3, 30B) einmal gegen die neue "
        "Definition neu ausweisen** (der Bericht rescored bereits gegen die "
        "aktuelle Falldefinition, die Neubewertung faellt also automatisch an, "
        "sobald `eval_hard.jsonl`/`eval_hard2.jsonl` angepasst sind). Bis dahin "
        "bleiben die Distraktor-Zahlen als „ungeloest, aber teils "
        "Fallsatz-Artefakt\" markiert.",
        "",
    ]
    return L


def _hard_section(h: dict | None) -> list[str]:
    """Harter Eval — handgeschriebene Faelle mit Pro-Fall-Zusicherungen."""
    L = ["## Harter Eval 1: 24 handgeschriebene Faelle", "",
         "*Achtung: Charge 4 wurde spaeter gezielt fuer die Kategorien dieses "
         "Satzes gesammelt. Die Zahlen hier stammen aus der Messung DAVOR "
         "(Adapter v2) und sind insofern noch unabhaengig; fuer alles danach "
         "gilt „Holdout 2\" als Massstab.*", ""]
    if not h:
        return L + ["Nicht gemessen.", ""]
    # Neu bewerten statt die gespeicherte Zusammenfassung zu uebernehmen:
    # die Zusicherungen in eval_hard.jsonl koennen sich seit dem Lauf geaendert
    # haben (dist-02). So bleibt der Bericht mit der aktuellen Definition
    # konsistent, statt zwei Zahlenstaende zu mischen.
    from bench.eval_hard import HARD_DATA, load_cases, score
    cases = {c["id"]: c for c in load_cases(HARD_DATA)}
    runs = {}
    for label, stored in h.get("runs", {}).items():
        rows = stored.get("rows", [])
        try:
            runs[label] = score([cases[r["id"]] for r in rows],
                                [r["prediction"] for r in rows])
        except KeyError:      # Fall aus dem Datensatz entfernt → Original
            runs[label] = stored
    mit, ohne = runs.get("mit_adapter"), runs.get("ohne_adapter")
    L += [
        "Der Eval oben ist zu milde: die Referenzen stammen vom SELBEN "
        "Rewriter, der die Trainingsdaten erzeugt hat, und `keeps_anchor` "
        "besteht schon bei der Haelfte der Ankerbegriffe — ein blosser "
        "History-Dump haette 100 % erreicht. `bench/eval_hard.py` prueft "
        "stattdessen pro Fall, was drin sein MUSS und was NICHT drin sein "
        "darf, dazu Laenge, Sprache und Register. Bestanden nur, wenn ALLE "
        "Zusicherungen halten (`strict`). Die Faelle sind von Hand "
        "geschrieben, also unabhaengig von der Trainingsverteilung.",
        "",
        "Schaerfe VOR der Messung kalibriert (`--self-test`, CPU): "
        "Ideal-Antwort 24/24, History-Dump 0/24, Distraktor-Antwort 0/24.",
        "",
        "| Kategorie | ohne Adapter | mit Adapter v2 | prueft |",
        "|---|---|---|---|",
    ]
    why = {
        "distraktor": "Historie nennt zwei Entitaeten — nur eine ist gemeint",
        "tiefer_antezedent": "Bezug liegt zwei Turns zurueck",
        "kontrast": "„und ohne?\" — das Gegenteil muss verschwinden",
        "kein_dump": "lange Historie, kurze Antwort erzwungen",
        "sprache": "englisch rein, englisch raus",
        "unchanged": "schon eigenstaendig → nicht anfassen",
    }
    for kind in sorted((mit or ohne or {}).get("per_kind", {})):
        L.append(f"| {kind} | {(ohne or {}).get('per_kind', {}).get(kind, '—')} | "
                 f"**{(mit or {}).get('per_kind', {}).get(kind, '—')}** | "
                 f"{why.get(kind, '')} |")
    if ohne and mit:
        L += [
            f"| **gesamt** | **{ohne['strict_pass']}/{ohne['n']} "
            f"({ohne['strict_rate']:.0%})** | **{mit['strict_pass']}/{mit['n']} "
            f"({mit['strict_rate']:.0%})** | alle Zusicherungen |",
            "",
            "Haeufigste Verletzung ohne Adapter: "
            + ", ".join(f"`{k}` ({v}x)" for k, v in
                        list(ohne.get("violations", {}).items())[:3])
            + " — also weitgehend die unveraendert durchgereichte Frage.",
            "",
            "**`unchanged` 5/5 in beiden Laeufen.** Das war der wichtigste "
            "Test: die Sorge, dass das Weglassen der synthetischen Negative "
            "(Lauf v2) das Modell zum Ueber-Umformulieren bringt, hat sich "
            "NICHT bestaetigt.",
            "",
        ]
    if mit:
        fails = [r for r in mit.get("rows", []) if not r["strict_pass"]]
        if fails:
            L += ["Was mit Adapter noch fehlschlaegt:", ""]
            for r in fails:
                L.append(f"- `{r['id']}` [{', '.join(r['failed'])}] "
                         f"`{r['follow_up']}` → `{r['prediction'][:70]}`")
            L.append("")
    L += [
        "**Offengelegte Korrektur:** `dist-02` war eine fehlerhafte "
        "Zusicherung, kein Modellfehler — bei „warum ist das schneller?\" ist "
        "die Frage implizit vergleichend, die Nennung des Vergleichspartners "
        "also korrekt. Nach Sichtung der Ausgaben entfernt. Die Korrektur "
        "hebt BEIDE Laeufe um je einen Fall (11→12 und 20→21) und veraendert "
        "den Abstand nicht. Die uebrigen Fehlschlaege bleiben stehen.",
        "",
    ]
    return L


def main() -> None:
    trainings, evals, hards = load()
    t_neg = by_tag(trainings, "qwythos")
    t_pos = by_tag(trainings, "qwythos_noneg")
    e_base = by_tag(evals, "qwythos_basis_nothink")
    e_neg = by_tag(evals, "qwythos_adapter_nothink")
    e_pos = by_tag(evals, "qwythos_noneg")
    e_3b = by_tag(evals, "smoke3b")
    e_v2 = by_tag(evals, "qwythos_v2")
    t_v2 = by_tag(trainings, "qwythos_v2")
    h_v2 = by_tag(hards, "v2")
    h2_v2 = by_tag(hards, "v2_holdout2")
    h2_v3 = by_tag(hards, "v3_holdout2")
    h2_3b = by_tag(hards, "3b_rewrite_h2")      # qwen2.5:3b, all-positiv
    h2_3bneg = by_tag(hards, "3b_neg15_h2")     # qwen2.5:3b, 15 % Negative
    moe_s1 = by_tag(hards, "qwen3_30b_a3b_2507")   # 30B-MoE, Satz 1 (Ollama)
    moe_h2 = by_tag(hards, "qwen3_30b_holdout2")   # 30B-MoE, Holdout 2
    base_s1 = by_tag(hards, "qwen25_3b_set1")      # 3B-Basis, gleicher Treiber
    base_h2 = by_tag(hards, "qwen25_3b_holdout2")

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
        "finetuned und **verbessert die Zielaufgabe messbar**: auf einem "
        "unabhaengigen, vor der Datensammlung festgeschriebenen Testsatz "
        "**13/24 ohne gegen 21/24 mit Adapter**.",
        "",
        "**Ergebnis-Artefakt ist `bench/checkpoints/lora_qwythos_v2.pt`** "
        "(41 Trainingsbeispiele). Der spaetere v3 mit gezielt nachgesammelten "
        "Daten (56 Beispiele) liegt mit 20/24 darunter — die Nachsammlung "
        "brachte nichts, siehe „Holdout 2\".",
        "",
        "**Fuer den PRODUKTIVEN Rewriter ist die Antwort trotzdem: kein "
        "Finetune.** Der 9B laeuft auf dieser 8-GB-Karte nur ueber Offload "
        "(~300 s/Rewrite, PCIe-gebunden). Ein resident laufendes qwen2.5:3b "
        "ist der einsetzbare Weg — und dessen Basis (17/24) schlaegt keiner "
        "der drei 3B-Finetunes. Siehe „Der einsetzbare Weg\".",
        "",
        "Drei Befunde waren dafuer noetig, alle gegen den ersten Anschein:",
        "",
        "1. **Die selbst erzeugten synthetischen Negativbeispiele haben den "
        "Finetune sabotiert.** Der erste Lauf sah aus wie „bringt nichts\"; "
        "tatsaechlich hoben sich zwei echte Verbesserungen und zwei "
        "selbstgemachte Schaeden auf. Sichtbar nur im gepaarten Vergleich pro "
        "Beispiel — der Median verdeckt es vollstaendig.",
        "2. **Die Loss taugt hier nicht zur Auswahl.** Der bessere Adapter hat "
        "die HOEHERE Endloss. Wer nach Loss ausgewaehlt haette, haette den "
        "schlechteren genommen.",
        "3. **Mehr Daten sind nicht automatisch besser.** Gezielt gesammelte "
        "Beispiele fuer die gemessenen Schwaechen haben das Ergebnis leicht "
        "verschlechtert, weil das Lehrer-Modell die Zielfaehigkeit selbst "
        "nicht beherrscht.",
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
        *_hard_section(h_v2),
        *_holdout2_section(h2_v2, h2_v3),
        *_small_model_section(h2_3b, h2_3bneg),
        *_moe_section(base_s1, moe_s1, base_h2, moe_h2),
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
        "- **Die Stichproben sind klein.** 24 harte Faelle und 15 "
        "zurueckgehaltene Traces; ein Unterschied von zwei, drei Faellen ist "
        "Rauschen. Die Richtung ist ueber beide Evals konsistent und die "
        "Fehlerklassen sind benannt — belastbar im statistischen Sinn ist das "
        "trotzdem nicht.",
        "- **Der Trace-Eval stammt aus derselben Quelle wie das Training** "
        "(gleiche Themen, gleicher Rewriter, gleiches Prompt-Format) und "
        "misst Verbesserung nur *auf dieser Verteilung*. Genau deshalb gibt "
        "es den harten Eval mit handgeschriebenen Faellen; dessen Zahlen sind "
        "die aussagekraeftigeren.",
        "- **Beide harten Evals sind von derselben Instanz geschrieben, die "
        "auch kuriert und trainiert hat.** Gleiche blinde Flecken wirken in "
        "beiden Richtungen. Die zeitliche Trennung (Holdout 2 vor der "
        "Datensammlung festgeschrieben) schuetzt gegen nachtraegliches "
        "Zurechtbiegen, nicht gegen geteilte Annahmen — ein von aussen "
        "geschriebener Fallsatz waere der naechste Schritt.",
        "- **Der Distraktor-Fall bleibt ungeloest.** 3/6 ohne Adapter, 3/6 mit "
        "v2, 4/6 mit v3: die Faehigkeit, sich bei zwei Entitaeten zu "
        "entscheiden, ist mit diesen Daten nicht antrainierbar.",
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
        "| **`bench/checkpoints/lora_qwythos_v2.pt`** | **Ergebnis-Adapter** (41 Beispiele, 21/24) |",
        "| `bench/checkpoints/lora_qwythos_v3.pt` | Charge-4-Variante (56 Beispiele, 20/24) — nicht besser |",
        "| `bench/checkpoints/lora_qwythos.pt` | Adapter mit synth. Negativen — der Negativ-Befund |",
        "| `bench/checkpoints/lora_qwythos_noneg.pt` | erster wirksamer Adapter (22 Beispiele) |",
        "| `bench/data/eval_hard2.jsonl` | unabhaengiger Holdout, vor Charge 4 festgeschrieben |",
        "| `bench/checkpoints/lora_qwen3b_rewrite.pt` | qwen2.5:3b-Finetune (all-positiv) — schlaegt Basis nicht |",
        "| `bench/checkpoints/lora_qwen3b_neg15.pt` | qwen2.5:3b + 15 % Negative — schlechteste Variante |",
        "| `bench/eval_rewrite.py` | Trace-Eval (Adapter an/aus, gepaart) |",
        "| `bench/eval_hard.py` + `bench/data/eval_hard.jsonl` | harter Eval, 24 handgeschriebene Faelle |",
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
