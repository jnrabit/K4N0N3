# Vorregistrierung — Basismodell-Entscheid per Messung

**Festgeschrieben VOR dem ersten Messlauf** (Auftrag „Basismodell-Entscheid",
Abschnitt Vorregistrierung). Wer das nachträglich ändert, macht die Messung
wertlos. Abgleich erfolgt im Bericht, auch und gerade wenn er peinlich ausfällt.

Datum: 2026-07-29
Autor der Vorhersagen: Claude (Opus 4.8), auf Basis der bisherigen Messreihen.

## Setup, auf das sich die Zahlen beziehen

Alle vier Kandidaten über DENSELBEN Ollama-Treiber, alle **Q4_K_M**,
temp=0, seed=42, num_ctx identisch. eval_hard mit **Thinking AUS**,
Probe-Suite mit **Thinking AN**. Zahlen sind nur innerhalb dieses Setups
vergleichbar (17-vs-13-Befund).

Referenzpunkte aus demselben Ollama-q4-Treiber (nicht Altzahlen aus
transformers!): `qwen2.5:3b` 15/24 (Satz 1) & 13/24 (Holdout 2);
`qwen3:30b-a3b-2507` 19/24 & 18/24.

## Vorhergesagte eval_hard-Ergebnisse (strict, roh)

| Kandidat | Satz 1 | Holdout 2 | Summe | T4 (Pflicht-Call) | T5 (Kein-Call-Falle) | T7 Think |
|---|---|---|---|---|---|---|
| **A: Qwythos-9B** | 14/24 | 13/24 | **27** | besteht (tools-fähig) | Risiko unnötiger Call | lang (>500 tok) |
| **B: Qwen3.5-9B-Instruct** | 17/24 | 16/24 | **33** | besteht | besteht | mittel |
| **C: Qwen3-8B** | 16/24 | 15/24 | **31** | besteht (natives TC) | besteht | mittel |
| **D: deepseek-r1:8b-0528** | 12/24 | 11/24 | **23** | wackelig | Risiko unnötiger Call | Timeout erwartet |

Erwartete Fehlerbilder (damit nicht nachträglich uminterpretiert wird):

- **A**: Register-/Identity-Effekte der Claude-Distillation; unter
  Thinking-AUS eher Geschwätzigkeit als Kürze; Distraktor-Kategorie wie alle
  anderen bei ~2–3/6 (Fallsatz-Präferenz, s. Fußnote).
- **B**: sauberstes Instruction-Following der vier; Hauptrisiko ist
  Über-Umformulieren bei den UNCHANGED-Fällen.
- **C**: eine Generation älter, minimal schwächer als B, aber stabil.
- **D**: R1-Distill; Hauptrisiko ist Nicht-Abschalten des Think-Modus und
  Antwortlänge (Längen-/Ein-Zeilen-Zusicherungen reißen).

## Die zu prüfende Vorab-These

> **These (aus dem Auftrag): „B ≈ A bei Rewrite-Qualität, B sauberer bei
> Register/Identity."**

**Mein Abweichungsvermerk, vorab offengelegt:** ich erwarte NICHT „B ≈ A",
sondern **B deutlich vor A** (+6 Punkte Summe). Begründung: die
Claude-Trace-Distillation optimierte auf Dialog-/Assistenz-Verhalten, nicht
auf knappe Instruktionstreue; die dokumentierten Nebenwirkungen
(Identity-Prompt, englische Think-Monologe, Registerkippen) sind Symptome
genau davon. Wenn die Messung „B ≈ A" zeigt, ist meine Vorhersage widerlegt
und die These bestätigt — das wird so berichtet.

**Erwartetes Distillations-Verdikt (A vs. B):** die Distillation kostet
messbar Instruktionstreue und bringt für die Rewriter-/Agenten-Rolle
nichts Nachweisbares. Falls A ≥ B ausfällt, ist das ein starkes Signal für
den Distill-Wert und ich habe mich geirrt.

## Vorhergesagtes Verdikt

**B (Qwen3.5-9B-Instruct) wird WorkflowAgent-Basis und Finetune-Ziel.**
Fallback bei Gate-Riss von B: C. A verliert seinen Kandidatenstatus, wenn
B ihn bei bereinigtem eval_hard um > 2 Punkte schlägt UND bei T5/T6
mindestens gleichzieht.
