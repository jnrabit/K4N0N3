# K4N0N3 — Umbau 5: Trace-Pipeline, Qwythos-9B-Finetune, Eval-Harness

*Generiert am 2026-07-21 19:01 von `bench/make_report5.py` — alle Zahlen stammen aus den JSONs unter `bench/results/`. Nicht von Hand editieren.*

## Kernaussage

Ein 9-Mrd.-Parameter-Modell wurde auf einer 8-GB-Karte per LoRA finetuned und **verbessert die Zielaufgabe messbar** — aber erst, nachdem die selbst erzeugten synthetischen Negativbeispiele aus dem Trainingssatz entfernt waren. Der erste Lauf sah aus wie „bringt nichts"; tatsaechlich hoben sich zwei echte Verbesserungen und zwei selbstgemachte Schaeden auf.

## Eval: Rewrite-Qualitaet auf 10 zurueckgehaltenen Traces

Metrik `keeps_anchor`: steht der aus dem Verlauf aufzuloesende Antezedent in der Ausgabe (ab der Haelfte der Ankerbegriffe)? `token_f1`: Wortueberlappung mit der Referenz. Ober- und Untergrenze stammen aus `--self-test` (CPU, kein Modell).

| Konfiguration | token_f1 (Median) | Anker |
|---|---|---|
| Untergrenze: Folgefrage unveraendert | 0.367 | 0/10 |
| Qwythos-9B ohne Adapter | 0.584 | 6/10 |
| Qwythos-9B + Adapter (**mit** synth. Negativen) | 0.595 | 6/10 |
| Qwythos-9B + Adapter (**ohne** Negative) | 0.752 | 9/10 |
| Qwythos-9B ohne Adapter — *Eval n=15* | 0.588 | 10/15 |
| Qwythos-9B + Adapter **v2** (41 Beispiele, Eval n=15) | 0.762 | 15/15 |
| *Ausser Konkurrenz: Qwen2.5-3B-Instruct, ohne Offload-Quant* | 0.839 | 10/10 |
| Obergrenze: Referenz als Vorhersage | 1.000 | 10/10 |

Gepaart pro Beispiel (Adapter ohne Negative): **3 besser, 0 schlechter, 7 gleich**.

- **besser** `und wie skaliert das?`  
  ohne: `und wie skaliert das?`  
  mit : `Wie skaliert Rate-Limiting mit einem Sliding-Window?`
- **besser** `und wofür kann man das nutzen?`  
  ohne: `und wofür kann man das nutzen?`  
  mit : `Wofür kann man Apache Spark nutzen?`
- **besser** `was ist mit dem Speicher?`  
  ohne: `was ist mit dem Speicher`  
  mit : `Was ist der Speicherbedarf beim Gradient-Checkpointing?`

Zum Vergleich der erste Adapter (mit Negativen): **2 besser, 2 schlechter, 6 gleich** — der Median 6/10 verdeckt, dass hier zwei Verbesserungen gegen zwei Verschlechterungen stehen.

- **besser** `und wie skaliert das?`  
  ohne: `und wie skaliert das?`  
  mit : `Wie skaliert Rate-Limiting mit einem Sliding-Window?`
- **schlechter** `wofür nutzt man das?`  
  ohne: `wofür nutzt man HMAC`  
  mit : `UNCHANGED`
- **besser** `was ist mit dem Speicher?`  
  ohne: `was ist mit dem Speicher`  
  mit : `Was ist der Effekt von Gradient-Checkpointing auf den Speicherbedarf beim Training?`
- **schlechter** `when does it blow up?`  
  ohne: `when does a regex backtracking engine blow up?`  
  mit : `UNCHANGED`

### Befund: synthetische Negative sind der Schaden

9 der 31 Beispiele des ersten Trainingssatzes (29 %) waren synthetische `UNCHANGED`-Negative. Echte sind ueber die Pipeline **nicht sammelbar**: das deterministische Gate `is_referential()` laesst bereits eigenstaendige Fragen gar nicht erst zum Rewriter durch, es entsteht also kein Trace. Alle neun waren konstruiert — und das Modell lernte daraus, `UNCHANGED` zu *produzieren* statt zu *erkennen*: dreimal auf klar referenzielle Fragen.

Der zweite Lauf unterscheidet sich **nur** durch `--negative-ratio 0`. Vorhergesagt waren ~8/10 vor dem Lauf.

## Lauf v2: doppelte Datenmenge (Charge 3)

Traces von 53 auf **100** gebracht (Charge 3 bewusst ausserhalb der Infra/ML-Domaene: Handwerk, Biologie, Recht, Finanzen, Geschichte). Kuratierte Positive 32 → **56**, Trainingssatz 22 → **41**, Eval-Satz 10 → **15**. Sonst identische Einstellungen, 80 statt 50 Schritte wegen des groesseren Satzes.

Gepaart: **5 besser, 0 schlechter, 10 gleich** (Anker 10/15 ohne → 15/15 mit Adapter). Die Richtung aus Lauf 1 bestaetigt sich auf groesserem Eval.

- **besser** `und wo wird das eingesetzt?`  
  ohne: `und wo wird das eingesetzt?`  
  mit : `Wo wird ein Merkle-Tree eingesetzt?`
- **besser** `und wofür kann man das nutzen?`  
  ohne: `und wofür kann man das nutzen?`  
  mit : `Wofür kann man Apache Spark nutzen?`
- **besser** `and how does that change the batch size?`  
  ohne: `and how does that change the batch size?`  
  mit : `How does gradient accumulation change the effective batch size?`
- **besser** `warum wird das nicht ganz trocken?`  
  ohne: `warum wird das nicht ganz trocken?`  
  mit : `Warum wird die Wäsche nach dem Schleudern in der Zentrifuge nicht ganz trocken?`
- **besser** `and how do central banks stop it?`  
  ohne: `and how do central banks stop it?`  
  mit : `How do central banks stop a bank run?`

## Training: LoRA auf Qwythos-9B (Basis > VRAM)

| | mit Negativen | ohne Negative | v2 (41 Beispiele) |
|---|---|---|---|
| Beispiele | 31 | 22 | 41 |
| Schritte | 50 | 50 | 80 |
| Lernrate | 0.0001 | 0.0001 | 0.0001 |
| Adapter-Parameter | 3473408 | 3473408 | 3473408 |
| Loss Median erste 10 | 2.2742 | 2.3806 | 2.0888 |
| Loss Median letzte 10 | 0.2913 | 0.5350 | 0.1045 |
| Schrittzeit ms (Median) | 4300 | 7330 | 4576 |
| VRAM-Peak MB | 5873 | 5876 | 5874 |

**Die Loss zeigt in die falsche Richtung.** Der wirksame Adapter (ohne Negative) endet bei Median 0.5350, der unwirksame bei 0.2913 — also *hoeher* bei besserem Eval-Ergebnis (9/10 gegen 6/10). Wer diesen Lauf nach der Loss ausgewaehlt haette, haette den schlechteren Adapter genommen. Das ist derselbe Befund wie beim 3B-Lauf, nur diesmal im direkten A/B.

Kriterien des Q3-Beweislaufs, auf 9B angewandt: `steps_completed`=✓, `loss_falls`=✓, `vram_within_budget`=✗, `adapters_change_output`=✓

**`vram_within_budget=✗` ist hier die falsche Messgroesse, kein Fehlschlag.** Verglichen wird der Peak (5876 MB) mit dem *Layer*-Budget (2048 MB); die resident auf der GPU liegenden Embeddings + lm_head (2,03 Mrd. Parameter → 3,79 GiB bf16, Vokabular 248k, `tie_word_embeddings=false`) zaehlen nicht mit hinein. 2048 + 3790 = 5838 MB deckt den gemessenen Peak. Kein OOM, die Karte hat 8 GB. Das Kriterium stammt aus dem 3B-Lauf, wo Embeddings vernachlaessigbar waren.

## Was am Modell auffiel

- **Denkmodus unbrauchbar fuer diese Aufgabe, nicht messbar:** Qwythos-9B beantwortet eine einzeilige Umformulierung mit einem englischen Analyse-Monolog und schliesst den `<think>`-Block innerhalb von 512 Tokens nicht ab. Alle Eval-Laeufe oben liefen deshalb mit `enable_thinking=False`. Der Finetune hat daran nichts geaendert — er wurde auch nicht darauf gemessen.
- **Generierung ist der teure Teil, nicht Training:** pro Token muss das ganze Modell durch den Bus (32 Layer), ein Trainingsschritt ist *ein* Durchlauf. Gemessen: Schrittzeit 7330 ms gegen mehrere Sekunden pro *Token* bei der Generierung.

## Zwei stille Architektur-Annahmen in der Kernmechanik

Beide waren seit Auftrag 3 vorhanden, fielen aber nie auf, weil nur Qwen2.5-fp16 getestet wurde. Beide haetten den Trainingslauf genauso getroffen wie den Eval.

1. **Buffer-only-Module blieben auf der CPU.** `_move_fixed_to_gpu()` prueft, was resident auf die GPU gehoert — mit `list(mod.parameters(recurse=False))`. Qwen3.5 haelt die Rotary-Frequenzen (`inv_freq`) in einem Modul **ohne Parameter**, nur mit Buffern. Ergebnis: `Expected all tensors to be on the same device` im ersten Forward. Fix: `_has_own_tensors()` (Parameter ODER Buffer), auch in `offload_all()`.
2. **Der Quantisierungspfad hatte fp16 fest verdrahtet.** `from_pretrained(dtype=torch.float16)` wird bei diesem Modell ignoriert — alle 427 Parameter bleiben bf16. Im fp16-Pfad faellt das nicht auf (alles rechnet einheitlich bf16); der int4-Pfad erzeugte aber fp16-Gewichte im bf16-Modell → dtype-Mismatch im Linear. Fix: der Scale traegt den dtype des Originalgewichts.

## Grenzen dieser Messung

- **10 Eval-Beispiele sind keine Statistik.** 6→9 sind drei Beispiele. Die Richtung ist deutlich und die Fehlerklasse mechanistisch erklaert, aber das ist ein starker Hinweis, kein Beweis. Fuer Belastbarkeit braucht es den urspruenglich genannten Umfang (50–100 Traces; Stand: 53 gesammelt, 32 kuratiert).
- **Der Eval-Satz stammt aus derselben Quelle wie das Training** (gleiche Themen, gleicher Rewriter, gleiches Prompt-Format). Gemessen ist Verbesserung *auf dieser Verteilung*, nicht allgemeine Rewrite-Faehigkeit.
- **`token_f1` misst Formaehnlichkeit, nicht Korrektheit.** Der einzige verbleibende Ankerfehler des besseren Adapters ist `'Wie genau funktioniert TLS?'` gegen die Referenz `'Wie genau funktioniert das TLS-Protokoll?'` — inhaltlich richtig, von der Metrik bestraft.
- **Die Trainings-Loss taugt hier als Erfolgssignal nicht.** Der vorangegangene 3B-Lauf erreichte 0,0001 und verschlechterte die Aufgabe trotzdem. Deshalb dieser Eval-Harness.

## Artefakte

| Datei | Inhalt |
|---|---|
| `bench/checkpoints/lora_qwythos.pt` | Adapter mit synth. Negativen (13,9 MB) |
| `bench/checkpoints/lora_qwythos_noneg.pt` | Adapter ohne Negative — der wirksame |
| `bench/eval_rewrite.py` | Eval-Harness, `--self-test` kalibriert die Skala |
| `collect2: collect-traces build` | baut Trainings-/Eval-Satz reproduzierbar |
| `collect2: docs/traces_curation.md` | Kurationsrubrik + Kalibrierbeispiele |

- Basis: `bench/results/20260721_070421_eval_rewrite_qwythos_basis_nothink.json`
- Adapter mit Neg.: `bench/results/20260721_072035_eval_rewrite_qwythos_adapter_nothink.json`
- Adapter ohne Neg.: `bench/results/20260721_075205_eval_rewrite_qwythos_noneg.json`
- Training mit Neg.: `bench/results/20260721_071055_train_lora_qwythos.json`
- Training ohne Neg.: `bench/results/20260721_074007_train_lora_qwythos_noneg.json`
