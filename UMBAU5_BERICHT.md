# K4N0N3 — Umbau 5: Trace-Pipeline, Qwythos-9B-Finetune, Eval-Harness

*Generiert am 2026-07-22 14:10 von `bench/make_report5.py` — alle Zahlen stammen aus den JSONs unter `bench/results/`. Nicht von Hand editieren.*

## Kernaussage

Ein 9-Mrd.-Parameter-Modell wurde auf einer 8-GB-Karte per LoRA finetuned und **verbessert die Zielaufgabe messbar**: auf einem unabhaengigen, vor der Datensammlung festgeschriebenen Testsatz **13/24 ohne gegen 21/24 mit Adapter**.

**Ergebnis-Artefakt ist `bench/checkpoints/lora_qwythos_v2.pt`** (41 Trainingsbeispiele). Der spaetere v3 mit gezielt nachgesammelten Daten (56 Beispiele) liegt mit 20/24 darunter — die Nachsammlung brachte nichts, siehe „Holdout 2".

**Fuer den PRODUKTIVEN Rewriter ist die Antwort trotzdem: kein Finetune.** Der 9B laeuft auf dieser 8-GB-Karte nur ueber Offload (~300 s/Rewrite, PCIe-gebunden). Ein resident laufendes qwen2.5:3b ist der einsetzbare Weg — und dessen Basis (17/24) schlaegt keiner der drei 3B-Finetunes. Siehe „Der einsetzbare Weg".

Drei Befunde waren dafuer noetig, alle gegen den ersten Anschein:

1. **Die selbst erzeugten synthetischen Negativbeispiele haben den Finetune sabotiert.** Der erste Lauf sah aus wie „bringt nichts"; tatsaechlich hoben sich zwei echte Verbesserungen und zwei selbstgemachte Schaeden auf. Sichtbar nur im gepaarten Vergleich pro Beispiel — der Median verdeckt es vollstaendig.
2. **Die Loss taugt hier nicht zur Auswahl.** Der bessere Adapter hat die HOEHERE Endloss. Wer nach Loss ausgewaehlt haette, haette den schlechteren genommen.
3. **Mehr Daten sind nicht automatisch besser.** Gezielt gesammelte Beispiele fuer die gemessenen Schwaechen haben das Ergebnis leicht verschlechtert, weil das Lehrer-Modell die Zielfaehigkeit selbst nicht beherrscht.

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

## Harter Eval 1: 24 handgeschriebene Faelle

*Achtung: Charge 4 wurde spaeter gezielt fuer die Kategorien dieses Satzes gesammelt. Die Zahlen hier stammen aus der Messung DAVOR (Adapter v2) und sind insofern noch unabhaengig; fuer alles danach gilt „Holdout 2" als Massstab.*

Der Eval oben ist zu milde: die Referenzen stammen vom SELBEN Rewriter, der die Trainingsdaten erzeugt hat, und `keeps_anchor` besteht schon bei der Haelfte der Ankerbegriffe — ein blosser History-Dump haette 100 % erreicht. `bench/eval_hard.py` prueft stattdessen pro Fall, was drin sein MUSS und was NICHT drin sein darf, dazu Laenge, Sprache und Register. Bestanden nur, wenn ALLE Zusicherungen halten (`strict`). Die Faelle sind von Hand geschrieben, also unabhaengig von der Trainingsverteilung.

Schaerfe VOR der Messung kalibriert (`--self-test`, CPU): Ideal-Antwort 24/24, History-Dump 0/24, Distraktor-Antwort 0/24.

| Kategorie | ohne Adapter | mit Adapter v2 | prueft |
|---|---|---|---|
| distraktor | 1/6 | **4/6** | Historie nennt zwei Entitaeten — nur eine ist gemeint |
| kein_dump | 1/2 | **2/2** | lange Historie, kurze Antwort erzwungen |
| kontrast | 2/4 | **4/4** | „und ohne?" — das Gegenteil muss verschwinden |
| sprache | 3/3 | **3/3** | englisch rein, englisch raus |
| tiefer_antezedent | 0/4 | **3/4** | Bezug liegt zwei Turns zurueck |
| unchanged | 5/5 | **5/5** | schon eigenstaendig → nicht anfassen |
| **gesamt** | **12/24 (50%)** | **21/24 (88%)** | alle Zusicherungen |

Haeufigste Verletzung ohne Adapter: `antezedent` (8x), `kein_konjunktionsanfang` (8x), `kein_postgres` (1x) — also weitgehend die unveraendert durchgereichte Frage.

**`unchanged` 5/5 in beiden Laeufen.** Das war der wichtigste Test: die Sorge, dass das Weglassen der synthetischen Negative (Lauf v2) das Modell zum Ueber-Umformulieren bringt, hat sich NICHT bestaetigt.

Was mit Adapter noch fehlschlaegt:

- `dist-03` [antezedent, kein_postgres, kein_postgresql] `wie viele gleichzeitige Schreiber vertraegt das?` → `Wie viele gleichzeitige Schreiber verträgt PostgreSQL?`
- `dist-06` [kein_rsa] `welche Schluessellaengen sind da ueblich?` → `Welche Schlüssellängen sind für RSA und AES üblich?`
- `hist-04` [antezedent] `und wie prueft man damit einen einzelnen Eintrag?` → `Wie prüft man damit einen einzelnen Eintrag?`

**Offengelegte Korrektur:** `dist-02` war eine fehlerhafte Zusicherung, kein Modellfehler — bei „warum ist das schneller?" ist die Frage implizit vergleichend, die Nennung des Vergleichspartners also korrekt. Nach Sichtung der Ausgaben entfernt. Die Korrektur hebt BEIDE Laeufe um je einen Fall (11→12 und 20→21) und veraendert den Abstand nicht. Die uebrigen Fehlschlaege bleiben stehen.

## Holdout 2: der unabhaengige Test

Charge 4 (36 Paare) zielte gezielt auf die oben gemessenen Schwaechen — damit misst `eval_hard.jsonl` nicht mehr Generalisierung, sondern In-Distribution-Leistung. `eval_hard2.jsonl` wurde deshalb **vor** der Datensammlung geschrieben und in einem eigenen Commit festgeschrieben (durchgehend andere Entitaeten und Domaenen). Das ist die unabhaengige Zahl.

| Kategorie | ohne Adapter | v2 (41 Beisp.) | v3 (56 Beisp., Charge 4) |
|---|---|---|---|
| distraktor | 3/6 | 3/6 | 4/6 |
| kein_dump | 1/2 | 2/2 | 2/2 |
| kontrast | 2/4 | 4/4 | 3/4 |
| sprache | 2/3 | 3/3 | 3/3 |
| tiefer_antezedent | 0/4 | 4/4 | 3/4 |
| unchanged | 5/5 | 5/5 | 5/5 |
| **gesamt** | **13/24** | **21/24** | 20/24 |

### Befund: Charge 4 hat nichts gebracht

v3 (20/24) liegt unter v2 (21/24). Der Unterschied ist Rauschen, aber es gibt keinen Grund, v3 zu bevorzugen — und keinen, die 36 zusaetzlichen Traces als Fortschritt auszuweisen. **v2 bleibt der Adapter der Wahl.**

Aufgeschluesselt: v3 gewinnt einen Distraktor-Fall (`SSD- oder HDD-Speicher` → `eine SSD`), verliert aber je einen bei tiefem Antezedenten und Kontrast. Das Distraktor-Training wirkt also — nur zu schwach, um die Verwaesserung aufzuwiegen.

**Warum es schiefging, steht in der Kuration:** von 14 gesammelten Distraktor-Paaren ueberlebten **3**, weil qwen2.5:3b bei zwei Entitaeten fast immer hedgt (beide nennt). Drei Beispiele gegen 56 im Satz bewegen nichts, waehrend die 15 neuen Beispiele insgesamt die Verteilung verschoben und zwei bestehende Faehigkeiten leicht verwaesserten.

**Die Lehre:** gezielt sammeln funktioniert nur, wenn das Lehrer-Modell die Zielfaehigkeit selbst beherrscht. Aus einer Quelle, die sich bei zwei Entitaeten nicht entscheiden kann, laesst sich die Entscheidung nicht destillieren — dafuer braeuchte es ein staerkeres Modell oder handgeschriebene Targets.

**Korrektur zum Protokoll:** waehrend des Laufs wurde der Sprung beim tiefen Antezedenten (0/4 → 3/4) zunaechst Charge 4 zugeschrieben. Das war falsch — verglichen worden war gegen die BASIS, nicht gegen v2. v2 hatte dort bereits 4/4, ganz ohne Zwei-Turn-Trainingsdaten.

## Der einsetzbare Weg: qwen2.5:3b (resident statt offloaded)

Der 9B laeuft auf dieser 8-GB-Karte nur ueber Offload — PCIe-4.0-x8-gebunden, ~300 s pro Rewrite, nicht interaktiv. Ein kleines Modell passt resident: kein Transfer pro Token, Millisekunden statt Minuten. Also dieselbe Finetune-Frage fuer `qwen2.5:3b`, gemessen auf **demselben** unabhaengigen Holdout 2.

| Variante | gesamt | Distraktor | UNCHANGED |
|---|---|---|---|
| **qwen2.5:3b Basis (Pipeline-Default)** | **17/24** | 1/6 | 5/5 |
| + Finetune (56 pos, `--negative-ratio 0`) | **17/24** | 3/6 | 3/5 |
| + Finetune (56 pos + 9 neg, `0.15`) | **16/24** | 1/6 | 3/5 |

**Kein Finetune schlaegt die Basis.** Der all-positive Lauf tauscht nur um: **+Distraktor** (1/6→3/6, das Modell lernt sich zu entscheiden) gegen **−UNCHANGED** (5/5→3/5, es formt jetzt schon eigenstaendige Fragen um) — netto 17=17. Der Versuch, das mit 15 % synthetischen Negativen zu balancieren, war die schlechteste Variante (16/24): er zerstoerte den Distraktor-Gewinn UND stellte UNCHANGED nicht wieder her.

**Vorhergesagt waren ~19/24, gemessen 16/24 — die Hypothese ist widerlegt.** Damit ist der Negativ-Befund doppelt belegt: die synthetischen UNCHANGED-Negative helfen KEINEM Modell (sie verschlechterten schon den 9B). Sie sind konstruiert, nicht echt, und uebertragen nicht auf echte eigenstaendige Fragen. Echte Negative kann die Pipeline nicht sammeln — `is_referential()` faengt eigenstaendige Fragen vor dem Rewriter ab. Diese Gate-Grenze deckelt den Finetune-Ansatz.

**Konsequenz fuer den Betrieb:** der Rewriter bleibt auf der unfinetunten `qwen2.5:3b` (bereits `rewrite_model`/`decompose_model` im Default) — resident, schnell, und so gut wie jeder Finetune hier. Der 9B-Finetune bleibt als belegte Faehigkeits-Demonstration (13→21 auf Holdout 2), nicht als Produktionsartefakt.

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

- **Die Stichproben sind klein.** 24 harte Faelle und 15 zurueckgehaltene Traces; ein Unterschied von zwei, drei Faellen ist Rauschen. Die Richtung ist ueber beide Evals konsistent und die Fehlerklassen sind benannt — belastbar im statistischen Sinn ist das trotzdem nicht.
- **Der Trace-Eval stammt aus derselben Quelle wie das Training** (gleiche Themen, gleicher Rewriter, gleiches Prompt-Format) und misst Verbesserung nur *auf dieser Verteilung*. Genau deshalb gibt es den harten Eval mit handgeschriebenen Faellen; dessen Zahlen sind die aussagekraeftigeren.
- **Beide harten Evals sind von derselben Instanz geschrieben, die auch kuriert und trainiert hat.** Gleiche blinde Flecken wirken in beiden Richtungen. Die zeitliche Trennung (Holdout 2 vor der Datensammlung festgeschrieben) schuetzt gegen nachtraegliches Zurechtbiegen, nicht gegen geteilte Annahmen — ein von aussen geschriebener Fallsatz waere der naechste Schritt.
- **Der Distraktor-Fall bleibt ungeloest.** 3/6 ohne Adapter, 3/6 mit v2, 4/6 mit v3: die Faehigkeit, sich bei zwei Entitaeten zu entscheiden, ist mit diesen Daten nicht antrainierbar.
- **`token_f1` misst Formaehnlichkeit, nicht Korrektheit.** Der einzige verbleibende Ankerfehler des besseren Adapters ist `'Wie genau funktioniert TLS?'` gegen die Referenz `'Wie genau funktioniert das TLS-Protokoll?'` — inhaltlich richtig, von der Metrik bestraft.
- **Die Trainings-Loss taugt hier als Erfolgssignal nicht.** Der vorangegangene 3B-Lauf erreichte 0,0001 und verschlechterte die Aufgabe trotzdem. Deshalb dieser Eval-Harness.

## Artefakte

| Datei | Inhalt |
|---|---|
| **`bench/checkpoints/lora_qwythos_v2.pt`** | **Ergebnis-Adapter** (41 Beispiele, 21/24) |
| `bench/checkpoints/lora_qwythos_v3.pt` | Charge-4-Variante (56 Beispiele, 20/24) — nicht besser |
| `bench/checkpoints/lora_qwythos.pt` | Adapter mit synth. Negativen — der Negativ-Befund |
| `bench/checkpoints/lora_qwythos_noneg.pt` | erster wirksamer Adapter (22 Beispiele) |
| `bench/data/eval_hard2.jsonl` | unabhaengiger Holdout, vor Charge 4 festgeschrieben |
| `bench/checkpoints/lora_qwen3b_rewrite.pt` | qwen2.5:3b-Finetune (all-positiv) — schlaegt Basis nicht |
| `bench/checkpoints/lora_qwen3b_neg15.pt` | qwen2.5:3b + 15 % Negative — schlechteste Variante |
| `bench/eval_rewrite.py` | Trace-Eval (Adapter an/aus, gepaart) |
| `bench/eval_hard.py` + `bench/data/eval_hard.jsonl` | harter Eval, 24 handgeschriebene Faelle |
| `collect2: collect-traces build` | baut Trainings-/Eval-Satz reproduzierbar |
| `collect2: docs/traces_curation.md` | Kurationsrubrik + Kalibrierbeispiele |

- Basis: `bench/results/20260721_070421_eval_rewrite_qwythos_basis_nothink.json`
- Adapter mit Neg.: `bench/results/20260721_072035_eval_rewrite_qwythos_adapter_nothink.json`
- Adapter ohne Neg.: `bench/results/20260721_075205_eval_rewrite_qwythos_noneg.json`
- Training mit Neg.: `bench/results/20260721_071055_train_lora_qwythos.json`
- Training ohne Neg.: `bench/results/20260721_074007_train_lora_qwythos_noneg.json`
