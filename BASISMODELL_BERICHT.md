# Basismodell-Entscheid per Messung — Bericht

*Generiert am 2026-07-29 06:57 von `bench/make_report_basemodel.py` — alle Zahlen aus den JSONs unter `bench/results/`. Nicht von Hand editieren.*

## Setup (harte Regel: ein Treiber, eine Quant-Klasse, ein Seed)

| Kandidat | Modell | Quant | arch | Digest |
|---|---|---|---|---|
| **A** | `pdurlej/qwythos-9b-claude-mythos-5-1m:latest` | Q4_K_M | qwen35 | `6efe22249827` |
| **B** | `qwen3.5:9b-q4_K_M` | Q4_K_M | qwen35 | `6488c96fa5fa` |
| **C** | `qwen3:8b` | Q4_K_M | qwen3 | `500a1f067a9f` |
| **D** | `deepseek-r1:8b-0528-qwen3-q4_K_M` | Q4_K_M | qwen3 | `6995872bfe4c` |

Probe-Suite mit **Thinking AN** (dort wird Think-Ökonomie gemessen), eval_hard mit **Thinking AUS** (Vergleichbarkeit mit der bestehenden Methodik). Seed und num_ctx für alle Kandidaten identisch.

### Messkorrektur: ein dokumentierter Nachbesserungslauf für alle

Der Erstlauf lief mit `num_predict=64` (bisherige Methodik). Dabei erzielte **D 0/24 — ein Messartefakt, kein Modellbefund**: bei D unterdrückt `think:false` das Denken nicht, es verschiebt es nur in ein separates `thinking`-Feld; das Token-Budget wird trotzdem verbraucht, der Antworttext blieb 24× leer (`<LEER>`). **Das ist zugleich der vom Auftrag verlangte Befund zu D: Thinking ist dort über die API nicht abschaltbar, nur umleitbar.**

Weil das ein Confounder des Messaufbaus ist und kein Modellverhalten, wurde er per **einem** Nachbesserungslauf **für alle vier gleich** entfernt: `num_predict=512`. Alle Zahlen unten stammen aus diesem Lauf. Der 64er-Erstlauf bleibt in `bench/results/*_base_*` erhalten; bei A/B/C änderte er die Ergebnisse nur um ≤1 Fall (das Budget war für sie nicht bindend), bei D von 0/24 auf einen echten Wert.

## Gates (vorab festgelegte Reihenfolge)

**Gate 2 nach Auftrags-Wortlaut:** ausgeschlossen wird nur, wer Think bei einfachen Aufgaben *nicht abschließt* **UND** *nicht abschaltbar* ist. Ein hoher T7-Think-Wert allein (⚠ = >500 tok, Schwelle aus dem alten Testplan) ist ein berichtetes Warnsignal, kein Ausschlussgrund — sonst wäre ein Kriterium erfunden, das der Auftrag nicht gesetzt hat. „Abschaltbar“ ist belegt, wenn die eval_hard-Läufe mit `think:false` think-freie Antworten lieferten.

| Kandidat | Gate 1 Tool-Disziplin (T4+T5) | Gate 2 Think-Ökonomie | Timeouts | im Ranking? |
|---|---|---|---|---|
| **A** | ✓ (OK (vault_search, query gesetzt) / OK (kein Call)) | ✓ (T7 ~139 tok, abschaltbar: ja) | 0 | ja |
| **B** | ✓ (OK (vault_search, query gesetzt) / OK (kein Call)) | ✓ (T7 ~2526 tok ⚠, abschaltbar: ja) | 0 | ja |
| **C** | ✓ (OK (vault_search, query gesetzt) / OK (kein Call)) | ✓ (T7 ~347 tok, abschaltbar: ja) | 0 | ja |
| **D** | ✗ (FAIL (kein tool_call) / OK (kein Call)) | ✓ (T7 ~123 tok, abschaltbar: ja) | 0 | **nein** |

## eval_hard — roh und bereinigt

> **Auswertungs-Fußnote (Pflicht):** Die vier Vergleichs-Folgefragen `h2-dist-01`, `h2-dist-02`, `h2-dist-03`, `h2-dist-06` sind als **Fallsatz-Präferenz** markiert — 3B und 30B beantworteten sie identisch mit „beide Entitäten“, was für einen Retrieval-Rewriter vertretbar ist; die Re-Spezifikation der Kategorie steht aus. Der Gesamtscore wird deshalb zusätzlich **ohne** diese vier ausgewiesen. **Das Verdikt stützt sich auf die bereinigte Zahl.**

| Kandidat | Satz 1 | Holdout 2 | Summe roh | Summe bereinigt | Latenz median |
|---|---|---|---|---|---|
| **A** | 12/24 | 14/24 | **26**/48 | **26**/44 | 0.9s / 0.9s |
| **B** | 8/24 | 11/24 | **19**/48 | **19**/44 | 1.1s / 1.1s |
| **C** | 20/24 | 20/24 | **40**/48 | **38**/44 | 0.6s / 0.6s |
| **D** | 19/24 | 17/24 | **36**/48 | **35**/44 | 8.3s / 7.5s |

### Pro-Kategorie (Holdout 2)

| Kandidat | distraktor | kein_dump | kontrast | sprache | tiefer_antezedent | unchanged |
|---|---|---|---|---|---|---|
| **A** | 1/6 | 0/2 | 3/4 | 3/3 | 2/4 | 5/5 |
| **B** | 0/6 | 0/2 | 2/4 | 2/3 | 3/4 | 4/5 |
| **C** | 4/6 | 2/2 | 3/4 | 2/3 | 4/4 | 5/5 |
| **D** | 2/6 | 1/2 | 4/4 | 2/3 | 4/4 | 4/5 |

### Nebenbefund: die Distraktor-Fähigkeit existiert doch

Der frühere MoE-Befund lautete: „die Distraktor-Schwäche war eine **Fallsatz-Präferenz, kein Modell-Gap**“ — begründet damit, dass qwen2.5:3b und qwen3:30b-a3b bei Vergleichs-Folgefragen *identisch* beide Entitäten nannten.

**Diese Messung korrigiert das teilweise.** C (qwen3:8b) erreicht Distraktor **5/6 / 4/6** und wählt dabei tatsächlich EINE Entität („Wie viele gleichzeitige Schreiber unterstützt SQLite?“, „Wie verteilt Kubernetes die Last?“). Von den vier als Fallsatz-Präferenz markierten Fällen löst C **2 von 4** korrekt auf (`h2-dist-02`, `h2-dist-03`).

Präzisierung des alten Befunds: die Fähigkeit, sich bei zwei Entitäten zu entscheiden, **existiert** — sie fehlte nur den beiden bis dahin gemessenen Modellen. „Kein Modell-Gap“ war zu stark formuliert; richtig ist: *3B und 30B-a3b teilen die Hedging-Neigung, qwen3:8b nicht*. Die Re-Spezifikation der Kategorie bleibt trotzdem sinnvoll (bei `h2-dist-01` wählt C die FALSCHE Entität — ein anderes Fehlerbild als Hedging), aber sie ist nicht mehr die einzige Erklärung.


## Manuelle Bewertung (T1/T2/T6)

Automatische Checks reichen hier nicht; Rubrik aus `qwythos_testplan.md`. Quelle der Urteile steht in jedem JSON (`manual_verdict_source`).

**T1 — Rewrite Standard-Deixis**

- **A**: BESTANDEN — eigenstaendige Stichwort-Query, Thema (Goertzel+Netzbrummen+ESP32) erhalten, eine Zeile.  
  Ausgabe: `Robustheit Goertzel-Filter bei Netzbrummen auf ESP32`
- **B**: BESTANDEN — eigenstaendige Stichwort-Query, Thema erhalten, eine Zeile.  
  Ausgabe: `Robustheit von Goertzel-Filterung bei netzbrummen auf ESP32`
- **C**: BESTANDEN — vollstaendige eigenstaendige Frage, Thema erhalten, eine Zeile.  
  Ausgabe: `Wie robust ist die Goertzel-Filterung auf einem ESP32 bei Netzbrummen?`
- **D**: BESTANDEN — eigenstaendige Frage, Thema erhalten, eine Zeile.  
  Ausgabe: `wie robust ist der goertzel algorithmus gegen störungen durch netzbrummen?`

**T2 — Referenz über zwei Sprünge**

- **A**: BESTANDEN — richtige Referenz (LSB-Timing-Jitter) aufgeloest, nicht Lorenz.  
  Ausgabe: `Ist LSB-Timing-Jitter auf einem ESP32 ausreichend für kryptographische Zwecke?`
- **B**: DURCHGEFALLEN — antwortet UNCHANGED auf eine klar referenzielle Folgefrage; loest den Antezedenten gar nicht auf. Gleiche Fehlerklasse wie unchanged 1/5 im eval_hard.  
  Ausgabe: `UNCHANGED`
- **C**: BESTANDEN — richtige Referenz, praezisester der vier (nennt auch die Quelle des Jitters).  
  Ausgabe: `Ist LSB-Timing-Jitter aus Timer-Messungen bei asynchronen Ereignissen fuer kryptographische Zwecke a`
- **D**: BESTANDEN — richtige Referenz (LSB-Timing als Entropiequelle).  
  Ausgabe: `Kann LSB-Timing-Jitter als Entropiequelle fuer kryptographisch sichere Zufallszahlengeneratoren verw`

**T6 — Komprimiertes Deutsch / Register**

- **A**: DURCHGEFALLEN — antwortet auf deutsche Frage ENGLISCH und mit einbetoniertem Identity-Prompt ('Qwythos here from Empero AI'). Genau die dokumentierte Distillations-Nebenwirkung, hier reproduziert.  
  Ausgabe: `Qwythos here from Empero AI — k=60 still makes sense for 260k docs; you can keep it or tune slightly`
- **B**: DURCHGEFALLEN — antwortet auf deutsche Frage auf CHINESISCH (Hoeflichkeitsform). Vollstaendiger Sprachbruch.  
  Ausgabe: `您好，我理解您对于文档处理系统中参数设置的困惑。关于 `rrf`（Reciprocal Rank Fusion）、融合算法以及常数 $k$ 的选择问题，这通常涉及到搜索或向量数据库的排名优化场景。  `
- **C**: BESTANDEN — durchgehend Deutsch, technisch, Register gehalten, RRF-Formel korrekt.  
  Ausgabe: `Die **RRF (Reciprocal Rank Fusion)**-Methode kombiniert Suchergebnisse aus mehreren Quellen, indem d`
- **D**: BESTANDEN mit Vorbehalt — Sprache/Register gehalten (Deutsch, lockerer Ton), aber die angegebene RRF-Formel ist inhaltlich falsch.  
  Ausgabe: `Okay, klar:  RRF (Reciprocal Rank Fusion) ist eine Methode, um mehrere Suchergebnisse aus verschiede`


## Distillations-Verdikt: A vs. B

**Die Distillation trägt messbar bei.** A liegt bereinigt 7 Punkte vor der nackten Basis B.

- A bereinigt **26/44**, B bereinigt **19/44**.

**Das ist die Antwort auf die Frage, für die dieser Auftrag existiert — und sie fällt gegen beide Vorab-Aussagen aus.** Die Auftrags-These lautete „B ≈ A bei Rewrite-Qualität, B sauberer bei Register/Identity“; meine vorregistrierte Gegenthese lautete „B deutlich vor A (+6)“. Gemessen ist das Gegenteil von beidem: **A liegt 7 Punkte vor B.**

Woran es liegt, zeigt der Kategorien-Aufriss: B versagt bei `unchanged` (formuliert eigenständige Fragen um und hängt ihnen fremden Kontext aus der Historie an) und antwortet umgekehrt `UNCHANGED` auf klar referenzielle Fragen (T2, `dist-06`). Es trifft die Rewrite-Entscheidung also in beide Richtungen falsch. Die Distillation hat A genau diese Instruktionstreue beigebracht.

**Der Preis der Distillation ist aber belegt und nicht klein:** A fällt in T6 durch — es antwortet auf eine deutsche Frage **auf Englisch** und stellt sich dabei selbst vor („Qwythos here from Empero AI“). Der einbetonierte Identity-Prompt und das Registerkippen sind damit reproduziert, nicht nur behauptet. Für eine Agenten-Rolle mit deutschem Nutzertext ist das ein echter Mangel — er kostet A hier aber nicht den zweiten Platz, weil B in T6 noch härter durchfällt (Antwort auf **Chinesisch**).

## Ranking (nur Kandidaten, die beide Gates bestanden)

1. **C — Qwen3-8B**: bereinigt 38/44 (roh 40/48)
2. **A — Qwythos-9B (Claude-Distill)**: bereinigt 26/44 (roh 26/48)
3. **B — Qwen3.5-9B-Instruct (nackte Basis)**: bereinigt 19/44 (roh 19/48)

**Verdikt: C — Qwen3-8B wird WorkflowAgent-Basis und Finetune-Ziel.**

Nicht im Ranking (Gate gerissen, aber vollständig gemessen und berichtet): **D**.

**Zu D ausdrücklich, damit der Ausschluss nicht falsch gelesen wird:** D ist mit bereinigt 35/44 der **zweitbeste Rewriter im Feld** — es scheitert nicht an der Sprachfähigkeit, sondern an Gate 1: es ruft das Pflicht-Tool nicht auf, sondern halluziniert stattdessen einen medizinischen Kontext zum erfundenen Begriff. Für die Rewriter-Rolle allein wäre D ein ernsthafter Kandidat; für die **Agenten**-Rolle disqualifiziert es die fehlende Tool-Disziplin — und genau die müsste ein Finetune erst erzeugen, was laut Lehrer-Modell-Befund das schlechte Geschäft ist. Dazu kommt betrieblich: ~8 s statt ~0,6 s pro Rewrite, weil sich das Denken nicht abschalten lässt.

## Vorhersagen-Abgleich

Vorregistriert in `bench/predictions_basemodel.md` vor der ersten Messung.

| Kandidat | vorhergesagt (Summe roh) | gemessen (Summe roh) | Δ |
|---|---|---|---|
| **A** | 27 | 26 | -1 |
| **B** | 33 | 19 | -14 |
| **C** | 31 | 40 | +9 |
| **D** | 23 | 36 | +13 |

**Bilanz der Vorhersagen: überwiegend falsch, und zwar deutlich.** Nur A wurde getroffen (Δ−1). B wurde um 14 Punkte überschätzt, C um 9 und D um 13 unterschätzt. Auch die qualitativen Vorhersagen stimmten nur teilweise: für A war „T7 lang (>500 tok)“ vorhergesagt — gemessen sind es ~139; für D „Timeout erwartet“ — es gab keinen einzigen Timeout, dafür riss D das Tool-Gate, was als „wackelig“ immerhin angedeutet war.

**Die geprüfte These des Auftrags („B ≈ A bei Rewrite-Qualität, B sauberer bei Register/Identity“) ist widerlegt** — in beiden Hälften: B ist bei der Rewrite-Qualität nicht gleichauf, sondern deutlich schlechter, und bei Register/Identity nicht sauberer, sondern schlechter (Chinesisch statt Deutsch gegen Englisch statt Deutsch). Meine eigene Gegenthese ist ebenfalls widerlegt, nur in die andere Richtung. Der reale Befund war von keiner der beiden Seiten vorhergesagt.

## Artefakte

| Datei | Inhalt |
|---|---|
| `bench/probe_suite.py` | T1–T7, ein Treiber für alle Kandidaten |
| `bench/eval_hard_ollama.py` | eval_hard, Bewertung unverändert importiert |
| `bench/predictions_basemodel.md` | Vorregistrierung |

- A Probe: `bench/results/20260729_062659_probe_A.json`
- A eval S1: `bench/results/20260729_064416_eval_hard_ollama_np512_A_set1.json`
- A eval H2: `bench/results/20260729_064445_eval_hard_ollama_np512_A_h2.json`
- B Probe: `bench/results/20260729_063120_probe_B.json`
- B eval S1: `bench/results/20260729_064532_eval_hard_ollama_np512_B_set1.json`
- B eval H2: `bench/results/20260729_064602_eval_hard_ollama_np512_B_h2.json`
- C Probe: `bench/results/20260729_063435_probe_C.json`
- C eval S1: `bench/results/20260729_064644_eval_hard_ollama_np512_C_set1.json`
- C eval H2: `bench/results/20260729_064702_eval_hard_ollama_np512_C_h2.json`
- D Probe: `bench/results/20260729_063658_probe_D.json`
- D eval S1: `bench/results/20260729_065029_eval_hard_ollama_np512_D_set1.json`
- D eval H2: `bench/results/20260729_065337_eval_hard_ollama_np512_D_h2.json`
