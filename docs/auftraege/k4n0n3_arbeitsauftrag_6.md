# K4N0N3 — Arbeitsauftrag 6: Transferpfad-Endausbau (Staging-Blob, Overlap, spekulatives Decoding)

## Kontext

Stand: 3B int8 bei 0,94 tok/s. Das entspricht einer effektiven Transferrate
von ~2,8 GB/s (0,94 tok/s × ~3 GB int8/Token). Die manuelle Diagnose
(2026-07-28) korrigiert die frühere Annahme „~12 GB/s Gen4-x8-Maximum": eine
Roh-Bandbreiten-Probe (einzelner gepinnter Blob, 16 MB–1 GB) liefert auf dieser
Maschine ebenfalls nur **2,84 GB/s** — pinned wie pageable, größenunabhängig,
unverändert unter `force_performance_level=high`. Effektive Rate und
Roh-Blob-Rate fallen also zusammen. Das verschiebt die Frage: NICHT „warum ist
die effektive Rate 4× unter roh" (sie ist es nicht), sondern **„warum liegt die
Roh-Rate selbst 4–5× unter der Gen4-x8-Theorie"** — Bus, Link-Zustand oder
Treiber-Pfad. Bereits gemessen: Link unter Last 16,0 GT/s ×8, ReBAR an,
Clock/Power ausgeschlossen; die Ursache der 2,84 GB/s bleibt offen und ist
Gegenstand von T. Für einen ZUSÄTZLICHEN Verlust im K4N0N3-Layer-Pfad (viele
kleine Copies statt großer Blob) zwei ältere Verdächtige, beide seit Auftrag 4
notiert — mit der Vorab-Einschränkung, dass die effektive Rate schon auf der
Roh-Blob-Rate liegt, ihr Beitrag also klein sein dürfte:

1. **Fragmentierung:** der Upload macht pro Parameter einzelne
   `.to()`-Calls — viele kleine PCIe-Transaktionen statt einem Block pro
   Layer. Passt zum Befund, dass der Pin-Fix (12/36 → 36/36 gepinnt) nur
   3 % brachte: async nützt nichts, wenn jeder Einzeltransfer zu klein ist,
   um Latenz zu verstecken.
2. **Fehlender Overlap:** Warm Forward verhält sich wie Transfer + Compute
   in Summe statt max(Transfer, Compute) — der Prefetch-Stream überlappt
   nicht wirklich mit der Rechnung.

Dazu der einzige Trick, der das Grundproblem angreift statt es zu polieren:
**spekulatives Decoding** amortisiert die Transferkosten über mehrere
Tokens — ein residentes Draft-Modell schlägt k Tokens vor, das gestreamte
Modell verifiziert alle k in EINEM Durchlauf. Training profitiert aus
demselben Grund (ein Durchlauf, viele Tokens) schon heute.

**Ehrliche Einordnung vorab (gehört auch in den Bericht):** Dieser Auftrag
ist das Performance-Schlusskapitel des Forschungsartefakts. Ziel ist, die
Lücke zum eigenen theoretischen Maximum zu schließen und den Abstand zur
Ollama-Referenz korrekt neu zu beziffern — nicht, mit llama.cpp zu
konkurrieren. Erwartungsrahmen: 0,94 → 3–5 tok/s wäre voller Erfolg.
Interaktiv wird das nicht, und das steht vorher fest.

**Eingangsdaten aus manueller Diagnose (2026-07-28, liegen vor):** Link-Status
unter Last = 16,0 GT/s ×8 (Gen4, ReBAR an); Roh-H2D-Bandbreite = 2,84 GB/s
(großer Blob, `scratchpad/pcie_bw2.py`); prompt/decode-Split der MoE-Läufe
(betrifft nur die Ollama/llama.cpp-Pfade, nicht diesen Offload-Auftrag). Damit
sind T1 und die Roh-Bandbreite aus T2 bereits erledigt — T holt nur noch die
Ist-Bandbreite des echten 3B-int8-Layer-Uploads nach und bildet das Verhältnis.
Starke Vorab-Erwartung: Ist ≈ Roh (die effektive 3B-Rate deckt sich mit der
Roh-Blob-Rate) → **T-Gate schließt, U wird übersprungen.** U bleibt trotzdem
als Contingency ausformuliert (falls die Ist-Messung doch Kopfraum zeigt).

Umgebung unverändert. Reihenfolge T → U → V → W. Regeln wie immer:
keine Zahl ohne Harness-JSON, bei verfehlten Kriterien stoppen und
dokumentieren.

---

## Auftrag T — Diagnose & Instrumentierung (entscheidet, ob U überhaupt lohnt)

### T1: Link-Status in den Harness

`bench/harness.py` erfasst pro Lauf `pcie_link_speed` und `pcie_link_width`
aus `/sys/class/drm/card*/device/current_link_{speed,width}` — gelesen
WÄHREND der Messphase (nicht davor: ASPM trainiert den Link im Idle
runter). Felder in jedes JSON.

### T2: Transfer-Mikrobenchmark (`bench/pcie_probe.py`)

Drei Messungen, jeweils Median aus 10 Wiederholungen, als JSON:

1. **Roh-Bandbreite:** ein einzelner gepinnter 150-MB-Blob per
   `.to("cuda", non_blocking=True)` + Event-Timing → erreichbare GB/s
   als Referenzlinie dieser Maschine.
2. **Ist-Zustand:** ein echter 3B-int8-Layer über den bestehenden
   Upload-Pfad → effektive GB/s (Layer-Bytes / gemessene Zeit) und
   Anzahl der Einzel-Copies (Zähler in den Upload-Pfad instrumentieren).
3. **Fragmentierungs-Simulation:** dieselben Bytes als N Einzeltensoren
   in den Größen der echten Layer-Params → isoliert den
   Fragmentierungsanteil von allem anderen.

### Entscheidungs-Gate

- Ist-Bandbreite ≥ 80 % der Roh-Bandbreite → Fragmentierung ist NICHT das
  Problem; U wird ÜBERSPRUNGEN (Befund dokumentieren, direkt zu V, und
  der fehlende Overlap wird dort bzw. in W als offener Punkt vermerkt).
- Ist-Bandbreite deutlich darunter → U ist begründet; Zielwert für U ist
  die gemessene Roh-Bandbreite.

---

## Auftrag U — Staging-Blob + Double-Buffering (nur wenn T-Gate es begründet)

### U1: Master als zusammenhängender Blob

Pro Layer EIN gepinnter uint8-Blob statt vieler Einzeltensoren, plus
Manifest: `[(name, offset, nbytes, dtype, shape, kind)]` mit
kind ∈ {plain, q8, q4, scale}. Offsets auf 16 Bytes gepadded
(Alignment für spätere dtype-Views). Quantisierte Einträge legen q und
scale als getrennte Manifest-Einträge im selben Blob ab. Aufbau ersetzt
die bisherige Master-Dict-Struktur hinter derselben Schnittstelle;
Zwei-Pass-Logik (Auftrag O) und Partial-Pinning bleiben erhalten —
ein ungepinnter Layer ist einfach ein ungepinnter Blob.

### U2: GPU-Slot-Ring + Upload als ein Copy

- Ring aus (prefetch_depth + 1) GPU-Slots, jeder so groß wie der größte
  Layer-Blob. Slots werden einmal allokiert, nie freigegeben —
  eliminiert nebenbei Allocator-Churn.
- Prefetch = EIN `blob.to(slot, non_blocking=True)` im Prefetch-Stream,
  Event danach.
- Konsum (`_ensure_on_gpu`): Event abwarten, dann pro Manifest-Eintrag
  View in den Slot schneiden (`slot[a:b].view(dtype).view(shape)`),
  quantisierte Einträge dequantisieren (wie gehabt, aus den Views),
  `p.data` setzen. `record_stream`-Behandlung auf dem Slot, Kommentar
  wie an den bestehenden Stellen.
- Slot-Lebenszyklus: frei bei Offload des Layers. Assertion, dass nie
  mehr Layer GPU-resident sind als Slots existieren (sonst ist die
  Budget-Logik verletzt — harter Fehler, kein stilles Überschreiben).
- Wichtig: `p.data` zeigt bei plain-fp16-Einträgen direkt in den Slot —
  beim Offload/Drop müssen diese Views gelöst werden BEVOR der Slot
  wiederverwendet wird (sonst rechnet ein alter Layer auf neuen Bytes).
  Test dafür schreiben (zwei Layer nacheinander durch denselben Slot,
  Logits gegen Referenz).

### U3: Overlap verifizieren, nicht behaupten

Messgröße: warm_forward_ms gegen die Summe (gemessene Transferzeit aller
Layer + gemessene reine Compute-Zeit). Ziel: warm_forward nähert sich
max() statt Summe. Wenn nicht: per Event-Timestamps pro Layer prüfen, ob
der Prefetch des Layers n+1 zeitlich im Compute von Layer n liegt — und
den Blocker benennen (typischer Kandidat: der Dequant läuft im
Prefetch-Stream und serialisiert gegen den Copy).

### Akzeptanz U

- Mechanik-Greedy-Check (mit vs. ohne Offloading, identische Tokens) —
  hartes Kriterium, gerade wegen der View-Wiederverwendung.
- Ist-Bandbreite (T2-Messung wiederholt) ≥ 90 % der Roh-Bandbreite.
- warm_forward_ms und tok/s im Harness, ehrlicher Vergleich zur
  Auftrag-4-Basis. Training-Smoke (10 LoRA-Schritte 3B) läuft und
  Loss fällt — der Trainingspfad nutzt denselben Core.
- CPU-only-Pfad degradiert sauber.

---

## Auftrag V — Spekulatives Decoding (Transfer über k Tokens amortisieren)

### V1: Draft-Modell + Verdrahtung

- Draft: Qwen2.5-0.5B fp16, VOLLSTÄNDIG GPU-resident (~1 GB — passt
  neben Budget + Reserve; im MemoryManager als fixe Reserve verbuchen,
  Mechanik aus Auftrag E wiederverwenden). Gleicher Tokenizer wie das
  3B-Ziel (Voraussetzung von HF assisted generation — verifizieren,
  Vokabular-Identität prüfen, sonst harter Fehler mit Meldung).
- Verdrahtung über den vorhandenen HF-Mechanismus, KEIN eigener
  Decoding-Loop: `ZeroFlushModel.generate(..., assistant_model=draft)`
  hinter neuem Parameter `speculative: bool = False` (+ optional
  `draft_model_name`). HF übernimmt Vorschlagen, Batch-Verifikation
  und KV-Rollback.
- Erwartete Interaktion mit den Hooks: die Verifikations-Forwards laufen
  durch dasselbe gehookte Modell, nur mit >1 Token pro Durchlauf — die
  Mechanik ist dieselbe. Instrumentierung: Zähler "Layer-0-Pre-Hook-
  Feuerungen pro generiertem Token" ins Harness-JSON — das ist die
  direkte Messung des Amortisierungsfaktors (1,0 = kein Gewinn;
  0,25 = Transfer auf ein Viertel).

### V2: Messen

- Konfigurationen: 3B int8, greedy, max_new_tokens=64: baseline vs.
  speculative mit num_assistant_tokens ∈ {4, 8, 12}.
- **Hartes Korrektheitskriterium:** spekulatives Greedy-Decoding ist
  verlustfrei — die Token-Sequenz MUSS identisch zur nicht-spekulativen
  Greedy-Referenz sein. Jede Abweichung ist ein Bug (vermutlich
  Hook×KV-Rollback), dann stoppen und Repro bauen.
- Berichten: tok/s, Amortisierungsfaktor, Akzeptanzverhalten über k
  (steigende k lohnen nur bei hoher Akzeptanz — den Sweet Spot aus den
  Daten benennen, nicht raten).

### Akzeptanz V

Greedy-Identität bestanden; tok/s-Tabelle über k; Verdikt in einem Satz,
welcher k-Wert Default wird (oder dass es sich nicht lohnt — auch das
mit Zahlen).

---

## Auftrag W — Bericht (Schlusskapitel Inference)

Via make_report: Tabelle Auftrag-4-Stand vs. U vs. U+V, PCIe-Link-Felder
ausgewiesen, Roh- vs. Ist-Bandbreite, Amortisierungsfaktor. Die
Ollama-Referenzzeile (74 tok/s) bleibt drin; der Abstand wird neu
beziffert. Abschlusssatz zur Einordnung: was der Streaming-Pfad auf
dieser Hardware maximal kann, und dass damit das Inference-Kapitel
geschlossen ist — verbleibende Nische unverändert Training
(+ ggf. Batch-Generation, wo tok/s egal sind).

**Datenpunkt aus der Diagnose (llama.cpp, MoE-Placement) — ehrlich einordnen,
nicht als K4N0N3-Ergebnis verkaufen:** für MoE-Modelle (nicht den dichten 9B)
liefert ein `llama-server` mit `--override-tensor 'ffn_.*_exps.=CPU'`
30B-Qualität (18/24 auf beiden harten Sätzen, ±1 zu Ollama) bei ~5,4 s/Rewrite
— ~9× schneller als Ollamas statischer 68/32-Split, weil nur die dünn-aktiven
Experten-FFNs im RAM liegen und Attention/Router/KV auf der GPU. Das ist ein
ANDERER Pfad (statisches MoE-Placement, kein K4N0N3-Streaming) und
MoE-spezifisch — es rettet den dichten 9B nicht. Betrieblich: die 5,4 s
qualifizieren das 30B für **latenztolerante Rollen** (Batch-Synthese,
Harvester-Klassifikation, schwere Einzelfälle), NICHT für den Gate-Pfad (der
bleibt Sub-Sekunde und damit beim 3B); die 18,6 GB Residenz kollidieren mit
K4N0N3-Trainings-RAM. Also **startbarer On-Demand-/Zeitfenster-Dienst, nicht
resident neben Training** — ein Satz, der spätere Verwunderung über swappende
Trainingsläufe erspart.

## Allgemeine Regeln

Commits thematisch (T eigenständig, U hinter der bestehenden Schnittstelle,
V hinter Flag). Keine neuen Pflicht-Dependencies (Draft-Modell wird zur
Laufzeit geladen, nichts Neues zu installieren). Timebox: wenn U3-Overlap
nach zwei fokussierten Anläufen nicht erreichbar ist, Befund mit
Event-Timeline dokumentieren und zu V weitergehen — V funktioniert auch
ohne perfekten Overlap.
