# K4N0N3 — Arbeitsauftrag 4: Pin-Budget-Fix, Group-wise int4, Training-Offload (LoRA)

## Kontext

Stand nach Auftrag 3: Mess-Harness etabliert (jede Zahl aus JSON), bitsandbytes
als inkompatibel belegt, Custom int8 + On-GPU-Dequant hinter `quantize_transfer`
funktioniert und ist der Gewinner: warm forward 1078 ms vs. 2078 ms fp16,
0,93 vs. 0,48 tok/s, Mechanik-Korrektheit bestanden. Referenz außer Konkurrenz:
Ollama q4 resident = 74 tok/s. K4N0N3s Nische ist damit klar umrissen:
PyTorch-Ökosystem + Modelle/Workloads, die nicht ins VRAM passen — vor allem
**Training**, wo llama.cpp nichts anbietet.

Offene Befunde aus Auftrag 3, die dieser Auftrag angeht:

1. **Pin-Budget wird im falschen Moment gemessen:** MemAvailable wird einmal
   beim Master-Aufbau geprobt — da liegt das volle fp16-Modell (~6,2 GB) noch
   im RAM. Der int8-Master braucht nur ~2,9 GB; nach Freigabe der fp16-Originale
   müssten 36/36 Layer pinnbar sein statt 12–15. Vermutlich der billigste
   verbleibende Speedup: mit voll gepinntem Master fallen die synchronen
   Pageable-Copies weg, die den Async-Overlap aktuell deckeln.
2. **int4 per-Channel ist qualitativ unbrauchbar** (Divergenz ab Token 0) —
   erwartbar, 16 Stufen pro ganzem Output-Channel sind zu grob. Standard-Fix:
   Group-wise-Skalen (Gruppengröße 64–128).
3. **int4 brachte nur 12 % statt ~2×** — nach dem Pin-Fix neu messen, um zu
   sehen, ob ungepinnte Layer gedeckelt haben oder der Unpack-Overhead frisst.
4. **`training.py` wurde seit Umbau 1 bewusst nicht angefasst** — der
   `TrainingManager` nutzt noch die alten `module.to()`-Pfade (pageable,
   D2H-Copies). Für den Trainings-Auftrag Q muss die Master/Drop-Mechanik
   dorthin portiert werden.

**Umgebung unverändert:** PyTorch/ROCm (HIP), RX 7600 (8 GB VRAM), 15 GB RAM.
Aufträge **in Reihenfolge** (P baut auf Os Neumessung auf, Q auf beidem).
Regel aus Auftrag 3 gilt weiter: **keine Zahl ohne Harness-JSON.**

---

## Auftrag O — Pin-Budget-Fix + Neumessung

### O1: Zwei-Pass-Aufbau im `quantize_transfer`-Pfad

Reihenfolge umbauen: **Pass 1** quantisiert alle Layer und gibt die
fp16-Originale sofort layerweise frei (`p.data`-Referenz lösen, damit der
Storage freigegeben wird; nach Pass 1 einmal `gc.collect()`). **Pass 2** pinnt
die int8-Master mit frisch geprobtem Budget. So wird das Budget gemessen,
wenn der RAM-Zustand dem Endzustand entspricht.

### O2: Per-Layer-Reprobe im fp16-Pfad

Im nicht-quantisierten Pfad gibt es keinen Freigabe-Pass (Pinnen ersetzt das
Original direkt), aber auch dort sinkt MemAvailable nicht monoton wie die
Einmal-Probe annimmt: `available_ram_mb()` vor **jeder** Layer-Pin-Entscheidung
neu lesen (ein /proc-Read pro Layer, vernachlässigbar). Die statische
Budget-Variable entfällt; `pin_ram_fraction` wird pro Probe angewendet.
`pin_ram_fraction=0.0` muss weiterhin den ungepinnten Pfad erzwingen
(Regressionstest aus H bleibt gültig).

### O3: Neumessung mit dem Harness

- 3B int8-custom: Erwartung **36/36 gepinnt**, warm_forward < 1078 ms.
- 3B int4-per-channel (M5-Stand, nur für die Mechanik-Messung — Qualität ist
  bekannt kaputt): neu messen, um Befund 3 zu klären.
- Wenn int4 nach dem Fix immer noch nicht deutlich unter int8 liegt:
  Copy vs. Unpack/Dequant per `synchronize()`-Timing trennen und den
  Zeitfresser benennen (das entscheidet, ob P überhaupt lohnt bzw. ob der
  Unpack in P anders gebaut werden muss).
- fp16 partial-pin einmal mitmessen (profitiert von O2, neue Referenz).

### Akzeptanz O

- Harness-JSONs für alle drei Konfigurationen; pinned_layers und
  warm_forward_ms beantworten die Erwartungen mit Messwerten (auch wenn
  die Antwort "Erwartung verfehlt" ist — dann mit dem Timing-Split aus O3).
- Leak-Test und Mechanik-Greedy-Check (Auftrag-3-Kriterien) weiterhin grün.
- Kein Verhalten geändert bei `quantize_transfer=False` + `pin_ram_fraction`
  Default außer dem Reprobe-Timing (Tests grün).

---

## Auftrag P — Group-wise int4 (ersetzt den M5-Stand)

### P1: Quantisierungsschema

- Symmetrisch, **group-wise entlang der Input-Dimension**: Gruppengröße
  `group_size=128` (Default, als Parameter), pro Gruppe ein fp16-Scale →
  Scale-Shape `[out_features, in_features/group_size]`.
- `in_features` nicht durch group_size teilbar: letzte Gruppe kürzer (Padding
  vermeiden, lieber Restgruppe behandeln — im Code kommentieren).
- Packing wie M5 (zwei Nibbles pro Byte), aber Unpack + Dequant in einem
  Schritt: `w_fp16 = unpack(q).to(fp16) * scale_expandiert`. Auf Kernel-Fusion
  nicht versteifen — erst korrekt, dann per O3-Timing-Methodik prüfen, ob der
  Unpack überhaupt messbar ist.
- Master-Struktur: `{"q4": packed_uint8(pinned), "scale": fp16(pinned),
  "meta": {group_size, orig_shape}}`.

### P2: Qualität beziffern

Gleiche Methodik wie M3, gegen dieselbe fp16-Referenz (greedy_tokens im JSON):

- greedy-Divergenzpunkt in 32 Tokens + mittlere |Logit-Differenz| Token 1.
- Erwartung bei g=128: keine oder späte Divergenz. Wenn unbrauchbar: einmal
  g=64 probieren. Wenn auch das divergiert ab Token 0: stoppen und
  dokumentieren — dann wäre als nächstes asymmetrische Quantisierung
  (Zero-Point pro Gruppe) der Kandidat, aber NICHT mehr in diesem Auftrag.

### P3: Messen

Harness, gegen die frischen O3-Referenzen: master_ram_mb (Erwartung ~1,6 GB
+ Scales), pinned_layers (36/36), warm_forward_ms, tokens_per_s.
Erwartungswert grob: Transfer nochmal ~×0,55 gegenüber int8 (Scales kommen
dazu) — wenn warm_forward nicht entsprechend reagiert, Timing-Split wie O3.

### Akzeptanz P

- Qualität: greedy-Divergenz erst nach Token 16 oder gar nicht (g=128 oder
  g=64), Logit-Differenz im JSON. Sonst: dokumentierter Stopp.
- Mechanik-Greedy-Check (mit vs. ohne Offloading, identisch) bestanden.
- M5-per-Channel-Code entfernt oder klar als deprecated markiert — es darf
  nicht zwei int4-Pfade geben.

---

## Auftrag Q — Training-Offload: LoRA auf einem Modell, das nicht ins VRAM passt

**Das ist der Auftrag, bei dem K4N0N3 etwas kann, was der kurze Weg
(llama.cpp/Ollama) prinzipiell nicht kann.** Ziel bewusst schmal: EIN
funktionierender LoRA-Finetune-Lauf mit sinkender Loss auf der 8-GB-Karte,
mit einem Basismodell, dessen fp16-Gewichte das VRAM sprengen. Kein
Trainings-Framework-Ausbau, keine Hyperparameter-Optimierung.

### Q1: `TrainingManager` auf Master/Drop portieren

`training.py` teilt viel Logik mit `hooks.py`, nutzt aber noch `module.to()`.
Refactoring:

- Gemeinsame Basis extrahieren (Master-Aufbau, Pinning inkl. O-Fixes,
  Upload/Drop, Prefetch-Stream, Budget-Anbindung) — z. B. `_OffloadCore`,
  von der `LayerManager` und `TrainingManager` erben oder die sie komponieren.
  Duplizierten Code in `training.py` (Discovery etc.) dabei gleich auflösen.
- **Wichtige Einschränkung:** Der Drop-Offload verwirft GPU-Tensoren ersatzlos —
  das ist nur korrekt für Gewichte, die sich nicht ändern. Für Q gilt daher:
  **Basisgewichte sind frozen** (`requires_grad=False`), trainiert werden
  ausschließlich LoRA-Adapter. Der `TrainingManager` muss das erzwingen:
  beim `prepare()` prüfen, dass kein getrackter Basis-Parameter
  `requires_grad=True` hat, sonst harte Fehlermeldung mit Erklärung.
  (Voll-Finetuning mit Offload = anderes Projekt: bräuchte D2H-Writeback +
  Optimizer-State-Offload. Im Code-Kommentar festhalten.)
- Backward-Hooks anpassen: `bw_pre` holt den Layer via Master-Upload zurück
  (statt `.to()`), `bw_post` droppt. Prefetch-Richtung im Backward ist
  **rückwärts** (Layer n-1 als nächstes) — im `bw_post` entsprechend
  rückwärts prefetchen.
- `quantize_transfer` soll auch hier funktionieren (frozen Basis + int8/int4-
  Transfer ist genau das QLoRA-Muster). Wenn das ohne Zusatzaufwand geht:
  aktivieren; wenn es hakt: fp16-Master reicht für den Q-Beweis, Befund notieren.

### Q2: LoRA-Setup

- Adapter via `peft` (nur als Spike-/Dev-Dependency, nicht Pflicht) ODER
  minimal von Hand (zwei kleine fp16-Matrizen A/B pro Ziel-Linear, r=8,
  auf `q_proj`/`v_proj`) — Entscheidung nach ROCm-Verträglichkeit von peft,
  Handvariante ist ausdrücklich okay und ~100 Zeilen.
- Adapter liegen **permanent auf der GPU** (wenige MB) und sind von der
  Offload-Mechanik ausgenommen — sicherstellen, dass Discovery/Hooks sie
  nicht mit offloaden.
- Achtung Hook-Interaktion: Wenn HF `gradient_checkpointing` nötig wird,
  feuern die Forward-Pre-Hooks während der Backward-Rekomputation erneut,
  und zwar in Rückwärts-Reihenfolge der Layer — prüfen, ob die
  Prefetch-Logik damit klarkommt (idempotente States sollten das abfedern).
  Erst OHNE Checkpointing versuchen (Batch 1, seq_len 256); nur wenn
  Aktivierungen das VRAM sprengen, Checkpointing dazunehmen und die
  Interaktion explizit testen.

### Q3: Der Beweis-Lauf

- Modell: Qwen2.5-3B (fp16-Basis ~6,2 GB > 8 GB VRAM abzüglich Aktivierungen
  — Bedingung "passt nicht" gegeben; falls es wider Erwarten doch knapp
  passt, VRAM-Budget künstlich auf 3 GB setzen, damit der Beweis sauber ist).
- Daten: winziger fester Textdatensatz (z. B. 200 Beispiele, gern aus einem
  seiner Projekte-READMEs generiert — Inhalt egal, Reproduzierbarkeit zählt:
  fester Seed, Datei ins Repo unter `bench/data/`).
- 50 Optimizer-Schritte, AdamW nur auf Adapter-Parametern, Batch 1,
  Loss pro Schritt ins Harness-JSON (Harness um Trainings-Metriken erweitern:
  loss_curve, step_time_ms Median, vram_peak_mb).
- Erfolgskriterien: (1) 50 Schritte ohne OOM/Crash, (2) Loss fällt erkennbar
  (Median letzte 10 < Median erste 10), (3) VRAM-Peak ≤ Budget + Reserve,
  (4) Funktionsprobe: generate() mit Adaptern liefert erkennbar anderen
  Output als ohne (Adapter an/aus vergleichen, greedy_tokens beider Läufe
  ins JSON).
- step_time_ms ehrlich einordnen: Erwartung ist LANGSAM (jeder Schritt
  streamt alle Layer zweimal, Forward + Backward). Die Zahl ist der
  Datenpunkt, nicht das Problem — es geht um "geht überhaupt", nicht
  "geht schnell".

### Akzeptanz Q

- Alle vier Erfolgskriterien aus Q3 mit JSON-Beleg, ODER dokumentierter
  Stopp mit präziser Ursache (z. B. Checkpointing × Hooks inkompatibel —
  dann mit minimalem Repro-Skript unter `spike/`).
- `TrainingManager`-Refactoring: bestehende Tests grün, neue Tests für den
  requires_grad-Guard und den Backward-Prefetch (CPU-only lauffähig via
  Mock/kleines Modell).
- Timebox-Hinweis: Q1+Q2 sind der eigentliche Aufwand; wenn Q3 nach
  funktionierendem Q1/Q2 an einem einzelnen ROCm-Detail scheitert, Befund
  dokumentieren statt endlos debuggen.

---

## Auftrag R — Bericht

Via `bench/make_report.py` aus den JSONs:

1. Inference-Tabelle neu: fp16 partial-pin (O) / int8 (O) / int4-groupwise (P),
   plus die alte Ollama-Referenzzeile. Vorher/Nachher-Spalte für den
   Pin-Fix-Effekt (Auftrag-3-JSONs vs. neue).
2. Qualitätstabelle int8 vs. int4-g128 (Divergenzpunkt, Logit-Differenz).
3. Trainings-Abschnitt: Loss-Kurve (als Zahlenreihe oder simple
   ASCII/Matplotlib-Grafik unter bench/results/), step_time, VRAM-Peak,
   Funktionsprobe-Ergebnis.
4. Abschluss-Einordnung: was K4N0N3 jetzt nachweislich kann, was es
   nachweislich nicht sein will (Inference-Ersatz für llama.cpp), und die
   2–3 sinnvollsten nächsten Schritte NUR falls das Projekt weitergeht —
   ausdrücklich inklusive der Option "hier ist ein guter Abschluss".

## Allgemeine Regeln

- Reihenfolge O → P → Q, Messungen jeweils gegen die frischesten Referenzen.
- Commits thematisch getrennt; Q1-Refactoring als eigener Commit VOR der
  LoRA-Logik (revertierbar).
- Neue Dependencies (peft) nur optional/dev, Handvariante bevorzugen, wenn
  peft auf ROCm zickt.
- CPU-only-Smoke-Test nach O und Q1.
- Bei Unklarheiten oder verfehlten Akzeptanzkriterien: stoppen, Befund
  dokumentieren, nicht drumherum patchen. Keine Zahl ohne JSON.
