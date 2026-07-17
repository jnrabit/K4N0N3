# K4N0N3 — Arbeitsauftrag 2: RAM-Fallback, Partial Pinning, Quantisierungs-Spike

## Kontext

Stand nach Umbau 1 (Pinned-Master + Drop-Offload, Budget-Tracking, Wrap-Around):
funktioniert und ist validiert für Modelle, deren Pinned-Master komplett in den
RAM passt (0.5B: 683 MB gepinnt, 0.3 ms/Layer Upload async). Beim Zielszenario
**Qwen2.5-3B auf RX 7600 / 15 GB RAM** greift jedoch der Pinning-Fallback und
das System fällt auf synchrone pageable Copies zurück (~3 ms/Layer, Prefetch
überlappt nicht, Warm Forward 2.3 s ≈ 0.43 tok/s).

Befunde aus dem 3B-Lauf, die dieser Auftrag behebt:

1. **Fallback verschwendet RAM:** "11.5 GB nötig" bei einem 6.2-GB-Modell heißt,
   der Fallback legt eine *zweite* pageable Kopie neben den Original-Tensoren an.
   Das ist sinnlos — ungepinnt sind die Originale selbst der Master.
2. **Pinning ist alles-oder-nichts:** Wenn nicht alle Layer gepinnt werden können,
   wird gar nichts gepinnt. Partial Pinning (so viel wie der RAM hergibt) würde
   den Transferpfad anteilig beschleunigen.
3. **fp16-Transfervolumen ist die harte Decke:** 6.2 GB PCIe-Traffic pro Token
   deckelt 3B fp16 bei ~1.5–2 tok/s selbst mit perfektem Pinning. 4-bit-Weights
   (~1.9 GB) passen komplett gepinnt in den RAM UND vierteln den Transfer.
   → Quantisierung ist der Hebel, der beide Probleme gleichzeitig löst.

**Umgebung unverändert:** PyTorch/ROCm (HIP), eine GPU (RX 7600, 8 GB), 15 GB RAM,
CPU-only-Pfad muss weiter sauber degradieren. Keine API-Brüche nach außen.

Aufträge **in Reihenfolge** abarbeiten, nach jedem Auftrag Akzeptanzkriterien
prüfen, erst dann weiter.

---

## Auftrag G — Fallback-Fix: Originale als Master, keine Duplikat-Kopie (`hooks.py`)

### G1: Master-Aufbau umstellen

`_build_master_copies` (bzw. wie die Methode jetzt heißt) so ändern:

- **Pfad "gepinnt":** wie bisher — gepinnte Kopie anlegen, `p.data` auf die
  gepinnte Kopie setzen, Original wird freigegeben.
- **Pfad "ungepinnt" (Fallback):** KEINE Kopie anlegen. Stattdessen den
  bestehenden CPU-Tensor direkt als Master registrieren:
  `self._cpu_master[name][pname] = p.data` (bzw. `p.detach()` mit identischem
  Storage — es darf kein Clone entstehen). RAM-Mehrverbrauch des Fallbacks: 0.
- Das Flag muss **pro Layer** geführt werden, nicht global
  (`self._pinned: dict[str, bool]` statt `self._pinned: bool`) — das braucht
  Auftrag H sowieso. Bestehende Abfragen von `self._pinned` entsprechend anpassen.

### G2: Konsistenz Upload/Offload prüfen

- Upload- und Drop-Pfad funktionieren identisch, egal ob der Master gepinnt ist
  oder nicht (`pinned.to("cuda", non_blocking=True)` ist auf ungepinnten Tensoren
  einfach faktisch synchron — das ist okay, kein Sonderpfad nötig).
- Sicherstellen, dass beim Drop (`p.data = master`) im ungepinnten Fall wirklich
  derselbe Storage referenziert wird, der beim Aufbau registriert wurde — kein
  neuer Tensor, kein Copy.

### Akzeptanz G

- 3B-Modell laden mit erzwungenem Fallback (Pin-Budget künstlich auf 0 setzen,
  siehe H): RSS des Prozesses nach Master-Aufbau ≤ RSS direkt nach
  `from_pretrained` + max. 200 MB Toleranz (messen via
  `/proc/self/status` VmRSS vorher/nachher, im Test loggen).
- Logits-Vergleich gegen Referenz weiterhin identisch (fp16-Toleranz).
- Leak-Test (5× Offload→Prepare→Forward, `memory_allocated` konstant) grün.

---

## Auftrag H — Partial Pinning mit RAM-Probe (`hooks.py` + `utils.py`)

### H1: Pin-Budget bestimmen

Neue Funktion in `utils.py`:

```python
def available_ram_mb() -> float:
    """MemAvailable aus /proc/meminfo in MB. Fallback: psutil-frei, nur Linux."""
```

- Liest `MemAvailable` aus `/proc/meminfo`. Wenn nicht lesbar (kein Linux):
  konservativ 0 zurückgeben → dann wird nichts gepinnt (Fallback G greift).
- Pin-Budget = `available_ram_mb() * pin_ram_fraction`, Default
  `pin_ram_fraction=0.7`. Als neuer optionaler Parameter durch
  `LayerManager.__init__` und `ZeroFlushModel.__init__` durchreichen
  (Keyword-only, Default 0.7 → keine API-Brüche). `pin_ram_fraction=0.0`
  erzwingt den ungepinnten Pfad (wird im Akzeptanztest G gebraucht).

### H2: Layerweise pinnen bis Budget erschöpft

- Beim Master-Aufbau Layer in **Forward-Reihenfolge** durchgehen. Pro Layer:
  wenn Layer-Größe ≤ Pin-Restbudget → pinnen (Pfad "gepinnt"), Budget abziehen;
  sonst → Fallback-Pfad (Original als Master), NICHT abbrechen, weitere Layer
  prüfen (kleinere Layer wie finale Norm können noch passen).
- Zusätzlich jeden einzelnen `pin_memory()`-Call in try/except RuntimeError
  wrappen: schlägt er trotz Budget fehl, diesen Layer auf Fallback setzen und
  weitermachen (MemAvailable ist eine Schätzung, keine Garantie).
- `verbose=True` loggt eine Zusammenfassung:
  `[K4N0N3] Pinned 21/36 layers (3087/6174 MB), pin budget 3200 MB`.

### H3: Warum Forward-Reihenfolge — dokumentieren

Kurzer Code-Kommentar: Bei Wrap-Around-Prefetch profitiert jeder Layer gleich
oft, die Reihenfolge ist daher sekundär; Forward-Reihenfolge ist schlicht
deterministisch und debugbar. KEINE cleveren Heuristiken (z. B. "größte zuerst")
einbauen.

### Akzeptanz H

- 3B-Lauf auf der Zielmaschine: Log zeigt partielles Pinning (> 0 und < 36 Layer
  gepinnt). Kein OOM-Kill, kein `pin_memory`-Crash.
- Warm Forward messbar schneller als die 2.315 ms des reinen Pageable-Laufs
  (Erwartung: grob proportional zum gepinnten Anteil; Messwert im Bericht
  festhalten, keine Schönrechnung — wenn's nicht schneller ist, als Befund
  dokumentieren und stoppen).
- `pin_ram_fraction=0.0` → Verhalten identisch zu Auftrag G (Regressionstest).
- 0.5B-Modell: weiterhin 100% gepinnt, Zahlen aus Umbau 1 reproduzierbar
  (0.3 ms/Layer-Größenordnung).

---

## Auftrag I — Quantisierungs-Spike: 4-bit-Modell durch die Hook-Mechanik (`spike/`, kein Produktionscode)

**Ziel des Spikes:** Beweisen (oder widerlegen), dass die K4N0N3-Mechanik mit
quantisierten Modellen funktioniert, und tok/s messen. Explizit als Spike:
Erkenntnisse zählen, nicht Code-Schönheit. Ergebnis ist ein Bericht + ggf. eine
kurze Liste nötiger Produktänderungen — KEIN großer Umbau in diesem Auftrag.

### I1: Modellwahl

- Erste Wahl: ein fertiges GPTQ-Quantisat von Qwen2.5-3B-Instruct von HF
  (z. B. offizielles `Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4`), geladen via
  `transformers` + `gptqmodel`/`auto-gptq` — **vorher prüfen, ob das Paket auf
  ROCm installierbar ist**. Wenn nein:
- Zweite Wahl: torchao int4 weight-only (`torchao.quantization`), auf das
  fp16-Modell angewendet. Auch hier: ROCm-Kompatibilität zuerst prüfen
  (torchao int4-Kernels sind teils CUDA-only — wenn der Kernel auf HIP nicht
  läuft, ist genau DAS der dokumentierte Befund).
- Dritte Wahl (wenn beides scheitert): int8 via torchao als Datenpunkt —
  halbiert Transfer immerhin, und der Mechanik-Beweis (Buffers!) gilt genauso.
- Egal welcher Weg: im Bericht festhalten, welcher gewählt wurde und warum.

### I2: Mechanik-Prüfung — Buffers sind der Knackpunkt

Quantisierte Layer tragen ihre Daten teils in **Buffers** statt Parametern
(GPTQ: `qweight`, `qzeros`, `scales`, `g_idx`). Prüfen:

- Deckt `_build_master_copies` via `named_buffers()` diese wirklich ab?
  (War in Umbau 1 / A1 gefordert — hier zeigt sich, ob es stimmt.)
- Deckt `_measure_layer_sizes` und `_module_param_bytes` in `memory.py` Buffers
  mit ab? Aktuell zählen beide nur `parameters()` → Layer-Größen und
  Budget-Buchhaltung wären für GPTQ-Layer massiv falsch (qweight ist ein
  Buffer!). Falls ja: das ist eine der "nötigen Produktänderungen" für den
  Bericht — im Spike als minimaler Patch fixen (Buffers mitzählen), damit
  gemessen werden kann.
- Drop-Offload: funktioniert `b.data = master` auch für Buffers mit
  Integer-Dtypes? (Sollte, aber verifizieren — `is_pinned`-Stichprobe wie gehabt.)

### I3: Messen

Gleiche Messreihe wie beim 3B-fp16-Lauf, damit direkt vergleichbar:

- Master-Größe gesamt (Erwartung: ~1.9–2.2 GB für int4 → sollte **komplett
  gepinnt** in den RAM passen — verifizieren, das ist die Kernthese)
- Anteil gepinnter Layer (Erwartung: 36/36)
- Cold Forward, Warm Forward, Upload/Layer, D2H-Offload
- **tok/s bei echtem `generate()`** mit `max_tokens=64`, `do_sample=False` —
  nicht schätzen, messen. Timeout-Schutz: wenn ein einzelner Forward > 30 s
  dauert, abbrechen und Teilmessung dokumentieren.
- Korrektheit: greedy-decode des quantisierten Modells MIT K4N0N3 vs. dasselbe
  quantisierte Modell OHNE Offloading (voll auf GPU, passt ja mit ~2 GB) —
  identische Token-Sequenz erwartet. NICHT gegen fp16 vergleichen
  (Quantisierungsabweichung ist erwartbar und hier nicht das Thema).

### Akzeptanz I

- Bericht beantwortet drei Fragen mit Messwerten: (1) Läuft die Hook-Mechanik
  mit quantisierten Layern korrekt? (2) Passt der int4-Master komplett gepinnt
  in 15 GB RAM? (3) Wie viele tok/s — und wie verhält sich das zur groben
  Zielmarke 5–8 tok/s?
- Liste "nötige Produktänderungen" (z. B. Buffers in Größenmessung, Loader-Pfad
  für GPTQ in `ZeroFlushModel`), jeweils mit 1-Satz-Aufwandsschätzung.
- Wenn keiner der drei Quantisierungswege auf ROCm läuft: sauber dokumentieren
  woran es scheitert (Fehlermeldungen, Paketversionen) — das ist dann das
  valide Ergebnis des Spikes, nicht drumherum hacken.

---

## Auftrag J — Abschlussbericht

Markdown-Bericht mit:

1. **Vergleichstabelle** über alle drei Konfigurationen (3B fp16 pageable /
   3B fp16 partial-pinned / 3B int4 full-pinned): Master-RAM, gepinnte Layer,
   Warm Forward, tok/s, VRAM-Peak.
2. Getrennte Ausweisung Messwert vs. Schätzung. tok/s IMMER aus echtem
   `generate()` ableiten, nie aus Einzel-Forward hochrechnen.
3. Kausalaussagen nur mit Beleg — "X ist schneller wegen Y" braucht eine
   Messung, die Y isoliert, sonst als Vermutung kennzeichnen.
4. Empfehlung: lohnt der Ausbau des Quantisierungspfads zu Produktionscode,
   und was wären die nächsten 2–3 konkreten Schritte.

## Allgemeine Regeln

- Kleine, thematisch getrennte Commits pro Auftrag; Spike-Code unter `spike/`
  oder in einem Branch, nicht ins Paket mischen.
- Neue Dependencies NUR für den Spike (gptqmodel/torchao), nicht als
  Pflicht-Dependency des Pakets.
- CPU-only-Smoke-Test nach G und H.
- Bei Unklarheiten oder verfehlten Akzeptanzkriterien: stoppen, Befund
  dokumentieren, nicht drumherum patchen.
