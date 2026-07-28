# K4N0N3 — Arbeitsauftrag 3: Messung reparieren, Offload-Druck-Test, Custom-Dequant-Spike

## Kontext

Stand nach Umbau 2: Fallback-Zero-Copy (G) und Partial Pinning (H) funktionieren
und sind validiert. Der int8-Spike (I) hat dagegen **nicht gemessen, was er
behauptet**: bitsandbytes hält das quantisierte Modell permanent GPU-resident
(VRAM-Peak 3394 MB ≈ Modellgröße), es fand also **kein Offloading statt** —
die schnellen Pre-Hooks und der 2.4s-Forward sind trivial, wenn nichts
transferiert wird. Zusätzlich hat die Vergleichstabelle Cold- und Warm-Zahlen
vermischt (104s ist der Cold-Wert von fp16, nicht Warm) und tok/s erneut
geschätzt statt gemessen.

Dieser Auftrag hat drei Ziele:

1. **Messinfrastruktur**, die solche Fehler strukturell verhindert (K)
2. **Klären**, ob bitsandbytes unter echtem Offload-Druck überhaupt mit der
   K4N0N3-Mechanik kompatibel ist — Erwartung: eher nein, und auch das ist
   ein sauberes Ergebnis (L)
3. **Custom Weight-only-Quantisierung** als Spike: int8-Master + On-GPU-Dequant,
   ohne fremde Kernel-Abhängigkeiten, passend zur Master/Drop-Architektur (M)

**Umgebung unverändert:** PyTorch/ROCm (HIP), RX 7600 (8 GB VRAM), 15 GB RAM,
CPU-only-Pfad degradiert sauber, keine API-Brüche.

**Wichtigste Regel dieses Auftrags:** Keine Zahl in einen Bericht, die nicht
aus dem Mess-Harness (K) stammt. Wenn eine Messung nicht durchläuft (Timeout,
OOM, Crash), steht im Bericht "nicht messbar, weil X" — niemals ein
geschätzter Ersatzwert an ihrer Stelle.

---

## Auftrag K — Mess-Harness (`bench/harness.py`, wird Teil des Repos)

### K1: Ein Skript für alle Konfigurationen

Ein einziges Skript, das eine Konfiguration (Modell, dtype/Quant, Budget,
prefetch_depth, pin_ram_fraction) als Argumente nimmt und **immer dieselbe
Messreihe** in derselben Reihenfolge fährt. Output: eine JSON-Datei pro Lauf
(`bench/results/<timestamp>_<config>.json`) plus lesbare Konsolen-Zusammenfassung.
Berichte werden aus den JSONs generiert, nie von Hand getippt.

### K2: Definitionen (im Code als Docstring, damit unstrittig)

- **Cold Forward:** erster Forward nach frischem `prepare()` in einem frischen
  Prozess. Enthält CUDA-Init und Allocator-Warmup — deshalb separat ausweisen
  und NIE mit Warm vergleichen.
- **Warm Forward:** Median aus 5 Forwards nach 2 Discard-Warmups, jeweils mit
  `torch.cuda.synchronize()` vor Start und nach Ende der Zeitnahme.
- **tok/s:** ausschließlich aus echtem `model.generate(prompt,
  max_new_tokens=32, do_sample=False)`: Wallclock der generate()-Zeit geteilt
  durch tatsächlich neu generierte Tokens (aus der Output-Länge, nicht aus
  max_new_tokens). Timeout-Budget 10 Minuten pro generate-Lauf; bei Timeout
  wird `tokens_per_s: null` mit `timeout: true` geloggt.
- **VRAM-Peak:** `torch.cuda.max_memory_allocated()`, resettet via
  `reset_peak_memory_stats()` direkt vor der Messphase (nicht vor dem Laden —
  Ladephase separat erfassen).
- **Offload-Wirksamkeit:** `memory_allocated()` unmittelbar vor und nach einem
  erzwungenen `_offload(layer)` — Differenz muss ≈ Layer-GPU-Größe sein.
  Das ist DIE Kennzahl, die beim bitsandbytes-Lauf gefehlt hat.

### K3: Pflichtfelder pro JSON

config (alle Parameter), git-Commit-Hash, Zeitstempel, torch/ROCm-Version,
cold_forward_ms, warm_forward_ms (Median + alle 5 Einzelwerte),
generate_tokens, generate_wallclock_s, tokens_per_s, vram_peak_mb,
offload_frees_mb (aus K2 letzter Punkt), pinned_layers, master_ram_mb,
greedy_tokens (die ersten 32 Token-IDs des Greedy-Outputs, für
Korrektheitsvergleiche zwischen Läufen).

### K4: Alte Zahlen neu erheben

Nach Fertigstellung des Harness die zwei fp16-Konfigurationen (pageable via
`pin_ram_fraction=0.0`, partial-pinned Default) einmal sauber durchmessen.
Diese JSONs sind ab jetzt die Referenz — die Tabellen aus Umbau-Bericht 2
gelten als ungültig.

### Akzeptanz K

- Zwei aufeinanderfolgende Läufe derselben Konfiguration weichen bei
  warm_forward_ms um < 10 % voneinander ab (sonst Messaufbau prüfen).
- fp16-partial-pinned zeigt messbar besseres warm_forward als pageable
  (Richtung und Größenordnung plausibel zum Pin-Anteil; Werte im JSON).
- tok/s-Feld ist bei mindestens einer Konfiguration ein echter Messwert
  (kein Timeout) ODER der Timeout ist dokumentiert — keine Schätzwerte im JSON.

---

## Auftrag L — bitsandbytes unter echtem Offload-Druck (Spike, timeboxed)

**Ziel:** Die offene Frage aus Umbau 2 beantworten: funktioniert die
Master/Drop-Mechanik mit bitsandbytes-Int8Params, wenn Eviction wirklich
stattfindet? **Erwartung: vermutlich nein** (Int8Params quantisieren beim
GPU-Transfer und tragen Zustand wie SCB-Scales außerhalb von `.data`). Ein
sauber dokumentiertes "inkompatibel, weil X" ist ein voller Erfolg dieses
Auftrags. **Timebox: nicht mehr als einen fokussierten Arbeitsblock —
keine Rettungsversuche, keine Monkey-Patches in bitsandbytes.**

### L1: Druck erzwingen

int8-3B-Modell laden wie im Spike I, aber VRAM-Budget des MemoryManager
künstlich auf 2 × Layer-GPU-Größe setzen, `prefetch_depth=1` → Eviction ist
bei jedem Layer-Schritt garantiert. Vorher verifizieren (Log), dass die Layer
überhaupt als CPU-Master vorliegen und nicht schon beim Laden GPU-resident
sind — falls bitsandbytes das Laden auf CPU verweigert bzw. beim `.to("cpu")`
dequantisiert, ist der Spike an dieser Stelle beendet: Befund dokumentieren.

### L2: Drei Messpunkte mit dem Harness

1. `offload_frees_mb`: gibt der Drop wirklich VRAM frei? (Wenn SCB/Scales
   GPU-resident bleiben oder Int8Params interne Referenzen halten: nein →
   dokumentieren, welche Tensoren übrig bleiben, via
   `torch.cuda.memory_snapshot()` oder Zählung der GPU-Tensoren im Modul.)
2. Re-Upload nach Eviction: rechnet der Layer danach korrekt weiter, oder
   korrumpiert der `.data`-Swap den Int8Params-Zustand?
3. `greedy_tokens` vs. Referenzlauf desselben int8-Modells voll auf GPU
   (ohne Hooks): identische Sequenz erwartet. Abweichung = Mechanik
   korrumpiert Zustand → dokumentieren, Spike beendet.

### Akzeptanz L

Bericht-Abschnitt mit klarem Verdikt: kompatibel / inkompatibel, mit den drei
Messpunkten als Beleg. Bei "inkompatibel" ein Satz zur Ursache (welcher
Zustand liegt außerhalb der Master/Drop-Mechanik). Kein Produktcode geändert.

---

## Auftrag M — Custom Weight-only int8 + On-GPU-Dequant (Spike → Kandidat für Produktcode)

**Kernidee:** Quantisierung selbst machen, damit die Weights schlichte
Integer-Tensoren sind, die sich exakt wie bisher pinnen, kopieren und droppen
lassen. Kein fremder Kernel, kein Paket, das gegen die Architektur arbeitet.

**Wichtig fürs Erwartungsmanagement (auch im Bericht so ausweisen):**
Dieser Ansatz spart **PCIe-Transfer** (halbiert bei int8), NICHT GPU-VRAM —
nach dem Dequant liegt der Layer als fp16 im VRAM, GPU-Größe unverändert.
Für K4N0N3 ist das der richtige Trade: VRAM hält eh nur 2–3 Layer, der
Engpass ist der Bus. `MemoryManager` bucht daher weiterhin die
**fp16-GPU-Größe**, nicht die int8-Transfergröße.

### M1: Quantisierung beim Master-Aufbau

- Nur `torch.nn.Linear`-Weights innerhalb der getrackten Layer quantisieren
  (Bias, Norms, Embeddings, alles außerhalb der Layer: bleibt fp16).
- Symmetrisch, per-Output-Channel: `scale[c] = max(|W[c,:]|) / 127`,
  `W_int8[c,:] = round(W[c,:] / scale[c])`, dtype int8, scale als fp16-Vektor.
- Master-Struktur pro quantisiertem Param: `{"q": int8_tensor(pinned),
  "scale": fp16_tensor(pinned)}`; nicht-quantisierte Params wie bisher als
  direkte Master. Kennzeichnung im Master-Dict, welcher Typ vorliegt.
- Aktivierung über neuen Keyword-Parameter `quantize_transfer: bool = False`
  an `LayerManager`/`ZeroFlushModel` (Default aus → keinerlei Verhaltensänderung
  für Bestandsnutzung).
- RAM-Erwartung 3B: ~2.9 GB int8-Master + Scales + fp16-Reste → sollte
  **vollständig pinnbar** sein. Loggen und verifizieren (das ist Kernthese 1).

### M2: Upload-Pfad mit Dequant

Im Prefetch-/Ensure-Pfad für quantisierte Params:

1. int8-Tensor + scale async auf GPU kopieren (Prefetch-Stream, wie gehabt)
2. Im selben Stream dequantisieren: `w_fp16 = q.to(torch.float16) * scale.unsqueeze(1)`
3. `p.data = w_fp16`; das int8-GPU-Staging und der scale-GPU-Tensor verlieren
   danach ihre Referenz (Allocator gibt frei). Event nach dem Dequant recorden,
   nicht nach dem Copy — der Konsument braucht das fertige fp16.
4. `record_stream`-Behandlung wie beim bestehenden Pfad (gleicher Kommentar,
   gleiche Stelle).
- Drop-Offload unverändert: `p.data = master["q"]`? **Nein** — Achtung:
  `p.data` muss nach dem Drop wieder ein Tensor sein, mit dem der Layer auf
  CPU theoretisch rechnen könnte bzw. der beim nächsten Upload als Quelle
  dient. Sauberste Lösung: `p.data` zeigt nach Drop auf einen leeren
  fp16-Meta-/Dummy-Tensor ODER auf den int8-Master, und der Upload-Pfad liest
  grundsätzlich aus der Master-Struktur, nie aus `p.data`. Entscheidung
  treffen, im Code begründen, und sicherstellen, dass der CPU-only-Pfad
  (`quantize_transfer=True` ohne CUDA) entweder sauber dequantisiert auf CPU
  rechnet oder mit klarer Fehlermeldung ablehnt — kein stiller Müll-Output.

### M3: Korrektheit — Quantisierungsfehler beziffern, nicht wegdiskutieren

- **Mechanik-Korrektheit:** derselbe quantisierte Zustand MIT Offloading vs.
  OHNE (alle Layer dauerhaft auf GPU, Hooks entfernt, gleiche int8-Master als
  Quelle): `greedy_tokens` müssen **identisch** sein. Das isoliert
  Mechanik-Fehler von Quantisierungsfehlern.
- **Quantisierungsqualität:** greedy_tokens int8 vs. fp16-Referenz — Divergenz
  ab Token N notieren; zusätzlich mittlere |Logit-Differenz| am ersten Token.
  Erwartung: kleine Abweichung, per-Channel int8 ist etabliert harmlos.
  Zahlen in den Bericht, Bewertung dem Menschen überlassen.

### M4: Messen (Harness, volle Reihe)

- Konfigurationen: 3B fp16 partial-pin (Referenz aus K4) vs. 3B int8-custom.
- Kernthesen, die die Messung bestätigen oder widerlegen soll:
  (1) 36/36 Layer gepinnt bei int8, (2) warm_forward deutlich unter dem
  fp16-partial-pin-Wert (Transferanteil halbiert + alles async),
  (3) tok/s als echter generate()-Messwert.
- Wenn (2) nicht eintritt: mit `torch.cuda.synchronize()`-Timing um
  Copy vs. Dequant herausmessen, wo die Zeit hingeht (Dequant-Kernel auf
  ROCm könnte langsamer sein als erhofft — dann ist das der Befund).

### M5: Stretch (nur wenn M1–M4 grün und Zeit übrig): int4-gepackt

Zwei Nibbles pro Byte packen (`(q[:, 0::2] & 0xF) | (q[:, 1::2] << 4)`),
Unpack im Dequant-Schritt. Halbiert den Transfer nochmal. Nur angehen, wenn
int8 sauber läuft — und als separater Commit, damit es einzeln revertierbar ist.

### Akzeptanz M

- Mechanik-Korrektheit (M3, Punkt 1) bestanden — das ist hartes Kriterium,
  bei Abweichung stoppen und Ursache dokumentieren.
- Harness-JSONs für beide Konfigurationen vorhanden, Kernthesen (1)–(3)
  jeweils mit Messwert beantwortet (auch wenn die Antwort "nein" ist).
- Code hinter `quantize_transfer=False` per Default inaktiv; bestehende
  Tests grün.

---

## Auftrag N — Bericht

Aus den Harness-JSONs generiert (Skript `bench/make_report.py`, liest alle
JSONs, baut die Tabelle):

1. Tabelle: fp16 pageable / fp16 partial-pin / int8-custom (+ int4 falls M5):
   master_ram_mb, pinned_layers, warm_forward_ms, tokens_per_s, vram_peak_mb,
   offload_frees_mb. Cold Forward separat darunter, ausdrücklich als
   nicht-vergleichsrelevant markiert.
2. **Referenzzeile außer Konkurrenz:** dasselbe Modell als GGUF q4 über
   Ollama/llama.cpp-ROCm auf derselben Maschine, tok/s per Stoppuhr-Skript
   (`ollama run` mit festem Prompt, Tokens/Zeit aus der Ollama-Ausgabe).
   Kein K4N0N3-Vergleich im engen Sinn, sondern die ehrliche Antwort auf
   "was wäre der kurze Weg gewesen" — gehört zur Einordnung des Projekts.
3. L-Verdikt (bitsandbytes kompatibel ja/nein + Ursache).
4. Empfehlung: int8-custom in Produktcode überführen ja/nein, und ob int4
   der nächste Schritt ist — begründet ausschließlich mit Zahlen aus den JSONs.

## Allgemeine Regeln

- Commits: K als eigener Commit (Harness ist bleibende Infrastruktur),
  L nur unter `spike/`, M hinter Feature-Flag, N generiert.
- Keine neuen Pflicht-Dependencies. bitsandbytes bleibt Spike-only.
- CPU-only-Smoke-Test nach K und M.
- Bei Unklarheiten oder verfehlten Akzeptanzkriterien: stoppen, Befund
  dokumentieren, nicht drumherum patchen. Insbesondere: keine Zahl ohne
  JSON dahinter.
