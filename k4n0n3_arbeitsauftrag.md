# K4N0N3 — Umbau auf Pinned-Memory-Offloading (Arbeitsauftrag)

## Kontext

K4N0N3 ist ein Python-Paket für transparentes Layer-Offloading von HF-Transformer-Modellen: Forward-Hooks holen Layer just-in-time auf die GPU, ein `MemoryManager` (LRU) hält ein VRAM-Budget ein, ein separater CUDA-Stream prefetcht kommende Layer. Ziel des Projekts: **größere lokale LLMs auf einer 8-GB-Karte (AMD RX 7600, ROCm) benutzbar machen.**

Wichtige Dateien:
- `hooks.py` — `LayerManager` (Kern: Discovery, Pre/Post-Hooks, Prefetch, Offload)
- `memory.py` — `MemoryManager` (Budget-Tracking, LRU-Eviction)
- `huggingface.py` — `ZeroFlushModel` (HF-Wrapper, `generate()`)
- `training.py` — `TrainingManager` (Backward-Hooks, hier NICHT anfassen außer wo explizit gesagt)
- `cli.py` — `run` / `bench` Subcommands
- `cache.py`, `utils.py`, `parallel.py`, `tensor.py`, `gguf_reader.py` — Nebenschauplätze

**Umgebung:** PyTorch mit ROCm-Backend. `torch.cuda.*` API funktioniert via HIP, `pin_memory()` wird unterstützt. Es gibt genau EINE GPU. Alle Änderungen müssen weiterhin sauber degradieren, wenn `torch.cuda.is_available()` False ist (CPU-only-Pfad darf nicht brechen).

**Bekannte Probleme, die dieser Umbau behebt:**
1. Prefetch kopiert von pageable CPU-Memory → `non_blocking=True` ist wirkungslos, Copies sind faktisch synchron, Prefetch überlappt nicht mit Compute.
2. Offload macht einen echten D2H-Copy zurück auf CPU, obwohl Weights bei Inference immutable sind → unnötiger PCIe-Traffic, und der Offload läuft synchron im Post-Hook (Hot Path).
3. `MemoryManager` erfährt von geprefetchten Layern erst im Pre-Hook → während des Prefetch-Fensters liegen real bis zu `prefetch_depth` Layer mehr im VRAM als getrackt → OOM-Risiko bei knappem Budget.
4. Kein Wrap-Around: nach dem letzten Layer wird nichts für den nächsten Token-Durchlauf geprefetcht → Layer 0 startet jeden Token kalt.
5. Kleinkram: O(n)-`list.index()` in jedem Hook, TTL-Inkonsistenz in `LRUCache.__contains__`, fehlendes `record_stream()` beim Cross-Stream-Prefetch.

Arbeite die Aufträge **in Reihenfolge** ab. Nach jedem Auftrag: kurzer Selbsttest (siehe Akzeptanzkriterien), erst dann weiter. Keine API-Brüche nach außen — `ZeroFlushModel(...)`, `LayerManager(...)`-Signaturen und die CLI bleiben kompatibel.

---

## Auftrag A — Pinned CPU-Master-Kopie + Drop-Offload (Kern-Umbau, `hooks.py`)

### A1: Master-Kopie anlegen

Im `LayerManager.__init__` (nur wenn CUDA verfügbar), nach `_measure_layer_sizes()`:

- Baue pro Layer ein Dict `self._cpu_master: dict[str, dict[str, torch.Tensor]]`, das für jeden Parameter (und Buffer!) des Layers eine **gepinnte** CPU-Kopie hält: `p.detach().to("cpu").pin_memory()`.
- Key-Struktur: äußerer Key = Layer-Name, innerer Key = Parameter-/Buffer-Name relativ zum Layer (aus `named_parameters()` / `named_buffers()`).
- Nach dem Anlegen der Master-Kopie: setze die `.data` der Original-Parameter direkt auf die gepinnten Tensoren (`p.data = pinned`), damit nicht doppelt CPU-RAM belegt wird (die ursprünglichen pageable Tensoren werden dadurch freigegeben).
- Achtung Speicher: Pinned Memory ist nicht swappbar. Gib bei `verbose=True` eine Zeile aus, wie viel MB gepinnt wurden. Wenn `pin_memory()` mit RuntimeError fehlschlägt (zu wenig RAM), logge eine Warnung und fahre mit ungepinnten Kopien fort (Flag `self._pinned: bool`).

### A2: Upload = Copy vom Master

Ersetze die Logik in `_prefetch_async` und im synchronen Pfad von `_ensure_on_gpu`:

- Statt `module.to("cuda", non_blocking=True)`: iteriere über die Params/Buffers des Layers und setze `p.data = self._cpu_master[name][pname].to("cuda", non_blocking=True)`.
- Im Prefetch-Stream-Kontext ausführen wie bisher, Event danach recorden.
- **`record_stream`:** Direkt nach jedem `.to("cuda")` im Prefetch-Stream rufe auf dem GPU-Tensor `t.record_stream(torch.cuda.current_stream())` NICHT auf — stattdessen: wenn der Default-Stream den Tensor später konsumiert, muss beim **Offload** `record_stream` auf den Prefetch-Stream gesetzt werden, falls der Tensor im Prefetch-Stream alloziert wurde. Konkret und einfach korrekt: alloziere im Prefetch-Stream, und rufe im `_ensure_on_gpu`-Pfad (wenn auf das Event gewartet wurde) für jeden Param `p.data.record_stream(torch.cuda.current_stream())` auf. Kommentiere im Code, warum das nötig ist (Caching-Allocator + Cross-Stream).

### A3: Offload = Drop statt Copy

Ersetze `_offload`:

- Kein `.to("cpu")` mehr. Stattdessen pro Param/Buffer: `p.data = self._cpu_master[name][pname]` (Zeiger zurück auf die gepinnte Master-Kopie). Der GPU-Tensor verliert seine letzte Referenz und der Allocator gibt den Block frei.
- Pending Prefetch-Event für den Layer entfernen wie bisher, State auf `ON_CPU`, `memory.mark_off_gpu(name)`.
- Wichtig: Falls der Layer gerade `PREFETCHING` ist und offgeloadet werden soll (Edge Case), erst auf das Event synchronisieren, dann droppen — sonst droppst du in einen laufenden Copy hinein.

### A4: `prepare()` und `offload_all()` anpassen

- `prepare()` darf die Layer nicht mehr per `mod.to("cpu")` zurückschieben, sondern nutzt den neuen Drop-Pfad (bzw. setzt initial alle `.data` auf die Master-Kopien).
- `ZeroFlushModel.offload_all()` in `huggingface.py` ruft aktuell `self.model.to("cpu")` — das würde die gepinnten `.data`-Zeiger durch neue pageable Tensoren ersetzen und die ganze Master-Struktur invalidieren. Ändere es so, dass für die getrackten Layer der Drop-Pfad des `LayerManager` genutzt wird und nur die Fixed-Module (Embeddings, Norms, Head) per `.to("cpu")` gehen.

### Akzeptanz A

- `pytest` (falls Tests existieren, sonst Mini-Skript): Mit einem kleinen Modell (`sshleifer/tiny-gpt2` oder `Qwen/Qwen2.5-0.5B`) läuft `ZeroFlushModel(...).generate("Hallo", max_length=20)` durch, Output ist nicht-leer.
- Prüfe nach mehreren `generate()`-Aufrufen hintereinander, dass `torch.cuda.memory_allocated()` nicht monoton wächst (kein Leak durch verwaiste GPU-Tensoren).
- Verifiziere per `p.data.is_pinned()` stichprobenartig, dass offgeloadete Layer wirklich auf den gepinnten Mastern liegen.

---

## Auftrag B — Budget-Tracking beim Prefetch (`hooks.py` + `memory.py`)

- `_prefetch_async` muss **vor** dem Copy `self.memory.mark_on_gpu(name, module)` aufrufen und die zurückgegebene Eviction-Liste abarbeiten (`self._offload(...)` für jeden). Damit reserviert Prefetch sein Budget sofort.
- Im Pre-Hook wird `mark_on_gpu` dann für bereits geprefetchte Layer nur noch zum LRU-Touch (move_to_end) — das kann `mark_on_gpu` schon (early return, wenn Key existiert). Prüfe, dass dieser Pfad keine Doppel-Eviction auslöst.
- Schutz: Prefetch darf niemals den **aktuell rechnenden** Layer oder den unmittelbar nächsten evicten. Gib `mark_on_gpu` einen optionalen Parameter `protected: set[str] | None`, der Namen von der Eviction ausnimmt. Wenn ohne Eviction der geschützten Layer kein Platz ist, brich den Prefetch ab (State bleibt `ON_CPU`, kein Fehler — der Pre-Hook holt den Layer dann synchron).
- `TrainingManager` nutzt denselben `MemoryManager` — die Signaturerweiterung muss dort kompilieren, Verhalten dort unverändert lassen (protected=None).

### Akzeptanz B

- Konstruiere einen Testfall mit absichtlich zu kleinem Budget (z. B. Budget = 1,5 × Layergröße, `prefetch_depth=2`): Es darf kein OOM und keine Eviction des aktiven Layers auftreten; das Modell rechnet korrekt (Vergleich der Logits gegen einen Lauf ohne Offloading, `torch.allclose` mit fp16-Toleranz).
- `MemoryManager.report()` Peak darf das Budget nie überschreiten.

---

## Auftrag C — Wrap-Around-Prefetch für autoregressive Generation (`hooks.py`)

- Im Post-Hook: Prefetch-Ziele modulo Layer-Anzahl berechnen (`(idx + offset) % n_layers`), sodass die letzten Layer bereits Layer 0, 1, … für den nächsten Token vorziehen.
- Ebenso das Offload-Fenster im Post-Hook modulo rechnen — aber Vorsicht: bei sehr kleinen Modellen (n_layers ≤ prefetch_depth + 3) dürfen sich Prefetch- und Offload-Fenster nicht überlappen. Wenn Überlappung droht, Offload-Fenster verkleinern statt Layer zu evicten, die gleich gebraucht werden.
- Der Sonderfall "erster Forward nach `prepare()`" (Layer 0 ist schon da) darf durch den Wrap-Around nicht doppelt geprefetcht werden — `_prefetch_async` ist über den State-Check idempotent, verifiziere das nur.

### Akzeptanz C

- Mit `verbose=True` und einem kurzen `generate` (≥ 5 Tokens): im Log muss sichtbar sein, dass während der letzten Layer eines Durchlaufs bereits `PREF` für Layer 0/1 erscheint.
- Miss tok/s vor/nach (CLI `bench` oder `run`) und notiere die Zahlen im Abschlussbericht.

---

## Auftrag D — Kleinreparaturen

1. **`hooks.py` + `training.py`:** `self._layer_list.index(name)` in Hooks durch ein einmalig gebautes `self._layer_idx: dict[str, int]` ersetzen.
2. **`cache.py`:** `__contains__` respektiert TTL nicht → auf `self.get(key) is not None` umstellen (oder TTL-Check duplizieren, aber Verhalten muss konsistent mit `get` sein). Achtung: `get` hat Seiteneffekt (move_to_end) — implementiere `__contains__` ohne LRU-Touch, aber mit TTL-Check und Lazy-Delete.
3. **`hooks.py` `_move_fixed_to_gpu`:** `startswith(layer_prefixes)` matcht auch `model.layers.10` bei Prefix `model.layers.1` — Präfixe mit angehängtem `.` prüfen bzw. exakte Namensmengen verwenden. (Subtiler Bug: Fixed-Module-Erkennung.)
4. **`cli.py` `cmd_bench`:** `model.model.to("cuda")` für den Standard-Vergleichslauf kollidiert nach Auftrag A mit der Master-Struktur (gleicher Mechanismus wie `offload_all`). Reihenfolge umbauen: erst K4N0N3-Messung, dann für den Standard-Lauf die Hooks entfernen (`remove_hooks`) und das Modell frisch nach CUDA schieben — oder den Standard-Lauf überspringen, wenn das Modell nicht komplett ins VRAM passt (try/except OOM mit sauberer Meldung).
5. **`huggingface.py`:** `generate()` ruft bei jedem Aufruf `self.prepare()` → nach Auftrag A prüfen, ob ein zweiter `prepare()` auf bereits präparierten Zustand idempotent und billig ist (early return, wenn `self._prepared` und Layer-States konsistent).

### Akzeptanz D

- Kein Verhaltensbruch: bestehende Aufrufe aus `cli.py` laufen unverändert.
- Für Punkt 2 einen kleinen Unit-Test schreiben (TTL abgelaufen → `key in cache` ist False).

---

## Auftrag E — KV-Cache-Budget berücksichtigen (`memory.py` + `huggingface.py`)

Der KV-Cache wächst bei Generation im VRAM, taucht im Budget aber nicht auf → er frisst das Layer-Budget still weg.

- Pragmatischer Ansatz (kein Deep-Hook in HF): `ZeroFlushModel.generate()` schätzt vor dem Lauf die KV-Cache-Obergrenze aus Config (`num_hidden_layers × 2 × num_key_value_heads × head_dim × max_length × dtype_size × batch`) und zieht sie als Reserve vom `MemoryManager`-Budget ab (neue Methode `MemoryManager.set_reserve(bytes)`, die `available_bytes`/Eviction-Schwelle entsprechend senkt).
- Bei `verbose=True`: Reserve in MB loggen.
- Fallback: wenn Config-Felder fehlen, Reserve = 10 % des Budgets.

### Akzeptanz E

- `run` mit langem `--max-tokens` (z. B. 256) auf knappem Budget läuft ohne OOM durch.

---

## Auftrag F — Validierung & Abschlussbericht

1. **Korrektheit:** Logits-Vergleich offloaded vs. voll auf GPU (kleines Modell, fp16-Toleranz `atol=1e-3, rtol=1e-3`), zusätzlich greedy-decode-Vergleich (identische Token-Sequenz bei `do_sample=False`).
2. **Performance:** `bench` mit einem realistisch großen Modell (so groß, wie CPU-RAM erlaubt, z. B. 3B in fp16) bei Budget 2048 MB, `prefetch_depth` 1 und 2. Tabelle: tok/s vorher (git stash / alter Stand) vs. nachher.
3. **Speicher:** `torch.cuda.max_memory_allocated()` nach einem `generate` loggen und gegen Budget + Reserve prüfen.
4. Schreib am Ende einen kurzen Bericht (Markdown): was geändert, Messwerte, offene Punkte. Bekannte offene Punkte, die du NICHT bearbeiten sollst, aber im Bericht erwähnen: Quantisierung (GPTQ/AWQ/torchao-int4) als nächster großer Hebel für tok/s; `parallel.py` Multi-GPU-Pfad ist für dieses Setup irrelevant und ungetestet.

## Allgemeine Regeln

- Kleine, thematisch getrennte Commits pro Auftrag (A1–A4 dürfen ein Commit sein, wenn atomar sinnvoller).
- Keine neuen Pflicht-Dependencies. `gguf` und `transformers` bleiben optional wie bisher.
- CPU-only-Pfad (`torch.cuda.is_available() == False`) nach jedem Auftrag einmal smoke-testen (Hooks dürfen dann No-Ops sein, Modell rechnet normal auf CPU).
- Bei Unklarheiten oder wenn ein Akzeptanzkriterium nicht erreichbar ist: stoppen, Befund dokumentieren, nicht drumherum patchen.
