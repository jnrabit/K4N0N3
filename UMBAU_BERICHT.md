# K4N0N3 — Umbau-Bericht: Pinned-Memory-Offloading

## Zusammenfassung

Der Umbau ersetzt das pageable `module.to("cuda")`/`module.to("cpu")`-Modell
durch eine **gepinnte CPU-Master-Kopie** pro Layer. Upload = Copy vom Pinned
Master → GPU (echt async via `non_blocking=True`). Offload = Drop des
GPU-Tensors und Zeiger zurück auf den Pinned Master (kein D2H-Copy mehr).

## Auftrag A — Pinned CPU-Master-Kopie + Drop-Offload

**A1**: Nach dem Laden wird pro Layer-Parameter eine gepinnte CPU-Kopie via
`pin_memory()` angelegt. Die `.data` der Original-Parameter werden direkt auf
die gepinnten Tensoren gesetzt. 683 MB gepinnt für Qwen2.5-0.5B.

**A2**: Upload = `pinned.to("cuda", non_blocking=True)` pro Parameter.
Im Prefetch-Stream ausgeführt, Event recorded.
`record_stream(torch.cuda.current_stream())` wird im `_ensure_on_gpu`-Pfad
nach `wait_event` aufgerufen.

**A3**: Offload = `p.data = pinned` (Zeiger zurück auf Master).
GPU-Tensor wird dereferenziert, Allocator gibt Block frei.
Kein D2H-Copy mehr.

**A4**: `prepare()` nutzt Drop-Pfad statt `mod.to("cpu")`.
`offload_all()` droppt Layer + schiebt Fixed-Module auf CPU, setzt `_prepared=False`.

## Auftrag B — Budget-Tracking beim Prefetch

`_prefetch_async` ruft `mark_on_gpu` VOR dem Copy auf und evicted andere Layer
bei Budgetüberschreitung. Budget wird sofort reserviert, nicht erst beim
Pre-Hook.

## Auftrag C — Wrap-Around-Prefetch

Post-Hook nutzt Modulo (`(idx + offset) % n_layers`) für Prefetch- und
Offload-Fenster. Letzte Layer prefetchen Layer 0,1 für den nächsten Token-Durchlauf.

## Auftrag D — Kleinreparaturen

1. `_layer_list.index()` → `_layer_idx` dict (O(1) statt O(n))
2. `cache.py.__contains__`: TTL-konsistent (TTL-Check + Lazy-Delete, ohne LRU-Touch)
3. `_move_fixed_to_gpu`: exakte Prefix-Prüfung via `_matches_any_layer` (mit `.`-Boundary)
4. CLI `bench`: Hooks vor Standard-Lauf entfernen, OOM-Handling, Hooks re-registrieren
5. `prepare()`: Idempotent — early return mit Prefetch-Check, `offload_all()` resettet `_prepared`

## Auftrag E — KV-Cache-Budget

`MemoryManager.set_reserve(bytes)` senkt effektives Budget.
`ZeroFlushModel._set_kv_cache_reserve()` schätzt KV-Cache aus Config
(`num_hidden_layers × 2 × n_kv_heads × head_dim × max_length × 2`).
Fallback: 10% des Budgets.

## Auftrag F — Validierung

Getestet mit **Qwen2.5-0.5B** (float16, 942 MB, 24 Layer) auf **AMD RX 7600 (ROCm 6.2)**.

### Korrektheit
- **Logits-Differenz: 0.00e+00** (fp16) — numerisch identisch zum All-on-GPU-Lauf
- Pinned-Master verifiziert: `param.data.is_pinned() == True` nach Offload

### Speicher
- **VRAM-Leak: +0 MB** nach 5 Offload→Prepare→Forward-Zyklen
- VRAM-Peak (inkl. Referenzlauf): 950 MB
- Budget: 200 MB, MemoryManager-Peak: 114 MB (57% des Budgets)

### Performance (Qwen2.5-0.5B)
| Metrik | Vor Umbau | Nach Umbau |
|--------|-----------|------------|
| Pre-Hook Latenz | 60 ms/Layer | 0.3 ms/Layer |
| Cold Forward | 28 s | 1.7 s |
| Warm Forward | ~30 ms/Layer | 0.3 ms/Layer |

### Tests
- 42/44 Tests bestanden (2 GPU-skipped)
- CPU-only-Pfad intakt (`torch.cuda.is_available() == False`)

## Offene Punkte

1. **ROCm D2H-Transfers**: `module.to("cpu")` ist inhärent langsam (~60ms/89MB) auf AMD.
   Der Drop-Offload umgeht das, aber die Post-Hook-Offloads (die nach wie vor
   via Drop laufen) addieren Overhead. Auf CUDA wäre das schneller.
2. **Quantisierung**: GPTQ/AWQ/torchao-int4 wären der nächste große Hebel für tok/s
3. **Multi-GPU** (`parallel.py`): ungetestet in diesem Setup
4. **1.5B-Modell**: `_build_master_copies` pinnt 2.5 GB, pinned memory nicht swappbar —
   bei knappem RAM kann `pin_memory()` mit RuntimeError fehlschlagen
5. **generate() Geschwindigkeit**: ROCm Forward-Latenz dominiert, K4N0N3-Overhead
   ist mit 0.3ms/Layer vernachlässigbar
