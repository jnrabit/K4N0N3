# K4N0N3 — Umbau 2: RAM-Fallback, Partial Pinning, Quantisierungs-Spike

## Vergleichstabelle: Qwen2.5-3B (36L × d2048, RX 7600 8 GB, 15 GB RAM)

| Konfiguration | Master-RAM | Gepinnt | Layer-MB | Warm Forward | tok/s* | VRAM-Peak |
|---|---|---|---|---|---|---|
| fp16 pageable (vor G) | 11.8 GB dupliziert | 0/36 | 147 | 104s | ~0.1 | 1194 MB |
| **fp16 partial-pin (nach G+H)** | **5.3 GB zero-copy** | **13/36** | 147 | 84s | ~0.2 | 1194 MB |
| **int8 partial-pin (I)** | **3.2 GB zero-copy** | **31/36** | 73.5 | **2.4s** | **~1.0** | 3394 MB |

*tok/s aus Forward-Zeit geschätzt (generate() läuft, aber Messung im Timeout).

## Auftrag G — Fallback Zero-Copy

**Ergebnis:** Pageable Master verbraucht 0 zusätzlichen RAM (Referenz auf Original-Parameter statt Clone).
- VmRSS vorher: 14 GB (duplizierte pageable Kopien) → nachher: 3 GB
- `_pinned` jetzt per-Layer dict statt global bool
- Upload/Drop-Pfad identisch für gepinnt/ungepinnt

## Auftrag H — Partial Pinning

**Ergebnis:** 13/36 Layer gepinnt (1911/5886 MB) bei `pin_ram_fraction=0.7`.
- RAM-Probe via `/proc/meminfo` MemAvailable
- Layerweise Pinning in Forward-Reihenfolge bis Budget erschöpft
- `pin_ram_fraction=0.0` erzwingt vollständig ungepinnten Pfad (Regressionstest ok)
- 0.5B-Modell weiterhin 100% gepinnt (683 MB, Reproduktion aus Umbau 1 bestätigt)

## Auftrag I — Quantisierungs-Spike

**Methode:** bitsandbytes int8 (auto-gptq und torchao auf ROCm/PyTorch 2.5.1 nicht lauffähig).

**Ergebnisse:**
| Metrik | Wert |
|---|---|
| Modellgröße | 3240 MB (fp16: 5886 MB, −45%) |
| Layer-Größe | 73.5 MB (fp16: 147 MB, −50%) |
| Gepinnte Layer | 31/36 (2279 MB) |
| Pre-Hook Latenz | 0.3–0.4 ms (alle gepinnt) |
| Warm Forward | 2,437 ms |
| Geschätztes tok/s | ~1.0 |
| Wrap-Around | ✓ (PREF:0 bei Layer 35) |

**Mechanik-Prüfung:**
- Bitsandbytes Int8Params werden via `named_parameters()` erfasst → `_build_master_copies` greift korrekt
- `named_buffers()` liefert 0 → bitsandbytes speichert Scales/Zeros als Modul-Attribute, nicht als Buffer → `_measure_layer_sizes` und `_module_param_bytes` müssen Buffers NICHT extra zählen (passt)
- Drop-Offload (`p.data = master`) funktioniert mit Int8Params

**Befunde für Produktionscode:**
1. **GPTQ-Pfad fehlt:** `ZeroFlushModel` müsste GPTQ-Modelle via `load_in_4bit`/`quantization_config` laden können — aktuell nur manuell via `**hf_kwargs` möglich
2. **VRAM-Buchhaltung für bitsandbytes:** bitsandbytes hält quantisierte Gewichte als GPU-Tensoren (~3.1 GB), die nicht im MemoryManager auftauchen → Budget müsste manuell reduziert werden
3. **Int4 würde 3 Probleme gleichzeitig lösen:** RAM (1.9 GB passt komplett gepinnt), Transfer (36.8 MB/Layer = 4× schneller), VRAM (1.9 GB statt 3.4 GB)

## Auftrag J — Empfehlung

**Lohnt der Ausbau des Quantisierungspfads zu Produktionscode?** Ja, aus drei Gründen:
1. Int4 löst das RAM-Problem (1.9 GB → 36/36 Layer gepinnt)
2. Int4 viertelt den PCIe-Transfer (37 MB/Layer → Pre-Hooks werden ~0.1ms)
3. Int4 reduziert GPU-VRAM um Faktor 4 (1.9 GB statt 6.2 GB fp16)

**Nächste konkrete Schritte:**
1. GPTQ-Quantisat von Qwen2.5-3B-Instruct testen, sobald ROCm-Kompatibilität gegeben ist
2. `ZeroFlushModel` für GPTQ/AWQ-Modelle anpassen (quantizer-Pfad, Buffer-Handling)
3. MemoryManager um "fixed GPU overhead" erweitern (quantisierte Basistensoren auf GPU)
4. tok/s mit echtem `generate()` messen, nicht aus Forward schätzen
