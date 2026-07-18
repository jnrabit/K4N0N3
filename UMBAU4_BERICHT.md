# K4N0N3 — Umbau 4: Pin-Fix, Group-wise int4, Training-Offload (LoRA)

*Generiert am 2026-07-18 06:32 von `bench/make_report.py` — alle Zahlen stammen aus den JSONs unter `bench/results/`. Nicht von Hand editieren.*

## Inference: Qwen/Qwen2.5-3B (RX 7600 8 GB, ROCm) — Stand nach Pin-Fix

| Konfiguration | Master-RAM MB | Gepinnt | Warm Forward ms | tok/s | VRAM-Peak MB |
|---|---|---|---|---|---|
| fp16 partial-pin | 5292 | 6/36 | 1875.3 | **0.52** | 1196 |
| int8-custom | 2648 | 36/36 | 1049.4 | **0.94** | 1263 |
| int4-per-channel (M5, deprecated) | 1325 | 36/36 | 950.6 | **1.06** | 1340 |
| int4-g128 (group-wise) | 1365 | 36/36 | 721.1 | **1.38** | 1229 |
| int4-g64 (group-wise) | 1406 | 36/36 | 736.7 | **1.36** | 1230 |
| *Referenz ausser Konkurrenz: Ollama GGUF q4 (resident)* | — | — | — | 74.2 | — |

## Pin-Fix-Effekt (O): Auftrag-3-Laeufe vs. neu

| Konfiguration | Gepinnt vorher → nachher | Warm ms vorher → nachher | tok/s vorher → nachher |
|---|---|---|---|
| fp16 partial-pin | 6/36 → **6/36** | 2077.7 → **1875.3** | 0.48 → **0.52** |
| int8-custom | 12/36 → **36/36** | 1078.1 → **1049.4** | 0.93 → **0.94** |
| int4-per-channel (M5, deprecated) | 26/36 → **36/36** | 964.5 → **950.6** | 1.04 → **1.06** |

Randnotiz (im O2-Commit dokumentiert): Der Per-Layer-Reprobe brauchte zusaetzlich einen RAM-Floor (1,5 GB MemAvailable-Rest) — ohne ihn pinnt der fp16-Pfad das System in Swap-Hunger; der erste fp16-Neumesslauf wurde deshalb nach 71 min abgebrochen (14 % Memory-Stall, GPU idle).

### Wiederholbarkeit (< 10 % bei warm_forward_ms, Nachher-Laeufe)

- int8-custom: 1052.3 ms vs. 1049.4 ms → 0.3 % (OK)
- int4-g128 (group-wise): 719.8 ms vs. 721.1 ms → 0.2 % (OK)
- int4-g64 (group-wise): 735.4 ms vs. 736.7 ms → 0.2 % (OK)

## Qualitaet (P2, greedy vs. fp16-Referenz, 32 Tokens)

| Variante | Mechanik (mit=ohne Offload) | Divergenz ab Token | mittl. \|Logit-Diff\| Token 1 |
|---|---|---|---|
| int8 | identisch ✓ | keine (32/32) | 0.0821 |
| int4-per-channel (M5) | identisch ✓ | 0 | — |
| int4-g128 | identisch ✓ | 1 | 0.6322 |
| int4-g64 | identisch ✓ | 17 | 0.7680 |

Akzeptanz P (Divergenz erst nach Token 16 oder gar nicht): Bewertung im Abschluss. Mechanik-Korrektheit ist von Quantisierungsqualitaet getrennt zu lesen.

## Training-Offload: LoRA auf Basis > VRAM (Q3)

Lauf: `20260718_062153_train_lora_beweislauf.json` — 50 Schritte, Batch 1, seq_len 256, Budget 3072 MB, quantize_transfer=int8, grad_checkpointing=True, 1,843,200 Adapter-Parameter.

- (✓) Alle Schritte ohne OOM/Crash
- (✓) Loss faellt: Median erste 10 = 3.6099 → letzte 10 = 1.6220
- (✓) VRAM-Peak 1473 MB ≤ Budget 3072 MB + Reserve
- (✓) Funktionsprobe: generate() mit vs. ohne Adapter unterscheidet sich (greedy_tokens im JSON)

Loss-Kurve: `▆█▃▄▇▄▅▄▃▁▃▆▅▃▁▁▂▃▂▁▆▃▁▄▂` (4.173 → 0.208)

step_time Median: **2082 ms** — bewusst langsam: jeder Schritt streamt alle Layer zweimal (Forward + Backward). Die Zahl ist der Datenpunkt, nicht das Problem.

## Abschluss-Einordnung

- 3B-Inference unter 8 GB VRAM mit int8-Transfer: 1049.4 ms warm forward, **0.94** tok/s (fp16: 1875.3 ms, **0.52**) — Mechanik-Korrektheit und Offload-Wirksamkeit mit Messwerten belegt.
- int4 group-wise: g=64 besteht das Qualitaets-Gate (Divergenz ab Token 17 > 16) bei 736.7 ms / **1.36** tok/s; g=128 ist minimal schneller (721.1 ms), divergiert aber ab Token 1 — Empfehlung: g=64 als Default, wenn int4 genutzt wird.
- **LoRA-Training auf einem Basismodell, das nicht ins VRAM passt** — das kann der kurze Weg (llama.cpp/Ollama) prinzipiell nicht; alle vier Q3-Kriterien mit JSON-Beleg erfuellt.

Was K4N0N3 nachweislich nicht sein will: ein Inference-Ersatz fuer llama.cpp — die Ollama-Referenz (Weights resident, eigene Kernel) liegt bei 74.2 tok/s, Faktor ~50-70 vor dem Layer-Streaming. K4N0N3s Nische ist das PyTorch-Oekosystem bei zu kleinem VRAM: Inference, wenn es sein muss — und Training, wo es sonst gar nicht ginge.

Sinnvolle naechste Schritte, NUR falls das Projekt weitergeht: (1) asymmetrische int4-Quantisierung (Zero-Point pro Gruppe) gegen das Qualitaets-Gate, (2) Optimizer-State/Writeback-Offload fuer echtes Voll-Finetuning, (3) Multi-GPU-/NVMe-Staging. Ausdruecklich ebenso vertretbar: **hier ist ein guter Abschluss** — die Kernthese (transparentes Layer-Streaming im PyTorch-Oekosystem, inkl. Training) ist belegt, die Grenzen sind vermessen und dokumentiert.
