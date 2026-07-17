# K4N0N3 — Umbau 3: Mess-Harness, bitsandbytes-Verdikt, Custom int8

*Generiert am 2026-07-17 13:29 von `bench/make_report.py` — alle Zahlen stammen aus den JSONs unter `bench/results/`. Nicht von Hand editieren.*

## Vergleich: Qwen/Qwen2.5-3B (RX 7600 8 GB, ROCm)

| Konfiguration | Master-RAM MB | Gepinnt | Warm Forward ms | tok/s | VRAM-Peak MB | Offload-Free MB |
|---|---|---|---|---|---|---|
| fp16 pageable | 5292 | 0/36 | 1989.5 | **0.45** | 1196 | 150.0 |
| fp16 partial-pin | 5292 | 6/36 | 2077.7 | **0.48** | 1196 | 150.0 |
| int8-custom | 2648 | 12/36 | 1078.1 | **0.93** | 1263 | 150.0 |
| int4-custom (gepackt) | 1325 | 26/36 | 964.5 | **1.04** | 1340 | 150.0 |

`Offload-Free` = real freigegebenes VRAM bei erzwungenem Layer-Drop (muss ≈ Layer-GPU-Groesse sein — die Kennzahl, die beim bitsandbytes-Lauf in Umbau 2 fehlte).

### Cold Forward (separat — enthaelt CUDA-Init/Allocator-Warmup, NICHT mit Warm vergleichen)

| Konfiguration | Cold Forward ms | Ladezeit s |
|---|---|---|
| fp16 pageable | 5157.7 | 33.7 |
| fp16 partial-pin | 9807.5 | 43.4 |
| int8-custom | 1934.8 | 57.5 |
| int4-custom (gepackt) | 1757.4 | 81.1 |

### Wiederholbarkeit (Akzeptanz K: < 10 % Abweichung bei warm_forward_ms)

- fp16 partial-pin: 1949.1 ms vs. 2077.7 ms → 6.6 % (OK)
- int8-custom: 1075.7 ms vs. 1078.1 ms → 0.2 % (OK)
- int4-custom (gepackt): 961.3 ms vs. 964.5 ms → 0.3 % (OK)

## Referenz ausser Konkurrenz: GGUF q4 via Ollama/llama.cpp-ROCm

`qwen2.5:3b` auf derselben Maschine: **74.15 tok/s** (108 Tokens, aus der Ollama---verbose-Statistik).

Kein K4N0N3-Vergleich im engen Sinn (anderes Format, eigene Kernel, Weights dauerhaft im VRAM) — sondern die ehrliche Antwort auf "was waere der kurze Weg gewesen".

## L-Verdikt: bitsandbytes unter echtem Offload-Druck

**inkompatibel** — Drop gibt 0.0 statt ~74 MB frei; Offloading findet real nicht statt (Spike-Verdikt woertlich: "teilkompatibel — Ergebnis korrekt, aber Drop gibt VRAM nicht (voll) frei").

- Layer 0 nach Laden: 147.1 MB GPU, 26 GPU-Tensoren
- L1 q_proj.weight: {'dtype': 'torch.int8', 'device': 'cuda:0', 'scb': 'False'} → {'dtype': 'torch.int8', 'device': 'cpu', 'scb_device': 'None'}
- Messpunkt 1 offload_frees_mb=0.0126953125 (erwartet ~74)
- Nach Drop verbleiben auf GPU: 0.0 MB (0 Tensoren)
- Messpunkt 3 greedy identisch mit Referenz: True

Ursachen-Nachmessung (`20260717_123833_l_spike_bnb_residual.json`):
- nach Masse-.to(cpu): memory_allocated=3246 MB
- nach Druck-Forward: memory_allocated=3321 MB (Modell int8 ~3240 MB — bleibt alles liegen?)
- Layer 0 (gedroppt) — GPU-Tensoren in Linear8bitLt.state: 21
-   self_attn.q_proj.state.CB:torch.int8:4.0MB
-   self_attn.q_proj.state.SCB:torch.float32:0.0MB
-   self_attn.q_proj.state.idx:torch.int64:0.0MB
-   self_attn.k_proj.state.CB:torch.int8:0.5MB
-   self_attn.k_proj.state.SCB:torch.float32:0.0MB
-   self_attn.k_proj.state.idx:torch.int64:0.0MB
- global via gc erreichbare GPU-Tensoren: 3390 MB

## M3: Korrektheit int8/int4-custom

### int8-custom
- **Mechanik-Korrektheit bestanden**: greedy_tokens mit vs. ohne Offloading (gleiche Quant-Master) identisch.
- Greedy vs. fp16-Referenz: keine Divergenz innerhalb der 32 Tokens (Referenz: `20260717_122139_qwen2.5-3b_fp16_pin0.7_bud4096_pf1_k4_run2.json`).
- Mittlere |Logit-Differenz| am ersten Token: 0.0821.

### int4-custom
- **Mechanik-Korrektheit bestanden**: greedy_tokens mit vs. ohne Offloading (gleiche Quant-Master) identisch.
- Greedy vs. fp16-Referenz: Divergenz ab Token 0 von 32 (Referenz: `20260717_122139_qwen2.5-3b_fp16_pin0.7_bud4096_pf1_k4_run2.json`).

Bewertung der Qualitaet: dem Menschen ueberlassen — das sind die Zahlen.

## Empfehlung

- Warm Forward: int8-custom 1078.1 ms vs. fp16 partial-pin 2077.7 ms → Faktor 1.93.
- Pinning: int8-custom 12/36 vs. fp16 6/36 (Master-RAM 2648 vs. 5292 MB).
- tok/s: int8-custom **0.93** vs. fp16 **0.48**.
- int4-gepackt (M5): warm 964.5 ms (vs. int8 1078.1 ms, Faktor 1.12), tok/s **1.04**, Master-RAM 1325 MB, gepinnt 26/36 — Qualitaet siehe M3-Abschnitt.

**int8-custom in Produktcode ueberfuehren: ja** — Transferhalbierung schlaegt messbar durch (Zahlen oben).
**int4 in dieser Form: nein** — nur Faktor 1.12 schneller als int8, aber greedy divergiert bereits ab Token 0 von der fp16-Referenz (per-Channel ohne Grouping zu grob). Vor einem weiteren int4-Anlauf: Group-wise-Quantisierung, dann neu messen.
