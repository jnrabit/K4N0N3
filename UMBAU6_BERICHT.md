# K4N0N3 — Umbau 6: Inferenz-Schlusskapitel (Transfer-Decke & Streaming-Maximum)

*Generiert am 2026-07-28 18:01 von `bench/make_report6.py` — alle Zahlen aus den JSONs unter `bench/results/`. Nicht von Hand editieren.*

## Kernaussage

Auftrag 5 verließ das Inferenz-Kapitel mit einer offenen Flanke: „9B nur über Offload, ~300 s/Rewrite, PCIe-gebunden“ — als Vermutung. Auftrag 6 schließt sie mit zwei Zahlen, die es vorher nicht gab:

1. **Die echte, gemessene H2D-Decke dieser Maschine: ~2.88 GB/s** — nicht die angenommenen 12 GB/s. Alle behebbaren Ursachen (Link-Zustand, Clock, Power, ReBAR, Fragmentierung, Stream-Concurrency) sind gemessen ausgeschlossen; es bleibt der ROCm-H2D-DMA-Pfad selbst.
2. **Das Maximum, das der Streaming-Pfad innerhalb dieser Decke herausholt: ~3.06 tok/s** (3B, int4 × spekulativ) — von 0,94 tok/s naiv, also ~3,3×, und verlustfrei bewiesen.

Der Weg dahin folgte einer eingebauten Bremse gegen unnötige Arbeit: die Diagnose (T) durfte den Umbau (U) verbieten, bevor er entstand — und tat es.

**Interaktiv wird Offload-Inferenz damit nicht** (3 tok/s ≈ 0,3 s/Token, ein 64-Token-Rewrite ≈ 21 s), und das stand vorab fest. Die Nische des Streaming-Pfads bleibt Training und Batch-Generierung, wo tok/s zweitrangig ist. Für einen schnellen Rewriter bleibt es beim residenten 3B; für ein starkes MoE-Modell siehe den llama.cpp-Datenpunkt unten.

## Auftrag T — die echte Decke der Maschine (Gate: U übersprungen)

Die frühere Annahme „~12 GB/s Gen4-x8“ war nie gemessen. T misst sie — und findet eine ganz andere Zahl. Alle Werte Median aus 10 CUDA-Event-Zeiten.

- **Link unter Last:** 16.0 GT/s PCIe / x8 (Max 16.0 GT/s PCIe / x8) — voll Gen4 x8, nichts verloren.
- **Roh-Bandbreite** (1 Blob 150 MB): **2.832 GB/s**.
- **Multistream-Decke** (1–4 Streams): 2.884 GB/s (1:2.832, 2:2.884, 3:2.868, 4:2.861) — flach, also ist die Roh-Rate die WAHRE Decke, kein Pro-Queue-Limit.
- **Ist** (echter int8-Layer `model.layers.18`, 73.6 MB, 19 Einzel-Copies): transfer-only **2.803 GB/s**, voll inkl. Dequant 2.526 GB/s.
- **Fragmentierung**: dieselben Bytes als 1 Blob 2.828 GB/s.

**Gate ZU → U ÜBERSPRINGEN: transfer-only Ist 2.803 / Roh 2.832 = 0.99 (≥0,80 = Fragmentierung nicht das Problem)**

Der echte 19-Copy-Layer-Upload liegt bei 0.99 der Roh-Rate — Fragmentierung kostet ~1 %, nicht Faktor 4. **Auftrag U (Staging-Blob) wurde ÜBERSPRUNGEN**, bevor eine Zeile davon entstand: die eingebaute Bremse, die Messung vor Code stellt. Der Dequant kostet ~10 % (Compute, gehört zu V), keine Transfer-Sache.

**Die echte Decke dieser Maschine ist ~2,83–2,88 GB/s H2D** — 4–5× unter der Gen4-x8-Theorie. Ursache ist der ROCm/amdgpu-H2D-DMA-Pfad selbst (Link, Clock, Power, ReBAR, Fragmentierung, Stream-Concurrency alle gemessen ausgeschlossen), **kein PCIe-Bus-Limit im engeren Sinn** und in K4N0N3 nicht behebbar. Damit ist der Boden pro Decode-Token fix: gestreamte Bytes ÷ 2,83 GB/s.

## Auftrag V — spekulatives Decoding: der einzige verbleibende Hebel

Bei gedeckeltem Bus (T) hilft nur, den EINEN teuren Offload-Forward über mehr Tokens zu amortisieren: ein resident laufendes 0,5B-Draft schlägt k Tokens vor, der gestreamte 3B verifiziert alle in einem Durchlauf. Verdrahtet über HFs assisted generation (kein eigener Decode-Loop).

**Scharfrichter:** spekulatives Greedy ist mathematisch verlustfrei — die Token-Sequenz MUSS identisch zur nicht-spekulativen Referenz sein. Jede Abweichung wäre ein Bug im Hook×KV-Rollback, kein „nah genug“.

**Amortisierung direkt gemessen** am Pre-Hook des ersten Layers (Forwards/Token), nicht aus tok/s zurückgerechnet.

| Quant | k | Baseline tok/s | spekulativ tok/s | Fwd/Tok | Akz/Fwd | verlustfrei |
|---|---|---|---|---|---|---|
| int8 | 4 | 0.94 | 1.69 | 0.469 | 2.13 | ✓ |
| int8 | 8 | 0.94 | 1.97 | 0.453 | 2.21 | ✓ ⭐ |
| int8 | 12 | 0.94 | 1.93 | 0.453 | 2.21 | ✓ |
| int4 | 4 | 1.34 | 2.28 | 0.438 | 2.29 | ✓ |
| int4 | 8 | 1.34 | 3.06 | 0.391 | 2.56 | ✓ ⭐ |
| int4 | 12 | 1.34 | 3.00 | 0.391 | 2.56 | ✓ |

**Scharfrichter: ALLE Läufe verlustfrei bestanden ✓** — das Hook×KV-Rollback-Zusammenspiel ist korrekt.

- **int8** (vorregistriert, Marke 3 tok/s): 0.94 → **1.97 tok/s** (k=8) — **VERFEHLT**. Wand ist die Draft-Akzeptanz (~2,2 Tok/Forward saturiert; höheres k bringt nichts).
- **int4** (orthogonaler Hebel — halbe Bytes/Forward): 1.34 → **3.06 tok/s** (k=8) — **ERREICHT**.

**Das Maximum, das der Streaming-Pfad herausholt: ~3,06 tok/s** (int4 × spekulativ). Von der naiven int8-Baseline (0,94 tok/s) sind das ~3,3× — aus zwei orthogonalen Hebeln: weniger Bytes/Forward (int4) und weniger Forwards/Token (spekulativ). U (Staging) trägt nichts bei, weil der Bus schon voll ausgereizt ist.

## Einordnung: der parallele Deployment-Pfad (llama.cpp, MoE)

Nicht jedes starke Modell muss gestreamt werden. Für MoE-Modelle (nicht den dichten 9B) hält ein `llama-server` mit `--override-tensor 'ffn_.*_exps.=CPU'` nur die dünn-aktiven Experten-FFNs im RAM und alles andere auf der GPU:

- qwen3:30b-a3b-2507 (q4): **18/24** auf Holdout 2, **median 5.38 s/Rewrite** — ~9× schneller als Ollamas statischer Split, gleiche Qualität.

Das ist ein ANDERER Pfad (statisches MoE-Placement, kein K4N0N3-Streaming) und MoE-spezifisch. Betrieblich: latenztolerante Rollen (Batch-Synthese, Harvester-Klassifikation), **On-Demand-Dienst, nicht resident neben Training** (18,6 GB kollidieren mit Trainings-RAM).

## Das Inferenz-Kapitel ist geschlossen

Die offene Flanke aus Auftrag 5 („PCIe-gebunden“, Vermutung) ist eine Messung geworden: die Decke ist ~2.88 GB/s (ROCm-DMA), das Maximum des Streaming-Pfads darunter ~3.06 tok/s, verlustfrei. Mehr gibt diese Hardware für gestreamte Inferenz nicht her — Staging (U) war nachweislich kein Hebel, spekulatives Decoding (V) und int4 sind die zwei, die es gibt, und sie sind ausgereizt. Verbleibende Nische: Training und Batch.

## Artefakte

| Datei | Inhalt |
|---|---|
| `bench/pcie_probe.py` + `_pcie_probe_int8_*.json` | T: Roh/Multistream/Ist/Fragmentierung + Gate |
| `bench/eval_speculative.py` + `_eval_speculative_int8/int4_*.json` | V: Amortisierung, verlustfrei-Check, tok/s |
| `bench/harness.py` (`_pcie_link`) | T1: Link-Felder in jedem Lauf |
| `k4n0n3/hooks.py` | Copy-Zähler (T) + Layer-0-Feuerungs-Zähler (V) |
| `k4n0n3/huggingface.py` | `generate(speculative=…)` über HF assisted generation |

- T-Probe: `bench/results/20260728_172255_pcie_probe_int8_final.json`
- V int8: `bench/results/20260728_173815_eval_speculative_int8_v_first.json`
- V int4: `bench/results/20260728_175613_eval_speculative_int4_v_int4.json`
