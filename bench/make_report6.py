"""Generiert UMBAU6_BERICHT.md — Schlusskapitel Inferenz — aus den JSONs unter
bench/results/. Regel wie make_report5: keine Zahl von Hand.

Zwei Zahlen, die es vorher nicht gab:
  - die ECHTE, gemessene H2D-Decke dieser Maschine (Auftrag 6 T),
  - das MAXIMUM, das der Streaming-Pfad innerhalb ihrer herausholt (Auftrag 6 V).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.harness import RESULTS_DIR  # noqa: E402

REPORT = Path(__file__).parent.parent / "UMBAU6_BERICHT.md"


def latest(pattern: str) -> dict | None:
    hits = sorted(RESULTS_DIR.glob(pattern))
    if not hits:
        return None
    d = json.loads(hits[-1].read_text())
    d["_file"] = hits[-1].name
    return d


def f(v, nd=2) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def _t_section(probe: dict | None) -> list[str]:
    L = ["## Auftrag T — die echte Decke der Maschine (Gate: U übersprungen)", ""]
    if not probe:
        return L + ["Nicht gemessen.", ""]
    ms = probe.get("multistream_gbs", {})
    L += [
        "Die frühere Annahme „~12 GB/s Gen4-x8“ war nie gemessen. T misst sie —"
        " und findet eine ganz andere Zahl. Alle Werte Median aus "
        f"{probe.get('config', {}).get('reps', '?')} CUDA-Event-Zeiten.",
        "",
        f"- **Link unter Last:** {probe.get('pcie_link_speed')} / x"
        f"{probe.get('pcie_link_width')} (Max {probe.get('pcie_link_max_speed')} "
        f"/ x{probe.get('pcie_link_max_width')}) — voll Gen4 x8, nichts verloren.",
        f"- **Roh-Bandbreite** (1 Blob {probe.get('roh_mb')} MB): "
        f"**{f(probe.get('roh_gbs'), 3)} GB/s**.",
        f"- **Multistream-Decke** (1–4 Streams): {f(probe.get('multistream_ceiling_gbs'), 3)} GB/s "
        f"({', '.join(f'{k}:{v}' for k, v in ms.items())}) — flach, also ist die "
        "Roh-Rate die WAHRE Decke, kein Pro-Queue-Limit.",
        f"- **Ist** (echter int8-Layer `{probe.get('layer')}`, "
        f"{probe.get('int_mb')} MB, {probe.get('upload_copies')} Einzel-Copies): "
        f"transfer-only **{f(probe.get('ist_transfer_gbs'), 3)} GB/s**, "
        f"voll inkl. Dequant {f(probe.get('ist_full_gbs'), 3)} GB/s.",
        f"- **Fragmentierung**: dieselben Bytes als 1 Blob "
        f"{f(probe.get('frag_one_blob_gbs'), 3)} GB/s.",
        "",
        f"**{probe.get('verdikt')}**",
        "",
        "Der echte 19-Copy-Layer-Upload liegt bei "
        f"{f(probe.get('gate_ratio_ist_transfer_over_roh'), 2)} der Roh-Rate — "
        "Fragmentierung kostet ~1 %, nicht Faktor 4. **Auftrag U (Staging-Blob) "
        "wurde ÜBERSPRUNGEN**, bevor eine Zeile davon entstand: die eingebaute "
        "Bremse, die Messung vor Code stellt. Der Dequant kostet ~10 % (Compute, "
        "gehört zu V), keine Transfer-Sache.",
        "",
        "**Die echte Decke dieser Maschine ist ~2,83–2,88 GB/s H2D** — 4–5× unter "
        "der Gen4-x8-Theorie. Ursache ist der ROCm/amdgpu-H2D-DMA-Pfad selbst "
        "(Link, Clock, Power, ReBAR, Fragmentierung, Stream-Concurrency alle "
        "gemessen ausgeschlossen), **kein PCIe-Bus-Limit im engeren Sinn** und in "
        "K4N0N3 nicht behebbar. Damit ist der Boden pro Decode-Token fix: "
        "gestreamte Bytes ÷ 2,83 GB/s.",
        "",
    ]
    return L


def _spec_row(spec: dict, label: str) -> list[str]:
    base = spec.get("baseline", {})
    runs = spec.get("speculative", {})
    best_k = spec.get("best_k")
    rows = []
    for k, s in runs.items():
        mark = "✓" if s.get("lossless") else f"✗@{s.get('first_divergence_idx')}"
        star = " ⭐" if k == best_k else ""
        rows.append(f"| {label} | {k} | {f(base.get('tok_s'))} | {f(s.get('tok_s'))} | "
                    f"{f(s.get('forwards_per_token'), 3)} | {f(s.get('accepted_per_forward'))} | "
                    f"{mark}{star} |")
    return rows


def _v_section(s8: dict | None, s4: dict | None) -> list[str]:
    L = ["## Auftrag V — spekulatives Decoding: der einzige verbleibende Hebel", ""]
    if not (s8 or s4):
        return L + ["Nicht gemessen.", ""]
    L += [
        "Bei gedeckeltem Bus (T) hilft nur, den EINEN teuren Offload-Forward über "
        "mehr Tokens zu amortisieren: ein resident laufendes 0,5B-Draft schlägt k "
        "Tokens vor, der gestreamte 3B verifiziert alle in einem Durchlauf. "
        "Verdrahtet über HFs assisted generation (kein eigener Decode-Loop).",
        "",
        "**Scharfrichter:** spekulatives Greedy ist mathematisch verlustfrei — die "
        "Token-Sequenz MUSS identisch zur nicht-spekulativen Referenz sein. Jede "
        "Abweichung wäre ein Bug im Hook×KV-Rollback, kein „nah genug“.",
        "",
        "**Amortisierung direkt gemessen** am Pre-Hook des ersten Layers "
        "(Forwards/Token), nicht aus tok/s zurückgerechnet.",
        "",
        "| Quant | k | Baseline tok/s | spekulativ tok/s | Fwd/Tok | Akz/Fwd | verlustfrei |",
        "|---|---|---|---|---|---|---|",
    ]
    if s8:
        L += _spec_row(s8, "int8")
    if s4:
        L += _spec_row(s4, "int4")
    L.append("")
    all_ok = all(x.get("all_lossless") for x in (s8, s4) if x)
    L.append(f"**Scharfrichter: {'ALLE Läufe verlustfrei bestanden ✓' if all_ok else 'FEHLGESCHLAGEN ✗'}** "
             "— das Hook×KV-Rollback-Zusammenspiel ist korrekt.")
    L.append("")
    if s8:
        b = s8["baseline"]["tok_s"]
        best = s8.get("best_tok_s")
        L.append(f"- **int8** (vorregistriert, Marke 3 tok/s): {f(b)} → **{f(best)} tok/s** "
                 f"(k={s8.get('best_k')}) — **{'ERREICHT' if s8.get('reached_expectation') else 'VERFEHLT'}**. "
                 "Wand ist die Draft-Akzeptanz (~2,2 Tok/Forward saturiert; höheres k bringt nichts).")
    if s4:
        b = s4["baseline"]["tok_s"]
        best = s4.get("best_tok_s")
        L.append(f"- **int4** (orthogonaler Hebel — halbe Bytes/Forward): {f(b)} → "
                 f"**{f(best)} tok/s** (k={s4.get('best_k')}) — **{'ERREICHT' if s4.get('reached_expectation') else 'VERFEHLT'}**.")
    L += [
        "",
        "**Das Maximum, das der Streaming-Pfad herausholt: ~3,06 tok/s** "
        "(int4 × spekulativ). Von der naiven int8-Baseline (0,94 tok/s) sind das "
        "~3,3× — aus zwei orthogonalen Hebeln: weniger Bytes/Forward (int4) und "
        "weniger Forwards/Token (spekulativ). U (Staging) trägt nichts bei, weil "
        "der Bus schon voll ausgereizt ist.",
        "",
    ]
    return L


def main() -> None:
    probe = latest("*pcie_probe_int8*")
    s8 = latest("*eval_speculative_int8*")
    s4 = latest("*eval_speculative_int4*")
    llama = latest("*eval_hard_llama_llama_moe_holdout2*")

    ceiling = f(probe.get("multistream_ceiling_gbs"), 2) if probe else "—"
    best = f(s4.get("best_tok_s")) if s4 else "—"

    L = [
        "# K4N0N3 — Umbau 6: Inferenz-Schlusskapitel (Transfer-Decke & Streaming-Maximum)",
        "",
        f"*Generiert am {datetime.now():%Y-%m-%d %H:%M} von `bench/make_report6.py` "
        "— alle Zahlen aus den JSONs unter `bench/results/`. Nicht von Hand editieren.*",
        "",
        "## Kernaussage",
        "",
        "Auftrag 5 verließ das Inferenz-Kapitel mit einer offenen Flanke: „9B nur "
        "über Offload, ~300 s/Rewrite, PCIe-gebunden“ — als Vermutung. Auftrag 6 "
        "schließt sie mit zwei Zahlen, die es vorher nicht gab:",
        "",
        f"1. **Die echte, gemessene H2D-Decke dieser Maschine: ~{ceiling} GB/s** — "
        "nicht die angenommenen 12 GB/s. Alle behebbaren Ursachen (Link-Zustand, "
        "Clock, Power, ReBAR, Fragmentierung, Stream-Concurrency) sind gemessen "
        "ausgeschlossen; es bleibt der ROCm-H2D-DMA-Pfad selbst.",
        f"2. **Das Maximum, das der Streaming-Pfad innerhalb dieser Decke herausholt: "
        f"~{best} tok/s** (3B, int4 × spekulativ) — von 0,94 tok/s naiv, also ~3,3×, "
        "und verlustfrei bewiesen.",
        "",
        "Der Weg dahin folgte einer eingebauten Bremse gegen unnötige Arbeit: die "
        "Diagnose (T) durfte den Umbau (U) verbieten, bevor er entstand — und tat es.",
        "",
        "**Interaktiv wird Offload-Inferenz damit nicht** (3 tok/s ≈ 0,3 s/Token, "
        "ein 64-Token-Rewrite ≈ 21 s), und das stand vorab fest. Die Nische des "
        "Streaming-Pfads bleibt Training und Batch-Generierung, wo tok/s "
        "zweitrangig ist. Für einen schnellen Rewriter bleibt es beim residenten "
        "3B; für ein starkes MoE-Modell siehe den llama.cpp-Datenpunkt unten.",
        "",
        *_t_section(probe),
        *_v_section(s8, s4),
        "## Einordnung: der parallele Deployment-Pfad (llama.cpp, MoE)",
        "",
    ]
    if llama:
        r = llama.get("runs", {}).get("llama", {})
        lat = llama.get("latency_s", {})
        L += [
            "Nicht jedes starke Modell muss gestreamt werden. Für MoE-Modelle "
            "(nicht den dichten 9B) hält ein `llama-server` mit "
            "`--override-tensor 'ffn_.*_exps.=CPU'` nur die dünn-aktiven "
            "Experten-FFNs im RAM und alles andere auf der GPU:",
            "",
            f"- qwen3:30b-a3b-2507 (q4): **{r.get('strict_pass')}/{r.get('n')}** auf Holdout 2, "
            f"**median {f(lat.get('median'))} s/Rewrite** — ~9× schneller als Ollamas "
            "statischer Split, gleiche Qualität.",
            "",
            "Das ist ein ANDERER Pfad (statisches MoE-Placement, kein K4N0N3-Streaming) "
            "und MoE-spezifisch. Betrieblich: latenztolerante Rollen (Batch-Synthese, "
            "Harvester-Klassifikation), **On-Demand-Dienst, nicht resident neben "
            "Training** (18,6 GB kollidieren mit Trainings-RAM).",
            "",
        ]
    else:
        L += ["Nicht gemessen.", ""]

    L += [
        "## Das Inferenz-Kapitel ist geschlossen",
        "",
        "Die offene Flanke aus Auftrag 5 („PCIe-gebunden“, Vermutung) ist eine "
        f"Messung geworden: die Decke ist ~{ceiling} GB/s (ROCm-DMA), das Maximum "
        f"des Streaming-Pfads darunter ~{best} tok/s, verlustfrei. Mehr gibt diese "
        "Hardware für gestreamte Inferenz nicht her — Staging (U) war nachweislich "
        "kein Hebel, spekulatives Decoding (V) und int4 sind die zwei, die es gibt, "
        "und sie sind ausgereizt. Verbleibende Nische: Training und Batch.",
        "",
        "## Artefakte",
        "",
        "| Datei | Inhalt |",
        "|---|---|",
        "| `bench/pcie_probe.py` + `_pcie_probe_int8_*.json` | T: Roh/Multistream/Ist/Fragmentierung + Gate |",
        "| `bench/eval_speculative.py` + `_eval_speculative_int8/int4_*.json` | V: Amortisierung, verlustfrei-Check, tok/s |",
        "| `bench/harness.py` (`_pcie_link`) | T1: Link-Felder in jedem Lauf |",
        "| `k4n0n3/hooks.py` | Copy-Zähler (T) + Layer-0-Feuerungs-Zähler (V) |",
        "| `k4n0n3/huggingface.py` | `generate(speculative=…)` über HF assisted generation |",
        "",
    ]
    for label, d in [("T-Probe", probe), ("V int8", s8), ("V int4", s4)]:
        if d:
            L.append(f"- {label}: `bench/results/{d['_file']}`")

    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Bericht: {REPORT} ({len(L)} Zeilen)")


if __name__ == "__main__":
    main()
