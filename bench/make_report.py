"""Auftrag N — Bericht aus den Harness-JSONs generieren.

Liest alle JSONs unter bench/results/, gruppiert nach Konfiguration
(neuester Lauf pro Konfiguration zaehlt, Wiederholungslaeufe dienen der
<10%-Wiederholbarkeits-Pruefung) und schreibt UMBAU3_BERICHT.md.
Jede Zahl im Bericht stammt aus einem JSON — nichts wird von Hand getippt.
Fehlende Messungen erscheinen als "nicht messbar, weil X", nie als Schaetzwert.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
REPORT = Path(__file__).parent.parent / "UMBAU3_BERICHT.md"

MAIN_MODEL = "Qwen/Qwen2.5-3B"


def row_label(cfg: dict) -> str | None:
    """Ordnet eine Harness-Konfiguration einer Berichtszeile zu (None = ignorieren)."""
    if cfg.get("model") != MAIN_MODEL:
        return None
    if cfg.get("quantize_transfer") == "int4":
        return "int4-custom (gepackt)"
    if cfg.get("quantize_transfer"):
        return "int8-custom"
    if cfg.get("pin_ram_fraction", 0) == 0.0:
        return "fp16 pageable"
    return "fp16 partial-pin"


def fmt(v, nd=1) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def tok_s_cell(run: dict) -> str:
    if run.get("tokens_per_s") is not None:
        return f"**{run['tokens_per_s']:.2f}**"
    if run.get("timeout"):
        return f"Timeout >{run['config'].get('generate_timeout_s', 600):.0f}s"
    if run.get("generate_error"):
        return "Fehler (s. JSON)"
    return "—"


def main() -> None:
    runs, ollama, spikes, m3s = [], [], [], []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        data["_file"] = f.name
        if data.get("reference_only"):
            ollama.append(data)
        elif data.get("spike"):
            spikes.append(data)
        elif data.get("m3"):
            m3s.append(data)
        elif "config" in data:
            runs.append(data)

    # Gruppieren: Label -> [Laeufe chronologisch]
    groups: dict[str, list[dict]] = {}
    for r in runs:
        label = row_label(r["config"])
        if label:
            groups.setdefault(label, []).append(r)

    order = ["fp16 pageable", "fp16 partial-pin", "int8-custom", "int4-custom (gepackt)"]
    lines = [
        "# K4N0N3 — Umbau 3: Mess-Harness, bitsandbytes-Verdikt, Custom int8",
        "",
        f"*Generiert am {datetime.now():%Y-%m-%d %H:%M} von `bench/make_report.py` — "
        "alle Zahlen stammen aus den JSONs unter `bench/results/`. "
        "Nicht von Hand editieren.*",
        "",
    ]

    # -- Haupttabelle --------------------------------------------------------
    lines += [
        f"## Vergleich: {MAIN_MODEL} (RX 7600 8 GB, ROCm)",
        "",
        "| Konfiguration | Master-RAM MB | Gepinnt | Warm Forward ms | tok/s | VRAM-Peak MB | Offload-Free MB |",
        "|---|---|---|---|---|---|---|",
    ]
    latest: dict[str, dict] = {}
    for label in order:
        if label not in groups:
            continue
        r = groups[label][-1]
        latest[label] = r
        wf = r.get("warm_forward_ms") or {}
        lines.append(
            f"| {label} | {fmt(r.get('master_ram_mb'), 0)} | {r.get('pinned_layers', '—')} "
            f"| {fmt(wf.get('median'))} | {tok_s_cell(r)} "
            f"| {fmt(r.get('vram_peak_mb'), 0)} | {fmt(r.get('offload_frees_mb'))} |"
        )
    lines += [
        "",
        "`Offload-Free` = real freigegebenes VRAM bei erzwungenem Layer-Drop "
        "(muss ≈ Layer-GPU-Groesse sein — die Kennzahl, die beim "
        "bitsandbytes-Lauf in Umbau 2 fehlte).",
        "",
    ]

    # -- Cold separat --------------------------------------------------------
    lines += [
        "### Cold Forward (separat — enthaelt CUDA-Init/Allocator-Warmup, "
        "NICHT mit Warm vergleichen)",
        "",
        "| Konfiguration | Cold Forward ms | Ladezeit s |",
        "|---|---|---|",
    ]
    for label, r in latest.items():
        lines.append(f"| {label} | {fmt(r.get('cold_forward_ms'))} | {fmt(r.get('load_s'))} |")
    lines.append("")

    # -- Wiederholbarkeit (Akzeptanz K) --------------------------------------
    lines += ["### Wiederholbarkeit (Akzeptanz K: < 10 % Abweichung bei warm_forward_ms)", ""]
    any_repeat = False
    for label, g in groups.items():
        if len(g) < 2:
            continue
        any_repeat = True
        m1 = g[-2]["warm_forward_ms"]["median"]
        m2 = g[-1]["warm_forward_ms"]["median"]
        dev = abs(m2 - m1) / m1 * 100
        status = "OK" if dev < 10 else "VERFEHLT — Messaufbau pruefen"
        lines.append(f"- {label}: {m1} ms vs. {m2} ms → {dev:.1f} % ({status})")
    if not any_repeat:
        lines.append("- nicht messbar, weil keine Konfiguration doppelt gelaufen ist")
    lines.append("")

    # -- Ollama-Referenz -----------------------------------------------------
    lines += ["## Referenz ausser Konkurrenz: GGUF q4 via Ollama/llama.cpp-ROCm", ""]
    if ollama:
        o = ollama[-1]
        lines += [
            f"`{o['config']['model']}` auf derselben Maschine: "
            f"**{fmt(o.get('tokens_per_s'), 2)} tok/s** "
            f"({o.get('generate_tokens')} Tokens, aus der Ollama---verbose-Statistik).",
            "",
            "Kein K4N0N3-Vergleich im engen Sinn (anderes Format, eigene Kernel, "
            "Weights dauerhaft im VRAM) — sondern die ehrliche Antwort auf "
            "\"was waere der kurze Weg gewesen\".",
        ]
    else:
        lines.append("nicht messbar, weil kein Ollama-Referenzlauf vorliegt "
                     "(`bench/ollama_reference.py` ausfuehren).")
    lines.append("")

    # -- L-Verdikt -----------------------------------------------------------
    lines += ["## L-Verdikt: bitsandbytes unter echtem Offload-Druck", ""]
    if spikes:
        s = spikes[-1]
        lines.append(f"**{s.get('verdict', 'kein Verdikt')}**")
        lines.append("")
        for f_ in s.get("findings", []):
            lines.append(f"- {f_}")
    else:
        lines.append("nicht messbar, weil kein L-Spike-Lauf vorliegt.")
    lines.append("")

    # -- M3: Korrektheit int8-custom -----------------------------------------
    lines += ["## M3: Korrektheit int8-custom", ""]
    if m3s:
        m = m3s[-1]
        mech = m.get("mechanik_identisch")
        if mech is True:
            lines.append("- **Mechanik-Korrektheit bestanden**: greedy_tokens mit vs. "
                         "ohne Offloading (gleiche int8-Master) identisch.")
        elif mech is False:
            lines.append("- **Mechanik-Korrektheit VERFEHLT**: greedy_tokens weichen ab "
                         "— hartes Kriterium, Details im JSON "
                         f"(`{m['_file']}`).")
        else:
            lines.append("- Mechanik-Check nicht messbar, weil: "
                         f"{m.get('full_gpu_error', 'unbekannt')}")
        if m.get("divergenz_vs_fp16_ab_token") is not None:
            lines.append(f"- Quantisierungsqualitaet: Divergenz zur fp16-Referenz "
                         f"ab Token {m['divergenz_vs_fp16_ab_token']} von 32 "
                         f"(Referenz: `{m.get('fp16_greedy_source')}`).")
        if m.get("mean_abs_logit_diff_first_token") is not None:
            lines.append(f"- Mittlere |Logit-Differenz| am ersten Token: "
                         f"{m['mean_abs_logit_diff_first_token']}.")
        lines.append("- Bewertung der Qualitaet: dem Menschen ueberlassen — "
                     "das sind die Zahlen.")
    else:
        lines.append("nicht messbar, weil kein M3-Lauf vorliegt "
                     "(`bench/m3_correctness.py` ausfuehren).")
    lines.append("")

    # -- Empfehlung (nur aus JSON-Zahlen) ------------------------------------
    lines += ["## Empfehlung", ""]
    fp16 = latest.get("fp16 partial-pin")
    int8 = latest.get("int8-custom")
    if fp16 and int8:
        wf_fp16 = fp16["warm_forward_ms"]["median"]
        wf_int8 = int8["warm_forward_ms"]["median"]
        factor = wf_fp16 / wf_int8 if wf_int8 else None
        lines.append(
            f"- Warm Forward: int8-custom {fmt(wf_int8)} ms vs. fp16 partial-pin "
            f"{fmt(wf_fp16)} ms → Faktor {fmt(factor, 2)}."
        )
        lines.append(
            f"- Pinning: int8-custom {int8.get('pinned_layers')} vs. fp16 "
            f"{fp16.get('pinned_layers')} (Master-RAM {fmt(int8.get('master_ram_mb'), 0)} "
            f"vs. {fmt(fp16.get('master_ram_mb'), 0)} MB)."
        )
        lines.append(f"- tok/s: int8-custom {tok_s_cell(int8)} vs. fp16 {tok_s_cell(fp16)}.")
        if factor and factor > 1.5 and int8.get("tokens_per_s"):
            lines.append("")
            lines.append(
                "**int8-custom in Produktcode ueberfuehren: ja** — Transferhalbierung "
                "schlaegt messbar durch (Zahlen oben). Naechster Schritt int4-gepackt, "
                "sofern der Dequant-Anteil laut Messung nicht dominiert."
            )
        else:
            lines.append("")
            lines.append(
                "**int8-custom in Produktcode ueberfuehren: anhand der Zahlen oben "
                "entscheiden** — der erwartete Vorsprung ist nicht (oder nicht klar) "
                "eingetreten; erst Ursache klaeren (Copy- vs. Dequant-Anteil), dann int4."
            )
    else:
        lines.append("nicht messbar, weil fp16-partial-pin- und/oder int8-custom-Lauf fehlen.")
    lines.append("")

    REPORT.write_text("\n".join(lines))
    print(f"Bericht: {REPORT}")


if __name__ == "__main__":
    main()
