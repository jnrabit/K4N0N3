# K4N0N3 — Arbeitsauftrag 5: 9B-Mechanik-Stresstest (LoRA auf Qwythos-9B)

## Voraussetzung — NICHT vorher starten

RAM-Upgrade eingebaut und verifiziert: `free -h` zeigt ~31 GB total,
`available_ram_mb()` > 24000 im Leerlauf. Wenn nicht: STOPP.

## Kontext

Der Q3-Beweislauf (Auftrag 4) hat LoRA-Training mit K4N0N3 auf Qwen2.5-3B
demonstriert (50 Schritte, Loss 3,6 → 1,6 Median, VRAM-Peak 1473 MB,
2,1 s/Schritt, int8-Transfer, grad_checkpointing). Dieser Auftrag skaliert
den Beweis auf das reale Zielmodell: **Qwythos-9B** (Qwen3.5-9B-Basis),
das später auf collect2-Traces finegetunt werden soll (Traces entstehen in
einem separaten collect2-Auftrag — hier noch Dummy-Daten).

**Das ist ein Mechanik-Stresstest, kein Qualitäts-Finetune.** Bewiesen werden
soll: die K4N0N3-Pipeline (int8-Master, Pinning, Forward/Backward-Streaming,
LoRA, Checkpointing) funktioniert bei 9B auf RX 7600 (8 GB) + 31 GB RAM.
Loss-Werte sind Mechanik-Indikator (fällt sie?), nicht Modellbewertung.

Bekannte Risiken vorab (im Plan adressieren, nicht erst beim Crash):

1. **Architektur-Support:** Qwen3.5 ist neu — prüfen, ob die installierte
   transformers-Version das model_type kennt. Falls Upgrade nötig: zuerst
   in einem venv testen, dass der bestehende K4N0N3-Teststack damit grün
   bleibt (Qwen2.5-0.5B-Smoke), erst dann global. Falls die Architektur
   auch mit aktuellem transformers nicht ladbar ist: STOPP, Befund
   dokumentieren — das wäre ein Blocker, den kein Code-Umweg lösen soll.
2. **Layer-Prefix:** vermutlich weiterhin `model.layers`; Discovery-Fallback
   existiert. Im Log verifizieren, Layer-Anzahl und Layer-Größe berichten.
3. **YaRN/RoPE:** Qwythos shipped mit YaRN-Scaling für 1M-Kontext. Für
   Training mit seq_len ≤ 2048 ist das irrelevant, aber die Config nicht
   anfassen und nicht "optimieren" — laden wie sie ist, seq_len klein.
4. **RAM-Rechnung:** int8-Master ~9 GB + fp16-Reste (Embeddings/Norms/Head,
   bei 9B nicht klein — beziffern!) + gepinnte Scales + Prozess-Overhead.
   Erwartung: passt in 31 GB mit Luft; `available_ram_mb()`-Verlauf
   vor/nach Master-Aufbau ins JSON.

## S1 — Modell laden + Master-Aufbau

- HF-Download `empero-ai/Qwythos-9B-Claude-Mythos-5-1M` (Achtung ~18 GB
  Download — Plattenplatz prüfen).
- `ZeroFlushModel(..., quantize_transfer=True)` mit int8: Master-Aufbau,
  Zwei-Pass (Auftrag-O-Mechanik), Pinning-Zusammenfassung ins Log.
- Messpunkte (Harness erweitern wo nötig): master_ram_mb, pinned_layers
  (Erwartung: alle), Aufbauzeit, RAM-Verlauf.
- Smoke: ein Forward (seq 256) + Mechanik-Greedy-Check (mit vs. ohne
  Offloading, 16 Tokens, identisch — auf 9B teuer, aber einmal Pflicht).

## S2 — LoRA-Trainingslauf (analog Q3, skaliert)

- Setup wie Q3: frozen Basis (requires_grad-Guard greift), LoRA r=8 auf
  q_proj/v_proj, AdamW nur Adapter, Batch 1, grad_checkpointing an.
- **seq_len:** den Wert aus dem Trace-Validator-Report nehmen (p90 der
  collect2-Trace-Längen), falls der Report schon existiert; sonst 1024
  als Default und im Bericht vermerken, dass die echte Länge nachzuziehen
  ist.
- Dummy-Daten: bench/data aus Q3 wiederverwenden (Inhalt egal, Seed fest).
- 50 Schritte. Messpunkte: loss_curve, step_time_ms (Median + p90),
  vram_peak_mb gegen Budget, RAM-Verlauf während Training (Swap-Wächter:
  wenn Swap-Nutzung > 500 MB auftritt, abbrechen und dokumentieren statt
  stundenlang thrashen).
- Erwartung step_time: Q3 hatte 2,1 s bei 3B/seq256. 9B skaliert Transfer
  ×3, seq_len ×4 skaliert Compute+Aktivierungen — grobe Erwartung
  10–25 s/Schritt. Deutlich darüber → Timing-Split (Transfer vs. Compute
  vs. Checkpointing-Recompute) statt raten.

## S3 — Abbruchkriterien & Verdikt

Erfolg = die vier Q3-Kriterien auf 9B: (1) 50 Schritte ohne OOM/Crash,
(2) Loss fällt (Median letzte 10 < erste 10), (3) VRAM-Peak ≤ Budget +
Reserve, (4) Adapter verändern generate()-Output nachweisbar
(greedy_tokens an/aus im JSON).

Teilerfolg ist auch ein Ergebnis: wenn z. B. der Master passt, aber
Checkpointing × 40+ Layer einen neuen Hook-Konflikt zeigt → minimales
Repro unter spike/, Befund in den Bericht, STOPP. Kein Drumherum-Patchen.

## S4 — Bericht

Via make_report.py: S1-Kennzahlen, Trainings-Kennzahlen, Vergleichszeile
zu Q3 (3B), Verdikt in einem Satz: "9B-Finetune auf dieser Hardware
machbar: ja/nein, Kosten X s/Schritt" — das ist die Zahl, die entscheidet,
wie groß der echte Trace-Finetune dimensioniert werden kann
(Schritte × Sekunden = Nachtbudget?).

## Allgemeine Regeln

Wie gehabt: keine Zahl ohne JSON, Commits thematisch, keine neuen
Pflicht-Dependencies, bei verfehlten Kriterien stoppen und dokumentieren.
