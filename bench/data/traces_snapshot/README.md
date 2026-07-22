# Trace-Snapshot (2026-07-21)

Eingefrorene Kopie des Trainingskorpus aus `~/collect2/data/traces`, damit die
Zahlen in `UMBAU5_BERICHT.md` **aus diesem Repo allein** reproduzierbar sind.
Das Original liegt unter `collect2/data/`, und das ist dort gitignored.

Stand: 136 eindeutige Rewrite-Traces, 71 kuratierte Positive, Trainingssatz 56.

## Inhalt

- `2026-07-*.jsonl` — rohe Traces, append-only, eine Datei pro Tag
- `migrated.jsonl` — Alt-Traces aus dem flachen Vorgaengerformat
- `curated/curation.jsonl` — die Kurationsverdikte (gut/schlecht + Begruendung).
  **Das ist die eigentliche Handarbeit**; die Traces selbst sind
  nachproduzierbar, diese Entscheidungen nicht.
- `curated/training_set.jsonl`, `curated/eval_set.jsonl` — daraus gebaut

Roh-Zeilen > eindeutige Traces: die Tagesdateien enthalten Wiederholungen,
`load_all()` dedupliziert ueber `trace_id`.

## Trainings-/Eval-Satz reproduzieren

Verifiziert: aus diesem Snapshot baut `collect-traces build` denselben
Trainings- und Eval-Satz bit-identisch nach.

```bash
cp -r bench/data/traces_snapshot/* ~/collect2/data/traces/   # oder COLLECT_TRACES_DIR setzen
cd ~/collect2 && .venv/bin/python -m collect.traces.cli build --eval-n 15 --negative-ratio 0
```

`--negative-ratio 0` ist Absicht: synthetische UNCHANGED-Negative haben den
Finetune messbar verschlechtert.

## Snapshot aktualisieren

Kein Symlink, keine Automatik — bewusst ein manueller Schnappschuss, damit ein
Lauf gegen einen eingefrorenen Stand laeuft. Nach neuen Chargen:

```bash
rm bench/data/traces_snapshot/validation_report.json 2>/dev/null
cp -r ~/collect2/data/traces/* bench/data/traces_snapshot/
rm bench/data/traces_snapshot/validation_report.json   # generiertes Artefakt, nicht committen
```
