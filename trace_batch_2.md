# Trace-Charge 2 — Rewrite-Traces sammeln

**So läuft es:** `collect-repl` (braucht Verlauf im selben Lauf!). Erst die
Basisfrage stellen, Antwort abwarten, **dann** die Folgefrage als zweiten Turn.
Nur so entsteht ein Rewrite-Trace — beides zusammen in einen Prompt geht nicht.

Der Trace entsteht unabhängig davon, ob die Antwort inhaltlich gut ist. Du musst
nichts bewerten, nur fragen. Danach:

```
collect-traces stats
collect-traces curate          # nur die Urteils-Fälle kommen dir vor
collect-traces build --eval-n 10
```

Ziel: ~36 neue Paare → zusammen mit den 22 vorhandenen landen wir bei ~58.

Charge 1 war fast nur Infra/ML und fast nur „und wie/was …?"-Pronomen. Diese
Charge streut die **Referenztypen** absichtlich — Pronomen sind zu leicht, das
Modell lernt sonst nur ein Muster.

---

## A · Pronomen-Referenz (das / es / die / dem) — 8

1. `Was macht ein Merkle-Tree?` → `und wo wird das eingesetzt?`
2. `Wie funktioniert ein B-Tree-Index in PostgreSQL?` → `wann bringt der nichts?`
3. `Wofür ist ein JWT gedacht?` → `wie lange sollte es gültig sein?`
4. `Was besagt das CAP-Theorem?` → `welchen Teil davon geben Datenbanken meist auf?`
5. `Wie arbeitet der Garbage Collector in Java?` → `wann pausiert er die Anwendung?`
6. `Was macht der Borrow-Checker in Rust?` → `warum nervt er am Anfang so?`
7. `Wie funktioniert DNS-Auflösung?` → `wie lange wird die zwischengespeichert?`
8. `Was ist ein CRDT?` → `wie lösen die Konflikte auf?`

## B · Ellipse ohne Pronomen — 7

Die schwersten Fälle: es gibt kein Wort, das zurückzeigt — das Thema muss aus
dem Verlauf ergänzt werden.

9. `Wie funktioniert Kafka-Partitionierung?` → `und bei nur einem Consumer?`
10. `Was bringt mmap beim Lesen großer Dateien?` → `und bei Schreibzugriffen?`
11. `Wie arbeitet der Raft-Konsens-Algorithmus?` → `und wenn der Leader ausfällt?`
12. `Wozu dient Backpressure in Datenströmen?` → `und ohne?`
13. `Was macht BPE bei der Tokenisierung?` → `und bei Umlauten?`
14. `Wie funktioniert Chunking beim RAG?` → `und bei Tabellen?`
15. `Was macht Dropout beim Training?` → `und zur Inferenzzeit?`

## C · Vergleich / Alternative — 6

16. `Wie funktioniert HNSW als Vektorindex?` → `und wie schneidet IVF-Flat dagegen ab?`
17. `Was macht Adam als Optimierer besser als SGD?` → `und bei kleinen Batches?`
18. `Wie arbeitet btrfs mit Copy-on-Write?` → `wie macht ext4 das stattdessen?`
19. `Was ist Beam-Search bei der Textgenerierung?` → `worin unterscheidet es sich von Greedy?`
20. `Wie funktioniert OAuth2 mit Authorization Code Flow?` → `und der implizite Flow?`
21. `Was macht FlashAttention schneller?` → `im Vergleich zur normalen Attention?`

## D · Auswahl / Rückbezug auf eine Aufzählung — 5

Hier muss der Rewriter erkennen, worauf sich „davon" / „der zweite" bezieht.

22. `Welche Isolationsstufen kennt SQL?` → `welche davon verhindert Phantom-Reads?`
23. `Welche Docker-Storage-Driver gibt es?` → `welcher davon ist heute Standard?`
24. `Welche Kompressionsverfahren nutzt HTTP?` → `welches spart am meisten?`
25. `Welche Speculative-Decoding-Varianten gibt es?` → `welche braucht kein zweites Modell?`
26. `Welche systemd-Unit-Typen gibt es?` → `welcher startet beim Booten?`

## E · Bereits eigenständige Folgefrage → soll UNCHANGED liefern — 5

Wichtig: bisher sind **alle** UNCHANGED-Beispiele synthetisch. Echte gehören
dazu, sonst lernt das Modell UNCHANGED nur am künstlichen Satzbau.

27. `Was ist ein Bloom-Filter?` → `Wie funktioniert die Huffman-Kodierung?`
28. `Wie arbeitet SIMD auf modernen CPUs?` → `Was ist ein Speicherbarriere-Befehl?`
29. `Was macht ONNX als Austauschformat?` → `Wie unterscheidet sich Protobuf von JSON?`
30. `Wie funktioniert ein Reranker im Retrieval?` → `Was ist Batch-Normalisierung?`
31. `Was ist Idempotenz bei einer HTTP-API?` → `Wie funktioniert ein Rebase in Git?`

## F · Englisch — Sprache muss erhalten bleiben — 5

Charge 1 hatte nur zwei englische, beide zu „Apache Spark".

32. `How does copy-on-write forking work?` → `and what about shared memory?`
33. `What is the Python GIL?` → `why does it hurt CPU-bound code?`
34. `How does a WebSocket handshake work?` → `and how is it kept alive?`
35. `What does a regex backtracking engine do?` → `when does it blow up?`
36. `How does gradient accumulation work?` → `and how does it affect the effective batch size?`

---

## Woran ich beim Kurieren messe

Rubrik steht in `collect2/docs/traces_curation.md`. Die mechanischen Regeln
(Sie-Register, Konjunktions-Anfang, zu kurz, Passthrough) sieben vorab; dir
kommen nur die semantischen Grenzfälle vor. Die zwei Urteilsfragen dabei:

- **Antezedent erhalten?** Wurde aus „und bei Schreibzugriffen?" wirklich
  „Wie verhält sich mmap bei Schreibzugriffen?" — oder eine themenlose
  Allgemeinfrage?
- **Begriff sauber?** Keine zerhackten Terme (Kalibrierbeispiel aus Charge 1:
  „Lorenz-**Akt attractor**").

Bei Gruppe E ist die Erwartung anders: dort ist `UNCHANGED` die **richtige**
Antwort. Formt der Rewriter die schon eigenständige Frage trotzdem um, ist das
ein Ablehnungsgrund — genau der Fehler, gegen den die Negativbeispiele zielen.
