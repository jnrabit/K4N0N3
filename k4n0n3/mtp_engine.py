"""MTP (Multi-Token Prediction) Verification Engine — Phase 3.

Verifiziert Draft-Token aus den MTP/Draft-Head-Modulen gegen die echten
Logits des Modells und treibt einen greedy-korrekten Multi-Token-Decode-Loop.

Die Engine ist bewusst modell-unwissend: Sie arbeitet nur mit Tensoren und
einem ``forward_fn``, der die echten Logits plus die Draft-Logits liefert.
Die Kopplung an LayerManager/HF liegt im Aufrufer (ZeroFlushModel).
"""
from __future__ import annotations

import torch


def _head_out_features(head) -> int | None:
    if head is None:
        return None
    return getattr(head, "out_features", None)


class MTPVerificationEngine:
    """Greedy (top-1) Multi-Token-Prediction-Verifikation.

    ``temperature=0.0`` ist der deterministische Hauptpfad. Bei
    ``temperature > 0`` wird der Haupt-Token gesamplet, die Draft-Verifikation
    bleibt greedy — exaktes lossless Sampling (rejection) folgt separat.

    ``num_branches`` (> 1) aktiviert Multi-Branch-Tree-Drafting: pro Draft-
    Position werden Top-K Kandidaten extrahiert und K parallele Branches in
    einem einzigen Batch-Forward verifiziert.
    """

    def __init__(self, temperature: float = 0.0, num_branches: int = 1):
        self.temperature = temperature
        self.num_branches = max(1, int(num_branches))

    # -- pure helpers -------------------------------------------------------

    def extract_draft_logits(self, buffer: dict, head) -> list[torch.Tensor]:
        """Buffer (dict mtp_name -> list[output]) -> Draft-Logits je Modul.

        Nimmt pro MTP-Modul den letzten Output (aktueller Durchlauf) und die
        letzte Sequenzposition. Hat der Output bereits Vokabular-Breite
        (== head.out_features), wird er direkt als Logits verwendet, sonst
        wird der ``head`` (unembedding/lm_head) angewendet.

        Robust gegen mehrere Output-Formen (multi-head / projiziert):
          [batch, seq, vocab|hidden] -> letzte Position -> [batch, ...]
          [batch, vocab|hidden]       -> unveraendert
        """
        vocab = _head_out_features(head)
        logits: list[torch.Tensor] = []
        for outputs in buffer.values():
            if not outputs:
                continue
            h = outputs[-1]
            if h.dim() == 3:
                # [batch, seq, dim] -> letzte Sequenzposition [batch, dim]
                h = h[:, -1]
            elif h.dim() == 2:
                pass
            else:
                raise ValueError(
                    f"MTP-Output unerwartete Dimension {tuple(h.shape)}: "
                    f"erwartet [batch, seq, dim] oder [batch, dim]."
                )
            if head is not None and (vocab is None or h.shape[-1] != vocab):
                h = head(h)
                if h.dim() == 3:
                    h = h[:, -1]
            logits.append(h)
        return logits

    @staticmethod
    def greedy_tokens(logits: list[torch.Tensor]) -> list[int]:
        return [int(torch.argmax(l, dim=-1)[0]) for l in logits]

    @staticmethod
    def _pick(logits: torch.Tensor, temperature: float) -> int:
        if temperature == 0.0:
            return int(torch.argmax(logits, dim=-1)[0])
        probs = torch.softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probs, num_samples=1)[0])

    def verify_drafts(self, draft_tokens: list[int],
                      target_logits: list[torch.Tensor],
                      temperature: float | None = None) -> int:
        """Anzahl akzeptierter Drafts (greedy top-1).

        Draft ``i`` wird akzeptiert, wenn ``argmax(target_logits[i]) ==
        draft_tokens[i]``; Abbruch bei der ersten Abweichung. Rueckgabe 0
        bedeutet 1-Token-Fallback im Decode-Loop.
        """
        temp = self.temperature if temperature is None else temperature
        n = 0
        for i, draft in enumerate(draft_tokens):
            if i >= len(target_logits):
                break
            real = self._pick(target_logits[i], temp)
            if draft == real:
                n += 1
            else:
                break
        return n

    # -- multi-branch tree-drafting -----------------------------------------

    def topk_branches(self, draft_logits: list[torch.Tensor]) -> list[list[int]]:
        """Baut K parallele Branches (diagonal) aus den Draft-Logits.

        Pro Draft-Position die Top-K-Token; Branch ``i`` nimmt die i-te
        Top-K-Wahl je Position. Ergebnis: list[list[int]] mit K Branches,
        jeder Branch ein Pfad der Laenge ``len(draft_logits)``.
        """
        K = self.num_branches
        if not draft_logits:
            return []
        topk_per_pos: list[list[int]] = []
        for logit in draft_logits:
            flat = logit if logit.dim() == 1 else logit.reshape(-1)
            k = min(K, flat.shape[0])
            topk_per_pos.append(torch.topk(flat, k, dim=-1).indices.tolist())
        branches: list[list[int]] = []
        for i in range(K):
            branch = [pos[i] if i < len(pos) else pos[-1] for pos in topk_per_pos]
            branches.append(branch)
        return branches

    def _verify_tree(self, forward_fn, ids: torch.Tensor,
                     branches: list[list[int]]) -> tuple[int, list[int]]:
        """Verifiziert alle Branches parallel in einem Batch-Forward.

        ``ids`` ist der aktuelle Kontext [1, L] (inkl. Haupt-Token). Zeile i des
        Batch = [ids, branch_i] -> [K, L+k]. Die echten argmax der k
        Folge-Positionen werden vektorisiert mit den Branches verglichen; die
        gueltige Praefix-Laenge je Branch wird per ``cumprod`` bestimmt.
        Gibt ``(n_acc, prefix)`` des laengsten gueltigen Pfads zurueck.
        """
        K = len(branches)
        k = len(branches[0]) if branches else 0
        if K == 0 or k == 0:
            return 0, []
        L = ids.shape[1]
        branch_t = torch.tensor(branches, dtype=ids.dtype, device=ids.device)  # [K, k]
        rows = [torch.cat([ids, branch_t[i:i + 1]], dim=1) for i in range(K)]
        ids_batch = torch.cat(rows, dim=0)  # [K, L+k]
        real_logits, _ = forward_fn(ids_batch)  # [K, L+k, V]
        real_tokens = torch.argmax(real_logits, dim=-1)  # [K, L+k]
        predicted = real_tokens[:, L - 1:L - 1 + k]  # [K, k] echte argmax je Draft-Position
        match = branch_t == predicted  # [K, k]
        valid_len = torch.cumprod(match, dim=1).sum(dim=1)  # [K]
        best = int(torch.argmax(valid_len))
        n_acc = int(valid_len[best])
        return n_acc, branches[best][:n_acc]

    # -- decode loop --------------------------------------------------------

    def generate(self, forward_fn, input_ids: torch.Tensor, max_new_tokens: int,
                 eos_token_id: int | None = None,
                 temperature: float | None = None) -> list[int]:
        """Greedy-korrekter MTP-Decode-Loop.

        ``forward_fn(ids) -> (logits, draft_logits)`` mit ``ids`` [1, L] (bzw.
        [K, L] fuer die Batch-Verifikation), ``logits`` [*, L, V] und
        ``draft_logits`` eine Liste von [1, V].

        Bei ``num_branches == 1`` exakt der bisherige Single-Path; sonst
        Multi-Branch-Tree-Drafting. Liefert die neu generierten Token-IDs
        (ohne den Prompt).
        """
        if self.num_branches <= 1:
            return self._generate_single_path(forward_fn, input_ids,
                                              max_new_tokens, eos_token_id, temperature)
        return self._generate_tree(forward_fn, input_ids,
                                   max_new_tokens, eos_token_id, temperature)

    def _generate_single_path(self, forward_fn, input_ids, max_new_tokens,
                              eos_token_id, temperature) -> list[int]:
        """Single-Path-MTP (num_branches=1): sequentielle Draft-Verifikation."""
        temp = self.temperature if temperature is None else temperature
        ids = input_ids
        generated: list[int] = []
        steps = 0
        while len(generated) < max_new_tokens:
            steps += 1
            logits, draft_logits = forward_fn(ids)
            t0 = self._pick(logits[:, -1], temp)
            accepted = [t0]
            ids = torch.cat([ids, ids.new_tensor([[t0]])], dim=1)
            if t0 == eos_token_id:
                generated.append(t0)
                break
            drafts = self.greedy_tokens(draft_logits) if draft_logits else []
            for draft in drafts:
                if len(generated) + len(accepted) >= max_new_tokens:
                    break
                real_logits, _ = forward_fn(ids)
                real_tok = int(torch.argmax(real_logits[:, -1], dim=-1)[0])
                if draft == real_tok:
                    accepted.append(draft)
                    ids = torch.cat([ids, ids.new_tensor([[draft]])], dim=1)
                    if draft == eos_token_id:
                        break
                else:
                    break
            generated.extend(accepted)
            if eos_token_id is not None and eos_token_id in accepted:
                break
        self.last_run_stats = {
            "steps": steps,
            "accepted_tokens": len(generated),
            "accepted_per_step": (len(generated) / steps) if steps else 0.0,
        }
        return generated[:max_new_tokens]

    def _generate_tree(self, forward_fn, input_ids, max_new_tokens,
                       eos_token_id, temperature) -> list[int]:
        """Multi-Branch-Tree-Drafting: K Branches, ein Batch-Forward je Schritt."""
        temp = self.temperature if temperature is None else temperature
        ids = input_ids
        generated: list[int] = []
        steps = 0
        while len(generated) < max_new_tokens:
            steps += 1
            logits, draft_logits = forward_fn(ids)
            t0 = self._pick(logits[:, -1], temp)
            ids = torch.cat([ids, ids.new_tensor([[t0]])], dim=1)
            if t0 == eos_token_id:
                generated.append(t0)
                break
            accepted = [t0]
            if draft_logits:
                branches = self.topk_branches(draft_logits)
                n_acc, prefix = self._verify_tree(forward_fn, ids, branches)
                n_acc = min(n_acc, max_new_tokens - len(generated) - 1)
                if n_acc > 0:
                    prefix_acc = prefix[:n_acc]
                    if eos_token_id is not None and eos_token_id in prefix_acc:
                        stop = prefix_acc.index(eos_token_id)
                        prefix_acc = prefix_acc[:stop + 1]
                        n_acc = len(prefix_acc)
                    accepted.extend(prefix_acc)
                    ids = torch.cat([ids, ids.new_tensor([prefix_acc])], dim=1)
            generated.extend(accepted)
            if eos_token_id is not None and eos_token_id in accepted:
                break
        self.last_run_stats = {
            "steps": steps,
            "accepted_tokens": len(generated),
            "accepted_per_step": (len(generated) / steps) if steps else 0.0,
        }
        return generated[:max_new_tokens]
