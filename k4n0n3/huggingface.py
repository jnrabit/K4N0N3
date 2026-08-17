from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from .hooks import LayerManager
from .mtp_engine import MTPVerificationEngine
from .utils import auto_vram_budget, estimate_model_size


class ZeroFlushModel:
    """Wraps a Hugging Face model with transparent layer offloading."""

    _LAYER_PREFIX_MAP: dict[str, str] = {
        "llama": "model.layers",
        "mistral": "model.layers",
        "falcon": "transformer.h",
        "gpt2": "transformer.h",
        "gpt_neo": "transformer.h",
        "gpt_neox": "gpt_neox.layers",
        "gptj": "transformer.h",
        "opt": "model.decoder.layers",
        "bloom": "transformer.h",
        "gemma": "model.layers",
        "gemma2": "model.layers",
        "phi": "model.layers",
        "phi3": "model.layers",
        "qwen2": "model.layers",
        "mixtral": "model.layers",
        "stablelm": "model.layers",
        "distilbert": "distilbert.transformer.layer",
        "cohere": "model.layers",
        "dbrx": "transformer.blocks",
        "mpt": "transformer.blocks",
        "olmo": "model.transformer.blocks",
    }

    def __init__(
        self,
        model_name: str,
        vram_budget_mb: int | None = None,
        prefetch_depth: int = 1,
        device: str = "cuda",
        torch_dtype: torch.dtype | None = None,
        *,
        verbose: bool = False,
        pin_ram_fraction: float = 0.7,
        quantize_transfer: bool | str = False,  # False | True/"int8" | "int4"
        int4_group_size: int = 64,
        use_mtp: bool = False,
        pinned_layers: list[int | str] | None = None,
        mtp_num_branches: int = 1,
        **hf_kwargs: Any,
    ):
        self.model_name = model_name
        self.prefetch_depth = prefetch_depth
        self._device = device
        self.verbose = verbose
        self.use_mtp = use_mtp
        self.pinned_layers = pinned_layers
        self.mtp_num_branches = mtp_num_branches

        if torch_dtype is None and "cuda" in device:
            torch_dtype = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch_dtype,
                **hf_kwargs,
            )
            self.model.to("cpu")
        except (ValueError, KeyError, ImportError):
            self.model = AutoModel.from_pretrained(
                model_name,
                dtype=torch_dtype,
                **hf_kwargs,
            )
            self.model.to("cpu")

        self.model.eval()

        # MTP-Gewicht-Rekonstruktion: HF ignoriert mtp.*/model.layers.N+.
        # Hier die Draft-Heads aus dem Checkpoint lesen und anhaengen, BEVOR
        # der LayerManager die Discovery laufen laesst.
        if use_mtp:
            from .mtp_loader import reconstruct_and_attach_mtp
            reconstruct_and_attach_mtp(self.model, model_name, dtype=torch_dtype)

        model_size_mb = estimate_model_size(self.model)
        if vram_budget_mb is None:
            gpu_info = auto_vram_budget()
            if verbose:
                print(
                    f"[K4N0N3] Model size: {model_size_mb:.0f} MB | "
                    f"Auto VRAM budget: {gpu_info} MB"
                )
            vram_budget_mb = gpu_info
        elif verbose:
            print(
                f"[K4N0N3] Model size: {model_size_mb:.0f} MB | "
                f"VRAM budget: {vram_budget_mb} MB (manual)"
            )

        self.vram_budget_mb = vram_budget_mb

        prefix = self._guess_layer_prefix()
        self.layer_manager = LayerManager(
            self.model,
            layer_prefix=prefix,
            vram_budget_mb=vram_budget_mb,
            prefetch_depth=prefetch_depth,
            verbose=verbose,
            pin_ram_fraction=pin_ram_fraction,
            quantize_transfer=quantize_transfer,
            int4_group_size=int4_group_size,
            use_mtp=use_mtp,
            pinned_layers=pinned_layers,
        )

    def _guess_layer_prefix(self) -> str:
        model_type = getattr(self.model.config, "model_type", "")
        return self._LAYER_PREFIX_MAP.get(model_type, "model.layers")

    def prepare(self) -> None:
        self.layer_manager.prepare()

    def generate(self, prompt: str, *, speculative: bool = False,
                 draft_model_name: str = "Qwen/Qwen2.5-0.5B",
                 num_assistant_tokens: int | None = None,
                 **gen_kwargs: Any) -> str:
        if self.use_mtp:
            return self._generate_mtp(prompt, **gen_kwargs)
        if not hasattr(self.model, "generate"):
            raise AttributeError(
                f"Model '{self.model_name}' ({type(self.model).__name__}) "
                f"has no generate() method. Use forward() instead, "
                f"or load a causal LM (e.g. 'gpt2', 'meta-llama/Llama-*')."
            )
        if speculative:
            draft = self._get_draft(draft_model_name)
            if num_assistant_tokens is not None:
                # k fixieren, sonst passt HF ihn adaptiv an und die Messung
                # ueber k wird unsauber.
                draft.generation_config.num_assistant_tokens = num_assistant_tokens
                draft.generation_config.num_assistant_tokens_schedule = "constant"
            gen_kwargs["assistant_model"] = draft
        self._set_kv_cache_reserve(**gen_kwargs)
        self.prepare()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if "cuda" in self._device and torch.cuda.is_available():
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        # Nur die neu erzeugten Tokens — für den verlustfrei-Vergleich (V2)
        self._last_new_token_ids = output_ids[0][input_len:].tolist()
        result = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        self.layer_manager.memory.set_reserve(0)
        return result

    def _generate_mtp(self, prompt: str, **gen_kwargs: Any) -> str:
        """MTP-speculativer Decode-Loop (use_mtp=True).

        Verifiziert die im LayerManager gebufferten MTP-Drafts gegen die echten
        Logits. Greedy (temperature=0.0) ist verlustfrei und liefert exakt den
        Standard-Greedy-Output. Nutzt Recompute (kein KV-Cache), daher gibt es
        keinen geteilten Cache-Zustand zu korrumpieren.
        """
        self.prepare()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if "cuda" in self._device and torch.cuda.is_available():
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        head = self._resolve_lm_head()
        engine = MTPVerificationEngine(num_branches=self.mtp_num_branches)
        self._mtp_engine = engine
        max_new_tokens = gen_kwargs.get("max_new_tokens", gen_kwargs.get("max_length", 2048))
        eos = getattr(self.tokenizer, "eos_token_id", None)
        temperature = float(gen_kwargs.get("temperature", 0.0))

        def forward_fn(ids: torch.Tensor):
            with torch.no_grad():
                logits = self.model(input_ids=ids).logits
            draft_logits = engine.extract_draft_logits(
                self.layer_manager.get_mtp_buffer(), head)
            self.layer_manager.clear_mtp_buffer()
            return logits, draft_logits

        with torch.no_grad():
            new_ids = engine.generate(forward_fn, input_ids, max_new_tokens,
                                      eos_token_id=eos, temperature=temperature)
        self.layer_manager.memory.set_reserve(0)

        self._last_new_token_ids = new_ids
        self._mtp_stats = engine.last_run_stats
        new_tensor = torch.tensor([new_ids], dtype=input_ids.dtype, device=input_ids.device)
        full_ids = torch.cat([input_ids, new_tensor], dim=1)
        return self.tokenizer.decode(full_ids[0], skip_special_tokens=True)

    def _resolve_lm_head(self) -> torch.nn.Module:
        """Findet den unembedding/lm_head des Modells (fuer MTP-Draft-Logits)."""
        for attr in ("lm_head", "embed_out", "head"):
            m = getattr(self.model, attr, None)
            if isinstance(m, torch.nn.Module):
                return m
        raise AttributeError(
            f"use_mtp=True braucht einen unembedding/lm_head am Modell "
            f"'{self.model_name}'. Keines der Attribute lm_head/embed_out/head "
            f"gefunden."
        )

    def _get_draft(self, name: str):
        """Lazy-lädt das Draft-Modell für spekulatives Decoding: VOLL GPU-
        resident (fp16), gecacht. Hartes Kriterium: identisches Vokabular wie
        das Ziel (Voraussetzung von HF assisted generation)."""
        if getattr(self, "_draft", None) is not None and self._draft_name == name:
            return self._draft
        draft = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16)
        draft.to("cuda").eval()
        tgt_vocab = getattr(self.model.config, "vocab_size", None)
        dft_vocab = getattr(draft.config, "vocab_size", None)
        if tgt_vocab != dft_vocab:
            raise ValueError(
                f"Draft '{name}' Vokabular {dft_vocab} != Ziel {tgt_vocab}. "
                "Assisted generation braucht identisches Vokabular — anderes "
                "Draft-Modell derselben Familie wählen.")
        self._draft = draft
        self._draft_name = name
        self._draft_reserve_bytes = sum(
            p.numel() * p.element_size() for p in draft.parameters())
        if self.verbose:
            print(f"[K4N0N3] Draft {name} resident: "
                  f"{self._draft_reserve_bytes / 1024**2:.0f} MB")
        return self._draft

    def _set_kv_cache_reserve(self, **gen_kwargs: Any) -> None:
        """Estimate KV-cache upper bound and reserve that VRAM budget.

        Ein resident laufendes Draft-Modell (spekulativ) belegt zusätzlich VRAM,
        das der MemoryManager nicht als Layer sieht — als fixe Reserve mitbuchen.
        """
        draft_reserve = getattr(self, "_draft_reserve_bytes", 0)
        cfg = self.model.config
        try:
            n_layers = getattr(cfg, "num_hidden_layers", 0) or 0
            n_kv_heads = getattr(cfg, "num_key_value_heads", 0) or getattr(cfg, "num_attention_heads", 0)
            head_dim = getattr(cfg, "hidden_size", 0) // max(getattr(cfg, "num_attention_heads", 1), 1)
            max_len = gen_kwargs.get("max_length", gen_kwargs.get("max_new_tokens", 2048))
            dtype_size = 2  # float16
            kv_bytes = n_layers * 2 * n_kv_heads * head_dim * max_len * dtype_size
            reserve = kv_bytes + draft_reserve
            if reserve > 0:
                self.layer_manager.memory.set_reserve(reserve)
                if self.verbose:
                    print(f"[K4N0N3] Reserve: KV {kv_bytes / 1024**2:.0f} MB "
                          f"+ Draft {draft_reserve / 1024**2:.0f} MB")
        except Exception:
            # Fallback: 10% of budget + Draft
            reserve = int(self.vram_budget_mb * 1024 * 1024 * 0.10) + draft_reserve
            self.layer_manager.memory.set_reserve(reserve)
            if self.verbose:
                print(f"[K4N0N3] Reserve (fallback): {reserve / 1024**2:.0f} MB")

    def forward(self, prompt: str) -> torch.Tensor:
        self.prepare()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if "cuda" in self._device and torch.cuda.is_available():
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs

    def offload_all(self) -> None:
        self.layer_manager.offload_all()

    def get_mtp_buffer(self):
        return self.layer_manager.get_mtp_buffer()

    def clear_mtp_buffer(self) -> None:
        self.layer_manager.clear_mtp_buffer()

    def report(self) -> str:
        return self.layer_manager.memory.report()
