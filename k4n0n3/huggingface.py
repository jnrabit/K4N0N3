from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from .hooks import LayerManager
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
        int4_group_size: int = 128,
        **hf_kwargs: Any,
    ):
        self.model_name = model_name
        self.prefetch_depth = prefetch_depth
        self._device = device
        self.verbose = verbose

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
        )

    def _guess_layer_prefix(self) -> str:
        model_type = getattr(self.model.config, "model_type", "")
        return self._LAYER_PREFIX_MAP.get(model_type, "model.layers")

    def prepare(self) -> None:
        self.layer_manager.prepare()

    def generate(self, prompt: str, **gen_kwargs: Any) -> str:
        if not hasattr(self.model, "generate"):
            raise AttributeError(
                f"Model '{self.model_name}' ({type(self.model).__name__}) "
                f"has no generate() method. Use forward() instead, "
                f"or load a causal LM (e.g. 'gpt2', 'meta-llama/Llama-*')."
            )
        self._set_kv_cache_reserve(**gen_kwargs)
        self.prepare()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if "cuda" in self._device and torch.cuda.is_available():
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        result = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        self.layer_manager.memory.set_reserve(0)
        return result

    def _set_kv_cache_reserve(self, **gen_kwargs: Any) -> None:
        """Estimate KV-cache upper bound and reserve that VRAM budget."""
        cfg = self.model.config
        try:
            n_layers = getattr(cfg, "num_hidden_layers", 0) or 0
            n_kv_heads = getattr(cfg, "num_key_value_heads", 0) or getattr(cfg, "num_attention_heads", 0)
            head_dim = getattr(cfg, "hidden_size", 0) // max(getattr(cfg, "num_attention_heads", 1), 1)
            max_len = gen_kwargs.get("max_length", gen_kwargs.get("max_new_tokens", 2048))
            dtype_size = 2  # float16
            kv_bytes = n_layers * 2 * n_kv_heads * head_dim * max_len * dtype_size
            if kv_bytes > 0:
                self.layer_manager.memory.set_reserve(kv_bytes)
                if self.verbose:
                    print(f"[K4N0N3] KV-cache reserve: {kv_bytes / 1024**2:.0f} MB")
        except Exception:
            # Fallback: 10% of budget
            reserve = int(self.vram_budget_mb * 1024 * 1024 * 0.10)
            self.layer_manager.memory.set_reserve(reserve)
            if self.verbose:
                print(f"[K4N0N3] KV-cache reserve (fallback): {reserve / 1024**2:.0f} MB")

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

    def report(self) -> str:
        return self.layer_manager.memory.report()
