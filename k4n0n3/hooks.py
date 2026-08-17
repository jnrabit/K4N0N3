"""LayerManager with pinned-memory master copies — upload=copy, offload=drop."""
from __future__ import annotations

import inspect
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import torch

from .memory import MemoryManager


class LayerState(Enum):
    ON_CPU = auto()
    PREFETCHING = auto()
    ON_GPU = auto()


@dataclass
class LayerInfo:
    state: LayerState = LayerState.ON_CPU
    size_mb: float = 0.0
    transfer_time_ms: float = 0.0
    compute_time_ms: float = 0.0


# -- M1: custom weight-only int8 (symmetrisch, per-Output-Channel) -----------


def quantize_per_channel_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """scale[c] = max(|W[c,:]|) / 127 (fp16-Vektor), W_int8[c,:] = round(W[c,:]/scale[c])."""
    wf = w.detach().float()
    scale = wf.abs().amax(dim=1).div_(127.0).clamp_(min=2**-24)
    q = torch.round(wf / scale.unsqueeze(1)).clamp_(-127, 127).to(torch.int8)
    # dtype des Gewichts erhalten: Qwen3.5 kommt als bf16, und ein fp16-Scale
    # macht die dequantisierten Gewichte inkompatibel zum Rest des Modells.
    return q, scale.to(w.dtype)


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(scale.dtype) * scale.unsqueeze(1)


# -- P: int4 group-wise gepackt (ersetzt den M5-per-Channel-Stand) -----------


def quantize_groupwise_int4(
    w: torch.Tensor, group_size: int = 64
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Symmetrisch, group-wise entlang der Input-Dimension (P1).

    scale-Shape [out, ceil(in/group_size)] als fp16; q in [-7, 7], Spalten
    2j/2j+1 teilen sich ein Byte. Ist in_features nicht durch group_size
    teilbar, ist die LETZTE Gruppe kuerzer — bewusst Restgruppe statt Padding,
    damit die q4-Packung exakt in/2 Bytes bleibt und kein Padding-Wert die
    Scale der Randgruppe verzerrt. Gerade Spaltenzahl vorausgesetzt
    (Aufrufer prueft und faellt sonst auf int8 zurueck).
    """
    out_f, in_f = w.shape
    wf = w.detach().float()
    n_groups = (in_f + group_size - 1) // group_size
    scale = torch.empty(out_f, n_groups, dtype=torch.float32)
    q = torch.empty(out_f, in_f, dtype=torch.int16)
    for g in range(n_groups):
        lo, hi = g * group_size, min((g + 1) * group_size, in_f)
        blk = wf[:, lo:hi]
        sc = blk.abs().amax(dim=1).div_(7.0).clamp_(min=2**-24)
        scale[:, g] = sc
        q[:, lo:hi] = torch.round(blk / sc.unsqueeze(1)).clamp_(-7, 7).to(torch.int16)
    nib = q & 0xF  # Zweierkomplement-Nibble
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).to(torch.uint8)
    meta = {"group_size": group_size, "orig_shape": (out_f, in_f)}
    return packed, scale.to(w.dtype), meta


def dequantize_groupwise_int4(
    packed: torch.Tensor, scale: torch.Tensor, meta: dict
) -> torch.Tensor:
    """Unpack + Dequant in einem Schritt.

    Unpack bewusst ohne torch.where/int16-Zwischentensoren: Sign-Extend eines
    Nibbles v ist (v ^ 8) - 8, direkt in uint8→int8 — der Timing-Split (O3)
    hat gezeigt, dass die naive where/stack-Variante den halbierten Transfer
    wieder auffrisst (12.7 ms Dequant vs. 2.9 ms bei int8).
    """
    group_size = meta["group_size"]
    out_f, in_f = meta["orig_shape"]
    q = torch.empty(out_f, in_f, dtype=torch.int8, device=packed.device)
    q[:, 0::2] = ((packed & 0x0F) ^ 8).view(torch.int8) - 8
    q[:, 1::2] = ((packed >> 4) ^ 8).view(torch.int8) - 8
    w = q.to(scale.dtype)  # Scale traegt den dtype des Originalgewichts
    n_groups = scale.shape[1]
    if in_f == n_groups * group_size:
        # Broadcast ueber Gruppen-View statt eines materialisierten
        # [out, in]-Scale-Tensors
        w.view(out_f, n_groups, group_size).mul_(scale.unsqueeze(2))
        return w
    # Restgruppe kuerzer: Scale explizit expandieren
    reps = torch.full((n_groups,), group_size, dtype=torch.long, device=scale.device)
    reps[-1] = in_f - group_size * (n_groups - 1)
    return w.mul_(scale.repeat_interleave(reps, dim=1))


class LayerManager:
    """Per-layer device placement via forward hooks with async prefetch.

    Weights live in pinned CPU memory as a canonical master copy.
    "Upload"  = copy from pinned master → GPU (async).
    "Offload" = drop GPU tensor, point .data back to pinned master (no D2H).

    quantize_transfer=True (M): Linear-Weights innerhalb der Layer liegen als
    int8-Master ({"q", "scale"}) im RAM; der Upload kopiert int8 + scale und
    dequantisiert auf der GPU zu fp16. Spart PCIe-Transfer (halbiert), NICHT
    GPU-VRAM — der MemoryManager bucht weiterhin die fp16-GPU-Groesse.
    """

    #: Q1: Wenn True (TrainingManager), bekommen trainierbare Params
    #: (requires_grad=True, d. h. Adapter) keinen Master-Eintrag — sie werden
    #: nie gedroppt/hochgeladen und bleiben permanent auf der GPU.
    _skip_trainable: bool = False

    #: Namensmuster fuer MTP/Draft-Head-Module (Qwen3, DeepSeek-V3 MTP,
    #: blk.*.nextn). Case-insensitiv auf den vollen Modulpfad gematcht.
    _MTP_NAME_PATTERNS: tuple[str, ...] = ("mtp", "draft_head", "nextn", "next_n")

    def __init__(
        self,
        model: torch.nn.Module,
        layer_prefix: str = "model.layers",
        vram_budget_mb: int = 4096,
        prefetch_depth: int = 1,
        *,
        verbose: bool = False,
        pin_ram_fraction: float = 0.7,
        quantize_transfer: bool | str = False,
        int4_group_size: int = 64,
        use_mtp: bool = False,
        pinned_layers: list[int | str] | None = None,
    ):
        self.model = model
        self._int4_group_size = int4_group_size
        self.prefetch_depth = max(1, prefetch_depth)
        self.verbose = verbose
        self.use_mtp = use_mtp
        self.memory = MemoryManager(vram_budget_mb)

        self._layers: OrderedDict[str, torch.nn.Module] = OrderedDict()
        self._layer_list: list[str] = []
        self._layer_info: dict[str, LayerInfo] = {}
        # MTP/Draft-Head-Module — Phase 1 nur gemappt, Phase 2 im Dual-Pass ausgefuehrt.
        self._mtp_layers: OrderedDict[str, torch.nn.Module] = OrderedDict()
        self._mtp_layer_list: list[str] = []
        # Zuordnung Standard-Layer -> zugehoerige MTP-Module (Namen).
        self._layer_to_mtp: dict[str, list[str]] = {}
        # Draft-Tiefe pro MTP-Modul (1-basiert, fuer position_ids-Shift).
        self._mtp_depth: dict[str, int] = {}
        # Buffer fuer MTP/Draft-Outputs: mtp_name -> Liste der Outputs pro Durchlauf.
        self._mtp_buffer: dict[str, list[torch.Tensor]] = {}
        # Forward-Kontext (input_ids/attention_mask/position_ids) aus dem
        # Wurzel-Pre-Hook, damit _run_mtp_pass reale MTP-Signaturen bedienen kann.
        self._forward_ctx: dict[str, torch.Tensor] = {}
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._root_hook_handle: torch.utils.hooks.RemovableHandle | None = None
        self._prepared = False

        self._cuda_available = torch.cuda.is_available()
        # False → aus; True/"int8" → int8-Master; "int4" → int4-gepackt (M5)
        if quantize_transfer is True:
            quantize_transfer = "int8"
        if quantize_transfer not in (False, "int8", "int4"):
            raise ValueError(f"quantize_transfer: erwartet False/True/'int8'/'int4', "
                             f"bekommen: {quantize_transfer!r}")
        self._quantize_transfer = quantize_transfer
        if quantize_transfer and not self._cuda_available:
            raise ValueError(
                "quantize_transfer=True requires CUDA/ROCm: die Layer-Weights "
                "liegen dann nur als int8-Master vor und ein CPU-Forward wuerde "
                "auf int8-Daten rechnen (stiller Muell-Output). Ohne GPU den "
                "Default quantize_transfer=False verwenden."
            )
        self._prefetch_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream() if self._cuda_available else None
        )
        self._prefetch_events: dict[str, torch.cuda.Event] = {}

        # Master copies: layer_name -> {param_name: pinned_tensor | {"q","scale"}}
        self._cpu_master: dict[str, dict] = {}
        # Fast param lookup: layer_name -> {param_name: nn.Parameter}
        self._param_refs: dict[str, dict[str, torch.nn.Parameter]] = {}
        # Per-layer pin status
        self._pinned: dict[str, bool] = {}
        self._pin_ram_fraction = max(0.0, min(1.0, pin_ram_fraction))
        # fp16-GPU-Groesse pro Layer (vor Quantisierung gemessen) — Basis der
        # VRAM-Buchung, auch wenn der Transfer int8 ist
        self._layer_gpu_bytes: dict[str, int] = {}

        self._discover_layers(layer_prefix)
        self.pinned_layer_indices = self._normalize_pinned_layers(pinned_layers)
        self._pinned_names = {self._layer_list[i] for i in self.pinned_layer_indices}
        self._measure_layer_sizes()
        if self._cuda_available:
            self._build_master_copies()
        self._register_hooks()

    # -- discovery ----------------------------------------------------------

    def _discover_layers(self, prefix: str) -> None:
        module = _resolve_module(self.model, prefix)
        if module is None:
            modules = dict(self.model.named_modules())
            prefix = self._guess_layer_prefix(modules) or prefix
            module = _resolve_module(self.model, prefix)
        if module is None:
            raise ValueError(
                f"Could not find transformer layers at '{prefix}'. "
                f"Specify layer_prefix (e.g. 'model.layers' for Llama, "
                f"'transformer.h' for GPT-2)."
            )
        for idx, child in enumerate(module.children()):
            name = f"{prefix}.{idx}"
            self._layers[name] = child
            self._layer_list.append(name)
            self._layer_info[name] = LayerInfo()
        if self.verbose:
            print(f"[K4N0N3] Discovered {len(self._layers)} layers at '{prefix}'")
        if self.use_mtp:
            self._discover_mtp_layers()
            self._associate_mtp_layers()

    def _discover_mtp_layers(self) -> None:
        """MTP/Draft-Head-Module namensbasiert erkennen und separat mappen.

        Phase 1 registriert sie nur im Mapping (keine Hooks, kein Prefetch/
        Offload). Gedeckt sind drei Faelle:
          - Leaf-Module mit MTP-Namen und eigenen Tensoren
            (model.layers.0.mtp, model.blk.0.nextn, model.draft_heads.0),
          - Container (ModuleList/ModuleDict) mit MTP-Namen → direkte Kinder,
          - komplexe MTP-Module ohne eigene Tensoren (z. B. model.nextn mit
            Submodulen enorm/hnorm/...) → als Ganzes.
        Submodule eines bereits registrierten MTP-Moduls werden uebersprungen,
        damit die internen Linear-/Norm-Layer nicht als eigene Draft-Heads
        auftauchen. Standard-Layer bleiben unberuehrt.
        """
        skip: set[str] = set()
        for name, mod in self.model.named_modules():
            if any(name == p or name.startswith(p + ".") for p in skip):
                continue
            if not self._is_mtp_module(name):
                continue
            if _has_own_tensors(mod):
                self._add_mtp_layer(name, mod)
            elif isinstance(mod, (torch.nn.ModuleList, torch.nn.ModuleDict)):
                for child_name, child in mod.named_children():
                    full = f"{name}.{child_name}" if name else child_name
                    self._add_mtp_layer(full, child)
                    skip.add(full)
                skip.add(name)
            else:
                # Komplexes MTP-Modul (eigene Struktur, keine Blatt-Tensoren).
                self._add_mtp_layer(name, mod)
                skip.add(name)
        if self.verbose:
            print(f"[K4N0N3] Discovered {len(self._mtp_layers)} MTP/draft module(s)")

    @classmethod
    def _is_mtp_module(cls, name: str) -> bool:
        low = name.lower()
        return any(pattern in low for pattern in cls._MTP_NAME_PATTERNS)

    def _add_mtp_layer(self, name: str, mod: torch.nn.Module) -> None:
        if name in self._mtp_layers:
            return
        self._mtp_layers[name] = mod
        self._mtp_layer_list.append(name)

    def _normalize_pinned_layers(self, pinned: list[int | str] | None) -> set[int]:
        """Normalisiert pinned_layers zu einem Set gueltiger Layer-Indizes.

        int -> Index (negativ: n + i). str -> reine Ziffer als Index, sonst
        exakter _layer_list-Name, sonst Suffix-Match (letztes Segment).
        """
        if not pinned:
            return set()
        n = len(self._layer_list)
        indices: set[int] = set()
        for entry in pinned:
            idx = self._resolve_pinned_index(entry, n)
            if not (0 <= idx < n):
                raise ValueError(
                    f"pinned_layers: Index {idx} ausserhalb von [0, {n}). "
                    f"Eintrag: {entry!r}."
                )
            indices.add(idx)
        return indices

    def _resolve_pinned_index(self, entry: int | str, n: int) -> int:
        if isinstance(entry, bool):
            raise ValueError(f"pinned_layers: bool nicht erlaubt: {entry!r}.")
        if isinstance(entry, int):
            return entry if entry >= 0 else n + entry
        s = entry
        if s.lstrip("-").isdigit():
            i = int(s)
            return i if i >= 0 else n + i
        if s in self._layer_list:
            return self._layer_list.index(s)
        for idx, name in enumerate(self._layer_list):
            if name.rsplit(".", 1)[-1] == s:
                return idx
        raise ValueError(
            f"pinned_layers: {entry!r} ist weder Index noch Layer-Name. "
            f"Verfuegbare Layer: {self._layer_list}."
        )

    def _associate_mtp_layers(self) -> None:
        """Ordnet jedes MTP-Modul einem Standard-Layer zu (layer -> [mtp...]).

        Heuristik in Reihenfolge:
          1. MTP-Modul liegt INNERHALB eines Standard-Layers
             (model.layers.0.mtp -> model.layers.0).
          2. Sonst: Index aus dem MTP-Namen (model.mtp_layers.0, blk.0.nextn)
             -> Standard-Layer mit gleichem Index.
          3. Sonst (z. B. model.nextn ohne Index) -> letzter Standard-Layer.
        Zusaetzlich wird die Draft-Tiefe (1-basiert) je MTP-Modul bestimmt:
        letzter numerischer Pfad-Index + 1, sonst 1.
        """
        self._layer_to_mtp = {name: [] for name in self._layer_list}
        if not self._mtp_layer_list:
            return
        by_index: dict[int, str] = {}
        for name in self._layer_list:
            tail = name.rsplit(".", 1)[-1]
            if tail.isdigit():
                by_index[int(tail)] = name
        last_layer = self._layer_list[-1]

        for mtp_name in self._mtp_layer_list:
            target: str | None = None
            for ln in self._layer_list:
                if mtp_name == ln or mtp_name.startswith(ln + "."):
                    target = ln
                    break
            if target is None:
                for token in mtp_name.split("."):
                    if token.isdigit() and int(token) in by_index:
                        target = by_index[int(token)]
                        break
            if target is None:
                target = last_layer
            self._layer_to_mtp[target].append(mtp_name)
            self._mtp_depth[mtp_name] = self._depth_from_name(mtp_name)

    @staticmethod
    def _depth_from_name(mtp_name: str) -> int:
        for token in reversed(mtp_name.split(".")):
            if token.isdigit():
                return int(token) + 1
        return 1

    @staticmethod
    def _guess_layer_prefix(modules: dict[str, torch.nn.Module]) -> str | None:
        candidates = []
        for name, mod in modules.items():
            if isinstance(mod, torch.nn.ModuleList):
                candidates.append((name, len(mod)))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _measure_layer_sizes(self) -> None:
        for name, mod in self._layers.items():
            nbytes = sum(
                p.numel() * p.element_size()
                for p in mod.parameters()
                if p.device.type != "meta"
            )
            self._layer_info[name].size_mb = nbytes / (1024 * 1024)
            self._layer_gpu_bytes[name] = nbytes
        if self.verbose:
            total = sum(li.size_mb for li in self._layer_info.values())
            avg = total / max(len(self._layers), 1)
            budget_layers = self.memory.budget_bytes / (avg * 1024 * 1024) if avg > 0 else 0
            print(
                f"[K4N0N3] Layer sizes: {avg:.1f} MB avg, {total:.1f} MB total "
                f"({len(self._layers)} layers) | Budget ~{budget_layers:.1f} layers"
            )

    # -- G+H: pinned/pageable master copies (partial pinning) ----------------

    #: Pinnen stoppt, bevor MemAvailable unter diesen Rest faellt. Ohne Floor
    #: pinnt der Per-Layer-Reprobe im fp16-Fall das System in Swap-Hunger:
    #: jede Probe sieht avail*fraction > layer_bytes, bis nichts mehr frei ist —
    #: gepinnte Seiten sind unswappbar, der Rest des Systems thrasht
    #: (empirisch: 14 % Memory-Stall, GPU idle, Messwerte unbrauchbar).
    PIN_RAM_FLOOR_BYTES: int = 1536 * 1024 * 1024

    def _can_pin(self, layer_bytes: int) -> bool:
        """O2: MemAvailable vor JEDER Pin-Entscheidung frisch lesen.

        Die fruehere Einmal-Probe beim Master-Aufbau hat das Budget in einem
        Moment gemessen, in dem der RAM-Zustand nicht dem Endzustand entsprach
        (z. B. fp16-Originale noch nicht freigegeben). Ein /proc-Read pro
        Layer ist vernachlaessigbar.
        """
        if self._pin_ram_fraction <= 0.0:
            return False
        avail = _available_ram_bytes()
        if avail <= 0:
            return False
        if avail - layer_bytes < self.PIN_RAM_FLOOR_BYTES:
            return False
        return layer_bytes <= int(avail * self._pin_ram_fraction)

    def _build_master_copies(self) -> None:
        """Build CPU master copies — pinned where budget allows, pageable (zero-copy) otherwise."""
        if self._quantize_transfer:
            # O1 Pass 1: ALLE Layer quantisieren; die fp16-Originale werden
            # dabei layerweise freigegeben (p.data zeigt auf den Quant-Master).
            for name in self._layer_list:
                master, pref, _ = self._build_quantized_master(self._layers[name])
                self._cpu_master[name] = master
                self._param_refs[name] = pref
                self._pinned[name] = False
            import gc
            gc.collect()
            # O1 Pass 2: Pinnen mit frisch geprobtem Budget — jetzt entspricht
            # der RAM-Zustand dem Endzustand (fp16 weg, nur Quant-Master da).
            for name in self._layer_list:
                master = self._cpu_master[name]
                layer_bytes = sum(_entry_bytes(e) for e in master.values())
                if self._can_pin(layer_bytes) and _pin_master_inplace(
                        self._layers[name], master, self._param_refs[name]):
                    self._pinned[name] = True
            self._log_pinning()
            return

        for name in self._layer_list:
            mod = self._layers[name]
            master: dict[str, torch.Tensor] = {}
            pref: dict[str, torch.nn.Parameter] = {}
            layer_bytes = sum(p.numel() * p.element_size() for p in mod.parameters())

            can_pin = self._can_pin(layer_bytes)

            if can_pin:
                pinned_ok = True
                for pname, param in mod.named_parameters():
                    if self._skip_trainable and param.requires_grad:
                        continue
                    try:
                        pinned = param.detach().to("cpu").pin_memory()
                    except RuntimeError:
                        pinned_ok = False
                        break
                    param.data = pinned
                    master[pname] = pinned
                    pref[pname] = param
                if pinned_ok:
                    for bname, buf in mod.named_buffers():
                        try:
                            pinned_b = buf.detach().to("cpu").pin_memory()
                        except RuntimeError:
                            pinned_ok = False
                            break
                        mod._buffers[bname] = pinned_b
                        master[bname] = pinned_b
                    if pinned_ok:
                        self._pinned[name] = True
                        self._cpu_master[name] = master
                        self._param_refs[name] = pref
                        continue

            # Pageable fallback — G: zero-copy, original IS the master
            self._pinned[name] = False
            for pname, param in mod.named_parameters():
                if self._skip_trainable and param.requires_grad:
                    continue
                master[pname] = param.data  # reference, no clone
                pref[pname] = param
            for bname, buf in mod.named_buffers():
                master[bname] = buf
            self._cpu_master[name] = master
            self._param_refs[name] = pref

        self._log_pinning()

    def _log_pinning(self) -> None:
        if not self.verbose:
            return
        n_pinned = sum(1 for v in self._pinned.values() if v)
        n_total = len(self._layer_list)
        total_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**2
        pinned_mb = sum(
            sum(_entry_bytes(t) for t in d.values())
            for n, d in self._cpu_master.items() if self._pinned.get(n, False)
        ) / 1024**2
        print(f"[K4N0N3] Pinned {n_pinned}/{n_total} layers "
              f"({pinned_mb:.0f}/{total_mb:.0f} MB), "
              f"pin fraction {self._pin_ram_fraction*100:.0f}% (per-layer reprobe)"
              + (f" | quantize_transfer={self._quantize_transfer}"
                 if self._quantize_transfer else ""))

    def _build_quantized_master(self, mod: torch.nn.Module) -> tuple[dict, dict, int]:
        """M1: Linear-Weights → int8-Master {"q","scale"}; alles andere direkte Master.

        Drop-Entscheidung (M2): p.data zeigt nach Drop auf den int8-Master —
        es existiert bewusst keine fp16-CPU-Kopie mehr (RAM-Halbierung ist
        Kernthese 1). Der Upload-Pfad liest grundsaetzlich aus der
        Master-Struktur, nie aus p.data. CPU-Forward ist damit unmoeglich,
        deshalb verlangt __init__ bei quantize_transfer=True eine GPU.
        """
        linear_weights = {
            (f"{sub}.weight" if sub else "weight")
            for sub, m in mod.named_modules()
            if isinstance(m, torch.nn.Linear)
        }
        master: dict = {}
        pref: dict[str, torch.nn.Parameter] = {}
        for pname, param in mod.named_parameters():
            if self._skip_trainable and param.requires_grad:
                continue
            if pname in linear_weights and param.dim() == 2:
                if self._quantize_transfer == "int4" and param.shape[1] % 2 == 0:
                    packed, scale, meta = quantize_groupwise_int4(
                        param.data, self._int4_group_size)
                    entry = {"q4": packed, "scale": scale, "meta": meta}
                else:  # int8 — auch Fallback fuer ungerade Spaltenzahl bei int4
                    q, scale = quantize_per_channel_int8(param.data)
                    entry = {"q": q, "scale": scale}
                master[pname] = entry
                # Inference-only: int-.data ist mit requires_grad unvereinbar
                param.requires_grad_(False)
                param.data = _packed_tensor(entry)  # fp16-Original wird freigegeben
            else:
                master[pname] = param.data
            pref[pname] = param
        for bname, buf in mod.named_buffers():
            master[bname] = buf
        layer_bytes = sum(_entry_bytes(e) for e in master.values())
        return master, pref, layer_bytes

    # -- hooks --------------------------------------------------------------

    def _register_hooks(self) -> None:
        for name, module in self._layers.items():
            pre = module.register_forward_pre_hook(self._make_pre_hook(name))
            post = module.register_forward_hook(self._make_post_hook(name))
            self._hook_handles.extend([pre, post])
        if self.use_mtp:
            try:
                self._root_hook_handle = self.model.register_forward_pre_hook(
                    self._make_root_pre_hook(), with_kwargs=True)
            except TypeError:
                # torch < 2.0: kein with_kwargs — nur input_ids via args[0].
                self._root_hook_handle = self.model.register_forward_pre_hook(
                    self._make_root_pre_hook())

    def _make_root_pre_hook(self) -> Callable:
        """Sammelt input_ids/attention_mask/position_ids des aktuellen Forwards.

        Der Layer-Post-Hook sieht nur hidden_states; echte MTP-Heads brauchen
        aber die Input-Ebene. Der Wurzel-Pre-Hook speichert sie in _forward_ctx.
        """
        def hook(module: torch.nn.Module, args, kwargs=None):
            ctx: dict[str, torch.Tensor] = {}
            if kwargs is not None and "input_ids" in kwargs:
                ctx["input_ids"] = kwargs["input_ids"]
            elif args and torch.is_tensor(args[0]):
                ctx["input_ids"] = args[0]
            if kwargs is not None:
                for key in ("attention_mask", "position_ids"):
                    if key in kwargs and kwargs[key] is not None:
                        ctx[key] = kwargs[key]
            self._forward_ctx = ctx
        return hook

    def _make_pre_hook(self, name: str) -> Callable:
        def hook(module: torch.nn.Module, args):
            if not self._prepared:
                self.prepare()
            if self._layer_list and name == self._layer_list[0]:
                global _layer0_fire_count
                _layer0_fire_count += 1
                # Neuer Forward-Durchlauf: MTP-Buffer zuruecksetzen, damit
                # zwischen Generation-Steps keine Alt-Referenzen kumulieren.
                if self.use_mtp:
                    self._mtp_buffer.clear()
            t0 = time.perf_counter()
            if name not in self._pinned_names:
                self.memory.mark_on_gpu(name, module, self._layer_gpu_bytes.get(name))
            self._ensure_on_gpu(name)
            dt = (time.perf_counter() - t0) * 1000
            self._layer_info[name].transfer_time_ms = dt
            if self.verbose:
                used = self.memory.used_bytes() / 1024**2
                print(f"  [pre]  {name} | {dt:5.1f}ms | {used:5.0f}MB | {_state_summary(self._layer_info)}")
        return hook

    def _make_post_hook(self, name: str) -> Callable:
        def hook(module: torch.nn.Module, args, output):
            # Dual-Pass: zugehoerige MTP/Draft-Module auf dem Layer-Output
            # ausfuehren, SOLANGE der Layer noch geladen ist (fail-fast).
            if self.use_mtp and self._layer_to_mtp.get(name):
                self._run_mtp_pass(name, output)
            idx = self._layer_idx[name]
            n = len(self._layer_list)
            # Wrap-around prefetch for autoregressive generation
            for offset in range(1, self.prefetch_depth + 1):
                nxt = (idx + offset) % n
                self._prefetch_async(self._layer_list[nxt])
            # Offload layers far behind (modulo-safe)
            for offset in range(self.prefetch_depth + 1, self.prefetch_depth + 3):
                prev = (idx - offset) % n
                # Don't offload layers in the prefetch window
                self._offload(self._layer_list[prev])
            if self.verbose:
                print(f"  [post] {name} | {_state_summary(self._layer_info)}")
        return hook

    def _run_mtp_pass(self, layer_name: str, hidden_states: torch.Tensor) -> None:
        """Fuehrt die dem Layer zugeordneten MTP-Module aus und buffert die Outputs.

        Inspiziert die forward-Signatur dynamisch: erwartet das Modul nur
        hidden_states, wird der direkte Ein-Argument-Aufruf genutzt (Fallback).
        Verlangt es position_ids/input_ids/attention_mask, werden sie aus
        _forward_ctx ergaenzt, position_ids um die Draft-Tiefe geshiftet und
        auf das Device der hidden_states gelegt.
        """
        for mtp_name in self._layer_to_mtp.get(layer_name, []):
            mod = self._mtp_layers[mtp_name]
            depth = self._mtp_depth.get(mtp_name, 1)
            kwargs = self._build_mtp_kwargs(mod, hidden_states, depth)
            with torch.no_grad():
                if kwargs:
                    out = mod(hidden_states, **kwargs)
                else:
                    out = mod(hidden_states)
            self._mtp_buffer.setdefault(mtp_name, []).append(out)

    def _build_mtp_kwargs(
        self, mod: torch.nn.Module, hidden_states: torch.Tensor, depth: int
    ) -> dict[str, torch.Tensor] | None:
        """Baut kwargs fuer einen MTP-Aufruf; None bedeutet single-arg-Aufruf."""
        names = self._forward_param_names(mod)
        if names is None or len(names) <= 1:
            return None
        ctx = self._forward_ctx
        kwargs: dict[str, torch.Tensor] = {}
        if "position_ids" in names:
            kwargs["position_ids"] = self._shifted_position_ids(hidden_states, depth)
        if "input_ids" in names and "input_ids" in ctx:
            kwargs["input_ids"] = ctx["input_ids"].to(hidden_states.device)
        if "attention_mask" in names and ctx.get("attention_mask") is not None:
            kwargs["attention_mask"] = ctx["attention_mask"].to(hidden_states.device)
        return kwargs or None

    @staticmethod
    def _forward_param_names(mod: torch.nn.Module) -> list[str] | None:
        try:
            sig = inspect.signature(mod.forward)
        except (ValueError, TypeError):
            return None
        names = [
            name for name, p in sig.parameters.items()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          inspect.Parameter.KEYWORD_ONLY)
            and name != "self"
        ]
        return names

    def _shifted_position_ids(self, hidden_states: torch.Tensor, depth: int) -> torch.Tensor:
        """position_ids fuer Draft-Tiefe ``depth`` (Basis + depth).

        Basis aus _forward_ctx, sonst automatisch arange(seq_len) auf dem
        Device/Dtype der hidden_states.
        """
        dev = hidden_states.device
        seq_len = hidden_states.shape[1]
        base = self._forward_ctx.get("position_ids")
        if base is None:
            base = torch.arange(seq_len, device=dev, dtype=torch.long).unsqueeze(0)
        else:
            base = base.to(device=dev)
        return base + depth

    # -- A2: upload = copy from pinned master -------------------------------

    def _ensure_on_gpu(self, name: str) -> None:
        if not self._cuda_available:
            return
        info = self._layer_info[name]
        if info.state == LayerState.ON_GPU:
            return
        if info.state == LayerState.PREFETCHING:
            event = self._prefetch_events.pop(name, None)
            if event is not None:
                torch.cuda.current_stream().wait_event(event)
            # record_stream: tell allocator these tensors are now on main stream
            for pname in self._cpu_master.get(name, {}):
                param = _get_param_by_name(self._layers[name], pname)
                if param is not None and param.device.type == "cuda":
                    param.data.record_stream(torch.cuda.current_stream())
            info.state = LayerState.ON_GPU
            return
        # Cold upload
        _upload_layer(self._layers[name], self._cpu_master.get(name, {}), self._param_refs.get(name))
        torch.cuda.current_stream().synchronize()
        info.state = LayerState.ON_GPU

    def _prefetch_async(self, name: str) -> None:
        if not self._cuda_available:
            return
        if name in self._pinned_names:
            return  # gepinnt: bleibt dauerhaft auf GPU, nie prefetchen/buchen
        info = self._layer_info[name]
        if info.state != LayerState.ON_CPU:
            return
        # B: reserve budget before copying — evict if needed
        evicted = self.memory.mark_on_gpu(name, self._layers[name], self._layer_gpu_bytes.get(name))
        for ev_name in evicted:
            if ev_name != name:
                self._offload(ev_name)
        info.state = LayerState.PREFETCHING
        with torch.cuda.stream(self._prefetch_stream):
            _upload_layer(self._layers[name], self._cpu_master.get(name, {}), self._param_refs.get(name))
        # Event NACH dem Dequant (letzter Kernel im Stream) — der Konsument
        # braucht das fertige fp16, nicht nur die Kopie
        self._prefetch_events[name] = self._prefetch_stream.record_event()

    # -- A3: offload = drop GPU tensor, point back to pinned master ---------

    def _offload(self, name: str) -> None:
        if name in self._pinned_names:
            return  # gepinnt: bleibt dauerhaft auf GPU (resident)
        info = self._layer_info[name]
        if info.state == LayerState.ON_CPU:
            return
        # If prefetch still in flight, wait for it before dropping
        if info.state == LayerState.PREFETCHING:
            event = self._prefetch_events.pop(name, None)
            if event is not None:
                event.synchronize()
        # Drop: point .data back to pinned master
        _drop_layer(self._layers[name], self._cpu_master.get(name, {}), self._param_refs.get(name))
        self._prefetch_events.pop(name, None)
        info.state = LayerState.ON_CPU
        self.memory.mark_off_gpu(name)

    # -- public API ---------------------------------------------------------

    def prepare(self) -> None:
        """Move non-layer modules to GPU, drop layers to CPU, prefetch first batch."""
        if self._prepared and self.memory.used_bytes() >= 0:
            # Already prepared — just verify consistency and prefetch first layers
            for i in range(min(self.prefetch_depth + 1, len(self._layer_list))):
                name = self._layer_list[i]
                if self._layer_info[name].state == LayerState.ON_CPU:
                    self._prefetch_async(name)
            if self.verbose:
                print(f"[K4N0N3] Re-prepare: {_state_summary(self._layer_info)}")
            return

        self.memory.clear()
        self._prefetch_events.clear()

        for name in self._layer_list:
            _drop_layer(self._layers[name], self._cpu_master.get(name, {}), self._param_refs.get(name))
            self._layer_info[name].state = LayerState.ON_CPU

        if self._cuda_available:
            self._move_fixed_to_gpu()
            # Gepinnte Layer vorgluehen: dauerhaft auf GPU, ohne LRU-Buchung.
            for name in self._pinned_names:
                self._ensure_on_gpu(name)

        limit = min(self.prefetch_depth + 1, len(self._layer_list))
        for i in range(limit):
            name = self._layer_list[i]
            if i == 0:
                self._ensure_on_gpu(name)
                if name not in self._pinned_names:
                    self.memory.mark_on_gpu(name, self._layers[name], self._layer_gpu_bytes.get(name))
            else:
                self._prefetch_async(name)

        self._prepared = True
        if self.verbose:
            print(f"[K4N0N3] Prepared: {_state_summary(self._layer_info)}")

    def _move_fixed_to_gpu(self) -> None:
        """Move non-layer leaf modules to GPU without touching layer params."""
        layer_set = set(self._layers.keys())
        for mod_name, mod in self.model.named_modules():
            # Exact prefix match with dot boundary
            if _matches_any_layer(mod_name, layer_set):
                continue
            if _has_own_tensors(mod):
                mod.to("cuda", non_blocking=True)

    def offload_all(self) -> None:
        """Drop all tracked layers to CPU. Non-layer modules go to CPU via .to()."""
        for name in self._layer_list:
            self._offload(name)
        # Move remaining (non-layer) modules to CPU
        layer_set = set(self._layers.keys())
        for mod_name, mod in self.model.named_modules():
            if _matches_any_layer(mod_name, layer_set):
                continue
            if _has_own_tensors(mod):
                mod.to("cpu")
        self._prepared = False

    @property
    def _layer_idx(self) -> dict[str, int]:
        if not hasattr(self, "_layer_idx_cache"):
            self._layer_idx_cache = {n: i for i, n in enumerate(self._layer_list)}
        return self._layer_idx_cache

    # -- MTP/Layer-Mapping ---------------------------------------------------

    @property
    def standard_layers(self) -> list[torch.nn.Module]:
        """Die Standard-Transformer-Blöcke (Haupt-Layer) als Modul-Liste."""
        return list(self._layers.values())

    @property
    def mtp_layers(self) -> list[torch.nn.Module]:
        """Erkannte MTP/Draft-Head-Module (leer, wenn keine gefunden)."""
        return list(self._mtp_layers.values())

    @property
    def layer_map(self) -> dict[str, list[torch.nn.Module]]:
        """Strukturiertes Layer-Mapping: standard_layers + mtp_layers."""
        return {
            "standard_layers": self.standard_layers,
            "mtp_layers": self.mtp_layers,
        }

    def get_mtp_buffer(self) -> dict[str, list[torch.Tensor]]:
        """Gepufferte MTP/Draft-Outputs des letzten Forward-Durchlaufs.

        Nicht destruktiv: Key = MTP-Modulname, Value = Liste der Outputs.
        """
        return self._mtp_buffer

    def clear_mtp_buffer(self) -> None:
        """Leert den MTP-Buffer (gibt Referenzen fuer den GC frei)."""
        self._mtp_buffer.clear()

    def stats(self) -> dict:
        return {
            name: {
                "size_mb": info.size_mb,
                "transfer_ms": info.transfer_time_ms,
                "compute_ms": info.compute_time_ms,
            }
            for name, info in self._layer_info.items()
        }

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        if self._root_hook_handle is not None:
            self._root_hook_handle.remove()
            self._root_hook_handle = None


# -- A2/M2 helpers: per-parameter upload/drop --------------------------------

# T-Diagnose (Auftrag 6 T): zaehlt einzelne H2D-Copies im Upload-Pfad, um
# Fragmentierung (viele kleine Transfers statt einem Block) zu beziffern. Ein
# int-Increment je .to("cuda") ist gegen die Kopie selbst vernachlaessigbar;
# der Zaehler bleibt immer an, der Probe nullt ihn vor der Messung.
_upload_copy_count = 0

# V-Diagnose (Auftrag 6 V): Feuerungen des Pre-Hooks des ERSTEN Layers = Anzahl
# voller Modell-Durchläufe. Geteilt durch erzeugte Tokens ergibt den
# Amortisierungsfaktor (1,0 = ein Forward je Token, <1 = spekulativ gewonnen).
_layer0_fire_count = 0


def reset_upload_copy_count() -> int:
    """Gibt den bisherigen Zaehlerstand zurueck und setzt ihn auf 0."""
    global _upload_copy_count
    n, _upload_copy_count = _upload_copy_count, 0
    return n


def reset_layer0_fire_count() -> int:
    """Feuerungen des ersten Layers seit dem letzten Reset; setzt auf 0."""
    global _layer0_fire_count
    n, _layer0_fire_count = _layer0_fire_count, 0
    return n


def _upload_layer(module: torch.nn.Module, master: dict,
                  param_refs: dict[str, torch.nn.Parameter] | None = None) -> None:
    """Copy each param/buffer from master to GPU (non-blocking).

    int8-Master ({"q","scale"}): Copy + On-GPU-Dequant im aufrufenden Stream.
    Die int8/scale-Staging-Tensoren verlieren nach dem Dequant ihre Referenz,
    der Allocator gibt sie frei.
    """
    global _upload_copy_count
    if param_refs is None:
        param_dict = dict(module.named_parameters())
    else:
        param_dict = param_refs
    for pname, entry in master.items():
        param = param_dict.get(pname)
        if isinstance(entry, dict):  # M2/P: int8/int4 + scale → fp16 auf der GPU
            if param is not None:
                q_gpu = _packed_tensor(entry).to("cuda", non_blocking=True)
                s_gpu = entry["scale"].to("cuda", non_blocking=True)
                _upload_copy_count += 2
                if "q4" in entry:
                    param.data = dequantize_groupwise_int4(q_gpu, s_gpu, entry["meta"])
                else:
                    param.data = dequantize_int8(q_gpu, s_gpu)
            continue
        if param is not None:
            param.data = entry.to("cuda", non_blocking=True)
            _upload_copy_count += 1
        elif pname in module._buffers:
            module._buffers[pname] = entry.to("cuda", non_blocking=True)
            _upload_copy_count += 1


def _drop_layer(module: torch.nn.Module, master: dict,
                param_refs: dict[str, torch.nn.Parameter] | None = None) -> None:
    """Point each param/buffer back to its CPU master — GPU tensor freed.

    int8-Master: p.data zeigt nach dem Drop auf den int8-Master (siehe
    _build_quantized_master fuer die Begruendung).
    """
    if param_refs is None:
        param_dict = dict(module.named_parameters())
    else:
        param_dict = param_refs
    for pname, entry in master.items():
        param = param_dict.get(pname)
        if isinstance(entry, dict):
            if param is not None:
                param.data = _packed_tensor(entry)
            continue
        if param is not None:
            param.data = entry
        elif pname in module._buffers:
            module._buffers[pname] = entry


def _has_own_tensors(module: torch.nn.Module) -> bool:
    """Eigene Parameter ODER Buffer (nicht rekursiv).

    Buffer muessen mit: Qwen3.5 haelt die Rotary-Frequenzen (`inv_freq`) in
    einem Modul ganz ohne Parameter. Wer nur auf Parameter prueft, laesst es
    auf der CPU liegen — der erste Forward stirbt dann mit „two devices".
    """
    return bool(list(module.parameters(recurse=False))
                or list(module.buffers(recurse=False)))


def _get_param_by_name(module: torch.nn.Module, name: str) -> torch.nn.Parameter | None:
    for n, p in module.named_parameters():
        if n == name:
            return p
    return None


# -- helpers --------------------------------------------------------------


def _entry_bytes(entry) -> int:
    """Bytes eines Master-Eintrags — plain Tensor oder Quant-Dict {"q"|"q4","scale","meta"}."""
    if isinstance(entry, dict):
        return sum(t.numel() * t.element_size()
                   for t in entry.values() if isinstance(t, torch.Tensor))
    return entry.numel() * entry.element_size()


def _packed_tensor(entry: dict) -> torch.Tensor:
    """Der Transfer-Tensor eines Quant-Eintrags (int8-Matrix oder int4-Packung)."""
    return entry["q4"] if "q4" in entry else entry["q"]


def _pin_master_inplace(mod: torch.nn.Module, master: dict,
                        pref: dict[str, torch.nn.Parameter]) -> bool:
    """Pin all master entries in place; re-point param/buffer data. False on failure."""
    try:
        for key in list(master.keys()):
            entry = master[key]
            if isinstance(entry, dict):
                pinned_entry = {k: (t.pin_memory() if isinstance(t, torch.Tensor) else t)
                                for k, t in entry.items()}
                master[key] = pinned_entry
                if key in pref:
                    pref[key].data = _packed_tensor(pinned_entry)
            else:
                pinned = entry if entry.is_pinned() else entry.pin_memory()
                master[key] = pinned
                if key in pref:
                    pref[key].data = pinned
                elif key in mod._buffers:
                    mod._buffers[key] = pinned
    except RuntimeError:
        return False
    return True


def _matches_any_layer(mod_name: str, layer_set: set[str]) -> bool:
    """Check if mod_name is inside any tracked layer (exact prefix with dot)."""
    for ln in layer_set:
        if mod_name == ln or mod_name.startswith(ln + "."):
            return True
    return False


def _state_summary(info: dict[str, LayerInfo]) -> str:
    on_gpu = [n for n, i in info.items() if i.state == LayerState.ON_GPU]
    pref = [n for n, i in info.items() if i.state == LayerState.PREFETCHING]
    cpu = [n for n, i in info.items() if i.state == LayerState.ON_CPU]
    parts = []
    if on_gpu:
        parts.append(f"GPU:{_range_str(on_gpu)}")
    if pref:
        parts.append(f"PREF:{_range_str(pref)}")
    if cpu:
        parts.append(f"CPU:{_range_str(cpu)}")
    return " ".join(parts)


def _range_str(names: list[str]) -> str:
    if len(names) <= 3:
        return ",".join(n.rsplit(".", 1)[-1] for n in names)
    first = names[0].rsplit(".", 1)[-1]
    last = names[-1].rsplit(".", 1)[-1]
    return f"{first}..{last}({len(names)})"


def _available_ram_bytes() -> int:
    """Read MemAvailable from /proc/meminfo (Linux). Returns 0 on failure."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # kB → bytes
    except Exception:
        pass
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        return 0


def _resolve_module(root: torch.nn.Module, dotted: str) -> torch.nn.Module | None:
    parts = dotted.split(".")
    current = root
    for part in parts:
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current
