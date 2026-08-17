"""Phase 5 — Echte-Modell-Integrationstests (lokal, CPU oder GPU).

Prueft, dass use_mtp=True auf Modellen OHNE MTP-Module graceful degradiert:
mtp_layers == [] und generate() ist verlustfrei identisch zum Standard-Greedy
(die Engine laeuft dann mit leeren Drafts = reiner Greedy). Skippt, wenn das
Modell nicht lokal verfuegbar ist.
"""
import pytest
import torch

from k4n0n3 import ZeroFlushModel

PROMPT = "The quick brown fox jumps over the lazy dog"


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load(model_name: str, use_mtp: bool) -> ZeroFlushModel:
    try:
        return ZeroFlushModel(model_name, device=_device(),
                              vram_budget_mb=2048, use_mtp=use_mtp)
    except Exception as e:  # noqa: BLE001 — OSError wenn Modell nicht lokal
        pytest.skip(f"Modell {model_name} nicht verfuegbar: {type(e).__name__}")


@pytest.fixture(scope="module")
def gpt2_base():
    return _load("openai-community/gpt2", use_mtp=False)


@pytest.fixture(scope="module")
def gpt2_mtp():
    return _load("openai-community/gpt2", use_mtp=True)


@pytest.fixture(scope="module")
def qwen_base():
    return _load("Qwen/Qwen2.5-0.5B", use_mtp=False)


@pytest.fixture(scope="module")
def qwen_mtp():
    return _load("Qwen/Qwen2.5-0.5B", use_mtp=True)


@pytest.mark.integration
def test_gpt2_standard_generate_deterministic(gpt2_base):
    a = gpt2_base.generate(PROMPT, max_new_tokens=16, do_sample=False)
    b = gpt2_base.generate(PROMPT, max_new_tokens=16, do_sample=False)
    assert a == b
    assert len(gpt2_base._last_new_token_ids) == 16


@pytest.mark.integration
def test_gpt2_use_mtp_true_graceful_degradation(gpt2_base, gpt2_mtp):
    assert gpt2_mtp.layer_manager.mtp_layers == []
    gpt2_base.generate(PROMPT, max_new_tokens=16, do_sample=False)
    gpt2_mtp.generate(PROMPT, max_new_tokens=16, do_sample=False)
    assert gpt2_base._last_new_token_ids == gpt2_mtp._last_new_token_ids


@pytest.mark.integration
def test_resolve_lm_head(gpt2_mtp):
    head = gpt2_mtp._resolve_lm_head()
    assert head is gpt2_mtp.model.lm_head


@pytest.mark.integration
def test_qwen25_05b_use_mtp_true_graceful_degradation(qwen_base, qwen_mtp):
    assert qwen_mtp.layer_manager.mtp_layers == []
    qwen_base.generate(PROMPT, max_new_tokens=16, do_sample=False)
    qwen_mtp.generate(PROMPT, max_new_tokens=16, do_sample=False)
    assert qwen_base._last_new_token_ids == qwen_mtp._last_new_token_ids
