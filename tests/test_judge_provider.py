"""Tests for the JUDGE_PROVIDER override in UnifiedLLMClient.

Generator and judge both use Ollama by default, with independent model names.
The judge is identified by constructing the client with JUDGE_HF_MODEL.
"""

import pytest

from src.config import JUDGE_HF_MODEL, JUDGE_NVIDIA_MODEL, JUDGE_OLLAMA_MODEL
from src.core.llm_client import UnifiedLLMClient


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("JUDGE_PROVIDER", raising=False)
    # src.config is imported before fixtures run; neutralize any JUDGE_PROVIDER
    # value loaded from the notebook-created .env so this test represents the
    # true no-override/default configuration.
    monkeypatch.setattr("src.config.JUDGE_PROVIDER", "ollama")


def test_hybrid_judge_on_huggingface_generator_on_ollama(monkeypatch):
    monkeypatch.setenv("JUDGE_PROVIDER", "huggingface")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    judge = UnifiedLLMClient(model=JUDGE_HF_MODEL)
    generator = UnifiedLLMClient(model=None)

    assert judge.provider == "huggingface"
    assert judge.model == JUDGE_HF_MODEL  # Qwen/Qwen2.5-72B-Instruct
    assert generator.provider == "ollama"
    assert generator.model == "llama3.1:8b"


def test_no_override_judge_follows_llm_provider():
    judge = UnifiedLLMClient(model=JUDGE_HF_MODEL)
    generator = UnifiedLLMClient(model=None)

    assert judge.provider == "ollama"
    # Whatever JUDGE_OLLAMA_MODEL the environment configured (default or .env).
    assert judge.model == JUDGE_OLLAMA_MODEL
    assert generator.provider == "ollama"
    assert generator.model == "llama3.1:8b"


def test_judge_provider_nvidia(monkeypatch):
    monkeypatch.setenv("JUDGE_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi_test")

    judge = UnifiedLLMClient(model=JUDGE_HF_MODEL)
    generator = UnifiedLLMClient(model=None)

    assert judge.provider == "nvidia"
    assert judge.model == JUDGE_NVIDIA_MODEL
    assert generator.provider == "ollama"


def test_judge_provider_explicit_kwarg_still_wins(monkeypatch):
    monkeypatch.setenv("JUDGE_PROVIDER", "huggingface")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    # An explicit provider argument takes precedence over the env override.
    judge = UnifiedLLMClient(model=JUDGE_HF_MODEL, provider="ollama")
    assert judge.provider == "ollama"
    assert judge.model == JUDGE_OLLAMA_MODEL


def test_generator_untouched_by_judge_provider(monkeypatch):
    monkeypatch.setenv("JUDGE_PROVIDER", "huggingface")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    # Generator-style model names never trigger the judge override.
    generator = UnifiedLLMClient(model="llama3.1:8b")
    assert generator.provider == "ollama"
    assert generator.model == "llama3.1:8b"
