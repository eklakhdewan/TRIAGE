"""Tests for Phi-4 running as the Ollama judge (not Transformers)."""

import json

from src.config import JUDGE_HF_MODEL
from src.core.llm_client import UnifiedLLMClient


def test_phi4_ollama_is_selected_for_judge(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("JUDGE_PROVIDER", "ollama")
    monkeypatch.setenv("JUDGE_OLLAMA_MODEL", "phi4:14b")

    judge = UnifiedLLMClient(model=JUDGE_HF_MODEL, temperature=0.0, max_tokens=32)

    assert judge.provider == "ollama"
    assert judge.model == "phi4:14b"


def test_phi4_judge_uses_gpu_while_generator_is_cpu(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("JUDGE_PROVIDER", "ollama")
    monkeypatch.setenv("JUDGE_OLLAMA_MODEL", "phi4:14b")

    captured = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "PHI4_TEST_OK"}}

    def fake_post(url, json=None, timeout=None):
        captured["payload"] = json
        return Response()

    monkeypatch.setattr("requests.post", fake_post)

    judge = UnifiedLLMClient(model=JUDGE_HF_MODEL, temperature=0.0, max_tokens=32)
    generator = UnifiedLLMClient(model=None, temperature=0.1, max_tokens=32)

    assert judge.chat([{"role": "user", "content": "test"}]) == "PHI4_TEST_OK"
    judge_options = captured["payload"]["options"]
    assert "num_gpu" not in judge_options

    assert generator.chat([{"role": "user", "content": "test"}]) == "PHI4_TEST_OK"
    generator_options = captured["payload"]["options"]
    assert generator_options["num_gpu"] == 0
