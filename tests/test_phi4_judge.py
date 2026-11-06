"""Tests for the local Transformers judge backend used by Kaggle Phi-4 runs."""

import sys
import types

import torch

from src.config import JUDGE_HF_MODEL
from src.core.llm_client import UnifiedLLMClient


def test_transformers_judge_loads_and_generates_without_provider_fallback(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("JUDGE_PROVIDER", "transformers")
    monkeypatch.setenv("JUDGE_MODEL", "microsoft/phi-4")
    monkeypatch.setenv("PHI4_LOAD_IN_4BIT", "1")
    monkeypatch.setenv("PHI4_MAX_INPUT_TOKENS", "4096")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 2
        pad_token = None
        eos_token = "</s>"

        @classmethod
        def from_pretrained(cls, model_name):
            assert model_name == "microsoft/phi-4"
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            return "rendered prompt"

        def __call__(self, prompt, return_tensors):
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

        def decode(self, tokens, skip_special_tokens=True):
            return "PHI4_TEST_OK"

    class FakeModel:
        device = torch.device("cpu")

        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            assert model_name == "microsoft/phi-4"
            assert kwargs["quantization_config"].load_in_4bit is True
            return cls()

        def eval(self):
            return self

        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3]])

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=FakeTokenizer,
        AutoModelForCausalLM=FakeModel,
        BitsAndBytesConfig=FakeBitsAndBytesConfig,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    judge = UnifiedLLMClient(
        model=JUDGE_HF_MODEL,
        temperature=0.0,
        max_tokens=32,
    )

    assert judge.provider == "transformers"
    assert judge.model == "microsoft/phi-4"
    assert judge.chat([{"role": "user", "content": "test"}]) == "PHI4_TEST_OK"


def test_transformers_backend_does_not_override_generator(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("JUDGE_PROVIDER", "transformers")

    generator = UnifiedLLMClient(model="llama3.1:8b")

    assert generator.provider == "ollama"
    assert generator.model == "llama3.1:8b"
