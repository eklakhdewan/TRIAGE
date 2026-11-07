"""Unified LLM client (audit fixes: no stateful provider mutation, Ollama timeout,
honest token budgets, explicit JSON mode)."""
import logging
import os
import time
from typing import Optional

import requests

from src.config import LLM_OLLAMA_TIMEOUT_S

logger = logging.getLogger(__name__)

OLLAMA_TIMEOUT_S = LLM_OLLAMA_TIMEOUT_S

class UnifiedLLMClient:
    """Unified LLM interface supporting Hugging Face Inference API with fallback.

    Fixes applied :
    - A transient HF failure no longer permanently mutates the instance; the
      fallback is scoped to the failing call.
    - Ollama calls carry an explicit timeout.
    - `max_tokens` is honored (no silent 768 cap).
    - JSON mode is an explicit parameter, not substring sniffing.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ):
        from src.config import (
            LLM_PROVIDER, LLM_HF_MODEL, JUDGE_HF_MODEL, JUDGE_MODEL,
            JUDGE_PROVIDER, JUDGE_OLLAMA_MODEL, LLM_MODEL, HF_TOKEN,
        )

        self.is_judge = model in (JUDGE_HF_MODEL, JUDGE_MODEL, "judge")
        self.provider = (provider or os.getenv("LLM_PROVIDER") or LLM_PROVIDER).lower()
        # An explicit provider kwarg always wins over the judge override.
        if not provider and self.is_judge:
            judge_provider = (os.getenv("JUDGE_PROVIDER") or JUDGE_PROVIDER).strip().lower()
            if judge_provider:
                self.provider = judge_provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.token = os.getenv("HF_TOKEN") or HF_TOKEN
        self.ollama_model = (
            JUDGE_OLLAMA_MODEL
            if self.is_judge
            else LLM_MODEL
            if model is None or model in (LLM_MODEL, LLM_HF_MODEL)
            else model
        )

        if self.provider == "transformers":
            if not self.is_judge:
                raise ValueError("The Transformers backend is reserved for the configured judge.")
            self.model = JUDGE_MODEL
            self._load_transformers_judge()
            return

        if self.provider == "nvidia":
            from src.config import NVIDIA_API_KEY, NVIDIA_BASE_URL, LLM_NVIDIA_MODEL, JUDGE_NVIDIA_MODEL
            self.nvidia_key = os.getenv("NVIDIA_API_KEY") or NVIDIA_API_KEY
            self.nvidia_url = (os.getenv("NVIDIA_BASE_URL") or NVIDIA_BASE_URL).rstrip("/") + "/chat/completions"
            if model == JUDGE_HF_MODEL or model == "judge":
                self.model = JUDGE_NVIDIA_MODEL
            elif not model or model in (LLM_MODEL, LLM_HF_MODEL, "llama3.1:8b"):
                self.model = LLM_NVIDIA_MODEL
            else:
                self.model = model
            if not self.nvidia_key:
                logger.warning("NVIDIA_API_KEY not found; falling back to Ollama.")
                self.provider = "ollama"
                self.model = self.ollama_model
            else:
                return

        if self.provider == "huggingface" and self.token:
            self.model = LLM_HF_MODEL if (not model or model == LLM_MODEL) else model
            try:
                from huggingface_hub import InferenceClient
                self._hf_client = InferenceClient(api_key=self.token, timeout=90.0)
                return
            except Exception as e:
                logger.warning("Failed to initialize HF client: %s. Using Ollama.", e)
        else:
            if self.provider != "nvidia":
                logger.info("HF provider unavailable (no token) - using Ollama.")
        self.provider = "ollama"
        self.model = self.ollama_model

    def _load_transformers_judge(self):
        """Load the local causal-LM judge; Phi-4 is quantized for Kaggle T4."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError(
                "JUDGE_PROVIDER=transformers requires CUDA. Enable Kaggle GPU T4."
            )

        load_in_4bit = os.getenv("PHI4_LOAD_IN_4BIT", "1").strip() == "1"
        max_input_tokens = int(os.getenv("PHI4_MAX_INPUT_TOKENS", "4096"))
        if not 512 <= max_input_tokens <= 12000:
            raise ValueError("PHI4_MAX_INPUT_TOKENS must be between 512 and 12000.")

        self._judge_tokenizer = AutoTokenizer.from_pretrained(self.model)
        if self._judge_tokenizer.pad_token_id is None:
            self._judge_tokenizer.pad_token = self._judge_tokenizer.eos_token

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        logger.info(
            "Loading judge %s (4-bit NF4=%s, input budget=%d)",
            self.model, load_in_4bit, max_input_tokens,
        )
        self._judge_model = AutoModelForCausalLM.from_pretrained(
            self.model,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: "12GiB", "cpu": "24GiB"},
            low_cpu_mem_usage=True,
        )
        self._judge_model.eval()
        self._judge_max_input_tokens = max_input_tokens
        logger.info(
            "Transformers judge loaded; GPU allocated %.2f GiB",
            torch.cuda.memory_allocated() / (1024 ** 3),
        )

    def _chat_transformers(self, messages, temp, tokens, json_mode):
        """Run deterministic local chat generation through the loaded judge."""
        import torch

        prompt_messages = list(messages)
        if json_mode:
            prompt_messages.append({
                "role": "system",
                "content": "Return only valid JSON. Do not include Markdown fences.",
            })
        prompt = self._judge_tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._judge_tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        if input_ids.shape[-1] > self._judge_max_input_tokens:
            # Preserve the answer and output schema at the end of the prompt.
            input_ids = input_ids[:, -self._judge_max_input_tokens:]
            inputs["input_ids"] = input_ids
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, -self._judge_max_input_tokens:]

        inputs = {key: value.to(self._judge_model.device) for key, value in inputs.items()}
        pad_id = self._judge_tokenizer.pad_token_id or self._judge_tokenizer.eos_token_id
        with torch.inference_mode():
            output = self._judge_model.generate(
                **inputs,
                max_new_tokens=tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_id,
                eos_token_id=self._judge_tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[-1]:]
        return self._judge_tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip()

    def _chat_ollama(self, messages, temp, tokens, json_mode, model):
        import ollama
        options = {"temperature": temp, "num_predict": tokens}
        # Keep the answer generator on CPU; reserve the Kaggle T4 for Phi-4.
        if not self.is_judge:
            options["num_gpu"] = 0
        kwargs = {"options": options}
        if json_mode:
            kwargs["format"] = "json"
        if model.startswith("qwen3:"):
            kwargs["think"] = False
        try:
            res = ollama.chat(model=model, messages=messages, **kwargs)
        except TypeError:
            kwargs.pop("think", None)
            res = ollama.chat(model=model, messages=messages, **kwargs)
        return res.message.content or ""

    def _chat_ollama_http(self, messages, temp, tokens, json_mode, model):
        """Timeout-guarded HTTP call (the ollama pkg has no timeout kwarg)."""
        payload = {
            "model": model, "messages": messages, "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": tokens,
                **({"num_gpu": 0} if not self.is_judge else {}),
            },
        }
        if json_mode:
            payload["format"] = "json"
        try:
            resp = requests.post(
                "http://localhost:11434/api/chat", json=payload, timeout=OLLAMA_TIMEOUT_S,
            )
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "") or ""
        except Exception as e:
            logger.warning("Ollama HTTP call failed (%s); retrying via SDK.", e)
            return self._chat_ollama(messages, temp, tokens, json_mode, model)

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """Execute chat completion across Hugging Face or Ollama.

        Provider fallback is scoped to this call; the instance state is never
        mutated, so a judge client cannot silently degrade mid-benchmark.
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        ollama_target = self.model if self.provider == "ollama" else self.ollama_model

        if self.provider == "transformers":
            # Never silently fall back: mixed judges would invalidate the run.
            return self._chat_transformers(messages, temp, tokens, json_mode)

        if self.provider == "nvidia":
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": tokens,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            headers = {
                "Authorization": f"Bearer {self.nvidia_key}",
                "Content-Type": "application/json",
            }
            try:
                resp = requests.post(self.nvidia_url, json=payload, headers=headers, timeout=60.0)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"] or ""
            except Exception as e:
                logger.warning("NVIDIA NIM call failed (%s); falling back to Ollama.", e)

        if self.provider == "huggingface":
            for attempt in range(2):
                try:
                    res = self._hf_client.chat.completions.create(
                        model=self.model, messages=messages,
                        max_tokens=tokens, temperature=temp,
                    )
                    return res.choices[0].message.content or ""
                except Exception as e:
                    err = str(e)
                    logger.warning("HF attempt %d failed: %s", attempt + 1, e)
                    if "402" in err or "Payment Required" in err or "depleted" in err:
                        break
                    if attempt == 0:
                        time.sleep(1.0)
            logger.warning("Falling back to Ollama (%s) for this call only.", ollama_target)

        return self._chat_ollama_http(messages, temp, tokens, json_mode, ollama_target)
