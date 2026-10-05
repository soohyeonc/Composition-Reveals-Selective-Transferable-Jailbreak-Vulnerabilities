#!/usr/bin/env python3
"""Run a resumable Ollama target transfer using frozen mutation artifacts.

The source run supplies the exact baseline mutations, chained mutations, and
persistence labels. This runner changes only the target model, then evaluates
the new target responses with the same OpenAI safety and intent judges. Reusing
the frozen transformations removes stochastic mutation drift from the
cross-target comparison and avoids paying to regenerate mutation artifacts.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

import requests

import experiment as core


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "deepseek-r1:8b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


class OllamaExperiment(core.Experiment):
    def __init__(
        self,
        *args: Any,
        ollama_urls: list[str],
        target_model: str,
        target_timeout: float,
        think: bool,
        num_ctx: int,
        target_seed: int | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not ollama_urls:
            raise ValueError("At least one Ollama URL is required")
        self.ollama_urls = [url.rstrip("/") for url in ollama_urls]
        self.target_model = target_model
        self.target_timeout = target_timeout
        self.think = think
        self.num_ctx = num_ctx
        self.target_seed = target_seed
        self.ollama_local = threading.local()
        self.ollama_url_lock = threading.Lock()
        self.ollama_url_index = 0

    def ollama_session(self) -> requests.Session:
        if not hasattr(self.ollama_local, "session"):
            self.ollama_local.session = requests.Session()
        return self.ollama_local.session

    def completion(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[str, core.Usage]:
        if model != self.target_model:
            return super().completion(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )

        if self.stop.is_set():
            raise core.BudgetExceeded("Experiment stop requested")
        self.budget.reserve()
        self.rate_limiter.wait()
        options: dict[str, Any] = {
            "num_predict": max_tokens,
            "num_ctx": self.num_ctx,
        }
        if temperature is not None:
            options["temperature"] = temperature
        request_seed = getattr(self.ollama_local, "request_seed", self.target_seed)
        if request_seed is not None:
            options["seed"] = request_seed
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": self.think,
            "keep_alive": "30m",
            "options": options,
        }
        with self.ollama_url_lock:
            ollama_url = self.ollama_urls[
                self.ollama_url_index % len(self.ollama_urls)
            ]
            self.ollama_url_index += 1
        try:
            response = self.ollama_session().post(
                f"{ollama_url}/api/chat",
                json=payload,
                timeout=(10.0, self.target_timeout),
            )
            response.raise_for_status()
            result = response.json()
        except Exception:
            self.budget.cancel()
            raise

        self.budget.settle(0.0)
        message = result.get("message") or {}
        text = message.get("content") or ""
        usage = core.Usage(
            requested_model=model,
            response_model=result.get("model"),
            prompt_tokens=int(result.get("prompt_eval_count") or 0),
            cached_tokens=0,
            completion_tokens=int(result.get("eval_count") or 0),
            reasoning_tokens=0,
            cost_usd=0.0,
            request_id=None,
            system_fingerprint=None,
        )
        return text, usage

    def task_seed(self, task_kind: str, task_id: str) -> int | None:
        if self.target_seed is None:
            return None
        material = f"{self.target_seed}\x1f{task_kind}\x1f{task_id}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")

    def call_and_save(
        self,
        *,
        table: str,
        task_kind: str,
        task_id: str,
        stage: str,
        field: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        if model != self.target_model:
            return super().call_and_save(
                table=table,
                task_kind=task_kind,
                task_id=task_id,
                stage=stage,
                field=field,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )

        self.ollama_local.request_seed = self.task_seed(task_kind, task_id)
        try:
            return super().call_and_save(
                table=table,
                task_kind=task_kind,
                task_id=task_id,
                stage=stage,
                field=field,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
        finally:
            del self.ollama_local.request_seed
