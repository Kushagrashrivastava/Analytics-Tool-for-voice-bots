"""OpenAI-compatible client for Groq (graded default) and same-model dev providers, with retries and caching."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import openai
from openai import OpenAI
from pydantic import ValidationError
from tenacity import RetryCallState, retry, retry_if_exception, stop_after_attempt

from .prompts import PROMPT_VERSION, SYSTEM_PROMPT
from .schemas import Evidence

log = logging.getLogger("aster_vale.llm")

PROVIDERS = {
    # name: (base_url, api-key env var, default model id)
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", "gpt-oss-120b"),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY", "openai/gpt-oss-120b"),
}


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "groq"
    model: str | None = None
    temperature: float = 0.0
    seed: int = 7
    reasoning_effort: str = "medium"
    max_completion_tokens: int = 6000
    timeout_s: float = 90.0
    max_attempts: int = 5
    cache_path: str | None = ".cache/llm_cache.sqlite"

    @property
    def model_id(self) -> str:
        return self.model or PROVIDERS[self.provider][2]


@dataclass
class LLMResult:
    evidence: Evidence
    prompt_tokens: int
    completion_tokens: int
    cached: bool
    latency_s: float


def _strict_schema() -> dict:
    """Pydantic schema -> self-contained strict JSON schema (refs inlined, no titles/defaults)."""
    raw = Evidence.model_json_schema()
    defs = raw.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            return {k: walk(v) for k, v in node.items() if k not in ("title", "default")}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(raw)


EVIDENCE_SCHEMA = _strict_schema()


class _Cache:
    def __init__(self, path: str | None):
        self._lock = threading.Lock()
        self._db = None
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")

    def get(self, key: str) -> str | None:
        if not self._db:
            return None
        with self._lock:
            row = self._db.execute("SELECT v FROM c WHERE k=?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str) -> None:
        if self._db:
            with self._lock:
                self._db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (key, value))
                self._db.commit()


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
                        openai.InternalServerError, ValidationError, json.JSONDecodeError)):
        return True
    return isinstance(exc, openai.APIStatusError) and exc.status_code in (408, 409, 425, 498, 500, 502, 503)


def _wait(state: RetryCallState) -> float:
    exc = state.outcome.exception() if state.outcome else None
    if isinstance(exc, openai.APIStatusError):
        ra = exc.response.headers.get("retry-after")
        if ra:
            try:
                return min(float(ra) + 0.5, 90.0)
            except ValueError:
                pass
    return min(2.0 * 2 ** (state.attempt_number - 1), 60.0)


class EvidenceExtractor:
    def __init__(self, cfg: LLMConfig, api_key: str | None = None):
        if cfg.provider not in PROVIDERS:
            raise ValueError(f"unknown provider {cfg.provider!r}; choose from {list(PROVIDERS)}")
        base_url, key_env, _ = PROVIDERS[cfg.provider]
        key = api_key or os.environ.get(key_env)
        if not key:
            raise RuntimeError(f"{key_env} is not set")
        self.cfg = cfg
        self.client = OpenAI(base_url=base_url, api_key=key, timeout=cfg.timeout_s, max_retries=0)
        self.cache = _Cache(cfg.cache_path)
        self._strict = True

    def _key(self, user_msg: str) -> str:
        h = hashlib.sha256()
        for part in (self.cfg.provider, self.cfg.model_id, PROMPT_VERSION, self.cfg.reasoning_effort, user_msg):
            h.update(part.encode())
            h.update(b"\0")
        return h.hexdigest()

    def extract(self, user_msg: str) -> LLMResult:
        key = self._key(user_msg)
        hit = self.cache.get(key)
        if hit:
            return LLMResult(Evidence.model_validate_json(hit), 0, 0, True, 0.0)

        @retry(stop=stop_after_attempt(self.cfg.max_attempts), wait=_wait, retry=retry_if_exception(_retryable),
               reraise=True, before_sleep=lambda s: log.warning("retry %s: %r", s.attempt_number,
                                                                s.outcome.exception()))
        def call() -> LLMResult:
            t0 = time.monotonic()
            resp = self._create(user_msg)
            content = resp.choices[0].message.content or ""
            evidence = Evidence.model_validate_json(content)
            u = resp.usage
            self.cache.put(key, evidence.model_dump_json())
            return LLMResult(evidence, getattr(u, "prompt_tokens", 0) or 0,
                             getattr(u, "completion_tokens", 0) or 0, False, time.monotonic() - t0)

        return call()

    def _create(self, user_msg: str):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}]
        if self._strict:
            fmt = {"type": "json_schema",
                   "json_schema": {"name": "evidence", "strict": True, "schema": EVIDENCE_SCHEMA}}
        else:
            schema_hint = "\n\nJSON schema:\n" + json.dumps(EVIDENCE_SCHEMA, separators=(",", ":"))
            messages[0] = {"role": "system", "content": SYSTEM_PROMPT + schema_hint}
            fmt = {"type": "json_object"}
        extra = {"reasoning_effort": self.cfg.reasoning_effort}
        if self.cfg.provider == "groq":
            extra["include_reasoning"] = False
        try:
            return self.client.chat.completions.create(
                model=self.cfg.model_id, messages=messages, temperature=self.cfg.temperature,
                seed=self.cfg.seed, max_completion_tokens=self.cfg.max_completion_tokens,
                response_format=fmt, extra_body=extra,
            )
        except openai.BadRequestError as e:
            if self._strict and "schema" in str(e).lower():
                log.warning("strict json_schema rejected by provider; falling back to json_object: %s", e)
                self._strict = False
                return self._create(user_msg)
            raise
