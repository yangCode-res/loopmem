"""Small OpenAI-compatible JSON client extracted for the runtime adapter."""
from __future__ import annotations

from dataclasses import dataclass
import http.client
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_MODEL = "loopmem-controller"

def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()

def _extract_json_object(text: str) -> dict[str, Any]:
    candidate = _strip_code_fences(text)
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Some OpenAI-compatible providers append a short explanation or even a
    # second draft after an otherwise complete JSON object.  A greedy
    # ``\{.*\}`` match turns that into ``Extra data``.  Decode from each object
    # start and stop at the first complete dictionary instead; the trailing
    # provider chatter never enters the Controller wire response.
    decoder = json.JSONDecoder()
    for start in (match.start() for match in re.finditer(r"\{", candidate)):
        try:
            parsed, _ = decoder.raw_decode(candidate[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    match = re.search(r"\{.*\}", candidate, re.S)
    if match:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("LLM response did not contain a valid JSON object")

class OpenAIChatJSONClient:
    """Very small OpenAI-compatible JSON caller built on stdlib HTTP."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 60.0,
        max_tokens: int = 2048,
        max_attempts: int = 6,
        retry_base_seconds: float = 1.0,
        retry_max_seconds: float = 30.0,
        thinking_mode: str | None = None,
        capture_logprobs: bool = False,
        top_logprobs: int = 0,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self.thinking_mode = thinking_mode
        self.capture_logprobs = bool(capture_logprobs)
        self.top_logprobs = max(0, min(20, int(top_logprobs)))
        self.temperature = float(temperature)
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be a finite non-negative number")
        self.seed = None if seed is None else int(seed)
        self.request_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.last_completion_text = ""
        self.last_completion_token_logprobs: list[dict[str, Any]] = []
        self.last_completion_logprob: float | None = None

    @classmethod
    def from_env(cls, *, model: str | None = None, base_url: str | None = None) -> "OpenAIChatJSONClient" | None:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LOOPMEM_OPENAI_API_KEY")
        if not api_key:
            return None
        return cls(
            api_key=api_key,
            model=model or os.getenv("LOOPMEM_OPENAI_MODEL") or DEFAULT_MODEL,
            base_url=base_url or os.getenv("LOOPMEM_OPENAI_BASE_URL") or DEFAULT_BASE_URL,
            timeout_seconds=float(os.getenv("LOOPMEM_OPENAI_TIMEOUT_SECONDS", "60")),
            max_tokens=int(os.getenv("LOOPMEM_OPENAI_MAX_TOKENS", "2048")),
            capture_logprobs=os.getenv("LOOPMEM_CONTROLLER_CAPTURE_LOGPROBS", "").strip().lower()
            in {"1", "true", "yes", "on"},
            top_logprobs=int(os.getenv("LOOPMEM_CONTROLLER_TOP_LOGPROBS", "0")),
        )

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.last_completion_text = ""
        self.last_completion_token_logprobs = []
        self.last_completion_logprob = None
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.capture_logprobs:
            payload["logprobs"] = True
            if self.top_logprobs:
                payload["top_logprobs"] = self.top_logprobs
        thinking_mode = (
            self.thinking_mode
            if self.thinking_mode is not None
            else os.getenv("LOOPMEM_CONTROLLER_THINKING_MODE", "")
        ).strip().lower()
        if thinking_mode in {"enabled", "disabled"}:
            # DeepSeek defaults to high-effort thinking. Structured Controller
            # calls need the JSON content, not a long hidden reasoning prelude;
            # disabling thinking prevents the completion budget from being
            # exhausted before any JSON is emitted.
            payload["thinking"] = {"type": thinking_mode}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        body = ""
        for attempt in range(self.max_attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:  # pragma: no cover - network path
                detail = exc.read().decode("utf-8", errors="ignore")
                retryable = exc.code in {408, 409, 429, 500, 502, 503, 504}
                if retryable and attempt + 1 < self.max_attempts:
                    retry_after = 0.0
                    if exc.headers:
                        try:
                            retry_after = float(exc.headers.get("Retry-After") or 0)
                        except (TypeError, ValueError):
                            retry_after = 0.0
                    delay = min(
                        self.retry_max_seconds,
                        max(retry_after, self.retry_base_seconds * (2**attempt)),
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"OpenAI request failed with status {exc.code}: {detail}") from exc
            except (urllib.error.URLError, http.client.IncompleteRead, TimeoutError, ConnectionError) as exc:
                if attempt + 1 == self.max_attempts:  # pragma: no cover - network path
                    detail = getattr(exc, "reason", str(exc))
                    raise RuntimeError(
                        f"OpenAI request failed after {self.max_attempts} attempts: {detail}"
                    ) from exc
                time.sleep(
                    min(
                        self.retry_max_seconds,
                        self.retry_base_seconds * (2**attempt),
                    )
                )

        data = json.loads(body)
        self.request_count += 1
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("OpenAI response did not contain any choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("OpenAI response did not contain assistant text content")
        self.last_completion_text = content
        if self.capture_logprobs:
            raw_tokens = ((choices[0].get("logprobs") or {}).get("content") or [])
            normalized_tokens: list[dict[str, Any]] = []
            sequence_logprob = 0.0
            for item in raw_tokens:
                if not isinstance(item, dict):
                    continue
                try:
                    logprob = float(item.get("logprob"))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(logprob):
                    continue
                normalized = {
                    "token": str(item.get("token") or ""),
                    "logprob": logprob,
                }
                if isinstance(item.get("bytes"), list):
                    normalized["bytes"] = item["bytes"]
                if isinstance(item.get("top_logprobs"), list):
                    normalized["top_logprobs"] = item["top_logprobs"]
                normalized_tokens.append(normalized)
                sequence_logprob += logprob
            self.last_completion_token_logprobs = normalized_tokens
            if normalized_tokens:
                self.last_completion_logprob = sequence_logprob
        return _extract_json_object(content)
