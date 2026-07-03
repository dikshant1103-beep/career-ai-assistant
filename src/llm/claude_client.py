"""Anthropic Claude client with retry + structured-JSON helper.

Uses the official ``anthropic`` SDK. Two modes are exposed:

* ``complete(prompt, ...)`` -- plain text answer.
* ``complete_json(prompt, ...)`` -- forces JSON output by prefilling ``{`` and
  parsing the response. Falls back to extracting the first valid JSON object
  from the text if needed.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.config import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)


_RETRY_EXCEPTIONS = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


class ClaudeClient:
    """Thin wrapper around the Anthropic SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        s = get_settings()
        s.assert_api_key()
        self.api_key = api_key or s.ANTHROPIC_API_KEY
        self.model = model or s.CLAUDE_MODEL
        self.max_tokens = max_tokens or s.CLAUDE_MAX_TOKENS
        self.temperature = temperature if temperature is not None else s.CLAUDE_TEMPERATURE
        self.client = anthropic.Anthropic(api_key=self.api_key)

    # ------------------------------------------------------------------ #
    @retry(
        retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
        wait=wait_exponential(multiplier=1.0, min=2, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _create(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> anthropic.types.Message:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        return self.client.messages.create(**kwargs)

    # ------------------------------------------------------------------ #
    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Single-turn completion. Returns the assistant's text content."""
        msg = self._create(
            messages=[{"role": "user", "content": prompt}],
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return _extract_text(msg)

    # ------------------------------------------------------------------ #
    def complete_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Like :meth:`complete` but coerces the answer to JSON.

        We use the assistant-prefill trick: prefilling ``{`` strongly biases
        Claude to produce a single JSON object. We then prepend the ``{`` back
        and parse. If parsing still fails, we attempt to locate the first
        balanced ``{...}`` block in the text.
        """
        msg = self._create(
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "{"},
            ],
            system=system,
            max_tokens=max_tokens,
            temperature=temperature if temperature is not None else 0.1,
        )
        raw = "{" + _extract_text(msg)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            log.warning("Direct JSON parse failed; attempting extraction.")
            extracted = _extract_json_object(raw)
            if extracted is None:
                raise ValueError(
                    "Claude did not return valid JSON. Raw output:\n" + raw[:500]
                )
            return extracted


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _extract_text(msg: anthropic.types.Message) -> str:
    """Concatenate all text blocks of a Claude response."""
    out: list[str] = []
    for block in msg.content:
        if block.type == "text":
            out.append(block.text)
    return "".join(out).strip()


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort: find the first balanced JSON object in a string."""
    match = _JSON_OBJ_RE.search(text)
    if not match:
        return None
    candidate = match.group(0)
    # walk back to a balanced brace count
    depth = 0
    end = -1
    for i, ch in enumerate(candidate):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return None
    try:
        return json.loads(candidate[:end])
    except json.JSONDecodeError:
        return None
