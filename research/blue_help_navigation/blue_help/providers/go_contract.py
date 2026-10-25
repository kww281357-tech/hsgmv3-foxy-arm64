"""Pure Go request/response validation. No network or credential lookup.

The explicitly constructed go_transport client is separate. Building or parsing
these values does not imply provider authorization, vision support or paid use.
"""
from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

MODEL = "deepseek-v4.1-flash"
ENDPOINT = "https://opencode.ai/inference/go/openai/v1/chat/completions"
USER_AGENT = "r2rce-navigation-research-offline-contract/0.1"
MAX_REQUEST_BYTES = 8 * 1024 * 1024  # Our conservative cap, not a provider limit.
MAX_RESPONSE_BYTES = 1024 * 1024


class ContractError(ValueError):
    """Messages deliberately omit user text, images, response bodies and secrets."""


@dataclass(frozen=True)
class PreparedRequest:
    endpoint: str
    headers: dict[str, str] = field(repr=False)
    body: bytes = field(repr=False)
    image_count: int

    def audit_metadata(self) -> dict[str, Any]:
        """No request body, image, instruction, header or credential in logs."""
        return {"model": MODEL, "endpoint": self.endpoint,
                "request_bytes": len(self.body), "image_count": self.image_count,
                "body_sha256": hashlib.sha256(self.body).hexdigest(),
                "live_request_performed": False}


@dataclass(frozen=True)
class ParsedResponse:
    content: str = field(repr=False)
    model: str
    usage: dict[str, Any] = field(default_factory=dict)


def session_id(experiment_id: str, episode_id: str, arm: str, attempt: int = 0) -> str:
    """Stable within one episode/arm attempt; no history sharing across arms."""
    if not all(isinstance(v, str) and v and len(v) <= 128
               for v in (experiment_id, episode_id, arm)):
        raise ContractError("Invalid session components")
    if arm not in {"no_ask", "max3_ask"} or type(attempt) is not int or attempt < 0:
        raise ContractError("Invalid arm or attempt")
    raw = json.dumps([experiment_id, episode_id, arm, attempt], separators=(",", ":"))
    return "r2rce-" + hashlib.sha256(raw.encode()).hexdigest()[:32]


def _image(url: Any) -> None:
    if not isinstance(url, str) or len(url) > MAX_REQUEST_BYTES:
        raise ContractError("Invalid or oversized image")
    match = re.fullmatch(r"data:image/(png|jpeg);base64,([A-Za-z0-9+/=]+)", url)
    if not match:
        raise ContractError("Only inline PNG/JPEG images are accepted; no remote fetch")
    try:
        data = base64.b64decode(match[2], validate=True)
    except (binascii.Error, ValueError):
        raise ContractError("Invalid image base64") from None
    # Header check only: renderer must still provide a complete decodable image.
    good = data.startswith(b"\x89PNG\r\n\x1a\n") if match[1] == "png" else data.startswith(b"\xff\xd8\xff")
    if not good:
        raise ContractError("Image header does not match declared media type")


def prepare_request(messages: list[dict[str, Any]], *, session: str,
                    max_tokens: int = 2048) -> PreparedRequest:
    """Build bytes only. No key is read, stored, accepted, or sent.

    Action schema and information-privilege validation belong to the restored
    planner, not this provider-level module. Never supply complete episode JSON.
    """
    if not isinstance(session, str) or not re.fullmatch(r"r2rce-[a-f0-9]{32}", session):
        raise ContractError("Invalid session identifier")
    if type(max_tokens) is not int or not 128 <= max_tokens <= 8192:
        raise ContractError("max_tokens outside offline safety bounds")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 64:
        raise ContractError("Invalid message list")
    accounted_bytes = 1024

    def account(value: str) -> None:
        # Bound cumulative allocation BEFORE deepcopy/whole-request encoding.
        # Per-string serialization is bounded by the individual field caps.
        nonlocal accounted_bytes
        try:
            accounted_bytes += len(json.dumps(value, ensure_ascii=False).encode("utf-8")) + 128
        except UnicodeError:
            raise ContractError("Invalid Unicode in request") from None
        if accounted_bytes > MAX_REQUEST_BYTES:
            raise ContractError("Request exceeds cumulative offline safety bound")

    image_count = 0
    final_user_images = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ContractError("Only role/content message fields are accepted")
        role, content = message["role"], message["content"]
        if not isinstance(role, str) or role not in {"system", "user", "assistant"}:
            raise ContractError("Unsupported message role")
        if isinstance(content, str):
            if not content.strip() or len(content) > 100_000:
                raise ContractError("Empty or oversized text")
            account(content)
            continue
        if role != "user" or not isinstance(content, list) or not 1 <= len(content) <= 32:
            raise ContractError("Only user messages may contain image blocks")
        for block in content:
            if not isinstance(block, dict):
                raise ContractError("Invalid content block")
            if set(block) == {"type", "text"} and block["type"] == "text":
                if not isinstance(block["text"], str) or not block["text"].strip() or len(block["text"]) > 100_000:
                    raise ContractError("Empty or oversized block text")
                account(block["text"])
            elif set(block) == {"type", "image_url"} and block["type"] == "image_url":
                payload = block["image_url"]
                if not isinstance(payload, dict) or set(payload) not in ({"url"}, {"url", "detail"}):
                    raise ContractError("Invalid image block fields")
                if "detail" in payload and (not isinstance(payload["detail"], str) or payload["detail"] not in {"low", "high", "auto"}):
                    raise ContractError("Unsupported image detail")
                if not isinstance(payload["url"], str) or len(payload["url"]) > MAX_REQUEST_BYTES:
                    raise ContractError("Invalid or oversized image")
                account(payload["url"])
                _image(payload["url"])
                image_count += 1
                if index == len(messages) - 1:
                    final_user_images += 1
            else:
                raise ContractError("Unsupported content block")
    if messages[-1]["role"] != "user" or not final_user_images:
        raise ContractError("Navigation requires a current image in the final user message")
    body = json.dumps({"model": MODEL, "messages": copy.deepcopy(messages),
                       "stream": False, "max_tokens": max_tokens},
                      ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    if len(body) > MAX_REQUEST_BYTES:
        raise ContractError("Request exceeds offline safety bound")
    # No Authorization header; an approved future transport must inject a fresh
    # secure credential. Do not impersonate OpenCode or another coding harness.
    return PreparedRequest(ENDPOINT, {"Content-Type": "application/json",
                           "User-Agent": USER_AGENT, "x-opencode-session": session},
                           body, image_count)


def _finite_number(value: Any) -> bool:
    if type(value) is int:
        return 0 <= value <= 10 ** 15
    return type(value) is float and 0 <= value <= 10 ** 15 and math.isfinite(value)


def _bounded_int(value: str) -> int:
    if len(value.lstrip("-")) > 16:
        raise ContractError("Oversized integer in response")
    return int(value)


def _invalid_constant(_: str) -> None:
    raise ContractError("Nonfinite JSON value")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("Duplicate response JSON field")
        result[key] = value
    return result


def parse_response(raw: bytes) -> ParsedResponse:
    """Parse a captured/mock Chat Completions body, never execute its contents.

    Content remains untrusted. Existing v1.1 legal-action/JSON validation MUST run
    after this function. Reasoning text, errors and arbitrary fields are not logged.
    """
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= MAX_RESPONSE_BYTES:
        raise ContractError("Invalid response size or type")
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_object,
                             parse_int=_bounded_int, parse_constant=_invalid_constant)
    except ContractError:
        raise
    except (UnicodeError, ValueError, RecursionError):
        raise ContractError("Response is not valid JSON") from None
    if not isinstance(payload, dict) or payload.get("error") is not None:
        raise ContractError("Provider response failed")
    if payload.get("model") != MODEL:
        raise ContractError("Unexpected model; no fallback allowed")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ContractError("Exactly one response choice required")
    choice = choices[0]
    message = choice.get("message")
    if type(choice.get("index")) is not int or choice["index"] != 0 or choice.get("finish_reason") != "stop":
        raise ContractError("Incomplete, refused, or unexpected completion")
    if (not isinstance(message, dict) or message.get("role") != "assistant"
            or message.get("tool_calls") or message.get("function_call") or message.get("refusal")):
        raise ContractError("Tool calls or refusals cannot become navigation decisions")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip() or len(content) > 50_000:
        raise ContractError("Missing or oversized decision content")
    try:
        content.encode("utf-8")
    except UnicodeError:
        raise ContractError("Invalid Unicode in decision content") from None
    usage = payload.get("usage", {})
    if not isinstance(usage, dict) or not {"cost", "opencode"} <= set(usage):
        raise ContractError("Missing explicit billing metadata; do not continue")
    safe_usage: dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
        if key in usage:
            if not _finite_number(usage[key]):
                raise ContractError("Invalid numeric usage field")
            safe_usage[key] = usage[key]
    provider = usage.get("opencode", {})
    if not isinstance(provider, dict) or not {"billing_source", "allowance"} <= set(provider):
        raise ContractError("Missing explicit Go allowance metadata; do not continue")
    if provider:
        billing = provider.get("billing_source")
        if not isinstance(billing, str) or billing not in {"go", "go-plus"}:
            raise ContractError("Unexpected billing source; stop rather than fall back")
        safe_usage["billing_source"] = billing
        for key in ("allowance", "list_cost"):
            if key in provider:
                if not _finite_number(provider[key]):
                    raise ContractError("Invalid provider cost field")
                safe_usage[key] = provider[key]
    if safe_usage.get("cost", 0) > 0:
        raise ContractError("Balance charge reported; stop rather than continue")
    # A response-time check cannot prevent an already charged request. Disable
    # Extra Usage in the provider console and approve caps BEFORE any live run.
    return ParsedResponse(content, MODEL, safe_usage)


def send_live(*args: Any, **kwargs: Any) -> None:
    """Deliberately impossible to enable via config/env; no network implementation."""
    raise RuntimeError("Live inference unavailable: source, provider-use and spend approval pending")
