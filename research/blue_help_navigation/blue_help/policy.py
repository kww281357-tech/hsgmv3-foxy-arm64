"""Instruction-conditioned visual policy boundary: never accepts an Episode/goal."""
from __future__ import annotations
import base64
from dataclasses import dataclass
import io
import json
from typing import Callable
from .contracts import SensorFrame
from .help_offer import OFFER_TEXT
from .prompt_catalog import load_prompts
from .providers.go_contract import prepare_request, parse_response, session_id


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class PolicyContext:
    public: dict
    image_data_uri: str


@dataclass(frozen=True)
class Decision:
    action: str
    offer_choice: str | None = None
    offer_id: str | None = None
    question: str | None = None
    uncertainty: str = "low"
    short_reason: str = ""


def rgb_to_data_uri(rgb) -> str:
    if isinstance(rgb, str) and rgb.startswith("data:image/"):
        return rgb  # Synthetic tests may pass a pre-encoded fixture.
    from PIL import Image
    image = Image.fromarray(rgb).convert("RGB")
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def build_policy_context(*, instruction: str, frame: SensorFrame, history: list[dict],
                         interaction_state: dict, active_offer: dict | None,
                         last_delivered_answer: str | None, is_lost: bool,
                         lost_reasons: tuple[str, ...] | list[str], dummy_aliases: dict | None = None,
                         answer_metadata: dict | None = None) -> PolicyContext:
    """Explicit field allowlists prevent accidental goal/path/pose serialization.

    The caller must not place private data inside free-text instruction/history.
    Here history is re-built from actual action/displacement/collision fields only.
    """
    history_public = [{key: row[key] for key in ("step", "action", "collision", "displacement_m", "yaw_delta_rad") if key in row}
                      for row in history[-12:]]
    remaining = interaction_state["remaining_delivered_answers"]
    candidates = []
    aliases = dummy_aliases if dummy_aliases is not None else {}
    for row in frame.dummy_candidates:
        # The environment registry is privileged; occluded actor IDs/distances are not observations.
        if row.get("visible") is not True:
            continue
        if row["dummy_id"] not in aliases:
            aliases[row["dummy_id"]] = f"DUMMY_{len(aliases)+1:03d}"
        candidates.append({"dummy_id": aliases[row["dummy_id"]],
                           "spatial_eligible": row.get("spatial_eligible") is True,
                           "horizontal_distance_m": row["horizontal_distance_m"],
                           "interaction_radius_m": row["interaction_radius_m"]})
    offer = None
    if active_offer and remaining > 0 and active_offer["dummy_id"] in aliases:
        offer = {key: active_offer[key] for key in ("offer_id", "expires_step")}
        offer["dummy_id"] = aliases[active_offer["dummy_id"]]
        offer.update(text=OFFER_TEXT, options=["accept", "decline", "ignore"])
    specs = {}
    for action,spec in getattr(frame,"action_specs",{}).items():
        if action not in frame.legal_actions:continue
        clean={key:value for key,value in spec.items() if key in ("relative_direction","requested_distance_m","turn_degrees")}
        if "turn_degrees" not in clean and "degrees" in spec:
            clean["turn_degrees"]=spec["degrees"]
        specs[action]=clean
    meta = {key: answer_metadata[key] for key in ("question_step", "delivery_step") if key in answer_metadata} if answer_metadata else None
    public = {"original_instruction": instruction, "observation_step": frame.step,
              "legal_candidate_ids": list(frame.legal_actions), "action_specs": specs, "executed_history": history_public,
              "interaction_candidates": candidates, "remaining_delivered_answers": remaining,
              "active_offer": offer, "last_delivered_answer": last_delivered_answer,"last_answer_metadata":meta,
              "observed_uncertainty": {"is_lost": bool(is_lost), "reasons": list(lost_reasons)},
              "evidence_ids": [f"OBS_{frame.step}"] + [f"ACTION_{row['step']}" for row in history_public if "step" in row]}
    return PolicyContext(public, rgb_to_data_uri(frame.rgb))


def _english_text(value, *, optional=False, limit=1000):
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise PolicyError("Invalid policy text")
    # A narrow script check, not a claim to classify every possible language.
    if any("\u3400" <= ch <= "\u9fff" for ch in value):
        raise PolicyError("English runtime output is required")
    return value


def parse_decision(content: str, context: PolicyContext) -> Decision:
    if not isinstance(content, str) or len(content) > 20_000:
        raise PolicyError("Invalid decision content")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise PolicyError("Duplicate decision field")
            result[key] = value
        return result
    try:
        data = json.loads(content, object_pairs_hook=unique)
    except (json.JSONDecodeError, RecursionError):
        raise PolicyError("Decision must be strict JSON") from None
    expected = {"intent", "offer_choice", "offer_id", "candidate_id", "uncertainty", "evidence_ids", "short_reason", "question"}
    if not isinstance(data, dict) or set(data) != expected:
        raise PolicyError("Decision schema mismatch")
    intent, choice = data["intent"], data["offer_choice"]
    if not isinstance(intent, str) or intent not in {"NAVIGATE", "REQUEST_HELP", "STOP", "WAIT"}:
        raise PolicyError("Invalid intent")
    if choice is not None and (not isinstance(choice, str) or choice not in {"accept", "decline", "ignore"}):
        raise PolicyError("Invalid offer choice")
    active = context.public["active_offer"]
    if choice is not None:
        if active is None or data["offer_id"] != active["offer_id"]:
            raise PolicyError("Unknown or stale offer")
    elif data["offer_id"] is not None:
        raise PolicyError("Offer ID without a choice")
    uncertainty = data["uncertainty"]
    if not isinstance(uncertainty, str) or uncertainty not in {"low", "medium", "high"}:
        raise PolicyError("Invalid uncertainty")
    evidence = data["evidence_ids"]
    if (not isinstance(evidence, list) or any(not isinstance(x, str) for x in evidence)
            or not set(evidence) <= set(context.public["evidence_ids"])):
        raise PolicyError("Invented evidence ID")
    reason = _english_text(data["short_reason"], limit=600)
    question = _english_text(data["question"], optional=True)
    if intent == "REQUEST_HELP":
        if choice != "accept" or data["candidate_id"] is not None or question is None:
            raise PolicyError("Help requires acceptance, a question, and no movement")
        if uncertainty == "low" and not context.public["observed_uncertainty"]["is_lost"]:
            raise PolicyError("Do not ask merely because an offer is present")
        action = "wait"
    else:
        if choice == "accept" or question is not None:
            raise PolicyError("Acceptance/question requires REQUEST_HELP")
        if intent == "NAVIGATE":
            action = data["candidate_id"]
            if action not in ("move_forward", "turn_left", "turn_right"):
                raise PolicyError("NAVIGATE requires a movement candidate")
        else:
            if data["candidate_id"] is not None:
                raise PolicyError("Non-movement intent cannot select a candidate")
            action = "stop" if intent == "STOP" else "wait"
    if action not in context.public["legal_candidate_ids"]:
        raise PolicyError("Illegal current action")
    return Decision(action, choice, data["offer_id"], question, uncertainty, reason)


class VisualLanguagePolicy:
    """Actual multimodal request/decision adapter, with an injected approved client.

    No default network transport or credential loading. Unit tests use fake bytes.
    External transport authorization/rate/cost enforcement belongs to the runner's
    explicitly configured future provider integration, not a hidden fallback.
    """
    def __init__(self, transport: Callable | None = None):
        self.transport = transport
        self._session = None
        self.api_attempts = 0
        self.usage = []

    def reset(self, experiment_id: str, episode_id: str, arm: str):
        self._session = session_id(experiment_id, episode_id, arm)
        self.api_attempts = 0
        self.usage = []

    def decide(self, context: PolicyContext) -> Decision:
        if self.transport is None:
            raise RuntimeError("No authorized model transport configured; no request sent")
        if self._session is None:
            raise RuntimeError("Policy episode was not reset")
        request = prepare_request([
            {"role": "system", "content": load_prompts()["policy"]},
            {"role": "user", "content": [
                {"type": "text", "text": json.dumps(context.public, ensure_ascii=False, allow_nan=False)},
                {"type": "image_url", "image_url": {"url": context.image_data_uri}}]}], session=self._session)
        self.api_attempts += 1
        raw = self.transport(request)
        response = parse_response(raw)
        self.usage.append(response.usage)
        return parse_decision(response.content, context)


class ReplayPolicy:
    """Deterministic test/interaction-smoke actions, NOT a language-navigation model."""
    def __init__(self, decisions):
        self.decisions = tuple(decisions)
        self.index = 0

    def reset(self, experiment_id, episode_id, arm):
        self.index = 0

    def decide(self, context):
        if self.index >= len(self.decisions):
            return Decision("stop", short_reason="Replay completed; not a model judgment.")
        decision = self.decisions[self.index]
        self.index += 1
        return decision
