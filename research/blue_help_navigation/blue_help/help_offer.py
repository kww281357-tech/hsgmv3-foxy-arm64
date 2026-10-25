"""Independent offline offer/accept/decline/ignore state machine.

No simulator, model transport or credentials. The trusted environment must supply
fresh spatial eligibility and validated answer/pose facts, never the model itself.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import uuid
from .interaction_range import RangeConfig


MAX_ANSWER_CHARACTERS = 8192

OFFER_TEXT = "Do you need any help? You can accept, decline, or ignore this offer and continue navigating."


@dataclass(frozen=True)
class OfferConfig:
    radius_m: float = 1.0
    exit_hysteresis_m: float = 0.2
    cooldown_steps: int = 10
    offer_timeout_steps: int = 5
    answer_timeout_steps: int = 30
    max_delivered_answers: int = 3

    def range_config(self, max_floor_height_delta_m: float = 0.5) -> RangeConfig:
        """Single source of truth for the environment's spatial range query."""
        return RangeConfig(self.radius_m, max_floor_height_delta_m)

    def __post_init__(self):
        for value in (self.radius_m, self.exit_hysteresis_m):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Positive finite radius and hysteresis required")
        for value in (self.cooldown_steps, self.offer_timeout_steps, self.answer_timeout_steps):
            if type(value) is not int or value <= 0:
                raise ValueError("Positive integer step settings required")
        if type(self.max_delivered_answers) is not int or not 0 <= self.max_delivered_answers <= 3:
            raise ValueError("Delivered-answer budget must be 0..3")


class HelpOfferMachine:
    def __init__(self, config: OfferConfig = OfferConfig()):
        self.config = config
        self._lock = threading.Lock()
        self._instance = uuid.uuid4().hex
        self._epoch = -1
        self.reset("unstarted")

    def reset(self, episode_id: str) -> None:
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("episode_id required")
        with self._lock:
            self._epoch += 1
            self.episode_id = episode_id
            self.delivered = 0
            self.attempts = 0
            self._sequence = 0
            self._step = -1
            self._next_offer_step = 0
            self._offered_this_visit = {}
            self._offer = None
            self._pending = None
            self.events = []

    def _advance(self, step: int) -> None:
        if type(step) is not int or step < 0 or step < self._step:
            raise ValueError("Steps must be nonnegative and monotonic")
        self._step = step
        if self._offer is not None and step >= self._offer["expires_step"]:
            self._event("ignored_timeout", self._offer["dummy_id"])
            self._offer = None
        if self._pending is not None and step >= self._pending["expires_step"]:
            self._event("answer_timeout", self._pending["dummy_id"], self._pending["request_id"])
            self._pending = None

    def _event(self, kind: str, dummy_id: str = "", request_id: str = "") -> None:
        self.events.append({"event": kind, "step": self._step,
                            "dummy_id": dummy_id, "request_id": request_id,
                            "delivered_answers": self.delivered, "attempts": self.attempts})

    def observe(self, candidates: list[dict], *, step: int) -> dict | None:
        """Candidates are freshly checked range/LOS/floor/visible/reachable facts.

        Use spatial_eligible from evaluate_interaction, NOT its budget/cooldown
        qualified eligible flag. This machine owns offer cooldown and choice state.

        Absent/occluded people do not imply leaving the disk. Rearm only on a
        measured foot-center distance > radius + hysteresis. Nearest wins ties
        deterministically by dummy_id. No raw image or world coordinates exposed.
        """
        validated = []
        seen = set()
        for item in candidates:
            if not isinstance(item, dict):
                raise ValueError("Invalid candidate")
            identity, distance, eligible = item.get("dummy_id"), item.get("horizontal_distance_m"), item.get("spatial_eligible")
            if (not isinstance(identity, str) or not identity or identity in seen
                    or type(distance) not in (int, float) or not math.isfinite(distance) or distance < 0
                    or type(eligible) is not bool):
                raise ValueError("Invalid candidate evidence")
            if item.get("interaction_radius_m") != self.config.radius_m:
                raise ValueError("Environment range and offer configuration must match")
            seen.add(identity)
            validated.append((identity, distance, eligible))
        with self._lock:
            self._advance(step)
            for identity, distance, _ in validated:
                if distance > self.config.radius_m + self.config.exit_hysteresis_m:
                    self._offered_this_visit[identity] = False
            # Losing eligibility cancels the offer; returning without a measured
            # exit does not immediately issue another one.
            eligible_ids = {identity for identity, distance, ok in validated
                            if ok and distance <= self.config.radius_m}
            if self._offer and self._offer["dummy_id"] not in eligible_ids:
                self._event("offer_cancelled_not_eligible", self._offer["dummy_id"])
                self._offer = None
            if (self._offer or self._pending or self.delivered >= self.config.max_delivered_answers
                    or step < self._next_offer_step):
                return None
            choices = [(distance, identity) for identity, distance, ok in validated
                       if ok and distance <= self.config.radius_m
                       and not self._offered_this_visit.get(identity, False)]
            if not choices:
                return None
            _, identity = min(choices)
            self._sequence += 1
            offer_id = f"offer-{self._instance}-{self._epoch}-{self._sequence}"
            self._offer = {"offer_id": offer_id, "dummy_id": identity,
                           "expires_step": step + self.config.offer_timeout_steps}
            self._offered_this_visit[identity] = True
            self._next_offer_step = step + self.config.cooldown_steps
            self._event("offered", identity)
            return {**self._offer, "text": OFFER_TEXT,
                    "options": ["accept", "decline", "ignore"],
                    "remaining_delivered_answers": self.config.max_delivered_answers - self.delivered,
                    "route_information_in_offer": False}

    def choose(self, offer_id: str, choice: str, *, step: int,
               spatial_eligible_now: bool) -> dict:
        """No API call is made here. Accept returns a request token for a future caller."""
        if not isinstance(choice, str) or choice not in {"accept", "decline", "ignore"} or type(spatial_eligible_now) is not bool:
            raise ValueError("Invalid choice or trusted eligibility")
        with self._lock:
            if self._offer is None or self._offer["offer_id"] != offer_id:
                return {"status": "stale_or_missing_offer", "budget_consumed": False}
            self._advance(step)
            if self._offer is None:
                return {"status": "stale_or_missing_offer", "budget_consumed": False}
            identity = self._offer["dummy_id"]
            self._offer = None
            if choice != "accept":
                self._event("declined" if choice == "decline" else "ignored", identity)
                return {"status": choice, "budget_consumed": False, "resume_navigation": True}
            if (not spatial_eligible_now or self._pending is not None
                    or self.delivered >= self.config.max_delivered_answers):
                self._event("accept_rejected_not_eligible", identity)
                return {"status": "not_eligible", "budget_consumed": False}
            self.attempts += 1
            request_id = f"request-{self._instance}-{self._epoch}-{self._sequence}"
            self._pending = {"request_id": request_id, "dummy_id": identity,
                             "expires_step": step + self.config.answer_timeout_steps}
            self._next_offer_step = max(self._next_offer_step, step + self.config.cooldown_steps)
            self._event("accepted", identity, request_id)
            return {"status": "accepted", "request_id": request_id,
                    "dummy_id": identity, "budget_consumed": False, "api_call_made": False}

    def deliver(self, request_id: str, answer_text: str | None, *, step: int,
                answer_validated: bool, spatial_eligible_now: bool,
                pose_reference_valid: bool) -> dict:
        """Trusted environment only. Atomic successful delivery + budget increment.

        Exactly one caller gets the accepted answer; concurrent/late duplicate
        callbacks do not re-deliver or count twice. Invalid/failed/expired delivery
        consumes no successful-answer budget; prior API cost is a separate matter.
        """
        if any(type(x) is not bool for x in (answer_validated, spatial_eligible_now, pose_reference_valid)):
            raise ValueError("Trusted validation flags must be bool")
        with self._lock:
            if self._pending is None or self._pending["request_id"] != request_id:
                return {"status": "stale_or_duplicate_answer", "budget_consumed": False}
            self._advance(step)
            if self._pending is None:
                return {"status": "stale_or_duplicate_answer", "budget_consumed": False}
            identity = self._pending["dummy_id"]
            self._pending = None
            valid = (answer_validated and spatial_eligible_now and pose_reference_valid
                     and isinstance(answer_text, str) and bool(answer_text.strip())
                     and len(answer_text) <= MAX_ANSWER_CHARACTERS and self.delivered < self.config.max_delivered_answers)
            if not valid:
                self._event("answer_failed_or_invalid", identity, request_id)
                return {"status": "answer_failed_or_invalid", "budget_consumed": False}
            self.delivered += 1
            self._event("answer_delivered", identity, request_id)
            return {"status": "answer_delivered", "answer_text": answer_text,
                    "budget_consumed": True,
                    "remaining_delivered_answers": self.config.max_delivered_answers - self.delivered,
                    "resume_navigation": True}

    def snapshot(self) -> dict:
        with self._lock:
            return {"episode_id": self.episode_id, "delivered_answers": self.delivered,
                    "question_attempts": self.attempts,
                    "remaining_delivered_answers": self.config.max_delivered_answers - self.delivered,
                    "offer_pending": self._offer is not None, "answer_pending": self._pending is not None}
