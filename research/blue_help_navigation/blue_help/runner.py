"""One-episode orchestration with an explicit private/public information boundary.

The runner/evaluator/oracle know the episode goal. A policy sees ONLY the allow-
listed PolicyContext, never the environment, Episode, world pose, path, or metrics.
No transport is created here. In-process separation is an auditable design boundary,
not an OS sandbox against intentionally malicious Python plugins.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
from .contracts import Episode
from .help_offer import HelpOfferMachine, OfferConfig
from .lost import LostDetector
from .oracle import RuleDirectionOracle, answer_is_stale
from .policy import Decision, build_policy_context, PolicyError
from .evaluation import evaluate_episode


@dataclass(frozen=True)
class RunConfig:
    max_steps: int = 200
    seed: int = 42
    experiment_id: str = "demo20-v1"
    success_distance_m: float = 3.0
    backend_label: str = "unverified_backend"
    policy_label: str = "unverified_policy"

    def __post_init__(self):
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 500:
            raise ValueError("max_steps must be 1..500")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if not math.isfinite(self.success_distance_m) or self.success_distance_m <= 0:
            raise ValueError("Invalid success distance")
        if not all(isinstance(v,str) and v for v in (self.experiment_id,self.backend_label,self.policy_label)):
            raise ValueError("Explicit provenance labels required")


def _eligible(frame, identity):
    return any(row.get("dummy_id") == identity and row.get("spatial_eligible") is True
               and row.get("visible") is True for row in frame.dummy_candidates)


def _safe_length(env, start, goal):
    points = env.find_safe_path(start, goal)
    if not points or any(len(p)!=3 or any(not math.isfinite(v) for v in p) for p in points):
        return None
    return sum(math.dist(a,b) for a,b in zip(points,points[1:]))


def run_episode(env, policy, episode: Episode, dummies, *, arm="max3_ask",
                config=RunConfig(), metric_distance=None, oracle_factory=RuleDirectionOracle,
                frame_sink=None):
    """Runs a single episode and returns private audit logs plus evaluator results.

    metric_distance (optional) is a trusted evaluator-only geodesic callable. By
    default the adapter's human-safe path length is used and marked accordingly.
    frame_sink(frame,event) may save privileged debug images, NEVER feed them to
    the policy. A failure is recorded as invalid, not disguised as a nav failure.
    Caller owns closing the environment so paired resets can share one simulator.
    """
    if arm not in ("no_ask", "max3_ask"):
        raise ValueError("Unknown experimental arm")
    radius = getattr(getattr(env,"config",None),"interaction_radius_m",1.0)
    machine = HelpOfferMachine(OfferConfig(radius_m=radius,max_delivered_answers=3 if arm=="max3_ask" else 0))
    machine.reset(episode.episode_id)
    detector = LostDetector()
    history, decision_log, dialogue = [], [], []
    active_offer = None
    last_answer = None
    answer_metadata = None
    dummy_aliases = {}
    previous_action = None
    previous_uncertainty = 0.0
    stopped, travelled, error = False, 0.0, None
    initial_distance = final_distance = None
    metric = metric_distance or (lambda a,b: _safe_length(env,a,b))
    frame = None
    oracle = None
    try:
        frame = env.reset(episode, dummies, seed=config.seed)
        initial_distance = metric(frame.pose.position,episode.goal_position)
        if metric_distance is not None and (type(initial_distance) not in (int,float) or not math.isfinite(initial_distance) or initial_distance<0):
            raise ValueError("Initial evaluator geodesic is unavailable; do not spend model requests")
        policy.reset(config.experiment_id, episode.episode_id, arm)
        if arm=="max3_ask":
            oracle = oracle_factory(episode.goal_position,env.find_safe_path,scene_id=episode.scene_id)
        for _ in range(config.max_steps):
            offer = machine.observe(frame.dummy_candidates,step=frame.step)
            if offer is not None:
                active_offer = offer
                dialogue.append({"step":frame.step,"role":"dummy","kind":"offer","text":offer["text"],"dummy_id":offer["dummy_id"]})
            if not machine.snapshot()["offer_pending"]:
                active_offer = None
            lost = detector.update_frame(frame,uncertainty=previous_uncertainty,previous_action=previous_action)
            context = build_policy_context(instruction=episode.instruction,frame=frame,history=history,
                interaction_state=machine.snapshot(),active_offer=active_offer,last_delivered_answer=last_answer,
                is_lost=lost.is_lost,lost_reasons=lost.reasons,dummy_aliases=dummy_aliases,answer_metadata=answer_metadata)
            if frame_sink:
                frame_sink(frame,{"phase":"observation","arm":arm,"interaction":machine.snapshot(),"active_offer":active_offer})
            decision = policy.decide(context)
            if not isinstance(decision,Decision) or decision.action not in frame.legal_actions:
                raise PolicyError("Policy selected an invalid primitive")
            if decision.offer_choice not in (None,"accept","decline","ignore"):
                raise PolicyError("Policy selected an invalid offer choice")
            if decision.uncertainty not in ("low","medium","high"):
                raise PolicyError("Invalid public uncertainty")
            decision_log.append({"step":frame.step,"decision":asdict(decision),"public_context":context.public})
            if decision.offer_choice is not None:
                if active_offer is None or decision.offer_id != active_offer["offer_id"]:
                    raise PolicyError("Stale or invented offer")
                if decision.offer_choice=="accept" and (
                    decision.action != "wait" or not isinstance(decision.question,str) or not decision.question.strip()
                    or (decision.uncertainty=="low" and not lost.is_lost)):
                    raise PolicyError("Acceptance requires uncertainty, a question, and a stationary action")
                fresh = env.observe()
                choice = machine.choose(decision.offer_id,decision.offer_choice,step=fresh.step,
                    spatial_eligible_now=_eligible(fresh,active_offer["dummy_id"]))
                if choice["status"]=="accepted":
                    dialogue.append({"step":fresh.step,"role":"vehicle","kind":"question","text":decision.question,
                                     "dummy_id":choice["dummy_id"]})
                    answer = oracle.answer(fresh.pose,fresh.step,public_history=history)
                    current = env.observe()
                    delivery = machine.deliver(choice["request_id"],answer.text if answer else None,step=current.step,
                        answer_validated=answer is not None,spatial_eligible_now=_eligible(current,choice["dummy_id"]),
                        pose_reference_valid=answer is not None and not answer_is_stale(answer,current.pose,current.step))
                    if delivery["status"]=="answer_delivered":
                        last_answer = delivery["answer_text"]
                        answer_metadata = {"question_step":answer.question_step,"delivery_step":current.step}
                        dialogue.append({"step":current.step,"role":"dummy","kind":"answer","text":last_answer,
                                         "dummy_id":choice["dummy_id"]})
                active_offer = None
            elif decision.question is not None:
                raise PolicyError("Question without explicit acceptance")
            before = frame
            frame = env.step(decision.action)
            displacement = math.dist(before.pose.position,frame.pose.position)
            travelled += displacement
            history.append({"step":frame.step,"action":decision.action,"collision":bool(frame.collision),"displacement_m":displacement,"yaw_delta_rad":(frame.pose.yaw_rad-before.pose.yaw_rad+math.pi)%(2*math.pi)-math.pi})
            previous_action = decision.action
            previous_uncertainty = {"low":0.0,"medium":.7,"high":1.0}[decision.uncertainty]
            if frame_sink:
                frame_sink(frame,{"phase":"after_action","arm":arm,"action":decision.action,"interaction":machine.snapshot(),"last_dialogue":dialogue[-1] if dialogue else None})
            if decision.action=="stop":
                stopped = True
                break
        final_distance = metric(frame.pose.position,episode.goal_position)
    except Exception as exc:
        # No raw provider error, response body or credential-bearing string is persisted.
        error = {"type":type(exc).__name__,"phase":"episode_execution","detail":"See locally controlled sanitized diagnostics; no raw provider body retained."}
    provenance = "mock" if "mock" in config.backend_label else ("real" if config.backend_label=="real_habitat" else "replay")
    evaluation = evaluate_episode(stopped=stopped,final_geodesic_m=final_distance,
        shortest_path_m=initial_distance,travelled_path_m=travelled,question_count=machine.delivered,
        provenance=provenance,success_threshold_m=config.success_distance_m,runtime_valid=error is None,
        error=error["type"] if error else None,distance_source="geodesic" if metric_distance else "safe_path_proxy")
    metrics = {"valid":evaluation.status=="valid","success":int(evaluation.success) if evaluation.success is not None else None,
        "spl":evaluation.spl,"path_length_m":travelled,"shortest_start_distance_m":evaluation.shortest_path_m,"final_distance_m":evaluation.final_geodesic_m,
        "stopped":stopped,"steps":len(history),"question_count":machine.delivered,"question_attempts":machine.attempts,
        "metric_path_kind":"supplied_evaluator_geodesic" if metric_distance else "human_safe_adapter_path",
        "success_distance_m":config.success_distance_m,"invalid_reason":evaluation.invalid_reason}
    return {"schema_version":1,"episode_id":episode.episode_id,"scene_id":episode.scene_id,"arm":arm,"seed":config.seed,
        "backend":config.backend_label,"policy":config.policy_label,
        "is_autonomous_vln_result":config.policy_label=="approved_visual_language_policy" and config.backend_label=="real_habitat",
        "metrics":metrics,"evaluation":evaluation.to_dict(),"error":error,"history":history,"decisions":decision_log,"dialogue":dialogue,
        "interaction_events":machine.events,"api_attempts":getattr(policy,"api_attempts",0),
        "provider_usage":getattr(policy,"usage",[])}
