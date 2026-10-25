import json
import unittest
from blue_help.contracts import Pose, SensorFrame
from blue_help.policy import build_policy_context, parse_decision, PolicyError, VisualLanguagePolicy

PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jDxkAAAAASUVORK5CYII="


def context(offer=True):
    frame = SensorFrame(PNG, None, Pose((123, 0, 456), 0), 3,
                        dummy_candidates=[{"dummy_id":"blue", "spatial_eligible":True,"visible":True,
                                           "horizontal_distance_m":.8,"interaction_radius_m":1,
                                           "hidden_goal":"must not copy"}])
    return build_policy_context(instruction="Go to the bed, then stop.",frame=frame,
        history=[{"step":2,"action":"turn_left","collision":False,"displacement_m":0,
                  "reference_path":"must not copy"}], interaction_state={"remaining_delivered_answers":3},
        active_offer={"offer_id":"O1","dummy_id":"blue","expires_step":7,"private":"must not copy"} if offer else None,
        last_delivered_answer=None,is_lost=False,lost_reasons=[])


def decision(**updates):
    result={"intent":"NAVIGATE","offer_choice":None,"offer_id":None,"candidate_id":"move_forward",
            "uncertainty":"low","evidence_ids":["OBS_3"],"short_reason":"The route ahead is clear.","question":None}
    result.update(updates);return result


class PolicyTests(unittest.TestCase):
    def test_context_excludes_internal_goal_pose_and_reference_fields(self):
        public=json.dumps(context().public)
        for text in ("hidden_goal","reference_path","must not copy","123","456"):
            self.assertNotIn(text,public)
        self.assertIn("Go to the bed, then stop.",public)

    def test_movement_and_english_accept_parse(self):
        self.assertEqual(parse_decision(json.dumps(decision()),context()).action,"move_forward")
        result=decision(intent="REQUEST_HELP",offer_choice="accept",offer_id="O1",candidate_id=None,
                        uncertainty="high",question="Which way should I go from my current heading?")
        self.assertEqual(parse_decision(json.dumps(result),context()).offer_choice,"accept")

    def test_decline_can_continue_navigation(self):
        result=decision(offer_choice="decline",offer_id="O1")
        parsed=parse_decision(json.dumps(result),context())
        self.assertEqual((parsed.action,parsed.offer_choice),("move_forward","decline"))

    def test_reject_invented_candidate_evidence_and_offer(self):
        for update in ({"candidate_id":"teleport"},{"evidence_ids":["GOAL"]},{"offer_choice":"accept","offer_id":"fabricated"}):
            with self.subTest(update=update),self.assertRaises(PolicyError):
                parse_decision(json.dumps(decision(**update)),context())

    def test_no_offer_no_accept(self):
        with self.assertRaises(PolicyError):
            parse_decision(json.dumps(decision(offer_choice="decline",offer_id="O1")),context(False))

    def test_no_default_network_transport(self):
        policy=VisualLanguagePolicy();policy.reset("demo","40","no_ask")
        with self.assertRaisesRegex(RuntimeError,"no request sent"):
            policy.decide(context())
        self.assertEqual(policy.api_attempts,0)

    def test_fake_provider_receives_image_and_complete_instruction(self):
        received=[]
        def fake(request):
            received.append(json.loads(request.body))
            return json.dumps({"model":"deepseek-v4.1-flash","choices":[{"index":0,"finish_reason":"stop",
                "message":{"role":"assistant","content":json.dumps(decision())}}],
                "usage":{"cost":0,"opencode":{"billing_source":"go","allowance":.001}}}).encode()
        policy=VisualLanguagePolicy(fake);policy.reset("demo","40","max3_ask")
        self.assertEqual(policy.decide(context()).action,"move_forward")
        self.assertIn("Go to the bed, then stop.",received[0]["messages"][1]["content"][0]["text"])
        self.assertEqual(received[0]["messages"][1]["content"][1]["image_url"]["url"],PNG)
        self.assertEqual(policy.api_attempts,1)

    def test_unknown_extra_fields_and_duplicate_json_rejected(self):
        bad=decision();bad["goal_position"]=[0,0,0]
        with self.assertRaises(PolicyError):parse_decision(json.dumps(bad),context())
        with self.assertRaises(PolicyError):parse_decision('{"intent":"WAIT","intent":"STOP"}',context())

class RegistryPrivacyTests(unittest.TestCase):
    def test_hidden_registry_actor_is_not_observed(self):
        frame=SensorFrame(PNG,None,Pose((0,0,0),0),0,dummy_candidates=[
            {"dummy_id":"hidden-secret","visible":False,"horizontal_distance_m":.7,"interaction_radius_m":1,"spatial_eligible":False}])
        ctx=build_policy_context(instruction="Continue.",frame=frame,history=[],interaction_state={"remaining_delivered_answers":3},
            active_offer=None,last_delivered_answer=None,is_lost=False,lost_reasons=[])
        self.assertEqual(ctx.public["interaction_candidates"],[])
        self.assertNotIn("hidden-secret",json.dumps(ctx.public))

class ActionSpecTests(unittest.TestCase):
    def test_actual_adapter_degrees_key_is_normalized_not_dropped(self):
        frame=SensorFrame(PNG,None,Pose((0,0,0),0),0)
        frame.action_specs={'turn_left':{'relative_direction':'left','degrees':15,'hidden_goal':'private'}}
        c=build_policy_context(instruction='Continue.',frame=frame,history=[],interaction_state={'remaining_delivered_answers':3},
            active_offer=None,last_delivered_answer=None,is_lost=False,lost_reasons=[])
        self.assertEqual(c.public['action_specs']['turn_left'],{'relative_direction':'left','turn_degrees':15})
