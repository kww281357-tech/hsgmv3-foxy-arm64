import unittest
import math
from blue_help.contracts import Episode,Pose,SensorFrame
from blue_help.policy import Decision,ReplayPolicy
from blue_help.runner import run_episode,RunConfig

PNG='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jDxkAAAAASUVORK5CYII='
EP=Episode('40','test_scene','Continue forward.',(0,0,0),(0,0,0,1),(0,0,-1))
CFG=RunConfig(max_steps=40,backend_label='mock_unit',policy_label='replay_test')

class FakeEnv:
    def __init__(self,visible=True,oracle_path=True):self.visible=visible;self.oracle_path=oracle_path
    def reset(self,episode,dummies,seed=42):self.step_id=0;self.z=0;return self.observe()
    def observe(self):
        inside=self.step_id%11==0
        return SensorFrame(PNG,None,Pose((0,0,self.z),0),self.step_id,dummy_candidates=[{
            'dummy_id':'human_guide_kitchen','horizontal_distance_m':.8 if inside else 1.5,
            'interaction_radius_m':1.0,'spatial_eligible':inside and self.visible,'visible':self.visible}])
    def step(self,action):
        self.step_id+=1
        if action=='move_forward':self.z-=.1
        return self.observe()
    def find_safe_path(self,a,b):return [a,b] if self.oracle_path else None

class HelpPolicy:
    def __init__(self,choice='accept'):self.choice=choice;self.contexts=[]
    def reset(self,*args):self.contexts=[]
    def decide(self,context):
        self.contexts.append(context.public)
        offer=context.public['active_offer']
        if offer:
            if self.choice=='accept':return Decision('wait','accept',offer['offer_id'],'Which way should I go from my current heading?','high')
            return Decision('move_forward',self.choice,offer['offer_id'])
        return Decision('stop' if context.public['observation_step']>=35 else 'move_forward')

class RunnerTests(unittest.TestCase):
    def test_max3_then_no_fourth_and_navigation_resumes(self):
        result=run_episode(FakeEnv(),HelpPolicy(),EP,[],config=CFG)
        self.assertIsNone(result['error'])
        self.assertEqual(result['metrics']['question_count'],3)
        self.assertEqual(result['metrics']['question_attempts'],3)
        self.assertEqual(len([x for x in result['dialogue'] if x['kind']=='answer']),3)
        self.assertGreater(result['metrics']['path_length_m'],2)
        self.assertFalse(result['is_autonomous_vln_result'])

    def test_reset_prevents_budget_cross_episode(self):
        env,policy=FakeEnv(),HelpPolicy()
        for i in range(2):self.assertEqual(run_episode(env,policy,EP,[],config=CFG)['metrics']['question_count'],3)

    def test_noask_never_calls_oracle(self):
        def fail(*a,**k):raise AssertionError('No oracle in control arm')
        result=run_episode(FakeEnv(),HelpPolicy(),EP,[],arm='no_ask',config=CFG,oracle_factory=fail)
        self.assertIsNone(result['error']);self.assertEqual(result['metrics']['question_count'],0)
        self.assertEqual(result['dialogue'],[])

    def test_decline_ignore_do_not_consume_and_continue(self):
        for choice in ('decline','ignore'):
            result=run_episode(FakeEnv(),HelpPolicy(choice),EP,[],config=CFG)
            self.assertEqual(result['metrics']['question_count'],0)
            self.assertEqual(result['metrics']['question_attempts'],0)
            self.assertGreater(result['metrics']['path_length_m'],3)

    def test_no_visible_dummy_no_questions_and_no_registry_leak(self):
        policy=HelpPolicy();result=run_episode(FakeEnv(False),policy,EP,[],config=CFG)
        self.assertEqual(result['dialogue'],[])
        self.assertTrue(all(x['interaction_candidates']==[] for x in policy.contexts))

    def test_private_semantic_names_and_goal_not_sent(self):
        import json
        policy=HelpPolicy();run_episode(FakeEnv(),policy,EP,[],config=CFG)
        public=json.dumps(policy.contexts)
        self.assertNotIn('human_guide_kitchen',public)
        self.assertNotIn('goal_position',public)
        self.assertNotIn('reference_pose',public)
        self.assertIn('DUMMY_001',public)
        answered=[c for c in policy.contexts if c['last_answer_metadata']]
        self.assertEqual(answered[0]['last_answer_metadata'],{'question_step':0,'delivery_step':0})

    def test_unreachable_answers_do_not_consume_and_invalid_metrics_not_failure(self):
        result=run_episode(FakeEnv(oracle_path=False),HelpPolicy(),EP,[],config=CFG)
        self.assertEqual(result['metrics']['question_count'],0)
        self.assertGreater(result['metrics']['question_attempts'],0)
        self.assertFalse(result['metrics']['valid']);self.assertIsNone(result['metrics']['success'])

    def test_basic_navigation_and_stop_evaluation(self):
        result=run_episode(FakeEnv(),ReplayPolicy([Decision('move_forward'),Decision('stop')]),EP,[],arm='no_ask',config=CFG,metric_distance=math.dist)
        self.assertEqual(result['metrics']['success'],1)
        self.assertAlmostEqual(result['metrics']['path_length_m'],.1)

    def test_no_stop_is_not_success(self):
        config=RunConfig(max_steps=1,backend_label='mock_unit',policy_label='replay_test')
        result=run_episode(FakeEnv(),ReplayPolicy([Decision('move_forward')]),EP,[],arm='no_ask',config=config,metric_distance=math.dist)
        self.assertEqual(result['metrics']['success'],0)

    def test_provider_failure_invalid_not_silent_failure(self):
        class Bad(HelpPolicy):
            def decide(self,c):raise RuntimeError('A secret MUST NOT be retained')
        result=run_episode(FakeEnv(),Bad(),EP,[],config=CFG)
        self.assertEqual(result['error']['type'],'RuntimeError')
        self.assertNotIn('secret',str(result))
        self.assertIsNone(result['metrics']['success'])

    def test_execution_rechecks_eligibility(self):
        class Vanishes(FakeEnv):
            def reset(self,*a,**k):self.calls=0;return super().reset(*a,**k)
            def observe(self):
                self.calls+=1
                frame=super().observe()
                if self.calls>1:frame.dummy_candidates[0]['spatial_eligible']=False
                return frame
        result=run_episode(Vanishes(),HelpPolicy(),EP,[],config=CFG)
        self.assertEqual(result['metrics']['question_count'],0)
        self.assertEqual(result['metrics']['question_attempts'],0)

class RadiusConfigTests(unittest.TestCase):
    def test_trusted_environment_radius_is_shared_with_state_machine(self):
        from types import SimpleNamespace
        class SmallRange(FakeEnv):
            config=SimpleNamespace(interaction_radius_m=.9)
            def observe(self):
                frame=super().observe()
                for candidate in frame.dummy_candidates:candidate['interaction_radius_m']=.9
                return frame
        result=run_episode(SmallRange(),HelpPolicy(),EP,[],config=CFG)
        self.assertIsNone(result['error'])
        self.assertEqual(result['metrics']['question_count'],3)

class EvaluationGuardTests(unittest.TestCase):
    def test_invalid_initial_geodesic_blocks_all_model_decisions(self):
        import json
        class Counter(HelpPolicy):
            calls=0
            def decide(self,c):self.calls+=1;return super().decide(c)
        policy=Counter()
        result=run_episode(FakeEnv(),policy,EP,[],config=CFG,metric_distance=lambda a,b:float('nan'))
        self.assertEqual(policy.calls,0)
        self.assertFalse(result['metrics']['valid'])
        self.assertIsNone(result['metrics']['shortest_start_distance_m'])
        json.dumps(result,allow_nan=False)
