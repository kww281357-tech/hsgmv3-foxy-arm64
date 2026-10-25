import tempfile,unittest,math
from pathlib import Path
from blue_help.data import FROZEN_IDS
from blue_help.contracts import Episode
from blue_help.harness import run_paired
from blue_help.runner import RunConfig
from blue_help.policy import ReplayPolicy,Decision
from test_runner import FakeEnv

class ClosedEnv(FakeEnv):
    closes=0
    def close(self):type(self).closes+=1

class HarnessTests(unittest.TestCase):
    def episodes(self):return [Episode(str(i),'scene','Continue.',(0,0,0),(0,0,0,1),(0,0,-1)) for i in FROZEN_IDS]
    def run_batch(self,path,metric=True):
        return run_paired(self.episodes(),lambda e:[],env_factory=ClosedEnv,
            policy_factory=lambda eid,arm:ReplayPolicy([Decision('stop')]),output_dir=path,
            config=RunConfig(max_steps=1,backend_label='mock_unit',policy_label='replay_test'),
            metric_factory=(lambda env:math.dist) if metric else None)
    def test_default_two_episodes_four_serial_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=self.run_batch(Path(tmp)/'out')
            self.assertEqual(len(m['runs']),4)
            self.assertEqual([r['episode_id'] for r in m['runs']],['40','40','41','41'])
            self.assertEqual([r['arm'] for r in m['runs']],['no_ask','max3_ask']*2)
            self.assertEqual(m['status'],'requested_batches_complete')
    def test_invalid_proxy_pauses_after_first_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=self.run_batch(Path(tmp)/'out',False)
            self.assertEqual(len(m['runs']),1);self.assertEqual(m['status'],'paused_invalid_measurement')
    def test_no_overwrite_previous_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'out';self.run_batch(p)
            with self.assertRaisesRegex(ValueError,'fresh'):self.run_batch(p)
    def test_real_evaluation_needs_evaluator_before_backend_creation(self):
        with tempfile.TemporaryDirectory() as tmp,self.assertRaisesRegex(ValueError,'geodesic'):
            run_paired(self.episodes(),lambda e:[],env_factory=lambda:1/0,policy_factory=lambda *x:1/0,
                output_dir=Path(tmp)/'out',config=RunConfig(backend_label='real_habitat'))

class InvalidMetricFactoryTests(unittest.TestCase):
    def test_real_none_metric_blocks_policy_factory(self):
        calls=[]
        with tempfile.TemporaryDirectory() as tmp,self.assertRaisesRegex(ValueError,'callable'):
            run_paired(HarnessTests().episodes(),lambda e:[],env_factory=ClosedEnv,
                policy_factory=lambda *x:calls.append(x),metric_factory=lambda env:None,
                output_dir=Path(tmp)/'out',config=RunConfig(backend_label='real_habitat'))
        self.assertEqual(calls,[])
