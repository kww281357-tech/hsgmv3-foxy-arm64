"""CLI guard tests: subprocesses run no simulator, provider, or credential access."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from blue_help.prompt_catalog import load_prompts

ROOT = Path(__file__).resolve().parents[1]

class PortableCLITests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, '-m', 'blue_help', *args], cwd=ROOT,
                              text=True, capture_output=True, timeout=15)

    def test_root_help_has_explicit_commands(self):
        result = self.run_cli('--help')
        self.assertEqual(result.returncode, 0)
        for word in ('trial', 'probe', 'survey', 'validate-data', 'plan'):
            self.assertIn(word, result.stdout)

    def test_plan_is_pending_and_serial_not_an_execution(self):
        result = self.run_cli('plan', '--batch-size', '2')
        self.assertEqual(result.returncode, 0)
        plan = json.loads(result.stdout)
        self.assertEqual(len(plan['runs']), 40)
        self.assertTrue(all(r['status'] == 'pending_not_executed' for r in plan['runs']))
        self.assertTrue(all(b['concurrency'] == 1 for b in plan['batches']))

    def test_probe_gate_precedes_image_and_credential_read(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)/'never_created.json'
            result = self.run_cli('probe', '--image', 'does-not-exist.png', '--output', str(output))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Extra Usage OFF', result.stderr)
            self.assertFalse(output.exists())

    def test_live_trial_gate_precedes_data_or_secret_read(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)/'never_created'
            result = self.run_cli('trial', '--data-root', 'missing', '--review-root', 'missing',
                                  '--staging-report', 'missing', '--output', str(output))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Extra Usage OFF', result.stderr)
            self.assertFalse(output.exists())

    def test_trial_output_cannot_be_inside_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            result=self.run_cli('trial','--fixture-preflight','--data-root',str(root),
                '--review-root',str(root/'review'),'--staging-report','missing',
                '--output',str(root/'new_output'))
            self.assertNotEqual(result.returncode,0)
            self.assertIn('must not modify frozen data',result.stderr)
            self.assertFalse((root/'new_output').exists())

    def test_probe_image_human_review_required(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)/'never_created'
            result = self.run_cli('trial', '--confirmed-extra-usage-off',
                '--successful-probe', 'missing', '--data-root', 'missing', '--review-root', 'missing',
                '--staging-report', 'missing', '--output', str(output))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('compare the probe description', result.stderr)
            self.assertFalse(output.exists())

    def test_packaged_resources_and_current_route_origin(self):
        import blue_help
        root = Path(blue_help.__file__).parent
        self.assertTrue((root/'assets/blue_dummy/light_blue_human.obj').is_file())
        prompts = load_prompts()
        self.assertIn('fixed safe route-start point', prompts['policy'])
        self.assertIn('Do not truncate', prompts['dummy_oracle'])
        self.assertIn('safe standing point beside the helper', prompts['question'])

    def test_staged_trial_invalidates_both_benchmark_score_views(self):
        from blue_help.trial.run_trial import mark_staged_not_benchmark
        result={'metrics':{'valid':True,'success':True,'spl':1.0},
                'evaluation':{'status':'valid','success':True,'spl':1.0}}
        mark_staged_not_benchmark(result)
        self.assertFalse(result['metrics']['valid'])
        self.assertTrue(result['measurement_valid'])
        self.assertEqual(result['evaluation']['status'], 'invalid')
        for name in ('metrics','evaluation'):
            self.assertIsNone(result[name]['success'])
            self.assertIsNone(result[name]['spl'])

    def test_runtime_sources_have_no_workspace_path_or_sys_path_injection(self):
        for path in (ROOT/'blue_help').rglob('*.py'):
            source = path.read_text()
            self.assertNotIn('/workspace/scratch/', source)
            self.assertNotIn('sys.path.insert', source)

if __name__ == '__main__':
    unittest.main()
