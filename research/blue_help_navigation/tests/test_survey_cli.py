"""Portable survey contract tests using synthetic fixtures, never real Habitat."""
import ast
import contextlib
from dataclasses import replace
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from blue_help.contracts import DummySpec, Episode, Pose, SensorFrame
from blue_help.manual_route import survey


class FakeSurveyEnvironment:
    """Synthetic step-only facade: no pose setter exists after reset."""
    def __init__(self, *, block=False, safe=True):
        self.reset_calls = 0
        self.step_calls = []
        self.closed = False
        self._proxy_collision = False
        self.block, self.safe = block, safe
        self.last_path_check = {"status": "fixture_rejected"}
        self._agent = NS(agent_config=NS(action_space={name:NS(actuation=NS(amount=0))
            for name in ("move_forward","turn_left","turn_right")}))
        self.sim = NS(pathfinder=NS(find_path=self.find_path))

    def reset(self, episode, dummies, seed):
        self.reset_calls += 1
        if self.reset_calls != 1:
            raise AssertionError("Survey reset/teleport after initial anchor")
        self.position = np.asarray(episode.start_position, dtype=float)
        q = episode.start_rotation_xyzw
        self.yaw = 2*math.atan2(q[1],q[3])
        self.count = 0
        self.goal = episode.goal_position
        return self.frame()

    def frame(self):
        return SensorFrame(np.zeros((24,32,3),dtype=np.uint8),
            np.ones((24,32),dtype=np.float32), Pose(tuple(self.position),self.yaw),
            self.count,self._proxy_collision,[{"spatial_eligible":True,"dummy_id":"synthetic"}])

    def step(self, name):
        amount = self._agent.agent_config.action_space[name].actuation.amount
        self.step_calls.append((name,amount))
        self.count += 1
        if name == "move_forward":
            if self.block:
                self._proxy_collision = True
            else:
                self.position += amount*np.array([-math.sin(self.yaw),0,-math.cos(self.yaw)])
        else:
            self.yaw += math.radians(amount)*(1 if name == "turn_left" else -1)
            self.yaw = (self.yaw+math.pi)%(2*math.pi)-math.pi
        return self.frame()

    def find_safe_path(self, start, end):
        return [start,end] if self.safe else None

    def find_path(self, query):
        a,b = np.asarray(query.requested_start),np.asarray(query.requested_end)
        query.points = [a,(a+b)/2,b]
        query.geodesic_distance = float(np.linalg.norm(a-b))
        return True

    def debug_camera(self):
        return {"camera_position":tuple(self.position+[0,1.25,0]),
                "camera_rotation_xyzw":(0,math.sin(self.yaw/2),0,math.cos(self.yaw/2)),
                "hfov_degrees":90}

    def close(self):
        self.closed = True


class SurveyCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "external_input"
        self.data.mkdir()
        self.stage = self.root / "external_stage.json"
        # All coordinates below are invented unit-test geometry, not user data.
        self.episode = Episode("40","synthetic_scene.glb","Synthetic instruction.",
                               (0,0,0),(0,0,0,1),(.3,0,0))
        self.dummy = DummySpec("synthetic-helper",(0,0,-.85),0)
        self.inputs = survey.SurveyInputs(self.data,self.stage,self.episode,self.episode,
            self.dummy,{"status":"synthetic_fixture_not_real_data"},"0"*64,"unchanged_historical_sidecar")

    def tearDown(self):
        self.temporary.cleanup()

    def fixture_files(self, *, episode_id="40", stage_changes=None):
        episode_file = self.data/survey.EPISODES_REL
        episode_file.parent.mkdir(parents=True,exist_ok=True)
        episode_file.write_text(json.dumps({"episodes":[{
            "episode_id":episode_id,"scene_id":"synthetic_scene.glb",
            "instruction":{"instruction_text":"Synthetic instruction."},
            "start_position":[0,0,0],"start_rotation":[0,0,0,1],
            "goals":[{"position":[.3,0,0]}]}]}))
        dummy_file = self.data/survey.DUMMIES_REL
        dummy_file.parent.mkdir(parents=True,exist_ok=True)
        dummy_file.write_text(json.dumps({"episodes":[{
            "episode_id":episode_id,"scene_id":"synthetic_scene.glb",
            "dummies":[{"dummy_id":"synthetic-helper","position":[0,0,-.85],"yaw_rad":0}]}]}))
        stage = {"dummy_id":"synthetic-helper","dummy_ground_position":[0,0,-.85],
                 "camera_ground_position":[0,0,0],"camera_rotation_xyzw":[0,0,0,1]}
        stage.update(stage_changes or {})
        self.stage.write_text(json.dumps({"staged_view":stage}))

    def test_cli_requires_all_explicit_locations(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): survey.main([])
        with patch.object(survey,"load_survey_inputs",return_value=self.inputs) as loader:
            result = {"kind":"privileged_offline_oracle_survey","episode_id":"40",
                      "walk_complete":True,"actual_walked_m":.3}
            with patch.object(survey,"run_survey",return_value=result) as runner:
                with contextlib.redirect_stdout(io.StringIO()) as text:
                    code = survey.main(["--data-root",str(self.data),"--staging-report",str(self.stage),
                                        "--output",str(self.root/"new_private"),"--max-actions","80"])
        self.assertEqual(code,0)
        loader.assert_called_once_with(self.data,self.stage)
        runner.assert_called_once_with(self.inputs,self.root/"new_private",max_actions=80,
                                     helper_profile="game-v7", helper_asset=None)
        self.assertFalse(json.loads(text.getvalue())["is_benchmark_result"])

    def test_python_api_defaults_to_game_not_implicit_blue(self):
        import inspect
        self.assertEqual(inspect.signature(survey.run_survey).parameters['helper_profile'].default,'game-v7')

    def test_help_does_not_import_habitat(self):
        code = """import sys
from blue_help.manual_route.survey import main
try:
    main(['--help'])
except SystemExit as exc:
    assert exc.code == 0
assert 'habitat_sim' not in sys.modules
"""
        result = subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,
                                cwd=Path(survey.__file__).resolve().parents[2])
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn("--staging-report",result.stdout)

    def test_dataset_pin_checked_before_simulator_or_output(self):
        self.fixture_files()
        with patch.object(survey,"validate_frozen_data",side_effect=ValueError("checksum")):
            with patch.object(survey,"run_survey") as run:
                with self.assertRaisesRegex(ValueError,"checksum"):
                    survey.main(["--data-root",str(self.data),"--staging-report",str(self.stage),
                                 "--output",str(self.root/"unused")])
                run.assert_not_called()
        self.assertFalse((self.root/"unused").exists())

    def test_first_episode_must_be_pinned_40(self):
        self.fixture_files(episode_id="41")
        with patch.object(survey,"validate_frozen_data",return_value={}):
            with self.assertRaisesRegex(ValueError,"pinned"):
                survey.load_survey_inputs(self.data,self.stage)

    def test_external_anchor_but_goal_only_from_dataset(self):
        self.fixture_files(stage_changes={"camera_ground_position":[.1,0,0],
                                          "goal_position":[999,999,999]})
        with patch.object(survey,"validate_frozen_data",return_value={"status":"fixture"}) as check:
            loaded = survey.load_survey_inputs(self.data,self.stage)
        check.assert_called_once_with(self.data.resolve())
        self.assertEqual(loaded.staged_episode.start_position,(.1,0,0))
        self.assertEqual(loaded.staged_episode.goal_position,(.3,0,0))
        self.assertEqual(loaded.dummy.dummy_id,"synthetic-helper")

    def test_relocation_requires_explicit_metadata(self):
        self.fixture_files(stage_changes={"dummy_ground_position":[.1,0,-.85]})
        with patch.object(survey,"validate_frozen_data",return_value={}):
            with self.assertRaisesRegex(ValueError,"relocation"):
                survey.load_survey_inputs(self.data,self.stage)
        self.fixture_files(stage_changes={"dummy_ground_position":[.1,0,-.85],
            "placement_kind":"explicit_goal_independent_demo_relocation"})
        with patch.object(survey,"validate_frozen_data",return_value={}):
            loaded = survey.load_survey_inputs(self.data,self.stage)
        self.assertEqual(loaded.dummy.ground_position,(.1,0,-.85))

    def test_nonfinite_duplicate_and_invalid_rotation_fail_closed(self):
        self.fixture_files(stage_changes={"camera_rotation_xyzw":[0,0,0,2]})
        with patch.object(survey,"validate_frozen_data",return_value={}):
            with self.assertRaisesRegex(ValueError,"unit"):
                survey.load_survey_inputs(self.data,self.stage)
        with self.assertRaises(ValueError): survey._strict_json('{"x":1,"x":2}')
        with self.assertRaises(ValueError): survey._strict_json('{"x":NaN}')

    def test_packaged_asset_is_not_a_workspace_or_sibling_path(self):
        asset = survey.packaged_object_config()
        self.assertEqual(asset.parent.parent.parent,Path(survey.__file__).resolve().parents[1])
        config = json.loads(asset.read_text())
        self.assertTrue((asset.parent/config["render_asset"]).is_file())
        self.assertTrue((asset.parent/config["collision_asset"]).is_file())

    def run_fixture(self, env, output=None, *, inputs=None, max_actions=10000):
        captured = []
        def make_environment(scene,navmesh,asset,*,config):
            captured.append((scene,navmesh,asset,config))
            return env
        report = survey.run_survey(inputs or self.inputs,output or self.root/"evidence",
            environment_factory=make_environment,shortest_path_factory=NS,max_actions=max_actions,
            helper_profile="blue-reference")
        return report,captured

    def test_full_mock_walk_steps_only_single_reset_and_private_evidence(self):
        env = FakeSurveyEnvironment()
        report,captured = self.run_fixture(env)
        self.assertTrue(report["walk_complete"])
        self.assertEqual(env.reset_calls,1)
        self.assertTrue(env.closed)
        self.assertGreater(len(env.step_calls),0)
        self.assertAlmostEqual(report["actual_walked_m"],.3)
        self.assertLessEqual(report["endpoint_error_m"],.05)
        self.assertTrue(report["goal_used_by_survey"])
        self.assertFalse(report["policy_executed"])
        self.assertFalse(report["is_benchmark_result"])
        self.assertEqual(report["model_api_calls"],0)
        config = captured[0][3]
        self.assertTrue(config.allow_sliding)
        self.assertEqual((config.agent_radius_m,config.forward_step_m,config.turn_degrees),(.10,.10,15))
        actions = json.loads((self.root/"evidence/actions.private.json").read_text())
        self.assertTrue(all(row["amount"] <= .10+1e-9 for row in actions if row["action"]=="move_forward"))
        rows = json.loads((self.root/"evidence/frames.private.json").read_text())
        self.assertTrue(any(row["kind"] == "stationary_left_view" for row in rows))
        self.assertTrue(any(row["kind"] == "stationary_right_view" for row in rows))
        self.assertTrue(list((self.root/"evidence").glob("*.png")))
        self.assertTrue(list((self.root/"evidence").glob("*_depth.npz")))

    def test_output_immutable_and_not_inside_frozen_data(self):
        output = self.root/"existing"
        output.mkdir()
        with self.assertRaisesRegex(ValueError,"already exists"):
            self.run_fixture(FakeSurveyEnvironment(),output)
        with self.assertRaisesRegex(ValueError,"frozen data"):
            self.run_fixture(FakeSurveyEnvironment(),self.data/"forbidden_output")

    def test_human_guard_failure_saved_and_env_closed(self):
        env = FakeSurveyEnvironment(block=True)
        with self.assertRaisesRegex(RuntimeError,"Human collision"):
            self.run_fixture(env)
        self.assertTrue(env.closed)
        report = json.loads((self.root/"evidence/audit.private.json").read_text())
        self.assertFalse(report["walk_complete"])
        self.assertEqual(report["human_proxy_rejections"],1)

    def test_action_budget_is_bounded_and_failure_evidence_retained(self):
        env = FakeSurveyEnvironment()
        with self.assertRaisesRegex(RuntimeError,"action budget"):
            self.run_fixture(env,max_actions=1)
        self.assertTrue(env.closed)
        report = json.loads((self.root/"evidence/audit.private.json").read_text())
        self.assertFalse(report["walk_complete"])
        self.assertEqual(report["action_count"],1)

    def test_unsafe_route_never_executes_movement(self):
        env = FakeSurveyEnvironment(safe=False)
        with self.assertRaisesRegex(RuntimeError,"No audited safe route"):
            self.run_fixture(env)
        self.assertEqual(env.step_calls,[])
        self.assertTrue(env.closed)

    def test_source_has_no_pose_setter_or_workspace_root(self):
        source = Path(survey.__file__).read_text()
        tree = ast.parse(source)
        methods = {node.func.attr for node in ast.walk(tree)
                   if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)}
        self.assertFalse(methods & {"set_state","initialize_agent","set_translation","teleport"})
        self.assertNotIn("sys.path",source)
        self.assertNotIn("/workspace/",source)
        self.assertNotIn("/home/",source)
        self.assertNotIn("import habitat_sim",source)
        self.assertIn("from habitat_sim import ShortestPath",source)  # function-local only


if __name__ == "__main__":
    unittest.main()
