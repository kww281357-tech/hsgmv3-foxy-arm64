"""Offline regression tests for failure-segment preparation and diagnostic hard-off guards."""
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for p in (str(SRC), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import prepare_failure_segments as pfs


def _extract_agent_decomp_runner():
    """Extract the exact decomposition AST block from PathPlannerAgent.__init__."""
    agent_path = SRC / "agent" / "agent.py"
    source = agent_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    target_stmts = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            start_idx = None
            end_idx = None
            for idx, stmt in enumerate(node.body):
                src_seg = ast.get_source_segment(source, stmt)
                if "nav_cfg =" in src_seg and "navigation" in src_seg:
                    start_idx = idx
                if start_idx is not None and "self.instruction_obj =" in src_seg:
                    end_idx = idx
                    break
            if start_idx is not None and end_idx is not None:
                target_stmts = node.body[start_idx : end_idx + 1]
                break
    assert target_stmts is not None, "Could not locate decomposition block in PathPlannerAgent.__init__"

    func_def = ast.FunctionDef(
        name="_run_decomp_block",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg="self"),
                ast.arg(arg="config"),
                ast.arg(arg="instruction"),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=target_stmts,
        decorator_list=[],
    )
    ast.fix_missing_locations(func_def)
    namespace = {
        "os": os,
        "Instruction": lambda full_instruction, sub_instructions: NS(
            full_instruction=full_instruction, sub_instructions=sub_instructions
        ),
    }
    exec(compile(ast.Module(body=[func_def], type_ignores=[]), str(agent_path), "exec"), namespace)
    return namespace["_run_decomp_block"]


def _extract_r2r_env_harness(extra_globals=None):
    """Extract R2REnv._initialize_episode and _save_episode_results into a lightweight class."""
    run_exp_path = SRC / "run_experiments.py"
    tree = ast.parse(run_exp_path.read_text(encoding="utf-8"))
    init_node = None
    save_node = None
    run_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "R2REnv":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_initialize_episode":
                    init_node = item
                elif isinstance(item, ast.FunctionDef) and item.name == "_save_episode_results":
                    save_node = item
                elif isinstance(item, ast.FunctionDef) and item.name == "_run_episode":
                    run_node = item
    if save_node is None:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_save_episode_results":
                save_node = node
                break
    assert init_node is not None and save_node is not None

    class _BaseEnv:
        def _initialize_episode(self, episode_ndx: int):
            self._super_initialized = episode_ndx

    cls_def = ast.ClassDef(
        name="ExtractedR2REnv",
        bases=[ast.Name(id="_BaseEnv", ctx=ast.Load())],
        keywords=[],
        body=[init_node, save_node, run_node],
        decorator_list=[],
    )
    ast.fix_missing_locations(cls_def)
    namespace = {
        "_BaseEnv": _BaseEnv,
        "os": os,
        "np": np,
        "json": json,
        "tempfile": tempfile,
        "logging": __import__("logging"),
        "EpisodeVideoRecorder": type("EpisodeVideoRecorder", (), {}),
        "PolarAction": NS(null="NULL_ACTION"),
        "SimWrapper": Mock(side_effect=lambda cfg: NS(cfg=cfg, close=Mock(), step=Mock(return_value={"rgb": None}), get_path=Mock(return_value=5.2))),
        "GPTAgent": Mock(side_effect=lambda sim, cfg, inst, **kw: NS(sim=sim, cfg=copy.deepcopy(cfg), instruction=inst, kwargs=kw, bind_episode_targets=Mock(), _instruction_decomposition_enabled=False, instruction_obj=NS(sub_instruction_keys=["P1"], num_subtasks=1))),
    }
    if extra_globals:
        namespace.update(extra_globals)
    exec(compile(ast.Module(body=[cls_def], type_ignores=[]), str(run_exp_path), "exec"), namespace)
    return namespace["ExtractedR2REnv"], namespace


class TestFailureSegments(unittest.TestCase):
    def test_actual_run_entry_saves_initialization_failure_without_previous_agent_state(self):
        Env, ns = _extract_r2r_env_harness()
        with tempfile.TemporaryDirectory() as td:
            env = Env()
            env.env_cfg = {'name': 'slice', 'output_root': td, 'max_steps': 100}
            env.all_episodes = [{'diagnostic_segment': {'schema': 1, 'max_steps': 24,
                                   'restore_mode': 'rounded_pose_cold_start', 'segment_id': 'rejected'}}]
            env.agent = NS(_execution_ledger=[{'step': 5, 'action_type': 'stop'}],
                           _instruction_decomposition_enabled=False)
            env.agent_trajectory = [[99, 99, 99]]
            env.current_episode = {'instruction': 'previous successful episode'}
            metrics = env._run_episode(0, show_visualization=False)
            self.assertEqual(metrics['termination_reason'], 'diagnostic_initialization_failed')
            saved = json.loads((Path(td)/'results_slice/episode_0_results.json').read_text())
            self.assertEqual(saved['diagnostic_segment']['segment_id'], 'rejected')
            self.assertIsNone(saved['instruction_decomposition_enabled'])
            self.assertFalse(saved['diagnostic_contract_satisfied'])
            self.assertFalse(saved['execution_summary']['stop_executed'])
            self.assertEqual(saved['agent_trajectory'], [])
            ns['GPTAgent'].assert_not_called()
            ns['SimWrapper'].assert_not_called()

    def test_recovery_events_do_not_count_as_execution_records(self):
        _, result = self._sample_dataset_and_result()
        result['subtask']['P1']['rollback_events'] = [{'step': 4, 'event': 'started'}]
        row = pfs.extract_step_executed_action(result, 4)
        self.assertIs(row, result['subtask']['P1']['record'][1])

    def test_cross_episode_source_and_existing_output_are_rejected(self):
        dataset, result = self._sample_dataset_and_result()
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            dp, rp, cp, op = [base / name for name in ('data.json', 'result.json', 'cases.json', 'out.json')]
            dp.write_text(json.dumps(dataset))
            cp.write_text(json.dumps([{'segment_id': 'test', 'episode_id': 101, 'step': 4,
                                      'source_result': 'result.json', 'instruction': 'stop on the stairs.'}]))
            result['subtask']['P1']['instruction'] = 'A different original episode instruction.'
            rp.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'original single instruction'):
                pfs.prepare_failure_segments(dp, cp, op)
            self.assertFalse(op.exists())
            result['subtask']['P1']['instruction'] = dataset['episodes'][0]['instruction']['instruction_text']
            rp.write_text(json.dumps(result))
            op.touch()
            with self.assertRaisesRegex(ValueError, 'overwrite existing'):
                pfs.prepare_failure_segments(dp, cp, op)
            self.assertEqual(op.read_bytes(), b'')

    def test_cli_hard_off_flag_sets_the_production_config(self):
        import argparse
        source = (SRC / 'run_experiments.py').read_text()
        tree = ast.parse(source)
        main = next(n for n in tree.body if isinstance(n, ast.If)
                    and '__name__' in ast.get_source_segment(source, n.test))
        add_argument = next(n for n in main.body if isinstance(n, ast.Expr)
                            and '--no_subtask_decomposition' in ast.get_source_segment(source, n))
        apply_flag = next(n for n in main.body if isinstance(n, ast.If)
                          and ast.get_source_segment(source, n.test) == 'args.no_subtask_decomposition')
        for flag in ('--no-subtask-decomposition', '--no_subtask_decomposition'):
            ns = {'parser': argparse.ArgumentParser(), 'sim_config': {'navigation': {}}}
            exec(compile(ast.Module(body=[add_argument], type_ignores=[]), '<actual CLI>', 'exec'), ns)
            ns['args'] = ns['parser'].parse_args([flag])
            exec(compile(ast.Module(body=[apply_flag], type_ignores=[]), '<actual config>', 'exec'), ns)
            self.assertIs(ns['sim_config']['navigation']['disable_subtask_decomposition'], True)

    def test_segment_results_are_excluded_from_whole_episode_comparisons(self):
        from run_failed_case_retest import summarize_result
        with self.assertRaisesRegex(ValueError, 'full-episode'):
            summarize_result({'diagnostic_segment': {'segment_id': 'door'},
                              'metrics': {'goal_reached': True}}, {})

    def _sample_dataset_and_result(self):
        dataset = {
            "dataset_name": "r2r_val_unseen",
            "split": "val_unseen",
            "episodes": [
                {
                    "episode_id": 101,
                    "scene_id": "mp3d/2t7WUuJeko7/2t7WUuJeko7.glb",
                    "start_position": [1.25, -0.4, 3.5],
                    "start_rotation": [0.0, 0.0, 0.0, 1.0],
                    "instruction": {
                        "instruction_text": "Walk past the bed, turn left at the doorway, and stop on the stairs.",
                        "instruction_tokens": [1, 2, 3],
                    },
                    "goals": [{"position": [5.0, -0.4, 8.0], "radius": 3.0}],
                    "reference_path": [[1.25, -0.4, 3.5], [3.0, -0.4, 6.0], [5.0, -0.4, 8.0]],
                    "info": {"geodesic_distance": 7.85},
                }
            ],
        }
        result = {
            "metrics": {"distance_to_goal": 4.1},
            "agent_trajectory": [[1.25, -0.4, 3.5], [99.0, 99.0, 99.0]],
            "subtask": {
                "P1": {
                    "instruction": dataset["episodes"][0]["instruction"]["instruction_text"],
                    "record": [
                        {
                            "step": 1,
                            "executed_action": "Move Forward 0.20m",
                            "action_type": "move",
                            "pose_before": [0.0, 0.0, 0.0],
                            "yaw_before": 0.0,
                        },
                        {
                            "step": 4,
                            "executed_action": "Move Forward 0.20m",
                            "action_type": "move",
                            "pose_before": [2.0, -1.5, 0.3],
                            "yaw_before": 0.5,
                        },
                    ],
                }
            },
        }
        return dataset, result

    def test_pose_and_yaw_conversion_boundaries_and_known_pose(self):
        pos = pfs.mapper_pose_to_habitat_position(
            start_position=[1.25, -0.4, 3.5],
            pose_before=[2.0, -1.5, 0.3],
            camera_height=1.2,
        )
        self.assertAlmostEqual(pos[0], 3.25, places=7)
        self.assertAlmostEqual(pos[1], 1.1, places=7)
        self.assertAlmostEqual(pos[2], 2.0, places=7)

        pos_custom_cam = pfs.mapper_pose_to_habitat_position(
            start_position=[1.25, -0.4, 3.5],
            pose_before=[2.0, -1.5, 0.3],
            camera_height=1.45,
        )
        self.assertAlmostEqual(pos_custom_cam[1], 1.35, places=7)

        # Boundary yaws: alpha = -pi/2 -> theta = 0 -> identity xyzw quaternion [0, 0, 0, 1]
        q_zero_theta = pfs.mapper_yaw_to_habitat_quaternion(-math.pi / 2.0)
        self.assertAlmostEqual(q_zero_theta[0], 0.0, places=7)
        self.assertAlmostEqual(q_zero_theta[1], 0.0, places=7)
        self.assertAlmostEqual(q_zero_theta[2], 0.0, places=7)
        self.assertAlmostEqual(q_zero_theta[3], 1.0, places=7)

        for alpha in (0.0, math.pi / 2.0, -math.pi / 2.0, math.pi, -math.pi, 0.732):
            expected_theta = -alpha - (math.pi / 2.0)
            self.assertAlmostEqual(pfs.mapper_yaw_to_habitat_theta(alpha), expected_theta, places=7)
            q = pfs.mapper_yaw_to_habitat_quaternion(alpha)
            self.assertEqual(q[0], 0.0)
            self.assertEqual(q[2], 0.0)
            self.assertAlmostEqual(q[1], math.sin(expected_theta / 2.0), places=7)
            self.assertAlmostEqual(q[3], math.cos(expected_theta / 2.0), places=7)
            norm = math.sqrt(sum(c * c for c in q))
            self.assertAlmostEqual(norm, 1.0, places=7)

    def test_reject_missing_yaw_nan_duplicate_step_and_non_verbatim_slice(self):
        dataset, result = self._sample_dataset_and_result()

        # Missing yaw_before
        bad_no_yaw = copy.deepcopy(result)
        del bad_no_yaw["subtask"]["P1"]["record"][1]["yaw_before"]
        with self.assertRaises(ValueError):
            pfs.extract_step_executed_action(bad_no_yaw, 4)

        # Missing pose_before
        bad_no_pose = copy.deepcopy(result)
        bad_no_pose["subtask"]["P1"]["record"][1]["pose_before"] = None
        with self.assertRaises(ValueError):
            pfs.extract_step_executed_action(bad_no_pose, 4)

        # NaN in yaw_before or pose_before
        bad_nan_yaw = copy.deepcopy(result)
        bad_nan_yaw["subtask"]["P1"]["record"][1]["yaw_before"] = float("nan")
        with self.assertRaises(ValueError):
            pfs.extract_step_executed_action(bad_nan_yaw, 4)

        bad_inf_pose = copy.deepcopy(result)
        bad_inf_pose["subtask"]["P1"]["record"][1]["pose_before"] = [0.0, float("inf"), 0.0]
        with self.assertRaises(ValueError):
            pfs.extract_step_executed_action(bad_inf_pose, 4)

        # Duplicate step in subtask records
        bad_dup_step = copy.deepcopy(result)
        bad_dup_step["subtask"]["P1"]["record"].append(copy.deepcopy(bad_dup_step["subtask"]["P1"]["record"][1]))
        with self.assertRaises(ValueError):
            pfs.extract_step_executed_action(bad_dup_step, 4)

        # Refuses fallback to agent_trajectory when subtask is missing
        only_traj = {"agent_trajectory": [[1.0, 2.0, 3.0]]}
        with self.assertRaises(ValueError):
            pfs.extract_step_executed_action(only_traj, 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ds_path = tmp / "dataset.json"
            res_path = tmp / "episode_0_results.json"
            cases_path = tmp / "cases.json"
            out_path = tmp / "out.json"

            ds_path.write_text(json.dumps(dataset), encoding="utf-8")
            res_path.write_text(json.dumps(result), encoding="utf-8")

            # Non-verbatim slice (paraphrased text)
            cases_bad_slice = [
                {
                    "segment_id": "seg_bad",
                    "source_result": "episode_0_results.json",
                    "episode_id": 101,
                    "step": 4,
                    "instruction": "turn left at the door and stop",
                }
            ]
            cases_path.write_text(json.dumps(cases_bad_slice), encoding="utf-8")
            with self.assertRaises(ValueError):
                pfs.prepare_failure_segments(ds_path, cases_path, out_path)
            self.assertFalse(out_path.exists())

            # Duplicate segment_id
            valid_case = {
                "segment_id": "seg_dup",
                "source_result": "episode_0_results.json",
                "episode_id": 101,
                "step": 4,
                "instruction": "turn left at the doorway, and stop on the stairs.",
            }
            cases_path.write_text(json.dumps([valid_case, copy.deepcopy(valid_case)]), encoding="utf-8")
            with self.assertRaises(ValueError):
                pfs.prepare_failure_segments(ds_path, cases_path, out_path)

            # Refuse overwriting dataset or result file
            cases_path.write_text(json.dumps([valid_case]), encoding="utf-8")
            with self.assertRaises(ValueError):
                pfs.prepare_failure_segments(ds_path, cases_path, ds_path)
            with self.assertRaises(ValueError):
                pfs.prepare_failure_segments(ds_path, cases_path, res_path)

    def test_artifact_traceability_and_original_data_immutability(self):
        dataset, result = self._sample_dataset_and_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ds_path = tmp / "dataset.json"
            res_dir = tmp / "results"
            res_dir.mkdir()
            res_path = res_dir / "episode_0_results.json"
            cases_path = tmp / "cases.json"
            out_path = tmp / "diag_segments.json"

            ds_bytes = json.dumps(dataset, indent=2).encode("utf-8")
            res_bytes = json.dumps(result, indent=2).encode("utf-8")
            ds_path.write_bytes(ds_bytes)
            res_path.write_bytes(res_bytes)
            ds_sha_before = hashlib.sha256(ds_bytes).hexdigest()
            res_sha_before = hashlib.sha256(res_bytes).hexdigest()

            slice_text = "turn left at the doorway, and stop on the stairs."
            cases = [
                {
                    "segment_id": "ep101_step4_doorway_stairs",
                    "source_result": "results/episode_0_results.json",
                    "episode_id": 101,
                    "step": 4,
                    "instruction": slice_text,
                }
            ]
            cases_path.write_text(json.dumps(cases), encoding="utf-8")

            rc = pfs.main(
                [
                    "--dataset",
                    str(ds_path),
                    "--cases",
                    str(cases_path),
                    "--output",
                    str(out_path),
                    "--camera-height",
                    "1.2",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertEqual(hashlib.sha256(ds_path.read_bytes()).hexdigest(), ds_sha_before)
            self.assertEqual(hashlib.sha256(res_path.read_bytes()).hexdigest(), res_sha_before)

            out = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(out["dataset_name"], "r2r_val_unseen")
            self.assertEqual(out["split"], "val_unseen")
            self.assertEqual(len(out["episodes"]), 1)

            ep = out["episodes"][0]
            self.assertEqual(ep["episode_id"], 0)
            self.assertEqual(ep["instruction"]["instruction_text"], slice_text)
            self.assertIsNone(ep["reference_path"])
            self.assertEqual(ep["goals"], dataset["episodes"][0]["goals"])
            self.assertTrue(ep["info"]["geodesic_distance_requires_recalculation"])

            diag = ep["diagnostic_segment"]
            self.assertEqual(diag["schema"], 1)
            self.assertEqual(diag["segment_id"], "ep101_step4_doorway_stairs")
            self.assertEqual(diag["source_episode_id"], 101)
            self.assertEqual(diag["source_step"], 4)
            self.assertEqual(diag["source_dataset_sha256"], ds_sha_before)
            self.assertEqual(diag["source_result_sha256"], res_sha_before)
            self.assertEqual(diag["instruction"], slice_text)
            self.assertEqual(
                diag["source_original_instruction"],
                dataset["episodes"][0]["instruction"]["instruction_text"],
            )
            self.assertEqual(diag["restore_mode"], "rounded_pose_cold_start")
            self.assertFalse(diag["decomposition_enabled"])
            self.assertEqual(diag["max_steps"], 24)

    def test_agent_hard_off_blocks_decomposition_under_env_and_config(self):
        run_decomp = _extract_agent_decomp_runner()
        inst_text = "  turn left at the doorway, and stop on the stairs.  "

        with patch.dict(os.environ, {"HSGM_DECOMPOSE_INSTRUCTION": "1"}, clear=False):
            # Hard-off must override both enable_subtask_decomposition=True and HSGM_DECOMPOSE_INSTRUCTION=1
            agent_hard_off = NS(decompose_instruction=Mock(return_value=["part1", "part2"]))
            cfg_hard_off = {
                "navigation": {
                    "enable_subtask_decomposition": True,
                    "disable_subtask_decomposition": True,
                }
            }
            run_decomp(agent_hard_off, cfg_hard_off, inst_text)
            agent_hard_off.decompose_instruction.assert_not_called()
            self.assertFalse(agent_hard_off._instruction_decomposition_enabled)
            self.assertEqual(agent_hard_off.instruction_obj.sub_instructions, [inst_text.strip()])

            # Normal mode without hard-off preserves existing behavior
            agent_normal = NS(decompose_instruction=Mock(return_value=["step A", "step B"]))
            cfg_normal = {"navigation": {"enable_subtask_decomposition": False}}
            run_decomp(agent_normal, cfg_normal, inst_text)
            agent_normal.decompose_instruction.assert_called_once_with(inst_text)
            self.assertTrue(agent_normal._instruction_decomposition_enabled)
            self.assertEqual(agent_normal.instruction_obj.sub_instructions, ["step A", "step B"])

    def test_initialize_episode_diagnostic_isolation_and_max_steps_guard(self):
        ExtractedR2REnv, ns = _extract_r2r_env_harness()
        diag_meta = {
            "schema": 1,
            "segment_id": "seg_0",
            "source_episode_id": 101,
            "source_step": 4,
            "source_result_sha256": "a" * 64,
            "source_dataset_sha256": "b" * 64,
            "instruction": "stop on the stairs.",
            "source_original_instruction": "Walk and stop on the stairs.",
            "restore_mode": "rounded_pose_cold_start",
            "decomposition_enabled": False,
            "max_steps": 24,
        }
        diag_ep = {
            "episode_id": 0,
            "scene_id": "mp3d/sceneA/sceneA.glb",
            "start_position": [1.0, 2.0, 3.0],
            "start_rotation": [0.0, 0.0, 0.0, 1.0],
            "instruction": {"instruction_text": "stop on the stairs."},
            "goals": [{"position": [4.0, 2.0, 6.0]}],
            "reference_path": None,
            "info": {"geodesic_distance": 7.0, "geodesic_distance_requires_recalculation": True},
            "diagnostic_segment": diag_meta,
        }
        normal_ep = {
            "episode_id": 1,
            "scene_id": "mp3d/sceneA/sceneA.glb",
            "start_position": [0.0, 0.0, 0.0],
            "start_rotation": [0.0, 0.0, 0.0, 1.0],
            "instruction": {"instruction_text": "Walk and stop on the stairs."},
            "goals": [{"position": [4.0, 2.0, 6.0]}],
            "reference_path": [[0.0, 0.0, 0.0], [4.0, 2.0, 6.0]],
            "info": {"geodesic_distance": 7.0},
        }

        env = ExtractedR2REnv()
        env.all_episodes = [diag_ep, normal_ep]
        env.simWrapper = None
        env.path_calculator = NS(requested_ends=None, requested_start=None)
        env.sim_cfg = {
            "model_name": "gpt-test",
            "navigation": {"enable_subtask_decomposition": True},
        }

        # Reject max_steps > 24 (e.g. default 100 or 25)
        env.env_cfg = {"max_steps": 100}
        with self.assertRaises(ValueError):
            env._initialize_episode(0)
        ns["GPTAgent"].assert_not_called()

        # Reject max_steps <= 0
        env.env_cfg = {"max_steps": 0}
        with self.assertRaises(ValueError):
            env._initialize_episode(0)

        # Valid diagnostic max_steps <= 24 sets hard-off on copied config without polluting normal episode
        env.env_cfg = {"max_steps": 24}
        env._initialize_episode(0)
        self.assertEqual(ns["GPTAgent"].call_count, 1)
        diag_agent_cfg = ns["GPTAgent"].call_args[0][1]
        self.assertTrue(diag_agent_cfg["navigation"]["disable_subtask_decomposition"])
        self.assertFalse(diag_agent_cfg["navigation"]["enable_subtask_decomposition"])
        self.assertEqual(env.current_episode["diagnostic_segment"]["segment_id"], "seg_0")

        # Original env.sim_cfg['navigation'] remains unpolluted for normal episode 1
        self.assertTrue(env.sim_cfg["navigation"]["enable_subtask_decomposition"])
        self.assertNotIn("disable_subtask_decomposition", env.sim_cfg["navigation"])

        env._initialize_episode(1)
        normal_agent_cfg = ns["GPTAgent"].call_args[0][1]
        self.assertTrue(normal_agent_cfg["navigation"]["enable_subtask_decomposition"])
        self.assertFalse(normal_agent_cfg["navigation"].get("disable_subtask_decomposition", False))
        self.assertNotIn("diagnostic_segment", env.current_episode)

    def test_save_episode_results_diagnostic_tags_and_normal_unchanged(self):
        ExtractedR2REnv, _ = _extract_r2r_env_harness()
        diag_meta = {
            "schema": 1,
            "segment_id": "seg_save_test",
            "source_episode_id": 101,
            "source_step": 4,
            "source_result_sha256": "a" * 64,
            "source_dataset_sha256": "b" * 64,
            "instruction": "stop on the stairs.",
            "source_original_instruction": "Walk and stop on the stairs.",
            "restore_mode": "rounded_pose_cold_start",
            "decomposition_enabled": False,
            "max_steps": 24,
        }
        metrics = {
            "distance_to_goal": 1.25,
            "oracle_navigation_error": 1.1,
            "traveled_distance": 3.4,
            "success_threshold": 3.0,
        }
        subtasks_p1 = {"P1": [{"step": 1, "executed_action": {"step": 1, "action_type": "stop"}}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            env = ExtractedR2REnv()
            env.env_cfg = {"name": "diag_unit", "output_root": tmpdir}
            env.agent_trajectory = [[1.0, 2.0, 3.0]]
            env.agent = NS(
                _execution_ledger=[{"step": 1, "action_type": "stop"}],
                _instruction_decomposition_enabled=False,
                instruction_obj=NS(sub_instruction_keys=["P1"], num_subtasks=1),
                navigation_trace=[],
            )

            # 1. Diagnostic episode writes diagnostic_segment and instruction_decomposition_enabled=False
            env.current_episode = {"diagnostic_segment": diag_meta}
            env._save_episode_results(0, [], dict(metrics), [], subtasks=subtasks_p1)
            diag_res_path = Path(tmpdir) / "results_diag_unit" / "episode_0_results.json"
            diag_saved = json.loads(diag_res_path.read_text(encoding="utf-8"))
            self.assertEqual(diag_saved["diagnostic_segment"], diag_meta)
            self.assertIs(diag_saved["instruction_decomposition_enabled"], False)

            self.assertTrue(diag_saved['diagnostic_contract_satisfied'])
            # Preserve error results honestly instead of inventing single-P1 evidence.
            env._save_episode_results(0, [], dict(metrics), [], subtasks={"P1": [], "P2": []})
            invalid = json.loads(diag_res_path.read_text())
            self.assertFalse(invalid['diagnostic_contract_satisfied'])
            del env.agent._instruction_decomposition_enabled
            env._save_episode_results(0, [], dict(metrics), [], subtasks=subtasks_p1)
            unknown = json.loads(diag_res_path.read_text())
            self.assertIsNone(unknown['instruction_decomposition_enabled'])
            self.assertFalse(unknown['diagnostic_contract_satisfied'])

            # 3. Normal episode does not add diagnostic keys
            env.current_episode = {"instruction": "normal"}
            env._save_episode_results(1, [], dict(metrics), [], subtasks=subtasks_p1)
            normal_res_path = Path(tmpdir) / "results_diag_unit" / "episode_1_results.json"
            normal_saved = json.loads(normal_res_path.read_text(encoding="utf-8"))
            self.assertNotIn("diagnostic_segment", normal_saved)
            self.assertNotIn("instruction_decomposition_enabled", normal_saved)


if __name__ == "__main__":
    unittest.main()

