"""CPU-only oracle tests; supplied polylines are explicit synthetic fixtures."""
from dataclasses import fields
import math
import re
import unittest

from blue_help.contracts import Pose
from blue_help.oracle import (COCO80_NAMES, DirectionAnswer, OracleConfig, RuleDirectionOracle,
                              VerifiedLandmark, answer_is_stale)


class OracleTests(unittest.TestCase):
    def oracle(self, path, goal=None, **kwargs):
        goal = goal if goal is not None else path[-1]
        return RuleDirectionOracle(goal, lambda start, end: path, **kwargs)

    def test_origin_is_vehicle_question_pose_not_dummy_foot(self):
        calls = []
        pose = Pose((2., 0., 5.), math.pi / 2)
        def safe_path(start, end):
            calls.append((start, end))
            return [start, end]
        oracle = RuleDirectionOracle((-2, 0, 5), safe_path)
        answer = oracle.answer(pose, 7)
        self.assertEqual(calls, [((2., 0., 5.), (-2., 0., 5.))])
        self.assertEqual(answer.reference_pose, pose)
        self.assertEqual(answer.question_step, 7)
        self.assertIn('go forward about 4.0 meters', answer.text)
        self.assertNotIn('turn left', answer.text)
        self.assertNotIn('turn right', answer.text)

    def test_answer_schema_contains_no_goal_or_path(self):
        self.assertEqual({f.name for f in fields(DirectionAnswer)},
                         {'text', 'reference_pose', 'question_step', 'terminal', 'landmark_ids'})
        answer = self.oracle([(12, 0, 8), (12, 0, 4)]).answer(Pose((12, 0, 8), 0), 0)
        self.assertNotIn('12', answer.text)
        self.assertNotIn('coordinate', answer.text)
        self.assertNotIn('waypoint', answer.text)
        self.assertFalse(hasattr(answer, 'goal_position'))
        self.assertFalse(hasattr(answer, 'path'))
        self.assertFalse(hasattr(answer, 'waypoints'))

    def test_left_right_and_back_are_relative_to_current_vehicle_heading(self):
        cases = [((-2, 0, 0), 0, 'turn left about 90 degrees'),
                 ((2, 0, 0), 0, 'turn right about 90 degrees'),
                 ((0, 0, 2), 0, 'turn around'),
                 ((0, 0, -2), math.pi / 2, 'turn right about 90 degrees')]
        for target, yaw, expected in cases:
            answer = self.oracle([(0, 0, 0), target]).answer(Pose((0, 0, 0), yaw), 0)
            with self.subTest(target=target, yaw=yaw):
                self.assertIn(expected, answer.text)

    def test_basic_curve_uses_sequential_ego_frame_and_preserves_corners(self):
        path = [(0, 0, 0), (0, 0, -2), (2, 0, -2), (2, 0, -4)]
        answer = self.oracle(path).answer(Pose((0, 0, 0), 0), 0)
        self.assertTrue(answer.terminal)
        self.assertEqual(answer.text.count('go forward'), 3)
        self.assertIn('turn right about 90 degrees', answer.text)
        self.assertIn('turn left about 90 degrees', answer.text)
        self.assertLess(answer.text.index('turn right'), answer.text.index('turn left'))

    def test_maximum_eight_metres_is_a_local_partial_answer(self):
        answer = self.oracle([(0, 0, 0), (0, 0, -20)]).answer(Pose((0, 0, 0), 0), 0)
        self.assertFalse(answer.terminal)
        self.assertIn('8.0 meters', answer.text)
        self.assertNotIn('20', answer.text)
        self.assertIn('not confirmation of arrival', answer.text)

    def test_maximum_three_segments_does_not_shortcut_fourth_corner(self):
        path = [(0, 0, 0), (0, 0, -1), (1, 0, -1), (1, 0, -2), (2, 0, -2)]
        answer = self.oracle(path).answer(Pose((0, 0, 0), 0), 0)
        self.assertFalse(answer.terminal)
        self.assertEqual(answer.text.count('go forward'), 3)
        distances = [float(x) for x in re.findall(r'forward about ([\d.]+) meters', answer.text)]
        self.assertEqual(sum(distances), 3)

    def test_dense_collinear_vertices_merge_without_using_segment_budget(self):
        path = [(0, 0, -n) for n in range(6)]
        answer = self.oracle(path).answer(Pose((0, 0, 0), 0), 0)
        self.assertTrue(answer.terminal)
        self.assertEqual(answer.text.count('go forward'), 1)
        self.assertIn('5.0 meters', answer.text)

    def test_reversing_path_is_not_merged_into_a_shortcut(self):
        path = [(0, 0, 0), (0, 0, -2), (0, 0, -1), (1, 0, -1)]
        answer = self.oracle(path).answer(Pose((0, 0, 0), 0), 0)
        self.assertEqual(answer.text.count('go forward'), 3)
        self.assertIn('turn around', answer.text)

    def test_unreachable_path_returns_no_answer(self):
        oracle = RuleDirectionOracle((0, 0, -2), lambda start, end: None)
        self.assertIsNone(oracle.answer(Pose((0, 0, 0), 0), 0))

    def test_backend_failure_returns_no_invented_route(self):
        def unavailable(start, end):
            raise RuntimeError('synthetic pathfinder unavailable')
        oracle = RuleDirectionOracle((0, 0, -2), unavailable)
        self.assertIsNone(oracle.answer(Pose((0, 0, 0), 0), 0))

    def test_invalid_partial_nonfinite_or_wrong_origin_paths_fail_closed(self):
        routes = [[], [(0, 0, 0)], [(0, 0, 0), (0, 0, -1)],
                  [(9, 0, 0), (0, 0, -2)], [(0, 0, 0), (0, math.nan, -2)]]
        for path in routes:
            with self.subTest(path=path):
                self.assertIsNone(self.oracle(path, goal=(0, 0, -2)).answer(Pose((0, 0, 0), 0), 0))

    def test_vertical_only_path_does_not_invent_stairs_or_elevator(self):
        self.assertIsNone(self.oracle([(0, 0, 0), (0, 2, 0)]).answer(Pose((0, 0, 0), 0), 0))

    def test_sloped_path_described_without_invented_stair_type(self):
        answer = self.oracle([(0, 0, 0), (0, .5, -2)]).answer(Pose((0, 0, 0), 0), 0)
        self.assertIn('rising route', answer.text)
        self.assertNotIn('stairs', answer.text)

    def test_arrival_requires_available_safe_path(self):
        oracle = RuleDirectionOracle((0, 0, 0), lambda a, b: None)
        self.assertIsNone(oracle.answer(Pose((0, 0, 0), 0), 0))
        answer = self.oracle([(0, 0, 0)]).answer(Pose((0, 0, 0), 0), 0)
        self.assertTrue(answer.terminal)
        self.assertIn('already at the destination', answer.text)

    def test_nearby_euclidean_goal_with_long_safe_route_is_not_arrival(self):
        path = [(0, 0, 0), (0, 0, -1), (.1, 0, -1), (.1, 0, 0)]
        answer = self.oracle(path).answer(Pose((0, 0, 0), 0), 0)
        self.assertNotIn('already at the destination', answer.text)
        self.assertEqual(answer.text.count('go forward'), 3)

    def test_unknown_wrong_turn_cause_is_not_invented_from_oracle_geometry(self):
        answer = self.oracle([(0, 0, 0), (0, 0, 2)]).answer(Pose((0, 0, 0), 0), 0,
            public_history=[{'step': 0, 'made_up_reason': 'You chose the wrong corridor'}])
        self.assertIn('cause of getting lost is unknown', answer.text)
        self.assertNotIn('wrong corridor', answer.text)

    def test_only_fresh_distinct_observed_collisions_support_cause_text(self):
        oracle = self.oracle([(0, 0, 0), (0, 0, -2)])
        history = [{'step': 8, 'collision': True}, {'step': 9, 'collision': True}]
        answer = oracle.answer(Pose((0, 0, 0), 0), 10, history)
        self.assertIn('repeated collisions', answer.text)
        self.assertIn('wrong turn is unknown', answer.text)
        duplicate = oracle.answer(Pose((0, 0, 0), 0), 10, [history[0], history[0]])
        self.assertIn('cause of getting lost is unknown', duplicate.text)
        stale = oracle.answer(Pose((0, 0, 0), 0), 30, history)
        self.assertIn('cause of getting lost is unknown', stale.text)

    def test_stalled_forward_history_supports_only_observed_symptom(self):
        history = [dict(step=n, action='move_forward', displacement_m=0.) for n in (4, 5)]
        answer = self.oracle([(0, 0, 0), (0, 0, -2)]).answer(Pose((0, 0, 0), 0), 6, history)
        self.assertIn('very little movement', answer.text)
        self.assertIn('wrong turn is unknown', answer.text)

    def test_default_answer_invents_no_landmark(self):
        answer = self.oracle([(0, 0, 0), (0, 0, -2)]).answer(Pose((0, 0, 0), 0), 0)
        self.assertEqual(answer.landmark_ids, ())
        self.assertNotIn('chair', answer.text)
        self.assertNotIn('door', answer.text)
        self.assertEqual(len(COCO80_NAMES), 80)

    def test_only_explicit_verified_matching_scene_landmark_can_be_named(self):
        landmark = VerifiedLandmark('chair-17', 'chair', (.1, 0, -2), 'scene-A', 'semantic-instance:17; observed-frame:4')
        answer = self.oracle([(0, 0, 0), (0, 0, -2)], scene_id='scene-A',
            verified_landmarks=(landmark,)).answer(Pose((0, 0, 0), 0), 0)
        self.assertEqual(answer.landmark_ids, ('chair-17',))
        self.assertIn('scene-verified chair', answer.text)
        for scene in (None, 'scene-B'):
            other = self.oracle([(0, 0, 0), (0, 0, -2)], scene_id=scene,
                verified_landmarks=(landmark,)).answer(Pose((0, 0, 0), 0), 0)
            self.assertEqual(other.landmark_ids, ())
            self.assertNotIn('chair', other.text)

    def test_landmark_without_supported_name_or_evidence_is_rejected(self):
        with self.assertRaises(ValueError):
            VerifiedLandmark('x', 'doorway', (0, 0, 0), 'scene', 'semantic:7')
        with self.assertRaises(ValueError):
            VerifiedLandmark('x', 'chair', (0, 0, 0), 'scene', '')

    def test_stale_direction_check_detects_age_translation_and_rotation(self):
        answer = self.oracle([(0, 0, 0), (0, 0, -2)]).answer(Pose((0, 0, 0), 0), 4)
        self.assertFalse(answer_is_stale(answer, Pose((0, 0, 0), 2*math.pi), 5))
        self.assertTrue(answer_is_stale(answer, Pose((.4, 0, 0), 0), 5))
        self.assertTrue(answer_is_stale(answer, Pose((0, 0, 0), math.pi/2), 5))
        self.assertTrue(answer_is_stale(answer, Pose((0, 0, 0), 0), 13))
        self.assertTrue(answer_is_stale(answer, Pose((0, 0, 0), 0), 3))

    def test_repeat_query_uses_new_vehicle_frame(self):
        oracle = self.oracle([(0, 0, 0), (0, 0, -2)])
        a = oracle.answer(Pose((0, 0, 0), 0), 1)
        b = oracle.answer(Pose((0, 0, 0), math.pi/2), 2)
        self.assertNotIn('turn right', a.text)
        self.assertIn('turn right', b.text)
        self.assertNotEqual(a.reference_pose, b.reference_pose)

    def test_small_real_curve_does_not_silently_drop_a_turn(self):
        length = 2.0
        angle = math.radians(4)
        path = [(0, 0, 0), (0, 0, -1),
                (-math.sin(angle)*length, 0, -1-math.cos(angle)*length)]
        answer = self.oracle(path).answer(Pose((0, 0, 0), 0), 0)
        self.assertIn('turn left about 4 degrees', answer.text)
        self.assertEqual(answer.text.count('go forward'), 2)

    def test_negative_or_waiting_collision_records_do_not_invent_cause(self):
        oracle = self.oracle([(0, 0, 0), (0, 0, -2)])
        for history in ([{'step': -1, 'collision': True}, {'step': -2, 'collision': True}],
                        [{'step': 0, 'collision': True, 'action': 'wait'},
                         {'step': 1, 'collision': True, 'action': 'wait'}]):
            answer = oracle.answer(Pose((0, 0, 0), 0), 2, history)
            self.assertIn('cause of getting lost is unknown', answer.text)

    def test_canonical_stall_evidence_from_actual_runner_history(self):
        from blue_help.contracts import Episode, SensorFrame
        from blue_help.policy import Decision, ReplayPolicy
        from blue_help.runner import RunConfig, run_episode
        image = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jDxkAAAAASUVORK5CYII='
        class StationaryEnv:
            def reset(self, episode, dummies, seed=42):
                self.tick = 0
                return self.observe()
            def observe(self):
                return SensorFrame(image, None, Pose((0, 0, 0), 0), self.tick)
            def step(self, action):
                self.tick += 1
                return self.observe()
            def find_safe_path(self, start, end):
                return [start, end]
        episode = Episode('40', 'synthetic-test-scene', 'Move forward.', (0,0,0), (0,0,0,1), (0,0,-2))
        run = run_episode(StationaryEnv(), ReplayPolicy([Decision('move_forward'),
            Decision('move_forward'), Decision('stop')]), episode, [], arm='no_ask',
            config=RunConfig(max_steps=3, backend_label='mock_unit', policy_label='replay_test'),
            metric_distance=math.dist)
        self.assertIsNone(run['error'])
        self.assertIn('displacement_m', run['history'][0])
        answer = self.oracle([(0,0,0),(0,0,-2)]).answer(Pose((0,0,0),0), 3, run['history'])
        self.assertIn('very little movement', answer.text)
        self.assertIn('wrong turn is unknown', answer.text)

    def test_caps_and_inputs_are_validated(self):
        for config in ({'max_distance_m': 8.1}, {'max_segments': 4}, {'max_segments': True},
                       {'max_distance_m': math.nan}, {'goal_radius_m': 0}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                OracleConfig(**config)
        oracle = self.oracle([(0, 0, 0), (0, 0, -2)])
        with self.assertRaises(ValueError):
            oracle.answer(Pose((0, 0, math.inf), 0), 0)
        with self.assertRaises(ValueError):
            oracle.answer(Pose((0, 0, 0), 0), True)


if __name__ == '__main__':
    unittest.main()
