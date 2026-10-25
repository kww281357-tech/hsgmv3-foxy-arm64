"""Lost-detection tests use only synthetic observed poses/actions, never goals."""
import inspect
import math
import unittest

from blue_help.contracts import Pose, SensorFrame
from blue_help.lost import LostConfig, LostDetector


def pose(x=0, z=0, yaw=0, y=0):
    return Pose((x, y, z), yaw)


class LostTests(unittest.TestCase):
    def test_basic_progress_is_not_lost(self):
        detector = LostDetector()
        for step in range(40):
            status = detector.update(pose(z=-step*.25), step=step, previous_action='move_forward')
            self.assertFalse(status.is_lost)

    def test_basic_curve_with_progress_is_not_lost(self):
        detector = LostDetector()
        for step in range(20):
            position = pose(x=max(0, step-8)*.25, z=-min(step, 8)*.25,
                            yaw=-math.pi/2 if step > 8 else 0)
            status = detector.update(position, step=step, previous_action='move_forward')
            self.assertFalse(status.is_lost)

    def test_persistent_uncertainty_requires_patience(self):
        detector = LostDetector()
        self.assertFalse(detector.update(pose(), step=0, uncertainty=.8).is_lost)
        self.assertFalse(detector.update(pose(), step=1, uncertainty=.8).is_lost)
        status = detector.update(pose(), step=2, uncertainty=.8)
        self.assertTrue(status.is_lost)
        self.assertTrue(status.newly_lost)
        self.assertEqual(status.reasons, ('persistent_public_uncertainty',))

    def test_repeated_triggers_are_not_new_lost_edges(self):
        detector = LostDetector({'uncertainty_patience': 1})
        self.assertTrue(detector.update(pose(), step=0, uncertainty=.9).newly_lost)
        for step in range(1, 20):
            status = detector.update(pose(), step=step, uncertainty=.9)
            self.assertTrue(status.is_lost)
            self.assertFalse(status.newly_lost)

    def test_recovery_has_hysteresis_then_can_trigger_new_episode_of_uncertainty(self):
        detector = LostDetector({'uncertainty_patience': 1, 'recovery_patience': 3})
        detector.update(pose(), step=0, uncertainty=.9)
        self.assertTrue(detector.update(pose(z=-1), step=1).is_lost)
        self.assertTrue(detector.update(pose(z=-2), step=2).is_lost)
        self.assertFalse(detector.update(pose(z=-3), step=3).is_lost)
        self.assertTrue(detector.update(pose(z=-4), step=4, uncertainty=.9).newly_lost)

    def test_observed_collisions_trigger_without_goal_progress_metric(self):
        detector = LostDetector()
        for step in range(3):
            status = detector.update(pose(), step=step, collision=True, previous_action='move_forward')
        self.assertTrue(status.is_lost)
        self.assertIn('repeated_observed_collisions', status.reasons)

    def test_waiting_does_not_amplify_stale_collision_flag(self):
        detector = LostDetector()
        for step in range(15):
            status = detector.update(pose(), step=step, collision=True, previous_action='wait')
        self.assertFalse(status.is_lost)

    def test_stalled_forward_motion_requires_forward_attempts(self):
        detector = LostDetector()
        for step in range(6):
            status = detector.update(pose(), step=step, previous_action='move_forward')
        self.assertTrue(status.is_lost)
        self.assertIn('stalled_forward_motion', status.reasons)
        for action in ('turn_left', 'turn_right', 'wait', 'stop', None):
            detector.reset()
            for step in range(12):
                status = detector.update(pose(yaw=.2*step), step=step, previous_action=action)
            self.assertFalse(status.is_lost, action)

    def test_revisit_requires_observed_travel_and_time_separation(self):
        detector = LostDetector({'min_revisit_gap_steps': 4})
        locations = [(0,0), (1,0), (1,-1), (0,-1), (0,0)]
        for step, (x,z) in enumerate(locations):
            status = detector.update(pose(x,z), step=step, previous_action='move_forward')
        self.assertTrue(status.is_lost)
        self.assertIn('revisited_observed_location', status.reasons)

    def test_position_noise_does_not_fake_a_travelled_loop(self):
        detector = LostDetector()
        for step in range(20):
            status = detector.update(pose(x=.005*(step % 2)), step=step, previous_action='wait')
        self.assertFalse(status.is_lost)

    def test_reset_removes_streaks_history_and_repeat_state(self):
        detector = LostDetector({'uncertainty_patience': 1})
        detector.update(pose(), step=8, uncertainty=1)
        detector.reset()
        status = detector.update(pose(), step=0)
        self.assertFalse(status.is_lost)
        self.assertEqual(status.reasons, ())

    def test_identical_observation_idempotent_conflict_or_stale_rejected(self):
        detector = LostDetector({'uncertainty_patience': 2})
        first = detector.update(pose(), step=3, uncertainty=.9)
        self.assertEqual(detector.update(pose(), step=3, uncertainty=.9), first)
        self.assertFalse(first.is_lost)
        with self.assertRaises(ValueError):
            detector.update(pose(x=1), step=3, uncertainty=.9)
        with self.assertRaises(ValueError):
            detector.update(pose(), step=2)
        self.assertTrue(detector.update(pose(), step=4, uncertainty=.9).is_lost)

    def test_observation_gap_does_not_fake_consecutive_evidence(self):
        detector = LostDetector()
        detector.update(pose(), step=0, collision=True, uncertainty=.9)
        detector.update(pose(), step=1, collision=True, uncertainty=.9)
        status = detector.update(pose(), step=20, collision=True, uncertainty=.9)
        self.assertFalse(status.is_lost)

    def test_absent_or_unreachable_dummy_does_not_change_lost_evidence(self):
        empty = LostDetector({'uncertainty_patience': 1})
        unreachable = LostDetector({'uncertainty_patience': 1})
        a = SensorFrame(None, None, pose(), 0, dummy_candidates=[])
        b = SensorFrame(None, None, pose(), 0, dummy_candidates=[
            {'id': 'not-eligible', 'reachable': False, 'private_goal': 'must-be-ignored'}])
        self.assertEqual(empty.update_frame(a, uncertainty=.9), unreachable.update_frame(b, uncertainty=.9))
        self.assertTrue(empty.update_frame(a, uncertainty=.9).is_lost)

    def test_no_oracle_goal_path_or_success_inputs_exist(self):
        parameters = set(inspect.signature(LostDetector.update).parameters)
        self.assertTrue(parameters.isdisjoint({'goal', 'goal_position', 'path', 'reference_path',
                                               'distance_to_goal', 'geodesic_distance', 'success', 'oracle'}))
        self.assertEqual(parameters, {'self', 'pose', 'step', 'collision', 'uncertainty', 'previous_action'})
        detector = LostDetector()
        self.assertFalse(any('oracle' in name or 'goal' in name for name in vars(detector)))

    def test_validation_fails_closed(self):
        for config in ({'uncertainty_threshold': math.nan}, {'collision_patience': 0},
                       {'stagnation_window': 2}, {'min_translation_m': -1},
                       {'revisit_window': 5, 'min_revisit_gap_steps': 6}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                LostConfig(**config)
        detector = LostDetector()
        for uncertainty in (math.inf, -1, True, 1.1):
            with self.subTest(uncertainty=uncertainty), self.assertRaises(ValueError):
                detector.update(pose(), step=0, uncertainty=uncertainty)
        with self.assertRaises(ValueError):
            detector.update(pose(x=math.nan), step=0)
        with self.assertRaises(ValueError):
            detector.update(pose(), step=True)


if __name__ == '__main__':
    unittest.main()
