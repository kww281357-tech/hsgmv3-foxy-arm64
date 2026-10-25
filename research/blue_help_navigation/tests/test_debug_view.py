import hashlib
import unittest
import numpy as np
from blue_help.debug_view import draw_interaction_circle, draw_topdown_inset
from blue_help.contracts import DummySpec, Pose


class DebugViewTests(unittest.TestCase):
    def test_overlay_is_separate_copy_and_does_not_change_raw(self):
        rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        rgb.setflags(write=False)
        before = hashlib.sha256(rgb.tobytes()).hexdigest()
        result = draw_interaction_circle(rgb, robot_ground_position=(0,0,0),
                   camera_position=(0,.5,0), camera_rotation_xyzw=(0,0,0,1))
        self.assertEqual(before, hashlib.sha256(rgb.tobytes()).hexdigest())
        self.assertFalse(np.shares_memory(rgb, result))
        self.assertGreater(np.count_nonzero(result), 0)
        self.assertTrue(np.any(result[30:] != rgb[30:]))  # circle, not only text

    def test_invalid_projection_rejected(self):
        with self.assertRaises(ValueError):
            draw_interaction_circle(np.zeros((4,4,3), dtype=np.uint8),
                robot_ground_position=(0,0,0), camera_position=(0,1,0),
                camera_rotation_xyzw=(0,0,0,2))

    def test_topdown_is_separate_anonymous_debug_geometry(self):
        rgb = np.zeros((240,320,3),dtype=np.uint8)
        rgb.setflags(write=False)
        before = hashlib.sha256(rgb.tobytes()).hexdigest()
        result = draw_topdown_inset(rgb,robot_pose=Pose((10,2,-3),0),
            dummies=[DummySpec("private_room_name",(10,2,-3.8),0)])
        self.assertEqual(before,hashlib.sha256(rgb.tobytes()).hexdigest())
        self.assertFalse(np.shares_memory(rgb,result))
        self.assertGreater(np.count_nonzero(result),0)
        # Changing a privileged room name must not change the anonymous diagram.
        other = draw_topdown_inset(rgb,robot_pose=Pose((10,2,-3),0),
            dummies=[DummySpec("different_metadata",(10,2,-3.8),0)])
        np.testing.assert_array_equal(result,other)

    def test_topdown_egocentric_rotation_has_consistent_geometry(self):
        rgb = np.zeros((240,320,3),dtype=np.uint8)
        front = draw_topdown_inset(rgb,robot_pose=Pose((0,0,0),0),
            dummies=[DummySpec("a",(0,0,-.8),0)])
        left_forward = draw_topdown_inset(rgb,robot_pose=Pose((0,0,0),np.pi/2),
            dummies=[DummySpec("a",(-.8,0,0),0)])
        np.testing.assert_array_equal(front,left_forward)

    def test_topdown_rejects_tiny_images(self):
        with self.assertRaises(ValueError):
            draw_topdown_inset(np.zeros((40,40,3),dtype=np.uint8),
                robot_pose=Pose((0,0,0),0),dummies=[])
