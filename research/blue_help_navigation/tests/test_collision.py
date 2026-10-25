import math
import unittest
from blue_help.collision import Cylinder, first_blocker, swept_cylinder_hits


class CollisionTests(unittest.TestCase):
    def setUp(self):
        self.body = Cylinder("person", (0, 0, 0), .32, 1.6)

    def hit(self, a, b, **kwargs):
        return swept_cylinder_hits(a, b, self.body, agent_radius_m=.18,
                                   agent_height_m=1.5, margin_m=.03, **kwargs)

    def test_continuous_tunneling_endpoints_clear(self):
        self.assertTrue(self.hit((-1, 0, 0), (1, 0, 0)))
        self.assertFalse(self.hit((-1, 0, 0), (-1, 0, 0)))
        self.assertFalse(self.hit((1, 0, 0), (1, 0, 0)))

    def test_narrow_body_not_one_metre_interaction_disk(self):
        self.assertFalse(self.hit((.6, 0, -1), (.6, 0, 1)))
        self.assertTrue(self.hit((.53, 0, -1), (.53, 0, 1)))

    def test_other_floor_not_physical_overlap(self):
        self.assertFalse(self.hit((-1, 3, 0), (1, 3, 0)))
        self.assertFalse(self.hit((-1, -2, 0), (1, -2, 0)))

    def test_vertical_overlap_checked_over_segment(self):
        self.assertTrue(self.hit((-1, 3, 0), (1, 0, 0)))
        self.assertFalse(self.hit((-1, 10, 0), (1, 3, 0)))

    def test_start_overlap_and_zero_length(self):
        self.assertTrue(self.hit((0, 0, 0), (0, 0, 0)))
        self.assertTrue(self.hit((0, 0, 0), (2, 0, 0)))

    def test_invalid_coordinates_fail(self):
        with self.assertRaises(ValueError):
            self.hit((math.nan, 0, 0), (0, 0, 0))
        with self.assertRaises(ValueError):
            Cylinder("x", (0, 0, 0), -1, 1)

    def test_first_blocker_no_obstacles(self):
        self.assertIsNone(first_blocker((0, 0, 0), (1, 0, 0), [],
                                        agent_radius_m=.18, agent_height_m=1.5))
