import unittest

from week4.experiments.experiment_09_multijoint_gripper import build_sequence as build_09
from week4.experiments.experiment_10_pick_and_place import build_sequence as build_10


class SequenceTests(unittest.TestCase):
    def test_experiment_09_frame_count_and_home(self):
        sequence = build_09()
        self.assertEqual(sum(step.steps for step in sequence), 180)
        self.assertEqual(tuple(sequence[-1].target_deg), (0.0,) * 6)

    def test_experiment_10_frame_count_and_home(self):
        sequence = build_10()
        self.assertEqual(sum(step.steps for step in sequence), 160)
        self.assertEqual(tuple(sequence[-1].target_deg), (0.0,) * 6)


if __name__ == "__main__":
    unittest.main()

