import unittest

import numpy as np

from week4.experiments.experiment_08_fk_chain import solve_forward_kinematics


class ForwardKinematicsTests(unittest.TestCase):
    def test_matches_guide_result(self):
        transform, rotation, position, _ = solve_forward_kinematics()
        expected_transform = np.array(
            [
                [0.6124, 0.3536, 0.7071, 2.3991],
                [0.6124, 0.3536, -0.7071, 2.3991],
                [-0.5, 0.8660, 0.0, 1.1],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(transform, expected_transform, atol=5e-5)
        np.testing.assert_allclose(rotation, transform[:3, :3], atol=1e-12)
        np.testing.assert_allclose(position, transform[:3, 3], atol=1e-12)


if __name__ == "__main__":
    unittest.main()

