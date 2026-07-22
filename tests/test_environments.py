# type: ignore

import sys
import unittest

import numpy as np

sys.path.append(".")

from c4dynamics.envs.environments import helicopter


class TestHelicopterEnvironment(unittest.TestCase):

    def setUp(self):
        self.env = helicopter(theta=0.1, psi=0.2, dtheta=-0.3, dpsi=0.4)

    def test_initial_state(self):
        self.assertTrue(np.allclose(self.env.X, np.array([0.1, 0.2, -0.3, 0.4])))

    def test_force_and_control_effectiveness(self):
        drift = self.env.F()
        control_matrix = self.env.G()

        self.assertEqual(drift.shape, (2,))
        self.assertEqual(control_matrix.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(drift)))
        self.assertTrue(np.all(np.isfinite(control_matrix)))

    def test_dynamics_returns_expected_shape(self):
        state_vector = np.array([0.1, 0.2, -0.3, 0.4])
        control_input = np.array([1.0, -1.0])

        derivative = self.env.dynamics(0.0, state_vector, control_input)

        self.assertEqual(derivative.shape, (4,))
        self.assertTrue(np.all(np.isfinite(derivative)))

    def test_reference_trajectory_shape(self):
        xd, xd_d, xd_dd = self.env.reference(0.5)

        self.assertEqual(xd.shape, (2,))
        self.assertEqual(xd_d.shape, (2,))
        self.assertEqual(xd_dd.shape, (2,))

    def test_split_state_vector(self):
        state_vector = np.array([0.1, 0.2, -0.3, 0.4])
        pos, vel = self.env.split(state_vector)

        self.assertTrue(np.allclose(pos, np.array([0.1, 0.2])))
        self.assertTrue(np.allclose(vel, np.array([-0.3, 0.4])))


if __name__ == "__main__":
    unittest.main()
