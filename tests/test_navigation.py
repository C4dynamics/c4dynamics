# type: ignore

import unittest
import numpy as np

import sys

sys.path.append("")
import c4dynamics as c4d
from c4dynamics.sensors.navigation import gps, imu, magnetometer


STATE_NAMES = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r"]


def make_state(**kw):
    x = np.zeros(12)
    for k, v in kw.items():
        x[STATE_NAMES.index(k)] = v
    return x


def make_rigidbody(**kw):
    # imu.measure() takes the true reference as a rigidbody object.
    return c4d.rigidbody(**kw)


class TestGPS(unittest.TestCase):

    def setUp(self):
        self.x_true = make_state(x=1.0, y=-2.0, z=3.5)

    def test_initialization(self):
        sensor = gps(noise_std=0.5)
        self.assertEqual(sensor.noise_std, 0.5)
        np.testing.assert_array_equal(sensor.bias, np.zeros(3))

    def test_zero_noise_is_exact(self):
        sensor = gps(noise_std=0.0)
        np.random.seed(0)
        z = sensor.measure(self.x_true)
        np.testing.assert_array_almost_equal(z, [1.0, -2.0, 3.5])

    def test_noise_statistics(self):
        np.random.seed(0)
        sensor = gps(noise_std=0.5)
        x = make_state(x=0.0, y=0.0, z=0.0)
        samples = np.array([sensor.measure(x) for _ in range(20000)])
        np.testing.assert_array_almost_equal(samples.mean(axis=0), [0, 0, 0], decimal=2)
        np.testing.assert_allclose(samples.std(axis=0), 0.5, rtol=0.05)

    def test_bias_is_additive(self):
        sensor = gps(noise_std=0.0, bias=[1.0, -1.0, 0.5])
        x = make_state(x=0.0, y=0.0, z=0.0)
        z = sensor.measure(x)
        np.testing.assert_array_almost_equal(z, [1.0, -1.0, 0.5])


class TestIMU(unittest.TestCase):

    def test_gyro_zero_noise_passes_through(self):
        sensor = imu(gyro_std=0.0)
        rb = make_rigidbody(p=0.1, q=-0.2, r=0.05)
        _, _, _, p, q, r = sensor.measure(rb)
        np.testing.assert_array_almost_equal([p, q, r], [0.1, -0.2, 0.05])

    def test_gyro_bias(self):
        sensor = imu(gyro_std=0.0, gyro_bias=[0.01, -0.02, 0.03])
        rb = make_rigidbody(p=0.0, q=0.0, r=0.0)
        _, _, _, p, q, r = sensor.measure(rb)
        np.testing.assert_array_almost_equal([p, q, r], [0.01, -0.02, 0.03])

    def test_accelerometer_gravity_only_at_rest(self):
        # On the first call there's no previous sample yet, so the
        # accelerometer reports the gravity projection alone:
        # [g*sin(theta), -g*sin(phi)*cos(theta), -g*cos(phi)*cos(theta)].
        sensor = imu(acc_std=0.0)
        phi, theta = 0.1, -0.15
        rb = make_rigidbody(phi=phi, theta=theta)
        ax, ay, az, _, _, _ = sensor.measure(rb)
        g = 9.81
        expected = [
            g * np.sin(theta),
            -g * np.sin(phi) * np.cos(theta),
            -g * np.cos(phi) * np.cos(theta),
        ]
        np.testing.assert_array_almost_equal([ax, ay, az], expected)

    def test_accelerometer_level_hover_reads_gravity_on_z_only(self):
        sensor = imu(acc_std=0.0)
        rb = make_rigidbody(phi=0.0, theta=0.0)
        ax, ay, az, _, _, _ = sensor.measure(rb)
        np.testing.assert_array_almost_equal([ax, ay, az], [0.0, 0.0, -9.81])

    def test_accelerometer_dynamic_term_requires_prev_state(self):
        # The first measure() call has no previous sample (gravity only);
        # imu keeps that sample internally, so the *next* call to the same
        # sensor picks up the dynamic (finite-difference) term automatically.
        sensor = imu(acc_std=0.0, dt=0.1)
        rb = make_rigidbody(phi=0.0, theta=0.0, vx=0.0)
        z_no_dyn = sensor.measure(rb)[:3]

        rb.vx = 1.0
        z_with_dyn = sensor.measure(rb)[:3]
        self.assertFalse(np.allclose(z_no_dyn, z_with_dyn))

    def test_accelerometer_bias(self):
        sensor = imu(acc_std=0.0, acc_bias=[0.1, -0.05, 0.2])
        rb = make_rigidbody(phi=0.0, theta=0.0)
        ax, ay, az, _, _, _ = sensor.measure(rb)
        np.testing.assert_array_almost_equal([ax, ay, az], [0.1, -0.05, -9.81 + 0.2])

    def test_measure_returns_tuple_of_accelerations_and_rates(self):
        sensor = imu(isideal=True)
        rb = make_rigidbody(phi=0.0, theta=0.0, p=1.0, q=2.0, r=3.0)
        out = sensor.measure(rb)
        self.assertEqual(len(out), 6)
        np.testing.assert_array_almost_equal(out[3:], [1.0, 2.0, 3.0])

    def test_store_records_sample_with_timestamp(self):
        sensor = imu(isideal=True)
        rb = make_rigidbody(p=0.5)
        sensor.measure(rb, t=1.5, store=True)
        t_hist, p_hist = sensor.data('p')
        np.testing.assert_array_almost_equal(t_hist, [1.5])
        np.testing.assert_array_almost_equal(p_hist, [0.5])

    def test_store_default_off(self):
        sensor = imu(isideal=True)
        rb = make_rigidbody(p=0.5)
        sensor.measure(rb)
        self.assertEqual(sensor.data().shape[0], 0)


class TestMagnetometer(unittest.TestCase):

    def test_zero_noise_is_exact(self):
        sensor = magnetometer(noise_std=0.0)
        x = make_state(psi=0.7)
        z = sensor.measure(x)
        self.assertAlmostEqual(z[0], 0.7)

    def test_noise_statistics(self):
        np.random.seed(0)
        sensor = magnetometer(noise_std=0.05)
        x = make_state(psi=0.0)
        samples = np.array([sensor.measure(x)[0] for _ in range(20000)])
        self.assertAlmostEqual(samples.mean(), 0.0, delta=0.005)
        self.assertAlmostEqual(samples.std(), 0.05, delta=0.005)

    def test_bias(self):
        sensor = magnetometer(noise_std=0.0, bias=0.1)
        x = make_state(psi=0.0)
        z = sensor.measure(x)
        self.assertAlmostEqual(z[0], 0.1)


if __name__ == "__main__":
    unittest.main()
