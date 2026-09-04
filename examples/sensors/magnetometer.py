# type: ignore

import sys
import os

sys.path.append(".")
import c4dynamics as c4d

import numpy as np
from matplotlib import pyplot as plt


savedir = os.path.join(os.getcwd(), "docs", "source", "_examples", "navigation")


def true_attitude():
    """Build a 12-state vector with a known yaw entry (index 8)."""
    x_true = np.zeros(12)
    x_true[8] = 0.5  # true heading [rad]
    return x_true


def ideal(x_true):
    """Demonstrate an ideal 3-axis magnetometer measurement (errors model muted)."""
    mag_ideal = c4d.sensors.magnetometer(isideal=True)

    print("Ideal magnetometer, level & north:", mag_ideal.measure(np.zeros(12)))
    # [0.5  0.  0.866]

    measurement = mag_ideal.measure(x_true)
    print("Ideal magnetometer, psi = 0.5 rad:", measurement)
    # [0.439  -0.24  0.866]

    return mag_ideal, measurement


def nonideal(x_true):
    """Demonstrate a magnetometer measurement with white noise."""
    np.random.seed(42)
    mag_sensor = c4d.sensors.magnetometer(noise_std=0.02)
    measurement = mag_sensor.measure(x_true)

    print(measurement)
    # [0.449  -0.242  0.879]

    return mag_sensor, measurement


def hard_iron(x_true):
    """Demonstrate a deterministic hard-iron offset with noise disabled."""
    mag_hi = c4d.sensors.magnetometer(noise_std=0, hard_iron=[0.1, 0, 0])
    mag_ref = c4d.sensors.magnetometer(isideal=True)

    print(mag_hi.measure(x_true) - mag_ref.measure(x_true))
    # [0.1  0.  0.]

    return mag_hi


def soft_iron():
    """Demonstrate a soft-iron gain on the body x axis."""
    mag_si = c4d.sensors.magnetometer(
        noise_std=0, soft_iron=np.diag([1.2, 1.0, 1.0]))

    print(mag_si.measure(np.zeros(12)))
    # [0.6  0.  0.866]

    return mag_si


def noise(x_true):
    """Demonstrate sample-to-sample per-axis magnetometer noise."""
    np.random.seed(1)
    mag_noise = c4d.sensors.magnetometer(noise_std=0.02)

    print("Repeated measurements of the same state:")
    for _ in range(3):
        print(mag_noise.measure(x_true))

    return mag_noise


def demo():
    """Run the built-in magnetometer demonstration."""
    fig = c4d.sensors.magnetometer.demo(show=True)

    if os.path.isdir(savedir):
        fig.savefig(
            c4d.j(savedir, "magnetometer_demo.png"),
            bbox_inches="tight",
            pad_inches=0.2,
            dpi=600,
        )

    return fig


def measure():
    """`measure()`'s own docstring example."""
    np.random.seed(42)
    mag_sensor = c4d.sensors.magnetometer(isideal=True)
    x = np.zeros(12)
    x[8] = 0.5
    print(mag_sensor.measure(x))
    # [0.439  -0.24  0.866]


if __name__ == "__main__":
    x_true = true_attitude()

    # Run the examples individually while experimenting.
    ideal(x_true)
    nonideal(x_true)
    hard_iron(x_true)
    soft_iron()
    noise(x_true)
    demo()

    measure()
