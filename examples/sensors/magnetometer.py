# type: ignore

import sys
import os

sys.path.append(".")
import c4dynamics as c4d

import numpy as np
from matplotlib import pyplot as plt


savedir = os.path.join(os.getcwd(), "docs", "source", "_examples", "navigation")


def true_heading():
    """Build a 12-state vector with a known yaw entry (index 8)."""
    x_true = np.zeros(12)
    x_true[8] = 0.5  # true heading [rad]
    return x_true


def ideal(x_true):
    """Demonstrate an ideal magnetometer measurement (errors model muted)."""
    mag_ideal = c4d.sensors.magnetometer(isideal=True)
    measurement = mag_ideal.measure(x_true)

    print("Ideal magnetometer:", measurement)
    # [0.5]

    return mag_ideal, measurement


def nonideal(x_true):
    """Demonstrate a magnetometer measurement with bias and white noise."""
    np.random.seed(42)
    mag_sensor = c4d.sensors.magnetometer(noise_std=0.05, bias=0.02)
    measurement = mag_sensor.measure(x_true)

    print(measurement)
    # [0.545]

    return mag_sensor, measurement


def bias(x_true):
    """Demonstrate a deterministic heading bias with noise disabled."""
    mag_bias = c4d.sensors.magnetometer(noise_std=0, bias=0.1)
    measurement = mag_bias.measure(x_true)

    print(measurement - x_true[8])
    # [0.1]

    return mag_bias, measurement


def noise(x_true):
    """Demonstrate sample-to-sample magnetometer noise."""
    np.random.seed(1)
    mag_noise = c4d.sensors.magnetometer(noise_std=0.05, bias=0)

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
    mag_sensor = c4d.sensors.magnetometer(noise_std=0, bias=0.1)
    x = np.zeros(12)
    x[8] = 0.5
    print(mag_sensor.measure(x))
    # [0.6]


if __name__ == "__main__":
    x_true = true_heading()

    # Run the examples individually while experimenting.
    ideal(x_true)
    nonideal(x_true)
    bias(x_true)
    noise(x_true)
    demo()

    measure()
