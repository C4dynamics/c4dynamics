# type: ignore

import sys
import os

sys.path.append(".")
import c4dynamics as c4d

import numpy as np
from matplotlib import pyplot as plt


savedir = os.path.join(os.getcwd(), "docs", "source", "_examples", "navigation")


def ideal():
    """Demonstrate an ideal imu measurement (errors model muted)."""
    imu_ideal = c4d.sensors.imu(isideal=True)
    rb = c4d.rigidbody(theta=-0.1, p=0.2, q=-0.1, r=0.05)
    ax, ay, az, p, q, r = imu_ideal.measure(rb)

    print(np.array([p, q, r]))
    # [0.2  -0.1  0.05]
    print(ax)
    # -0.979...
    print(az)
    # -9.760...

    return imu_ideal


def nonideal():
    """Demonstrate imu measurements with bias and white noise."""
    np.random.seed(100)
    imu_sensor = c4d.sensors.imu(gyro_std=0.01, acc_std=0.05)
    rb = c4d.rigidbody()

    print(imu_sensor.measure(rb))
    # (-0.012..., 0.049..., -9.784..., -0.017..., 0.003..., 0.011...)

    return imu_sensor


def store():
    """Demonstrate storing samples via measure(..., store = True)."""
    np.random.seed(200)
    imu_sensor = c4d.sensors.imu(isideal=True)
    rb = c4d.rigidbody()

    for t in np.arange(0, 0.02, 0.005):
        rb.p = 0.1 * t
        imu_sensor.measure(rb, t=t, store=True)

    print(imu_sensor.data("p"))
    # (array([0.   , 0.005, 0.01 , 0.015]), array([0.    , 0.0005, 0.001 , 0.0015]))

    return imu_sensor


def measure_ride():
    """`measure()`'s own docstring example: an imu riding a rigidbody
    through a short maneuver, storing the samples for later use."""
    np.random.seed(321)
    rb = c4d.rigidbody()
    imu_sensor = c4d.sensors.imu()
    dt = 0.005

    for t in np.arange(0, 1, dt):
        rb.inteqm(np.zeros(3), np.zeros(3), dt)
        imu_sensor.measure(rb, t=t, store=True)
        rb.store(t)

    print(imu_sensor.data("p")[1].shape)
    # (200,)

    return imu_sensor, rb


def demo():
    """Run the built-in imu demonstration."""
    fig = c4d.sensors.imu.demo(show=True)

    if os.path.isdir(savedir):
        fig.savefig(
            c4d.j(savedir, "imu_demo.png"),
            bbox_inches="tight",
            pad_inches=0.2,
            dpi=600,
        )

    return fig


if __name__ == "__main__":
    # Run the examples individually while experimenting.
    ideal()
    nonideal()
    store()
    measure_ride()
    demo()
