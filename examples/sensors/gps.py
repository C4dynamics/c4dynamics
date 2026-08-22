# type: ignore

import sys
import os

sys.path.append(".")
import c4dynamics as c4d

import numpy as np
from matplotlib import pyplot as plt


savedir = os.path.join(os.getcwd(), "docs", "source", "_examples", "navigation")


def true_trajectory(duration=20.0, dt=0.1):
    """Generate the trajectory used by the GPS documentation examples."""
    t = np.arange(0, duration, dt)
    x_true = np.zeros((len(t), 12))
    x_true[:, 0] = 5 * np.sin(0.3 * t)
    x_true[:, 1] = 4 * np.cos(0.25 * t)
    x_true[:, 2] = -2 + 0.5 * np.sin(0.6 * t)
    return t, x_true


def ideal(t, x_true):
    """Demonstrate an ideal GPS measurement."""
    gps_ideal = c4d.sensors.gps(noise_std=0, bias=[0, 0, 0])
    measurements = np.array([gps_ideal.measure(x) for x in x_true])

    print("Ideal GPS:", np.allclose(measurements, x_true[:, 0:3]))
    return gps_ideal, measurements


def nonideal(t, x_true):
    """Demonstrate GPS measurements with bias and white noise."""
    np.random.seed(42)

    gps_sensor = c4d.sensors.gps(
        noise_std=0.5,
        bias=[1.0, -0.5, 0.2],
    )
    measurements = np.array([gps_sensor.measure(x) for x in x_true])

    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    labels = ["x", "y", "z"]

    for i in range(3):
        ax[i].plot(t, x_true[:, i], lw=2, label="True")
        ax[i].plot(t, measurements[:, i], ".", ms=3, label="GPS measurement")
        ax[i].set_ylabel(f"{labels[i]} [m]")
        ax[i].grid(True)
        ax[i].legend()

    ax[-1].set_xlabel("Time [s]")
    fig.suptitle("GPS Position Measurements")
    fig.tight_layout()

    if os.path.isdir(savedir):
        plt.savefig(
            c4d.j(savedir, "gps.png"),
            bbox_inches="tight",
            pad_inches=0.2,
            dpi=600,
        )

    return gps_sensor, measurements, fig


def bias(x_true):
    """Demonstrate a deterministic position bias with noise disabled."""
    gps_bias = c4d.sensors.gps(noise_std=0, bias=[2, 0, 0])
    measurement = gps_bias.measure(x_true[0])
    # error = measurement - x_true[0, 0:3]

    print(measurement - x_true[0, 0:3])

    return gps_bias, measurement


def noise(x_true):
    """Demonstrate sample-to-sample GPS noise."""
    np.random.seed(1)
    gps_noise = c4d.sensors.gps(noise_std=0.5, bias=[0, 0, 0])

    print("Repeated measurements of the same state:")
    for _ in range(3):
        print(gps_noise.measure(x_true[0]))

    return gps_noise


def demo():
    """Run the built-in GPS demonstration."""
    fig = c4d.sensors.gps.demo(show=True)
    fig.savefig(
            c4d.j(savedir, "gps_demo.png"),
            bbox_inches="tight",
            pad_inches=0.2,
            dpi=600,
        )


def measure():
    import c4dynamics as c4d
    import numpy as np
    np.random.seed(42)
    gps_sensor = c4d.sensors.gps(noise_std=0, bias=[1, -2, 0.5])
    x = np.zeros(12)
    x[0:3] = [10, 20, 30]
    gps_sensor.measure(x)
    # array([11. , 18. , 30.5])

if __name__ == "__main__":
    t, x_true = true_trajectory()

    # Run the examples individually while experimenting.
    ideal(t, x_true)
    nonideal(t, x_true)
    bias(x_true)
    noise(x_true)
    demo()

    measure()

