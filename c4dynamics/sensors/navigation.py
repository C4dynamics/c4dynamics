"""

Navigation sensors — GPS, IMU, magnetometer
=============================================

`c4dynamics.sensors.navigation` provides generic navigation-sensor models
(GPS position, IMU gyroscope + accelerometer, magnetometer heading) that map
a *true* state vector to a noisy, biased *measurement* — the pattern used
throughout :mod:`c4dynamics.sensors` (compare :class:`radar
<c4dynamics.sensors.radar.radar>`, :class:`seeker
<c4dynamics.sensors.seeker.seeker>`).

Unlike :class:`radar`/:class:`seeker` (which operate on a
:class:`rigidbody <c4dynamics.states.lib.rigidbody.rigidbody>` origin and a
target), these sensors are written against a plain 12-state vector
``[x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]`` (the same ordering used
by :class:`rigidbody` and by :mod:`c4dynamics.controllers.quad_pid`), so they
drop directly into an EKF/UKF `predict`/`update` loop without an
intermediate adapter.

.. list-table::
  :header-rows: 0

  * - :class:`gps <c4dynamics.sensors.navigation.gps>`
    - Inertial position receiver
  * - :class:`imu <c4dynamics.sensors.navigation.imu>`
    - Gyroscope + accelerometer
  * - :class:`magnetometer <c4dynamics.sensors.navigation.magnetometer>`
    - Heading (yaw) sensor


See Also
========
.filters.ekf
.controllers.quad_pid


"""

import c4dynamics as c4d
import numpy as np
import sys


class gps:
    """
    GPS receiver.

    The :class:`gps` class models a GPS receiver that measures the inertial
    position of a vehicle in terms of ``[x, y, z]``.  The measurement is
    affected by a constant bias and sample-to-sample white Gaussian noise.


    Parameters
    ==========
    noise_std : float, optional
        Standard deviation of the position measurement noise, [m].
        Defaults to ``0.5``.
    bias : array_like, optional
        Constant position bias ``[bx, by, bz]``, [m].  Defaults to
        ``[0, 0, 0]``.


    See Also
    ========
    .ekf
    .seeker


    **Functionality**

    At each sample the GPS returns a position measurement based on the true
    inertial position of the vehicle.  If the true state is

    .. math::

        X = [x, y, z, v_x, v_y, v_z, \\varphi, \\theta, \\psi, p, q, r]^T

    the ideal position measurement is

    .. math::

        z_{ideal} = [x, y, z]^T.

    The simulated measurement is

    .. math::

        z = z_{ideal} + b + n

    where :math:`b` is the constant bias and :math:`n` is a zero-mean
    Gaussian random variable with standard deviation ``noise_std`` applied
    independently to each position coordinate.


    **Errors Model**

    The GPS measurement is subject to two error sources: bias and noise.

    - ``Bias``:
      represents a constant offset in the measured position.  It is set when
      the GPS object is constructed through the ``bias`` parameter and remains
      unchanged between measurements.  When ``bias`` is not provided, the
      bias is ``[0, 0, 0]``.
    - ``Noise``:
      represents random variations in the position measurement.  At every
      call to :meth:`measure`, an independent normally distributed random
      vector with mean zero and standard deviation ``noise_std`` is added to
      the position.

    The errors model can be disabled by setting ``noise_std = 0`` and
    ``bias = [0, 0, 0]``.  This produces an ideal position measurement.

    Unlike the :class:`seeker` model, the GPS implementation does not generate
    a random bias during construction and does not include a scale-factor
    error.  The supplied ``bias`` is deterministic for a given GPS instance.


    **Construction**

    A GPS instance is created by making a direct call to the constructor:

        >>> gps_sensor = c4d.sensors.gps()

    The measurement noise and constant bias can be specified when creating the
    sensor.


    Examples
    ========

    Import required packages:

    .. code::

        >>> import c4dynamics as c4d
        >>> from matplotlib import pyplot as plt
        >>> import numpy as np


    **True trajectory**

    For the examples below, generate a smooth 3D trajectory and store its
    position in a 12-state vector.

    .. code::

        >>> t = np.arange(0, 20, 0.1)
        >>> x_true = np.zeros((len(t), 12))
        >>> x_true[:, 0] = 5 * np.sin(0.3 * t)
        >>> x_true[:, 1] = 4 * np.cos(0.25 * t)
        >>> x_true[:, 2] = -2 + 0.5 * np.sin(0.6 * t)


    **Ideal GPS**

    An ideal GPS can be created by setting both the noise and the bias to
    zero:

    .. code::

        >>> gps_ideal = c4d.sensors.gps(noise_std=0, bias=[0, 0, 0])
        >>> measurements = np.array([gps_ideal.measure(x) for x in x_true])

    The measured position is then identical to the true position.

    .. code::

        >>> np.allclose(measurements, x_true[:, 0:3])
        True


    **Non-ideal GPS**

    A non-ideal GPS introduces a constant position bias and white measurement
    noise.  Set the random seed to make the example reproducible:

    .. code::

        >>> np.random.seed(42)
        >>> gps_sensor = c4d.sensors.gps(
        ...     noise_std=0.5,
        ...     bias=[1.0, -0.5, 0.2]
        ... )
        >>> measurements = np.array([gps_sensor.measure(x) for x in x_true])

    The result can be compared with the true position:

    .. code::

        >>> fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
        >>> labels = ['x', 'y', 'z']
        >>> for i in range(3):
        ...     ax[i].plot(t, x_true[:, i], lw=2, label='True')
        ...     ax[i].plot(t, measurements[:, i], '.', ms=3, label='GPS measurement')
        ...     ax[i].set_ylabel(f'{labels[i]} [m]')
        ...     ax[i].grid(True)
        ...     ax[i].legend()  # doctest: +IGNORE_OUTPUT
        >>> ax[-1].set_xlabel('Time [s]')  # doctest: +IGNORE_OUTPUT
        >>> fig.suptitle('GPS Position Measurements')  # doctest: +IGNORE_OUTPUT
        >>> fig.tight_layout()

    .. figure:: /_examples/navigation/gps.png


    **Bias**

    The bias is constant across all measurements.  For example, a GPS with a
    2m bias in the x direction can be created as follows:

    .. code::

        >>> gps_bias = c4d.sensors.gps(noise_std=0, bias=[2, 0, 0])
        >>> measurement = gps_bias.measure(x_true[0])
        >>> print(measurement - x_true[0, 0:3]) # doctest: +NUMPY_FORMAT
        [2.  0.  0.]

    The difference between the measurement and the true position is the
    specified bias.


    **Measurement noise**

    With zero bias, repeated measurements of the same state demonstrate the
    random noise generated at every call to :meth:`measure`:

    .. code::

        >>> np.random.seed(1)
        >>> gps_noise = c4d.sensors.gps(noise_std=0.5, bias=[0, 0, 0])
        >>> for _ in range(3): # doctest: +IGNORE_OUTPUT
        ...     print(gps_noise.measure(x_true[0]))
        [ 0.812  3.694  -2.264]
        [-0.536  4.433  -3.151]
        [ 0.872  3.619  -1.840]

    **Demo**

    The built-in :meth:`demo` method provides a compact demonstration of the
    GPS errors model and plots the true and measured positions:

    .. code::

        >>> fig = c4d.sensors.gps.demo(show=True)

    .. figure:: /_examples/navigation/gps_demo.png

    The same demonstration can be run without displaying the figure by using
    ``show=False``.
    """

    def __init__(self, noise_std=0.5, bias=None):
        self.noise_std = noise_std
        self.bias = np.zeros(3) if bias is None else np.asarray(bias, float)

    def measure(self, x_true):
        """
        Measure inertial position.

        The method extracts the first three elements of ``x_true`` and adds
        the GPS bias and a zero-mean Gaussian noise sample to each coordinate.

        Parameters
        ----------
        x_true : array_like
            True position vector
            ``[x, y, z]`` [m].

        Returns
        -------
        numpy.ndarray
            Measured inertial position ``[x, y, z]``, [m].


        **Errors Model**

        A single independent noise sample is generated for each position
        coordinate. The returned measurement is therefore:

        .. math::

            z = x_{true}[0:3] + bias + std \\cdot N(0, I)

        The bias is constant for the GPS instance, while the noise is
        regenerated at every call.

        Examples
        --------

        .. code::

            >>> import c4dynamics as c4d
            >>> import numpy as np
            >>> np.random.seed(42)
            >>> gps_sensor = c4d.sensors.gps(noise_std=0, bias=[1, -2, 0.5])
            >>> x = np.zeros(12)
            >>> x[0:3] = [10, 20, 30]
            >>> gps_sensor.measure(x)
            array([11.  18.  30.5])

        """
        return x_true[0:3] + self.bias + np.random.randn(3) * self.noise_std

    @staticmethod
    def demo(duration=20.0, dt=0.1, seed=1, show=True):
        """
        Demonstrate GPS position measurements.

        Simulates a smooth 3D trajectory and shows the noisy GPS position
        measurements in comparison with the true trajectory.

        Parameters
        ----------
        duration : float
            Simulation duration [s].
        dt : float
            Sampling interval [s].
        seed : int
            Random seed for reproducibility.
        show : bool
            If True, display the figure.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the true and measured x, y, and z positions.

        **Errors Model**

        The demonstration uses ``noise_std=0.5`` and zero bias.  The random
        seed controls the generated measurement noise so that the same
        demonstration can be reproduced.

        Examples
        --------

        Run the demonstration and display the result:

        .. code::

            >>> fig = gps.demo() # doctest: +ELLIPSIS

        To create the figure without displaying it:

        .. code::

            >>> fig = gps.demo(show=False) # doctest: +ELLIPSIS
        """
        import matplotlib.pyplot as plt

        np.random.seed(seed)

        sensor = gps(noise_std=0.5)

        t = np.arange(0, duration, dt)

        x_true = 5 * np.sin(0.3 * t)
        y_true = 4 * np.cos(0.25 * t)
        z_true = -2 + 0.5 * np.sin(0.6 * t)

        z_meas = np.zeros((len(t), 3))

        for k in range(len(t)):
            x = np.zeros(12)
            x[0:3] = [x_true[k], y_true[k], z_true[k]]
            z_meas[k] = sensor.measure(x)

        fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

        labels = ['x', 'y', 'z']
        truth = [x_true, y_true, z_true]

        for i in range(3):
            ax[i].plot(t, truth[i], 'k', lw=2, label='True')
            ax[i].plot(t, z_meas[:, i], '.', ms=3,
                       label='GPS measurement')
            ax[i].set_ylabel(f'{labels[i]} [m]')
            ax[i].grid(True)
            ax[i].legend()

        ax[-1].set_xlabel('Time [s]')
        fig.suptitle('GPS Position Measurements')
        fig.tight_layout()

        if show:
            plt.show()

        return fig


class imu:
    """Inertial measurement unit — gyroscope and accelerometer (200 Hz).

    The gyroscope measures body rates ``[p, q, r]``; the accelerometer measures
    body-frame specific force ``[ax, ay]``.  The accelerometer simulator adds
    the vehicle's translational (inertial) acceleration to the gravity
    projection, because that is what a real IMU senses; the EKF's measurement
    model :func:`accel_h` deliberately models only the gravity term, and the
    difference is absorbed by an inflated accelerometer ``R``.
    """

    def __init__(self, gyro_std=0.01, acc_std=0.05,
                 gyro_bias=None, acc_bias=None, g=9.81):
        self.gyro_std = gyro_std
        self.acc_std  = acc_std
        self.gyro_bias = np.zeros(3) if gyro_bias is None else np.asarray(gyro_bias, float)
        self.acc_bias  = np.zeros(2) if acc_bias  is None else np.asarray(acc_bias,  float)
        self.g = g

    def gyro(self, x_true):
        return x_true[9:12] + self.gyro_bias + np.random.randn(3) * self.gyro_std

    def accelerometer(self, x_true, x_true_prev=None, dt=0.005):
        phi, theta, psi = x_true[6], x_true[7], x_true[8]
        sp = np.sin(phi);  cp = np.cos(phi)
        st = np.sin(theta); ct = np.cos(theta)
        ss = np.sin(psi);  cs = np.cos(psi)

        ax = self.g * st            # gravity projection (modelled by accel_h)
        ay = -self.g * sp * ct

        if x_true_prev is not None:  # inertial term (NOT modelled by accel_h)
            # Rotated via BI = dcm321(phi,theta,psi) @ dcm321(phi=pi), matching
            # quad_pid.dynamics()'s actual body-from-inertial convention
            # (NOT the unflipped standard dcm321 — see derivation notes).
            dvx = (x_true[3] - x_true_prev[3]) / dt
            dvy = (x_true[4] - x_true_prev[4]) / dt
            dvz = (x_true[5] - x_true_prev[5]) / dt
            ax += (ct*cs)*dvx - (ct*ss)*dvy + st*dvz
            ay += (sp*st*cs - cp*ss)*dvx - (sp*st*ss + cp*cs)*dvy - (sp*ct)*dvz

        return np.array([ax, ay]) + self.acc_bias + np.random.randn(2) * self.acc_std

    @staticmethod
    def demo(duration=10.0, dt=0.01, seed=1, show=True):
        """
        Demonstrate IMU measurements.

        Simulates smooth body-rate motion and shows the gyroscope and
        accelerometer measurements in the presence of bias and white noise.

        Parameters
        ----------
        duration : float
            Simulation duration [s].
        dt : float
            Sampling interval [s].
        seed : int
            Random seed for reproducibility.
        show : bool
            If True, displays the figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        np.random.seed(seed)

        sensor = imu(
            gyro_std=0.02,
            acc_std=0.05,
            gyro_bias=[0.05, -0.03, 0.02],
            acc_bias=[0.10, -0.05],
        )

        t = np.arange(0, duration, dt)

        gyro_true = np.column_stack([
            0.5*np.sin(0.8*t),
            0.3*np.cos(0.6*t),
            0.2*np.sin(1.5*t),
        ])

        phi = np.deg2rad(10*np.sin(0.4*t))
        theta = np.deg2rad(8*np.cos(0.5*t))

        accel_true = np.column_stack([
            9.81*np.sin(theta),
            -9.81*np.sin(phi)*np.cos(theta),
        ])

        gyro_meas = np.zeros_like(gyro_true)
        accel_meas = np.zeros_like(accel_true)

        for k in range(len(t)):
            x = np.zeros(12)
            x[6] = phi[k]
            x[7] = theta[k]
            x[9:12] = gyro_true[k]

            gyro_meas[k] = sensor.gyro(x)
            accel_meas[k] = sensor.accelerometer(x)

        fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

        labels = ["p", "q", "r"]
        for i in range(3):
            ax[0].plot(t, gyro_true[:, i], lw=2, label=f"{labels[i]} true")
            ax[0].plot(t, gyro_meas[:, i], "--", lw=1,
                       label=f"{labels[i]} measured")

        ax[0].set_ylabel("Angular rate [rad/s]")
        ax[0].set_title("Gyroscope")
        ax[0].grid(True)
        ax[0].legend(ncol=3)

        ax[1].plot(t, accel_true[:, 0], lw=2, label="ax true")
        ax[1].plot(t, accel_meas[:, 0], "--", label="ax measured")
        ax[1].plot(t, accel_true[:, 1], lw=2, label="ay true")
        ax[1].plot(t, accel_meas[:, 1], "--", label="ay measured")

        ax[1].set_xlabel("Time [s]")
        ax[1].set_ylabel("Acceleration [m/s²]")
        ax[1].set_title("Accelerometer")
        ax[1].grid(True)
        ax[1].legend()

        plt.tight_layout()

        if show:
            plt.show()

        return fig


class magnetometer:
    """Magnetometer — measures heading (yaw) ``psi`` (50 Hz)."""

    def __init__(self, noise_std=0.05, bias=0.0):
        self.noise_std = noise_std
        self.bias = bias

    def measure(self, x_true):
        return x_true[8:9] + self.bias + np.random.randn(1) * self.noise_std

    @staticmethod
    def demo(duration=20.0, dt=0.02, seed=1, show=True):
        """
        Demonstrate magnetometer heading measurements.

        Simulates a changing vehicle heading and compares the true yaw angle
        with noisy magnetometer measurements.

        Parameters
        ----------
        duration : float
            Simulation duration [s].
        dt : float
            Sampling interval [s].
        seed : int
            Random seed for reproducibility.
        show : bool
            If True, display the figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        np.random.seed(seed)

        sensor = magnetometer(noise_std=0.05)

        t = np.arange(0, duration, dt)

        psi_true = 0.8 * np.sin(0.35 * t)

        psi_meas = np.zeros(len(t))

        for k in range(len(t)):
            x = np.zeros(12)
            x[8] = psi_true[k]
            psi_meas[k] = sensor.measure(x)[0]

        fig, ax = plt.subplots(figsize=(9, 4))

        ax.plot(t, np.rad2deg(psi_true),
                'k', lw=2, label='True heading')

        ax.plot(t, np.rad2deg(psi_meas),
                '.', ms=3, label='Magnetometer')

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Heading [deg]')
        ax.set_title('Magnetometer Measurements')
        ax.grid(True)
        ax.legend()

        fig.tight_layout()

        if show:
            plt.show()

        return fig



if __name__ == "__main__":

    from c4dynamics import rundoctests
    rundoctests(sys.modules[__name__])


