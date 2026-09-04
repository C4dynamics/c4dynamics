"""

Navigation sensors — GPS, IMU, magnetometer
=============================================

`c4dynamics.sensors.navigation` provides generic navigation-sensor models
(GPS position, IMU gyroscope + accelerometer, 3-axis magnetometer) that map
a *true* state vector to a noisy, biased *measurement* — the pattern used
throughout :mod:`c4dynamics.sensors` (compare :class:`radar
<c4dynamics.sensors.radar.radar>`, :class:`seeker
<c4dynamics.sensors.seeker.seeker>`).

Unlike :class:`radar`/:class:`seeker` (which operate on a
:class:`rigidbody <c4dynamics.states.lib.rigidbody.rigidbody>` origin and a
target), :class:`gps` and :class:`magnetometer` are written against a plain
12-state vector ``[x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]`` (the same
ordering used by :class:`rigidbody` and by
:mod:`c4dynamics.controllers.quad_pid`), so they drop directly into an
EKF/UKF `predict`/`update` loop without an intermediate adapter.
:class:`imu` measures rates and inertial acceleration, both of which are
derivatives of that state, so it takes the truth
:class:`rigidbody <c4dynamics.states.lib.rigidbody.rigidbody>` object itself
rather than a bare vector, and keeps its own previous-sample history
internally between calls to :meth:`imu.measure <imu.measure>`.

.. list-table::
  :header-rows: 0

  * - :class:`gps <c4dynamics.sensors.navigation.gps>`
    - Inertial position receiver
  * - :class:`imu <c4dynamics.sensors.navigation.imu>`
    - Gyroscope + accelerometer
  * - :class:`magnetometer <c4dynamics.sensors.navigation.magnetometer>`
    - 3-axis geomagnetic field sensor


See Also
========
.filters.ekf
.controllers.quad_pid


"""

import sys

import numpy as np

import c4dynamics as c4d


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
    isideal : bool, optional
        If ``True``, overrides ``noise_std`` and ``bias`` to zero, producing
        an ideal (noise-free, bias-free) GPS. Defaults to ``False``.


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

    def __init__(self, noise_std=0.5, bias=None, isideal=False):
        self.noise_std = noise_std
        self.bias = np.zeros(3) if bias is None else np.asarray(bias, float)
        if isideal:
            self.noise_std = 0.0
            self.bias = np.zeros(3)

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
            >>> gps_sensor.measure(x) # doctest: +NUMPY_FORMAT
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


class imu(c4d.state):
    """
    Inertial measurement unit — gyroscope and accelerometer.

    The `imu` class models a strapdown gyroscope and accelerometer pair.
    The gyroscope measures body rates ``[p, q, r]``.
    The accelerometer measures the full body-frame specific force
    ``[ax, ay, az]``.

    An `imu` instance is
    a :class:`state <c4dynamics.states.state.state>` whose state
    vector is the last measured sample, ``X = [ax, ay, az, p, q, r]``. This
    gives `imu` the :meth:`store() <c4dynamics.states.state.state.store>`
    and :meth:`data() <c4dynamics.states.state.state.data>` methods for free
    (:meth:`measure` uses `store` internally, see below).


    Parameters
    ==========
    gyro_std : float, optional
        Standard deviation of the gyroscope noise, [rad/s]. Defaults ``0.01``.
    acc_std : float, optional
        Standard deviation of the accelerometer noise, [m/s²]. Defaults ``0.05``.
    gyro_bias : array_like, optional
        Constant gyroscope bias ``[bp, bq, br]``, [rad/s]. Defaults ``[0, 0, 0]``.
    acc_bias : array_like, optional
        Constant accelerometer bias ``[bax, bay, baz]``, [m/s²]. Defaults ``[0, 0, 0]``.
    g : float, optional
        Gravitational acceleration, [m/s²]. Defaults ``9.81``.
    isideal : bool, optional
        A flag indicating whether the errors model is off. Defaults `False`.
    dt : float, optional
        Fallback timestep, [s], used for the accelerometer's finite-difference
        inertial term when consecutive calls to :meth:`measure` don't carry
        increasing `t` values. Defaults ``0.005`` (:math:`200Hz`).


    See Also
    ========
    .ekf
    .gps
    .magnetometer


    **Functionality**

    At each sample the imu returns rate and acceleration measurements based
    on the true state of a rigid body. Let the true state be:

    .. math::

        X = [x, y, z, v_x, v_y, v_z, \\varphi, \\theta, \\psi, p, q, r]^T

    Where:

    - :math:`x, y, z` are the inertial position coordinates
    - :math:`v_x, v_y, v_z` are the inertial velocity coordinates
    - :math:`\\varphi, \\theta, \\psi` are the Euler angles (roll, pitch, yaw)
    - :math:`p, q, r` are the body rates about the roll, pitch, yaw axes

    The gyroscope measurement is the last three states directly:

    .. math::

        [p, q, r]_{meas} = [p, q, r] + bias_{gyro} + noise_{gyro}

    The accelerometer measures the full body-frame specific force, i.e.
    gravity reaction plus the vehicle's own coordinate acceleration:

    .. math::

        [a_x, a_y, a_z]_{ideal} = [BI] \\cdot \\big(\\dot{v} + [0,\\ 0,\\ g]^T\\big)

    where :math:`[BI]` is the body-from-inertial DCM and :math:`\\dot{v}` is
    the inertial-velocity derivative. Since `measure` is given only the
    current true state, :math:`\\dot{v}` is approximated by a finite
    difference against the *previous* call's true state:

    .. math::

        \\dot{v} \\approx {v(t) - v(t_{prev}) \\over t - t_{prev}}

    which is why :meth:`measure` keeps the previous sample as an internal
    attribute (`t_prev`, `x_prev`) rather than asking the caller for it: the
    first call to `measure` (no previous sample yet) therefore reports the
    gravity term alone.



    **Errors Model**

    - ``Bias``: a constant offset, set at construction through `gyro_bias` /
      `acc_bias` and unchanged between measurements.
    - ``Noise``: a zero-mean Gaussian sample, drawn independently at every
      call to :meth:`measure`, with standard deviation `gyro_std` / `acc_std`.

    The errors model can be disabled by applying ``isideal = True`` at the
    imu construction stage: the gyroscope and accelerometer standard
    deviations and biases are then muted (forced to zero), regardless of
    the ``gyro_std`` / ``acc_std`` / ``gyro_bias`` / ``acc_bias`` arguments,
    and :meth:`measure` returns the noise-free, bias-free truth.


    **Construction**

    An imu instance is created by making a direct call to the constructor:

    .. code::

        >>> imu_sensor = c4d.sensors.imu()

    Initialization does not require any mandatory arguments.


    Examples
    ========

    Import required packages:

    .. code::

        >>> import c4dynamics as c4d
        >>> import numpy as np

    **Ideal imu**

    An ideal imu can be created by muting the errors model:

    .. code::

        >>> imu_ideal = c4d.sensors.imu(isideal = True)
        >>> rb = c4d.rigidbody(theta = -0.1, p = 0.2, q = -0.1, r = 0.05)
        >>> ax, ay, az, p, q, r = imu_ideal.measure(rb)
        >>> np.array([p, q, r]) # doctest: +NUMPY_FORMAT
        [0.2  -0.1  0.05]
        >>> ax # doctest: +ELLIPSIS
        -0.979...
        >>> az # doctest: +ELLIPSIS
        -9.760...

    **Non-ideal imu**

    .. code::

        >>> np.random.seed(100)
        >>> imu_sensor = c4d.sensors.imu(gyro_std = 0.01, acc_std = 0.05)
        >>> rb = c4d.rigidbody()
        >>> imu_sensor.measure(rb) # doctest: +ELLIPSIS
        (-0.012..., 0.049..., -9.784..., -0.017..., 0.003..., 0.011...)

    **Store**

    Passing ``store = True`` stores the sampled `[ax, ay, az, p, q, r]` along
    with the given timestamp; the histories are then available through
    :meth:`data() <c4dynamics.states.state.state.data>`:

    .. code::

        >>> np.random.seed(200)
        >>> imu_sensor = c4d.sensors.imu(isideal = True)
        >>> rb = c4d.rigidbody()
        >>> for t in np.arange(0, 0.02, 0.005):
        ...     rb.p = 0.1 * t
        ...     imu_sensor.measure(rb, t = t, store = True) # doctest: +IGNORE_OUTPUT
        >>> imu_sensor.data('p')
        (array([0.   , 0.005, 0.01 , 0.015]), array([0.    , 0.0005, 0.001 , 0.0015]))


    **Demo**

    The built-in :meth:`demo` method provides a compact demonstration of the
    imu errors model and plots the true and measured rates and accelerations:

    .. code::

        >>> fig = c4d.sensors.imu.demo(show = True)

    .. figure:: /_examples/navigation/imu_demo.png

    The same demonstration can be run without displaying the figure by using
    ``show = False``.
    """

    def __init__(self, gyro_std=0.01, acc_std=0.05, gyro_bias=None,
                 acc_bias=None, g=9.81, isideal=False, dt=0.005):
        super().__init__(ax=0.0, ay=0.0, az=0.0, p=0.0, q=0.0, r=0.0)

        self.gyro_std = gyro_std
        self.acc_std  = acc_std
        self.gyro_bias = np.zeros(3) if gyro_bias is None else np.asarray(gyro_bias, float)
        self.acc_bias  = np.zeros(3) if acc_bias  is None else np.asarray(acc_bias,  float)
        self.g  = g
        self.dt = dt

        if isideal:
            self.gyro_std  = 0.0
            self.acc_std   = 0.0
            self.gyro_bias = np.zeros(3)
            self.acc_bias  = np.zeros(3)

        self._x_prev = None   # true state at the previous call to measure()
        self._t_prev = None   # its timestamp

    def measure(self, rb: "c4d.rigidbody", t: float = -1, store: bool = False):  # type: ignore
        """
        Measures body rates and specific acceleration of a rigid body.

        If `store = True`, the method stores the measured sample
        `[ax, ay, az, p, q, r]` along with a timestamp (`t = -1` by default,
        if not provided otherwise).


        Parameters
        ----------
        rb : rigidbody
            A :class:`rigidbody <c4dynamics.states.lib.rigidbody.rigidbody>`
            object (or any 12-variable state object using the same
            ``[x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]`` ordering)
            providing the true reference state for the sample.
        t : float, optional
            Timestamp [seconds]. Defaults -1.
        store : bool, optional
            A flag indicating whether to store the measured values. Defaults `False`.

        Returns
        -------
        out : tuple
            Accelerations and rates, `(ax, ay, az, p, q, r)`,
            [m/s², m/s², m/s², rad/s, rad/s, rad/s].


        Note
        ----
        The accelerometer's inertial term requires a previous sample. As
        `measure` keeps that sample internally (there's no `x_true_prev`
        argument to pass), the very first call after construction reports
        the gravity-projection term alone.


        Example
        -------

        `measure` in a program simulating an imu riding a rigid body
        through a short maneuver, storing the samples for later use:

        .. code::

            >>> import c4dynamics as c4d
            >>> import numpy as np
            >>> np.random.seed(321)
            >>> rb = c4d.rigidbody()
            >>> imu_sensor = c4d.sensors.imu()
            >>> dt = 0.005
            >>> for t in np.arange(0, 1, dt):
            ...     rb.inteqm(np.zeros(3), np.zeros(3), dt) # doctest: +IGNORE_OUTPUT
            ...     imu_sensor.measure(rb, t = t, store = True) # doctest: +IGNORE_OUTPUT
            ...     rb.store(t)
            >>> imu_sensor.data('p')[1].shape
            (200,)
        """

        x_true = np.asarray(rb.X).ravel()

        # ---- gyroscope: body rates -------------------------------------
        p, q, r = x_true[9:12] + self.gyro_bias + np.random.randn(3) * self.gyro_std

        # ---- accelerometer: body-frame specific force -------------------
        phi, theta, psi = x_true[6], x_true[7], x_true[8]
        sp, cp = np.sin(phi), np.cos(phi)
        st, ct = np.sin(theta), np.cos(theta)

        ax = self.g * st            # gravity projection
        ay = -self.g * sp * ct
        az = -self.g * cp * ct

        if self._x_prev is not None:  # inertial term, needs a previous sample
            dtc = self.dt
            if t != -1 and self._t_prev is not None and self._t_prev != -1 and t > self._t_prev:
                dtc = t - self._t_prev

            ss, cs = np.sin(psi), np.cos(psi)
            dvx = (x_true[3] - self._x_prev[3]) / dtc
            dvy = (x_true[4] - self._x_prev[4]) / dtc
            dvz = (x_true[5] - self._x_prev[5]) / dtc
            ax += (ct*cs)*dvx - (ct*ss)*dvy + st*dvz
            ay += (sp*st*cs - cp*ss)*dvx - (sp*st*ss + cp*cs)*dvy - (sp*ct)*dvz
            az += (sp*ss + st*cp*cs)*dvx + (sp*cs - ss*st*cp)*dvy - (cp*ct)*dvz

        ax, ay, az = np.array([ax, ay, az]) + self.acc_bias + np.random.randn(3) * self.acc_std

        self.ax, self.ay, self.az, self.p, self.q, self.r = ax, ay, az, p, q, r

        self._x_prev, self._t_prev = x_true.copy(), t

        if store:
            self.store(t)

        return self.ax, self.ay, self.az, self.p, self.q, self.r

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
            acc_bias=[0.10, -0.05, 0.15],
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
            -9.81*np.cos(phi)*np.cos(theta),
        ])

        gyro_meas = np.zeros_like(gyro_true)
        accel_meas = np.zeros_like(accel_true)

        rb = c4d.rigidbody()

        for k in range(len(t)):
            rb.phi = phi[k]
            rb.theta = theta[k]
            rb.p, rb.q, rb.r = gyro_true[k]

            # rb.vx/vy/vz stay 0 throughout, so the accelerometer's
            # finite-difference inertial term evaluates to 0 regardless of
            # dt, leaving the gravity projection alone -- matching accel_true.
            ax_k, ay_k, az_k, p_k, q_k, r_k = sensor.measure(rb)
            gyro_meas[k] = [p_k, q_k, r_k]
            accel_meas[k] = [ax_k, ay_k, az_k]

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

        acc_labels = ["ax", "ay", "az"]
        for i in range(3):
            ax[1].plot(t, accel_true[:, i], lw=2, label=f"{acc_labels[i]} true")
            ax[1].plot(t, accel_meas[:, i], "--", label=f"{acc_labels[i]} measured")

        ax[1].set_xlabel("Time [s]")
        ax[1].set_ylabel("Acceleration [m/s²]")
        ax[1].set_title("Accelerometer")
        ax[1].grid(True)
        ax[1].legend(ncol=3)

        plt.tight_layout()

        if show:
            plt.show()

        return fig


class magnetometer:
    """
    Magnetometer — 3-axis geomagnetic field sensor.

    The :class:`magnetometer` class models a strapdown 3-axis magnetometer.
    It measures the local geomagnetic field vector expressed in the vehicle
    **body frame**: a fixed reference field in the navigation frame, rotated
    into the body frame by the true attitude, then corrupted by soft-iron and
    hard-iron distortion and sample-to-sample white Gaussian noise.


    Parameters
    ==========
    noise_std : float or array_like, optional
        Standard deviation of the field measurement noise, per body axis, in
        the same units as ``field_intensity`` (dimensionless when the field
        is left normalized).  A scalar is broadcast to all three axes.
        Defaults to ``0.02``.
    hard_iron : array_like, optional
        Constant additive bias ``[bx, by, bz]`` (hard-iron offset), in the
        same units as ``field_intensity``.  Defaults to ``[0, 0, 0]``.
    soft_iron : array_like, optional
        ``3 x 3`` soft-iron distortion matrix applied to the body-frame field
        before the hard-iron offset and the noise.  Defaults to the identity.
    field_intensity : float, optional
        Total intensity :math:`F` of the reference geomagnetic field.
        Defaults to ``1.0`` (the field is returned normalized).  Set a real
        value (e.g. ``50e-6`` T or ``50`` for µT) to work in physical units —
        ``noise_std`` and ``hard_iron`` then carry the same unit.
    inclination : float, optional
        Inclination (dip) angle :math:`I` of the reference field, [rad],
        positive pointing down.  Defaults to ``np.pi / 3`` (:math:`60^\\circ`,
        a typical mid-latitude value).
    declination : float, optional
        Declination angle :math:`D` of the reference field, [rad], positive
        east of north.  Defaults to ``0``.
    isideal : bool, optional
        If ``True``, overrides ``noise_std`` / ``hard_iron`` / ``soft_iron``
        to produce an ideal (noise-free, distortion-free) magnetometer.  The
        reference field itself is unaffected.  Defaults to ``False``.


    See Also
    ========
    .ekf
    .gps
    .imu


    **Functionality**

    The reference field is built from its total intensity, inclination and
    declination and held fixed in the navigation frame associated with the
    state's 3-2-1 Euler angles (``x`` forward/north, ``y`` right/east, ``z``
    down):

    .. math::

        m_{ref} = F \\cdot
            [\\cos I \\cos D,\\ \\cos I \\sin D,\\ \\sin I]^T

    At each sample, given the 12-state vector

    .. math::

        X = [x, y, z, v_x, v_y, v_z, \\varphi, \\theta, \\psi, p, q, r]^T

    the reference field is rotated into the body frame by the true attitude
    and distorted:

    .. math::

        m_{body} = [BI](\\varphi, \\theta, \\psi) \\cdot m_{ref}

        z = S_i \\cdot m_{body} + b_i + n

    where :math:`[BI]` is the body-from-inertial 3-2-1 DCM
    (:func:`dcm321 <c4dynamics.rotmat.rotmat.dcm321>`), :math:`S_i` is the
    soft-iron matrix, :math:`b_i` is the hard-iron offset, and :math:`n` is a
    zero-mean Gaussian vector with per-axis standard deviation ``noise_std``.

    ``measure`` returns the full 3-axis body-frame field vector.  A consumer
    that needs a heading derives it from the horizontal components after
    de-rotating roll and pitch (tilt compensation); a filter that consumes
    the vector directly (see :class:`ekf`) needs no yaw-wrapping, because the
    measurement is linear in the rotated field rather than in the angle
    itself.  In a typical setup the magnetometer is sampled at a lower rate
    than the IMU, e.g. :math:`50\\,Hz`.


    **Errors Model**

    The magnetometer measurement is subject to three error sources:

    - ``Hard iron``:
      a constant additive offset ``[bx, by, bz]`` (magnetized material fixed
      to the body).  Set through the ``hard_iron`` parameter and unchanged
      between measurements.  Defaults to ``[0, 0, 0]``.
    - ``Soft iron``:
      a constant ``3 x 3`` linear distortion :math:`S_i` (nearby ferrous
      material that reshapes the field).  Set through the ``soft_iron``
      parameter.  Defaults to the identity.
    - ``Noise``:
      a zero-mean Gaussian sample, drawn independently per axis at every call
      to :meth:`measure`, with standard deviation ``noise_std``.

    The errors model can be disabled by passing ``isideal = True`` at
    construction, which forces ``noise_std = 0``, ``hard_iron = [0, 0, 0]``
    and ``soft_iron = I`` regardless of the arguments.  Unlike the
    :class:`seeker` model, the magnetometer does not generate a random bias
    during construction; the supplied distortion is deterministic for a given
    instance.


    **Construction**

    A magnetometer instance is created by making a direct call to the
    constructor:

        >>> mag_sensor = c4d.sensors.magnetometer()

    The noise, the hard-iron / soft-iron distortion and the reference-field
    geometry can all be specified when creating the sensor.


    Examples
    ========

    Import required packages:

    .. code::

        >>> import c4dynamics as c4d
        >>> import numpy as np


    **True attitude**

    For the examples below, build a 12-state vector and set its attitude
    entries (indices 6, 7, 8 = roll, pitch, yaw):

    .. code::

        >>> x_true = np.zeros(12)
        >>> x_true[8] = 0.5   # true heading [rad]


    **Ideal magnetometer**

    An ideal magnetometer can be created by muting the errors model.  Level
    and heading north, it reads the reference field directly — horizontal
    component forward, vertical component (the dip) down:

    .. code::

        >>> mag_ideal = c4d.sensors.magnetometer(isideal=True)
        >>> mag_ideal.measure(np.zeros(12))   # doctest: +NUMPY_FORMAT
        [0.5  0.  0.866]

    Rotating to a heading of ``0.5`` rad swings the horizontal field into the
    body ``y`` axis while its magnitude and the vertical component are
    preserved:

    .. code::

        >>> mag_ideal.measure(x_true)   # doctest: +NUMPY_FORMAT
        [0.439  -0.24  0.866]


    **Non-ideal magnetometer**

    A non-ideal magnetometer adds white measurement noise (and, optionally,
    hard-iron / soft-iron distortion).  Set the random seed to make the
    example reproducible:

    .. code::

        >>> np.random.seed(42)
        >>> mag_sensor = c4d.sensors.magnetometer(noise_std=0.02)
        >>> mag_sensor.measure(x_true)   # doctest: +NUMPY_FORMAT
        [0.449  -0.242  0.879]


    **Hard iron**

    The hard-iron offset is constant across all measurements.  A magnetometer
    with a ``[0.1, 0, 0]`` offset can be created as follows:

    .. code::

        >>> mag_hi = c4d.sensors.magnetometer(noise_std=0, hard_iron=[0.1, 0, 0])
        >>> mag_ref = c4d.sensors.magnetometer(isideal=True)
        >>> mag_hi.measure(x_true) - mag_ref.measure(x_true)   # doctest: +NUMPY_FORMAT
        [0.1  0.  0.]

    The difference between the measurement and the ideal field is the
    specified offset.


    **Soft iron**

    The soft-iron matrix scales / mixes the body-frame field.  A diagonal
    ``soft_iron`` with a ``1.2`` gain on the body ``x`` axis:

    .. code::

        >>> mag_si = c4d.sensors.magnetometer(
        ...     noise_std=0, soft_iron=np.diag([1.2, 1.0, 1.0]))
        >>> mag_si.measure(np.zeros(12))   # doctest: +NUMPY_FORMAT
        [0.6  0.  0.866]


    **Measurement noise**

    With no distortion, repeated measurements of the same state demonstrate
    the random per-axis noise generated at every call to :meth:`measure`:

    .. code::

        >>> np.random.seed(1)
        >>> mag_noise = c4d.sensors.magnetometer(noise_std=0.02)
        >>> for _ in range(3): # doctest: +IGNORE_OUTPUT
        ...     print(mag_noise.measure(x_true))
        [0.471 -0.252  0.855]
        [0.417 -0.222  0.820]
        [0.474 -0.255  0.872]


    **Demo**

    The built-in :meth:`demo` method provides a compact demonstration of the
    magnetometer errors model and plots the true and measured body-frame
    field components through an attitude sweep:

    .. code::

        >>> fig = c4d.sensors.magnetometer.demo(show=True)

    .. figure:: /_examples/navigation/magnetometer_demo.png

    The same demonstration can be run without displaying the figure by using
    ``show=False``.
    """

    def __init__(self, noise_std=0.02, hard_iron=None, soft_iron=None,
                 field_intensity=1.0, inclination=np.pi / 3, declination=0.0,
                 isideal=False):
        self.noise_std = np.broadcast_to(
            np.asarray(noise_std, float), (3,)).astype(float)
        self.hard_iron = (np.zeros(3) if hard_iron is None
                          else np.asarray(hard_iron, float))
        self.soft_iron = (np.eye(3) if soft_iron is None
                          else np.asarray(soft_iron, float))
        self.field_intensity = field_intensity
        self.inclination = inclination
        self.declination = declination

        cI, sI = np.cos(inclination), np.sin(inclination)
        cD, sD = np.cos(declination), np.sin(declination)
        self.mref = field_intensity * np.array([cI * cD, cI * sD, sI])

        if isideal:
            self.noise_std = np.zeros(3)
            self.hard_iron = np.zeros(3)
            self.soft_iron = np.eye(3)

    def measure(self, x_true):
        """
        Measure the body-frame geomagnetic field.

        The method rotates the reference field ``mref`` into the body frame
        using the attitude entries of ``x_true`` (indices 6, 7, 8), then
        applies the soft-iron matrix, the hard-iron offset and a zero-mean
        Gaussian noise sample per axis.

        Parameters
        ----------
        x_true : array_like
            True state vector; only the attitude entries
            ``x_true[6:9]`` (roll, pitch, yaw) are used, [rad].

        Returns
        -------
        numpy.ndarray
            Measured body-frame field ``[mx, my, mz]``, in the units of
            ``field_intensity``.


        **Errors Model**

        .. math::

            z = S_i \\cdot [BI](\\varphi, \\theta, \\psi) \\cdot m_{ref}
                + b_i + std \\cdot N(0, I_3)

        The soft-iron matrix :math:`S_i` and hard-iron offset :math:`b_i` are
        constant for the magnetometer instance, while the noise is
        regenerated at every call.

        Examples
        --------

        .. code::

            >>> import c4dynamics as c4d
            >>> import numpy as np
            >>> np.random.seed(42)
            >>> mag_sensor = c4d.sensors.magnetometer(isideal=True)
            >>> x = np.zeros(12)
            >>> x[8] = 0.5
            >>> mag_sensor.measure(x)   # doctest: +NUMPY_FORMAT
            [0.439  -0.24  0.866]

        """
        x_true = np.asarray(x_true, float).ravel()
        phi, theta, psi = x_true[6], x_true[7], x_true[8]
        m_body = c4d.rotmat.dcm321(phi, theta, psi) @ self.mref
        return (self.soft_iron @ m_body
                + self.hard_iron
                + np.random.randn(3) * self.noise_std)

    @staticmethod
    def demo(duration=20.0, dt=0.02, seed=1, show=True):
        """
        Demonstrate 3-axis magnetometer measurements.

        Drives the sensor through a smooth roll / pitch / yaw sweep and
        compares the true body-frame geomagnetic field components with the
        noisy magnetometer measurements.

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
            Figure containing the true and measured mx, my, mz components.
        """
        import matplotlib.pyplot as plt

        np.random.seed(seed)

        sensor = magnetometer(noise_std=0.02)
        ideal = magnetometer(isideal=True)

        t = np.arange(0, duration, dt)

        phi_true = 0.15 * np.sin(0.50 * t)
        theta_true = 0.10 * np.cos(0.40 * t)
        psi_true = 0.80 * np.sin(0.35 * t)

        m_true = np.zeros((len(t), 3))
        m_meas = np.zeros((len(t), 3))

        for k in range(len(t)):
            x = np.zeros(12)
            x[6], x[7], x[8] = phi_true[k], theta_true[k], psi_true[k]
            m_true[k] = ideal.measure(x)
            m_meas[k] = sensor.measure(x)

        fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

        labels = ['m_x', 'm_y', 'm_z']
        for i in range(3):
            ax[i].plot(t, m_true[:, i], 'k', lw=2, label='True')
            ax[i].plot(t, m_meas[:, i], '.', ms=3, label='Magnetometer')
            ax[i].set_ylabel(f'{labels[i]} [norm.]')
            ax[i].grid(True)
            ax[i].legend()

        ax[-1].set_xlabel('Time [s]')
        fig.suptitle('Magnetometer Measurements (body-frame field)')
        fig.tight_layout()

        if show:
            plt.show()

        return fig



if __name__ == "__main__":

    from c4dynamics import rundoctests
    rundoctests(sys.modules[__name__])


