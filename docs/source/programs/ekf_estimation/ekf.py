"""
ekf.py — Extended Kalman Filter state estimation for the 12-state quadcopter
============================================================================

This module is the estimation companion to ``quad_pid_utils.py``.  Where the
cascade-PID module provides the *plant* and the *controller*, this module
provides the *sensors* and the *estimator* that let the same quadcopter fly its
figure-8 using an **estimated** state reconstructed from noisy GPS and IMU
measurements rather than from perfect truth.

It is organised exactly like ``quad_pid_utils.py`` — a single reusable module
that the notebook (``ekf.ipynb``) drives through a configuration dictionary and
a single call to :func:`run_fig8_ekf`.

Design
------
* The **truth** vehicle is a :class:`c4dynamics.rigidbody`, propagated with the
  shared :func:`quad_pid_utils.dynamics` (the *same* model the EKF uses as its
  process model).
* The **estimate** is an :class:`ekf <c4dynamics.filters.ekf>` whose 12 state
  variables carry the *same names and order* as the truth.  Because the
  framework EKF is itself a ``state`` object, the estimate stores itself
  (``store``/``data``), exposes named components (``est.phi`` …), and is read
  directly by the cascade-PID controllers — no proxy object is needed.
* The **sensors** follow the c4dynamics sensor pattern: each exposes a
  ``measure`` method that maps the *true* state to a noisy, biased measurement.
  Truth reaches the estimator only through these measurements.

The estimation algorithm (analytical Jacobian, 2nd-order discretization,
per-sensor innovation gating, adaptive GPS measurement noise, adaptive velocity
process noise, nonlinear accelerometer model) is expressed here through the framework's
``predict``/``update`` API.

State vector (shared with the truth rigidbody and the cascade-PID model)::

    [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
     0  1  2   3   4   5   6     7     8   9  10 11

Frames: body = forward-right-down (FRD), inertial = ENU, 3-2-1 Euler angles —
identical to ``quad_pid_utils.py`` (X motor configuration, dcm321-based
rotations). jacobian_F's translational blocks are computed numerically
against quad_pid_utils.dynamics() directly, so this
consistency is enforced by construction rather than by hand-derivation.

Contents
--------
    jacobian_F            process Jacobian df/dx (12x12): analytic kinematic/
                           rotational blocks + numeric translational block
    H_GPS, H_GYRO, H_MAG  constant measurement matrices
    accel_h, accel_H      nonlinear accelerometer measurement model + Jacobian
    gps, imu, magnetometer simulated sensors (truth -> measurement)
    ekf12                 the 12-state EKF (subclass of c4d.filters.ekf)
    default_ekf_config    reference EKF noise / initialization block
    run_fig8_ekf          single closed-loop estimation-control simulation
    compute_metrics       RMSE (true vs estimated) + filter-consistency (NEES)
    plot_estimation       true-vs-estimated visualisation with +-2 sigma bands
"""

import numpy as np
from collections import deque
from scipy.integrate import solve_ivp

import c4dynamics as c4d

from quad_pid_utils import (dynamics, position_reference, velocity_reference,
                            InitializeControllers, dcm321)

# State-variable names, in the canonical c4dynamics rigidbody order.
STATE_NAMES = ['x', 'y', 'z', 'vx', 'vy', 'vz',
               'phi', 'theta', 'psi', 'p', 'q', 'r']


# ============================================================================
#  PROCESS MODEL JACOBIAN   F = df/dx   (12 x 12)
# ============================================================================

def _numeric_translational_jacobian(x, quad, rotor_speeds, eps=1e-6):
    """
    Numerically differentiate the translational rows of
    :func:`quad_pid_utils.dynamics` (indices 3,4,5 = dvx,dvy,dvz) with respect
    to [vx,vy,vz,phi,theta,psi] (state indices 3..8), via central differences.

    quad_pid_utils2's translational model rotates velocity into the body
    frame, applies drag there, then rotates the resulting force back to
    inertial (``BI``/``dcm321``-based) — this couples drag to attitude and
    is no longer a simple closed-form block, so it is differentiated
    numerically here instead of hand-derived. This keeps the Jacobian
    correct automatically if quad_pid_utils.dynamics() changes again.

    Returns
    -------
    block : np.ndarray (3, 6) — d[dvx,dvy,dvz] / d[vx,vy,vz,phi,theta,psi]
    """
    idx = [3, 4, 5, 6, 7, 8]
    block = np.zeros((3, 6))
    for j, si in enumerate(idx):
        xp = x.copy(); xp[si] += eps
        xm = x.copy(); xm[si] -= eps
        fp = dynamics(0.0, xp, quad, rotor_speeds)[3:6]
        fm = dynamics(0.0, xm, quad, rotor_speeds)[3:6]
        block[:, j] = (fp - fm) / (2.0 * eps)
    return block


def jacobian_F(x, Omega, quad, rotor_speeds, params):
    """
    Jacobian of the rigid-body dynamics ``f(x, u)`` evaluated at the current
    state estimate.

    The process model ``f`` is :func:`quad_pid_utils.dynamics`. This is a
    hybrid Jacobian:
      - Blocks 1, 4, 5, 6 (position kinematics, Euler-angle kinematics,
        Euler's rotational equations + gyroscopic coupling) are closed-form.
        These depend only on body-frame angular rates and inertia.
      - Blocks 2, 3 (translational drag and thrust-to-attitude coupling) are
        computed numerically via :func:`_numeric_translational_jacobian`.

    Parameters
    ----------
    x : np.ndarray (12,)
        Current state estimate ``[x,y,z, vx,vy,vz, phi,theta,psi, p,q,r]``.
    Omega : float
        Net rotor speed ``w1 + w2 - w3 - w4`` [rad/s] for gyroscopic coupling.
    quad : object
        Quad-like object exposing the physical parameters as attributes
        (m, g, l, kT, kQ, Ixx, Iyy, Izz, Ar, IR, Ax, Ay, Az) — passed straight
        through to quad_pid_utils.dynamics() for the numeric block.
    rotor_speeds : np.ndarray (4,)
        Current actual rotor speeds [w1,w2,w3,w4], needed to re-evaluate
        dynamics() for the numeric block.
    params : dict
        Quadcopter physical parameters (mass, inertia, drag, rotor inertia).

    Returns
    -------
    F : np.ndarray (12, 12)
        Continuous-time Jacobian evaluated at ``x``.
    """
    _, _, _, vx, vy, vz, phi, theta, psi, p, q, r = x

    Ixx = params['Ixx']; Iyy = params['Iyy']; Izz = params['Izz']
    Ar  = params['Ar'];  IR  = params['IR']

    sp = np.sin(phi);   cp = np.cos(phi)
    st = np.sin(theta); ct = np.cos(theta)

    # Guard the Euler-angle singularity at theta = +-90 deg: the kinematic
    # block carries 1/cos(theta) and tan(theta) terms.  For nominal flight
    # (|theta| << 90 deg) this clamp is never active.
    if abs(ct) < 1e-3:
        ct = np.sign(ct) * 1e-3
    tt = st / ct

    F = np.zeros((12, 12))

    # Block 1 — d[pos]/d[vel] : identity
    F[0, 3] = F[1, 4] = F[2, 5] = 1.0

    # Blocks 2+3 — d[vel]/d[vel,att] : drag + thrust projection (numeric,
    # matches quad_pid_utils2.py's DCM/body-frame-drag translational model)
    F[3:6, 3:9] = _numeric_translational_jacobian(x, quad, rotor_speeds)

    # Block 4 — d[att-rate]/d[att] : Euler-angle kinematics
    sec2_theta = 1.0 / (ct**2)
    F[6, 6] = (cp*tt*q - sp*tt*r)
    F[6, 7] = (sp*q + cp*r) * sec2_theta
    F[7, 6] = (-sp*q - cp*r)
    F[8, 6] = ( cp/ct*q - sp/ct*r)
    F[8, 7] = (sp*q + cp*r)*tt/ct

    # Block 5 — d[att-rate]/d[body-rate]
    F[6, 9]  = 1.0
    F[6, 10] = sp * tt
    F[6, 11] = cp * tt
    F[7, 10] = cp
    F[7, 11] = -sp
    F[8, 10] = sp / ct
    F[8, 11] = cp / ct

    # Block 6 — d[body-rate]/d[body-rate] : Euler's equations + gyroscopic
    c1 = (Iyy - Izz) / Ixx
    c2 = (Izz - Ixx) / Iyy
    c3 = (Ixx - Iyy) / Izz
    F[9,  9]  = -Ar / Ixx
    F[9,  10] = c1 * r - (IR / Ixx) * Omega
    F[9,  11] = c1 * q
    F[10, 9]  = c2 * r + (IR / Iyy) * Omega
    F[10, 10] = -Ar / Iyy
    F[10, 11] = c2 * p
    F[11, 9]  = c3 * q
    F[11, 10] = c3 * p
    F[11, 11] = -Ar / Izz

    return F


# ============================================================================
#  MEASUREMENT MODELS
# ============================================================================
#
# GPS, gyroscope and magnetometer are linear in the state, so their measurement
# Jacobians are constant selector matrices.  The accelerometer is nonlinear
# (it measures the gravity projection), so its prediction and Jacobian are
# recomputed at the current estimate every update.

H_GPS = np.zeros((3, 12));  H_GPS[0, 0] = H_GPS[1, 1] = H_GPS[2, 2] = 1.0   # x,y,z
H_GYRO = np.zeros((3, 12)); H_GYRO[0, 9] = H_GYRO[1, 10] = H_GYRO[2, 11] = 1.0  # p,q,r
H_MAG = np.zeros((1, 12));  H_MAG[0, 8] = 1.0                                   # psi


def accel_h(x, quad=None, rotor_speeds=None, g=9.81):
    """Nonlinear accelerometer model: body-frame specific force.

    Full model: ``f_body = BI @ (a_inertial + [0,0,g])``, where
    ``a_inertial = dynamics(x)[3:6]`` is the actual (gravity-inclusive)
    translational acceleration and ``BI`` is quad_pid_utils.dynamics()'s own
    body-from-inertial matrix — i.e. this predicts exactly what
    ``imu.accelerometer`` simulates (gravity reaction + the vehicle's own
    drag/thrust-induced acceleration), not gravity alone. That match is what
    lets the accelerometer update carry real information about velocity
    instead of being mostly-discarded, R_acc-inflated noise.

    Backward-compatible fallback: if ``quad``/``rotor_speeds`` are omitted,
    returns the gravity-only term.
    """
    phi, theta = x[6], x[7]
    if quad is None or rotor_speeds is None:
        return np.array([g * np.sin(theta), -g * np.sin(phi) * np.cos(theta)])

    a_inertial = dynamics(0.0, x, quad, rotor_speeds)[3:6].copy()
    a_inertial[2] += g   # cancel dynamics()'s built-in "-g" so BI acts on pure specific force
    psi = x[8]
    BI = dcm321(phi, theta, psi) @ dcm321(phi=np.pi)
    f_body = BI @ a_inertial
    return f_body[:2]


def accel_H(x, quad=None, rotor_speeds=None, g=9.81, eps=1e-6):
    """Jacobian ``dh/dx`` of :func:`accel_h` at the current estimate (2 x 12).

    Numeric (central differences) when ``quad``/``rotor_speeds`` are given,
    since the full model routes through quad_pid_utils.dynamics() (DCM/
    body-frame-drag) and isn't practical to hand-differentiate reliably —
    same rationale as jacobian_F's Block 2/3. Falls back to the closed-form
    gravity-only Jacobian otherwise.
    """
    phi, theta = x[6], x[7]
    if quad is None or rotor_speeds is None:
        sp = np.sin(phi);  cp = np.cos(phi)
        st = np.sin(theta); ct = np.cos(theta)
        H = np.zeros((2, 12))
        H[0, 7] =  g * ct
        H[1, 6] = -g * cp * ct
        H[1, 7] =  g * sp * st
        return H

    H = np.zeros((2, 12))
    for si in (3, 4, 5, 6, 7, 8):   # vx,vy,vz,phi,theta,psi
        xp = x.copy(); xp[si] += eps
        xm = x.copy(); xm[si] -= eps
        H[:, si] = (accel_h(xp, quad, rotor_speeds, g) -
                    accel_h(xm, quad, rotor_speeds, g)) / (2.0 * eps)
    return H


# ============================================================================
#  SENSORS   (truth -> noisy, biased measurement)
# ============================================================================
#
# Each sensor follows the c4dynamics sensor pattern: it is constructed with its
# error parameters and exposes a ``measure`` method that maps the *true* state
# to a measurement.  Truth reaches the estimator only through these calls.
#
# White-noise standard deviations are 1-sigma.  A constant additive ``bias`` is
# supported for every channel (default zero, matching the reference prototype);
# enabling it reproduces the realistic GPS/IMU bias effects.

class gps:
    """GPS receiver — measures inertial position ``[x, y, z]`` (10 Hz)."""

    def __init__(self, noise_std=0.5, bias=None):
        self.noise_std = noise_std
        self.bias = np.zeros(3) if bias is None else np.asarray(bias, float)

    def measure(self, x_true):
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

        if x_true_prev is not None:  
            # inertial term (NOT modelled by accel_h)
            # Rotated via BI = dcm321(phi,theta,psi) @ dcm321(phi=pi), matching
            # quad_pid_utils.dynamics()'s actual body-from-inertial convention.
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

        c4d.plotdefaults(ax[0], 'Gyroscope', '', 'Heading [deg]', fontsize = 12)
        ax[0].legend(ncol=3)

        ax[1].plot(t, accel_true[:, 0], lw=2, label="ax true")
        ax[1].plot(t, accel_meas[:, 0], "--", label="ax measured")
        ax[1].plot(t, accel_true[:, 1], lw=2, label="ay true")
        ax[1].plot(t, accel_meas[:, 1], "--", label="ay measured")

        c4d.plotdefaults(ax[1], 'Accelerometer', 'Time [s]', 'Acceleration [m/s²]', fontsize = 12)
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

        c4d.plotdefaults(ax, 'Magnetometer Measurements', 'Time [s]', 'Heading [deg]', fontsize = 12)
        ax.legend()

        fig.tight_layout()

        if show:
            plt.show()

        return fig

# ============================================================================
#  EXTENDED KALMAN FILTER   (subclass of c4d.filters.ekf)
# ============================================================================

class ekf12(c4d.filters.ekf):
    """
    12-state quadcopter EKF.

    Built on :class:`c4dynamics.filters.ekf`, so the estimate is a first-class
    ``state`` object: it stores itself (``store``/``data``), prints and indexes
    by name (``est.phi``), and is read directly by the cascade-PID controllers.

    The class adds, as a thin layer over the framework's ``predict``/``update``:

    * a **2nd-order discretization** of the analytical Jacobian,
    * **innovation gating** (chi-squared NIS test) on every sensor,
    * **adaptive GPS measurement noise** (Mehra-style, two-phase),
    * **adaptive velocity process noise** during high-acceleration segments,
    * the **nonlinear accelerometer** measurement.

    The framework ``update`` is reused for the gain and covariance step; the
    (possibly nonlinear or angle-wrapped) innovation is supplied to it so the
    correction is identical to the reference prototype.
    """

    # Chi-squared 95% gates, by measurement dimension.
    _GATE_GPS  = 16.27
    _GATE_GYRO = 7.81
    _GATE_MAG  = 3.84
    _GATE_ACC  = 5.99

    def __init__(self, X0, P0, Q, R_gps, R_gyro, R_mag, R_acc, params):
        # X0 may be a dict {name: value} or a 12-vector.
        if not isinstance(X0, dict):
            X0 = {n: float(v) for n, v in zip(STATE_NAMES, np.asarray(X0).ravel())}
        super().__init__(X=X0, P0=P0, Q=Q.copy())

        self.R_gps  = R_gps.copy()
        self.R_gyro = R_gyro.copy()
        self.R_mag  = R_mag.copy()
        self.R_acc  = R_acc.copy()
        self._params = params

        # Attach physical parameters so this estimate can serve as the ``quad``
        # argument both to the shared dynamics (process model) and to
        # InitializeControllers — eliminating any proxy object.
        for k, v in params.items():
            setattr(self, k, v)

        # Adaptive GPS R (two-phase) — inflate R when innovations grow.
        self._R_gps_base      = R_gps.copy()
        self._r_scale         = 1.0
        self._r_scale_pending = 1.0
        self._innov_window    = deque(maxlen=10)
        self._innov_maxlen    = 10

        # Adaptive velocity Q — open the velocity channel during manoeuvres.
        self._Q_vel_base = np.diag(Q[3:6, 3:6]).copy()

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _wrap(a):
        return np.arctan2(np.sin(a), np.cos(a))

    def _gated_update(self, innov, H, R, gate):
        """Gate on NIS, then apply the framework update with innovation ``innov``.

        The framework ``update`` forms its innovation as ``z - H @ X``; feeding
        ``z = H @ X + innov`` makes that innovation exactly ``innov``, which lets
        the nonlinear-accelerometer and angle-wrapped-yaw corrections reuse the
        framework gain/covariance step unchanged.
        """
        innov = np.atleast_1d(innov).astype(float)
        S = H @ self.P @ H.T + R
        try:
            nis = float(innov @ np.linalg.solve(S, innov))
        except np.linalg.LinAlgError:
            return
        if nis > gate:
            return
        xvec = np.asarray(self.X).ravel()
        super().update(z=(H @ xvec + innov), H=H, R=R)

    # ── predict ──────────────────────────────────────────────────────────────

    def predict(self, dt, rotor_speeds):
        """
        EKF predict step.

        ``x <- x + dt f(x,u)`` (Euler), with the covariance propagated by the
        2nd-order discrete Jacobian ``F_d = I + dt F + dt^2/2 F^2`` and the
        adaptive process noise ``Q``.  The framework ``ekf.predict`` performs
        the state and covariance propagation; this method supplies ``f``, the
        adaptive ``Q``, and ``F_d`` evaluated at the predicted state.
        """
        x_now = np.asarray(self.X).ravel().copy()
        x_dot = dynamics(0.0, x_now, self, rotor_speeds)   # self carries params
        self._last_rotor_speeds = np.asarray(rotor_speeds).copy()

        # Adaptive velocity process noise (accel-magnitude scheduled).
        accel_mag = np.linalg.norm(x_dot[3:6])
        scale = min(2.0, 1.0 + (accel_mag - 0.4) * 1.5) if accel_mag > 0.4 else 1.0
        self.Q[3, 3] = self._Q_vel_base[0] * scale
        self.Q[4, 4] = self._Q_vel_base[1] * scale
        self.Q[5, 5] = self._Q_vel_base[2] * scale

        # Jacobian at the predicted, yaw-wrapped state.
        x_pred = x_now + dt * x_dot
        x_pred[8] = self._wrap(x_pred[8])
        # X-configuration gyro-coupling term (w1+w2-w3-w4), matching
        # quad_pid_utils.dynamics()'s motor layout.
        Omega = rotor_speeds[0] + rotor_speeds[1] - rotor_speeds[2] - rotor_speeds[3]
        Fc = jacobian_F(x_pred, Omega, self, rotor_speeds, self._params)
        F_d = np.eye(12) + dt * Fc + (0.5 * dt * dt) * (Fc @ Fc)

        # Framework predict: X += x_dot*dt ;  P = F_d P F_d^T + Q
        super().predict(F=F_d, fx=x_dot, dt=dt, Q=self.Q)
        self.psi = self._wrap(self.psi)     # wrap yaw after integration

    # ── per-sensor updates ────────────────────────────────────────────────────

    def update_gyro(self, z_gyro):
        innov = np.asarray(z_gyro).ravel() - (H_GYRO @ np.asarray(self.X).ravel())
        self._gated_update(innov, H_GYRO, self.R_gyro, self._GATE_GYRO)

    def update_accelerometer(self, z_acc):
        x = np.asarray(self.X).ravel()
        rs = getattr(self, '_last_rotor_speeds', None)
        innov = np.asarray(z_acc).ravel() - accel_h(x, self, rs)
        self._gated_update(innov, accel_H(x, self, rs), self.R_acc, self._GATE_ACC)

    def update_magnetometer(self, z_mag):
        # Circular innovation keeps the wrapped measurement off the state vector.
        innov = self._wrap(float(z_mag[0]) - float(self.psi))
        self._gated_update(np.array([innov]), H_MAG, self.R_mag, self._GATE_MAG)

    def update_gps(self, z_gps):
        """GPS position update with two-phase adaptive measurement noise.

        Phase A (always) collects the innovation and computes the *next* R
        scale; phase B applies the scale computed on the previous call so the
        gate and the gain use a consistent ``R``.
        """
        x = np.asarray(self.X).ravel()
        innov = np.asarray(z_gps).ravel() - (H_GPS @ x)

        # Phase A — observe innovation, compute pending scale.
        self._innov_window.append(innov.copy())
        if len(self._innov_window) == self._innov_maxlen:
            arr = np.array(self._innov_window)
            C_innov = (arr.T @ arr) / len(arr)
            S_expected = H_GPS @ self.P @ H_GPS.T + self.R_gps
            ratio = np.trace(C_innov) / np.trace(S_expected)
            alpha = 0.30
            new_scale = max(1.0, min(30.0, ratio))
            self._r_scale_pending = (1 - alpha) * self._r_scale + alpha * new_scale

        # Phase B — apply previous pending scale, gate, update.
        self._r_scale = self._r_scale_pending
        self.R_gps = self._R_gps_base * self._r_scale
        self._gated_update(innov, H_GPS, self.R_gps, self._GATE_GPS)


# ============================================================================
#  REFERENCE EKF CONFIGURATION BLOCK
# ============================================================================

def default_ekf_config():
    """Reference EKF noise / initialization block (process Q, measurement R,
    initial covariance, initial-offset and sensor-noise levels, sensor rates).

    Returned as a dict so the notebook can display and tune it as data — the
    estimation analogue of the controller-gain block in ``quad_pid_utils.py``.
    """
    Q = np.diag(np.array([
        0.005, 0.005, 0.008,   # x, y, z           [m]
        0.020, 0.020, 0.025,   # vx, vy, vz        [m/s]  
        0.008, 0.008, 0.010,   # phi, theta, psi   [rad]
        0.012, 0.012, 0.012,   # p, q, r           [rad/s]
    ])**2)
    P0 = np.diag(np.array([
        0.50, 0.50, 0.80,
        0.50, 0.50, 0.60,
        0.05, 0.05, 0.05,
        0.10, 0.10, 0.10,
    ])**2)
    return {
        'Q'      : Q,
        'P0'     : P0,
        'R_gps'  : np.diag([0.50**2, 0.50**2, 0.50**2]),
        'R_gyro' : np.diag([0.015**2, 0.015**2, 0.015**2]),
        'R_mag'  : np.array([[0.055**2]]),
        'R_acc'  : np.diag([0.10**2, 0.10**2]),
        # initial-estimate offset (1-sigma) from truth
        'x0_pos_sigma': 0.30,
        'x0_att_sigma': 0.05,
        # sensor white-noise levels (1-sigma) actually injected
        'gps_std' : 0.50,
        'gyro_std': 0.01,
        'mag_std' : 0.05,
        'acc_std' : 0.05,
        # sensor decimation relative to the 200 Hz master loop
        'gps_rate': 20,   # 10 Hz
        'mag_rate': 4,    # 50 Hz
        'seed'    : 42,
    }


# ============================================================================
#  CLOSED-LOOP ESTIMATION-CONTROL SIMULATION
# ============================================================================

def run_fig8_ekf(config, ekf_cfg=None):
    """
    Single closed-loop simulation: the quadcopter flies the figure-8 using the
    EKF estimate, while the EKF reconstructs that estimate from noisy GPS/IMU
    measurements of the truth.

    One time loop, two parallel ``state`` objects — the truth ``rigidbody`` and
    the ``ekf12`` estimate — both recorded via ``store``/``data``.

    Parameters
    ----------
    config : dict
        The same quad/trajectory/controller/sim configuration used by the
        cascade-PID example.
    ekf_cfg : dict, optional
        EKF noise / initialization block (see :func:`default_ekf_config`).

    Returns
    -------
    truth : c4d.rigidbody          true state history  (truth.data(...))
    est   : ekf12                  estimated state + covariance history
    diag  : dict                   {'t', 'nees', 'innov_gps_t', 'innov_gps'}
    """
    if ekf_cfg is None:
        ekf_cfg = default_ekf_config()

    dt, tf = config['sim']['dt'], config['sim']['tf']
    qp = config['quad']
    A, B, omega, z_ref = (config['trajectory']['A'], config['trajectory']['B'],
                          config['trajectory']['omega'], config['trajectory']['z_ref'])

    # ── Truth vehicle ────────────────────────────────────────────────────────
    truth = c4d.rigidbody()
    for k, v in qp.items():
        setattr(truth, k, v)
    truth.F = truth.m * truth.g
    truth.tau_phi = truth.tau_theta = truth.tau_psi = 0.0

    # ── Estimator (offset initial estimate) ──────────────────────────────────
    np.random.seed(ekf_cfg['seed'])
    x0 = np.zeros(12)                       # truth starts at rest on the ground
    x0[0:3] += np.random.randn(3) * ekf_cfg['x0_pos_sigma']
    x0[6:9] += np.random.randn(3) * ekf_cfg['x0_att_sigma']
    est = ekf12(x0, ekf_cfg['P0'], ekf_cfg['Q'],
                ekf_cfg['R_gps'], ekf_cfg['R_gyro'],
                ekf_cfg['R_mag'], ekf_cfg['R_acc'], qp)

    # ── Sensors ──────────────────────────────────────────────────────────────
    gps_sensor = gps(noise_std=ekf_cfg['gps_std'])
    imu_sensor = imu(gyro_std=ekf_cfg['gyro_std'], acc_std=ekf_cfg['acc_std'])
    mag_sensor = magnetometer(noise_std=ekf_cfg['mag_std'])
    gps_rate, mag_rate = ekf_cfg['gps_rate'], ekf_cfg['mag_rate']

    # ── Controllers read the ESTIMATE directly ────────────────────────────────
    outer_ctrl, mid_ctrl, inner_ctrl, allocator = InitializeControllers(
        config['controller'], est)

    Ts_outer, Ts_middle = 1.0 / 50.0, 1.0 / 100.0
    outer_time = middle_time = 0.0
    psi_d = phi_d = theta_d = 0.0
    p_d = q_d = r_d = 0.0
    T_cmd = truth.m * truth.g

    w_hover = np.sqrt(truth.m * truth.g / (4 * truth.kT))
    rotor_speeds = np.array([w_hover] * 4)

    N = int(round(tf / dt))
    t_hist = np.arange(N) * dt
    nees = np.zeros(N)
    innov_gps, innov_gps_t = [], []
    X_true_prev = np.asarray(truth.X).ravel().copy()
    mag_ctr = gps_ctr = 0

    print(f'EKF closed-loop  |  tf = {tf} s  |  dt = {dt} s')

    for k in range(N):
        t = t_hist[k]

        # 1. record truth, estimate (+ covariance via framework store), NEES
        truth.store(t)
        truth.storeparams(['F', 'tau_phi', 'tau_theta', 'tau_psi'], t=t)
        est.store(t)
        x_true = np.asarray(truth.X).ravel()
        x_err = x_true - np.asarray(est.X).ravel()
        try:
            nees[k] = float(x_err @ np.linalg.solve(est.P, x_err))
        except np.linalg.LinAlgError:
            nees[k] = np.nan

        # 2. predict
        est.predict(dt, rotor_speeds)

        # 3. correct — IMU every step, mag 50 Hz, GPS 10 Hz (truth via sensors)
        est.update_gyro(imu_sensor.gyro(x_true))
        est.update_accelerometer(
            imu_sensor.accelerometer(x_true, x_true_prev=X_true_prev, dt=dt))

        mag_ctr += 1
        if mag_ctr >= mag_rate:
            est.update_magnetometer(mag_sensor.measure(x_true))
            mag_ctr = 0

        gps_ctr += 1
        if gps_ctr >= gps_rate:
            z_gps = gps_sensor.measure(x_true)
            innov_gps.append(z_gps - np.asarray(est.X).ravel()[0:3]); innov_gps_t.append(t)
            est.update_gps(z_gps)
            gps_ctr = 0

        # 4. cascade PID on the ESTIMATE -> rotor speeds
        xd, yd, zd = position_reference(t, A, B, omega, z_ref, t_sim=tf)
        vxd_ff, vyd_ff = velocity_reference(t, A, B, omega, t_sim=tf)

        outer_time += dt
        if outer_time >= Ts_outer:
            T_cmd, phi_d, theta_d, psi_d = outer_ctrl.compute(
                xd, yd, zd, vxd_ff, vyd_ff, psi_d, est, Ts_outer)
            truth.F = T_cmd
            outer_time = 0.0

        middle_time += dt
        if middle_time >= Ts_middle:
            p_d, q_d, r_d = mid_ctrl.compute(phi_d, theta_d, psi_d, est, Ts_middle)
            middle_time = 0.0

        truth.tau_phi, truth.tau_theta, truth.tau_psi = inner_ctrl.compute(
            p_d, q_d, r_d, est, dt)
        rotor_speeds = np.array(allocator.allocate(
            truth.F, truth.tau_phi, truth.tau_theta, truth.tau_psi))

        # 5. propagate the TRUTH one step
        X_true_prev = x_true.copy()
        sol = solve_ivp(dynamics, [t, t + dt], truth.X,
                        args=(truth, rotor_speeds), method='RK45')
        truth.X = sol.y[:, -1]

    print('EKF closed-loop complete.')
    diag = {'t': t_hist, 'nees': nees,
            'innov_gps_t': np.array(innov_gps_t), 'innov_gps': np.array(innov_gps)}
    return truth, est, diag


# ============================================================================
#  METRICS  and  VISUALISATION
# ============================================================================

def compute_metrics(truth, est, diag, t0=8.0, t1=82.0, verbose=True):
    """Per-state RMSE (true vs estimated) and mean NEES over the figure-8 phase."""
    t = diag['t']
    mask = (t >= t0) & (t <= t1)
    rmse = {}
    for n in STATE_NAMES:
        xt = np.asarray(truth.data(n)[1])
        xe = np.asarray(est.data(n)[1])
        rmse[n] = float(np.sqrt(np.mean((xt[mask] - xe[mask])**2)))
    nees_mean = float(np.nanmean(diag['nees'][mask]))
    if verbose:
        print(f"{'state':<8}{'RMSE':>12}")
        print('-' * 20)
        for n in STATE_NAMES:
            print(f"{n:<8}{rmse[n]:>12.4f}")
        print('-' * 20)
        print(f"{'NEES':<8}{nees_mean:>12.2f}   (ideal ~ 12)")
    return {'rmse': rmse, 'nees_mean': nees_mean}


def plot_estimation(truth, est, diag, states=('x', 'z', 'phi', 'psi')):
    """True-vs-estimated trajectories with +-2 sigma covariance bands + NEES."""
    import matplotlib.pyplot as plt

    idx = {n: i for i, n in enumerate(STATE_NAMES)}
    nrows = len(states) + 1
    fig, axes = plt.subplots(nrows, 1, figsize=(9, 2.4 * nrows))

    for ax, n in zip(axes[:-1], states):
        t, xt = truth.data(n)
        _, xe = est.data(n)
        sig = np.sqrt(np.asarray(est.data(f'P{idx[n]}{idx[n]}')[1]))
        scale = c4d.r2d if n in ('phi', 'theta', 'psi', 'p', 'q', 'r') else 1.0
        xt = np.asarray(xt) * scale; xe = np.asarray(xe) * scale; sig = sig * scale
        ax.plot(t, xt, 'k-', lw=1.5, label='true')
        ax.plot(t, xe, 'C1--', lw=1.2, label='estimated')
        ax.fill_between(t, xe - 2*sig, xe + 2*sig, color='C1', alpha=0.2,
                        label=r'$\pm 2\sigma$')
        unit = '[deg]' if scale != 1.0 else '[m]'
        c4d.plotdefaults(ax, f'{n}', 'time [s]', f'{n} {unit}', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)

    ax = axes[-1]
    ax.plot(diag['t'], diag['nees'], 'C0-', lw=0.8)
    ax.axhline(12, color='k', ls=':', lw=1, label='ideal (n=12)')
    c4d.plotdefaults(ax, 'Filter consistency (NEES)', 'time [s]', 'NEES', fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 40)

    fig.tight_layout()
    return fig

def plot_trajectory(truth, est, config, t0=8.0, t1=82.0):
    """Figure-8 path: reference vs. true vs. estimated (top-down x-y view)."""
    import matplotlib.pyplot as plt
    from quad_pid_utils import position_reference

    # flown paths from the stored histories
    t,  x_true = truth.data('x');  _, y_true = truth.data('y')
    _,  x_est  = est.data('x');    _, y_est  = est.data('y')
    t = np.asarray(t)

    # rebuild the reference figure-8 over the same time vector
    A, B, omega, z_ref = (config['trajectory']['A'], config['trajectory']['B'],
                          config['trajectory']['omega'], config['trajectory']['z_ref'])
    tf = config['sim']['tf']
    ref = np.array([position_reference(ti, A, B, omega, z_ref, t_sim=tf) for ti in t])
    x_ref, y_ref = ref[:, 0], ref[:, 1]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(x_ref,  y_ref,  'k--', lw=1.5, label='reference')
    ax.plot(np.asarray(x_true), np.asarray(y_true), 'C0-', lw=1.5, label='true')
    ax.plot(np.asarray(x_est),  np.asarray(y_est),  'C1:', lw=1.8, label='estimated')
    ax.plot(x_true[0], y_true[0], 'go', ms=8, label='start')
    ax.set_aspect('equal', 'box')
    c4d.plotdefaults(ax, 'Figure-8 trajectory:  reference vs. true vs. estimated',
                     'x  [m]', 'y  [m]', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    return fig