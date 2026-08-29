"""
ekf.py — Extended Kalman Filter state estimation for the quadcopter
============================================================================

This module is the estimation companion to ``c4dynamics.controllers.quad_pid``.
Where the cascade-PID module provides the *plant* and the *controller*, this module
provides the *sensors* and the *estimator* that let the same quadcopter fly its
figure-8 using an estimated state reconstructed from noisy GPS and IMU
measurements rather than from perfect truth.

It is organised exactly like ``c4dynamics.controllers.quad_pid`` — a single reusable module
that the notebook (``ekf.ipynb``) drives through a configuration dictionary and
a single call to :func:`run_fig8_ekf`.

Design
------
* The truth vehicle is a :class:`c4dynamics.rigidbody`, propagated with the
  shared :func:`c4dynamics.controllers.quad_pid.dynamics <c4dynamics.controllers.quad_pid.dynamics>` (the *same* model the EKF uses as its
  process model).
* The estimate is an :class:`ekf <c4dynamics.filters.ekf>` whose 12 state
  variables carry the *same names and order* as the truth.  Because the
  framework EKF is itself a ``state`` object, the estimate stores itself
  (``store``/``data``), exposes named components (``est.phi`` …), and is read
  directly by the cascade-PID controllers — no proxy object is needed.
* The sensors follow the c4dynamics sensor pattern: each exposes a
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
identical to ``c4dynamics.controllers.quad_pid`` (X motor configuration, dcm321-based
rotations). jacobian_F's translational blocks are computed numerically
against dynamics() directly (see jacobian_F docstring) so this
consistency is enforced by construction rather than by hand-derivation.

Contents
--------
    jacobian_F            process Jacobian df/dx (12x12): analytic kinematic/
                           rotational blocks + numeric translational block
    H_GPS, H_GYRO, H_MAG  constant measurement matrices
    accel_h, accel_H      nonlinear accelerometer measurement model + Jacobian
    gps, imu, magnetometer simulated sensors (truth -> measurement)
    ekf_quad              the quadrotor EKF (subclass of c4d.filters.ekf)
    default_ekf_config    reference EKF noise / initialization block
                           (imported from the sibling ekf_config.py)
    run_fig8_ekf          single closed-loop estimation-control simulation
    compute_metrics       RMSE (true vs estimated) + filter-consistency (NEES)
    plot_estimation       true-vs-estimated visualisation with +-2 sigma bands
    plot_dashboard        2x3 results dashboard (reference/true/estimated),
                           mirroring quad_pid.plot_results
"""

import warnings
import numpy as np
from collections import deque
from scipy.integrate import solve_ivp

import c4dynamics as c4d
from c4dynamics.rotmat import dcm321
from c4dynamics.controllers.quad_pid import (dynamics, position_reference,
                                             velocity_reference, InitializeControllers,
                                             ground_contact)
from c4dynamics.sensors.navigation import gps, imu, magnetometer
from c4dynamics import g_ms2 as g
from docs.source.programs.ekf_estimation.ekf_config import default_ekf_config
EPS = 1e-6

# State-variable names, in the canonical c4dynamics rigidbody order.
STATE_NAMES = ['x', 'y', 'z', 'vx', 'vy', 'vz',
               'phi', 'theta', 'psi', 'p', 'q', 'r']


# ============================================================================
#  PROCESS MODEL JACOBIAN   F = df/dx   (12 x 12)
# ============================================================================

def _numeric_translational_jacobian(x, quad, rotor_speeds):
    """
    Numerically differentiate the translational rows of
    :func:`c4dynamics.controllers.quad_pid.dynamics <c4dynamics.controllers.quad_pid.dynamics>` (indices 3,4,5 = dvx,dvy,dvz) with respect
    to [vx,vy,vz,phi,theta,psi] (state indices 3..8), via central differences.

    c4dynamics.controllers.quad_pid's translational model rotates velocity into the body
    frame, applies drag there, then rotates the resulting force back to
    inertial (``BI``/``dcm321``-based) — this couples drag to attitude and
    is no longer a simple closed-form block, so it is differentiated
    numerically here instead of hand-derived. This keeps the Jacobian
    correct automatically if dynamics() changes again.

    Returns
    -------
    block : np.ndarray (3, 6) — d[dvx,dvy,dvz] / d[vx,vy,vz,phi,theta,psi]
    """
    idx = [3, 4, 5, 6, 7, 8]
    block = np.zeros((3, 6))
    for j, si in enumerate(idx):
        xp = x.copy(); xp[si] += EPS
        xm = x.copy(); xm[si] -= EPS
        fp = dynamics(0.0, xp, quad, rotor_speeds)[3:6]
        fm = dynamics(0.0, xm, quad, rotor_speeds)[3:6]
        block[:, j] = (fp - fm) / (2.0 * EPS)
    return block


def jacobian_F(x, Omega, quad, rotor_speeds, params):
    """
    Jacobian of the rigid-body dynamics ``f(x, u)`` evaluated at the current
    state estimate.

    The process model ``f`` is :func:`c4dynamics.controllers.quad_pid.dynamics <c4dynamics.controllers.quad_pid.dynamics>`. This is a
    hybrid Jacobian:
      - Blocks 1, 4, 5, 6 (position kinematics, Euler-angle kinematics,
        Euler's rotational equations + gyroscopic coupling) are closed-form.
        These depend only on body-frame angular rates and inertia, so they
        are unaffected by the inertial-frame/motor-layout convention and are
        identical regardless of inertial-frame/motor-layout convention.
      - Blocks 2, 3 (translational drag and thrust-to-attitude coupling) are
        computed numerically via :func:`_numeric_translational_jacobian`,
        since the DCM/body-frame-drag translational model couples drag to
        attitude and is impractical to hand-differentiate reliably.

    Parameters
    ----------
    x : np.ndarray (12,)
        Current state estimate ``[x,y,z, vx,vy,vz, phi,theta,psi, p,q,r]``.
    Omega : float
        Net rotor speed ``w1 + w2 - w3 - w4`` [rad/s] for gyroscopic coupling
        (X-configuration convention, matching dynamics()).
    quad : object
        Quad-like object exposing the physical parameters as attributes
        (m, g, l, kT, kQ, Ixx, Iyy, Izz, Ar, IR, Ax, Ay, Az) — passed straight
        through to dynamics() for the numeric block.
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
    # matches c4dynamics.controllers.quad_pid's DCM/body-frame-drag translational model)
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


def accel_h(x, quad=None, rotor_speeds=None):
    """Accelerometer model: body-frame specific force.

    Full model: ``f_body = BI @ (a_inertial + [0,0,g])``, where
    ``a_inertial = dynamics(x)[3:6]`` is the actual (gravity-inclusive)
    translational acceleration and ``BI`` is dynamics()'s own
    body-from-inertial matrix — i.e. this predicts exactly what
    ``imu.measure`` simulates on its accelerometer channel (gravity reaction
    + the vehicle's own drag/thrust-induced acceleration).
    That match is what
    lets the accelerometer update carry real information about velocity
    instead of being mostly-discarded, R_acc-inflated noise.

    Backward-compatible fallback: if ``quad``/``rotor_speeds`` are omitted,
    returns the gravity-only term (old behaviour).
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


def accel_H(x, quad=None, rotor_speeds=None):
    """Jacobian ``dh/dx`` of :func:`accel_h` at the current estimate (2 x 12).

    Numeric (central differences) when ``quad``/``rotor_speeds`` are given,
    since the full model routes through dynamics() (DCM/
    body-frame-drag) and isn't practical to hand-differentiate reliably —
    same rationale as jacobian_F's Block 2/3. Falls back to the closed-form
    gravity-only Jacobian otherwise.
    """
    phi, theta = x[6], x[7]
    if quad is None or rotor_speeds is None:
        sp = np.sin(phi)
        cp = np.cos(phi)
        st = np.sin(theta)
        ct = np.cos(theta)
        H = np.zeros((2, 12))
        H[0, 7] =  g * ct
        H[1, 6] = -g * cp * ct
        H[1, 7] =  g * sp * st
        return H

    H = np.zeros((2, 12))
    for si in (3, 4, 5, 6, 7, 8):   # vx,vy,vz,phi,theta,psi
        xp = x.copy()
        xp[si] += EPS
        xm = x.copy()
        xm[si] -= EPS
        H[:, si] = (accel_h(xp, quad, rotor_speeds) -
                    accel_h(xm, quad, rotor_speeds)) / (2.0 * EPS)
    return H



# ============================================================================
#  EXTENDED KALMAN FILTER   (subclass of c4d.filters.ekf)
# ============================================================================

class ekf_quad(c4d.filters.ekf):
    """
    Quadcopter EKF.

    Built on :class:`c4dynamics.filters.ekf`, so the estimate is a first-class
    ``state`` object:
    it stores itself (``store``/``data``), prints and indexes
    by name (``est.phi``),
    and is read directly by the cascade-PID controllers.

    The class adds, as a thin layer over the framework's ``predict``/``update``:

    * a 2nd-order discretization of the analytical Jacobian,
    * per-sensor innovation gating (chi-squared NIS test),
    * adaptive GPS measurement noise (Mehra-style, two-phase),
    * adaptive velocity process noise during high-acceleration segments,
    * the nonlinear accelerometer measurement.

    The NIS gate, the gain/covariance step, and covariance stabilization
    (symmetrize + tiny diagonal floor) are all native to the framework's
    ``kalman.predict``/``update`` (via ``gate=``/``innov=``/``P_jitter=``);
    this layer only has to supply the (possibly nonlinear or angle-wrapped)
    innovation itself.
    """

    # Chi-squared gates, by measurement dimension:
    _GATE_GPS  = 16.27 # 99.9% gate, 3 dimensions
    _GATE_GYRO = 7.81  # 95%, 3 dimensions
    _GATE_MAG  = 3.84  # 95%, 1 dimension
    _GATE_ACC  = 5.99  # 95%, 2 dimensions

    def __init__(self, X0, P0, Q, R_gps, R_gyro, R_mag, R_acc, params, dt_ref=0.005):
        # X0 may be a dict {name: value} or a 12-vector.
        if not isinstance(X0, dict):
            X0 = {n: float(v) for n, v in zip(STATE_NAMES, np.asarray(X0).ravel())}
        # P_jitter: tiny diagonal floor, symmetrized alongside; native to the
        # framework's predict()/update() now -- see kalman.__init__.
        super().__init__(X=X0, P0=P0, Q=Q.copy(), P_jitter=1e-9)

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
        self._alpha_mehra     = 0.30

        # ``Q`` (both the diag() config values and the adaptive velocity
        # scaling below) is tuned for one predict step of duration ``dt_ref``
        # (the 200 Hz baseline used throughout this example). ``predict()``
        # rescales it linearly by dt/dt_ref every call, so calling predict()
        # at a different step size (e.g. sub-stepped for a high-rate IMU
        # experiment) injects the same *continuous-time* noise density
        # instead of the same fixed amount per call regardless of dt.
        self._Q_ref   = Q.copy()
        self._dt_ref  = float(dt_ref)

        # Adaptive velocity Q — open the velocity channel during manoeuvres.
        self._Q_vel_base = np.diag(Q[3:6, 3:6]).copy()

        # jacobian_stride: recompute the *numeric* Jacobians (jacobian_F's
        # translational block, accel_H) only every `jacobian_stride` calls,
        # reusing the last value in between. Neither Jacobian affects the
        # state estimate (that's propagated directly from the nonlinear
        # dynamics()); they only shape the covariance P, which evolves far
        # more smoothly than the state does, so a slightly stale F_d/H_acc is
        # a standard, well-understood EKF approximation — not a correctness
        # risk of the kind a hand-derived-and-possibly-wrong closed form
        # would be. Default 1 = recompute every call (exactly today's
        # behaviour, bit-for-bit). Set >1 (e.g. via run_fig8_ekf) only to
        # afford a high IMU rate in a sweep; leave at 1 for the reference run.
        self.jacobian_stride = 1
        self._jac_ctr = 0
        self._accelH_ctr = 0
        self._F_d_cached = None
        self._accel_H_cached = None

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _wrap(a):
        return np.arctan2(np.sin(a), np.cos(a))


    # ── predict ──────────────────────────────────────────────────────────────

    def predict(self, dt, rotor_speeds):
        """
        EKF predict step.

        ``x <- x + dt f(x,u)`` (Euler), with the covariance propagated by the
        2nd-order discrete Jacobian ``F_d = I + dt F + dt^2/2 F^2`` and the
        adaptive process noise ``Q``.  The framework ``ekf.predict`` performs
        the state and covariance propagation; this method supplies ``f``, the
        adaptive ``Q``, and ``F_d`` evaluated at the predicted state.

        ``Q`` is first rescaled to the actual step size (``Q_ref * dt /
        dt_ref``, see ``__init__``), *then* the accel-magnitude-scheduled
        velocity scaling is applied on top — so calling ``predict`` more or
        less often than the ``dt_ref`` baseline (e.g. a sub-stepped, high
        IMU-rate run) injects the same total process noise per second rather
        than the same amount per call.
        """
        x_now = np.asarray(self.X).ravel().copy()
        x_dot = dynamics(0.0, x_now, self, rotor_speeds)   # self carries params
        self._last_rotor_speeds = np.asarray(rotor_speeds).copy()

        # Rescale the reference (dt_ref-tuned) Q to this step's actual dt.
        q_dt_scale = dt / self._dt_ref
        self.Q = self._Q_ref * q_dt_scale

        # Adaptive velocity process noise (accel-magnitude scheduled), applied
        # on top of the dt-rescaled base.
        accel_mag = np.linalg.norm(x_dot[3:6])
        scale = min(2.0, 1.0 + (accel_mag - 0.4) * 1.5) if accel_mag > 0.4 else 1.0
        self.Q[3, 3] = self._Q_vel_base[0] * q_dt_scale * scale
        self.Q[4, 4] = self._Q_vel_base[1] * q_dt_scale * scale
        self.Q[5, 5] = self._Q_vel_base[2] * q_dt_scale * scale

        # Jacobian at the predicted, yaw-wrapped state. The numeric block
        # inside jacobian_F (6 extra dynamics() calls) is the expensive part;
        # cache it across `jacobian_stride` calls (default 1 = every call,
        # unchanged) since only P uses it, not the state estimate above.
        x_pred = x_now + dt * x_dot
        x_pred[8] = self._wrap(x_pred[8])
        # X-configuration gyro-coupling term (w1+w2-w3-w4), matching
        # dynamics()'s motor layout.
        Omega = rotor_speeds[0] + rotor_speeds[1] - rotor_speeds[2] - rotor_speeds[3]
        if self._F_d_cached is None or self._jac_ctr % self.jacobian_stride == 0:
            self._Fc_cached = jacobian_F(x_pred, Omega, self, rotor_speeds, self._params)
        self._jac_ctr += 1
        Fc = self._Fc_cached
        F_d = np.eye(12) + dt * Fc + (0.5 * dt * dt) * (Fc @ Fc)
        self._F_d_cached = F_d

        # Framework predict: X += x_dot*dt ;  P = F_d P F_d^T + Q, stabilized
        # (symmetrized + tiny diagonal floor) natively since P_jitter is set.
        super().predict(F=F_d, fx=x_dot, dt=dt, Q=self.Q)
        self.psi = self._wrap(self.psi)     # wrap yaw after integration

    # ── per-sensor updates ────────────────────────────────────────────────────

    def update_gyro(self, z_gyro):
        innov = np.asarray(z_gyro).ravel() - (H_GYRO @ np.asarray(self.X).ravel())
        super().update(innov=innov, H=H_GYRO, R=self.R_gyro, gate=self._GATE_GYRO)

    def update_accelerometer(self, z_acc):
        x = np.asarray(self.X).ravel()
        rs = getattr(self, '_last_rotor_speeds', None)
        innov = np.asarray(z_acc).ravel() - accel_h(x, self, rs)   # always fresh
        # accel_H (the expensive part -- 12 extra dynamics() calls via central
        # differences) is cached across `jacobian_stride` calls, same
        # rationale as jacobian_F's numeric block in predict().
        if self._accel_H_cached is None or self._accelH_ctr % self.jacobian_stride == 0:
            self._accel_H_cached = accel_H(x, self, rs)
        self._accelH_ctr += 1
        super().update(innov=innov, H=self._accel_H_cached, R=self.R_acc, gate=self._GATE_ACC)

    def update_magnetometer(self, z_mag):
        # Circular innovation keeps the wrapped measurement off the state vector.
        innov = self._wrap(float(z_mag[0]) - float(self.psi))
        super().update(innov=innov, H=H_MAG, R=self.R_mag, gate=self._GATE_MAG)

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
            new_scale = max(1.0, min(30.0, ratio)) # never smaller than 1, never bigger than 30.
            self._r_scale_pending = (1 - self._alpha_mehra) * self._r_scale + self._alpha_mehra * new_scale

        # Phase B — apply previous pending scale, gate, update.
        self._r_scale = self._r_scale_pending
        self.R_gps = self._R_gps_base * self._r_scale
        super().update(innov=innov, H=H_GPS, R=self.R_gps, gate=self._GATE_GPS)

# ============================================================================
#  CLOSED-LOOP ESTIMATION-CONTROL SIMULATION
# ============================================================================
#
# default_ekf_config() (used below as the ekf_cfg fallback) lives in the
# sibling ekf_config.py, imported at the top of this module.

def run_fig8_ekf(
        config, ekf_cfg=None, imu_rate_hz=None, jacobian_stride=1,
        gps_dropout=None, imu_noise_density_model=True, verbose=True
    ):
    """
    Single closed-loop simulation: the quadcopter flies the figure-8 using the
    EKF estimate, while the EKF reconstructs that estimate from noisy GPS/IMU
    measurements of the truth.

    One time loop, two parallel ``state`` objects — the truth ``rigidbody`` and
    the ``ekf_quad`` estimate — both recorded via ``store``/``data``.

    Parameters
    ----------
    config : dict
        A dict with ``quad``/``trajectory``/``controller``/``sim`` blocks
        configuring the vehicle parameters, the figure-8 reference
        trajectory, the PID gains, and simulation timing.
        ``config['sim']['dt']`` is the control rate (200 Hz by default):
        the rate-loop controller and the rotor-speed commands always run
        at this fixed rate, independent of ``imu_rate_hz``.
    ekf_cfg : dict, optional
        EKF noise / initialization block (see :func:`default_ekf_config`).
    imu_rate_hz : float, optional
        Gyro/accelerometer sampling rate [Hz]. Defaults to ``1/config['sim']['dt']``
        (200 Hz) — the original, single-rate behaviour, byte-for-byte. Rates
        below 200 Hz simply skip IMU updates on some steps (like GPS/mag decimation
        below); rates above 200 Hz make the *EKF's own* predict/update cycle
        (and the truth propagation) sub-step *within* each 200 Hz control frame,
        holding the commanded rotor speeds fixed across the sub-steps — so the
        rate-controller bandwidth and every other loop stay exactly as
        configured. The simulated gyro/accel noise std and R_gyro/R_acc are
        also rescaled by ``sqrt(imu_rate_hz / 200)`` — a real IMU's noise
        *density* is fixed, not its per-sample noise, so a faster IMU gets
        noisier individual samples, not free extra precision.
        With both of these controlled for, IMU rate is the only thing that changes across
        a sweep.
    jacobian_stride : int, optional
        Recompute the numeric EKF Jacobians every N calls instead of every
        call (see :class:`ekf_quad`). Default 1 = off, byte-for-byte unchanged.
    gps_dropout : (float, float), optional
        ``(t_start, t_end)`` window [s] during which GPS updates are skipped
        entirely — magnetometer and IMU keep updating as usual. Simulates a
        GPS-denied period: with the one *absolute position* aiding source
        gone, the estimate has to dead-reckon on IMU (and gyro/mag for
        attitude) alone until GPS returns, which is exactly the regime where
        IMU rate should matter (Section 5.3) — unlike the always-aided case,
        where GPS's own 10 Hz correction dominates regardless of IMU rate.
        See :func:`run_gps_dropout_sweep`.

    Returns
    -------
    quad_true : c4d.rigidbody          true state history  (quad_true.data(...))
    est   : ekf_quad                  estimated state + covariance history
    diag  : dict                   {'t', 'nees', 'innov_gps_t', 'innov_gps'}
    """
    if ekf_cfg is None:
        ekf_cfg = default_ekf_config()

    dt, tf = config['sim']['dt'], config['sim']['tf']       # control ("inner-loop") rate, fixed
    qp = config['quad']
    A, B, omega, z_ref = (config['trajectory']['A'], config['trajectory']['B'],
                          config['trajectory']['omega'], config['trajectory']['z_ref'])
    # Takeoff/landing durations default to the reference example's 8 s/8 s,
    # but are configurable so a shorter run (e.g. the IMU-rate sweep below,
    # which uses much shorter tf to keep runtime reasonable) can scale them
    # down and still reach a real cruise phase instead of staying stuck
    # mid-climb the whole time.
    t_takeoff = config['trajectory'].get('t_takeoff', 8.0)
    t_land    = config['trajectory'].get('t_land', 8.0)

    if imu_rate_hz is None:
        imu_rate_hz = 1.0 / dt

    # `dt` (=Ts_inner) stays the actuator/rate-controller period throughout.
    # `dt_sim` is the actual per-iteration integration/estimation step: the
    # finer of the control period and the requested IMU period, snapped so it
    # divides Ts_inner exactly. IMU updates fire every `imu_decim` sim-steps.
    Ts_inner = dt
    dt_sim = min(Ts_inner, 1.0 / imu_rate_hz)
    n_per_ctrl = max(1, round(Ts_inner / dt_sim))
    dt_sim = Ts_inner / n_per_ctrl
    imu_decim = max(1, round((1.0 / imu_rate_hz) / dt_sim))

    # `ekf_cfg['gyro_std']/['acc_std']/['R_gyro']/['R_acc']` are calibrated for
    # one sample at the 200 Hz reference rate. A real IMU's noise *density* is
    # what's fixed, not its per-sample noise -- sampling faster means each
    # individual sample is noisier (sigma ~ sqrt(rate)), not free extra
    # precision. Rescale so a faster IMU is only better because it propagates
    # the estimate more often, not because each sample is unrealistically as
    # clean as a slower sensor's.
    imu_rate_ref = 1.0 / dt
    imu_noise_scale = np.sqrt(imu_rate_hz / imu_rate_ref) if imu_noise_density_model else 1.0

    # ── Truth vehicle ────────────────────────────────────────────────────────
    quad_true = c4d.rigidbody()
    for k, v in qp.items():
        setattr(quad_true, k, v)
    quad_true.T = quad_true.m * quad_true.g
    quad_true.tau_x = quad_true.tau_y = quad_true.tau_z = 0.0

    # ── Estimator (offset initial estimate) ──────────────────────────────────
    np.random.seed(ekf_cfg['seed'])
    x0 = np.zeros(12)                       # quad_true starts at rest on the ground
    x0[0:3] += np.random.randn(3) * ekf_cfg['x0_pos_sigma']
    x0[6:9] += np.random.randn(3) * ekf_cfg['x0_att_sigma']
    # dt_ref=dt: Q in ekf_cfg is tuned for one Ts_inner(=dt)-long predict step;
    # ekf_quad.predict() rescales it internally to whatever dt_sim actually is.
    R_gyro_eff = ekf_cfg['R_gyro'] * imu_noise_scale**2
    R_acc_eff  = ekf_cfg['R_acc']  * imu_noise_scale**2
    est = ekf_quad(x0, ekf_cfg['P0'], ekf_cfg['Q'],
                ekf_cfg['R_gps'], R_gyro_eff,
                ekf_cfg['R_mag'], R_acc_eff, qp, dt_ref=dt)
    est.jacobian_stride = jacobian_stride

    # ── Sensors ──────────────────────────────────────────────────────────────
    gps_sensor = gps(noise_std=ekf_cfg['gps_std'], isideal=ekf_cfg.get('ideal_gps', False))
    imu_sensor = imu(isideal=ekf_cfg['ideal_imu'], gyro_std=ekf_cfg['gyro_std'] * imu_noise_scale,
                      acc_std=ekf_cfg['acc_std'] * imu_noise_scale)
    mag_sensor = magnetometer(noise_std=ekf_cfg['mag_std'])
    # gps_rate/mag_rate are counts of Ts_inner periods (e.g. gps_rate=20 -> 10 Hz
    # at the 200 Hz baseline); convert to sim-step counts at the actual dt_sim.
    gps_decim = max(1, round(ekf_cfg['gps_rate'] * dt / dt_sim))
    mag_decim = max(1, round(ekf_cfg['mag_rate'] * dt / dt_sim))

    # ── Controllers read the ESTIMATE directly ────────────────────────────────
    outer_ctrl, mid_ctrl, inner_ctrl, allocator = InitializeControllers(
        config['controller'], est)

    Ts_outer, Ts_middle = 1.0 / 50.0, 1.0 / 100.0
    outer_time = middle_time = inner_time = 0.0
    psi_d = phi_d = theta_d = 0.0
    p_d = q_d = r_d = 0.0
    T_cmd = quad_true.m * quad_true.g

    w_hover = np.sqrt(quad_true.m * quad_true.g / (4 * quad_true.kT))
    rotor_speeds = np.array([w_hover] * 4)

    N = int(round(tf / dt_sim))
    t_hist = np.arange(N) * dt_sim
    nees = np.zeros(N)
    innov_gps, innov_gps_t = [], []
    mag_ctr = gps_ctr = imu_ctr = 0
    took_off = False

    if verbose:
        print(f'EKF closed-loop  |  tf = {tf} s  |  control dt = {dt} s  |  '
              f'IMU = {imu_rate_hz:g} Hz  (sim dt = {dt_sim:g} s)')

    for k in range(N):
        t = t_hist[k]

        # 1. record quad_true, estimate (+ covariance via framework store), NEES
        quad_true.store(t)
        quad_true.storeparams(['T', 'tau_x', 'tau_y', 'tau_z'], t=t)
        est.store(t)
        x_true = np.asarray(quad_true.X).ravel()
        x_err = x_true - np.asarray(est.X).ravel()
        try:
            nees[k] = float(x_err @ np.linalg.solve(est.P, x_err))
        except np.linalg.LinAlgError:
            nees[k] = np.nan

        # Ground-contact guard (shared with the plain cascade-PID example):
        # stop as soon as the real (quad_true) vehicle comes back down to z <= 0
        # after having been airborne -- there's no ground-collision model in
        # dynamics(), so continuing past this point would just integrate an
        # unphysical, below-ground trajectory.
        took_off, hit_ground = ground_contact(quad_true.z, took_off)
        if hit_ground:
            warnings.warn(f'Quadcopter hit the ground at t={t:.3f} s '
                           f'(z={quad_true.z:.4f} m); stopping simulation.', c4d.c4warn)
            t_hist, nees = t_hist[:k + 1], nees[:k + 1]
            break

        # 2. predict, at the actual sim step (may sub-step within a control frame)
        est.predict(dt_sim, rotor_speeds)

        # 3. correct — IMU at imu_rate_hz, mag 50 Hz, GPS 10 Hz (quad_true via sensors)
        # imu_sensor.measure() takes the quad_true rigidbody directly and keeps its own
        # previous-sample state internally, so the inertial term's dt is just t - t_prev.
        # az_meas is unused: this filter's predict step already knows the commanded
        # thrust, so az carries little the accelerometer update would add; only (ax, ay) — the tilt-sensitive axes — feed update_accelerometer.
        imu_ctr += 1
        if imu_ctr >= imu_decim:
            ax_meas, ay_meas, az_meas, p_meas, q_meas, r_meas = imu_sensor.measure(quad_true, t=t)
            est.update_gyro([p_meas, q_meas, r_meas])
            est.update_accelerometer([ax_meas, ay_meas])
            imu_ctr = 0

        mag_ctr += 1
        if mag_ctr >= mag_decim:
            est.update_magnetometer(mag_sensor.measure(x_true))
            mag_ctr = 0

        gps_ctr += 1
        if gps_ctr >= gps_decim:
            in_dropout = gps_dropout is not None and gps_dropout[0] <= t <= gps_dropout[1]
            if not in_dropout:
                z_gps = gps_sensor.measure(x_true)
                innov_gps.append(z_gps - np.asarray(est.X).ravel()[0:3])
                innov_gps_t.append(t)
                est.update_gps(z_gps)
            gps_ctr = 0

        # Divergence guard:
        # once X or P goes non-finite (e.g. an unbounded covariance during a
        # long GPS-denied stretch feeding an overflowed matmul),
        # stop rather than let a NaN/Inf state reach solve_ivp
        # --
        # scipy's adaptive RK45 can hang retrying ever-smaller steps against
        # a NaN derivative instead of failing fast.
        # Truncating the run here is what lets a diverged trial
        # actually finish (with garbage tail values a caller can
        # detect via NEES/isfinite) instead of never returning at all.
        if not (np.all(np.isfinite(np.asarray(est.X))) and np.all(np.isfinite(est.P))):
            if verbose:
                print(f'EKF diverged at t={t:.3f}s (non-finite state/covariance); '
                      f'stopping early.')
            # quad_true/est only have entries up to (and including) this
            # iteration's store() call; truncate the pre-allocated diagnostic
            # arrays to match so every returned array stays the same length.
            t_hist, nees = t_hist[:k + 1], nees[:k + 1]
            break

        # 4. cascade PID on the ESTIMATE -> rotor speeds
        xd, yd, zd = position_reference(t, A, B, omega, z_ref,
                                         t_takeoff=t_takeoff, t_land=t_land, t_sim=tf)
        vxd_ff, vyd_ff = velocity_reference(t, A, B, omega,
                                             t_takeoff=t_takeoff, t_land=t_land, t_sim=tf)

        outer_time += dt_sim
        if outer_time >= Ts_outer:
            T_cmd, phi_d, theta_d, psi_d = outer_ctrl.compute(
                xd, yd, zd, vxd_ff, vyd_ff, psi_d, est, Ts_outer)
            quad_true.T = T_cmd
            outer_time = 0.0

        middle_time += dt_sim
        if middle_time >= Ts_middle:
            p_d, q_d, r_d = mid_ctrl.compute(phi_d, theta_d, psi_d, est, Ts_middle)
            middle_time = 0.0

        # Rate loop + allocator stay pinned to Ts_inner (=dt): commanded rotor
        # speeds are held fixed across any finer IMU sub-steps within a frame.
        inner_time += dt_sim
        if inner_time >= Ts_inner:
            quad_true.tau_x, quad_true.tau_y, quad_true.tau_z = inner_ctrl.compute(
                p_d, q_d, r_d, est, Ts_inner)
            rotor_speeds = np.array(allocator.allocate(
                quad_true.T, quad_true.tau_x, quad_true.tau_y, quad_true.tau_z))
            inner_time = 0.0

        # 5. propagate the quad_true one sim step
        sol = solve_ivp(dynamics, [t, t + dt_sim], quad_true.X,
                        args=(quad_true, rotor_speeds), method='RK45')
        quad_true.X = sol.y[:, -1]

    if verbose:
        print('EKF closed-loop complete.')
    diag = {'t': t_hist, 'nees': nees,
            'innov_gps_t': np.array(innov_gps_t), 'innov_gps': np.array(innov_gps)}
    return quad_true, est, diag


# ============================================================================
#  GPS DROPOUT EXPERIMENT
# ============================================================================

def run_gps_dropout_sweep(
        config, ekf_cfg=None, imu_rates_hz=(20, 50, 100, 200, 400),
        dropout_window=None, n_trials=5, seeds=None,
        jacobian_stride=4, nees_diverged=50.0,
    ):
    """
    Disable GPS for a window mid-flight and measure how position error grows
    during that window, at several IMU rates.

    With GPS gone, the estimate has to dead-reckon on IMU (gyro + accel) and
    magnetometer alone until GPS returns — the strapdown-style regime
    described in Section 5.3, where IMU rate should actually matter, unlike
    the always-GPS-aided case (Section 8's original design) where GPS's own
    10 Hz correction dominates regardless of IMU rate.

    Each rate is run ``n_trials`` times (different seeds) since dropout drift
    is inherently noisy — it depends on the specific noise realization active
    right when GPS drops out, not just IMU rate. Rather than compare a single
    noisy endpoint, this averages the *entire* position-error-vs-time curve
    during the dropout across trials, which is far more stable.

    Parameters
    ----------
    config : dict
        Same shape as :func:`run_fig8_ekf`'s config.
    ekf_cfg : dict, optional
        As in :func:`run_fig8_ekf`.
    imu_rates_hz : sequence of float
        IMU rates to test.
    dropout_window : (float, float)
        ``(t_start, t_end)`` GPS-off window [s].
        Must fall within ``config['sim']['tf']``, comfortably inside the cruise phase (after ``t_takeoff``, before ``tf - t_land``).
        Keep this short, around 1.5 s. This EKF is a first-order (not iterated/unscented) filter:
        its correction is a linearization around the *current* estimate, and
        once real drift accumulates during a long GPS-denied stretch, each
        further correction is computed from an increasingly wrong
        linearization point -- a known EKF weakness, not specific to this
        codebase.
        Verified empirically: a ~1.5 s window shows a real, modest
        "faster IMU drifts less" effect; push much past ~2 s and the trend
        can invert as linearization breaks down, which demonstrates a
        different (and separate) point than intended here.
    n_trials : int
        Independent noise realizations averaged per rate.
    seeds : sequence of int, optional
        Explicit seeds, length >= n_trials. Defaults to ``1..n_trials``.
    jacobian_stride : int
        Forwarded to :func:`run_fig8_ekf` (default 4 here, not 1 — this
        experiment doesn't need every-step Jacobian freshness, and the
        speedup is verified negligible in accuracy; see the profiling in the
        module notes).
    nees_diverged : float
        Mean-NEES threshold (evaluated over the dropout window) above which a
        trial is excluded as diverged.

    Returns
    -------
    dict {'rate_hz', 'dropout_t' (seconds since dropout start),
          'drift_curve_mean': {rate: array}, 'drift_curve_std': {rate: array},
          'rms_during_mean': array, 'rms_during_std': array, 'n_ok': array}
    """
    if ekf_cfg is None:
        ekf_cfg = default_ekf_config()
    if dropout_window is None:
        tf = config['sim']['tf']
        dropout_window = (0.4 * tf, min(0.4 * tf + 1.5, 0.9 * tf))
    if seeds is None:
        seeds = list(range(1, n_trials + 1))
    t_start, t_end = dropout_window

    rate_list, rms_mean, rms_std, n_ok = [], [], [], []
    curve_mean, curve_std, curve_t = {}, {}, {}

    for rate in imu_rates_hz:
        trial_curves, trial_rms = [], []
        dropout_t_grid = np.array([])
        for seed in seeds[:n_trials]:
            cfg_trial = dict(ekf_cfg)
            cfg_trial['seed'] = seed
            truth, est, diag = run_fig8_ekf(config, cfg_trial, imu_rate_hz=rate,
                                             jacobian_stride=jacobian_stride,
                                             gps_dropout=dropout_window, verbose=False)
            t = diag['t']
            mask = (t >= t_start) & (t <= t_end)
            if not mask.any():
                continue
            xt = np.asarray(truth.data('x')[1])[mask]
            yt = np.asarray(truth.data('y')[1])[mask]
            zt = np.asarray(truth.data('z')[1])[mask]
            xe = np.asarray(est.data('x')[1])[mask]
            ye = np.asarray(est.data('y')[1])[mask]
            ze = np.asarray(est.data('z')[1])[mask]
            pos_err = np.sqrt((xt - xe)**2 + (yt - ye)**2 + (zt - ze)**2)
            nees_during = diag['nees'][mask]
            nees_mean = float(np.nanmean(nees_during))
            if (not np.isfinite(nees_mean) or nees_mean > nees_diverged
                    or not np.all(np.isfinite(pos_err))):
                print(f'  [rate={rate:g} Hz, seed={seed}] excluded '
                      f'(mean NEES during dropout = {nees_mean:.1f} > {nees_diverged:g}, '
                      f'or non-finite error, likely diverged)')
                continue
            if dropout_t_grid.size == 0:
                dropout_t_grid = t[mask] - t_start
            trial_curves.append(pos_err)
            trial_rms.append(float(np.sqrt(np.mean(pos_err**2))))

        n_ok.append(len(trial_rms))
        rate_list.append(rate)
        if trial_rms:
            # Trials at the same rate share dt_sim, so curves are equal-length.
            min_len = min(len(c) for c in trial_curves)
            stacked = np.array([c[:min_len] for c in trial_curves])
            curve_mean[rate] = stacked.mean(axis=0)
            curve_std[rate] = stacked.std(axis=0)
            curve_t[rate] = dropout_t_grid[:min_len]
            rms_mean.append(float(np.mean(trial_rms)))
            rms_std.append(float(np.std(trial_rms)))
        else:
            curve_mean[rate] = np.array([])
            curve_std[rate] = np.array([])
            curve_t[rate] = np.array([])
            rms_mean.append(np.nan)
            rms_std.append(np.nan)
        print(f'IMU rate = {rate:6g} Hz  ->  RMS position error during '
              f'{t_end - t_start:g}s GPS dropout = {rms_mean[-1]:.4f} m  '
              f'(n={n_ok[-1]}/{n_trials})')

    return {'rate_hz': np.array(rate_list, dtype=float),
            'dropout_window': dropout_window,
            'drift_curve_t': curve_t,
            'drift_curve_mean': curve_mean,
            'drift_curve_std': curve_std,
            'rms_during_mean': np.array(rms_mean),
            'rms_during_std': np.array(rms_std),
            'n_ok': np.array(n_ok)}


def plot_gps_dropout(sweep, show=True):
    """
    Two panels: (left) position error vs. time-since-dropout-start, one curve
    per IMU rate, shaded by trial std; (right) RMS position error during the
    dropout vs.
    """
    import matplotlib.pyplot as plt

    rates = sweep['rate_hz']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(rates)))
    for rate, color in zip(rates, cmap):
        t = sweep['drift_curve_t'][rate]
        m = sweep['drift_curve_mean'][rate]
        s = sweep['drift_curve_std'][rate]
        if len(t) == 0:
            continue
        ax1.plot(t, m, lw=2, color=color, label=f'{rate:g} Hz')
        ax1.fill_between(t, m - s, m + s, color=color, alpha=0.15)
    ax1.set_xlabel('Time since GPS dropout start [s]')
    ax1.set_ylabel('Position error [m]')
    ax1.set_title('Drift during GPS-denied dead-reckoning')
    ax1.legend(title='IMU rate', fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(rates, sweep['rms_during_mean'], yerr=sweep['rms_during_std'],
                 marker='o', ms=6, lw=2, capsize=4, color='tab:red')
    # ax2.set_xscale('log')
    ax2.set_xlabel('IMU update rate [Hz]')
    ax2.set_ylabel('RMS position error during dropout [m]')
    ax2.set_title('Summary: dropout-window RMS vs. IMU rate')
    ax2.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


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


def plot_estimation(truth, est, diag, states=('x', 'z', 'phi', 'psi'), t0=0.0, t1=100.0):
    """True-vs-estimated trajectories with +-2 sigma covariance bands + NEES.

    ``t0``/``t1`` window the time axis (seconds) shown in every panel,
    matching :func:`compute_metrics`'s default cruise-phase window.
    """
    import matplotlib.pyplot as plt

    idx = {n: i for i, n in enumerate(STATE_NAMES)}
    nrows = len(states) + 1
    fig, axes = plt.subplots(nrows, 1, figsize=(9, 2.4 * nrows))

    for ax, n in zip(axes[:-1], states):
        t, xt = truth.data(n)
        _, xe = est.data(n)
        sig = np.sqrt(np.asarray(est.data(f'P{idx[n]}{idx[n]}')[1]))
        t = np.asarray(t)
        mask = (t >= t0) & (t <= t1)
        t = t[mask]
        xt = np.asarray(xt)[mask]; xe = np.asarray(xe)[mask]; sig = sig[mask]
        scale = c4d.r2d if n in ('phi', 'theta', 'psi', 'p', 'q', 'r') else 1.0
        xt = xt * scale; xe = xe * scale; sig = sig * scale
        ax.plot(t, xt, 'k-', lw=1.5, label='true')
        ax.plot(t, xe, 'C1--', lw=1.2, label='estimated')
        ax.fill_between(t, xe - 2*sig, xe + 2*sig, color='C1', alpha=0.2,
                        label=r'$\pm 2\sigma$')
        unit = '[deg]' if scale != 1.0 else '[m]'
        c4d.plotdefaults(ax, f'{n}', 'time [s]', f'{n} {unit}', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)

    ax = axes[-1]
    t_nees = np.asarray(diag['t'])
    mask = (t_nees >= t0) & (t_nees <= t1)
    ax.plot(t_nees[mask], np.asarray(diag['nees'])[mask], 'C0-', lw=0.8)
    ax.axhline(12, color='k', ls=':', lw=1, label='ideal (n=12)')
    c4d.plotdefaults(ax, 'Filter consistency (NEES)', 'time [s]', 'NEES', fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 40)

    fig.tight_layout()
    return fig

def plot_trajectory(truth, est, config, t0=0.0, t1=100.0):
    """Figure-8 path: reference vs. true vs. estimated (top-down x-y view).

    ``t0``/``t1`` window the flown/reference paths (seconds) to this range,
    e.g. to exclude takeoff/landing.
    """
    import matplotlib.pyplot as plt
    from c4dynamics.controllers.quad_pid import position_reference

    # flown paths from the stored histories
    t,  x_true = truth.data('x');  _, y_true = truth.data('y')
    _,  x_est  = est.data('x');    _, y_est  = est.data('y')
    t = np.asarray(t)
    mask = (t >= t0) & (t <= t1)
    t = t[mask]
    x_true = np.asarray(x_true)[mask]; y_true = np.asarray(y_true)[mask]
    x_est  = np.asarray(x_est)[mask];  y_est  = np.asarray(y_est)[mask]

    # rebuild the reference figure-8 over the same (windowed) time vector
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


def plot_dashboard(truth, est, config, t0=8.0, t1=82.0):
    """
    Same 2x3 results dashboard as the plain cascade-PID example
    (:func:`c4dynamics.controllers.quad_pid.plot_results`), with the EKF
    estimate overlaid wherever truth is plotted: reference vs. true vs.
    estimated, instead of just reference vs. true.

    Panels: 3D trajectory | XY plane | horizontal position tracking
            Altitude (zoomed +-2 m around z_ref) | tracking/estimation error
            | attitude angles

    ``t0``/``t1`` window the time axis (seconds), matching
    :func:`compute_metrics`'s default cruise-phase window.
    """
    import matplotlib.pyplot as plt
    from c4dynamics.controllers.quad_pid import position_reference

    A, B, omega, z_ref = (config['trajectory']['A'], config['trajectory']['B'],
                          config['trajectory']['omega'], config['trajectory']['z_ref'])
    t_sim = config['sim']['tf']

    t, x_true = truth.data('x');  _, y_true = truth.data('y');  _, z_true = truth.data('z')
    _, x_est  = est.data('x');    _, y_est  = est.data('y');    _, z_est  = est.data('z')
    _, phi_true   = truth.data('phi',   scale=c4d.r2d)
    _, theta_true = truth.data('theta', scale=c4d.r2d)
    _, psi_true   = truth.data('psi',   scale=c4d.r2d)
    _, phi_est    = est.data('phi',   scale=c4d.r2d)
    _, theta_est  = est.data('theta', scale=c4d.r2d)
    _, psi_est    = est.data('psi',   scale=c4d.r2d)

    t = np.asarray(t)
    mask = (t >= t0) & (t <= t1)
    t = t[mask]
    x_true, y_true, z_true = (np.asarray(a)[mask] for a in (x_true, y_true, z_true))
    x_est,  y_est,  z_est  = (np.asarray(a)[mask] for a in (x_est, y_est, z_est))
    phi_true, theta_true, psi_true = (np.asarray(a)[mask] for a in (phi_true, theta_true, psi_true))
    phi_est,  theta_est,  psi_est  = (np.asarray(a)[mask] for a in (phi_est, theta_est, psi_est))

    ref = np.array([position_reference(ti, A, B, omega, z_ref, t_sim=t_sim) for ti in t])
    x_ref, y_ref, z_ref_hist = ref[:, 0], ref[:, 1], ref[:, 2]

    # tracking error (true vs. reference) and estimation error (true vs. estimate)
    pos_err = np.sqrt((x_true - x_ref)**2 + (y_true - y_ref)**2 + (z_true - z_ref_hist)**2)
    est_err = np.sqrt((x_true - x_est)**2 + (y_true - y_est)**2 + (z_true - z_est)**2)

    lw = 1.5
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('EKF Closed-Loop Quadcopter — Simulation Results', fontsize=16, fontweight='bold')

    # -- 3D Trajectory --
    ax3d = fig.add_subplot(2, 3, 1, projection='3d')
    ax3d.plot(x_true, y_true, z_true, 'b-', linewidth=lw, label='Actual')
    ax3d.plot(x_est, y_est, z_est, 'C1:', linewidth=lw + 0.3, label='Estimated')
    ax3d.plot(x_ref, y_ref, z_ref_hist, 'r--', linewidth=lw, label='Reference')
    ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
    ax3d.set_title('3D Trajectory')
    ax3d.legend(fontsize=8); ax3d.grid(True)

    # -- XY Plane --
    ax = fig.add_subplot(2, 3, 2)
    ax.plot(x_true, y_true, 'b-', linewidth=lw, label='Actual')
    ax.plot(x_est, y_est, 'C1:', linewidth=lw, label='Estimated')
    ax.plot(x_ref, y_ref, 'r--', linewidth=lw, label='Reference')
    c4d.plotdefaults(ax, 'XY Plane', xlabel='X (m)', ylabel='Y (m)')
    ax.legend(fontsize=8); ax.grid(True); ax.axis('equal')

    # -- Horizontal position tracking --
    ax = fig.add_subplot(2, 3, 3)
    ax.plot(t, x_true, 'b-', linewidth=lw, label='X actual')
    ax.plot(t, x_est, 'C1:', linewidth=lw, label='X estimated')
    ax.plot(t, x_ref, 'r--', linewidth=lw, label='X ref')
    ax.plot(t, y_true, 'g-', linewidth=lw, label='Y actual')
    ax.plot(t, y_est, 'C2:', linewidth=lw, label='Y estimated')
    ax.plot(t, y_ref, 'm--', linewidth=lw, label='Y ref')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Position (m)')
    ax.set_title('Horizontal Position Tracking')
    ax.legend(fontsize=6); ax.grid(True)

    # -- Altitude tracking (zoomed +-2 m around the z_ref plateau) --
    ax = fig.add_subplot(2, 3, 4)
    ax.plot(t, z_true, 'b-', linewidth=lw, label='Z actual')
    ax.plot(t, z_est, 'C1:', linewidth=lw, label='Z estimated')
    ax.plot(t, z_ref_hist, 'r--', linewidth=lw, label='Z ref')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Altitude (m)')
    ax.set_title('Altitude Tracking')
    # ax.set_ylim(z_ref - 2, z_ref + 2)
    ax.legend(fontsize=8); ax.grid(True)

    # -- Position tracking / estimation error --
    ax = fig.add_subplot(2, 3, 5)
    ax.plot(t, pos_err, 'r-', linewidth=lw, label='Tracking error (true - ref)')
    ax.plot(t, est_err, 'C1-', linewidth=lw, label='Estimation error (true - est)')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Error (m)')
    ax.set_title('Position Tracking / Estimation Error')
    ax.legend(fontsize=8); ax.grid(True)

    # -- Attitude angles --
    ax = fig.add_subplot(2, 3, 6)
    ax.plot(t, phi_true, 'b-', linewidth=lw, label='Roll (Phi)')
    ax.plot(t, phi_est, 'b:', linewidth=lw, label='Roll est.')
    ax.plot(t, theta_true, 'g-', linewidth=lw, label='Pitch (Theta)')
    ax.plot(t, theta_est, 'g:', linewidth=lw, label='Pitch est.')
    ax.plot(t, psi_true, 'r-', linewidth=lw, label='Yaw (Psi)')
    ax.plot(t, psi_est, 'r:', linewidth=lw, label='Yaw est.')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Angle (deg)')
    ax.set_title('Attitude Angles')
    ax.legend(fontsize=6); ax.grid(True)

    plt.tight_layout()
    return fig



