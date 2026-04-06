"""
Supporting module for the quadcopter cascade-PID notebook.

The notebook exposes the user-facing inputs. This file keeps the simulation,
control, plotting, and metric implementation details out of the notebook so the
workflow stays focused on: edit inputs -> run -> inspect results.
"""

import numpy as np
import matplotlib.pyplot as plt
import c4dynamics as c4d
from scipy.integrate import solve_ivp


# ============================================================
#  PID CONTROLLER CLASS
# ============================================================

class PID:
    """
    PID controller with:
      - Derivative on measurement  (prevents derivative kick)
      - First-order derivative filter  (reduces noise)
      - Integrator anti-windup clamping
    """

    def __init__(self, Kp, Ki, Kd, dt,
                 N=20,
                 output_limit=None,
                 integral_limit=None):
        """
        Parameters
        ----------
        Kp, Ki, Kd     : PID gains
        dt             : sample time for this loop [s]
        N              : derivative filter coefficient
                         N=50  inner rate loop
                         N=20  middle attitude loop
                         N=10  outer position loop
        output_limit   : symmetric clamp on output
        integral_limit : symmetric clamp on integrator (anti-windup)
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.N  = N
        self.output_limit   = output_limit
        self.integral_limit = integral_limit

        self._integral   = 0.0
        self._prev_meas  = None
        self._derivative = 0.0

    def reset(self):
        """Reset all internal states to zero."""
        self._integral   = 0.0
        self._prev_meas  = None
        self._derivative = 0.0

    def update(self, error, measurement=None):
        """
        Compute PID output.

        Parameters
        ----------
        error       : reference - measurement
        measurement : raw measurement value.
                      If provided, derivative is computed on
                      measurement (derivative on measurement).
                      If None, derivative is computed on error.

        Returns
        -------
        output : control command (clamped if output_limit set)
        """

        # Proportional
        P = self.Kp * error

        # Integral with anti-windup clamping
        self._integral += error * self.dt
        if self.integral_limit is not None:
            self._integral = np.clip(
                self._integral,
                -self.integral_limit,
                 self.integral_limit
            )
        I = self.Ki * self._integral

        # Derivative on measurement if provided, else on error
        if measurement is not None:
            if self._prev_meas is None:
                self._prev_meas = measurement
            raw_deriv = -(measurement - self._prev_meas) / self.dt
            self._prev_meas = measurement
        else:
            if self._prev_meas is None:
                self._prev_meas = error
            raw_deriv = (error - self._prev_meas) / self.dt
            self._prev_meas = error

        # First-order low-pass filter on derivative
        # alpha = N*dt / (1 + N*dt)
        alpha = self.N * self.dt / (1.0 + self.N * self.dt)
        self._derivative = (
            (1.0 - alpha) * self._derivative + alpha * raw_deriv
        )
        D = self.Kd * self._derivative

        # Total output with optional saturation
        output = P + I + D
        if self.output_limit is not None:
            output = np.clip(output, -self.output_limit, self.output_limit)

        return output


# ============================================================
#  QUADCOPTER NONLINEAR DYNAMICS
# ============================================================

def quad_dynamics(t, X, F, tau_phi, tau_theta, tau_psi,
                  m, g, Ixx, Iyy, Izz):
    """
    Full nonlinear Newton-Euler equations of motion.
    Called by solve_ivp at every integration step.

    State vector X = [x, y, z, vx, vy, vz,
                      phi, theta, psi, p, q, r]

    Parameters
    ----------
    t                         : current time [s]
    X                         : state vector [12]
    F                         : total thrust [N]
    tau_phi, tau_theta, tau_psi : body torques [N.m]
    m, g, Ixx, Iyy, Izz       : vehicle parameters

    Returns
    -------
    X_dot : state derivative vector [12]
    """

    x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = X

    # Trigonometric shorthands
    cphi   = np.cos(phi);    sphi   = np.sin(phi)
    ctheta = np.cos(theta);  stheta = np.sin(theta)
    cpsi   = np.cos(psi);    spsi   = np.sin(psi)
    ttheta = np.tan(theta)

    # Translational kinematics
    x_dot = vx
    y_dot = vy
    z_dot = vz

    # Translational dynamics (Newton - inertial frame)
    # Thrust projected via ZYX rotation matrix
    vx_dot = (F / m) * (cpsi * stheta * cphi + spsi * sphi)
    vy_dot = (F / m) * (spsi * stheta * cphi - cpsi * sphi)
    vz_dot = (F / m) * (ctheta * cphi) - g

    # Rotational kinematics - body rates to Euler angle rates
    phi_dot   = p + (q * sphi + r * cphi) * ttheta
    theta_dot = q * cphi - r * sphi
    psi_dot   = (q * sphi + r * cphi) / (ctheta + 1e-6)

    # Rotational dynamics - Euler equations
    p_dot = ((Iyy - Izz) / Ixx) * q * r + tau_phi   / Ixx
    q_dot = ((Izz - Ixx) / Iyy) * p * r + tau_theta / Iyy
    r_dot = ((Ixx - Iyy) / Izz) * p * q + tau_psi   / Izz

    return [x_dot, y_dot, z_dot,
            vx_dot, vy_dot, vz_dot,
            phi_dot, theta_dot, psi_dot,
            p_dot, q_dot, r_dot]


# ============================================================
#  REFERENCE TRAJECTORY
# ============================================================

def get_reference(t, A, B, omega, z_ref, t_start=5.0, t_ramp=12.0):
    """
    Figure-8 reference position with cosine ramp.

    A cosine ramp is applied for t_ramp seconds after takeoff
    to avoid velocity discontinuity at trajectory start.

    Parameters
    ----------
    t       : current time [s]
    A       : figure-8 X amplitude [m]
    B       : figure-8 Y amplitude [m]
    omega   : figure-8 angular frequency [rad/s]
    z_ref   : constant reference altitude [m]
    t_start : time to start figure-8 after simulation begins [s]
    t_ramp  : ramp duration [s]

    Returns
    -------
    x_ref, y_ref, z_ref : reference position [m]
    """
    if t < t_start:
        return 0.0, 0.0, z_ref

    t_fig = t - t_start

    if t_fig < t_ramp:
        ramp = 0.5 * (1.0 - np.cos(np.pi * t_fig / t_ramp))
    else:
        ramp = 1.0

    x_ref = ramp * A * np.sin(omega * t_fig)
    y_ref = ramp * B * np.sin(2.0 * omega * t_fig)

    return x_ref, y_ref, z_ref


def get_reference_velocity(t, A, B, omega, t_start=5.0, t_ramp=12.0):
    """
    Analytical time derivative of get_reference.
    Used for velocity feedforward in the position loop.

    Returns
    -------
    vx_ref, vy_ref : reference velocity [m/s]
    """
    if t < t_start:
        return 0.0, 0.0

    t_fig = t - t_start

    if t_fig < t_ramp:
        ramp     = 0.5 * (1.0 - np.cos(np.pi * t_fig / t_ramp))
        ramp_dot = 0.5 * (np.pi / t_ramp) * np.sin(np.pi * t_fig / t_ramp)
    else:
        ramp     = 1.0
        ramp_dot = 0.0

    vx_ref = (ramp_dot * A * np.sin(omega * t_fig)
              + ramp * A * omega * np.cos(omega * t_fig))

    vy_ref = (ramp_dot * B * np.sin(2.0 * omega * t_fig)
              + ramp * B * 2.0 * omega * np.cos(2.0 * omega * t_fig))

    return vx_ref, vy_ref


def get_steady_start(trajectory):
    """
    Return the start time of the steady-state tracking window.

    By default, steady-state begins after the initial hover delay plus the
    figure-8 ramp.
    """
    return trajectory.get('steady_start',
                          trajectory.get('t_start', 5.0)
                          + trajectory.get('t_ramp', 12.0))


# ============================================================
#  BUILD CONTROL ALLOCATION MATRIX
# ============================================================

def build_allocator(kT, kQ, l):
    """
    Build control allocation matrix and its inverse.

    Maps [F, tau_phi, tau_theta, tau_psi] -> [Omega1², Omega2², Omega3², Omega4²]

    Motor layout — plus (+) configuration:
      Motor 1: front (+x)  CW
      Motor 2: left  (+y)  CCW
      Motor 3: rear  (-x)  CW
      Motor 4: right (-y)  CCW

    Parameters
    ----------
    kT : thrust coefficient
    kQ : torque coefficient
    l  : arm length [m]

    Returns
    -------
    Gamma     : allocation matrix (4x4)
    Gamma_inv : inverse of allocation matrix
    """
    Gamma = np.array([
        [ kT,      kT,      kT,      kT     ],
        [ 0,      -kT * l,  0,       kT * l ],
        [ kT * l,  0,      -kT * l,  0      ],
        [ kQ,     -kQ,      kQ,     -kQ     ]
    ])
    Gamma_inv = np.linalg.inv(Gamma)
    return Gamma, Gamma_inv


# ============================================================
#  BUILD PID CONTROLLERS FROM CONFIG DICT
# ============================================================

def build_controllers(controller, dt_inner, dt_mid, dt_outer):
    """
    Instantiate all nine PID controllers from the
    controller configuration dictionary.

    Parameters
    ----------
    controller : dict of PID gains (from notebook user inputs)
    dt_inner, dt_mid, dt_outer : sample times [s]

    Returns
    -------
    Dictionary of PID instances keyed by axis name.
    """
    pids = {}

    # Inner loop — angular rate (200 Hz)
    pids['p'] = PID(
        Kp=controller['Kp_p'], Ki=controller['Ki_p'], Kd=controller['Kd_p'],
        dt=dt_inner, N=50, output_limit=50.0, integral_limit=10.0
    )
    pids['q'] = PID(
        Kp=controller['Kp_q'], Ki=controller['Ki_q'], Kd=controller['Kd_q'],
        dt=dt_inner, N=50, output_limit=50.0, integral_limit=10.0
    )
    pids['r'] = PID(
        Kp=controller['Kp_r'], Ki=controller['Ki_r'], Kd=controller['Kd_r'],
        dt=dt_inner, N=50, output_limit=20.0, integral_limit=5.0
    )

    # Middle loop — attitude (100 Hz)
    pids['phi'] = PID(
        Kp=controller['Kp_phi'], Ki=controller['Ki_phi'], Kd=controller['Kd_phi'],
        dt=dt_mid, N=20, output_limit=1.5, integral_limit=0.5
    )
    pids['theta'] = PID(
        Kp=controller['Kp_theta'], Ki=controller['Ki_theta'], Kd=controller['Kd_theta'],
        dt=dt_mid, N=20, output_limit=1.5, integral_limit=0.5
    )
    pids['psi'] = PID(
        Kp=controller['Kp_psi'], Ki=controller['Ki_psi'], Kd=controller['Kd_psi'],
        dt=dt_mid, N=20, output_limit=1.0, integral_limit=0.3
    )

    # Outer loop — position (50 Hz)
    pids['x'] = PID(
        Kp=controller['Kp_x'], Ki=controller['Ki_x'], Kd=controller['Kd_x'],
        dt=dt_outer, N=10, output_limit=0.35, integral_limit=0.3
    )
    pids['y'] = PID(
        Kp=controller['Kp_y'], Ki=controller['Ki_y'], Kd=controller['Kd_y'],
        dt=dt_outer, N=10, output_limit=0.35, integral_limit=0.3
    )
    pids['z'] = PID(
        Kp=controller['Kp_z'], Ki=controller['Ki_z'], Kd=controller['Kd_z'],
        dt=dt_outer, N=10, output_limit=10.0, integral_limit=3.0
    )

    return pids


# ============================================================
#  MAIN SIMULATION FUNCTION
# ============================================================

def run_fig8_pid(vehicle, trajectory, controller, sim):
    """
    Run the cascade PID quadcopter simulation.

    Parameters
    ----------
    vehicle : dict
        'm'   : mass [kg]
        'g'   : gravity [m/s²]
        'Ixx' : roll inertia [kg.m²]
        'Iyy' : pitch inertia [kg.m²]
        'Izz' : yaw inertia [kg.m²]
        'l'   : arm length [m]
        'kT'  : thrust coefficient
        'kQ'  : torque coefficient

    trajectory : dict
        'A'     : figure-8 X amplitude [m]
        'B'     : figure-8 Y amplitude [m]
        'omega' : angular frequency [rad/s]
        'z_ref' : hover altitude [m]
        't_end' : simulation duration [s]

    controller : dict
        PID gains for all nine controllers.
        Keys: Kp_p, Ki_p, Kd_p,
              Kp_q, Ki_q, Kd_q,
              Kp_r, Ki_r, Kd_r,
              Kp_phi, Ki_phi, Kd_phi,
              Kp_theta, Ki_theta, Kd_theta,
              Kp_psi, Ki_psi, Kd_psi,
              Kp_x, Ki_x, Kd_x,
              Kp_y, Ki_y, Kd_y,
              Kp_z, Ki_z, Kd_z,
              Kff_x, Kff_y

    sim : dict
        'dt'    : master timestep [s]
        't_end' : simulation end time [s]

    Returns
    -------
    results : dict with all simulation data for plotting and metrics
    """

    # ── Unpack vehicle parameters ──
    m   = vehicle['m']
    g   = vehicle.get('g', 9.81)
    Ixx = vehicle['Ixx']
    Iyy = vehicle['Iyy']
    Izz = vehicle['Izz']
    l   = vehicle['l']
    kT  = vehicle.get('kT', 1.0)
    kQ  = vehicle.get('kQ', 0.01)

    # ── Unpack trajectory parameters ──
    A     = trajectory['A']
    B     = trajectory['B']
    omega = trajectory['omega']
    z_ref = trajectory['z_ref']
    t_start = trajectory.get('t_start', 5.0)
    t_ramp = trajectory.get('t_ramp', 12.0)
    t_end = trajectory['t_end']

    # ── Unpack simulation settings ──
    dt    = sim['dt']

    # ── Sample rates ──
    dt_inner = dt          # 200 Hz — inner rate loop
    dt_mid   = dt * 2      # 100 Hz — middle attitude loop
    dt_outer = dt * 4      #  50 Hz — outer position loop

    mid_every   = 2
    outer_every = 4

    # ── Feedforward gains ──
    Kff_x = controller.get('Kff_x', 0.2479)
    Kff_y = controller.get('Kff_y', 0.35)

    # ── Build control allocator ──
    _, Gamma_inv = build_allocator(kT, kQ, l)

    # ── Build PID controllers ──
    pids = build_controllers(controller, dt_inner, dt_mid, dt_outer)

    # ── Initialize c4dynamics rigidbody ──
    quad       = c4d.rigidbody(z=z_ref)
    quad.mass  = m
    quad.I     = [Ixx, Iyy, Izz]
    quad.F         = m * g
    quad.tau_phi   = 0.0
    quad.tau_theta = 0.0
    quad.tau_psi   = 0.0

    # ── Desired values persistent across loop updates ──
    theta_des = 0.0
    phi_des   = 0.0
    p_des     = 0.0
    q_des     = 0.0
    r_des     = 0.0

    # ── Main simulation loop ──
    step = 0

    for ti in np.arange(0, t_end, dt):

        # Store state and control inputs at this timestep
        quad.store(ti)
        quad.storeparams(['F', 'tau_phi', 'tau_theta', 'tau_psi'], t=ti)

        # ── Outer loop — Position (50 Hz) ──
        if step % outer_every == 0:
            x_ref, y_ref, z_ref_t = get_reference(
                ti, A, B, omega, z_ref, t_start=t_start, t_ramp=t_ramp
            )
            vx_ref, vy_ref = get_reference_velocity(
                ti, A, B, omega, t_start=t_start, t_ramp=t_ramp
            )

            # Position PID + velocity feedforward
            theta_des = (pids['x'].update(x_ref - quad.x)
                         + Kff_x * vx_ref)
            phi_des   = -(pids['y'].update(y_ref - quad.y)
                         + Kff_y * vy_ref)

            # Altitude PID with hover feedforward
            delta_F = pids['z'].update(z_ref_t - quad.z, quad.vz)
            quad.F  = m * g + delta_F

        # ── Middle loop — Attitude (100 Hz) ──
        if step % mid_every == 0:
            p_des = pids['phi'].update(phi_des     - quad.phi)
            q_des = pids['theta'].update(theta_des - quad.theta)
            r_des = pids['psi'].update(0.0         - quad.psi)

        # ── Inner loop — Rate (200 Hz) ──
        quad.tau_phi   = pids['p'].update(p_des - quad.p, quad.p)
        quad.tau_theta = pids['q'].update(q_des - quad.q, quad.q)
        quad.tau_psi   = pids['r'].update(r_des - quad.r, quad.r)

        # ── Control allocator — torques to rotor speeds ──
        U         = np.array([quad.F,
                               quad.tau_phi,
                               quad.tau_theta,
                               quad.tau_psi])
        omega_sq  = np.clip(Gamma_inv @ U, 0, None)
        _         = np.sqrt(omega_sq)   # rotor speeds (available if needed)

        # ── Integrate nonlinear dynamics (RK45) ──
        sol    = solve_ivp(
            quad_dynamics,
            [ti, ti + dt],
            quad.X,
            args=(quad.F,
                  quad.tau_phi,
                  quad.tau_theta,
                  quad.tau_psi,
                  m, g, Ixx, Iyy, Izz),
            method='RK45'
        )
        quad.X = sol.y[:, -1]

        step += 1

    # ── Pack and return results ──
    results = {
        'quad'      : quad,
        'trajectory': trajectory,
        'vehicle'   : vehicle,
        't_end'     : t_end,
        'dt'        : dt,
    }
    return results


# ============================================================
#  PLOTTING HELPERS
# ============================================================

def plot_results(results):
    """
    Generate all result plots from simulation output.

    Produces:
      1. Position vs time
      2. Euler angles vs time
      3. Control inputs vs time
      4. 3D trajectory (steady state)

    Parameters
    ----------
    results : dict returned by run_fig8_pid()
    """
    quad       = results['quad']
    trajectory = results['trajectory']
    t_end      = results['t_end']
    dt         = results['dt']

    A     = trajectory['A']
    B     = trajectory['B']
    omega = trajectory['omega']
    z_ref = trajectory['z_ref']

    # ── Figure 1: Position, Euler Angles, Control Inputs ──
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.4)

    # Position
    axs[0].plot(*quad.data('x'), label='x')
    axs[0].plot(*quad.data('y'), label='y')
    axs[0].plot(*quad.data('z'), label='z')
    axs[0].axhline(y=z_ref, color='gray', linestyle='--',
                   linewidth=0.8, label='z_ref')
    axs[0].legend()
    c4d.plotdefaults(axs[0], 'Position', 'time [s]', 'm', fontsize=13)

    # Euler angles
    axs[1].plot(*quad.data('phi',   scale=c4d.r2d), label='roll φ')
    axs[1].plot(*quad.data('theta', scale=c4d.r2d), label='pitch θ')
    axs[1].plot(*quad.data('psi',   scale=c4d.r2d), label='yaw ψ')
    axs[1].legend()
    c4d.plotdefaults(axs[1], 'Euler Angles', 'time [s]', 'deg', fontsize=13)

    # Control inputs
    axs[2].plot(*quad.data('F'),         label='F [N]')
    axs[2].plot(*quad.data('tau_phi'),   label='τ_φ')
    axs[2].plot(*quad.data('tau_theta'), label='τ_θ')
    axs[2].plot(*quad.data('tau_psi'),   label='τ_ψ')
    axs[2].legend()
    c4d.plotdefaults(axs[2], 'Control Inputs',
                     'time [s]', 'N / N.m', fontsize=13)

    plt.tight_layout()
    plt.show()

    # ── Figure 2: 3D Trajectory (steady state only) ──
    fig3d = plt.figure(figsize=(8, 6))
    ax3d  = fig3d.add_subplot(111, projection='3d')

    t_hist = quad.data('x')[0]
    x_hist = quad.data('x')[1]
    y_hist = quad.data('y')[1]
    z_hist = quad.data('z')[1]

    # Show only steady-state tracking
    steady_idx = t_hist >= t_steady

    ax3d.plot(x_hist[steady_idx],
              y_hist[steady_idx],
              z_hist[steady_idx],
              'b', linewidth=1.5, label='Actual')

    t_vec = np.arange(t_steady, t_end, dt)
    x_r   = [get_reference(t, A, B, omega, z_ref, t_start, t_ramp)[0] for t in t_vec]
    y_r   = [get_reference(t, A, B, omega, z_ref, t_start, t_ramp)[1] for t in t_vec]
    z_r   = np.full(len(t_vec), z_ref)

    ax3d.plot(x_r, y_r, z_r, 'r--', linewidth=1.5, label='Reference')

    ax3d.set_xlabel('x [m]')
    ax3d.set_ylabel('y [m]')
    ax3d.set_zlabel('z [m]')
    ax3d.set_title('Figure-8 Trajectory Tracking (Steady State)')
    ax3d.legend()
    plt.show()


# ============================================================
#  PERFORMANCE METRICS
# ============================================================

def compute_metrics(results):
    """
    Compute and print RMSE tracking performance metrics.
    Metrics are computed over steady-state portion only
    (after the cosine ramp completes at t=17s).

    Parameters
    ----------
    results : dict returned by run_fig8_pid()
    """
    quad       = results['quad']
    trajectory = results['trajectory']
    A     = trajectory['A']
    B     = trajectory['B']
    omega = trajectory['omega']
    z_ref = trajectory['z_ref']
    t_start = trajectory.get('t_start', 5.0)
    t_ramp = trajectory.get('t_ramp', 12.0)
    t_steady = get_steady_start(trajectory)
    t_start = trajectory.get('t_start', 5.0)
    t_ramp = trajectory.get('t_ramp', 12.0)

    t_hist = quad.data('x')[0]
    x_hist = quad.data('x')[1]
    y_hist = quad.data('y')[1]
    z_hist = quad.data('z')[1]

    # Steady-state indices
    t_steady = get_steady_start(trajectory)
    steady_idx = t_hist >= t_steady
    t_ss = t_hist[steady_idx]

    # Reference at steady state times
    x_ref_ss = np.array([get_reference(t, A, B, omega, z_ref, t_start, t_ramp)[0]
                          for t in t_ss])
    y_ref_ss = np.array([get_reference(t, A, B, omega, z_ref, t_start, t_ramp)[1]
                          for t in t_ss])
    z_ref_ss = np.full(len(t_ss), z_ref)

    # RMSE
    rmse_x = np.sqrt(np.mean((x_hist[steady_idx] - x_ref_ss) ** 2))
    rmse_y = np.sqrt(np.mean((y_hist[steady_idx] - y_ref_ss) ** 2))
    rmse_z = np.sqrt(np.mean((z_hist[steady_idx] - z_ref_ss) ** 2))

    # Normalized errors
    norm_x = rmse_x / A * 100 if A else np.nan
    norm_y = rmse_y / B * 100 if B else np.nan
    norm_z = rmse_z / z_ref * 100 if z_ref else np.nan

    # Max altitude deviation
    max_z_dev = np.max(np.abs(z_hist[steady_idx] - z_ref))

    print("=" * 45)
    print("   TRACKING PERFORMANCE METRICS")
    print("=" * 45)
    print(f"  RMSE x : {rmse_x:.4f} m  ({norm_x:.1f}% of X amplitude)")
    print(f"  RMSE y : {rmse_y:.4f} m  ({norm_y:.1f}% of Y amplitude)")
    print(f"  RMSE z : {rmse_z:.6f} m  ({norm_z:.3f}% of altitude)")
    print(f"  Max altitude deviation : {max_z_dev*100:.2f} cm")
    print("=" * 45)

    return {
        'rmse_x'    : rmse_x,
        'rmse_y'    : rmse_y,
        'rmse_z'    : rmse_z,
        'norm_x'    : norm_x,
        'norm_y'    : norm_y,
        'norm_z'    : norm_z,
        'max_z_dev' : max_z_dev
    }
