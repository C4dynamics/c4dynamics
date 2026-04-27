"""
quad_pid_utils.py
=================
Supporting module for the Quadcopter Cascade PID notebook.


The notebook contains the main loop, parameters, and results.
This file contains ONLY the implementation classes and helpers.

Contents:
  - dynamics()                            : state derivatives
  - position_reference()                  : 3-phase trajectory
  - velocity_reference()                  : feedforward velocity
  - OuterPositionPID                      : position loop  (50 Hz)
  - MiddleAttitudePID                     : attitude loop (100 Hz)
  - InnerRatePID                          : rate loop    (200 Hz)
  - ControlAllocator                      : torques -> rotor speeds
  - plot_results()                        : time histories + 3D
  - compute_metrics()                     : RMSE over figure-8 phase
"""

import numpy as np
import c4dynamics as c4d
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from c4dynamics.rotmat import dcm321



# ============================================================
#  DYNAMICS  (called every integration step)
# ============================================================


def dynamics(t, y, quad, rotor_speeds, ENU_body=True):
    """
    Compute the 12-state derivatives for the quadcopter.

    Accepts and returns arrays compatible with c4d.rigidbody.X:
        X = [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]



    Frame and motor convention:

    Body frame:
    x forward
    y left
    z up

    Inertial frame (ENU):
    x east
    y north
    z up

    Rotation:
    ZYX (yaw-pitch-roll)

    Motor layout (plus configuration):
    w1: front (+x)
    w2: left  (+y)
    w3: rear  (-x)
    w4: right (-y)

    Torque mapping:
    roll  (phi)   = L * (T4 - T2)
    pitch (theta) = L * (T3 - T1)
    yaw   (psi)   = kQ * (-T1 + T2 - T3 + T4)



    Parameters
    ----------
    quad         : quadcopter physical parameters
    rotor_speeds : array [w1,w2,w3,w4]  rad/s

    Returns
    -------
    dX : array (12,) — state derivatives
    """

    x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = y
    w1, w2, w3, w4 = rotor_speeds

    m   = quad.m
    g   = quad.g
    L   = quad.l
    kT  = quad.kT
    kQ  = quad.kQ
    IR  = quad.IR
    Ixx = quad.Ixx
    Iyy = quad.Iyy
    Izz = quad.Izz
    Ax  = quad.Ax
    Ay  = quad.Ay
    Az  = quad.Az
    Ar  = quad.Ar

    # Motor thrusts
    T1 = kT * w1**2
    T2 = kT * w2**2
    T3 = kT * w3**2
    T4 = kT * w4**2

    T = T1 + T2 + T3 + T4
    M_phi = L * (T4 - T2)
    M_theta = L * (T3 - T1)
    M_psi = kQ * (-T1 + T2 - T3 + T4)
    Omega = w1 - w2 + w3 - w4  # net rotor speed for gyro coupling



    # ====================
    #  ROTATIONAL DYNAMICS
    # ====================

    # Euler angle kinematics
    dphi    = p + np.sin(phi) * np.tan(theta) * q + np.cos(phi) * np.tan(theta) * r
    dtheta  =                     np.cos(phi) * q -                 np.sin(phi) * r
    dpsi    =     np.sin(phi) / np.cos(theta) * q + np.cos(phi) / np.cos(theta) * r

    # Angular accelerations  (Euler's equations + gyro + aero drag)
    Mx = M_phi   - Ar * p - IR * q * Omega
    My = M_theta - Ar * q + IR * p * Omega
    Mz = M_psi   - Ar * r

    dp = (Mx - (Izz - Iyy) * q * r) / Ixx
    dq = (My - (Ixx - Izz) * p * r) / Iyy
    dr = (Mz - (Iyy - Ixx) * p * q) / Izz



    # =======================
    #  TRANSLATIONAL DYNAMICS
    # =======================


    # Compute body from inertial rotation matrix for velocity and force transformations
    if ENU_body:
        BI = dcm321(phi, theta, psi) # @ dcm321(phi = np.pi) # body from inertial dcm
    else:
        BI = dcm321(phi, theta, psi) @ dcm321(phi = np.pi) # body from inertial dcm
        T = -T





    # Velocity in body frame
    u, v, w = BI @ np.array([vx, vy, vz])

    # Thrust and aerodynamic drag in body frame
    Fb = np.array([-Ax * u, -Ay * v, T - Az * w])

    # Forces back to inertial frame
    Fi = BI.T @ Fb

    # Position kinematics
    dx = vx
    dy = vy
    dz = vz

    dvx, dvy, dvz = Fi / m

    # add gravity in the inertial frame (downward)
    dvz -= g # -g -> inertial z points upward.


    # Return in rigidbody order: [x,y,z, vx,vy,vz, phi,theta,psi, p,q,r]
    return np.array([dx, dy, dz, dvx, dvy, dvz, dphi, dtheta, dpsi, dp, dq, dr])


# ============================================================
#  REFERENCE TRAJECTORY
# ============================================================


def position_reference(t, A, B, omega, z_ref, t_takeoff=8.0, t_land=8.0, t_sim=90.0):
    """
    Three-phase reference trajectory: takeoff -> figure-8 -> landing.

    Phase 1  Takeoff  : Z rises from 0 to z_ref  (smooth S-curve)
    Phase 2  Figure-8 : x=A*sin(wt), y=B*sin(2wt), z=z_ref
    Phase 3  Landing  : X/Y return to origin, Z descends to 0

    Returns
    -------
    x_ref, y_ref, z_ref_out in inertial ENU frame
    """
    t_land_start = t_sim - t_land

    if t <= t_takeoff:
        frac = t / t_takeoff
        s = 3 * frac**2 - 2 * frac**3
        return 0.0, 0.0, z_ref * s

    elif t <= t_land_start:
        tau = t - t_takeoff
        return A * np.sin(omega * tau), B * np.sin(2 * omega * tau), z_ref

    else:
        frac = (t - t_land_start) / t_land
        s = 3 * frac**2 - 2 * frac**3
        tau_l = t_land_start - t_takeoff
        xl = A * np.sin(omega * tau_l)
        yl = B * np.sin(2 * omega * tau_l)
        return xl * (1 - s), yl * (1 - s), z_ref * (1 - s)


def velocity_reference(t, A, B, omega, t_takeoff=8.0, t_land=8.0, t_sim=90.0):
    """
    Analytical time derivative of position_reference.
    Used for velocity feedforward in the outer position loop.

    Returns
    -------
    vx_ref, vy_ref in inertial ENU frame
    """
    t_land_start = t_sim - t_land

    if t <= t_takeoff:
        return 0.0, 0.0
    elif t <= t_land_start:
        tau = t - t_takeoff
        return A * omega * np.cos(omega * tau), 2 * B * omega * np.cos(2 * omega * tau)
    else:
        return 0.0, 0.0


# ============================================================
#  OUTER POSITION PID  (50 Hz)
# ============================================================


class OuterPositionPID:
    """
    Outer loop: position -> desired angles + thrust.
    Runs at 50 Hz (every 4 master timesteps).
    """

    def __init__(self, params, m, g, kT):
        self.g = g
        self.m = m

        self.KP_Z = params["Kp_z"]
        self.KI_Z = params["Ki_z"]
        self.KD_Z = params["Kd_z"]
        self.KP_X = params["Kp_x"]
        self.KI_X = params["Ki_x"]
        self.KD_X = params["Kd_x"]
        self.KP_Y = params["Kp_y"]
        self.KI_Y = params["Ki_y"]
        self.KD_Y = params["Kd_y"]

        self.AW_Z = params["AW_z"]
        self.AW_X = params["AW_x"]
        self.AW_Y = params["AW_y"]

        self.T_max = params["T_max_factor"] * kT * params["omega_max"] ** 2
        self.T_min = params["T_min"]
        self.att_cmd_limit = params["att_cmd_limit"]

        self.FF_X = params["Kff_x"]
        self.FF_Y = params["Kff_y"]

        self.int_Z = self.int_X = self.int_Y = 0.0
        # self.Xd_prev = self.Yd_prev = 0.0

    def compute(self, Xd, Yd, Zd, Vxd, Vyd, Psi_sp, quad, Ts, ENU_body=True):
        """
        Compute the control commands for the outer position loop.

        Parameters
        ----------
        Xd, Yd, Zd : reference position in inertial ENU frame [m]
        Vxd, Vyd   : reference velocities in inertial ENU frame [m/s]
        Psi_sp      : desired yaw [rad]
        quad        : c4d.rigidbody — current state
        Ts          : sample time [s]

        Returns
        -------
        T_cmd, phi_d, theta_d, psi_d
        """

        x, y, z = quad.x, quad.y, quad.z
        vx, vy, vz = quad.vx, quad.vy, quad.vz
        phi, theta, psi = quad.phi, quad.theta, quad.psi


        # reference trajectory is given in inertial frame (ENU).
        # compute errors in inertial frame and rotate them to body for the PID calculations.


        # project the force on the body frame to account for thrust limit
        if ENU_body:
            BI = dcm321(phi, theta, psi) # @ dcm321(phi = np.pi)
            HE = dcm321(psi = psi) # body from inertial for horizontal error rotation
            Tmin = self.T_min
            Tmax = self.T_max
            Tfactor = 1
            pitch_factor = 1
            phi_factor = -1
        else:
            BI = dcm321(phi, theta, psi) @ dcm321(phi = np.pi)
            HE = dcm321(psi = psi) @ dcm321(phi = np.pi) # body from inertial for horizontal error rotation
            Tmin = -self.T_max
            Tmax = self.T_min
            Tfactor = -1
            pitch_factor = -1
            phi_factor = 1



        # position error in inertial frame.
        e_X = Xd - x
        e_Y = Yd - y
        e_Z = Zd - z

        # Altitude PID
        # required z command in inertial frame
        self.int_Z = np.clip(self.int_Z + Ts * e_Z, -self.AW_Z, self.AW_Z)
        az_cmd = self.KP_Z * e_Z + self.KI_Z * self.int_Z + self.KD_Z * (-vz)


        # project the force on the body frame to account for thrust limit
        Tcmd_b = BI @ [0, 0, self.m * (self.g + az_cmd)] # add g to compensate for gravity

        T_cmd = Tfactor * np.clip(
            Tcmd_b[2],
            Tmin,
            Tmax,
        )

        # Horizontal PID — errors rotated to body frame

        Xerr = [e_X, e_Y, 0]
        Xerr_b = HE @ Xerr

        V = [vx, vy, 0]
        Vb = HE @ V

        # e_U = Xerr_b[0] - Vb[0]
        # e_V = Xerr_b[1] - Vb[1]

        self.int_X = np.clip(self.int_X + Ts * Xerr_b[0], -self.AW_X, self.AW_X)
        self.int_Y = np.clip(self.int_Y + Ts * Xerr_b[1], -self.AW_Y, self.AW_Y)

        # Velocity feedforward
        Vff = [Vxd, Vyd, 0]
        Vff_b = HE @ Vff
        ff_theta = self.FF_X * Vff_b[0]
        ff_phi = -self.FF_Y * Vff_b[1]


        # theta_d → forward accel
        theta_d = np.clip(
            pitch_factor * (self.KP_X * Xerr_b[0] -
                            self.KP_X * Vb[0] +
                            self.KI_X * self.int_X +
                            self.KD_X * (Vff_b[0] - Vb[0]) +
                            ff_theta
                            ),
            -self.att_cmd_limit,
            self.att_cmd_limit,
        )

        # phi_d → lateral accel
        phi_d = np.clip(
            phi_factor * (self.KP_Y * Xerr_b[1] -
                          self.KP_Y * Vb[1] +
                          self.KI_Y * self.int_Y +
                          self.KD_Y * (Vff_b[1] - Vb[1]) +
                          ff_phi
                          ),
            -self.att_cmd_limit,
            self.att_cmd_limit,
        )

        # self.Xd_prev = Xd
        # self.Yd_prev = Yd

        return T_cmd, phi_d, theta_d, Psi_sp


# ============================================================
#  MIDDLE ATTITUDE PID  (100 Hz)
# ============================================================


class MiddleAttitudePID:
    """
    Middle loop: desired angles -> desired body rates.
    Runs at 100 Hz (every 2 master timesteps).
    """

    def __init__(self, params):
        self.KP_phi = params["Kp_phi"]
        self.KI_phi = params["Ki_phi"]
        self.KD_phi = params["Kd_phi"]
        self.KP_theta = params["Kp_theta"]
        self.KI_theta = params["Ki_theta"]
        self.KD_theta = params["Kd_theta"]
        self.KP_psi = params["Kp_psi"]
        self.KI_psi = params["Ki_psi"]
        self.KD_psi = params["Kd_psi"]

        self.AW_phi = params["AW_phi"]
        self.AW_theta = params["AW_theta"]
        self.AW_psi = params["AW_psi"]
        self.yaw_rate_limit = params["yaw_rate_limit"]

        self.int_phi = self.int_theta = self.int_psi = 0.0

    def compute(self, phi_d, theta_d, psi_d, quad, Ts):
        """
        Parameters
        ----------
        phi_d, theta_d, psi_d : desired angles [rad]
        quad : c4d.rigidbody
        Ts   : sample time [s]

        Returns
        -------
        p_d, q_d, r_d : desired body rates [rad/s]
        """
        e_phi = phi_d - quad.phi
        e_theta = theta_d - quad.theta
        # wrapping on circle:
        e_psi = np.arctan2(np.sin(psi_d - quad.psi), np.cos(psi_d - quad.psi))

        self.int_phi = np.clip(
            self.int_phi + Ts * e_phi,
            -self.AW_phi / self.KI_phi,
            self.AW_phi / self.KI_phi,
        )
        self.int_theta = np.clip(
            self.int_theta + Ts * e_theta,
            -self.AW_theta / self.KI_theta,
            self.AW_theta / self.KI_theta,
        )
        self.int_psi = np.clip(
            self.int_psi + Ts * e_psi,
            -self.AW_psi / self.KI_psi,
            self.AW_psi / self.KI_psi,
        )

        rl = self.yaw_rate_limit * 3
        p_d = np.clip(
            self.KP_phi * e_phi + self.KI_phi * self.int_phi - self.KD_phi * quad.p,
            -rl,
            rl,
        )
        q_d = np.clip(
            self.KP_theta * e_theta
            + self.KI_theta * self.int_theta
            - self.KD_theta * quad.q,
            -rl,
            rl,
        )
        r_d = np.clip(
            self.KP_psi * e_psi + self.KI_psi * self.int_psi - self.KD_psi * quad.r,
            -self.yaw_rate_limit,
            self.yaw_rate_limit,
        )

        # desired body rates
        return p_d, q_d, r_d


# ============================================================
#  INNER RATE PID  (200 Hz)
# ============================================================


class InnerRatePID:
    """
    Inner loop: desired body rates -> torque commands.
    Runs at 200 Hz (every master timestep).
    """

    def __init__(self, params, Ixx, Iyy, Izz, L, kT):
        self.KP_p = params["Kp_p"]
        self.KI_p = params["Ki_p"]
        self.KD_p = params["Kd_p"]
        self.KP_q = params["Kp_q"]
        self.KI_q = params["Ki_q"]
        self.KD_q = params["Kd_q"]
        self.KP_r = params["Kp_r"]
        self.KI_r = params["Ki_r"]
        self.KD_r = params["Kd_r"]

        self.N_rate = params["N_rate"]
        self.M_max = L * kT * params["omega_max"] ** 2

        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz

        self.int_p = self.int_q = self.int_r = 0.0
        self.ep_prev = self.eq_prev = self.er_prev = 0.0

    def compute(self, p_d, q_d, r_d, quad, Ts):
        """
        Parameters
        ----------
        p_d, q_d, r_d : desired body rates [rad/s]
        quad : c4d.rigidbody
        Ts   : sample time [s]

        Returns
        -------
        tau_phi, tau_theta, tau_psi : torque commands [N.m]
        """
        ep = p_d - quad.p
        eq = q_d - quad.q
        er = r_d - quad.r

        # Tustin integrator
        self.int_p += (Ts / 2) * (ep + self.ep_prev)
        self.int_q += (Ts / 2) * (eq + self.eq_prev)
        self.int_r += (Ts / 2) * (er + self.er_prev)

        # Filtered derivative
        d = 1 + self.N_rate * Ts
        dp = self.N_rate * (ep - self.ep_prev) / d
        dq = self.N_rate * (eq - self.eq_prev) / d
        dr = self.N_rate * (er - self.er_prev) / d

        tau_phi_raw = self.Ixx * (
            self.KP_p * ep + self.KI_p * self.int_p + self.KD_p * dp
        )
        tau_theta_raw = self.Iyy * (
            self.KP_q * eq + self.KI_q * self.int_q + self.KD_q * dq
        )
        tau_psi_raw = self.Izz * (
            self.KP_r * er + self.KI_r * self.int_r + self.KD_r * dr
        )

        tau_phi = np.clip(tau_phi_raw, -self.M_max, self.M_max)
        tau_theta = np.clip(tau_theta_raw, -self.M_max, self.M_max)
        tau_psi = np.clip(tau_psi_raw, -self.M_max, self.M_max)

        # Back-calculation anti-windup
        AW = 0.1
        self.int_p += AW * (tau_phi - tau_phi_raw) / (self.Ixx * self.KI_p + 1e-9)
        self.int_q += AW * (tau_theta - tau_theta_raw) / (self.Iyy * self.KI_q + 1e-9)
        self.int_r += AW * (tau_psi - tau_psi_raw) / (self.Izz * self.KI_r + 1e-9)

        self.ep_prev = ep
        self.eq_prev = eq
        self.er_prev = er

        # torques in body axes
        return tau_phi, tau_theta, tau_psi


# ============================================================
#  CONTROL ALLOCATOR
# ============================================================


class ControlAllocator:
    """
    Converts thrust + torques to individual rotor speeds.

    Motor layout — plus (+) configuration:

      w1: front (+x)  CW
      w2: left  (+y)  CCW
      w3: rear  (-x)  CW
      w4: right (-y)  CCW

    """

    def __init__(self, kT, kQ, L, omega_max):
        self.kT = kT
        self.kQ = kQ
        self.L = L
        self.sq_min = 0.0
        self.sq_max = omega_max**2

    def allocate(self, T_cmd, tau_phi, tau_theta, tau_psi):
        """
        Parameters
        ----------
        T_cmd     : total thrust [N]
        tau_phi   : roll  torque [N.m] (difference between left and right motors)
        tau_theta : pitch torque [N.m] (difference between front and rear motors)
        tau_psi   : yaw   torque [N.m]

        Returns
        -------
        w1, w2, w3, w4 : rotor speeds [rad/s]
        """
        T4K = T_cmd / (4 * self.kT)
        Mt = tau_theta / (2 * self.kT * self.L)
        Mp = tau_phi / (2 * self.kT * self.L)
        My = tau_psi / (4 * self.kQ)

        def cl(v):
            return np.clip(v, self.sq_min, self.sq_max)

        return (
            np.sqrt(cl(T4K - Mt - My)),
            np.sqrt(cl(T4K - Mp + My)),
            np.sqrt(cl(T4K + Mt - My)),
            np.sqrt(cl(T4K + Mp + My)),
        )


# ============================================================
#  MAIN LOOP
# ============================================================


def run_fig8_pid(config, enu = True):

    # Initialize the rigidbody — quadcopter starts at rest on the ground
    quad = c4d.rigidbody()

    for k, v in config["quad"].items():
        setattr(quad, k, v)

    # Control inputs stored alongside state
    if enu:
        quad.F = quad.m * quad.g  # thrust [N]  — initialized to hover
    else:
        quad.F = quad.m * quad.g  # thrust [N]  — initialized to hover

    quad.tau_phi = 0.0  # roll  torque [N.m]
    quad.tau_theta = 0.0  # pitch torque [N.m]
    quad.tau_psi = 0.0  # yaw   torque [N.m]

    # Trajectory parameters
    A, B, omega, z_ref = (
        config["trajectory"]["A"],
        config["trajectory"]["B"],
        config["trajectory"]["omega"],
        config["trajectory"]["z_ref"],
    )

    # Instantiate controllers
    outer_ctrl = OuterPositionPID(
        config["controller"], quad.m, quad.g, quad.kT
    )
    mid_ctrl   = MiddleAttitudePID(
        config["controller"]
    )
    inner_ctrl = InnerRatePID(
        config["controller"], quad.Ixx, quad.Iyy, quad.Izz, quad.l, quad.kT
    )
    allocator = ControlAllocator(
        quad.kT, quad.kQ, quad.l, config["controller"]["omega_max"]
    )

    # Loop rate counters
    Ts_outer = 1.0 / 50.0  # 0.020 s
    Ts_middle = 1.0 / 100.0  # 0.010 s
    outer_time = middle_time = 0.0

    # Initial setpoints
    dt, tf = config["sim"]["dt"], config["sim"]["tf"]
    psi_d = 0.0  # desired yaw — fixed
    phi_d = theta_d = 0.0
    p_d = q_d = r_d = 0.0
    T_cmd = quad.m * quad.g  # start at hover thrust
    rotor_speeds = np.array([np.sqrt(T_cmd / (4 * quad.kT))] * 4)

    print(f"Simulation start  |  tf = {tf} s  |  dt = {dt} s")

    for t in np.arange(0, tf, dt):

        # ── Store state and control inputs ──
        quad.store(t)
        quad.storeparams(["F", "tau_phi", "tau_theta", "tau_psi"], t=t)

        # ── Reference at current time ──
        xd, yd, zd = position_reference(t, A, B, omega, z_ref, t_sim=tf)
        vxd_ff, vyd_ff = velocity_reference(t, A, B, omega, t_sim=tf)

        # ── Outer loop — Position  (50 Hz) ──
        outer_time += dt
        if outer_time >= Ts_outer:
            T_cmd, phi_d, theta_d, psi_d = outer_ctrl.compute(
                xd, yd, zd, vxd_ff, vyd_ff, psi_d, quad, Ts_outer, enu
            )
            quad.F = T_cmd
            outer_time = 0.0

        # ── Middle loop — Attitude  (100 Hz) ──
        middle_time += dt
        if middle_time >= Ts_middle:
            p_d, q_d, r_d = mid_ctrl.compute(phi_d, theta_d, psi_d, quad, Ts_middle)
            middle_time = 0.0

        # ── Inner loop — Rate  (200 Hz, every step) ──
        quad.tau_phi, quad.tau_theta, quad.tau_psi = inner_ctrl.compute(
            p_d, q_d, r_d, quad, dt
        )

        # ── Control allocation — torques to rotor speeds ──
        rotor_speeds = np.array(
            allocator.allocate(quad.F, quad.tau_phi, quad.tau_theta, quad.tau_psi)
        )

        sol = solve_ivp(dynamics, [t, t + dt], quad.X, args=(quad, rotor_speeds, enu))
        quad.X = sol.y[:, -1]

        t += dt

    print("Simulation complete.")

    return quad


# ============================================================
#  PLOTTING
# ============================================================


def plot_results(quad, trajectory):
    """
    Generate result plots using quad.data() to retrieve stored histories.

    Figure 1 — Time histories (3 stacked subplots):
        Position (actual + reference), Euler angles, Control inputs

    Figure 2 — Simulation results dashboard (2x3 subplots):
        3D trajectory | XY plane | Horizontal position tracking
        Altitude      | Error    | Attitude angles

    Parameters
    ----------
    quad       : c4d.rigidbody — populated by the main loop
    trajectory : dict
    """
    A = trajectory["A"]
    B = trajectory["B"]
    omega = trajectory["omega"]
    z_ref = trajectory["z_ref"]
    t_takeoff = trajectory.get("t_takeoff", 8.0)
    t_land = trajectory.get("t_land", 8.0)
    t_sim = trajectory.get("t_sim", trajectory.get("t_end", 90.0))

    # ── Retrieve stored histories via quad.data() ──
    t_hist = quad.data("x")[0]
    x_hist = quad.data("x")[1]
    y_hist = quad.data("y")[1]
    z_hist = quad.data("z")[1]

    phi_hist = quad.data("phi", scale=c4d.r2d)[1]
    theta_hist = quad.data("theta", scale=c4d.r2d)[1]
    psi_hist = quad.data("psi", scale=c4d.r2d)[1]

    # Reference at every stored time
    ref = np.array(
        [
            position_reference(t, A, B, omega, z_ref, t_takeoff, t_land, t_sim)
            for t in t_hist
        ]
    )

    x_ref = ref[:, 0]
    y_ref = ref[:, 1]
    z_ref_hist = ref[:, 2]

    # Position error magnitude
    pos_err = np.sqrt(
        (x_hist - x_ref) ** 2 + (y_hist - y_ref) ** 2 + (z_hist - z_ref_hist) ** 2
    )

    lw = 1.5  # linewidth

    # ══════════════════════════════════════════════════════
    #  FIGURE 2 — Dashboard
    #  2 x 3 grid: 3D | XY | Horiz position
    #              Alt | Error | Attitude
    # ══════════════════════════════════════════════════════
    fig2 = plt.figure(figsize=(16, 10))
    fig2.suptitle(
        "Cascade PID Quadcopter — Simulation Results", fontsize=16, fontweight="bold"
    )

    # -- 3D Trajectory --
    ax3d = fig2.add_subplot(2, 3, 1, projection="3d")
    ax3d.plot(x_hist, y_hist, z_hist, "b-", linewidth=lw, label="Actual")
    ax3d.plot(x_ref, y_ref, z_ref_hist, "r--", linewidth=lw, label="Reference")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3D Trajectory")
    ax3d.legend(fontsize=8)
    ax3d.grid(True)

    # -- XY Plane --
    ax = fig2.add_subplot(2, 3, 2)
    ax.plot(x_hist, y_hist, "b-", linewidth=lw, label="Actual")
    ax.plot(x_ref, y_ref, "r--", linewidth=lw, label="Reference")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Plane")
    ax.legend(fontsize=8)
    ax.grid(True)
    ax.axis("equal")

    # -- Horizontal position tracking --
    ax = fig2.add_subplot(2, 3, 3)
    ax.plot(t_hist, x_hist, "b-", linewidth=lw, label="X actual")
    ax.plot(t_hist, x_ref, "r--", linewidth=lw, label="X ref")
    ax.plot(t_hist, y_hist, "g-", linewidth=lw, label="Y actual")
    ax.plot(t_hist, y_ref, "m--", linewidth=lw, label="Y ref")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.set_title("Horizontal Position Tracking")
    ax.legend(fontsize=8)
    ax.grid(True)

    # -- Altitude tracking --
    ax = fig2.add_subplot(2, 3, 4)
    ax.plot(t_hist, z_hist, "b-", linewidth=lw, label="Z actual")
    ax.plot(t_hist, z_ref_hist, "r--", linewidth=lw, label="Z ref")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Altitude Tracking")
    ax.legend(fontsize=8)
    ax.grid(True)

    # -- Position tracking error --
    ax = fig2.add_subplot(2, 3, 5)
    ax.plot(t_hist, pos_err, "r-", linewidth=lw)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (m)")
    ax.set_title("Position Tracking Error")
    ax.grid(True)

    # -- Attitude angles --
    ax = fig2.add_subplot(2, 3, 6)
    ax.plot(t_hist, phi_hist, "b-", linewidth=lw, label="Roll (Phi)")
    ax.plot(t_hist, theta_hist, "g-", linewidth=lw, label="Pitch (Theta)")
    ax.plot(t_hist, psi_hist, "r-", linewidth=lw, label="Yaw (Psi)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Attitude Angles")
    ax.legend(fontsize=8)
    ax.grid(True)

    plt.tight_layout()
    plt.show()


# ============================================================
#  METRICS
# ============================================================


def compute_metrics(quad, trajectory):
    """
    Compute RMSE tracking metrics over the figure-8 phase only.

    Uses quad.data() to retrieve stored histories.

    Parameters
    ----------
    quad       : c4d.rigidbody — populated by the main loop
    trajectory : dict

    Returns
    -------
    dict with rmse_x, rmse_y, rmse_z, norm_x, norm_y, norm_z, max_z_dev
    """
    A = trajectory["A"]
    B = trajectory["B"]
    omega = trajectory["omega"]
    z_ref = trajectory["z_ref"]
    t_takeoff = trajectory.get("t_takeoff", 8.0)
    t_land = trajectory.get("t_land", 8.0)
    t_sim = trajectory.get("t_sim", trajectory.get("t_end", 90.0))
    t_land_start = t_sim - t_land

    t_hist = quad.data("x")[0]
    x_hist = quad.data("x")[1]
    y_hist = quad.data("y")[1]
    z_hist = quad.data("z")[1]

    # Figure-8 phase indices
    idx = (t_hist >= t_takeoff) & (t_hist <= t_land_start)
    t_ss = t_hist[idx]

    pos_ref_ss = np.array(
        [
            position_reference(t, A, B, omega, z_ref, t_takeoff, t_land, t_sim)
            for t in t_ss
        ]
    )

    x_ref_ss = pos_ref_ss[:, 0]
    y_ref_ss = pos_ref_ss[:, 1]
    z_ref_ss = np.full(len(t_ss), z_ref)

    rmse_x = np.sqrt(np.mean((x_hist[idx] - x_ref_ss) ** 2))
    rmse_y = np.sqrt(np.mean((y_hist[idx] - y_ref_ss) ** 2))
    rmse_z = np.sqrt(np.mean((z_hist[idx] - z_ref_ss) ** 2))

    norm_x = rmse_x / A * 100
    norm_y = rmse_y / B * 100
    norm_z = rmse_z / z_ref * 100

    max_z_dev = np.max(np.abs(z_hist[idx] - z_ref))

    return {
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "rmse_z": rmse_z,
        "norm_x": norm_x,
        "norm_y": norm_y,
        "norm_z": norm_z,
        "max_z_dev": max_z_dev,
    }
