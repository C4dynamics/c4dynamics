"""
quad_pid_utils.py
=================
Supporting module for the Quadcopter Cascade PID notebook.


The notebook contains the main loop, parameters, and results.
This file contains ONLY the implementation classes and helpers.

Contents:
  - dynamics(quad, rotor_speeds, vehicle) : state derivatives
  - get_reference()                       : 3-phase trajectory
  - get_reference_velocity()              : feedforward velocity
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


# ============================================================
#  DYNAMICS  (called every integration step)
# ============================================================

def dynamics(quad, rotor_speeds, vehicle):
    """
    Compute the 12-state derivatives for the quadcopter.

    Accepts and returns arrays compatible with c4d.rigidbody.X:
        X = [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]

    Parameters
    ----------
    quad         : c4d.rigidbody  — current state lives in quad.X
    rotor_speeds : array [w1,w2,w3,w4]  rad/s
    vehicle      : dict of physical parameters

    Returns
    -------
    dX : array (12,) — state derivatives
    """

    x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = quad.X
    w1, w2, w3, w4 = rotor_speeds

    m   = vehicle['m'];   g   = vehicle['g']
    L   = vehicle['l'];   kT  = vehicle['kT'];  kQ = vehicle['kQ']
    IR  = vehicle['IR']
    Ixx = vehicle['Ixx']; Iyy = vehicle['Iyy']; Izz = vehicle['Izz']
    Ax  = vehicle['Ax'];  Ay  = vehicle['Ay'];  Az  = vehicle['Az']
    Ar  = vehicle['Ar']

    # Motor thrusts
    T1 = kT*w1**2;  T2 = kT*w2**2;  T3 = kT*w3**2;  T4 = kT*w4**2
    T       = T1+T2+T3+T4
    M_phi   = L*(T4-T2)
    M_theta = L*(T3-T1)
    M_psi   = kQ*(-T1+T2-T3+T4)
    Omega   = w1-w2+w3-w4          # net rotor speed for gyro coupling

    # Angular accelerations  (Euler's equations + gyro + aero drag)
    dp = ((Iyy-Izz)/Ixx)*q*r - (IR/Ixx)*q*Omega + M_phi/Ixx   - (Ar/Ixx)*p
    dq = ((Izz-Ixx)/Iyy)*p*r + (IR/Iyy)*p*Omega + M_theta/Iyy - (Ar/Iyy)*q
    dr = ((Ixx-Iyy)/Izz)*p*q                     + M_psi/Izz   - (Ar/Izz)*r

    # Euler angle kinematics
    dphi   = p + np.sin(phi)*np.tan(theta)*q + np.cos(phi)*np.tan(theta)*r
    dtheta = np.cos(phi)*q - np.sin(phi)*r
    dpsi   = np.sin(phi)/np.cos(theta)*q + np.cos(phi)/np.cos(theta)*r

    # Translational accelerations (inertial frame)
    # Thrust projected from body to inertial via ZYX rotation
    dvx = (np.sin(phi)*np.sin(psi) + np.cos(phi)*np.sin(theta)*np.cos(psi))*T/m - (Ax/m)*vx
    dvy = (-np.sin(phi)*np.cos(psi) + np.cos(phi)*np.sin(theta)*np.sin(psi))*T/m - (Ay/m)*vy
    dvz = -g + np.cos(phi)*np.cos(theta)*T/m - (Az/m)*vz

    # Position kinematics
    dx = vx;  dy = vy;  dz = vz

    # Return in rigidbody order: [x,y,z, vx,vy,vz, phi,theta,psi, p,q,r]
    return np.array([dx, dy, dz, dvx, dvy, dvz, dphi, dtheta, dpsi, dp, dq, dr])


# ============================================================
#  REFERENCE TRAJECTORY
# ============================================================

def get_reference(t, A, B, omega, z_ref,
                  t_takeoff=8.0, t_land=8.0, t_sim=90.0):
    """
    Three-phase reference trajectory: takeoff -> figure-8 -> landing.

    Phase 1  Takeoff  : Z rises from 0 to z_ref  (smooth S-curve)
    Phase 2  Figure-8 : x=A*sin(wt), y=B*sin(2wt), z=z_ref
    Phase 3  Landing  : X/Y return to origin, Z descends to 0

    Returns
    -------
    x_ref, y_ref, z_ref_out
    """
    t_land_start = t_sim - t_land

    if t <= t_takeoff:
        frac = t / t_takeoff
        s    = 3*frac**2 - 2*frac**3
        return 0.0, 0.0, z_ref*s

    elif t <= t_land_start:
        tau = t - t_takeoff
        return A*np.sin(omega*tau), B*np.sin(2*omega*tau), z_ref

    else:
        frac  = (t - t_land_start) / t_land
        s     = 3*frac**2 - 2*frac**3
        tau_l = t_land_start - t_takeoff
        xl    = A*np.sin(omega*tau_l)
        yl    = B*np.sin(2*omega*tau_l)
        return xl*(1-s), yl*(1-s), z_ref*(1-s)


def get_reference_velocity(t, A, B, omega,
                           t_takeoff=8.0, t_land=8.0, t_sim=90.0):
    """
    Analytical time derivative of get_reference.
    Used for velocity feedforward in the outer position loop.

    Returns
    -------
    vx_ref, vy_ref
    """
    t_land_start = t_sim - t_land

    if t <= t_takeoff:
        return 0.0, 0.0
    elif t <= t_land_start:
        tau = t - t_takeoff
        return A*omega*np.cos(omega*tau), 2*B*omega*np.cos(2*omega*tau)
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
        self.g = g;  self.m = m

        self.KP_Z = params['Kp_z'];  self.KI_Z = params['Ki_z'];  self.KD_Z = params['Kd_z']
        self.KP_X = params['Kp_x'];  self.KI_X = params['Ki_x'];  self.KD_X = params['Kd_x']
        self.KP_Y = params['Kp_y'];  self.KI_Y = params['Ki_y'];  self.KD_Y = params['Kd_y']

        self.AW_Z = params['AW_z'];  self.AW_X = params['AW_x'];  self.AW_Y = params['AW_y']

        self.T_max         = params['T_max_factor'] * kT * params['omega_max']**2
        self.T_min         = params['T_min']
        self.att_cmd_limit = params['att_cmd_limit']

        self.FF_X = params['Kff_x'];  self.FF_Y = params['Kff_y']

        self.int_Z = self.int_X = self.int_Y = 0.0
        self.Xd_prev = self.Yd_prev = 0.0

    def compute(self, Xd, Yd, Zd, Psi_sp, quad, Ts):
        """
        Parameters
        ----------
        Xd, Yd, Zd : reference position [m]
        Psi_sp      : desired yaw [rad]
        quad        : c4d.rigidbody — current state
        Ts          : sample time [s]

        Returns
        -------
        T_cmd, phi_d, theta_d, psi_d
        """
        x,y,z   = quad.x, quad.y, quad.z
        vx,vy,vz = quad.vx, quad.vy, quad.vz
        phi,theta,psi = quad.phi, quad.theta, quad.psi

        # Altitude PID
        e_Z = Zd - z
        self.int_Z = np.clip(self.int_Z + Ts*e_Z, -self.AW_Z, self.AW_Z)
        az_cmd = self.KP_Z*e_Z + self.KI_Z*self.int_Z + self.KD_Z*(-vz)
        T_cmd  = np.clip(self.m*(self.g + az_cmd) / max(0.5, np.cos(phi)*np.cos(theta)),
                         self.T_min, self.T_max)

        # Horizontal PID — errors rotated to body frame
        e_X_b =  (Xd-x)*np.cos(psi) + (Yd-y)*np.sin(psi)
        e_Y_b =  (Yd-y)*np.cos(psi) - (Xd-x)*np.sin(psi)
        vx_b  =  vx*np.cos(psi) + vy*np.sin(psi)
        vy_b  = -vx*np.sin(psi) + vy*np.cos(psi)
        e_U   = e_X_b - vx_b
        e_V   = e_Y_b - vy_b

        self.int_X = np.clip(self.int_X + Ts*e_U, -self.AW_X, self.AW_X)
        self.int_Y = np.clip(self.int_Y + Ts*e_V, -self.AW_Y, self.AW_Y)

        # Velocity feedforward
        Xd_dot = (Xd - self.Xd_prev) / Ts if Ts > 0 else 0.0
        Yd_dot = (Yd - self.Yd_prev) / Ts if Ts > 0 else 0.0
        ff_theta =  self.FF_X * (Xd_dot*np.cos(psi) + Yd_dot*np.sin(psi))
        ff_phi   = -self.FF_Y * (Yd_dot*np.cos(psi) - Xd_dot*np.sin(psi))

        theta_d = np.clip(self.KP_X*e_U + self.KI_X*self.int_X + self.KD_X*(-vx_b) + ff_theta,
                          -self.att_cmd_limit, self.att_cmd_limit)
        phi_d   = np.clip(-(self.KP_Y*e_V + self.KI_Y*self.int_Y + self.KD_Y*(-vy_b)) + ff_phi,
                          -self.att_cmd_limit, self.att_cmd_limit)

        self.Xd_prev = Xd;  self.Yd_prev = Yd
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
        self.KP_phi   = params['Kp_phi'];   self.KI_phi   = params['Ki_phi'];   self.KD_phi   = params['Kd_phi']
        self.KP_theta = params['Kp_theta']; self.KI_theta = params['Ki_theta']; self.KD_theta = params['Kd_theta']
        self.KP_psi   = params['Kp_psi'];   self.KI_psi   = params['Ki_psi'];   self.KD_psi   = params['Kd_psi']

        self.AW_phi   = params['AW_phi'];   self.AW_theta = params['AW_theta']; self.AW_psi = params['AW_psi']
        self.yaw_rate_limit = params['yaw_rate_limit']

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
        e_phi   = phi_d   - quad.phi
        e_theta = theta_d - quad.theta
        e_psi   = np.arctan2(np.sin(psi_d - quad.psi), np.cos(psi_d - quad.psi))

        self.int_phi   = np.clip(self.int_phi   + Ts*e_phi,
                                 -self.AW_phi/self.KI_phi,   self.AW_phi/self.KI_phi)
        self.int_theta = np.clip(self.int_theta + Ts*e_theta,
                                 -self.AW_theta/self.KI_theta, self.AW_theta/self.KI_theta)
        self.int_psi   = np.clip(self.int_psi   + Ts*e_psi,
                                 -self.AW_psi/self.KI_psi,   self.AW_psi/self.KI_psi)

        rl  = self.yaw_rate_limit * 3
        p_d = np.clip(self.KP_phi*e_phi     + self.KI_phi*self.int_phi   - self.KD_phi*quad.p,   -rl, rl)
        q_d = np.clip(self.KP_theta*e_theta + self.KI_theta*self.int_theta - self.KD_theta*quad.q, -rl, rl)
        r_d = np.clip(self.KP_psi*e_psi     + self.KI_psi*self.int_psi   - self.KD_psi*quad.r,
                      -self.yaw_rate_limit, self.yaw_rate_limit)
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
        self.KP_p = params['Kp_p']; self.KI_p = params['Ki_p']; self.KD_p = params['Kd_p']
        self.KP_q = params['Kp_q']; self.KI_q = params['Ki_q']; self.KD_q = params['Kd_q']
        self.KP_r = params['Kp_r']; self.KI_r = params['Ki_r']; self.KD_r = params['Kd_r']

        self.N_rate = params['N_rate']
        self.M_max  = L * kT * params['omega_max']**2

        self.Ixx = Ixx;  self.Iyy = Iyy;  self.Izz = Izz

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
        ep = p_d - quad.p;  eq = q_d - quad.q;  er = r_d - quad.r

        # Tustin integrator
        self.int_p += (Ts/2)*(ep + self.ep_prev)
        self.int_q += (Ts/2)*(eq + self.eq_prev)
        self.int_r += (Ts/2)*(er + self.er_prev)

        # Filtered derivative
        d = 1 + self.N_rate*Ts
        dp = self.N_rate*(ep - self.ep_prev)/d
        dq = self.N_rate*(eq - self.eq_prev)/d
        dr = self.N_rate*(er - self.er_prev)/d

        tau_phi_raw   = self.Ixx*(self.KP_p*ep + self.KI_p*self.int_p + self.KD_p*dp)
        tau_theta_raw = self.Iyy*(self.KP_q*eq + self.KI_q*self.int_q + self.KD_q*dq)
        tau_psi_raw   = self.Izz*(self.KP_r*er + self.KI_r*self.int_r + self.KD_r*dr)

        tau_phi   = np.clip(tau_phi_raw,   -self.M_max, self.M_max)
        tau_theta = np.clip(tau_theta_raw, -self.M_max, self.M_max)
        tau_psi   = np.clip(tau_psi_raw,   -self.M_max, self.M_max)

        # Back-calculation anti-windup
        AW = 0.1
        self.int_p += AW*(tau_phi   - tau_phi_raw)   / (self.Ixx*self.KI_p + 1e-9)
        self.int_q += AW*(tau_theta - tau_theta_raw) / (self.Iyy*self.KI_q + 1e-9)
        self.int_r += AW*(tau_psi   - tau_psi_raw)   / (self.Izz*self.KI_r + 1e-9)

        self.ep_prev = ep;  self.eq_prev = eq;  self.er_prev = er
        return tau_phi, tau_theta, tau_psi


# ============================================================
#  CONTROL ALLOCATOR
# ============================================================

class ControlAllocator:
    """
    Converts thrust + torques to individual rotor speeds.

    Motor layout — plus (+) configuration:
      w1: front (+x)  CW     w2: left  (+y)  CCW
      w3: rear  (-x)  CW     w4: right (-y)  CCW
    """

    def __init__(self, kT, kQ, L, omega_max):
        self.kT  = kT;  self.kQ = kQ;  self.L = L
        self.sq_min = 0.0;  self.sq_max = omega_max**2

    def allocate(self, T_cmd, tau_phi, tau_theta, tau_psi):
        """
        Parameters
        ----------
        T_cmd     : total thrust [N]
        tau_phi   : roll  torque [N.m]
        tau_theta : pitch torque [N.m]
        tau_psi   : yaw   torque [N.m]

        Returns
        -------
        w1, w2, w3, w4 : rotor speeds [rad/s]
        """
        T4K = T_cmd    / (4*self.kT)
        Mt  = tau_theta / (2*self.kT*self.L)
        Mp  = tau_phi   / (2*self.kT*self.L)
        My  = tau_psi   / (4*self.kQ)

        cl = lambda v: np.clip(v, self.sq_min, self.sq_max)
        return (np.sqrt(cl(T4K - Mt - My)),
                np.sqrt(cl(T4K - Mp + My)),
                np.sqrt(cl(T4K + Mt - My)),
                np.sqrt(cl(T4K + Mp + My)))


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
    A     = trajectory['A'];      B     = trajectory['B']
    omega = trajectory['omega'];  z_ref = trajectory['z_ref']
    t_takeoff = trajectory.get('t_takeoff', 8.0)
    t_land    = trajectory.get('t_land',    8.0)
    t_sim     = trajectory.get('t_sim', trajectory.get('t_end', 90.0))

    # ── Retrieve stored histories via quad.data() ──
    t_hist = quad.data('x')[0]
    x_hist = quad.data('x')[1]
    y_hist = quad.data('y')[1]
    z_hist = quad.data('z')[1]

    phi_hist   = quad.data('phi',   scale=c4d.r2d)[1]
    theta_hist = quad.data('theta', scale=c4d.r2d)[1]
    psi_hist   = quad.data('psi',   scale=c4d.r2d)[1]

    F_hist         = quad.data('F')[1]
    tau_phi_hist   = quad.data('tau_phi')[1]
    tau_theta_hist = quad.data('tau_theta')[1]
    tau_psi_hist   = quad.data('tau_psi')[1]

    # Reference at every stored time
    ref    = np.array([get_reference(t, A, B, omega, z_ref, t_takeoff, t_land, t_sim)
                       for t in t_hist])
    x_ref  = ref[:, 0];  y_ref = ref[:, 1];  z_ref_hist = ref[:, 2]

    # Position error magnitude
    pos_err = np.sqrt((x_hist - x_ref)**2 + (y_hist - y_ref)**2 + (z_hist - z_ref_hist)**2)

    lw = 1.5   # linewidth

    # ══════════════════════════════════════════════════════
    #  FIGURE 1 — Time histories
    #  3 stacked subplots: position | euler angles | control
    # ══════════════════════════════════════════════════════
    fig1, axs = plt.subplots(3, 1, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.4)

    # -- Position --
    axs[0].plot(t_hist, x_hist,   linewidth=lw,           label='x actual')
    axs[0].plot(t_hist, x_ref,    linewidth=lw, ls='--',  label='x ref')
    axs[0].plot(t_hist, y_hist,   linewidth=lw,           label='y actual')
    axs[0].plot(t_hist, y_ref,    linewidth=lw, ls='--',  label='y ref')
    axs[0].plot(t_hist, z_hist,   linewidth=lw,           label='z actual')
    axs[0].plot(t_hist, z_ref_hist, linewidth=lw, ls='--', label='z ref')
    axs[0].legend(fontsize=9)
    c4d.plotdefaults(axs[0], 'Position', 'time [s]', 'm', fontsize=13)

    # -- Euler angles --
    axs[1].plot(t_hist, phi_hist,   linewidth=lw, label=r'$\varphi$ roll')
    axs[1].plot(t_hist, theta_hist, linewidth=lw, label=r'$\theta$ pitch')
    axs[1].plot(t_hist, psi_hist,   linewidth=lw, label=r'$\psi$ yaw')
    axs[1].legend(fontsize=9)
    c4d.plotdefaults(axs[1], 'Euler Angles', 'time [s]', 'deg', fontsize=13)

    # -- Control inputs --
    axs[2].plot(t_hist, F_hist,         linewidth=lw, label='F [N]')
    axs[2].plot(t_hist, tau_phi_hist,   linewidth=lw, label=r'$\tau_\varphi$')
    axs[2].plot(t_hist, tau_theta_hist, linewidth=lw, label=r'$\tau_\theta$')
    axs[2].plot(t_hist, tau_psi_hist,   linewidth=lw, label=r'$\tau_\psi$')
    axs[2].legend(fontsize=9)
    c4d.plotdefaults(axs[2], 'Control Inputs', 'time [s]', 'N / N.m', fontsize=13)

    plt.tight_layout()
    plt.show()

    # ══════════════════════════════════════════════════════
    #  FIGURE 2 — Dashboard
    #  2 x 3 grid: 3D | XY | Horiz position
    #              Alt | Error | Attitude
    # ══════════════════════════════════════════════════════
    fig2 = plt.figure(figsize=(16, 10))
    fig2.suptitle('Cascade PID Quadcopter — Simulation Results',
                  fontsize=16, fontweight='bold')

    # -- 3D Trajectory --
    ax3d = fig2.add_subplot(2, 3, 1, projection='3d')
    ax3d.plot(x_hist, y_hist, z_hist,    'b-',  linewidth=lw, label='Actual')
    ax3d.plot(x_ref,  y_ref,  z_ref_hist,'r--', linewidth=lw, label='Reference')
    ax3d.set_xlabel('X (m)');  ax3d.set_ylabel('Y (m)');  ax3d.set_zlabel('Z (m)')
    ax3d.set_title('3D Trajectory')
    ax3d.legend(fontsize=8)
    ax3d.grid(True)

    # -- XY Plane --
    ax = fig2.add_subplot(2, 3, 2)
    ax.plot(x_hist, y_hist, 'b-',  linewidth=lw, label='Actual')
    ax.plot(x_ref,  y_ref,  'r--', linewidth=lw, label='Reference')
    ax.set_xlabel('X (m)');  ax.set_ylabel('Y (m)')
    ax.set_title('XY Plane')
    ax.legend(fontsize=8);  ax.grid(True);  ax.axis('equal')

    # -- Horizontal position tracking --
    ax = fig2.add_subplot(2, 3, 3)
    ax.plot(t_hist, x_hist, 'b-',  linewidth=lw, label='X actual')
    ax.plot(t_hist, x_ref,  'r--', linewidth=lw, label='X ref')
    ax.plot(t_hist, y_hist, 'g-',  linewidth=lw, label='Y actual')
    ax.plot(t_hist, y_ref,  'm--', linewidth=lw, label='Y ref')
    ax.set_xlabel('Time (s)');  ax.set_ylabel('Position (m)')
    ax.set_title('Horizontal Position Tracking')
    ax.legend(fontsize=8);  ax.grid(True)

    # -- Altitude tracking --
    ax = fig2.add_subplot(2, 3, 4)
    ax.plot(t_hist, z_hist,      'b-',  linewidth=lw, label='Z actual')
    ax.plot(t_hist, z_ref_hist,  'r--', linewidth=lw, label='Z ref')
    ax.set_xlabel('Time (s)');  ax.set_ylabel('Altitude (m)')
    ax.set_title('Altitude Tracking')
    ax.legend(fontsize=8);  ax.grid(True)

    # -- Position tracking error --
    ax = fig2.add_subplot(2, 3, 5)
    ax.plot(t_hist, pos_err, 'r-', linewidth=lw)
    ax.set_xlabel('Time (s)');  ax.set_ylabel('Error (m)')
    ax.set_title('Position Tracking Error')
    ax.grid(True)

    # -- Attitude angles --
    ax = fig2.add_subplot(2, 3, 6)
    ax.plot(t_hist, phi_hist,   'b-', linewidth=lw, label='Roll (Phi)')
    ax.plot(t_hist, theta_hist, 'g-', linewidth=lw, label='Pitch (Theta)')
    ax.plot(t_hist, psi_hist,   'r-', linewidth=lw, label='Yaw (Psi)')
    ax.set_xlabel('Time (s)');  ax.set_ylabel('Angle (deg)')
    ax.set_title('Attitude Angles')
    ax.legend(fontsize=8);  ax.grid(True)

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
    A     = trajectory['A'];      B     = trajectory['B']
    omega = trajectory['omega'];  z_ref = trajectory['z_ref']
    t_takeoff = trajectory.get('t_takeoff', 8.0)
    t_land    = trajectory.get('t_land',    8.0)
    t_sim     = trajectory.get('t_sim', trajectory.get('t_end', 90.0))
    t_land_start = t_sim - t_land

    t_hist = quad.data('x')[0]
    x_hist = quad.data('x')[1]
    y_hist = quad.data('y')[1]
    z_hist = quad.data('z')[1]

    # Figure-8 phase indices
    idx  = (t_hist >= t_takeoff) & (t_hist <= t_land_start)
    t_ss = t_hist[idx]

    ref_ss   = np.array([get_reference(t, A, B, omega, z_ref, t_takeoff, t_land, t_sim)
                         for t in t_ss])
    x_ref_ss = ref_ss[:,0];  y_ref_ss = ref_ss[:,1]
    z_ref_ss = np.full(len(t_ss), z_ref)

    rmse_x    = np.sqrt(np.mean((x_hist[idx] - x_ref_ss)**2))
    rmse_y    = np.sqrt(np.mean((y_hist[idx] - y_ref_ss)**2))
    rmse_z    = np.sqrt(np.mean((z_hist[idx] - z_ref_ss)**2))
    norm_x    = rmse_x / A     * 100
    norm_y    = rmse_y / B     * 100
    norm_z    = rmse_z / z_ref * 100
    max_z_dev = np.max(np.abs(z_hist[idx] - z_ref))

    print('=' * 45)
    print('   TRACKING PERFORMANCE METRICS')
    print('=' * 45)
    print(f'  RMSE x : {rmse_x:.4f} m  ({norm_x:.1f}% of X amplitude)')
    print(f'  RMSE y : {rmse_y:.4f} m  ({norm_y:.1f}% of Y amplitude)')
    print(f'  RMSE z : {rmse_z:.6f} m  ({norm_z:.3f}% of altitude)')
    print(f'  Max altitude deviation : {max_z_dev*100:.2f} cm')
    print('=' * 45)

    return {'rmse_x': rmse_x, 'rmse_y': rmse_y, 'rmse_z': rmse_z,
            'norm_x': norm_x, 'norm_y': norm_y, 'norm_z': norm_z,
            'max_z_dev': max_z_dev}
