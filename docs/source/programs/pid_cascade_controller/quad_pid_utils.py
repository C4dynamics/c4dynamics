import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


def compute_metrics(): pass
def get_reference(): pass
def get_reference_velocity(): pass


class QuadcopterPlant:
    """Quadcopter 12-state nonlinear dynamics model."""

    def __init__(self, params):
        """
        Initialize quadcopter plant with parameters.

        params : dict or module
            Dictionary/module containing physical parameters:
            - m, g, L, K_thrust, B_torque, IR
            - IXX, IYY, IZZ                         INERTIA MATRIX
            - Ax, Ay, Az, Ar                        AERODYNAMIC DRAG COEFFICIENTS
        """
        self.m = params['m']
        self.g = params['g']
        self.L = params['l']
        self.K_thrust = params['kT']
        self.B_torque = params['kQ']
        self.IR = params['IR']

        self.IXX = params['Ixx']
        self.IYY = params['Iyy']
        self.IZZ = params['Izz']

        self.Ax = params['Ax']
        self.Ay = params['Ay']
        self.Az = params['Az']
        self.Ar = params['Ar']


    def derivatives(self, state, rotor_speeds):
        """
        Compute state derivatives given current state and rotor speeds.

        state : array
            [P, Q, R, Phi, Theta, Psi, U, V, W, X, Y, Z]
        rotor_speeds : array
            [w1, w2, w3, w4] in rad/s

        Returns
        -------
        dstate : array
            Derivatives of state vector
        """

        P, Q, R, Phi, Theta, Psi, U, V, W, X, Y, Z = state
        w1, w2, w3, w4 = rotor_speeds

        # =============================================
        # CALCULATE THRUST AND TORQUES
        # =============================================

        T1 = self.K_thrust * w1**2
        T2 = self.K_thrust * w2**2
        T3 = self.K_thrust * w3**2
        T4 = self.K_thrust * w4**2

        T = T1 + T2 + T3 + T4      # total thrust
        M_phi = self.L * (T4 - T2) # roll torque
        M_theta = self.L * (T3 - T1) # pitch torque
        M_psi = self.B_torque * (-T1 + T2 - T3 + T4)  # yaw torque

        Omega = w1 - w2 + w3 - w4  # rotor speed sum for gyro coupling

        # =============================================
        # ANGULAR ACCELERATIONS (P, Q, R)
        # =============================================

        dP = ((self.IYY - self.IZZ) / self.IXX) * Q * R \
           - (self.IR / self.IXX) * Q * Omega \
           + M_phi / self.IXX \
           - (self.Ar / self.IXX) * P

        dQ = ((self.IZZ - self.IXX) / self.IYY) * P * R \
           + (self.IR / self.IYY) * P * Omega \
           + M_theta / self.IYY \
           - (self.Ar / self.IYY) * Q

        dR = ((self.IXX - self.IYY) / self.IZZ) * P * Q \
           + M_psi / self.IZZ \
           - (self.Ar / self.IZZ) * R

        # =============================================
        # EULER ANGLE RATES (Phi, Theta, Psi)
        # =============================================

        dPhi = P + np.sin(Phi) * np.tan(Theta) * Q + np.cos(Phi) * np.tan(Theta) * R
        dTheta = np.cos(Phi) * Q - np.sin(Phi) * R
        dPsi = np.sin(Phi) / np.cos(Theta) * Q + np.cos(Phi) / np.cos(Theta) * R

        # =============================================
        # BODY VELOCITY ACCELERATIONS (U, V, W)
        # =============================================

        dU = (np.sin(Phi) * np.sin(Psi) + np.cos(Phi) * np.sin(Theta) * np.cos(Psi)) * T / self.m \
           - (self.Ax / self.m) * U

        dV = (-np.sin(Phi) * np.cos(Psi) + np.cos(Phi) * np.sin(Theta) * np.sin(Psi)) * T / self.m \
           - (self.Ay / self.m) * V

        dW = -self.g + (np.cos(Phi) * np.cos(Theta)) * T / self.m - (self.Az / self.m) * W

        # MATLAB code: dX = U; dY = V; dZ = W;
        # These equations treat U, V, W as INERTIAL frame velocities
        # But for the controller to work, they should be BODY frame velocities
        # Convert inertial velocities to body velocities
        # Inverse of: Vinertial = DCM * Vbody
        # U_inertial = u*cos(θ)cos(ψ) + v*(sin(φ)sin(θ)cos(ψ) - cos(φ)sin(ψ)) + w*(cos(φ)sin(θ)cos(ψ) + sin(φ)sin(ψ))
        # For simplicity and controller compatibility, treat dX = U, dY = V, dZ = W
        # where U, V, W in the state are actually BODY frame velocities
        # This requires changing how we compute dU, dV, dW as BODY frame accelerations

        # Standard quadcopter body-frame dynamics for velocity:
        # In body frame, with thrust pointing in body Z direction:
        # The thrust T contributes only to du/dt in the magnitude of the copter in inertial coordinates
        # But we need to account for gravity in body frame

        # For body-frame velocity integration:
        dX = U
        dY = V
        dZ = W

        return np.array([dP, dQ, dR, dPhi, dTheta, dPsi, dU, dV, dW, dX, dY, dZ])

    def simulate(self, state_init, rotor_speeds_func, t_span, dt):
        """
        Simulate quadcopter dynamics.

        state_init : array
            Initial state [P, Q, R, Phi, Theta, Psi, U, V, W, X, Y, Z]
        rotor_speeds_func : callable
            Function that takes time and returns rotor speeds [w1, w2, w3, w4]
        t_span : tuple
            (t_start, t_end) simulation time
        dt : float
            Integration time step

        Returns
        -------
        t : array
            Time vector
        states : array
            State history, shape (N, 12)
        """

        t_start, t_end = t_span
        t = np.arange(t_start, t_end + dt, dt)
        N = len(t)

        states = np.zeros((N, 12))
        states[0] = state_init

        # Simple Euler integration (can upgrade to RK45 if needed)
        for i in range(N - 1):
            rotor_speeds = rotor_speeds_func(t[i])
            deriv = self.derivatives(states[i], rotor_speeds)
            states[i + 1] = states[i] + deriv * dt

        return t, states


class OuterPositionPID:
    """Outer loop position controller (50 Hz)."""

    def __init__(self, params, m, g, K_thrust):
        """Initialize outer position controller."""
        self.g = g
        self.m = m

        # Z gains
        self.KP_Z = params['Kp_z']
        self.KI_Z = params['Ki_z']
        self.KD_Z = params['Kd_z']

        # X gains
        self.KP_X = params['Kp_x']
        self.KI_X = params['Ki_x']
        self.KD_X = params['Kd_x']

        # Y gains
        self.KP_Y = params['Kp_y']
        self.KI_Y = params['Ki_y']
        self.KD_Y = params['Kd_y']

        # Anti-windup limits
        self.AW_Z = params['AW_z']
        self.AW_X = params['AW_x']
        self.AW_Y = params['AW_y']

        # Saturations
        self.T_max = params['T_max_factor'] * K_thrust * params['omega_max']**2
        self.T_min = params['T_min']
        self.att_cmd_limit = params['att_cmd_limit']

        # Feedforward gains
        self.FF_X = params['Kff_x']
        self.FF_Y = params['Kff_y']

        # Persistent states
        self.int_Z = 0.0
        self.int_X = 0.0
        self.int_Y = 0.0
        self.Xd_prev = 0.0
        self.Yd_prev = 0.0

    def compute(self, Xd, Yd, Zd, Psi_setpoint, X, Y, Z, U, V, W, Phi, Theta, Psi, Ts):
        """
        Compute thrust command and attitude setpoints.

        Returns
        -------
        T_cmd : float
            Thrust command [N]
        Phi_d : float
            Desired roll angle [rad]
        Theta_d : float
            Desired pitch angle [rad]
        Psi_d_out : float
            Desired yaw angle [rad]
        """

        # =============================================
        # ALTITUDE CONTROLLER (Z axis)
        # =============================================

        e_Z = Zd - Z

        # Integrator update
        self.int_Z += Ts * e_Z
        # Anti-windup clamping
        self.int_Z = np.clip(self.int_Z, -self.AW_Z, self.AW_Z)

        # Derivative on measurement
        dZ_meas = -W

        # PID output
        az_cmd = self.KP_Z * e_Z + self.KI_Z * self.int_Z + self.KD_Z * dZ_meas

        # Thrust command with gravity feedforward
        cos_corr = np.cos(Phi) * np.cos(Theta)
        cos_corr = max(0.5, cos_corr)
        T_raw = self.m * (self.g + az_cmd) / cos_corr

        # Saturate thrust
        T_cmd = np.clip(T_raw, self.T_min, self.T_max)

        # =============================================
        # HORIZONTAL POSITION CONTROLLER (X, Y)
        # =============================================

        # Inertial position errors
        e_X_inertial = Xd - X
        e_Y_inertial = Yd - Y

        # Rotate to body frame using current yaw
        e_X_body = e_X_inertial * np.cos(Psi) + e_Y_inertial * np.sin(Psi)
        e_Y_body = e_Y_inertial * np.cos(Psi) - e_X_inertial * np.sin(Psi)

        # Desired body velocities
        Ud_desired = e_X_body
        Vd_desired = e_Y_body

        # Velocity errors
        # States U,V are used as inertial rates in the plant model, so convert
        # to body-frame velocities before comparing against body-frame setpoints.
        U_body = U * np.cos(Psi) + V * np.sin(Psi)
        V_body = -U * np.sin(Psi) + V * np.cos(Psi)
        e_U = Ud_desired - U_body
        e_V = Vd_desired - V_body

        # Integrator update with anti-windup
        self.int_X += Ts * e_U
        self.int_Y += Ts * e_V
        self.int_X = np.clip(self.int_X, -self.AW_X, self.AW_X)
        self.int_Y = np.clip(self.int_Y, -self.AW_Y, self.AW_Y)

        # Derivative on measurement
        dU_meas = -U_body
        dV_meas = -V_body

        # Feedforward calculation
        Xd_dot_ref = (Xd - self.Xd_prev) / Ts if Ts > 0 else 0.0
        Yd_dot_ref = (Yd - self.Yd_prev) / Ts if Ts > 0 else 0.0

        # Rotate reference velocity from inertial to body frame
        Xd_dot_body = Xd_dot_ref * np.cos(Psi) + Yd_dot_ref * np.sin(Psi)
        Yd_dot_body = Yd_dot_ref * np.cos(Psi) - Xd_dot_ref * np.sin(Psi)

        # Feedforward attitude commands
        ff_theta = self.FF_X * Xd_dot_body
        ff_phi = -self.FF_Y * Yd_dot_body

        # =============================================
        # PID + FEEDFORWARD COMBINED OUTPUT
        # =============================================

        acmd_X = self.KP_X * e_U + self.KI_X * self.int_X + self.KD_X * dU_meas
        acmd_Y = self.KP_Y * e_V + self.KI_Y * self.int_Y + self.KD_Y * dV_meas

        Theta_d_raw = acmd_X + ff_theta
        Phi_d_raw = -(acmd_Y) + ff_phi

        # Saturate attitude commands
        Theta_d = np.clip(Theta_d_raw, -self.att_cmd_limit, self.att_cmd_limit)
        Phi_d = np.clip(Phi_d_raw, -self.att_cmd_limit, self.att_cmd_limit)
        Psi_d_out = Psi_setpoint

        # Store for feedforward next step
        self.Xd_prev = Xd
        self.Yd_prev = Yd

        return T_cmd, Phi_d, Theta_d, Psi_d_out


class MiddleAttitudePID:
    """Middle loop attitude controller (100 Hz)."""

    def __init__(self, params):
        """Initialize middle attitude controller."""

        # Roll gains
        self.KP_Phi = params['Kp_phi']
        self.KI_Phi = params['Ki_phi']
        self.KD_Phi = params['Kd_phi']

        # Pitch gains
        self.KP_Theta = params['Kp_theta']
        self.KI_Theta = params['Ki_theta']
        self.KD_Theta = params['Kd_theta']

        # Yaw gains
        self.KP_Psi = params['Kp_psi']
        self.KI_Psi = params['Ki_psi']
        self.KD_Psi = params['Kd_psi']

        # Anti-windup limits
        self.AW_Phi = params['AW_phi']
        self.AW_Theta = params['AW_theta']
        self.AW_Psi = params['AW_psi']

        self.yaw_rate_limit = params['yaw_rate_limit']

        # Persistent states
        self.int_Phi = 0.0
        self.int_Theta = 0.0
        self.int_Psi = 0.0


    def compute(self, Phi_d, Theta_d, Psi_d, Phi, Theta, Psi, P, Q, R, Ts):
        """
        Compute desired angular rates.

        Returns
        -------
        Pd, Qd, Rd : float
            Desired angular rates [rad/s]
        """

        # =============================================
        # ANGLE ERRORS
        # =============================================

        e_Phi = Phi_d - Phi
        e_Theta = Theta_d - Theta

        # Wrap yaw error to [-pi, pi]
        e_Psi_raw = Psi_d - Psi
        e_Psi = np.arctan2(np.sin(e_Psi_raw), np.cos(e_Psi_raw))

        # =============================================
        # INTEGRATOR UPDATE
        # =============================================

        self.int_Phi += Ts * e_Phi
        self.int_Theta += Ts * e_Theta
        self.int_Psi += Ts * e_Psi

        # Anti-windup clamping
        self.int_Phi = np.clip(self.int_Phi, -self.AW_Phi / self.KI_Phi, self.AW_Phi / self.KI_Phi)
        self.int_Theta = np.clip(self.int_Theta, -self.AW_Theta / self.KI_Theta, self.AW_Theta / self.KI_Theta)
        self.int_Psi = np.clip(self.int_Psi, -self.AW_Psi / self.KI_Psi, self.AW_Psi / self.KI_Psi)

        # =============================================
        # DERIVATIVE ON MEASUREMENT
        # Uses actual rates P, Q, R instead of differentiating error
        # =============================================

        dPhi_meas = -P
        dTheta_meas = -Q
        dPsi_meas = -R

        # =============================================
        # PID OUTPUT — desired angular rates
        # =============================================

        rate_lim = self.yaw_rate_limit * 3

        Pd_raw = self.KP_Phi * e_Phi + self.KI_Phi * self.int_Phi + self.KD_Phi * dPhi_meas
        Qd_raw = self.KP_Theta * e_Theta + self.KI_Theta * self.int_Theta + self.KD_Theta * dTheta_meas
        Rd_raw = self.KP_Psi * e_Psi + self.KI_Psi * self.int_Psi + self.KD_Psi * dPsi_meas

        # Saturate output rates
        Pd = np.clip(Pd_raw, -rate_lim, rate_lim)
        Qd = np.clip(Qd_raw, -rate_lim, rate_lim)
        Rd = np.clip(Rd_raw, -self.yaw_rate_limit, self.yaw_rate_limit)

        return Pd, Qd, Rd


class InnerRatePID:
    """Inner loop rate controller (200 Hz)."""

    def __init__(self, params, IXX, IYY, IZZ, L, K_thrust):
        """Initialize inner rate controller."""

        # Roll rate gains
        self.KP_P = params['Kp_p']
        self.KI_P = params['Ki_p']
        self.KD_P = params['Kd_p']

        # Pitch rate gains
        self.KP_Q = params['Kp_q']
        self.KI_Q = params['Ki_q']
        self.KD_Q = params['Kd_q']

        # Yaw rate gains
        self.KP_R = params['Kp_r']
        self.KI_R = params['Ki_r']
        self.KD_R = params['Kd_r']

        self.N_rate = params['N_rate']

        self.M_max = L * K_thrust * params['omega_max']**2

        # Inertias
        self.IXX = IXX
        self.IYY = IYY
        self.IZZ = IZZ

        # Persistent states
        self.int_P = 0.0
        self.int_Q = 0.0
        self.int_R = 0.0
        self.prev_eP = 0.0
        self.prev_eQ = 0.0
        self.prev_eR = 0.0


    def compute(self, Pd, Qd, Rd, P, Q, R, Ts):
        """
        Compute torque commands.

        Returns
        -------
        M_Phi, M_Theta, M_Psi : float
            Torque commands [N.m]
        """

        # =============================================
        # RATE ERRORS
        # =============================================

        eP = Pd - P
        eQ = Qd - Q
        eR = Rd - R

        # =============================================
        # INTEGRATOR UPDATE (Tustin)
        # =============================================

        self.int_P += (Ts / 2) * (eP + self.prev_eP)
        self.int_Q += (Ts / 2) * (eQ + self.prev_eQ)
        self.int_R += (Ts / 2) * (eR + self.prev_eR)

        # =============================================
        # DERIVATIVE WITH FILTER
        # =============================================

        denom_P = 1 + self.N_rate * Ts
        denom_Q = 1 + self.N_rate * Ts
        denom_R = 1 + self.N_rate * Ts

        dP = self.N_rate * (eP - self.prev_eP) / denom_P
        dQ = self.N_rate * (eQ - self.prev_eQ) / denom_Q
        dR = self.N_rate * (eR - self.prev_eR) / denom_R

        # =============================================
        # PID OUTPUT — torque commands
        # Multiply by inertia for physical units [N.m]
        # =============================================

        M_Phi_raw = self.IXX * (self.KP_P * eP + self.KI_P * self.int_P + self.KD_P * dP)
        M_Theta_raw = self.IYY * (self.KP_Q * eQ + self.KI_Q * self.int_Q + self.KD_Q * dQ)
        M_Psi_raw = self.IZZ * (self.KP_R * eR + self.KI_R * self.int_R + self.KD_R * dR)

        # Saturate torques
        M_Phi = np.clip(M_Phi_raw, -self.M_max, self.M_max)
        M_Theta = np.clip(M_Theta_raw, -self.M_max, self.M_max)
        M_Psi = np.clip(M_Psi_raw, -self.M_max, self.M_max)

        # Back-calculation anti-windup
        AW_gain = 0.1
        self.int_P += AW_gain * (M_Phi - M_Phi_raw) / (self.IXX * self.KI_P + 1e-9)
        self.int_Q += AW_gain * (M_Theta - M_Theta_raw) / (self.IYY * self.KI_Q + 1e-9)
        self.int_R += AW_gain * (M_Psi - M_Psi_raw) / (self.IZZ * self.KI_R + 1e-9)

        # =============================================
        # UPDATE MEMORY
        # =============================================

        self.prev_eP = eP
        self.prev_eQ = eQ
        self.prev_eR = eR

        return M_Phi, M_Theta, M_Psi


class ControlAllocator:
    """Control allocator: thrust + torques → motor speeds."""

    def __init__(self, K_thrust, B_torque, L, omega_max):
        """Initialize control allocator."""
        self.K_thrust = K_thrust
        self.B_torque = B_torque
        self.L = L
        self.omega_sq_min = 0.0           # min rotor speed squared [(rad/s)^2]
        self.omega_sq_max = omega_max**2  # max rotor speed squared [(rad/s)^2]



    def allocate(self, T_cmd, M_Phi, M_Theta, M_Psi):
        """
        Convert thrust and torques to motor speeds.

        Parameters
        ----------
        T_cmd : float
            Thrust command [N]
        M_Phi : float
            Roll torque command [N.m]
        M_Theta : float
            Pitch torque command [N.m]
        M_Psi : float
            Yaw torque command [N.m]

        Returns
        -------
        w1, w2, w3, w4 : float
            Rotor speeds [rad/s]
        """

        # =============================================
        # STEP 1: Compute omega squared (Eq. 3.24)
        # =============================================

        T_over_4K = T_cmd / (4 * self.K_thrust)
        M_theta_term = M_Theta / (2 * self.K_thrust * self.L)
        M_phi_term = M_Phi / (2 * self.K_thrust * self.L)
        M_psi_term = M_Psi / (4 * self.B_torque)

        w1_sq = T_over_4K - M_theta_term - M_psi_term
        w2_sq = T_over_4K - M_phi_term + M_psi_term
        w3_sq = T_over_4K + M_theta_term - M_psi_term
        w4_sq = T_over_4K + M_phi_term + M_psi_term

        # =============================================
        # STEP 2: Physical saturation
        # omega squared cannot be negative
        # =============================================

        w1_sq = np.clip(w1_sq, self.omega_sq_min, self.omega_sq_max)
        w2_sq = np.clip(w2_sq, self.omega_sq_min, self.omega_sq_max)
        w3_sq = np.clip(w3_sq, self.omega_sq_min, self.omega_sq_max)
        w4_sq = np.clip(w4_sq, self.omega_sq_min, self.omega_sq_max)

        # =============================================
        # STEP 3: Square root to get rotor speeds
        # =============================================

        w1 = np.sqrt(w1_sq)
        w2 = np.sqrt(w2_sq)
        w3 = np.sqrt(w3_sq)
        w4 = np.sqrt(w4_sq)

        return w1, w2, w3, w4


def trajectory_generator(params):


  t_sim = params['t_end']
  Z_hover = params['z_ref']
  A_fig8 = params['A']
  B_fig8 = params['B']
  w_fig8 = params['omega']


  n_points = 10000
  t_ref = np.linspace(0.0, t_sim, n_points)
  N_ref = len(t_ref)

  # ========================================================================
  # 14. FIGURE-8 TRAJECTORY REFERENCE
  # ========================================================================

  # Figure-8 parameters
  # A_fig8 = 4.0             # X amplitude [m]
  # B_fig8 = 2.0             # Y amplitude [m]
  # w_fig8 = 0.1             # angular frequency [rad/s]
  #                           # One cycle = 2π/0.1 ≈ 62.8 seconds

  # Phase durations
  t_takeoff_duration = 8.0   # seconds to rise to hover altitude
  t_land_duration = 8.0      # seconds to descend

  # Generate reference trajectory (exact MATLAB stage behavior)
  t_land_start = t_sim - t_land_duration

  Xd_traj = np.zeros_like(t_ref)
  Yd_traj = np.zeros_like(t_ref)
  Zd_traj = np.zeros_like(t_ref)

  for i, t_val in enumerate(t_ref):
      if t_val <= t_takeoff_duration:
          # PHASE 1: Takeoff using smooth S-curve (zero slope at endpoints)
          frac = t_val / t_takeoff_duration
          smooth_frac = 3.0 * frac**2 - 2.0 * frac**3
          Xd_traj[i] = 0.0
          Yd_traj[i] = 0.0
          Zd_traj[i] = Z_hover * smooth_frac

      elif t_val <= t_land_start:
          # PHASE 2: Figure-8 with phase origin at end of takeoff
          tau = t_val - t_takeoff_duration
          Xd_traj[i] = A_fig8 * np.sin(w_fig8 * tau)
          Yd_traj[i] = B_fig8 * np.sin(2.0 * w_fig8 * tau)
          Zd_traj[i] = Z_hover

      else:
          # PHASE 3: Smooth landing and smooth XY return to origin
          frac = (t_val - t_land_start) / t_land_duration
          smooth_frac = 3.0 * frac**2 - 2.0 * frac**3

          tau_land = t_land_start - t_takeoff_duration
          x_land_start = A_fig8 * np.sin(w_fig8 * tau_land)
          y_land_start = B_fig8 * np.sin(2.0 * w_fig8 * tau_land)

          Xd_traj[i] = x_land_start * (1.0 - smooth_frac)
          Yd_traj[i] = y_land_start * (1.0 - smooth_frac)
          Zd_traj[i] = Z_hover * (1.0 - smooth_frac)


  return t_ref, Xd_traj, Yd_traj, Zd_traj


def run_fig8_pid(vehicle, trajectory, controller, sim):
    """Run complete quadcopter cascade PID simulation."""

    print("Initializing Quadcopter Cascade PID Simulation...")

    # ====================================================================
    # 1. INITIALIZE PLANT AND CONTROLLERS
    # ====================================================================

    plant = QuadcopterPlant(vehicle)
    outer_controller = OuterPositionPID(controller,
                                        vehicle['m'],
                                        vehicle['g'],
                                        vehicle['kT'])
    middle_controller = MiddleAttitudePID(controller)
    inner_controller = InnerRatePID(controller,
                                    vehicle['Ixx'],
                                    vehicle['Iyy'],
                                    vehicle['Izz'],
                                    vehicle['l'],
                                    vehicle['kT'])
    allocator = ControlAllocator(vehicle['kT'],
                                 vehicle['kQ'],
                                 vehicle['l'],
                                 controller['omega_max'])

    # ====================================================================
    # 2. SETUP SIMULATION TIME AND REFERENCE TRAJECTORY
    # ====================================================================

    dt = sim.get('dt')
    t_sim = sim.get('t_sim')
    t = np.arange(0, t_sim + dt, dt)
    N = len(t)

    print(f"Simulation time: {t_sim} s, dt: {dt} s, N: {N} steps")


    t_ref, Xd_traj, Yd_traj, Zd_traj = trajectory_generator(trajectory)

    # Interpolate reference trajectory to simulation time grid
    interp_X = interp1d(t_ref, Xd_traj, kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_Y = interp1d(t_ref, Yd_traj, kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_Z = interp1d(t_ref, Zd_traj, kind='linear', bounds_error=False, fill_value='extrapolate')

    Xd_sim = interp_X(t)
    Yd_sim = interp_Y(t)
    Zd_sim = interp_Z(t)

    # ========================================================================
    # INITIAL CONDITIONS
    # Quadcopter starts on ground, stationary, flat
    # ========================================================================

    X0     = 0.0
    Y0     = 0.0
    Z0     = 0.0
    U0     = 0.0
    V0     = 0.0
    W0     = 0.0
    Phi0   = 0.0
    Theta0 = 0.0
    Psi0   = 0.0
    P0     = 0.0
    Q0     = 0.0
    R0     = 0.0

    IC = np.array([P0, Q0, R0, Phi0, Theta0, Psi0, U0, V0, W0, X0, Y0, Z0])


    # ========================================================================
    # 12. CONTROL LOOP RATES
    # ========================================================================

    f_outer  = 50.0          # Outer loop frequency [Hz]
    f_middle = 100.0         # Middle loop frequency [Hz]
    f_inner  = 200.0         # Inner loop frequency [Hz]

    Ts_outer  = 1.0 / f_outer   # 0.020 s
    Ts_middle = 1.0 / f_middle  # 0.010 s
    Ts_inner  = 1.0 / f_inner   # 0.005 s

    # ====================================================================
    # 3. INITIALIZE STATE AND OUTPUT ARRAYS
    # ====================================================================

    state = IC.copy()
    states_log = np.zeros((N, 12))
    states_log[0] = state

    # Log control signals
    T_log = np.zeros(N)
    Phi_d_log = np.zeros(N)
    Theta_d_log = np.zeros(N)
    Psi_d_log = np.zeros(N)
    Pd_log = np.zeros(N)
    Qd_log = np.zeros(N)
    Rd_log = np.zeros(N)
    M_Phi_log = np.zeros(N)
    M_Theta_log = np.zeros(N)
    M_Psi_log = np.zeros(N)
    w1_log = np.zeros(N)
    w2_log = np.zeros(N)
    w3_log = np.zeros(N)
    w4_log = np.zeros(N)

    # ====================================================================
    # 4. MAIN SIMULATION LOOP
    # ====================================================================

    # Counters for cascade control rates (in seconds)
    outer_time = 0.0
    middle_time = 0.0
    inner_time = 0.0

    # Initialize state
    state = IC.copy()
    P, Q, R, Phi, Theta, Psi, U, V, W, X, Y, Z = state


    Psi_d    = 0.0            # desired yaw angle [rad] — fixed

    # ====================================================================
    # INITIALIZE CONTROLLERS WITH FIRST CALL
    # ====================================================================

    Xd = Xd_sim[0]
    Yd = Yd_sim[0]
    Zd = Zd_sim[0]


    T_cmd, Phi_d, Theta_d, Psi_d_out = outer_controller.compute(
        Xd, Yd, Zd, Psi_d, X, Y, Z, U, V, W, Phi, Theta, Psi, Ts_outer
    )

    Pd, Qd, Rd = middle_controller.compute(
        Phi_d, Theta_d, Psi_d_out, Phi, Theta, Psi, P, Q, R, Ts_middle
    )

    M_Phi, M_Theta, M_Psi = inner_controller.compute(Pd, Qd, Rd, P, Q, R, dt)
    w1, w2, w3, w4 = allocator.allocate(T_cmd, M_Phi, M_Theta, M_Psi)




    print(f"\n=== INITIALIZATION ===" )
    print(f"Initial state: Z={Z:.3f}, Phi={np.rad2deg(Phi):.2f}°, Theta={np.rad2deg(Theta):.2f}°")
    print(f"Reference: Zd={Zd:.3f}, Xd={Xd:.3f}, Yd={Yd:.3f}")
    print(f"Thrust command: T_cmd={T_cmd:.3f} N (max {outer_controller.T_max})")
    print(f"Desired attitude: Phi_d={np.rad2deg(Phi_d):.2f}°, Theta_d={np.rad2deg(Theta_d):.2f}°")
    print(f"Desired rates: Pd={np.rad2deg(Pd):.2f}°/s, Qd={np.rad2deg(Qd):.2f}°/s")
    print(f"Motor speeds: w1={w1:.1f}, w2={w2:.1f}, w3={w3:.1f}, w4={w4:.1f} rad/s\n")

    # ====================================================================
    # MAIN SIMULATION LOOP
    # ====================================================================

    for i in range(N - 1):
        time = t[i]

        # Unpack state
        P, Q, R, Phi, Theta, Psi, U, V, W, X, Y, Z = state

        # Reference trajectory at this time step
        Xd = Xd_sim[i]
        Yd = Yd_sim[i]
        Zd = Zd_sim[i]

        # ====================================================================
        # OUTER LOOP  (50 Hz = Ts_outer = 0.02 s)
        # ====================================================================

        outer_time += dt
        if outer_time >= Ts_outer:
            T_cmd, Phi_d, Theta_d, Psi_d_out = outer_controller.compute(
                Xd, Yd, Zd, Psi_d,
                X, Y, Z, U, V, W, Phi, Theta, Psi,
                Ts_outer
            )
            outer_time = 0.0

        # ====================================================================
        # MIDDLE LOOP (100 Hz = Ts_middle = 0.01 s)
        # ====================================================================

        middle_time += dt
        if middle_time >= Ts_middle:
            Pd, Qd, Rd = middle_controller.compute(
                Phi_d, Theta_d, Psi_d_out,
                Phi, Theta, Psi, P, Q, R,
                Ts_middle
            )
            middle_time = 0.0

        # ====================================================================
        # INNER LOOP (200 Hz = dt = 0.005 s) — always runs
        # ====================================================================

        M_Phi, M_Theta, M_Psi = inner_controller.compute(
            Pd, Qd, Rd,
            P, Q, R,
            dt
        )

        # ====================================================================
        # CONTROL ALLOCATOR — runs every dt
        # ====================================================================

        w1, w2, w3, w4 = allocator.allocate(T_cmd, M_Phi, M_Theta, M_Psi)

        # ====================================================================
        # LOG SIGNALS
        # ====================================================================

        T_log[i] = T_cmd
        Phi_d_log[i] = Phi_d
        Theta_d_log[i] = Theta_d
        Psi_d_log[i] = Psi_d_out
        Pd_log[i] = Pd
        Qd_log[i] = Qd
        Rd_log[i] = Rd
        M_Phi_log[i] = M_Phi
        M_Theta_log[i] = M_Theta
        M_Psi_log[i] = M_Psi
        w1_log[i] = w1
        w2_log[i] = w2
        w3_log[i] = w3
        w4_log[i] = w4

        # ====================================================================
        # PLANT DYNAMICS INTEGRATION (RK4)
        # ====================================================================

        rotor_speeds = np.array([w1, w2, w3, w4])
        integration_substeps = 5
        h = dt / integration_substeps
        for _ in range(integration_substeps):
            k1 = plant.derivatives(state, rotor_speeds)
            k2 = plant.derivatives(state + 0.5 * h * k1, rotor_speeds)
            k3 = plant.derivatives(state + 0.5 * h * k2, rotor_speeds)
            k4 = plant.derivatives(state + h * k3, rotor_speeds)
            state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        states_log[i + 1] = state

    print("Simulation complete!")

    # ====================================================================
    # 5. RETURN RESULTS
    # ====================================================================

    results = {
        't': t,
        'states': states_log,
        'Xd': Xd_sim,
        'Yd': Yd_sim,
        'Zd': Zd_sim,
        'T_cmd': T_log,
        'Phi_d': Phi_d_log,
        'Theta_d': Theta_d_log,
        'Psi_d': Psi_d_log,
        'Pd': Pd_log,
        'Qd': Qd_log,
        'Rd': Rd_log,
        'M_Phi': M_Phi_log,
        'M_Theta': M_Theta_log,
        'M_Psi': M_Psi_log,
        'w1': w1_log,
        'w2': w2_log,
        'w3': w3_log,
        'w4': w4_log,
    }

    return results


def plot_results(results):
    """Plot simulation results."""

    t = results['t']
    states = results['states']
    Xd = results['Xd']
    Yd = results['Yd']
    Zd = results['Zd']

    # Unpack states
    Phi = states[:, 3]
    Theta = states[:, 4]
    Psi = states[:, 5]
    X = states[:, 9]
    Y = states[:, 10]
    Z = states[:, 11]

    # Compute tracking errors
    eX = Xd - X
    eY = Yd - Y
    eZ = Zd - Z
    pos_error = np.sqrt(eX**2 + eY**2 + eZ**2)

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Cascade PID Quadcopter — Simulation Results', fontsize=16, fontweight='bold')

    # 3D trajectory
    ax = fig.add_subplot(2, 3, 1, projection='3d')
    ax.plot(X, Y, Z, 'b-', linewidth=2, label='Actual')
    ax.plot(Xd, Yd, Zd, 'r--', linewidth=2, label='Reference')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory')
    ax.legend()
    ax.grid(True)

    # XY plane
    ax = fig.add_subplot(2, 3, 2)
    ax.plot(X, Y, 'b-', linewidth=2, label='Actual')
    ax.plot(Xd, Yd, 'r--', linewidth=2, label='Reference')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('XY Plane')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')

    # Position tracking
    ax = fig.add_subplot(2, 3, 3)
    ax.plot(t, X, 'b-', label='X actual', linewidth=1.5)
    ax.plot(t, Xd, 'r--', label='X ref', linewidth=1)
    ax.plot(t, Y, 'g-', label='Y actual', linewidth=1.5)
    ax.plot(t, Yd, 'm--', label='Y ref', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.set_title('Horizontal Position Tracking')
    ax.legend()
    ax.grid(True)

    # Altitude tracking
    ax = fig.add_subplot(2, 3, 4)
    ax.plot(t, Z, 'b-', label='Z actual', linewidth=2)
    ax.plot(t, Zd, 'r--', label='Z ref', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Altitude Tracking')
    ax.legend()
    ax.grid(True)

    # Position error
    ax = fig.add_subplot(2, 3, 5)
    ax.plot(t, pos_error, 'r-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (m)')
    ax.set_title('Position Tracking Error')
    ax.grid(True)

    # Attitude angles
    ax = fig.add_subplot(2, 3, 6)
    ax.plot(t, np.rad2deg(Phi), 'b-', label='Roll (Phi)', linewidth=1.5)
    ax.plot(t, np.rad2deg(Theta), 'g-', label='Pitch (Theta)', linewidth=1.5)
    ax.plot(t, np.rad2deg(Psi), 'r-', label='Yaw (Psi)', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.set_title('Attitude Angles')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

    # Print statistics
    print("\n" + "="*60)
    print("SIMULATION STATISTICS")
    print("="*60)
    print(f"Max position error: {np.max(pos_error):.4f} m")
    print(f"Mean position error: {np.mean(pos_error):.4f} m")
    print(f"Final position: X={X[-1]:.3f}, Y={Y[-1]:.3f}, Z={Z[-1]:.3f} m")
    print(f"Final reference: X={Xd[-1]:.3f}, Y={Yd[-1]:.3f}, Z={Zd[-1]:.3f} m")
    print(f"Max roll angle: {np.max(np.abs(Phi)):.3f} rad ({np.max(np.abs(np.rad2deg(Phi))):.1f}°)")
    print(f"Max pitch angle: {np.max(np.abs(Theta)):.3f} rad ({np.max(np.abs(np.rad2deg(Theta))):.1f}°)")
    print("="*60 + "\n")

