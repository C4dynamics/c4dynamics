"""
Reference cascade-PID controller gains.

Zero-dependency data module: :func:`default_controller_config` returns the
gain block (inner rate / middle attitude / outer position loops, velocity
feedforward, and actuator limits), tuned in the plain, truth-fed cascade-PID
figure-8 example against :func:`c4dynamics.models.quad.default_quad_config`'s
vehicle. Other examples (e.g. the EKF closed loop) import this as their
starting point and override specific entries explicitly where a different
regime calls for different tuning, rather than carrying an independent copy
that can silently diverge with no record of why.
"""


def default_controller_config() -> dict:
    """Reference cascade-PID gains.

    Returns a fresh dict on every call (mutate your own copy, or
    ``dict(default_controller_config(), Kp_z=..., ...)`` to override
    specific entries, without affecting other callers).

    Returns
    -------
    dict
        PID gains/limits, keyed as consumed by
        :func:`c4dynamics.controllers.quad_pid.InitializeControllers`.
    """
    return {
        # Inner loop - angular rate (200 Hz)
        'Kp_p': 0.80, 'Ki_p': 0.0001, 'Kd_p': 0.010,
        'Kp_q': 0.80, 'Ki_q': 0.0001, 'Kd_q': 0.010,
        'Kp_r': 0.60, 'Ki_r': 0.0001, 'Kd_r': 0.008,

        # Middle loop - attitude (100 Hz)
        'Kp_phi'  : 6.0, 'Ki_phi'  : 0.0001, 'Kd_phi'  : 0.80, 'AW_phi'  : 0.5,
        'Kp_theta': 6.0, 'Ki_theta': 0.0001, 'Kd_theta': 0.80, 'AW_theta': 0.5,
        'Kp_psi'  : 4.0, 'Ki_psi'  : 0.5,    'Kd_psi'  : 0.40, 'AW_psi'  : 0.5,

        # Outer loop - position (50 Hz)
        'Kp_x': 0.80, 'Ki_x': 0.00, 'Kd_x': 0.50, 'AW_x': 0.5,
        'Kp_y': 1.00, 'Ki_y': 0.00, 'Kd_y': 0.70, 'AW_y': 0.5,
        'Kp_z': 10.0, 'Ki_z': 10.0, 'Kd_z': 1.50, 'AW_z': 3.0,

        # Velocity feedforward
        'Kff_x': 0.35, 'Kff_y': 0.40,

        # Limits
        'N_rate'         : 50,      # Filter coefficient for derivative term in rate controller
        'omega_max'      : 1000.0,  # max rotor speed [rad/s]
        'T_max_factor'   : 4,       # max thrust factor: T_max = T_max_factor * K_thrust * omega_max**2 [N]
        'T_min'          : 0.0,     # min thrust [N]
        'att_cmd_limit'  : 0.314,   # Max attitude command [rad] = 18deg
        'yaw_rate_limit' : 1.0,     # Max yaw rate command [rad/s]
    }
