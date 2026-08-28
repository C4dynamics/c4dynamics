"""
Reference quadcopter physical model.

This module holds **only** the fixed physical description of a small
X-configuration quadrotor -- mass, geometry, rotor thrust/torque
coefficients, inertias, and aerodynamic drag. It contains no dynamics and
no control logic: the equations of motion that consume these parameters
live in :mod:`c4dynamics.controllers.quad_pid` (see
:func:`c4dynamics.controllers.quad_pid.dynamics`), which reads them off a
``quad``-like object's attributes (``quad.m``, ``quad.kT``, ...).

The vehicle is a hand-picked reference, not sourced from a real datasheet
-- roughly a hobbyist-class quadrotor (~0.5 kg, ~0.45 m motor-to-motor),
tuned so the default gains in
:mod:`c4dynamics.controllers.controller_config` fly a stable figure-8. It
is the vehicle shared by two examples built on top of it: the truth-fed
cascade-PID figure-8 (``quadcopter_pid.ipynb``) and the EKF closed-loop
estimation-control figure-8 (``quad_ekf.ipynb``) -- both import
:func:`default_quad_config` from here rather than each carrying an
independent copy of the same numbers.

For a physically-grounded alternative sourced from a real airframe (PX4's
default ``iris.sdf``), see the exploratory
``docs/source/programs/ekf_estimation/iris_quadcopter.py`` -- not yet
wired into this module, since it uses a more general per-rotor-position
geometry rather than this model's single scalar arm length.
"""


def default_quad_config() -> dict:
    """Reference quadcopter physical parameters.

    Returns a fresh dict on every call (mutate your own copy freely without
    affecting other callers).

    Returns
    -------
    dict
        Keys: ``m, g, l, kT, kQ, Ixx, Iyy, Izz, Ax, Ay, Az, Ar, IR``.
    """
    return {
        'm'  : 0.468,             # mass [kg]
        'g'  : 9.81,              # gravity [m/s^2]
        'l'  : 0.225,             # center to motor length [m]
        'kT' : 2.98e-6,           # thrust coefficient  [N/(rad/s)^2]
        'kQ' : 0.0382 * 2.98e-6,  # torque coefficient [N.m/(rad/s)^2] = gamma * kT, gamma = kQ/kT ~ 3.8 cm yaw-reaction lever
        'Ixx': 4.856e-3,          # roll  inertia [kg.m^2]
        'Iyy': 4.856e-3,          # pitch inertia [kg.m^2]
        'Izz': 8.801e-3,          # yaw   inertia [kg.m^2]
        'Ax' : 0.30,              # aero drag - x
        'Ay' : 0.30,              # aero drag - y
        'Az' : 0.25,              # aero drag - z
        'Ar' : 0.20,              # rotational drag - angular rates
        'IR' : 3.357e-5,          # rotor inertia [kg.m^2]
    }
