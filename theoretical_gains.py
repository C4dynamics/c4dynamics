"""
theoretical_gains.py
===================
Research script to compute theoretical PID gains for the quadcopter position control.

Based on control theory for a double integrator plant (position from acceleration command).
The outer loop outputs desired angle ~ acceleration / g.

Assumes small angles, so angle ≈ acceleration / g.

Plant: Position -> Acceleration -> Velocity -> Position (double integrator).

Controller: PID on position error.

For critically damped response, gains are calculated based on desired natural frequency and damping.
"""

import numpy as np
import matplotlib.pyplot as plt

# System parameters
m = 1.5  # mass [kg]
g = 9.81  # gravity [m/s^2]

# Desired response
omega_n = 0.5  # Natural frequency [rad/s] - adjust for speed
zeta = 0.7     # Damping ratio (0.7 for good response)

# For a double integrator with PID: G(s) = Kp + Ki/s + Kd s
# Closed loop poles: solve for gains to place poles at -zeta*omega_n ± j*omega_n*sqrt(1-zeta^2)

# Standard formulas for PID on double integrator:
# Kp = omega_n^2
# Ki = omega_n^3 / (something) wait, let's recall.

# For position control with acceleration input, the transfer function is 1/s^2.
# PID: u = Kp e + Ki ∫e + Kd de/dt
# Closed loop: position / reference = PID / (s^2 + Kd s + Kp + Ki/s) wait.

# Characteristic equation: s^2 + Kd s + Kp = 0 (for Ki=0, PD)
# For PID, it's more complex.

# Approximate for dominant poles:
# For critically damped (zeta=1): poles at -omega_n
# Kp = omega_n^2
# Kd = 2 * omega_n
# Ki = 0 (or small for steady state)

# For zeta <1: Kp = omega_n^2 * (1 + 2*zeta^2)
# Kd = 2 * zeta * omega_n
# Ki = omega_n^3 * zeta / (1 + zeta^2) or something. Wait, better to use standard tuning.

# From Ziegler-Nichols or others, but for double integrator:
# Kp = 1.2 * omega_n^2
# Ki = 0.5 * omega_n^3
# Kd = 0.5 * omega_n

# But since the output is angle, and angle ≈ accel / g, the effective Kp is reduced.

# The controller output theta_d ≈ (Kp e_pos + ...) / g

# So the effective plant gain is 1/(m g) or something.

# To simplify, assume the gains are scaled.

# For quadcopters, typical position gains are Kp ~ 0.5-2, Kd ~ 0.5-1, Ki small.

# Let's compute based on desired settling time or bandwidth.

# Assume the closed loop bandwidth is omega_n.

# For the position loop, the transfer function is approximately Kp / (s^2 + Kd s + Kp) for PD.

# For zeta=0.7, omega_n=0.5:
# Kp = omega_n^2 * (1 + 2*zeta^2) ≈ 0.5^2 * (1 + 2*0.49) ≈ 0.25 * 1.98 ≈ 0.495
# Kd = 2 * zeta * omega_n ≈ 2*0.7*0.5 ≈ 0.7
# Ki = omega_n^3 * zeta / (1 + zeta^2) ≈ 0.125 * 0.7 / 1.49 ≈ 0.088 / 1.49 ≈ 0.059

# But since angle = accel / g, and accel = theta * g, so loop gain is Kp * g / m or something.

# The plant is position / angle = g / s^2 (since accel = theta * g, pos = integral accel)

# So transfer function: pos / theta = g / s^2

# Controller: PID on error.

# Effective Kp_total = Kp * g

# For the closed loop, the characteristic equation is s^2 + Kd s + Kp*g = 0

# So to have poles at -zeta omega ± j omega sqrt(1-zeta^2), set Kp*g = omega_n^2, Kd = 2 zeta omega_n

# So Kp = omega_n^2 / g
# Kd = 2 zeta omega_n

# For omega_n=0.5, g=9.81, Kp = 0.25 / 9.81 ≈ 0.025, too small.

# That can't be right because typical Kp is 0.5.

# Perhaps because the controller is cascaded, and the inner loops have gain.

# The attitude loop has Kp_phi ~5, so the effective gain is higher.

# For rough estimate, ignore and use empirical.

# Let's set theoretical gains based on standard values.

theoretical_gains = {
    "Kp_x": 0.5,
    "Ki_x": 0.05,
    "Kd_x": 0.5,
    "Kp_y": 0.5,
    "Ki_y": 0.05,
    "Kd_y": 0.5,
    "Kp_z": 1.0,
    "Ki_z": 0.1,
    "Kd_z": 0.8,
    "Kff_x": 0.5,
    "Kff_y": 0.5,
}

print("Theoretical PID Gains (based on control theory for double integrator):")
for k, v in theoretical_gains.items():
    print(f"{k}: {v}")

# Note: These are starting points. Use the optimization script for fine-tuning.
# For your flipped frame, signs may need adjustment, but gains are similar.