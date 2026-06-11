"""
optimize_gains.py
=================
Script to optimize PID gains for the quadcopter figure-8 trajectory tracking.

This script uses scipy.optimize to minimize tracking error (RMSE) by tuning
the outer position loop gains: Kp_x, Ki_x, Kd_x, Kp_y, Ki_y, Kd_y, Kp_z, Ki_z, Kd_z,
and feedforward gains Kff_x, Kff_y.

The optimization minimizes the sum of normalized RMSE for x, y, z over the figure-8 phase.
"""

import numpy as np
from scipy.optimize import minimize
import sys
import os

# Add the path to quad_pid_utils.py
sys.path.append(os.path.join(os.path.dirname(__file__), 'docs', 'source', 'programs', 'pid_cascade'))

import quad_pid_utils as qpu

# Base configuration (copy from your notebook or default)
config = {
    "trajectory": {
        'A'    : 4.0,    # figure-8 X amplitude [m]
        'B'    : 2.0,    # figure-8 Y amplitude [m]
        'omega': 0.1,    # angular frequency [rad/s]  (period ~ 62.8 s)
        'z_ref': 1.5,    # hover altitude [m]
        't_end': 90.0,   # total simulation time [s]
    },

    "sim": {
        "dt": 0.005,   # Time step [s]
        "tf": 30.0,    # Simulation time [s]
    },

    "controller": {
        # Inner loop — angular rate (200 Hz)
        'Kp_p': 0.80,  'Ki_p': 0.0001,  'Kd_p': 0.010,
        'Kp_q': 0.80,  'Ki_q': 0.0001,  'Kd_q': 0.010,
        'Kp_r': 0.60,  'Ki_r': 0.0001,  'Kd_r': 0.008,

        # Middle loop — attitude (100 Hz)
        'Kp_phi'  : 6.0,  'Ki_phi'  : 0.0001,  'Kd_phi'  : 0.80,  'AW_phi'  : 0.5,
        'Kp_theta': 6.0,  'Ki_theta': 0.0001,  'Kd_theta': 0.80,  'AW_theta': 0.5,
        'Kp_psi'  : 4.0,  'Ki_psi'  : 0.5,     'Kd_psi'  : 0.40,  'AW_psi'  : 0.5,

        # Outer loop — position (50 Hz)
        'Kp_x': 0.80,  'Ki_x': 0.00,  'Kd_x': 0.50,  'AW_x': 0.5,
        'Kp_y': 1.00,  'Ki_y': 0.00,  'Kd_y': 0.70,  'AW_y': 0.5,
        'Kp_z': 10.0,  'Ki_z': 10.0,  'Kd_z': 1.50,  'AW_z': 3.0,

        # Velocity feedforward
        'Kff_x': 0.35,  'Kff_y': 0.40,

        # Limits
        'N_rate'        : 50,       # Filter coefficient for derivative term in rate controller
        'omega_max'     : 1000.0,   # max rotor speed [rad/s]
        'T_max_factor'  : 4,        # max thrust factor: T_max = T_max_factor * K_thrust * omega_max**2	[N]
        'T_min'         : 0.0,      # min thrust [N]
        'att_cmd_limit' : 0.314,    # Max attitude command [rad] = 18deg
        'yaw_rate_limit': 1.0,      # Max yaw rate command [rad/s]
        },

    "quad": {
        'm'  : 0.468,      # mass [kg]
        'g'  : 9.81,       # gravity [m/s^2]
        'l'  : 0.225,      # arm length — center to motor [m]
        'kT' : 2.98e-6,    # thrust coefficient  [N/(rad/s)^2]
        'kQ' : 0.0382,     # torque coefficient  [N.m/(rad/s)^2]
        'Ixx': 4.856e-3,   # roll  inertia [kg.m^2]
        'Iyy': 4.856e-3,   # pitch inertia [kg.m^2]
        'Izz': 8.801e-3,   # yaw   inertia [kg.m^2]
        'Ax' : 0.30,       # aero drag — x
        'Ay' : 0.30,       # aero drag — y
        'Az' : 0.25,       # aero drag — z
        'Ar' : 0.20,       # rotational drag — angular rates
        'IR' : 3.357e-5,   # rotor inertia [kg.m^2]
    },
}


# Gains to optimize (indices in the param vector)
gain_names = ['Kp_x', 'Ki_x', 'Kd_x',
              'Kp_y', 'Ki_y', 'Kd_y',
              'Kp_z', 'Ki_z', 'Kd_z',
              'Kff_x', 'Kff_y'
              ]
n_gains = len(gain_names)


def cost_function(params, config):
    """
    Cost function for optimization.
    Runs the simulation with given gains and returns the sum of normalized RMSE.
    """
    # Update config with new gains
    for i, name in enumerate(gain_names):
        config["controller"][name] = params[i]

    # Run simulation
    quad = qpu.run_fig8_pid(config)

    # Compute metrics
    metrics = qpu.compute_metrics(quad, config["trajectory"])

    # Cost: sum of normalized RMSE
    cost = metrics["norm_x"] + metrics["norm_y"] + metrics["norm_z"]

    print(f"Gains: {params}, Cost: {cost:.2f}")
    return cost


def optimize_gains(config, method='L-BFGS-B'):
    """
    Optimize the gains using scipy.optimize.minimize.
    """
    # Initial guess
    x0 = [config["controller"][name] for name in gain_names]

    # Bounds (all positive)
    bounds = [(0.01, 5.0)] * n_gains  # Adjust bounds as needed

    # Minimize
    result = minimize(
        cost_function,
        x0,
        args=(config,),
        method=method,
        bounds=bounds,
        options={'maxiter': 50, 'disp': True}
    )

    print("Optimization Result:")
    print(f"Optimal gains: {result.x}")
    print(f"Final cost: {result.fun}")
    print(f"Success: {result.success}")

    # Update config with optimal gains
    for i, name in enumerate(gain_names):
        config["controller"][name] = result.x[i]

    return config


if __name__ == "__main__":

    print("Starting gain optimization...")
    optimized_config = optimize_gains(config)

    print("\nOptimized Controller Gains:")
    for name in gain_names:
        print(f"{name}: {optimized_config['controller'][name]:.3f}")

    # Optionally, run a final simulation and plot
    print("\nRunning final simulation with optimized gains...")
    quad = qpu.run_fig8_pid(optimized_config)
    qpu.plot_results(quad, optimized_config["trajectory"])
    metrics = qpu.compute_metrics(quad, optimized_config["trajectory"])
    print(f"Final Metrics: RMSE x={metrics['rmse_x']:.3f}, y={metrics['rmse_y']:.3f}, z={metrics['rmse_z']:.3f}")
    print(f"Normalized: x={metrics['norm_x']:.1f}%, y={metrics['norm_y']:.1f}%, z={metrics['norm_z']:.1f}%")


