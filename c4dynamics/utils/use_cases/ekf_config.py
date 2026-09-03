"""
Reference EKF noise / initialization block.

Data module for the EKF closed-loop example: :func:`default_ekf_config`
returns the process noise ``Q``, measurement noise ``R`` (per sensor),
initial covariance ``P0``, initial-estimate offset, injected sensor-noise
levels, and sensor rates that :func:`quad_ekf.run_fig8_ekf` /
:func:`quad_ekf.run_gps_dropout_sweep` fall back on when a caller doesn't
supply ``ekf_cfg`` explicitly.
"""

import numpy as np


def default_ekf_config() -> dict:
    """Reference EKF noise / initialization block.

    Returns a fresh dict on every call (mutate your own copy freely without
    affecting other callers).

    Returns
    -------
    dict
        Keys: ``Q, P0, R_gps, R_gyro, R_mag, R_acc, x0_pos_sigma,
        x0_att_sigma, gps_std, gyro_std, mag_std, acc_std, gps_rate,
        mag_rate, seed, ideal_imu, ideal_magnetometer, ideal_gps``.
    """
    Q = np.diag(np.array([
        0.005, 0.005, 0.008,   # x, y, z        [m]
        0.020, 0.020, 0.025,   # vx, vy, vz     [m/s]  (adaptively scaled)
        0.008, 0.008, 0.010,   # phi, theta, psi[rad]
        0.012, 0.012, 0.012,   # p, q, r        [rad/s]
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
        'ideal_imu': False,
        'ideal_magnetometer': False,
        'ideal_gps': False,
    }
