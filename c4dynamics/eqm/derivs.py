from typing import Union
import numpy as np
import sys

# sys.path.append(".")
from c4dynamics.utils.math import sin, cos, tan
from c4dynamics import datapoint, rigidbody


def eqm3(dp: "datapoint", F: Union[np.ndarray, list]) -> np.ndarray:
    """
    Translational motion derivatives.

    These equations represent a set of first-order ordinary
    differential equations (ODEs) that describe the motion
    of a datapoint in three-dimensional space under the influence
    of external forces.

    Parameters
    ----------
    dp : :class:`datapoint <c4dynamics.states.lib.datapoint.datapoint>`
        C4dynamics' datapoint object for which the equations of motion are calculated.

    F : array_like
        Force vector :math:`[F_x, F_y, F_z]`

    Returns
    -------
    out : numpy.array
        :math:`[dx, dy, dz, dv_x, dv_y, dv_z]`
        6 derivatives of the equations of motion, 3 position derivatives,
        and 3 velocity derivatives.

    Examples
    --------

    Import required packages:

    .. code::

      >>> import c4dynamics as c4d
      >>> from matplotlib import pyplot as plt
      >>> import numpy as np


    .. code::

      >>> dp = c4d.datapoint()
      >>> dp.mass = 10                    # mass 10kg     # doctest: +IGNORE_OUTPUT
      >>> F  = [0, 0, c4d.g_ms2]          # g_ms2 = 9.8m/s^2
      >>> c4d.eqm.eqm3(dp, F)             # doctest: +NUMPY_FORMAT
      array([0  0  0  0  0  0.980665])


    Euler integration on the equations of motion of
    mass in a free fall:


    .. code::

      >>> h0 = 10000
      >>> pt = c4d.datapoint(z = 10000)
      >>> while pt.z > 0:
      ...   pt.store()
      ...   dx = c4d.eqm.eqm3(pt, [0, 0, -c4d.g_ms2])
      ...   pt.X += dx  # (dt = 1)
      >>> pt.plot('z')
      >>> # comapre to anayltic solution
      >>> t = np.arange(len(pt.data('t')))
      >>> z = h0 - .5 * c4d.g_ms2 * t**2
      >>> plt.gca().plot(t[z > 0], z[z > 0], 'c', linewidth = 1) # doctest: +IGNORE_OUTPUT

    .. figure:: /_examples/eqm/eqm3.png

    """

    dx = dp.vx
    dy = dp.vy
    dz = dp.vz

    dvx = F[0] / dp.mass
    dvy = F[1] / dp.mass
    dvz = F[2] / dp.mass

    return np.array([dx, dy, dz, dvx, dvy, dvz])


def eqm6(rb: "rigidbody", F: Union[np.ndarray, list], M: Union[np.ndarray, list]) -> np.ndarray:
    """
    Translational and angular motion derivatives.

    A set of first-order ordinary
    differential equations (ODEs) that describe the motion
    of a rigid body in three-dimensional space under the influence
    of external forces and moments.

    Parameters
    ----------
    rb : :class:`rigidbody <c4dynamics.states.lib.rigidbody.rigidbody>`
        C4dynamics' rigidbody object for which the
        equations of motion are calculated on.
    F : array_like
        Force vector :math:`[F_x, F_y, F_z]`
    M : array_like
        Moments vector :math:`[M_x, M_y, M_z]`

    Returns
    -------
    out : numpy.array
        :math:`[dx, dy, dz, dv_x, dv_y, dv_z, d\\varphi, d\\theta, d\\psi, dp, dq, dr]`

        12 total derivatives; 6 of translational motion, 6 of rotational motion.

    Examples
    --------
    Euler integration on the equations of motion of
    a stick fixed at one edge:

    (mass: 0.5 kg, moment of inertia about y: 0.4 kg*m^2,
    Length: 1m, initial Euler pitch angle: 80° (converted to radians))


    Import required packages:

    .. code::

      >>> import c4dynamics as c4d
      >>> import numpy as np

    Settings and initial conditions

    .. code::

      >>> dt = 0.5e-3
      >>> t = np.arange(0, 10, dt)
      >>> length =  1  # metter
      >>> rb = c4d.rigidbody(theta = 80 * c4d.d2r)
      >>> rb.mass = 0.5 # kg
      >>> rb.I = [0, 0.4, 0]


    Main loop:

    .. code::

      >>> for ti in t:
      ...    rb.store(ti)
      ...    tau_g = -rb.mass * c4d.g_ms2 * length / 2 * c4d.cos(rb.theta)
      ...    dx = c4d.eqm.eqm6(rb, np.zeros(3), [0, tau_g, 0])
      ...    rb.X += dx * dt
      >>> rb.plot('theta')

    .. figure:: /_examples/eqm/eqm3.png

    """

    ixx, iyy, izz = rb.I
    #
    # translational motion derivatives
    ##
    dx = rb.vx
    dy = rb.vy
    dz = rb.vz

    dvx = F[0] / rb.mass
    dvy = F[1] / rb.mass
    dvz = F[2] / rb.mass

    #
    # euler angles derivatives
    ##
    dphi = (rb.q * sin(rb.phi) + rb.r * cos(rb.phi)) * tan(rb.theta) + rb.p
    dtheta = rb.q * cos(rb.phi) - rb.r * sin(rb.phi)
    dpsi = (rb.q * sin(rb.phi) + rb.r * cos(rb.phi)) / cos(rb.theta)

    #
    # angular motion derivatives
    ##
    # dp     = (lA - q * r * (izz - iyy)) / ixx
    dp = 0 if ixx == 0 else (M[0] - rb.q * rb.r * (izz - iyy)) / ixx
    dq = 0 if iyy == 0 else (M[1] - rb.p * rb.r * (ixx - izz)) / iyy
    dr = 0 if izz == 0 else (M[2] - rb.p * rb.q * (iyy - ixx)) / izz

    #       0   1   2   3    4    5    6     7       8     9   10  11
    return np.array([dx, dy, dz, dvx, dvy, dvz, dphi, dtheta, dpsi, dp, dq, dr])


if __name__ == "__main__":

    from c4dynamics import rundoctests
    rundoctests(sys.modules[__name__])
