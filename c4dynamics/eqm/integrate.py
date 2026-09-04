import sys
from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray

from c4dynamics import datapoint, rigidbody

# sys.path.append(".")
from c4dynamics.eqm.derivs import eqm3, eqm6


def int3(
    dp: "datapoint", forces: Union[np.ndarray, list], dt: float, derivs_out: bool = False
) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """
    A step integration of the equations of translational motion.

    This method makes a numerical integration using the
    fourth-order Runge-Kutta method.

    The integrated derivatives are of three dimensional translational motion as
    given by
    :func:`eqm3 <c4dynamics.eqm.derivs.eqm3>`.


    The result is an integrated state in a single interval of time where the
    size of the step is determined by the parameter `dt`.


    Parameters
    ----------
    dp : :class:`datapoint <c4dynamics.states.lib.datapoint.datapoint>`
        The datapoint which state vector is to be integrated.
    forces : numpy.array or list
        An external forces array acting on the body.
    dt : float
        Time step for integration.
    derivs_out : bool, optional
        If true, returns the last three derivatives as an estimation for
        the acceleration of the datapoint.


    Returns
    -------
    X : numpy.float64
        An integrated state.
    dxdt4 : numpy.float64, optional
        The last three derivatives of the equations of motion.
        These derivatives can use as an estimation for the acceleration of the datapoint.
        Returned if `derivs_out` is set to `True`.



    **Algorithm**


    The integration steps follow the Runge-Kutta method:

    1. Compute k1 = f(ti, yi)

    2. Compute k2 = f(ti + dt / 2, yi + dt * k1 / 2)

    3. Compute k3 = f(ti + dt / 2, yi + dt * k2 / 2)

    4. Compute k4 = f(ti + dt, yi + dt * k3)

    5. Update yi = yi + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)



    Examples
    --------

    Runge-Kutta integration of the equations of motion on a mass in a free fall
    (compare to the same example in :func:`eqm3 <c4dynamics.eqm.derivs.eqm3>`
    with Euler integration):

    Import required packages


    .. code::

      >>> import c4dynamics as c4d

    .. code::

      >>> pt = c4d.datapoint(z = 10000)
      >>> while pt.z > 0:
      ...   pt.store()
      ...   pt.X = c4d.eqm.int3(pt, [0, 0, -c4d.g_ms2], dt = 1)

    .. figure:: /_examples/eqm/int3.png

    """

    x0 = dp.X
    X = dp.X

    # step 1
    dxdt1 = eqm3(dp, forces)
    # dp.update(x0 + dt / 2 * dxdt1)
    X = x0 + dt / 2 * dxdt1

    # step 2
    dxdt2 = eqm3(dp, forces)
    # dp.update(x0 + dt / 2 * dxdt2)
    X = x0 + dt / 2 * dxdt2

    # step 3
    dxdt3 = eqm3(dp, forces)
    # dp.update(x0 + dt * dxdt3)
    X = x0 + dt * dxdt3
    dxdt3 += dxdt2

    # step 4
    dxdt4 = eqm3(dp, forces)

    #
    # dp.update(np.concatenate((x0 + dt / 6 * (dxdt1 + dxdt4 + 2 * dxdt3), dxdt4[-3:]), axis = 0))
    X = x0 + dt / 6 * (dxdt1 + dxdt4 + 2 * dxdt3)
    # dp.ax, dp.ay, dp.az = dxdt4[-3:]

    if not derivs_out:
        return X

    # return also the derivatives.
    return X, dxdt4[-3:]

    ##


def int6(
    rb: "rigidbody",
    forces: Union[np.ndarray, list],
    moments: Union[np.ndarray, list],
    dt: float,
    derivs_out: bool = False,
) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """
    A step integration of the equations of motion.

    This method makes a numerical integration using the
    fourth-order Runge-Kutta method.

    The integrated derivatives are of three dimensional translational motion as
    given by
    :func:`eqm6 <c4dynamics.eqm.derivs.eqm6>`.


    The result is an integrated state in a single interval of time where the
    size of the step is determined by the parameter `dt`.


    Parameters
    ----------
    dp : :class:`datapoint <c4dynamics.states.lib.datapoint.datapoint>`
        The datapoint which state vector is to be integrated.
    forces : numpy.array or list
        An external forces array acting on the body.
    dt : float
        Time step for integration.
    derivs_out : bool, optional
        If true, returns the last three derivatives as an estimation for
        the acceleration of the datapoint.


    Returns
    -------
    X : numpy.float64
        An integrated state.
    dxdt4 : numpy.float64, optional
        The last six derivatives of the equations of motion.
        These derivatives can use as an estimation for the
        translational and angular acceleration of the datapoint.
        Returned if `derivs_out` is set to `True`.


    **Algorithm**


    The integration steps follow the Runge-Kutta method:

    1. Compute k1 = f(ti, yi)

    2. Compute k2 = f(ti + dt / 2, yi + dt * k1 / 2)

    3. Compute k3 = f(ti + dt / 2, yi + dt * k2 / 2)

    4. Compute k4 = f(ti + dt, yi + dt * k3)

    5. Update yi = yi + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


    Examples
    --------

    In the following example, the equations of motion of a
    cylinderical body are integrated by using the
    :func:`int6 <c4dynamics.eqm.integrate.int6>`.

    The results are compared to the results of the same equations
    integrated by using `scipy.odeint`.


    1. `int6`


    Import required packages

    .. code::

      >>> import c4dynamics as c4d
      >>> from matplotlib import pyplot as plt
      >>> from scipy.integrate import odeint
      >>> import numpy as np


    Settings and initial conditions

    .. code::

      >>> dt = 0.5e-3
      >>> t  = np.arange(0, 10, dt)
      >>> theta0 =  80 * c4d.d2r       # deg
      >>> q0     =  0 * c4d.d2r        # deg to sec
      >>> Iyy    =  .4                 # kg * m^2
      >>> length =  1                  # meter
      >>> mass   =  0.5                # kg


    Define the cylinderical-rigidbody object

    .. code::

      >>> rb = c4d.rigidbody(theta = theta0, q = q0)
      >>> rb.I = [0, Iyy, 0]
      >>> rb.mass = mass


    Main loop:

    .. code::

      >>> for ti in t:
      ...   rb.store(ti)
      ...   tau_g = -rb.mass * c4d.g_ms2 * length / 2 * c4d.cos(rb.theta)
      ...   rb.X = c4d.eqm.int6(rb, np.zeros(3), [0, tau_g, 0], dt)


    .. code::

      >>> rb.plot('theta')

    .. figure:: /_examples/eqm/int6.png


    2. `scipy.odeint`

    .. code::

      >>> def pend(y, t):
      ...  theta, omega = y
      ...  dydt = [omega, -rb.mass * c4d.g_ms2 * length / 2 * c4d.cos(theta) / Iyy]
      ...  return dydt
      >>> sol = odeint(pend, [theta0, q0], t)


    Compare to `int6`:

    .. code::

      >>> plt.plot(*rb.data('theta', c4d.r2d),
      ...       'm',
      ...       label = 'c4dynamics.int6'
      ... )  # doctest: +IGNORE_OUTPUT
      >>> plt.plot(t, sol[:, 0] * c4d.r2d, 'c', label = 'scipy.odeint') # doctest: +IGNORE_OUTPUT
      >>> c4d.plotdefaults(plt.gca(),
      ...       'Equations of Motion Integration ($\\theta$)',
      ...       'Time',
      ...       'degrees',
      ...       fontsize = 12
      ... )
      >>> plt.legend() # doctest: +IGNORE_OUTPUT


    .. figure:: /_examples/eqm/int6_vs_scipy.png


    **Note - Differences Between Scipy and C4dynamics Integration**

    The difference in the results derive from the method of delivering the
    forces and moments.
    scipy.odeint gets as input the function that caluclates the derivatives,
    where the forces and moments are included in it:

    `dydt = [omega, -rb.mass * c4d.g_ms2 * length / 2 * c4d.cos(theta) / Iyy]`

    This way, the forces and moments are recalculated for each step of
    the integration.

    c4dynamics.eqm.int6 on the other hand, gets the vectors of forces and moments
    only once when the function is called and therefore refer to them as constant
    for the four steps of integration.

    When external factors may vary quickly over time and a high level of
    accuracy is required, using other methods, like scipy.odeint, is recommanded.
    If computational resources are available, a decrement of the step-size
    may be a workaround to achieve high accuracy results.


    """

    # x, y, z, vx, vy, vz, phi, theta, psi, p, q, r
    x0 = rb.X
    X = rb.X

    # step 1
    h1 = eqm6(rb, forces, moments)
    X = x0 + dt / 2 * h1

    # step 2
    h2 = eqm6(rb, forces, moments)
    X = x0 + dt / 2 * h2

    # step 3
    h3 = eqm6(rb, forces, moments)
    X = x0 + dt * h3

    # step 4
    h4 = eqm6(rb, forces, moments)

    X = x0 + dt / 6 * (h1 + 2 * h2 + 2 * h3 + h4)

    if not derivs_out:
        return X

    # return also the derivatives.
    #                0   1   2   3    4    5    6     7       8     9   10  11
    # h4 = np.array([dx, dy, dz, dvx, dvy, dvz, dphi, dtheta, dpsi, dp, dq, dr])
    return X, np.concatenate([h4[3:6], h4[9:12]])  # X, [dvx, dvy, dvz, dp, dq, dr]

    ##


if __name__ == "__main__":

    from c4dynamics import rundoctests

    rundoctests(sys.modules[__name__])
