import sys

import numpy as np

# sys.path.append(".")
import c4dynamics as c4d
from c4dynamics.states.lib.datapoint import datapoint


class rigidbody(datapoint):
    """
    A rigid-body object


    The :class:`rigidbody` extends the
    :class:`datapoint <c4dynamics.states.lib.datapoint.datapoint>`
    class to form an elementary rigidbody object in space, i.e.
    an object with length and attitude.

    It introduces attributes related to rotational dynamics,
    such as angular position, angular velocity, and moment of inertia.
    As such its state vector consists of the following variables:


    .. math::

      X = [x, y, z, v_x, v_y, v_z, \\varphi, \\theta, \\psi, p, q, r)]^T

    - Position coordinates, velocity coordinates.
    - Angles, angular rates.



    **Arguments**

    x : float or int, optional
        The x-position of the datapoint. Default value :math:`x = 0`.
    y : float or int, optional
        The y-position of the datapoint. Default value :math:`y = 0`.
    z : float or int, optional
        The z-position of the datapoint. Default value :math:`z = 0`.
    vx : float or int, optional
        Component of velocity along the x-axis. Default value :math:`v_x = 0`.
    vy : float or int, optional
        Component of velocity along the y-axis. Default value :math:`v_y = 0`.
    vz : float or int, optional
        Component of velocity along the z-axis. Default value :math:`v_z = 0`.
    phi : float or int, optional
        Euler angle representing rotation around the x-axis (rad). Default value
        :math:`\\varphi = 0`.
    theta : float or int, optional
        Euler angle representing rotation around the y-axis (rad). Default value
        :math:`\\theta = 0`.
    psi : float or int, optional
        Euler angle representing rotation around the z-axis (rad). Default value
        :math:`\\psi = 0`.
    p : float or int, optional
        Angular rate around the x-axis (roll). (rad/sec). Default value :math:`p = 0`.
    q : float or int, optional
        Angular rate around the y-axis (pitch). (rad/sec). Default value :math:`q = 0`.
    r : float or int, optional
        Angular rate around the z-axis (yaw). (rad/sec). Default value :math:`r = 0`.


    The input arguments determine the initial values of the instance.
    The vector of initial conditions can be retrieved by calling
    :attr:`rigidbody.X0 <c4dynamics.states.state.state.X0>`:

    .. code::

      >>> from c4dynamics import rigidbody

    .. code::

      >>> rb = rigidbody(z = 1000, theta = 10 * c4d.d2r, q = 0.5 * c4d.d2r)
      >>> rb.X0 # doctest: +NUMPY_FORMAT
      [0  0  1000  0  0  0  0  0.174  0  0  0.0087  0]


    When the initial values are not known at the stage of constructing
    the state object, it's possible to pass zeros and override them later
    by direct assignment of the state variable with a `0` suffix.
    See more at :attr:`state.X0 <c4dynamics.states.state.state.X0>`.


    Parameters
    ==========

    mass : float
        The mass of the datapoint
    I : [float, float, float]
        An array of moments of inertia

    See Also
    ========
    .lib
    .rotmat
    .state
    .eqm


    Example
    =======

    A simplified model of an aircraft autopilot is given by:

    .. math::

      \\dot{z}(t) = 5 \\cdot \\theta(t)

      \\dot{\\theta}(t) = -0.5 \\cdot \\theta(t) - 0.1 \\cdot z(t)


    Where:

    - :math:`z` is the deviation of the aircraft from the required altitude
    - :math:`\\theta` is the pitch angle

    The aircraft is represented by a `rigidbody` object.
    `scipy's odeint` is employed to solve the
    dynamics equations and update the state vector `X`.


    import required packages:

    .. code::

      >>> import c4dynamics as c4d
      >>> from matplotlib import pyplot as plt
      >>> from scipy.integrate import odeint
      >>> import numpy as np


    Settings and initial conditions:

    .. code::

      >>> dt, tf = 0.01, 15
      >>> tspan = np.arange(0, tf, dt)
      >>> A = np.zeros((12, 12))
      >>> A[2, 7] =  5
      >>> A[7, 2] = -0.1
      >>> A[7, 7] = -0.5
      >>> f16 = c4d.rigidbody(z = 1, theta = 0)
      >>> for t in tspan:
      ...   f16.X = odeint(lambda y, t: A @ y, f16.X, [t, t + dt])[-1]
      ...   f16.store(t)


    .. code::

      >>> _, ax = plt.subplots(2, 1)
      >>> f16.plot('z', ax = ax[0])   # doctest: +IGNORE_OUTPUT
      >>> ax[0].set(xlabel = '')  # doctest: +IGNORE_OUTPUT
      >>> f16.plot('theta', ax = ax[1], scale = c4d.r2d)

    .. figure:: /_examples/rigidbody/intro_f16_autopilot.png

    The :meth:`animate` method allows the user to play the attitude
    histories given a 3D model (requires installation of `open3D`).

    The model in the example can be fetched using the c4dynamics datasets module
    (see :mod:`c4dynamics.datasets`):

    .. code::

      >>> modelpath = c4d.datasets.d3_model('f16')
      Fetched successfully

    .. code::

      >>> f16colors = np.vstack(([255, 215, 0], [255, 215, 0], [184, 134, 11], [0, 32, 38],
      ...             [218, 165, 32], [218, 165, 32], [54, 69, 79], [205, 149, 12],
      ...             [205, 149, 12])) / 255
      >>> f16.animate(modelpath, angle0 = [90 * c4d.d2r, 0, 180 * c4d.d2r],
      ...             modelcolor = f16colors)

    .. figure:: /_examples/rigidbody/rb_intro_ap.gif

    """

    phi: float
    theta: float
    psi: float
    p: float
    q: float
    r: float

    _ixx = 0
    _iyy = 0
    _izz = 0

    def __init__(self, x=0, y=0, z=0, vx=0, vy=0, vz=0, phi=0, theta=0, psi=0, p=0, q=0, r=0):

        rbargs = {}

        rbargs.setdefault("x", x)
        rbargs.setdefault("y", y)
        rbargs.setdefault("z", z)
        rbargs.setdefault("vx", vx)
        rbargs.setdefault("vy", vy)
        rbargs.setdefault("vz", vz)

        rbargs.setdefault("phi", phi)
        rbargs.setdefault("theta", theta)
        rbargs.setdefault("psi", psi)
        rbargs.setdefault("p", p)
        rbargs.setdefault("q", q)
        rbargs.setdefault("r", r)

        c4d.state.__init__(self, **rbargs)

    @property
    def I(self):
        """
        Gets and sets the array of moments of inertia.

        .. math::

          I = [I_{xx}, I_{yy}, I_{zz}]


        Default: :math:`I = [0, 0, 0]`


        Parameters
        ----------
        I : numpy.array or list
            An array of three moments of inertia about each
            one of the axes :math:`([I_{xx}, I_{yy}, I_{zz}])`.


        Returns
        -------
        out : numpy.array
            An array of the three moments of inertia :math:`[I_{xx}, I_{yy}, I_{zz}]`.


        Example
        -------

        The moment of inertia
        determines how much torque is required for a
        desired angular acceleration about a rotational axis.


        In this example, two physical pendulums with the same
        initial conditions
        show the effect of different moments of inertia
        on the time period of an oscillation:

        .. math::

          T = 2 \\cdot \\pi \\cdot \\sqrt{{I \\over m \\cdot g \\cdot l}}

        where here :math:`m` is the mass :math:`m = 1`,
        :math:`l` is the length from the center of mass :math:`l = 1`,
        :math:`g` is the gravity acceleration,
        and :math:`I` is the moment of inertia about :math:`y`,
        :math:`I_{yy1} = 0.5, I_{yy2} = 0.05`


        Import required packages:

        .. code::

          >>> import c4dynamics as c4d
          >>> from matplotlib import pyplot as plt
          >>> from scipy.integrate import odeint
          >>> import numpy as np


        Settings and initial condtions:

        .. code::

          >>> b = 0.5
          >>> dt = 0.01
          >>> g = c4d.g_ms2
          >>> theta0 = 80 * c4d.d2r
          >>> rb05  = c4d.rigidbody(theta = theta0)
          >>> rb05.I = [0, .5, 0]
          >>> rb005 = c4d.rigidbody(theta = theta0)
          >>> rb005.I = [0, .05, 0]



        Physical pendulum dynamics:

        .. code::

          >>> def pendulum(yin, t, Iyy):
          ...   theta, q = yin[7], yin[10]
          ...   yout = np.zeros(12)
          ...   yout[7] = q
          ...   yout[10] = -g * c4d.sin(theta) / Iyy - b * q
          ...   return yout



        Main loop

        .. code::

          >>> for ti in np.arange(0, 5, dt):
          ...   # Iyy = 0.5
          ...   rb05.X = odeint(pendulum, rb05.X, [ti, ti + dt], (rb05.I[1],))[1]
          ...   rb05.store(ti)
          ...   # Iyy = 0.05
          ...   rb005.X = odeint(pendulum, rb005.X, [ti, ti + dt], (rb005.I[1],))[1]
          ...   rb005.store(ti)


        Plot results:

        .. code::

          >>> rb05.plot('theta')
          >>> rb005.plot('theta', ax = plt.gca(), color = 'c')

        .. figure:: /_examples/rigidbody/Iyy_pendulum.png


        """
        return np.array([self._ixx, self._iyy, self._izz])

    @I.setter
    def I(self, I):
        self._ixx = I[0]
        self._iyy = I[1]
        self._izz = I[2]

    @property
    def angles(self):
        """
        Returns an array of Euler angles.


        .. math::

          angles = [\\varphi, \\theta, \\psi]



        Returns
        -------
        out : numpy.array
            An array of three Euler angles, about each one of the axes
            :math:`([\\varphi, \\theta, \\psi])`


        Examples
        --------

        .. code::

          >>> rb = c4d.rigidbody(phi = 135)
          >>> rb.angles # doctest: +NUMPY_FORMAT
          [135  0  0]

        """

        return np.array([self.phi, self.theta, self.psi])

    @property
    def ang_rates(self):
        """
        Returns an array of angular rates.

        .. math::

          angular rates = [p, q, r]



        Returns
        -------
        out : numpy.array
            An array of three angular rates of the body axes
            :math:`([p, q, r])`


        Examples
        --------

        .. code::

          >>> q0 = 30
          >>> rb = c4d.rigidbody(q = q0)
          >>> rb.ang_rates # doctest: +NUMPY_FORMAT
          [0  30  0]

        """
        return np.array([self.p, self.q, self.r])

    @property
    def BR(self):
        """
    Returns a Body-from-Reference Direction Cosine Matrix (DCM).


    Based on the current Euler angles, `BR` returns the DCM in a 3-2-1 order,
    i.e. first rotation about the z axis (yaw, :math:`\\psi`), then a rotation about the
    y axis (pitch, :math:`\\theta`), and finally a rotation about the x axis
    (roll, :math:`\\varphi`).

    The `DCM321` matrix is calculated by the
    :mod:`rotmat <c4dynamics.rotmat>` module and is given by:


    .. math::

      R = \\begin{bmatrix}
            c\\theta \\cdot c\\psi
          & c\\theta \\cdot s\\psi
          & -s\\theta \\\\
                s\\varphi \\cdot s\\theta \\cdot c\\psi - c\\varphi \\cdot s\\psi
              & s\\varphi \\cdot s\\theta \\cdot s\\psi + c\\varphi \\cdot c\\psi
              & s\\varphi \\cdot c\\theta \\\\
                    c\\varphi \\cdot s\\theta \\cdot c\\psi + s\\varphi \\cdot s\\psi
                  & c\\varphi \\cdot s\\theta \\cdot s\\psi - s\\varphi \\cdot c\\psi
                  & c\\varphi \\cdot c\\theta
          \\end{bmatrix}

    where

    - :math:`c\\varphi \\equiv cos(\\varphi)`
    - :math:`s\\varphi \\equiv sin(\\varphi)`
    - :math:`c\\theta \\equiv cos(\\theta)`
    - :math:`s\\theta \\equiv sin(\\theta)`
    - :math:`c\\psi \\equiv cos(\\psi)`
    - :math:`s\\psi \\equiv sin(\\psi)`



    For the background material regarding the rotational matrix operations,
    see :mod:`rotmat <c4dynamics.rotmat>`.

    Returns
    -------

    out : numpy.ndarray
        A 3x3 DCM matrix uses to rotate a vector
        to the body frame
        from a reference frame of coordinates.


    Example
    -------

    .. code::

      >>> v_inertial = [1, 0, 0]
      >>> rb = c4d.rigidbody(psi = 45 * c4d.d2r)
      >>> v_body = rb.BR @ v_inertial
      >>> v_body  # doctest: +NUMPY_FORMAT
      [0.707  -0.707  0.0]

    """

        return c4d.rotmat.dcm321(phi=self.phi, theta=self.theta, psi=self.psi)

    @property
    def RB(self):
        """
    Returns a Reference-from-Body Direction Cosine Matrix (DCM).


    Based on the current Euler angles, `RB` returns the
    transpose matrix of :attr:`BR <c4dynamics.states.lib.rigidbody.rigidbody.BR>`,
    where :attr:`BR <c4dynamics.states.lib.rigidbody.rigidbody.BR>`
    is the Body from Reference
    DCM in 3-2-1 order.

    The transpose matrix of the DCM generated by
    three Euler angles :math:`\\varphi` (rotation about `x`),
    :math:`\\theta` (about `y`), and :math:`\\psi` (about `z`) in 3-2-1 order,
    is given by:

    .. math::

        R = \\begin{bmatrix}
              c\\theta \\cdot c\\psi
            & s\\varphi \\cdot s\\theta \\cdot c\\psi - c\\varphi \\cdot s\\psi
            & c\\varphi \\cdot s\\theta \\cdot c\\psi + s\\varphi \\cdot s\\psi \\\\
                  c\\theta \\cdot s\\psi
                & s\\varphi \\cdot s\\theta \\cdot s\\psi + c\\varphi \\cdot c\\psi
                & c\\varphi \\cdot s\\theta \\cdot s\\psi - s\\varphi \\cdot c\\psi \\\\
                      -s\\theta
                    & s\\varphi \\cdot c\\theta
                    & c\\varphi \\cdot c\\theta
            \\end{bmatrix}

    where

    - :math:`c\\varphi \\equiv cos(\\varphi)`
    - :math:`s\\varphi \\equiv sin(\\varphi)`
    - :math:`c\\theta \\equiv cos(\\theta)`
    - :math:`s\\theta \\equiv sin(\\theta)`
    - :math:`c\\psi \\equiv cos(\\psi)`
    - :math:`s\\psi \\equiv sin(\\psi)`


    For the background material regarding the rotational matrix operations,
    see :mod:`rotmat <c4dynamics.rotmat>`.



    Returns
    -------

    out : numpy.ndarray
        A 3x3 DCM matrix uses to rotate a vector from a body frame
        to a reference frame of coordinates.


    Example
    -------

    .. code::

      >>> v_body = [np.sqrt(3), 0, 1]
      >>> rb = c4d.rigidbody(theta = 30 * c4d.d2r)
      >>> v_inertial = rb.RB @ v_body
      >>> v_inertial   # doctest: +NUMPY_FORMAT
      [2.0  0.0  0.0]

    """
        return np.transpose(self.BR)

    def inteqm(self, forces, moments, dt):  # type: ignore
        """
        Advances the state vector, :attr:`rigidbody.X <c4dynamics.states.state.state.X>`,
        with respect to the input
        forces and moments on a single step of time, `dt`.

        Integrates equations of six degrees motion using the Runge-Kutta method.

        This method numerically integrates the equations of motion for a dynamic system
        using the fourth-order Runge-Kutta method as given by
        :func:`eqm.int6 <c4dynamics.eqm.integrate.int6>`.

        The derivatives of the equations are of six dimensional motion as
        given by
        :py:func:`eqm.eqm6 <c4dynamics.eqm.derivs.eqm6>`.


        Parameters
        ----------
        forces : numpy.array or list
            An external forces vector acting on the body, `forces = [Fx, Fy, Fz]`
        moments : numpy.array or list
            An external moments vector acting on the body, `moments = [Mx, My, Mz]`
        dt : float or int
            Interval time step for integration.


        Returns
        -------
        out : numpy.float64
            An acceleration array at the final time step.


        Warning
        -------
        This method is not recommanded when the vectors
        of forces or moments depend on the state variables.
        Since the vectors of forces and moments are provided once at the
        entrance to the integration, they remain constant
        for the entire steps.
        Therefore, when the forces or moments depend on the state variables
        the results of this method are not accurate and may lead to instability.


        Examples
        --------
        A torque is applied by spacecraft thrusters to stabilize
        the roll in a constant rate.

        Import required packages:

        .. code::

          >>> import c4dynamics as c4d
          >>> from matplotlib import pyplot as plt
          >>> import numpy as np


        Settings and initial conditions:

        .. code::

          >>> dt = 0.001
          >>> torque = [0.1, 0, 0]
          >>> rb = c4d.rigidbody()
          >>> rb.I = [0.5, 0, 0]  # Moment of inertia about x


        Main loop:

        .. code::

          >>> for ti in np.arange(0, 5, dt):
          ...   rb.inteqm(np.zeros(3), torque, dt)  # doctest: +IGNORE_OUTPUT
          ...   if rb.p >= 10 * c4d.d2r:
          ...     torque = [0, 0, 0]
          ...   rb.store(ti)


        Plot results:

        .. code::

          >>> rb.plot('p')

        .. figure:: /_examples/rigidbody/inteqm_rollstable.png


        """
        # from c4dynamics.eqm import eqm6
        self.X, acc = c4d.eqm.int6(self, forces, moments, dt, derivs_out=True)
        return acc

    def animate(
        self,
        modelpath,
        angle0=[0, 0, 0],
        modelcolor=None,
        dt=1e-3,
        savedir=None,
        cbackground=[1, 1, 1],
    ):

        c4d.rotmat.animate(self, modelpath, angle0, modelcolor, dt, savedir, cbackground)


rigidbody.animate.__doc__ = c4d.rotmat.animate.__doc__


if __name__ == "__main__":

    from c4dynamics import rundoctests

    try:
        import open3d as o3d  # noqa: F401

        rundoctests(sys.modules[__name__])
    except ImportError:
        rundoctests(sys.modules[__name__], ["__main__.rigidbody.animate", "__main__.rigidbody"])
