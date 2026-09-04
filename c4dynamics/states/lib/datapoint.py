import sys
import warnings

import numpy as np

# sys.path.append(".")
import c4dynamics as c4d
from c4dynamics.states.state import state


def create(X):
    if len(X) > 6:
        rb = c4d.rigidbody()
        rb.X = X
        return rb

    dp = c4d.datapoint()
    dp.X = X
    return dp


class datapoint(state):
    """
    A data-point object.


    The :class:`datapoint <c4dynamics.states.lib.datapoint.datapoint>` is the most basic element
    in translational dynamics; it's a point in space with the following state vector:

    .. math::

      X = [x, y, z, v_x, v_y, v_z]^T

    - Position coordinates, velocity coordinates.


    As such, each one of the state variables is a parameter whose value determines
    its initial conditions:

    **Arguments**

    x : float or int, optional
        The initial x-position of the datapoint. Default value :math:`x = 0`.
    y : float or int, optional
        The initial y-position of the datapoint. Default value :math:`y = 0`.
    z : float or int, optional
        The initial z-position of the datapoint. Default value :math:`z = 0`.
    vx : float or int, optional
        The initial x-velocity of the datapoint. Default value :math:`v_x = 0`.
    vy : float or int, optional
        The initial y-velocity of the datapoint. Default value :math:`v_y = 0`.
    vz : float or int, optional
        The initial z-velocity of the datapoint. Default value :math:`v_z = 0`.


    The input arguments determine the initial values of the instance.
    The vector of initial conditions can be retrieved by calling
    :attr:`datapoint.X0 <c4dynamics.states.state.state.X0>`:


    .. code::

      >>> from c4dynamics import datapoint
      >>> dp = datapoint(x = 1000, vx = -100)
      >>> print(dp.X0) # doctest: +NUMPY_FORMAT
      [1000  0  0  -100  0  0]


    When the initial values are not known at the stage of constructing
    the state object, it's possible to pass zeros and override them later
    by direct assignment of the state variable with a `0` suffix.
    See more at :attr:`X0 <c4dynamics.states.state.state.X0>`.


    Parameters
    ==========

    mass : float
        The mass of the datapoint


    See Also
    ========
    .lib
    .state
    .eqm


    Example
    =======

    The following example simulates the
    motion of a body in a free fall.

    The example employs the :mod:`eqm <c4dynamics.eqm>`
    module to solve the equations of motion of a point-mass
    in the three dimensional space, and integrate them
    using the fourth-order Runge-Kutta method.


    Import required packages:

    .. code::

      >>> import c4dynamics as c4d
      >>> from matplotlib import pyplot as plt
      >>> import numpy as np


    Settings and initial conditions:

    .. code::

      >>> dp = c4d.datapoint(z = 100)
      >>> dt = 1e-2
      >>> t = np.arange(0, 10 + dt, dt)


    Main loop:

    .. code::

      >>> for ti in t:
      ...   dp.store(ti)
      ...   if dp.z < 0: break
      ...   dp.inteqm([0, 0, -c4d.g_ms2], dt) # doctest: +IGNORE_OUTPUT

    .. code::

      >>> dp.plot('z')

    .. figure:: /_examples/datapoint/intro_freefall.png


    """

    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float

    # Attributes
    # ==========

    # As mentioned earlier, reading and writing of the state vairables is allowed by using the
    # :attr:`X <datapoint.X>` property. The entire attributes which support
    # the reading and the updating of a datapoint instance are given in the following list:

    # .. automethod:: c4dynamics.datapoint
    #     the datapoint object is the most basic element in the translational dynamics domain.
    #     --
    #     TBD:
    # - there should be one abstrcact class \ inerface of a 'bodyw type which defines eqm(),
    # store() etc.
    #     and datapoint and rigidbody impement it. the body also includes the drawing functions
    # - all these nice things storage, plot etc. have to be move out of here.
    # - add an option in the constructor to select the variables required for storage.
    # - make a dictionary containing the variable name and the variable index in the
    # data storage to save and to extract for plotting.
    # - add total position, velocity, acceleration variables (path angles optional) and update
    # them for each update in the cartesian components.

    #   #
    #   # position
    #   ##
    #    maybe it's a better choise to work with vectors??
    #       maybe there's an option to define an array which will just designate its enteries.
    #       namely a docker that just references its variables
    #       -> this is actually a function!

    #       In Python, all variable names are references to values.
    # https://stackoverflow.com/questions/986006/how-do-i-pass-a-variable-by-reference

    #   https://docs.python.org/3/library/stdtypes.html

    #   Lists may be constructed in several ways:
    #     Using a pair of square brackets to denote the empty list: []
    #     Using square brackets, separating items with commas: [a], [a, b, c]
    #     Using a list comprehension: [x for x in iterable]
    #     Using the type constructor: list() or list(iterable)

    #   Tuples may be constructed in a number of ways:
    #       Using a pair of parentheses to denote the empty tuple: ()
    #       Using a trailing comma for a singleton tuple: a, or (a,)
    #       Separating items with commas: a, b, c or (a, b, c)
    #       Using the tuple() built-in: tuple() or tuple(iterable)

    #   The arguments to the range constructor must be integers

    # __slots__ = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'mass' # , 'ax', 'ay', 'az'
    #                 , 'x0', 'y0', 'z0', 'vx0', 'vy0', 'vz0'
    #                   , '_data', '_vardata', '_didx', '__dict__']

    _mass = 1

    def __init__(self, x=0, y=0, z=0, vx=0, vy=0, vz=0):
        # TODO: document the state type being float by default. (required after chaning vars to
        # 0.0 and the mechanism of generating the state vector array. )

        dpargs = {}
        dpargs.setdefault("x", x)
        dpargs.setdefault("y", y)
        dpargs.setdefault("z", z)
        dpargs.setdefault("vx", vx)
        dpargs.setdefault("vy", vy)
        dpargs.setdefault("vz", vz)

        super().__init__(**dpargs)

    @property
    def mass(self):
        """
        Gets and sets the object's mass.

        Default value :math:`mass = 1`.


        Parameters
        ----------
        mass : float or int
            Mass of the object.

        Returns
        -------
        out : float or int
            A scalar representing the object's mass.


        Example
        -------

        1. `datapoint`

        Two floating balloons of 1kg and 10kg float with total force of L = 0.5N
        and expreience a side wind of 10k.

        Import required packages:

        .. code::

          >>> import c4dynamics as c4d
          >>> from matplotlib import pyplot as plt
          >>> import numpy as np


        Settings and initial conditions:

        .. code::

          >>> dt = 0.01
          >>> tf = 10 + dt
          >>> F = [0, 0, .5]
          >>> #
          >>> bal1 = c4d.datapoint(vx = 10 * c4d.k2ms)
          >>> bal1.mass = 1
          >>> #
          >>> bal10 = c4d.datapoint(vx = 10 * c4d.k2ms)
          >>> bal10.mass = 10


        Main loop:

        .. code::

          >>> for t in np.arange(0, tf, dt):
          ...   bal1.store(t)
          ...   bal10.store(t)
          ...   bal1.X = c4d.eqm.int3(bal1, F, dt)
          ...   bal10.X = c4d.eqm.int3(bal10, F, dt)


        .. code::

          >>> bal1.plot('side')
          >>> bal10.plot('side', ax = plt.gca(), color = 'c')

        .. figure:: /_examples/datapoint/mass_balloon.png

        2. `rigidbody`

        The previous example for a `datapoint` object is directly applicable
        to the `rigidbody` object, as both classes share the same underlying principles
        concerning translational dynamics. Simply replace :code:`c4d.datapoint(vx = 10 * c4d.k2ms)`
        with :code:`c4d.rigidbody(vx = 10 * c4d.k2ms)`.


        """
        return self._mass

    @mass.setter
    def mass(self, mass):
        self._mass = mass

    #
    # runge kutta integration
    ##
    def inteqm(self, forces, dt):
        """
        Advances the state vector :attr:`datapoint.X <c4dynamics.states.state.state.X>`,
        with respect to the input
        forces on a single step of time, `dt`.

        Integrates equations of three degrees translational motion using the Runge-Kutta method.

        This method numerically integrates the equations of motion for a dynamic system
        using the fourth-order Runge-Kutta method as given by
        :func:`int3 <c4dynamics.eqm.integrate.int3>`.

        The derivatives of the equations are of three
        dimensional translational motion and
        produced with
        :func:`eqm3 <c4dynamics.eqm.derivs.eqm3>`


        Parameters
        ----------
        forces : numpy.array or list
            An external forces vector acting on the body, `forces = [Fx, Fy, Fz]`
        dt : float or int
            Interval time step for integration.


        Returns
        -------
        out : numpy.float64
            An acceleration array at the final time step.

        Warning
        -------
        This method is not recommanded when the vector
        of forces depends on the state variables.
        Since the force vector is provided once at the
        entrance to the integration, it remains constant
        for the entire steps.
        Therefore, when the forces depend on the state variables
        the results of this method are not accurate and may lead to instability.





        Example
        -------

        Simulation of the motion of a body in a free fall.

        Employing the :mod:`eqm <c4dynamics.eqm>`
        module to solve the equations of motion of a point-mass
        in the three dimensional space.
        Integrating the equations of motion
        using the fourth-order Runge-Kutta method.


        Import required packages:

        .. code::

          >>> import c4dynamics as c4d
          >>> from matplotlib import pyplot as plt
          >>> import numpy as np


        Settings and initial conditions:

        .. code::

          >>> dp = c4d.datapoint(z = 100)
          >>> dt = 1e-2
          >>> t = np.arange(0, 10 + dt, dt)


        Main loop:

        .. code::

          >>> for ti in t:
          ...   dp.store(ti)
          ...   if dp.z < 0: break
          ...   dp.inteqm([0, 0, -c4d.g_ms2], dt) # doctest: +IGNORE_OUTPUT


        .. code::

          >>> dp.plot('z')

        .. figure:: /_examples/datapoint/intro_freefall.png



        """
        self.X, acc = c4d.eqm.int3(self, forces, dt, derivs_out=True)
        return acc

    #
    # ploting functions
    ##

    def plot(self, var, scale=1, ax=None, filename=None, darkmode=True, **kwargs):
        """
        Draws plots of trajectories or variable evolution over time.

        `var` can be each one of the state variables, or `top`, `side`, for trajectories.


        Parameters
        ----------

        var : str
            The variable to be plotted.
            Possible variables for trajectories: `top`, `side`.
            For time evolution, any one of the state variables is possible:
            `x`, `y`, `z`, `vx`, `vy`, `vz` - for a datapoint object, and
            also `phi`, `theta`, `psi`, `p`, `q`, `r` - for a rigidbody object.

        scale : float or int, optional
            A scaling factor to apply to the variable values. Defaults to `1`.

        ax : matplotlib.axes.Axes, optional
            An existing Matplotlib axis to plot on.
            If None, a new figure and axis will be created. By default None.

        filename : str, optional
            Full file name to save the plot image.
            If None, the plot will not be saved, by default None.

        darkmode : bool, optional
            Directory path to save the plot image.
            If None, the plot will not be saved, by default None.

        **kwargs : dict, optional
            Additional key-value arguments passed to `matplotlib.pyplot.plot`.
            These can include any keyword arguments accepted by `plot`,
            such as `color`, `linestyle`, `marker`, etc.


        Notes
        -----
        - The method overrides the :meth:`plot <c4dynamics.states.state.state.plot>` of
          the parent :class:`state <c4dynamics.states.state.state>` object and is
          applicable to :class:`datapoint <c4dynamics.states.lib.datapoint.datapoint>`
          and its subclass :class:`rigidbody <c4dynamics.states.lib.rigidbody.rigidbody>`.

        - Uses matplotlib for plotting.

        - Trajectory views (`top` and `side`) show the crossrange vs
          downrange or downrange vs altitude.





        Examples
        --------

        Import necessary packages:

        .. code::

          >>> import c4dynamics as c4d
          >>> from matplotlib import pyplot as plt
          >>> import numpy as np
          >>> import scipy


        1) `datapoint`:

        .. code::

          >>> pt = c4d.datapoint()
          >>> for t in np.arange(0, 10, .01):
          ...   pt.x = 10 + np.random.randn()
          ...   pt.store(t)
          >>> pt.plot('x')

        .. figure:: /_examples/datapoint/plot.png


        2) `rigidbody`:

        A physical pendulum is represented by a rigidoby object.
        `scipy's odeint` integrates the equations of motion to simulate
        the angle of rotation of the pendulum over time.


        Settings and initial conditions:

        .. code::

          >>> dt =.01
          >>> pndlm  = c4d.rigidbody(theta = 80 * c4d.d2r)
          >>> pndlm.I = [0, .5, 0]


        Dynamic equations:

        .. code::

          >>> def pendulum(yin, t, Iyy):
          ...   yout = np.zeros(12)
          ...   yout[7]  =  yin[10]
          ...   yout[10] = -c4d.g_ms2 * c4d.sin(yin[7]) / Iyy - .5 * yin[10]
          ...   return yout


        Main loop:

        .. code::

          >>> for ti in np.arange(0, 4, dt):
          ...   pndlm.X = scipy.integrate.odeint(pendulum, pndlm.X, [ti, ti + dt], (pndlm.I[1],))[1]
          ...   pndlm.store(ti)


        Plot results:

        .. code::

          >>> pndlm.plot('theta', scale = c4d.r2d)


        .. figure:: /_examples/rigidbody/plot_pendulum.png


        """
        from matplotlib import pyplot as plt

        if var not in self._didx and var not in ["top", "side"]:
            warnings.warn(
                f"""{var} is not a state variable or a valid trajectory to plot.""", c4d.c4warn
            )
            return
        if not self._data:
            warnings.warn(f"""No stored data for {var}.""", c4d.c4warn)
            return

        if darkmode:
            plt.style.use("dark_background")
        else:
            plt.style.use("default")

        title = ""
        ylabel = ""

        if var.lower() == "top":
            # x axis: y data
            # y axis: x data
            x = self.data("y")[1]
            y = self.data("x")[1]
            xlabel = "Crossrange"
            ylabel = "Downrange"
            title = "Top View"
        elif var.lower() == "side":
            # x axis: x data
            # y axis: z data
            x = self.data("x")[1]
            y = self.data("z")[1]
            xlabel = "Downrange"
            ylabel = "Altitude"
            title = "Side View"
            # ax.invert_yaxis()
        else:

            if self._didx[var] >= 7:  # 7 and above are angular variables
                scale = 180 / np.pi

            if not len(np.flatnonzero(self.data("t") != -1)):  # values for t weren't stored
                x = range(len(self.data("t")))  # t is just indices
                xlabel = "Sample"
            else:
                x = self.data("t")
                xlabel = "Time"
            y = (
                np.array(self._data)[:, self._didx[var]] * scale if self._data else np.empty(1)
            )  # used selection

            if 1 <= self._didx[var] <= 6:
                # x, y, z, vx, vy, vz
                title = var.title()
                ylabel = var.title()
            elif 7 <= self._didx[var] <= 9:
                # phi, theta, psi
                title = "$\\" + var + "$"
                ylabel = title + " (deg)"
            elif 10 <= self._didx[var] <= 12:
                # p, q, r
                title = var.title()
                ylabel = var + " (deg/sec)"

        # Set default values in kwargs only if the user hasn't provided them
        kwargs.setdefault("color", "m")
        kwargs.setdefault("linewidth", 1.2)

        if ax is None:
            # _, ax = plt.subplots()
            # _, ax = plt.subplots(1, 1, dpi = 200, figsize = (3, 2.3)
            #           , gridspec_kw = {'left': .17, 'right': .9
            #                           , 'top': .85, 'bottom': .2
            #                             , 'hspace': 0.5, 'wspace': 0.3})

            # find the legnth of the number to adjust the left axis:
            # ndigits = len(str(int(np.max(y))))

            factorsize = 4
            aspectratio = 1080 / 1920
            _, ax = plt.subplots(
                1,
                1,
                dpi=200,
                figsize=(factorsize, factorsize * aspectratio),
                gridspec_kw={"left": 0.15, "right": 0.85, "top": 0.9, "bottom": 0.2},
            )

        ax.plot(x, y, **kwargs)
        c4d.plotdefaults(ax, title, xlabel, ylabel, 8)

        if filename:
            # plt.tight_layout(pad = 0)
            plt.savefig(filename, bbox_inches="tight", pad_inches=0.2, dpi=600)


if __name__ == "__main__":

    from c4dynamics import rundoctests

    rundoctests(sys.modules[__name__])
