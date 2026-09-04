import sys
import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

# sys.path.append(".")
import c4dynamics as c4d

# ======= top level
# ------- level 2
# ~~~~~~~ level 3
# ^^^^^^^ level 4
# ******* level 5


class state:
    """Custom state object.

    A state object represents a state vector and other attributes
    that form an entity of a physical (dynamic) system.

    A custom state object means any set of state variables is possible,
    while pre-defined states from the :mod:`state library<c4dynamics.states.lib>`
    are ready to use out of the box.

    Keyword Arguments
    =================

    **kwargs : float or int
        Keyword arguments representing the variables and their initial conditions.
        Each key is a variable name and each value is its initial condition.
        For example: :code:`s = c4d.state(x = 0, theta = 3.14)`


    See Also
    ========
    .states


    Examples
    ========


    ``Pendulum``

    .. code::

      >>> s = c4d.state(theta = 10 * c4d.d2r, omega = 0) # doctest: +IGNORE_OUTPUT
      [ θ  ω ]

    ``Strapdown navigation system``

    .. code::

      >>> s = c4d.state(x = 0, y = 0, z = 0,
      ...     vx = 0, vy = 0, vz = 0, q0 = 0,
      ...     q1 = 0, q2 = 0, q3 = 0, bax = 0,
      ...     bay = 0, baz = 0)   # doctest: +IGNORE_OUTPUT
      [ x  y  z  vx  vy  vz  q0  q1  q2  q3  bax  bay  baz ]

    ``Objects tracker``

    .. code::

      >>> s = c4d.state(x = 960, y = 540, w = 20, h = 10)   # doctest: +IGNORE_OUTPUT
      [ x  y  w  h ]

    ``Aircraft``

    .. code::

      >>> s = c4d.state(x = 0, y = 0, z = 0, vx = 0,
      ...       vy = 0, vz = 0, phi = 0, theta = 0,
      ...       psi = 0, p = 0, q = 0, r = 0)   # doctest: +IGNORE_OUTPUT
      [ x  y  z  vx  vy  vz  φ  θ  Ψ  p  q  r ]

    ``Self-driving car``

    .. code::

      >>> s = c4d.state(x = 0, y = 0, v = 0, theta = 0, omega = 0)   # doctest: +IGNORE_OUTPUT
      [ x  y  v  θ  ω ]

    ``Robot arm``

    .. code::

      >>> s = c4d.state(theta1 = 0, theta2 = 0, omega1 = 0, omega2 = 0)   # doctest: +IGNORE_OUTPUT
      [ θ1  θ2  ω1  ω2 ]
    """

    # TODO: define state space, i.e. define the state properties such
    #       as scope, limits, continuouty. etc.
    #     probably basicaly for rl algos.

    # Α α # Β β # Γ γ # Δ δ # Ε ε # Ζ ζ # Η η # Θ θ # Ι ι
    # Κ κ # Λ λ # Μ μ # Ν ν # Ξ ξ # Ο ο # Π π # Ρ ρ # Σ σ/ς
    # Τ τ # Υ υ # Φ φ # Χ χ # Ψ ψ # Ω ω
    _greek_unicode = (
        ("alpha", "\u03b1"),
        ("beta", "\u03b2"),
        ("gamma", "\u03b3"),
        ("delta", "\u03b4"),
        ("epsilon", "\u03b5"),
        ("zeta", "\u03b6"),
        ("eta", "\u03b7"),
        ("theta", "\u03b8"),
        ("iota", "\u03b9"),
        ("kappa", "\u03ba"),
        ("lambda", "\u03bb"),
        ("mu", "\u03bc"),
        ("nu", "\u03bd"),
        ("xi", "\u03be"),
        ("omicron", "\u03bf"),
        ("pi", "\u03c0"),
        ("rho", "\u03c1"),
        ("sigma", "\u03c3"),
        ("final_sigma", "\u03c2"),
        ("tau", "\u03c4"),
        ("upsilon", "\u03c5"),
        ("phi", "\u03c6"),
        ("chi", "\u03c7"),
        ("psi", "\u03c8"),
        ("omega", "\u03c9"),
        ("Alpha", "\u0391"),
        ("Beta", "\u0392"),
        ("Gamma", "\u0393"),
        ("Delta", "\u0394"),
        ("Epsilon", "\u0395"),
        ("Zeta", "\u0396"),
        ("Eta", "\u0397"),
        ("Theta", "\u0398"),
        ("Iota", "\u0399"),
        ("Kappa", "\u039a"),
        ("Lambda", "\u039b"),
        ("Mu", "\u039c"),
        ("Nu", "\u039d"),
        ("Xi", "\u039e"),
        ("Omicron", "\u039f"),
        ("Pi", "\u03a0"),
        ("Rho", "\u03a1"),
        ("Sigma", "\u03a3"),
        ("Tau", "\u03a4"),
        ("Upsilon", "\u03a5"),
        ("Phi", "\u03a6"),
        ("Chi", "\u03a7"),
        ("Psi", "\u03a8"),
        ("Omega", "\u03a9"),
    )

    _reserved_keys = (
        "X",
        "X_",
        "X0",
        "dist",
        "vel_mag",
        "Position",
        "Velocity",
        "norm",
        "normalize",
        "data",
        "_data",
        "_prmdata",
        "_didx",
    )

    _STATETYPE = np.float64

    def __init__(self, **kwargs):
        # TODO enable providing type for the setter.X output.

        # the problem with this is it cannot be used with seeker and other c4d objects
        # that takes datapoint objects.
        # beacuse sometimes it misses attributes such as y, z that are necessary for poistion etc

        # alternatives:

        # 1. all the attributes always exist but they are muted and most importantly not reflected
        #   in the state vector when doing dp.X

        # 2. they will not be used in this fucntions.

        # 3. the user must provide his implementations for poisiton velcity etc.. like used in
        # the dist() function that there i
        #   encountered the problem.

        # self._dtype = np.float32
        self._data = []  # for permanent class variables (t, x, y .. )
        self._prmdata = {}  # for user additional variables

        self._didx = {"t": 0}

        # if any of the input vars is floating point, then it casts the entire state type
        # (no partial types array)
        # if all of them are integer, it casts the staet type.
        # i think maybe the best way to achieve it is by generating a numpy array
        # and check its dtype:
        self._X = np.array(list(kwargs.values()), dtype=self._STATETYPE)
        # self._X = np.zeros(len(kwargs)) #, dtype = np.float64)
        # TODO: now the type is determined by the initial values.
        #       if any is floating point then the entire state is float.
        #       otherwise, integer.
        #       however, there cannot be partial types such as some variables.
        #       this must be covered in the documentation.

        for i, (k, v) in enumerate(kwargs.items()):
            if k in self._reserved_keys:
                raise ValueError(
                    f"{k} is a reserved key. Keys {self._reserved_keys} "
                    "cannot use as variable names."
                )
            # setattr(self, k, v)
            setattr(self, k + "0", v)
            self._didx[k] = 1 + i
            self._X[i] = v

    def __str__(self):

        self_str = "[ "
        for i, s in enumerate(self._didx.keys()):
            if s == "t":
                continue
            s = dict(self._greek_unicode).get(s, s)
            if i < len(self._didx.keys()) - 1:
                self_str += s + "  "
            else:
                self_str += s + " ]"

        return self_str

    def __repr__(self):
        # NOTE i think maybe to switch repr and str so
        # when i print >>> s   it show the variables and when
        # i do >>> print(s)   it show the entire description.
        # but then i need to iterate all the examples and remove the print from state presentations
        param_names = ", ".join(self._prmdata.keys())

        # FIXME Parameters is wrong. because currently parameters are determined only
        # by those which stored at least once with the storeparams method.
        return (
            f"<state object>\n"
            f"State Variables: {self.__str__()}\n"
            f"Initial Conditions (X0): {self.X0}\n"
            f"Current State Vector (X): {self.X}\n"
            f"Parameters: {param_names if param_names else 'None'}"
        )

    #
    # state operations
    ##

    def __getattr__(self, var):
        if var in self._didx:
            return self._X[self._didx[var] - 1]
        else:
            # default Python behavior
            raise AttributeError(f"{var} not found")

    def __setattr__(self, var, val):
        # intercept known state variables
        if "_didx" in self.__dict__ and var in self._didx:

            # if np.array(val).dtype != self._X.dtype:
            #   warnings.warn(
            #         f"Type mismatch! Casting X to {np.array(val).dtype}.
            #           (currently: {self._X.dtype})."
            #     , c4d.c4warn)
            #   self._X = self._X.astype(np.array(val).dtype)

            self._X[self._didx[var] - 1] = np.array(val).astype(self._STATETYPE).item()
        else:
            # regular attribute assignment
            super().__setattr__(var, val)

    @property
    def X(self) -> NDArray[Any]:
        """Gets and sets the state vector variables.

        Parameters
        ----------
        x : array_like
            Values vector to set the variables of the state.

        Returns
        -------
        out : numpy.array
            Values vector of the state.



        Examples
        --------

        Getter:

        .. code::

          >>> s = c4d.state(x1 = 0, x2 = -1)
          >>> s.X   # doctest: +NUMPY_FORMAT
          [0  -1]


        Setter:

        .. code::

          >>> s = c4d.state(x1 = 0, x2 = -1)
          >>> s.X += [0, 1] # equivalent to: s.X = s.X + [0, 1]
          >>> s.X   # doctest: +NUMPY_FORMAT
          [0  0]


        :class:`datapoint <c4dynamics.states.lib.datapoint.datapoint>` getter - setter:

        .. code::

          >>> dp = c4d.datapoint()
          >>> dp.X   # doctest: +NUMPY_FORMAT
          [0  0  0  0  0  0]
          >>> #       x     y    z  vx vy vz
          >>> dp.X = [1000, 100, 0, 0, 0, -100]
          >>> dp.X   # doctest: +NUMPY_FORMAT
          [1000  100  0  0  0  -100]
        """

        # xout = []

        # for k in self._didx.keys():
        #   if k == 't': continue
        #   # the alteast_1d() + the flatten() is necessary to
        #   # cope with non-homogenuous array
        #   # xout.append(np.atleast_1d(eval('self.' + k)))
        #   xout.append(np.atleast_1d(getattr(self, k)))

        # #
        # # XXX why float64? maybe it's just some default unlsess anything else required.
        # # pixelpoint:override the .X property: will distance the devs from a datapoint class.
        # #  con: much easier
        # #
        # # return np.array(xout).ravel().astype(self._dtype)
        # return np.array(xout).flatten().astype(np.float64)
        return self._X

    @X.setter
    def X(self, Xin):
        # Xin = np.atleast_1d(Xin).flatten()
        # i think to replace flatten() with ravel() is
        # safe also here because Xin is iterated over its elements which are
        # mutable. but lets keep tracking.
        Xin = np.atleast_1d(Xin).ravel().astype(self._STATETYPE)

        # if Xin.dtype != self._X.dtype:
        #   warnings.warn(
        #         f"Type mismatch! Casting X to {Xin.dtype}. (currently: {self._X.dtype})."
        #     , c4d.c4warn)
        #   self._X = self._X.astype(Xin.dtype)

        if len(Xin) != len(self._X):
            raise ValueError("Length mismatch")

        # for i, k in enumerate(self._didx.keys()):
        #   if k == 't': continue
        #   if i > xlen: break
        #   setattr(self, k, Xin[i - 1])

        self._X[:] = Xin

    @property
    def X0(self):
        """Returns the initial conditions of the state vector.

        The initial conditions are determined at the stage of constructing
        the state object.
        Modifying the initial conditions is possible by direct assignment
        of the state variable with a '0' suffix. For a state variable
        :math:`s.x`, its initial condition is modifyied by:
        :code:`s.x0 = x0`, where :code:`x0` is an arbitrary parameter.


        Returns
        -------
        out : numpy.array
            An array representing the initial values of the state variables.


        Examples
        --------

        .. code::

          >>> s = c4d.state(x1 = 0, x2 = -1)
          >>> s.X += [0, 1]
          >>> s.X0   # doctest: +NUMPY_FORMAT
          [0  -1]


        .. code::

          >>> s = c4d.state(x1 = 1, x2 = 1)
          >>> s.X0   # doctest: +NUMPY_FORMAT
          [1  1]
          >>> s.x10 = s.x20 = 0
          >>> s.X0   # doctest: +NUMPY_FORMAT
          [0  0]
        """
        xout = []

        for k in self._didx:
            if k == "t":
                continue
            # xout.append(eval('self.' + k + '0'))
            xout.append(getattr(self, k + "0"))

        return np.array(xout)

    def addvars(self, **kwargs):
        """
        Add state variables.

        Adding variables to the state outside the :class:`state <c4dynamics.states.state.state>`
        constructor is possible by using :meth:`addvars() <c4dynamics.states.state.state.addvars>`.


        Parameters
        ----------

        **kwargs : float or int
            Keyword arguments representing the variables and their initial conditions.
            Each key is a variable name and each value is its initial condition.


        Note
        ----
        If :meth:`store() <c4dynamics.states.state.state.store>` is called before
        adding the new variables, then the time histories of the new states
        are filled with zeros to maintain the same size as the other state variables.



        Examples
        --------

        .. code::

          >>> s = c4d.state(x = 0, y = 0)
          >>> print(s)
          [ x  y ]
          >>> s.addvars(vx = 0, vy = 0)
          >>> print(s)
          [ x  y  vx  vy ]

        calling :meth:`store() <c4dynamics.states.state.state.store>` before
        adding the new variables:

        .. code::

          >>> s = c4d.state(x = 1, y = 1)
          >>> s.store()
          >>> s.store()
          >>> s.store()
          >>> s.addvars(vx = 0, vy = 0)
          >>> s.data('x')[1]   # doctest: +NUMPY_FORMAT
          [1  1  1]
          >>> s.data('vx')[1]   # doctest: +NUMPY_FORMAT
          [0  0  0]
        """
        # NOTE: maybe useless because user can always create a new state object.

        b0 = len(self._didx)

        self._X = np.hstack((self._X, np.zeros(len(kwargs), dtype=self._STATETYPE)))

        for i, (k, v) in enumerate(kwargs.items()):
            if k in self._reserved_keys:
                raise ValueError(
                    f"{k} is a reserved key. Keys {self._reserved_keys} cannot use "
                    "as variable names."
                )

            setattr(self, k + "0", v)
            self._didx[k] = b0 + i  # including t
            self._X[b0 + i - 1] = v  # not including t

        if self._data:
            # add zero columns at the size of the new vars to avoid wrong broadcasting.
            dataarr = np.array(self._data)
            dataarr = np.hstack((dataarr, np.zeros((dataarr.shape[0], b0))))
            self._data = dataarr.tolist()

    #
    # data management operations
    ##

    def store(self, t=-1):
        """
        Stores the current state.

        The current state is defined by the vector of variables
        as given by :attr:`state.X <c4dynamics.states.state.state.X>`.
        :meth:`store() <c4dynamics.states.state.state.store>` is used to store the
        instantaneous state variables.


        Parameters
        ----------
        t : float or int, optional
            Time stamp for the stored state.

        Note
        ----
        1. Time `t` is an optional parameter with a default value of :math:`t = -1`.
        The time is always appended at the head of the array to store. However,
        if `t` is not given, default :math:`t = -1` is stored instead.

        2. The method :meth:`store() <c4dynamics.states.state.state.store>` goes together with
        the methods :meth:`data() <c4dynamics.states.state.state.data>`
        and :meth:`timestate() <c4dynamics.states.state.state.timestate>` as input and outputs.

        3. :meth:`store() <c4dynamics.states.state.state.store>` only stores
        state variables (those construct :attr:`state.X <c4dynamics.states.state.state.X>`).
        For other parameters, use :meth:`storeparams() <c4dynamics.states.state.state.storeparams>`.


        Examples
        --------

        .. code::

          >>> s = c4d.state(x = 1, y = 0, z = 0)
          >>> s.store()


        **Store with time stamp:**

        .. code::

          >>> s = c4d.state(x = 1, y = 0, z = 0)
          >>> s.store(t = 0.5)



        **Store in a for-loop:**

        .. code::

          >>> s = c4d.state(x = 1, y = 0, z = 0)
          >>> for t in np.linspace(0, 1, 3):
          ...   s.X = np.random.rand(3)
          ...   s.store(t)


        Usage of :meth:`store() <c4dynamics.states.state.state.store>`
        inside a program with a :class:`datapoint <c4dynamics.states.lib.datapoint.datapoint>`
        from the :mod:`states library <c4dynamics.states.lib>`:


        .. code::

          >>> t = 0
          >>> dt = 1e-3
          >>> h0 = 100
          >>> dp = c4d.datapoint(z = h0)
          >>> while dp.z >= 0:
          ...   dp.inteqm([0, 0, -c4d.g_ms2], dt) # doctest: +IGNORE_OUTPUT
          ...   t += dt
          ...   dp.store(t)
          >>> for z in dp.data('z'):   # doctest: +IGNORE_OUTPUT
          ...   print(z)
          99.9999950
          99.9999803
          99.9999558
          ...
          0.00033469
          -0.0439570
        """
        # FIXME: make that when t is not provided a counter is used instead.

        self._data.append([t] + self.X.tolist())

    def storeparams(self, params, t=-1.0):
        """
        Stores parameters.

        Parameters are data attributes which are not part of the state vector.
        :meth:`storeparams() <c4dynamics.states.state.state.storeparams>` is
        used to store the instantaneous parameters.

        Parameters
        ----------
        params : str or list of str
            Name or names of the parameters to store.
        t : float or int, optional
            Time stamp for the stored state.

        Note
        ----
        1. Time `t` is an optional parameter with a default value of :math:`t = -1`.
        The time is always appended at the head of the array to store. However,
        if `t` is not given, default :math:`t = -1` is stored instead.

        2. The method :meth:`storeparams() <c4dynamics.states.state.state.storeparams>`
        goes together with the method :meth:`data() <c4dynamics.states.state.state.data>`
        as input and output.


        Examples
        --------

        .. code::

          >>> s = c4d.state(x = 100, vx = 10)
          >>> s.mass = 25
          >>> s.storeparams('mass')
          >>> s.data('mass')[1]   # doctest: +NUMPY_FORMAT
          [25]


        **Store with time stamp:**

        .. code::

          >>> s = c4d.state(x = 100, vx = 10)
          >>> s.mass = 25
          >>> s.storeparams('mass', t = 0.1)
          >>> s.data('mass')
          (array([0.1]), array([25.]))


        **Store multiple parameters:**

        .. code::

          >>> s = c4d.state(x = 100, vx = 10)
          >>> s.x_std = 5
          >>> s.vx_std = 10
          >>> s.storeparams(['x_std', 'vx_std'])
          >>> s.data('x_std')[1]   # doctest: +NUMPY_FORMAT
          [5]
          >>> s.data('vx_std')[1] # doctest: +NUMPY_FORMAT
          [10]



        **Objects classification:**

        .. code::

          >>> s = c4d.state(x = 25, y = 25, w = 20, h = 10)
          >>> np.random.seed(44)
          >>> for i in range(3):
          ...   s.X += 1
          ...   s.w, s.h = np.random.randint(0, 50, 2)
          ...   if s.w > 40 or s.h > 20:
          ...     s.class_id = 'truck'
          ...   else:
          ...     s.class_id = 'car'
          ...   s.store() # stores the state
          ...   s.storeparams('class_id') # store the class_id parameter
          >>> print('   x    y    w    h    class')  # doctest: +IGNORE_OUTPUT
          >>> print(np.hstack((s.data()[:, 1:].astype(int),
          ...       np.atleast_2d(s.data('class_id')[1]).T)))  # doctest: +IGNORE_OUTPUT
          x   y   w   h   class
          26  26  20  35  truck
          27  27  49  45  car
          28  28  3   32  car


        The `morphospectra` implements a custom method `getdim` to update
        the dimension parameter `dim` with respect to the position coordinates:

        .. code::

          >>> import types
          >>> #
          >>> def getdim(s):
          ...   if s.X[2] != 0:
          ...     # z
          ...     s.dim = 3
          ...   elif s.X[1] != 0:
          ...     # y
          ...     s.dim = 2
          ...   elif s.X[0] != 0:
          ...     # x
          ...     s.dim = 1
          ...   else:
          ...     # none
          ...     s.dim = 0
          >>> #
          >>> morphospectra = c4d.state(x = 0, y = 0, z = 0)
          >>> morphospectra.dim = 0
          >>> morphospectra.getdim = types.MethodType(getdim, morphospectra)
          >>> #
          >>> for r in range(10):
          ...   morphospectra.X = np.random.choice([0, 1], 3)
          ...   morphospectra.getdim()
          ...   morphospectra.store()
          ...   morphospectra.storeparams('dim')
          >>> #
          >>> print('x y z  | dim')  # doctest: +IGNORE_OUTPUT
          >>> print('------------')  # doctest: +IGNORE_OUTPUT
          >>> for x, dim in zip(morphospectra.data().astype(int)[:, 1 : 4].tolist(),
          ...   morphospectra.data('dim')[1].tolist()):  # doctest: +IGNORE_OUTPUT
          ...       print(*(x + [' | '] + [dim]))
          x y z  | dim
          ------------
          0 1 0  |  2
          1 1 0  |  2
          1 0 0  |  1
          0 1 1  |  3
          1 0 0  |  1
          1 1 1  |  3
          1 1 0  |  2
          1 1 0  |  2
          1 0 1  |  3
          1 0 1  |  3
        """
        # TODO show example of multiple vars
        # maybe the example of the kalman variables store
        #   from the detect track exmaple.
        # TODO add test if the params is not 0 or 1 dim throw error or warning. why?
        # TODO document about the two if's down here: nonscalar param and empty param.
        # TODO add an option to provide the values to store becuase sometimes its not realy using
        # the state on realtime but just for storage
        from functools import reduce

        lparams = params if isinstance(params, list) else [params]
        for p in lparams:
            if p not in self._prmdata:
                self._prmdata[p] = []

            vval = np.atleast_1d(reduce(getattr, p.split("."), self)).flatten()
            if len(vval) == 0:  # empty, convert to none to keep homogenuous array
                vval = np.atleast_1d(np.nan)
            elif len(vval) > 1:
                # c4d.cprint(f'{p} is not a scalar. only first item is stored', 'r')
                warnings.warn(f"{p} is not a scalar. Only first item is stored", c4d.c4warn)
                vval = vval[:1]
            self._prmdata[p].append([t] + vval.tolist())

    def data(self, var=None, scale=1.0):
        """
        Returns arrays of stored time and data.

        :meth:`data() <c4dynamics.states.state.state.data>` returns a tuple containing two
        numpy arrays:
        the first consists of timestamps, and the second
        contains the values of a `var` corresponding to those timestamps.

        `var` may be each one of the state variables or the parameters.

        If `var` is not introduced, :meth:`data() <c4dynamics.states.state.state.data>`
        returns a single array of the entire state histories.

        If data were not stored, :meth:`data() <c4dynamics.states.state.state.data>`
        returns an empty array.


        Parameters
        ----------
        var : str
            The name of the variable or parameter of the required histories.

        scale : float or int, optional
            A scaling factor to apply to the variable values, by default 1.

        Returns
        -------
        out : array or tuple of numpy arrays
            if `var` is introduced, `out` is a tuple of a timestamps array
            and an array of `var` values corresponding to those timestamps.
            If `var` is not introduced, then :math:`n \\times m+1` numpy array is returned,
            where `n` is the number of stored samples, and `m+1` is the
            number of state variables and times.



        Examples
        --------

        Get all stored data:

        .. code::

          >>> np.random.seed(100) # to reproduce results
          >>> s = c4d.state(x = 1, y = 0, z = 0)
          >>> for t in np.linspace(0, 1, 3):
          ...   s.X = np.random.rand(3)
          ...   s.store(t)
          >>> s.data() # doctest: +NUMPY_FORMAT
          [[0.   0.543  0.278  0.424]
           [0.5  0.845  0.005  0.121]
           [1.   0.671  0.826  0.137]]


        Data of a variable:

        .. code::

          >>> time, x_data = s.data('x')
          >>> time  # doctest: +NUMPY_FORMAT
          [0.  0.5  1.]
          >>> x_data  # doctest: +NUMPY_FORMAT
          [0.543  0.845  0.671]
          >>> s.data('y')[1]  # doctest: +NUMPY_FORMAT
          [0.278  0.005  0.826]


        Get data with scaling:

        .. code::

          >>> s = c4d.state(phi = 0)
          >>> for p in np.linspace(0, c4d.pi):
          ...   s.phi = p
          ...   s.store()
          >>> s.data('phi', c4d.r2d)[1]  # doctest: +IGNORE_OUTPUT
          [0  3.7  7.3  ...  176.3  180]


        Data of a parameter

        .. code::

          >>> s = c4d.state(x = 100, vx = 10)
          >>> s.mass = 25
          >>> s.storeparams('mass', t = 0.1)
          >>> s.data('mass')
          (array([0.1]), array([25.]))
        """

        # FIXME: check if var is a state variable

        if var is None:
            # return all
            # XXX not sure this is applicable to the new change where arrays\ matrices
            #   are also stored.
            #   in fact the matrices are not stored in _data but in _prmdata
            return np.array(self._data)

        idx = self._didx.get(var, -1)
        if idx >= 0:
            if not self._data:
                # empty array
                # c4d.cprint('Warning: no history of state samples.', 'r')
                warnings.warn("""No history of state samples.""", c4d.c4warn)

                return np.array([])

            if idx == 0:
                return np.array(self._data)[:, 0]

            return (np.array(self._data)[:, 0], np.array(self._data)[:, idx] * scale)

        # else \ user defined variables
        if var not in self._prmdata:
            # c4d.cprint('Warning: no history samples of ' + var + '.', 'r')
            warnings.warn(f"""No history samples of {var}.""", c4d.c4warn)

            return np.array([])

        # if the var is text, dont multiply by scale
        if np.issubdtype(np.array(self._prmdata[var])[:, 1].dtype, np.number):
            return (np.array(self._prmdata[var])[:, 0], np.array(self._prmdata[var])[:, 1] * scale)
        else:
            return (np.array(self._prmdata[var])[:, 0], np.array(self._prmdata[var])[:, 1])

    def timestate(self, t):
        """
        Returns the state as stored at time `t`.

        The method searches the closest time
        to time `t` in the sampled histories and
        returns the state that stored at the time.

        If data were not stored returns None.

        Parameters
        ----------
        t : float or int
            The time at the required sample.

        Returns
        -------
        X : numpy.array
            An array of the state vector
            :attr:`state.X <c4dynamics.states.state.state.X>`
            at time `t`.


        Examples
        --------

        .. code::

          >>> s = c4d.state(x = 0, y = 0, z = 0)
          >>> for t in np.linspace(0, 1, 3):
          ...   s.X += 1
          ...   s.store(t)
          >>> s.timestate(0.5) # doctest: +NUMPY_FORMAT
          [2  2  2]


        .. code::

          >>> s = c4d.state(x = 1, y = 0, z = 0)
          >>> s.timestate(0.5)  # doctest: +IGNORE_OUTPUT
          Warning: no history of state samples.
          None
        """

        # TODO what about throwing a warning when dt is too long?
        # \\ what is dt and so what if its long?
        times = self.data("t")
        if len(times) == 0:
            X = None
        else:
            idx = min(range(len(times)), key=lambda i: abs(times[i] - t))
            X = np.array(self._data[idx][1:])

        return X

    def plot(self, var, scale=1, ax=None, filename=None, darkmode=True, block=False, **kwargs):
        """
        Draws plots of variable evolution over time.

        This method plots the evolution of a state variable over time.
        The resulting plot can be saved to a directory if specified.

        Parameters
        ----------
        var : str
            The name of the variable or parameter to be plotted.

        scale : float or int, optional
            A scaling factor to apply to the variable values. Defaults to `1`.

        ax : matplotlib.axes.Axes, optional
            An existing Matplotlib axis to plot on.
            If None, a new figure and axis will be created, by default None.

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


        Returns
        -------
        ax : matplotlib.axes.Axes.
            Matplotlib axis of the derived plot.


        Note
        ----
        - The default `color` is set to `'m'` (magenta).
        - The default `linewidth` is set to `1.2`.


        Examples
        --------

        Import required packages:

        .. code::

          >>> import c4dynamics as c4d
          >>> from matplotlib import pyplot as plt
          >>> import numpy as np

        Plot an arbitrary state variable and save:

        .. code::

          >>> s = c4d.state(x = 0, y = 0)
          >>> s.store()
          >>> for _ in range(100):
          ...   s.x = np.random.randint(0, 100, 1)
          ...   s.store()
          >>> s.plot('x', filename = 'x.png')   # doctest: +IGNORE_OUTPUT
          >>> plt.show()

        .. figure:: /_examples/state/plot_x.png


        **Interactive mode:**

        .. code::

          >>> s.plot('x')   # doctest: +IGNORE_OUTPUT
          >>> plt.show(block = True)


        **Dark mode off:**

          >>> s = c4d.state(x = 0)
          >>> s.xstd = 0.2
          >>> for t in np.linspace(-2 * c4d.pi, 2 * c4d.pi, 1000):
          ...   s.x = c4d.sin(t) + np.random.randn() * s.xstd
          ...   s.store(t)
          >>> s.plot('x', darkmode = False)    # doctest: +IGNORE_OUTPUT
          >>> plt.show()

        .. figure:: /_examples/state/plot_darkmode.png



        **Scale plot:**

        .. code::

          >>> s = c4d.state(phi = 0)
          >>> for y in c4d.tan(np.linspace(-c4d.pi, c4d.pi, 500)):
          ...   s.phi = c4d.atan(y)
          ...   s.store()
          >>> s.plot('phi', scale = c4d.r2d)  # doctest: +IGNORE_OUTPUT
          >>> plt.gca().set_ylabel('deg') # doctest: +IGNORE_OUTPUT
          >>> plt.show()

        .. figure:: /_examples/state/plot_scale.png


        **Given axis:**

        .. code::

          >>> plt.subplots(1, 1)  # doctest: +IGNORE_OUTPUT
          >>> plt.plot(np.linspace(-c4d.pi, c4d.pi, 500) * c4d.r2d, 'm')   # doctest: +IGNORE_OUTPUT
          >>> s.plot('phi', scale = c4d.r2d, ax = plt.gca(), color = 'c')  # doctest: +IGNORE_OUTPUT
          >>> plt.gca().set_ylabel('deg')  # doctest: +IGNORE_OUTPUT
          >>> plt.legend(['θ', 'φ'])  # doctest: +IGNORE_OUTPUT
          >>> plt.show()

        .. figure:: /_examples/state/plot_axis.png


        Top view + side view - options of
        :class:`datapoint <c4dynamics.states.lib.datapoint.datapoint>`
        and :class:`rigidbody <c4dynamics.states.lib.rigidbody.rigidbody>` objects:

        .. code::

          >>> dt = 0.01
          >>> floating_balloon = c4d.datapoint(vx = 10 * c4d.k2ms)
          >>> floating_balloon.mass = 0.1
          >>> for t in np.arange(0, 10, dt):
          ...   floating_balloon.inteqm(forces = [0, 0, .05], dt = dt)   # doctest: +IGNORE_OUTPUT
          ...   floating_balloon.store(t)
          >>> floating_balloon.plot('side')
          >>> plt.gca().invert_yaxis()
          >>> plt.show()

        .. figure:: /_examples/state/plot_dp_inteqm3.png
        """

        from matplotlib import pyplot as plt

        #
        # points to consider
        # -------------------
        #
        # figsize
        # -------
        #
        # i think the challenge here is to get view + save images with
        # 1920 x 1080 pixels (Full HD)
        # 72 DPI (the standard for web images). no way. 72dpi is poor res. at least 300.
        # --> alternative: 960x540 600dpi.       #
        #
        # get the screen dpi to get the desired resolution.
        #
        #
        #
        # backends
        # --------
        #
        # non-interactive backends: Agg, SVG, PDF:
        #     When saving plots to files.
        #     include plt.show()
        # interactive backends:
        #     TkAgg, Qt5Agg, etc.
        #     dont include plt.show()
        #
        # Check Backend: matplotlib.get_backend().
        # avoid hardcoding backend settings.
        # Avoid using features that are specific to certain backends.
        # Users should be able to override backend settings.
        #
        # Test your plotting functions across different backends.
        #
        # finally i think the best soultion regardint backends is not
        # to do anythink and let the user select the backend from outside.
        ##
        if var not in self._didx:
            warnings.warn(f"""{var} is not a state variable.""", c4d.c4warn)
            return None
        if not self._data:
            warnings.warn(f"""No stored data for {var}.""", c4d.c4warn)
            return None

        if darkmode:
            plt.style.use("dark_background")
        else:
            plt.style.use("default")

        # plt.switch_backend('TkAgg')
        # plt.switch_backend('TkAgg')

        # try:
        #   from IPython import get_ipython
        #   if get_ipython() is None:
        #       return False
        #   else:
        #       return True
        # except ImportError:
        #   return False

        # Set default values in kwargs only if the user hasn't provided them
        kwargs.setdefault("color", "m")
        kwargs.setdefault("linewidth", 1.2)

        if ax is None:
            # factorsize = 4
            # aspectratio = 1080 / 1920
            # _, ax = plt.subplots(1, 1, dpi = 200
            #               , figsize = (factorsize, factorsize * aspectratio)
            #                       , gridspec_kw = {'left': 0.15, 'right': .9
            #                                           , 'top': .9, 'bottom': .2})
            _, ax = c4d._figdef()

        if not len(np.flatnonzero(self.data("t") != -1)):  # values for t weren't stored
            x = range(len(self.data("t")))  # t is just indices
            xlabel = "Samples"
        else:
            x = self.data("t")
            xlabel = "Time"
        y = (
            np.array(self._data)[:, self._didx[var]] * scale if self._data else np.empty(1)
        )  # used selection

        if dict(self._greek_unicode).get(var, "") != "":
            title = "$\\" + var + "$"
        else:
            title = var

        ax.plot(x, y, **kwargs)
        c4d.plotdefaults(ax, title, xlabel, "", 8)

        if filename:
            # plt.tight_layout(pad = 0)
            plt.savefig(filename, bbox_inches="tight", pad_inches=0.2, dpi=600)

        if block:
            # be = plt.get_backend()
            # plt.switch_backend('tkagg') # NOTE requires tk installed!
            plt.show(block=True)
            # plt.switch_backend(be)

        return ax

    def reset(self):
        self._data = []
        self._prmdata = {}

    #
    # math operations
    ##

    @property
    def norm(self):
        """
        Returns the Euclidean norm of the state vector.

        Returns
        -------
        out : float
            The computed norm of the state vector. The return type specifically is a numpy.float64.


        Examples
        --------

        .. code::

          >>> s = c4d.state(x1 = 1, x2 = -1)
          >>> s.norm  # doctest: +ELLIPSIS
          1.414...
        """
        return np.linalg.norm(self.X)

    @property
    def normalize(self):
        """
        Returns a unit vector representation of the state vector.

        Returns
        -------
        out : numpy.array
            A normalized vector of the same direction and shape as `self.X`, where
            the norm of the vector is `1`.


        Examples
        --------

        .. code::

          >>> s = c4d.state(x = 1, y = 2, z = 3)
          >>> s.normalize   # doctest: +NUMPY_FORMAT
          [0.267  0.534  0.801]
        """

        return self.X / np.linalg.norm(self.X)

    # cartesian operations

    @property
    def Position(self):
        """
        Returns a vector of position coordinates.

        If the state doesn't include any position coordinate (x, y, z),
        an empty array is returned.

        Note
        ----
        In the context of :attr:`Position <c4dynamics.states.state.state.Position>`,
        only x, y, z, (case sensitive) are considered position coordinates.


        Returns
        -------
        out : numpy.array
            A vector containing the values of three position coordinates.


        Examples
        --------

        .. code::

          >>> s = c4d.state(theta = 3.14, x = 1, y = 2)
          >>> s.Position  # doctest: +NUMPY_FORMAT
          [1  2  0]


        .. code::

          >>> s = c4d.state(theta = 3.14, x = 1, y = 2, z = 3)
          >>> s.Position  # doctest: +NUMPY_FORMAT
          [1  2  3]


        .. code::

          >>> s = c4d.state(theta = 3.14, z = -100)
          >>> s.Position  # doctest: +NUMPY_FORMAT
          [0  0  -100]


        .. code::

          >>> s = c4d.state(theta = 3.14)
          >>> s.Position   # doctest: +IGNORE_OUTPUT
          Position is valid when at least one cartesian coordinate variable (x, y, z) exists...
          []
        """

        if not self.cartesian():
            # c4d.cprint('Warning: Position is valid when at least one cartesian'
            #             ' coordinate variable (x, y, z) exists.', 'm')
            warnings.warn(
                """Position is valid when at least one cartesian """
                """coordinate variable (x, y, z) exists.""",
                c4d.c4warn,
            )
            return np.array([])

        return np.array([getattr(self, var, 0) for var in ["x", "y", "z"]])

    @property
    def Velocity(self):
        """
        Returns a vector of velocity coordinates.

        If the state doesn't include any velocity coordinate (vx, vy, vz),
        an empty array is returned.

        Note
        ----
        In the context of :attr:`Velocity <c4dynamics.states.state.state.Velocity>`,
        only vx, vy, vz, (case sensitive) are considered velocity coordinates.

        Returns
        -------
        out : numpy.array
            A vector containing the values of three velocity coordinates.


        Examples
        --------

        .. code::

          >>> s = c4d.state(x = 100, y = 0, vx = -10, vy = 5)
          >>> s.Velocity # doctest: +NUMPY_FORMAT
          [-10  5  0]


        .. code::

          >>> s = c4d.state(x = 100, vz = -100)
          >>> s.Velocity # doctest: +NUMPY_FORMAT
          [0  0  -100]


        .. code::

          >>> s = c4d.state(z = 100)
          >>> s.Velocity  # doctest: +IGNORE_OUTPUT
          Warning: Velocity is valid when at least one velocity coordinate
          variable (vx, vy, vz) exists.
          []
        """

        if self.cartesian() < 2:
            # c4d.cprint('Warning: Velocity is valid when at least one velocity '
            #             'coordinate variable (vx, vy, vz) exists.', 'm')
            warnings.warn(
                """Velocity is valid when at least one velocity """
                """coordinate variable (vx, vy, vz) exists.""",
                c4d.c4warn,
            )
            return np.array([])

        return np.array([getattr(self, var, 0) for var in ["vx", "vy", "vz"]])

    def dist(self, state2=None):
        """
        Euclidean distance.

        Calculates the Euclidean distance between the self state object and
        a second object `state2`. If `state2` is not provided, then the self
        Euclidean distance is calculated.


        When a second state object is provided:

        .. math::

          dist = \\sum_{k=x,y,z} (self.k - state2.k)^2


        Otherwise:

        .. math::

          dist = \\sum_{k=x,y,z} self.k^2



        Raises
        ------
        TypeError
            If the states don't include any position coordinate (x, y, z).


        Note
        ----
        1. The provided states must have at least oneposition coordinate (x, y, z).
        2. In the context of :meth:`dist() <c4dynamics.states.state.state.dist>`,
           x, y, z, (case sensitive) are considered position coordinates.



        Parameters
        ----------
        state2 : :class:`state <c4dynamics.states.state.state>`
            A second state object for which the relative distance is calculated.


        Returns
        -------
        out : float
            Euclidean norm of the distance vector. The return type specifically is a numpy.float64.


        Examples
        --------

        .. code::

          >>> import c4dynamics as c4d

        .. code::

          >>> s = c4d.state(theta = 3.14, x = 1, y = 1)
          >>> s.dist()   # doctest: +ELLIPSIS
          1.414...

        .. code::

          >>> s  = c4d.state(theta = 3.14, x = 1, y = 1)
          >>> s2 = c4d.state(x = 1)
          >>> s.dist(s2)
          1.0

        .. code::

          >>> s  = c4d.state(theta = 3.14, x = 1, y = 1)
          >>> s2 = c4d.state(z = 1)
          >>> s.dist(s2)   # doctest: +ELLIPSIS
          1.73...


        For final example, import required packages:

        .. code::

          >>> import numpy as np
          >>> from matplotlib import pyplot as plt


        Settings and initial conditions:

        .. code::

          >>> camera = c4d.state(x = 0, y = 0)
          >>> car    = c4d.datapoint(x = -100, vx = 40, vy = -7)
          >>> dist   = []
          >>> time   = np.linspace(0, 10, 1000)



        Main loop:

        .. code::

          >>> for t in time:
          ...   car.inteqm(np.zeros(3), time[1] - time[0]) # doctest: +IGNORE_OUTPUT
          ...   dist.append(camera.dist(car))


        Show results:

        .. code::

          >>> plt.plot(time, dist, 'm') # doctest: +IGNORE_OUTPUT
          >>> c4d.plotdefaults(plt.gca(), 'Distance', 'Time (s)', '(m)')
          >>> plt.show()

        .. figure:: /_examples/states/state_dist.png
        """

        if not self.cartesian():
            raise TypeError("state must have at least one position coordinate (x, y, or z)")

        if state2 is None:
            state2 = c4d.datapoint()
        else:
            if not hasattr(state2, "cartesian") or not state2.cartesian():
                raise TypeError(
                    "state2 must be a state object with at least one position"
                    " coordinate (x, y, or z)"
                )

        dist = 0
        for var in ["x", "y", "z"]:
            dist += (getattr(self, var, 0) - getattr(state2, var, 0)) ** 2

        return np.sqrt(dist)

    def vel_mag(self):
        """
        Velocity Magnitude.

        Calculates the magnitude of the object velocity :

        .. math::

          vel mag = \\sum_{k=v_x,v_y,v_z} self.k^2

        If the state doesn't include any velocity coordinate (vx, vy, vz),
        a `ValueError` is raised.



        Returns
        -------
        out : float
            Euclidean norm of the velocity vector. The return type specifically is a numpy.float64.


        Raises
        ------
        TypeError
            If the state does not include any velocity coordinate (vx, vy, vz).

        Note
        ----
        In the context of :meth:`vel_mag() <c4dynamics.states.state.state.vel_mag>`,
        vx, vy, vz, (case sensitive) are considered velocity coordinates.


        Examples
        --------

        .. code::

          >>> s = c4d.state(vx = 7, vy = 24)
          >>> s.vel_mag()
          25.0

        .. code::

          >>> s = c4d.state(x = 100, y = 0, vx = -10, vy = 7)
          >>> s.vel_mag()   # doctest: +ELLIPSIS
          12.2...


        Uncommenting the following line throws a type error:

        .. code::

          >>> s = c4d.state(x = 100, y = 0)
          >>> # s.vel_mag()
          TypeError: state must have at least one velocity coordinate (vx, vy, or vz)
        """

        if self.cartesian() < 2:
            raise TypeError("state must have at least one velocity coordinate (vx, vy, or vz)")

        return np.linalg.norm(self.Velocity)

    def cartesian(self):
        # TODO document!
        if any(hasattr(self, var) for var in ["vx", "vy", "vz"]):
            return 2
        elif any(hasattr(self, var) for var in ["x", "y", "z"]):
            return 1
        else:
            return 0


if __name__ == "__main__":

    from c4dynamics import rundoctests

    rundoctests(sys.modules[__name__])
