import numpy as np

# from scipy.special import erfinv
import sys

# sys.path.append(".")
import c4dynamics as c4d
import warnings
from typing import Optional


# TODO: change 'seeker' to 'direction seeker'
class seeker(c4d.rigidbody):
    """
    Direction seeker.

    The :class:`seeker` class models sensors that
    measure the direction to a target in terms of azimuth and elevation.
    Such sensors typically use electro-optical or laser technologies,
    though other technologies may also be used.

    A `seeker` object can operate in an ideal mode,
    providing precise direction measurements.
    Alternatively, in a non-ideal mode,
    the measurements may be affected by errors such as
    `scale factor`, `bias`, and `noise`,
    as defined by the errors model.
    A random variable generation mechanism allows
    for Monte Carlo simulations.


    Parameters
    ==========
    origin : :class:`rigidbody <c4dynamics.states.lib.rigidbody.rigidbody>`, optional
        A `rigidbody` object whose state vector :attr:`X <c4dynamics.states.state.state.X>`
        determines the seeker's initial position and attitude.
        Defaults: a `rigidbody` object with zeros vector, `X = numpy.zeros(12)`.
    isideal : bool, optional
        A flag indicating whether the errors model is off.
        Defaults False.


    Keyword Arguments
    =================
    bias_std : float
        The standard deviation of the bias error, [radians]. Defaults :math:`0.1°`.
    scale_factor_std : float
        The standard deviation of the scale factor error, [dimensionless].
        Defaults :math:`0.05 (= 5\\%)`.
    noise_std : float
        The standard deviation of the seeker angular noise, [radians].
        Default value for non-ideal seeker: :math:`0.4°`.
    dt : float
        The time-constant of the operational rate of the seeker
        (below which the seeker measures return None), [seconds]. Default value: :math:`dt = -1sec`
        (no limit between calls to :meth:`measure`).


    See Also
    ========
    .filters
    .eqm
    .radar



    **Functionality**

    At each sample the seeker returns measures based on the true geometry
    with the target.

    Let the relative coordinates in an arbitrary frame of reference:

    .. math::

      dx = target.x - seeker.x

      dy = target.y - seeker.y

      dz = target.z - seeker.z


    The relative coordinates in the seeker body frame are given by:

    .. math::

      x_b = [BR] \\cdot [dx, dy, dz]^T

    where :math:`[BR]` is a
    Body from Reference DCM (Direction Cosine Matrix)
    formed by the seeker three Euler angles. See the `rigidbody` section below.

    The azimuth and elevation measures are then the spatial angles:

    .. math::

      az = tan^{-1}{x_b[1] \\over x_b[0]}

      el = tan^{-1}{x_b[2] \\over \\sqrt{x_b[0]^2 + x_b[1]^2}}



    Where:

    - :math:`az` is the azimuth angle
    - :math:`el` is the elevation angle
    - :math:`x_b` is the target-radar position vector in radar body frame

    .. figure:: /_architecture/skr_definitions.svg

      Fig-1: Azimuth and elevation angles definition



    **Errors Model**

    The azimuth and elevation angles are subject to errors: scale factor, bias, and noise.

    - ``Bias``:
      represents a constant offset or deviation from the
      true value in the seeker's measurements.
      It is a systematic error that consistently affects the measured values.
      The bias of a `seeker` instance is a normally distributed variable with `mean = 0`
      and `std = bias_std`, where `bias_std` is a parameter with default value of `0.1°`.
    - ``Scale Factor``:
      a multiplier applied to the true value of a measurement.
      It represents a scaling error in the measurements made by the seeker.
      The scale factor of a `seeker` instance is
      a normally distributed variable
      with `mean = 0` and `std = scale_factor_std`,
      , where `scale_factor_std` is a parameter with default value of `0.05`.
    - ``Noise``:
      represents random variations or fluctuations in the measurements
      that are not systematic.
      The noise at each seeker sample (:meth:`measure`)
      is a normally distributed variable
      with `mean = 0` and `std = noise_std`, where `noise_std`
      is a parameter with default value of `0.4°`.


    The errors model generates random variables for each seeker instance,
    allowing for the simulation of different scenarios or variations in the seeker behavior
    in a technique known as Monte Carlo.
    Monte Carlo simulations leverage this randomness to statistically analyze
    the impact of these biases and scale factors over a large number of iterations,
    providing insights into potential outcomes and system reliability.






    The errors model can be disabled by applying `isideal = True` at the seeker construction stage.

    The erros model operates as follows: the particular parameters
    that form each error, for example the standard deviation of the bias error,
    may be determined at the stage of creating the sensor object:
    :code:`seeker = c4d.sensors.seeker(bias_std = 0.1 * c4d.d2r)`,
    where `c4d.d2r` is a conversion from degrees to radians.
    Then, the errors model generates a normal random variable
    and establishes the seeker bias error.
    However, the user can override the generated bias by
    using the :attr:`bias` property and determine the seeker bias error:
    :code:`seeker.bias = 1 * c4d.d2r`, to have a 1° bias error.




    **rigidbody**

    The seeker class is a subclass of
    :class:`rigidbody <c4dynamics.states.lib.rigidbody.rigidbody>`, i.e.
    it suggests attributes of position and attitude and the manipulation of them.

    As a fundamental propety, the
    rigidbody's state vector
    :attr:`X <c4dynamics.states.state.state.X>`
    sets the spatial coordinates of the seeker:

    .. math::

      X = [x, y, z, v_x, v_y, v_z, {\\varphi}, {\\theta}, {\\psi}, p, q, r]^T

    The first six coordinates determine the translational position and velocity of the seeker
    while the last six determine its angular attitude in terms of Euler angles and
    the body rates.

    Passing a rigidbody parameter as an `origin` sets
    the initial conditions of the seeker.



    **Construction**

    A seeker instance is created by making a direct call
    to the seeker constructor:

    .. code::

      >>> skr = c4d.sensors.seeker()

    Initialization of the instance does not require any
    mandatory arguments, but the seeker parameters can be
    determined using the \\**kwargs argument as detailed above.




    Examples
    ========

    Import required packages:

    .. code::

      >>> import c4dynamics as c4d
      >>> from matplotlib import pyplot as plt
      >>> import numpy as np


    **Target**


    For the examples below let's generate the trajectory of a target with constant velocity:

    .. code::

      >>> tgt = c4d.datapoint(x = 1000, y = 0, vx = -80 * c4d.kmh2ms, vy = 10 * c4d.kmh2ms)
      >>> for t in np.arange(0, 60, 0.01):
      ...   tgt.inteqm(np.zeros(3), .01) # doctest: +IGNORE_OUTPUT
      ...   tgt.store(t)

    The method :meth:`inteqm <c4dynamics.states.lib.datapoint.datapoint.inteqm>`
    of the
    :class:`datapoint <c4dynamics.states.lib.datapoint.datapoint>` class
    integrates the 3 degrees of freedom equations of motion with respect to
    the input force vector (`np.zeros(3)` here).

    .. figure:: /_examples/seeker/target.png

    Since the call to :meth:`measure` requires a target as a `datapoint` object
    we utilize a custom `create` function that returns a new `datapoint` object for
    a given `X` state vector in time.

    - :code:`c4d.kmh2ms` converts kilometers per hour to meters per second.
    - :code:`c4d.r2d` converts radians to degrees.
    - :code:`c4d.d2r` converts degrees to radians.


    **Origin**

    Let's also introduce a pedestal as an origin for the seeker.
    The pedestal is a `rigidbody` object with position and attitude:

    .. code::

      >>> pedestal = c4d.rigidbody(z = 30, theta = -1 * c4d.d2r)



    **Ideal Seeker**

    Measure the target position with an ideal seeker:

    .. code::

      >>> skr_ideal = c4d.sensors.seeker(origin = pedestal, isideal = True)
      >>> for x in tgt.data():
      ...   skr_ideal.measure(c4d.create(x[1:]), t = x[0], store = True)  # doctest: +IGNORE_OUTPUT

    Comparing the seeker measurements with the true target angles requires
    converting the relative position to the seeker body frame:

    .. code::

      >>> dx = tgt.data('x')[1] - skr_ideal.x
      >>> dy = tgt.data('y')[1] - skr_ideal.y
      >>> dz = tgt.data('z')[1] - skr_ideal.z
      >>> Xb = np.array([skr_ideal.BR @ [X[1] - skr_ideal.x, X[2] - skr_ideal.y,
      ...       X[3] - skr_ideal.z] for X in tgt.data()])

    where :attr:`skr_ideal.BR <c4dynamics.states.lib.rigidbody.rigidbody.BR>` is a
    Body from Reference DCM (Direction Cosine Matrix)
    formed by the seeker three Euler angles

    Now `az_true` is the true target azimuth angle with respect to the seeker, and
    `el_true` is the true elevation angle (`atan2d` is an aliasing of `numpy's arctan2`
    with a modification returning the angles in degrees):

    .. code::

      >>> az_true = c4d.atan2d(Xb[:, 1], Xb[:, 0])
      >>> el_true = c4d.atan2d(Xb[:, 2], c4d.sqrt(Xb[:, 0]**2 + Xb[:, 1]**2))
      >>> # plot results
      >>> fig, axs = plt.subplots(2, 1)  # doctest: +IGNORE_OUTPUT
      >>> axs[0].plot(tgt.data('t'), az_true,
      ...     label = 'target'
      ... )  # doctest: +IGNORE_OUTPUT
      >>> axs[0].plot(*skr_ideal.data('az', scale = c4d.r2d),
      ...     label = 'seeker'
      ... )  # doctest: +IGNORE_OUTPUT
      >>> axs[1].plot(tgt.data('t'), el_true)  # doctest: +IGNORE_OUTPUT
      >>> axs[1].plot(*skr_ideal.data('el', scale = c4d.r2d))  # doctest: +IGNORE_OUTPUT

    .. figure:: /_examples/seeker/ideal.png


    **Non-ideal Seeker**


    Measure the target position with a *non-ideal* seeker.
    The seeker's errors model introduces bias, scale factor, and
    noise that corrupt the measurements.

    To reproduce the result, let's set the random generator seed (42 is arbitrary):

    .. code::

      >>> np.random.seed(42)

    .. code::

      >>> skr = c4d.sensors.seeker(origin = pedestal)
      >>> for x in tgt.data():
      ...   skr.measure(c4d.create(x[1:]), t = x[0], store = True) # doctest: +IGNORE_OUTPUT


    Results with respect to an ideal seeker:

    .. code::

      >>> fig, axs = plt.subplots(2, 1)
      >>> axs[0].plot(*skr_ideal.data('az', scale = c4d.r2d),
      ...       label = 'ideal'
      ... )  # doctest: +IGNORE_OUTPUT
      >>> axs[0].plot(*skr.data('az', scale = c4d.r2d),
      ...     label = 'non-ideal'
      ... )  # doctest: +IGNORE_OUTPUT
      >>> axs[1].plot(*skr_ideal.data('el', scale = c4d.r2d))  # doctest: +IGNORE_OUTPUT
      >>> axs[1].plot(*skr.data('el', scale = c4d.r2d))  # doctest: +IGNORE_OUTPUT

    .. figure:: /_examples/seeker/nonideal.png


    The bias, scale factor, and noise that used to generate these measures
    can be examined by:

    .. code::

      >>> skr.bias * c4d.r2d # doctest: +ELLIPSIS
      -0.01...
      >>> skr.scale_factor # doctest: +ELLIPSIS
      1.02...
      >>> skr.noise_std * c4d.r2d
      0.4

    Points to consider:

    - The scale factor error increases with the angle, such that for a :math:`5%`
      scale factor,
      the error of :math:`Azimuth = 100°` is :math:`5°`, whereas the error for
      :math:`Elevation = -15°` is only :math:`-0.75°`.
    - The standard deviation of the noise in the two channels is the same.
      However, as the `Elevation` values are confined to a smaller range, the effect
      appears more pronounced there.


    **Rotating Seeker**


    Measure the target position with a rotating seeker.
    The seeker origin is yawing (performed by the increment of :math:`\\psi`)
    in the direction of the target motion:


    .. code::

      >>> skr = c4d.sensors.seeker(origin = pedestal)
      >>> for x in tgt.data():
      ...   skr.psi += .02 * c4d.d2r
      ...   skr.measure(c4d.create(x[1:]), t = x[0], store = True) # doctest: +IGNORE_OUTPUT
      ...   skr.store(x[0])

    The seeker yaw angle:

    .. figure:: /_examples/seeker/psi.png

    And the target angles with respect to the yawing seeker are:

    .. code::

      >>> fig, axs = plt.subplots(2, 1)
      >>> axs[0].plot(*skr_ideal.data('az', c4d.r2d),
      ...     label = 'ideal static seeker'
      ... )  # doctest: +IGNORE_OUTPUT
      >>> axs[0].plot(*skr.data('az', c4d.r2d),
      ...     label = 'non-ideal yawing seeker'
      ... )  # doctest: +IGNORE_OUTPUT
      >>> axs[1].plot(*skr_ideal.data('el', scale = c4d.r2d))  # doctest: +IGNORE_OUTPUT
      >>> axs[1].plot(*skr.data('el', scale = c4d.r2d))  # doctest: +IGNORE_OUTPUT

    .. figure:: /_examples/seeker/yawing.png


    - The rotation of the seeker with the target direction
      keeps the azimuth angle limited, such that non-rotating seekers
      with limited FOV (field of view)
      would have lost the target.



    **Operation Time**

    By default, the seeker returns measurments for each
    call to :meth:`measure`. However, setting the parameter `dt`
    to a positive number makes `measure` return `None`
    for any `t < last_t + dt`, where `t` is the current measure time,
    `last_t` is the last measurement time, and `dt` is the seeker time-constant:

    .. code::

      >>> np.random.seed(770)
      >>> tgt = c4d.datapoint(x = 100, y = 100)
      >>> skr = c4d.sensors.seeker(dt = 0.01)
      >>> for t in np.arange(0, .025, .005):  # doctest: +ELLIPSIS
      ...   print(f'{t}: {skr.measure(tgt, t = t)}')
      0.0: (0.73..., 0.007...)
      0.005: (None, None)
      0.01: (0.73..., 0.0006...)
      0.015: (None, None)
      0.02: (0.71..., 0.006...)


    **Random Distribution**

    The distribution of normally generated random variables across mutliple seeker instances
    is shown for biases of two groups of seekers:

    - One group generated with a default `bias_std` of `0.1°`.
    - The second group with `bias_std` of 0.5°.

    .. code::

      >>> from c4dynamics.sensors import seeker
      >>> seekers_type_A = []
      >>> seekers_type_B = []
      >>> B_std = 0.5 * c4d.d2r
      >>> for i in range(1000):
      ...   seekers_type_A.append(seeker().bias * c4d.r2d)
      ...   seekers_type_B.append(seeker(bias_std = B_std).bias * c4d.r2d)

    The histogram highlights the broadening of the distribution
    as the standard deviation increases:

    .. code::

      >>> ax = plt.subplot()
      >>> ax.hist(seekers_type_A, 30, label = 'Type A')  # doctest: +IGNORE_OUTPUT
      >>> ax.hist(seekers_type_B, 30, label = 'Type B')  # doctest: +IGNORE_OUTPUT

    .. figure:: /_examples/seeker/bias2.png




    """

    _scale_factor = 1.0
    """ float; The scale factor error of the seeker angels.
  It is a normally distributed random variable with
  standard deviation scale_factor_std.
  When isideal seeker is configured scale_factor = 1.
  """

    # scale_factor_std = 0.05
    # ''' float; A standard deviation of the scale factor error '''

    _bias = 0.0
    """ float; The bias error of the seeker angels.
  It is a normally distributed random variable with
  standard deviation bias_std
  When isideal seeker is configured bias = 0.
  """

    # bias_std = 0.1 * c4d.d2r
    # ''' float; A standard deviation of the bias error. Defaults 0.1° '''

    # noise_std = 0.4 * c4d.d2r
    # ''' float; A standard deviation of the seeker angular noise. Default value for non-ideal
    # seeker: noise_std = 0.4° '''

    # dt = -1 # np.finfo(np.float64).eps
    # ''' float; The time-constant of the operational rate of the seeker ''' # .
    # default machine epsilon for float64

    _lastsample = -np.inf
    #

    # rng_noise_std = 0

    # def __init__(self, origin = None, isideal = False, **kwargs):
    def __init__(self, origin: Optional["c4d.rigidbody"] = None, isideal: bool = False, **kwargs):
        # A flag indicating whether to run the errors model
        # Initializes the Seeker object.
        # Args:
        #     isideal (bool): A flag indicating whether to run the errors model.
        # TODO
        # 1 limit field of view.

        isradar = kwargs.pop("radar", False)

        bias_std_def = 0.3 if isradar else 0.1
        noise_std_def = 0.8 if isradar else 0.4
        scale_factor_std_def = 0.07 if isradar else 0.05

        # self.__dict__.update(kwargs)
        self.dt = kwargs.pop("dt", -1)
        # self.rng_noise_std = kwargs.pop('rng_noise_std', 0)
        self.bias_std = kwargs.pop("bias_std", bias_std_def * c4d.d2r)
        self.noise_std = kwargs.pop("noise_std", noise_std_def * c4d.d2r)
        self.scale_factor_std = kwargs.pop("scale_factor_std", scale_factor_std_def)

        for k in kwargs.keys():
            if k == "rng_noise_std":
                if not isradar:
                    # c4d.cprint(f'Warning: {k} is not an attribute of seeker', 'r')
                    warnings.warn(f"""Warning: {k} is not an attribute of seeker""", c4d.c4warn)

                continue

            # c4d.cprint(f'Warning: {k} is not an attribute of seeker or radar', 'r')
            warnings.warn(f"""Warning: {k} is not an attribute of seeker or radar""", c4d.c4warn)

        super().__init__()

        self.measure_data = []

        self.az = 0
        self.el = 0

        if origin is not None:
            if not isinstance(origin, c4d.rigidbody):
                raise TypeError(
                    "origin must be a c4dynamics.rigidbody object whose state vector "
                    "origin.X represents the seeker initial position and attitude "
                    "initial conditions"
                )

            self.X = origin.X

        if isideal:
            self.noise_std = 0
        else:
            self._errors_model()

    @property
    def bias(self) -> float:
        """
        Gets and sets the object's bias.


        The bias is a random variable generated once at the stage of constructing the
        instance by the errors model:

        .. math::

          bias = std \\cdot randn

        Where `bias_std` is a parameter with default
        value of `0.1°` for :class:`seeker <c4dynamics.sensors.seeker.seeker>` object,
        and `0.3°` for
        :class:`radar <c4dynamics.sensors.radar.radar>` object.


        To get the bias generated by the errors model, or to override it, the user may call
        :attr:`bias` to get or set the final bias error.


        Parameters
        ----------
        bias : float
            Required bias, [radians].

        Returns
        -------
        bias : float
            Current bias, [radians].


        Example
        -------

        The following example for a `seeker` object is directly applicable
        to a `radar` object too.
        Simply replace :code:`c4d.sensors.seeker(origin = pedestal,...)`
        with :code:`c4d.sensors.radar(origin = pedestal,...)`.


        Required packages:


        .. code::

          >>> import c4dynamics as c4d
          >>> from matplotlib import pyplot as plt
          >>> import numpy as np


        Settings and initial conditions:

        (see :class:`seeker <c4dynamics.sensors.seeker.seeker>` examples for more details):


        Target:

        .. code::

          >>> tgt = c4d.datapoint(x = 1000, y = 0, vx = -80 * c4d.kmh2ms, vy = 10 * c4d.kmh2ms)
          >>> for t in np.arange(0, 60, 0.01):
          ...   tgt.inteqm(np.zeros(3), .01) # doctest: +IGNORE_OUTPUT
          ...   tgt.store(t)


        Origin:


        .. code::

          >>> pedestal = c4d.rigidbody(z = 30, theta = -1 * c4d.d2r)


        Ground truth reference:

        .. code::

          >>> skr_ideal = c4d.sensors.seeker(origin = pedestal, isideal = True)


        **Tracking with bias**

        Define a seeker with a bias error only (mute the scale factor and the noise)
        and set it to `0.5°` to track a constant velocity target:

        .. code::

          >>> skr = c4d.sensors.seeker(origin = pedestal, scale_factor_std = 0, noise_std = 0)
          >>> skr.bias = .5 * c4d.d2r
          >>> for x in tgt.data():
          ...   skr_ideal.measure(c4d.create(x[1:]), t = x[0],
          ...       store = True
          ...    )  # doctest: +IGNORE_OUTPUT
          ...   skr.measure(c4d.create(x[1:]), t = x[0], store = True)  # doctest: +IGNORE_OUTPUT


        Compare the biased seeker with the true target angles (ideal seeker):

        .. code::

          >>> ax = plt.subplot()
          >>> ax.plot(*skr_ideal.data('el', scale = c4d.r2d),
          ...     label = 'target'
          ... )  # doctest: +IGNORE_OUTPUT
          >>> ax.plot(*skr.data('el', scale = c4d.r2d), label = 'seeker')  # doctest: +IGNORE_OUTPUT

        .. figure:: /_examples/seeker/bias1.png


        Example
        -------


        The distribution of normally generated random variables
        is characterized by its bell-shaped curve, which is symmetric about the mean.
        The area under the curve represents probability, with about `68%` of the data
        falling within one standard deviation (1σ) of the mean, `95%` within two,
        and `99.7%` within three,
        making it a useful tool for understanding and predicting data behavior.


        In `radar` objects, and `seeker` objects in general,
        the `bias` and `scale factor` vary
        among different instances to allow a realistic simulation
        of performance behavior in a technique known as Monte Carlo.

        Let's examine the `bias` distribution across
        mutliple `radar` instances with a default `bias_std = 0.3°`
        in comparison to `seeker` instances with a default `bias_std = 0.1°`:








        .. code::

          >>> from c4dynamics.sensors import seeker, radar
          >>> seekers = []
          >>> radars = []
          >>> for i in range(1000):
          ...   seekers.append(seeker().bias * c4d.r2d)
          ...   radars.append(radar().bias * c4d.r2d)


        The histogram highlights the broadening of the distribution
        as the standard deviation increases:


          >>> ax = plt.subplot()
          >>> ax.hist(seekers, 30, label = 'Seekers') # doctest: +IGNORE_OUTPUT
          >>> ax.hist(radars, 30, label = 'Radars')  # doctest: +IGNORE_OUTPUT

        .. figure:: /_examples/radar/bias2.png




        """
        return self._bias

    @bias.setter
    def bias(self, bias: float):
        self._bias = bias

    @property
    def scale_factor(self) -> float:
        """
        Gets and sets the object's scale_factor.


        The scale factor is a random variable generated once at the stage of constructing the
        instance by the errors model:

        .. math::

          scalefactor = std \\cdot randn

        Where `scale_factor_std` is a parameter with default
        value of `0.05 (5%)` for :class:`seeker <c4dynamics.sensors.seeker.seeker>`
        object, and `0.07 (7%)` for
        :class:`radar <c4dynamics.sensors.radar.radar>` object.



        To get the scale factor generated by the errors model,
        or to override it, the user may call
        :attr:`scale_factor` to get or set the final scale factor error.


        Parameters
        ----------
        scale_factor : float
          Required scale factor, [dimensionless].


        Returns
        -------
        scale_factor : float
          Current scale factor, [dimensionless].


        Example
        -------

        The following example for a `seeker` object is directly applicable
        to a `radar` object too.
        Simply replace :code:`c4d.sensors.seeker(...)`
        with :code:`c4d.sensors.radar(...)`.



        Required packages:


        .. code::

          >>> import c4dynamics as c4d
          >>> from matplotlib import pyplot as plt
          >>> import numpy as np


        Settings and initial conditions:

        (see :class:`seeker <c4dynamics.sensors.seeker.seeker>`
        or :class:`radar <c4dynamics.sensors.radar.radar>`
        examples for more details):


        Target:

        .. code::

          >>> tgt = c4d.datapoint(x = 1000, y = 0, vx = -80 * c4d.kmh2ms, vy = 10 * c4d.kmh2ms)
          >>> for t in np.arange(0, 60, 0.01):
          ...   tgt.inteqm(np.zeros(3), .01) # doctest: +IGNORE_OUTPUT
          ...   tgt.store(t)


        Origin:


        .. code::

          >>> pedestal = c4d.rigidbody(z = 30, theta = -1 * c4d.d2r)


        Ground truth reference:

        .. code::

          >>> skr_ideal = c4d.sensors.seeker(origin = pedestal, isideal = True)


        **Tracking with scale factor**

        Define a seeker with a scale factor error only (mute the bias and the noise)
        and set it to `1.2 (= 20%)` to track a constant velocity target:


        .. code::

          >>> skr = c4d.sensors.seeker(origin = pedestal, bias_std = 0, noise_std = 0)
          >>> skr.scale_factor = 1.2
          >>> for x in tgt.data():
          ...   skr_ideal.measure(c4d.create(x[1:]), t = x[0],
          ...     store = True
          ...   )  # doctest: +IGNORE_OUTPUT
          ...   skr.measure(c4d.create(x[1:]), t = x[0], store = True)    # doctest: +IGNORE_OUTPUT


        Compare with the true target angles (ideal seeker):

        .. code::

          >>> ax = plt.subplot()
          >>> ax.plot(*skr_ideal.data('az', scale = c4d.r2d),
          ...     label = 'target'
          ... )  # doctest: +IGNORE_OUTPUT
          >>> ax.plot(*skr.data('az', scale = c4d.r2d),
          ...     label = 'seeker'
          ... )  # doctest: +IGNORE_OUTPUT

        .. figure:: /_examples/seeker/sf.png



        """
        return self._scale_factor

    @scale_factor.setter
    def scale_factor(self, scale_factor: float):
        self._scale_factor = scale_factor

    # def measure(self, target, t = -1, store = False):
    def measure(
        self, target: "c4d.state", t: float = -1, store: bool = False
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Measures azimuth and elevation between the seeker and a `target`.


        If the seeker time-constant `dt` was provided when the `seeker` was created,
        then `measure` returns `None` for any `t < last_t + dt`,
        where `t` is the time input, `last_t` is the last measurement time,
        and `dt` is the seeker time-constant.
        Default behavior, returns measurements for each call.

        If `store = True`, the method stores the measured
        azimuth and elevation along with a timestamp
        (`t = -1` by default, if not provided otherwise).


        Parameters
        ----------
        target : state
            A Cartesian state object to measure by the seeker, including at
            least one position coordinate (x, y, z).
        store : bool, optional
            A flag indicating whether to store the measured values. Defaults `False`.
        t : float, optional
            Timestamp [seconds]. Defaults -1.

        Returns
        -------
        out : tuple
            azimuth and elevation, [radians].


        Raises
        ------
        ValueError
            If `target` doesn't include any position coordinate (x, y, z).


        Example
        -------

        `measure` in a program simulating
        real-time tracking of a constant velcoity target.


        The target is represented by a
        :class:`datapoint <c4dynamics.states.lib.datapoint.datapoint>`
        and is simulated using the :mod:`eqm <c4dynamics.eqm>` module,
        which integrating the point-mass equations of motion.

        An ideal seeker uses as reference to the true position of the target.

        At each cycle, the the seekers take measurements and store the samples for
        later use in plotting the results.

        Import required packages:

        .. code::

          >>> import c4dynamics as c4d
          >>> from matplotlib import pyplot as plt
          >>> import numpy as np


        Settings and initial conditions:

        .. code::

          >>> dt = 0.01
          >>> np.random.seed(321)
          >>> tgt = c4d.datapoint(x = 1000, vx = -80 * c4d.kmh2ms, vy = 10 * c4d.kmh2ms)
          >>> pedestal = c4d.rigidbody(z = 30, theta = -1 * c4d.d2r)
          >>> skr = c4d.sensors.seeker(origin = pedestal, dt = 0.05)
          >>> skr_ideal = c4d.sensors.seeker(origin = pedestal, isideal = True)


        Main loop:

        .. code::

          >>> for t in np.arange(0, 60, dt):
          ...   tgt.inteqm(np.zeros(3), dt)   # doctest: +IGNORE_OUTPUT
          ...   skr_ideal.measure(tgt, t = t, store = True)     # doctest: +IGNORE_OUTPUT
          ...   skr.measure(tgt, t = t, store = True)     # doctest: +IGNORE_OUTPUT
          ...   tgt.store(t)



        Before viewing the results, let's examine the error parameters generated by
        the errors model (`c4d.r2d` converts radians to degrees):

        .. code::

          >>> skr.bias * c4d.r2d # doctest: +ELLIPSIS
          0.16...
          >>> skr.scale_factor # doctest: +ELLIPSIS
          1.008...
          >>> skr.noise_std * c4d.r2d
          0.4

        Then we excpect a constant bias of -0.02° and a 'compression' or 'squeeze' of `5%` with
        respect to the target (as represented by the ideal seeker):

        .. code::

          >>> ax = plt.subplot()   # doctest: +IGNORE_OUTPUT
          >>> ax.plot(*skr_ideal.data('az', scale = c4d.r2d), '.m', markersize = 1,
          ...           label = 'target')   # doctest: +IGNORE_OUTPUT
          >>> ax.plot(*skr.data('az', scale = c4d.r2d), '.c', markersize = 1,
          ...           label = 'seeker')   # doctest: +IGNORE_OUTPUT

        .. figure:: /_examples/seeker/measure.png

        The sample rate of the seeker was set by the parameter `dt = 0.05`.
        In cycles that don't satisfy `t < last_t + dt`, `measure` returns None,
        as shown in a close-up view:

        .. figure:: /_examples/seeker/measure_zoom.png


        """

        if not (hasattr(target, "cartesian") and target.cartesian()):
            raise TypeError(
                "target must be a state objects with at least one position coordinate, x, y, or z."
            )

        if t < self._lastsample + self.dt - 1e-10:
            return None, None

        self._lastsample = t

        # self: The rigid body object on which the seeker is installed
        # target: A datapoint object detected by the seeker

        # target-seeker position in inertial coordinates
        # self.range = self.dist(target) + self.rng_noise_std * np.random.randn()
        # rand1 = np.random.rand() # to preserve matlab normal
        # self.range = self.dist(target) + self.rng_noise_std
        #               * np.sqrt(2) * erfinv(2 * rand1 - 1)  # c4d.mrandn() #

        # target-seeker position in seeker-body coordinates
        x = target.Position - self.Position
        x_body = self.BR @ x

        # extract angles:
        az_true = c4d.atan2(x_body[1], x_body[0])
        el_true = c4d.atan2(x_body[2], c4d.sqrt(x_body[0] ** 2 + x_body[1] ** 2))

        self.az = (
            az_true * self._scale_factor + self._bias + self.noise_std * np.random.randn()
        )  # c4d.mrandn()
        self.el = (
            el_true * self._scale_factor + self._bias + self.noise_std * np.random.randn()
        )  # c4d.mrandn()

        if store:
            self.storeparams(["az", "el"], t=t)

        return self.az, self.el

    def _errors_model(self) -> None:
        """
        measured_angle = true_angle * scale_factor + bias + noise

        Applies the errors model to azimuth and elevation angles.
        Updates the scale factor, bias, and calculates noise.
        """
        self._scale_factor = 1 + self.scale_factor_std * np.random.randn()
        self._bias = self.bias_std * np.random.randn()


if __name__ == "__main__":

    from c4dynamics import rundoctests
    rundoctests(sys.modules[__name__])
