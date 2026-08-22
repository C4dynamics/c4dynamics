from scipy.linalg import (
    solve_discrete_are,
)  # NOTE only this file and maybe math.py use scipy. worth replacing and exclude from requirements.
from typing import Optional
import sys

# sys.path.append(".")
import c4dynamics as c4d
import numpy as np
import warnings


class kalman(c4d.state):
    """
    Kalman Filter.


    Discrete linear Kalman filter class for state estimation.

    :class:`kalman` provides methods for prediction and update (correct)
    phases of the filter.

    For background material, implementation, and examples,
    please refer to :mod:`filters <c4dynamics.filters>`.



    Parameters
    ==========
    X : dict
        Initial state variables and their values.
    F : numpy.ndarray
        Discrete-time state transition matrix. Defaults to None.
    H : numpy.ndarray
        Discrete-time measurement matrix. Defaults to None.
    steadystate : bool, optional
        Flag to indicate if the filter is in steady-state mode. Defaults to False.
    G : numpy.ndarray, optional
        Discrete-time control matrix. Defaults to None.
    P0 : numpy.ndarray, optional
        Covariance matrix, or standard deviations array, of the
        initial estimation error. Mandatory if `steadystate` is False.
        If P0 is one-dimensional array, standard deviation values are
        expected. Otherwise, variance values are expected.
    Q : numpy.ndarray, optional
        Process noise covariance matrix. Defaults to None.
    R : numpy.ndarray, optional
        Measurement noise covariance matrix. Defaults to None.
    P_jitter : float, optional
        If provided, `P` is symmetrized and floored by `P_jitter` times
        the identity after every `predict`/`update` that modifies it
        (skipped in `steadystate` mode, where `P` never changes). This
        guards against floating-point roundoff eroding `P`'s symmetry or
        positive-definiteness over many cycles; it is not a substitute for
        a correct process/measurement model. Defaults to `None` (off).

    Notes
    =====
    1. `kalman` is a subclass of :class:`state <c4dynamics.states.state.state>`,
    as such the variables provided within the parameter `X` form its state variables.
    Hence, `X` is a dictionary of variables and their initial values, for example:
    ``X = {'x': x0, 'y': y0, 'z': z0}``.

    2. Steady-state mode: if the underlying system is linear time-invariant (`LTI`),
    and the noise covariance matrices are time-invariant,
    then a steady-state mode of the Kalman filter can be employed.
    In steady-state mode the Kalman gain (`K`) and the estimation covariance matrix
    (`P`) are computed once and remain constant ('steady-state') for the entire run-time,
    performing as well as the time-varying filter.
    Note however that also in steady-state mode the predict and the update steps are separated
    and the user must call each of them at a time.



    Raises
    ======
    TypeError:
        If X is not a dictionary.
    ValueError:
        If P0 is not provided when steadystate is False.
    ValueError:
        If system matrices are not fully provided.



    See Also
    ========
    .filters
    .ekf
    .lowpass
    .seeker
    .eqm




    Examples
    ========

    The examples in the introduction to the
    :mod:`filters <c4dynamics.filters>`
    module demonstrate the operations of
    the Kalman filter for inputs from
    electromagnetic devices, such as an altimeter,
    which measures the altitude.




    An accurate Japaneese train travels 150 meters in one second
    (:math:`F = 1, u = 1, G = 150, Q = 0.05`).
    A sensor measures the train position with noise
    variance of :math:`200m^2` (:math:`H = 1, R = 200`).
    The initial position of the train is known with uncertainty
    of :math:`0.5m` (:math:`P0 = 0.5^2`).


    **Note**

    The system may be interpreted as follows:

    - :math:`F = 1`             - constant position

    - :math:`u = 1, G = 150`    - constant velocity control input

    The advantage of this model is in its being first order.
    However, a slight difference between the actual dynamics and
    the modeled process will result in a lag with the tracked object.



    Import required packages:

    .. code::

      >>> from c4dynamics.filters import kalman
      >>> from matplotlib import pyplot as plt
      >>> import c4dynamics as c4d


    Let's run a filter.

    First, since the covariance matrices are
    constant we can utilize the steady state mode of the filter.
    This requires initalization with the respective flag:

    .. code::

      >>> v = 150
      >>> sensor_noise = 200
      >>> kf = kalman({'x': 0}, P0 = 0.5**2, F = 1, G = v, H = 1
      ...                         , Q = 0.05, R = sensor_noise**2, steadystate = True)





    .. code::

      >>> for t in range(1, 26): #  seconds.
      ...   # store for later
      ...   kf.store(t)
      ...   # predict + correct
      ...   kf.predict(u = 1)
      ...   kf.detect = v * t + np.random.randn() * sensor_noise
      ...   kf.storeparams('detect', t)


    Recall that a :class:`kalman` object, as subclass of
    the :class:`state <c4dynamics.states.state.state>`,
    encapsulates the process state vector:

    .. code::

      >>> print(kf)
      [ x ]


    It can also employ the
    :meth:`plot <c4dynamics.states.state.state.plot>`
    or any other method of the `state` class:


    .. code::

      >>> kf.plot('x')  # doctest: +IGNORE_OUTPUT
      >>> plt.gca().plot(*kf.data('detect'), 'co', label = 'detection')   # doctest: +IGNORE_OUTPUT
      >>> plt.gca().legend()    # doctest: +IGNORE_OUTPUT
      >>> plt.show()

    .. figure:: /_examples/kf/steadystate.png


    Let's now assume that as the
    train moves farther from the station,
    the sensor measurements degrade.

    The measurement covariance matrix therefore increases accordingly,
    and the steady state mode cannot be used:


    .. code::

      >>> v = 150
      >>> kf = kalman({'x': 0}, P0 = 0.5*2, F = 1, G = v, H = 1, Q = 0.05)
      >>> for t in range(1, 26): #  seconds.
      ...   kf.store(t)
      ...   sensor_noise = 200 + 8 * t
      ...   kf.predict(u = 1)
      ...   kf.detect = v * t + np.random.randn() * sensor_noise
      ...   kf.K = kf.update(kf.detect, R = sensor_noise**2)
      ...   kf.storeparams('detect', t)


    .. figure:: /_examples/kf/varying_r.png



    """

    # FIX maybe change 'time histories' with 'time series' or 'time evolution'

    _Kinf = None
    _nonlinearF = False

    def __init__(
        self,
        X: dict,
        F: np.ndarray,
        H: np.ndarray,
        steadystate: bool = False,
        G: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        P_jitter: Optional[float] = None,
    ):
        #
        # P0 is mandatory and it is either the initial state covariance matrix itself or
        # a vector of the diagonal standard deviations.
        # dt is for the predict integration.
        # F and H are linear transition matrix and linear measure matrix for
        # a linear kalman filter.
        # Q and R are process noise and measure noise matrices when they are time invariant.
        ##

        if not isinstance(X, dict):
            raise TypeError("""X must be a dictionary containig pairs of variables
                          and initial conditions, e.g.: {''x'': 0, ''y'': 0}""")
        super().__init__(**X)

        #
        # verify shapes consistency:
        #   x = Fx + Gu + w
        #   y = Hx + v
        # X: nx1, F: nxn, G: nxm, u: mx1, y: 1xk, H: kxn
        # P: nxn, Q: nxn, R: kxk
        # state matrices should be 2dim array.
        ##
        def vershape(M1name, M1rows, M2name, M2columns):
            if M1rows.shape[0] != M2columns.shape[1]:
                raise ValueError(
                    f"The columns of {M2name} (= {M2columns.shape[1]}) must equal "
                    ""
                    f"the rows of {M1name} (= {M1rows.shape[0]})"
                )

        self.G = None
        if F is not None and H is not None:
            # discrete
            self.F = np.atleast_2d(F).astype(float)
            vershape("F", self.F, "F", self.F)  # F: nxn
            vershape("X", self.X.T, "F", self.F)  # F: n columns

            self.H = np.atleast_2d(H)
            vershape("X", self.X.T, "H", self.H)  # H: n columns

            if G is not None:
                self.G = np.atleast_2d(G).reshape(
                    self.F.shape[0], -1
                )  # now G is necessarily a column vector.

        else:
            raise ValueError(
                """F and H (G is optional) as a set of system matrices must be provided entirely:"""
                """\nx(k) = F*x(k-1) + G*u(k-1) + w(k-1), y(k) = H*x(k) + v(k)"""
            )

        self.Q = None
        self.R = None
        if Q is not None:
            self.Q = np.atleast_2d(Q)
            vershape("Q", self.Q, "Q", self.Q)  # Q: nxn
            vershape("X", self.X.T, "Q", self.Q)  # Q: n columns
        if R is not None:
            self.R = np.atleast_2d(R)
            vershape("R", self.R, "R", self.R)  # R: kxk
            vershape("H", self.H, "R", self.R)  # R: k columns

        if steadystate:
            # in steady state mode Q and R must be provided:
            if self.Q is None or self.R is None:
                raise ValueError(
                    """In steady-state mode, the noise matrices Q and R must be provided."""
                )

            self.P = solve_discrete_are(self.F.T, self.H.T, self.Q, self.R)
            self._Kinf = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)

        else:  # steady state is off
            if P0 is None:
                # NOTE maybe init with zeros and raising warning is better solution.
                raise ValueError(r"P0 must be provided (optional only in steadystate mode)")

            if np.array(P0).ndim == 1:
                # an array of standard deviations is provided
                self.P = np.diag(np.array(P0).ravel() ** 2)
            else:
                P0 = np.atleast_2d(P0)
                if P0.shape[0] == P0.shape[1]:
                    # square matrix
                    self.P = P0

        self._P_jitter = P_jitter
        self._Pdata = []

    def predict(self, u: Optional[np.ndarray] = None, Q: Optional[np.ndarray] = None):
        """
        Predicts the filter's next state and covariance matrix based
        on the current state and the process model.

        Parameters
        ----------
        u : numpy.ndarray, optional
            Control input. Defaults to None.
        Q : numpy.ndarray, optional
            Process noise covariance matrix. Defaults to None.


        Raises
        ------
        ValueError
            If `Q` is not missing (neither provided
            during construction nor passed to `predict`).
        ValueError
            If a control input is provided, but the number of elements in `u`
            does not match the number of columns of the input matrix `G`.


        Examples
        --------
        For more detailed usage,
        see the examples in the introduction to
        the :mod:`filters <c4dynamics.filters>` module and
        the :class:`kalman` class.




        Import required packages:

        .. code::

          >>> from c4dynamics.filters import kalman



        Plain `predict` step
        (predict in steady-state mode where the process variance matrix
        remains constant
        and is provided once to initialize the filter):

        .. code::

          >>> kf = kalman({'x': 0},
          ...       P0 = 0.5**2,
          ...       F = 1,
          ...       H = 1,
          ...       Q = 0.05,
          ...       R = 200,
          ...       steadystate = True
          ... )
          >>> print(kf)
          [ x ]
          >>> kf.X          # doctest: +NUMPY_FORMAT
          [0]
          >>> kf.P          # doctest: +NUMPY_FORMAT
          [[3.187...]]
          >>> kf.predict()
          >>> kf.X          # doctest: +NUMPY_FORMAT
          [0]
          >>> kf.P          # doctest: +NUMPY_FORMAT
          [[3.187...]]


        Predict with control input:

        .. code::

          >>> kf = kalman({'x': 0}, P0 = 0.5**2, F = 1, G = 150, H = 1, R = 200, Q = 0.05)
          >>> kf.X  # doctest: +NUMPY_FORMAT
          [0]
          >>> kf.P          # doctest: +NUMPY_FORMAT
          [[0.25...]]
          >>> kf.predict(u = 1)
          >>> kf.X      # doctest: +NUMPY_FORMAT
          [150]
          >>> kf.P  # doctest: +NUMPY_FORMAT
          [[0.3]]



        Predict with updated process noise covariance matrix:

        .. code::

          >>> kf = kalman({'x': 0}, P0 = 0.5**2, F = 1, G = 150, H = 1, R = 200, Q = 0.05)
          >>> kf.X  # doctest: +NUMPY_FORMAT
          [0]
          >>> kf.P # doctest: +NUMPY_FORMAT
          [[0.25]]
          >>> kf.predict(u = 1, Q = 0.01)
          >>> kf.X  # doctest: +NUMPY_FORMAT
          [150]
          >>> kf.P # doctest: +NUMPY_FORMAT
          [[0.26]]


        """

        if self._Kinf is None:

            if Q is not None:
                self.Q = np.atleast_2d(Q)
            elif self.Q is None:
                raise ValueError(
                    """Q is missing. It can be provided once at construction """
                    """or in every call to predict() """
                )

            self.P = self.F @ self.P @ self.F.T + self.Q
            if self._P_jitter is not None:
                self._stabilize_P()

        if not self._nonlinearF:
            self.X = self.F @ self.X

        if u is not None:
            if self.G is None:
                warnings.warn(
                    f"""\nWarning: u={u} is introduced as control
                    input but the input matrix G is zero!""",
                    c4d.c4warn,
                )
            else:
                u = np.atleast_2d(u)
                if len(u.ravel()) != self.G.shape[1]:
                    raise ValueError(
                        f"""The number of elements in u must equal the
                        number of columns of the input matrix G,
                        {len(u.ravel())} != {self.G.shape[1]}"""
                    )
                self.X += self.G @ u.ravel()

    def _stabilize_P(self):
        """
        Symmetrize `P` and add a small positive-definite floor to its
        diagonal (``P <- 0.5*(P + P.T) + P_jitter * I``).

        Defensive covariance hygiene: floating-point roundoff accumulated
        over many `predict`/`update` cycles can erode `P`'s symmetry and
        positive-definiteness enough to break a downstream inversion or
        `solve()`. This does not correct modeling errors (e.g. an
        unmodeled nonlinearity or a stale linearization) -- it only keeps
        `P` numerically well-formed. Called automatically, only when
        `P_jitter` was set at construction (see `__init__`).
        """
        self.P = 0.5 * (self.P + self.P.T)
        self.P += self._P_jitter * np.eye(self.P.shape[0])

    def update(
        self,
        z: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        hx: Optional[np.ndarray] = None,
        innov: Optional[np.ndarray] = None,
        gate: Optional[float] = None,
    ):
        """
        Updates (corrects) the state estimate based on the given measurements.

        Parameters
        ----------
        z : numpy.ndarray, optional
            Measurement vector. Required unless `innov` is provided directly.
        R : numpy.ndarray, optional
            Measurement noise covariance matrix. Defaults to None.
        hx : numpy.ndarray, optional
            Predicted measurement, i.e. h(x), used to form the innovation
            as ``z - hx``. Defaults to the linear prediction ``H @ X``.
            Useful for a nonlinear measurement function whose output does
            not equal ``H @ X`` (as in :class:`ekf <c4dynamics.filters.ekf.ekf>`).
            Ignored if `innov` is provided directly.
        innov : numpy.ndarray, optional
            The innovation (measurement residual) itself, overriding the
            default ``z - hx``. Use this when the residual is not a plain
            subtraction, for example a circular (angle-wrapped) measurement.
            `z` is not required when `innov` is given.
        gate : float, optional
            Chi-squared gating threshold on the normalized innovation
            squared (NIS = ``innov.T @ inv(S) @ innov``, with
            ``S = H @ P @ H.T + R``). If the NIS exceeds `gate`, the update
            is rejected: `X` and `P` are left unchanged and `update`
            returns `None`. Defaults to `None` (no gating).

        Returns
        -------
        K : numpy.ndarray or None
            Kalman gain, or `None` if the update was rejected by `gate`.


        Raises
        ------
        ValueError
            If neither `z` nor `innov` is provided.
        ValueError
            If the number of elements in `z` (or `innov`) does not match
            the number of rows in the measurement matrix H.
        ValueError
            If `R` is missing (neither provided
            during construction
            nor passed to `update`).

        Examples
        --------
        For more detailed usage,
        see the examples in the introduction to
        the :mod:`filters <c4dynamics.filters>` module and
        the :class:`kalman` class.



        Import required packages:

        .. code::

          >>> from c4dynamics.filters import kalman



        Plain update step
        (update in steady-state mode
        where the measurement covariance matrix remains
        and is provided once during filter initialization):

        .. code::

          >>> kf = kalman({'x': 0},
          ...       P0 = 0.5**2,
          ...       F = 1,
          ...       H = 1,
          ...       Q = 0.05,
          ...       R = 200,
          ...       steadystate = True
          ... )
          >>> print(kf)
          [ x ]
          >>> kf.X   # doctest: +NUMPY_FORMAT
          [0]
          >>> kf.P                # doctest: +NUMPY_FORMAT
          [[3.187...]]
          >>> kf.update(z = 100)  # returns Kalman gain   # doctest: +NUMPY_FORMAT
          [[0.0156...]]
          >>> kf.X                # doctest: +NUMPY_FORMAT
          [1.568...]
          >>> kf.P                # doctest: +NUMPY_FORMAT
          [[3.187...]]



        Update with modified measurement noise covariance matrix:

        .. code::

          >>> kf = kalman({'x': 0}, P0 = 0.5**2, F = 1, G = 150, H = 1, R = 200, Q = 0.05)
          >>> kf.X   # doctest: +NUMPY_FORMAT
          [0]
          >>> kf.P   # doctest: +NUMPY_FORMAT
          [[0.25]]
          >>> K = kf.update(z = 150, R = 0)
          >>> K   # doctest: +NUMPY_FORMAT
          [[1]]
          >>> kf.X  # doctest: +NUMPY_FORMAT
          [150]
          >>> kf.P  # doctest: +NUMPY_FORMAT
          [[0]]


        """

        # this H must be linear, but like F may it be linearized once about an equilibrium point for
        # the entire process (regular kalman) or at each
        # iteration about the current state (ekf).
        if innov is None:
            if z is None:
                raise ValueError("""Either z or innov must be provided.""")
            z = np.atleast_2d(z).ravel()
            if len(z) != self.H.shape[0]:
                raise ValueError(
                    f"""The number of elements in the input z must equal """
                    f"""the number of rows of the measurement matrix H, """
                    f"""{len(z.ravel())} != {self.H.shape[0]}"""
                )
        else:
            innov = np.atleast_1d(innov).astype(float)
            if len(innov) != self.H.shape[0]:
                raise ValueError(
                    f"""The number of elements in innov must equal """
                    f"""the number of rows of the measurement matrix H, """
                    f"""{len(innov)} != {self.H.shape[0]}"""
                )

        if self._Kinf is None:
            if R is not None:
                self.R = np.atleast_2d(R)
            elif self.R is None:
                raise ValueError(
                    """R is missing. It can be provided once at construction """
                    """or in every call to update() """
                )

            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
        else:
            K = self._Kinf
            S = None  # computed lazily below, only if a gate is requested

        # this H can be expressed as either linear or nonlinear function of x
        # (hx), and the residual itself may need to be formed in a
        # non-subtractive way (innov), e.g. for a circular measurement.
        if innov is None:
            if hx is None:
                hx = self.H @ self.X
            innov = z - hx

        if gate is not None:
            # Chi-squared test (NIS) on the innovation; reject the update
            # without touching X or P if it exceeds the gate.
            if S is None:
                S = self.H @ self.P @ self.H.T + self.R
            try:
                nis = float(innov @ np.linalg.solve(S, innov))
            except np.linalg.LinAlgError:
                return None
            if nis > gate:
                return None

        if self._Kinf is None:
            self.P = self.P - K @ self.H @ self.P
            if self._P_jitter is not None:
                self._stabilize_P()

        self.X += K @ innov  # type: ignore # nx1 = nxm @ (mx1 - mxn @ nx1)
        return K

    def store(self, t: int = -1):
        """
        Stores the current state and diagonal elements of the covariance matrix.

        The :meth:`store` method captures the current state of the Kalman filter,
        storing the state vector (`X`) and the error covariance matrix (`P`)
        at the specified time.


        Parameters
        ----------
        t : int, optional
            The current time at which the state is being stored. Defaults to -1.


        Notes
        -----
        1. The stored data can be accessed via :meth:`data <c4dynamics.states.state.state.data>`
           or other methods for
           post-analysis or visualization.
        2. The elements on the main diagonal of the covariance matrix are named
           according to their position, starting with 'P' followed by their row and column indices.
           For example, the first element is named 'P00', and so on.
        3. See also :meth:`store <c4dynamics.states.state.state.store>`
           and :meth:`data <c4dynamics.states.state.state.data>`
           for more details.

        Examples
        --------
        For more detailed usage,
        see the examples in the introduction to
        the :mod:`filters <c4dynamics.filters>` module and
        the :class:`kalman <c4dynamics.filters.kalman.kalman>` class.




        Import required packages:

        .. code::

          >>> from c4dynamics.filters import kalman



        .. code::

          >>> kf = kalman({'x': 0}, P0 = 0.5**2, F = 1, H = 1, Q = 0.05, R = 200)
          >>> # store initial conditions
          >>> kf.store()
          >>> kf.predict()
          >>> # store X after prediction
          >>> kf.store()
          >>> kf.update(z = 100)    # doctest: +NUMPY_FORMAT
          [[0.00149...]]
          >>> # store X after correct
          >>> kf.store()

        Access stored data:

        .. code::

          >>> kf.data('x')[1]  # doctest: +NUMPY_FORMAT
          [0  0  0.15])
          >>> kf.data('P00')[1]  # doctest: +NUMPY_FORMAT
          [0.25  0.3  0.299])

        """

        super().store(t)  # store state
        # store covariance:
        for i, p in enumerate(np.diag(self.P)):
            pstr = f"P{i}{i}"
            setattr(self, pstr, p)  # set
            self.storeparams(pstr, t)  # store

    @staticmethod
    def velocitymodel(dt: float, process_noise: float, measure_noise: float):
        """
        Defines a linear Kalman filter model for tracking position and velocity.

        Parameters
        ----------
        dt : float
            Time step for the system model.
        process_noise : float
            Standard deviation of the process noise.
        measure_noise : float
            Standard deviation of the measurement noise.

        Returns
        -------
        kf : kalman
            A Kalman filter object initialized with the linear system model.



        X = [x, y, w, h, vx, vy]
        #    0  1  2  3  4   5

        x'  = vx
        y'  = vy
        w'  = 0
        h'  = 0
        vx' = 0
        vy' = 0

        H = [1 0 0 0 0 0
            0 1 0 0 0 0
            0 0 1 0 0 0
            0 0 0 1 0 0]
        """
        from scipy.linalg import expm

        A = np.zeros((6, 6))
        A[0, 4] = A[1, 5] = 1
        F = expm(A * dt)
        H = np.zeros((4, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1

        Q = np.eye(6) * process_noise**2
        R = np.eye(4) * measure_noise**2

        kf = kalman(
            {"x": 0, "y": 0, "w": 0, "h": 0, "vx": 0, "vy": 0}, steadystate=True, F=F, H=H, Q=Q, R=R
        )
        return kf

    @staticmethod
    def nees(kf, true_obj):
        """normalized estimated error squared"""

        Ptimes = kf.data("P00")[0]
        err = []
        for t in kf.data("t"):

            xkf = kf.timestate(t)
            xtrain = true_obj.timestate(t)

            idx = min(range(len(Ptimes)), key=lambda i: abs(Ptimes[i] - t))
            P = kf.data("P00")[1][idx]

            xerr = xtrain - xkf
            err.append(xerr**2 / P)
        return np.mean(err)


if __name__ == "__main__":

    from c4dynamics import rundoctests

    rundoctests(sys.modules[__name__])
