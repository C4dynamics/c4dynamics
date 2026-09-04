import sys
from typing import Optional

import numpy as np

# sys.path.append(".")
# import c4dynamics as c4d
from c4dynamics.filters import kalman


class ekf(kalman):
    """
    Extended Kalman Filter class for handling nonlinear dynamics by
    incorporating functions for nonlinear state transitions and measurements.

    This subclass extends the base
    :class:`kalman <c4dynamics.filters.kalman.kalman>`
    class to handle cases where
    system dynamics or measurements are nonlinear. The Jacobian matrices
    `F` and `H` can be dynamically updated as linearizations of the
    nonlinear functions.


    Parameters
    ----------
    X : dict
        Initial state estimate dictionary, where key-value pairs
        represent state variables and their initial values.
    P0 : np.ndarray
        Initial error covariance matrix, defining the initial
        uncertainty for each state variable.
    F : np.ndarray, optional
        State transition Jacobian matrix; defaults to an identity matrix
        if not provided, assuming a linear system model.
    H : np.ndarray, optional
        Measurement Jacobian matrix; defaults to a zero matrix if not
        provided.
    G : np.ndarray, optional
        Control input matrix, mapping control inputs to the state.
    Q : np.ndarray, optional
        Process noise covariance matrix.
    R : np.ndarray, optional
        Measurement noise covariance matrix.
    P_jitter : float, optional
        Opt-in covariance stabilization; see
        :class:`kalman <c4dynamics.filters.kalman.kalman>`.


    Example
    -------
    A detailed example can be found in the introduction
    to the c4dynamics.filters module.
    The mechanism of this class is similar to
    the :class:`kalman <c4dynamics.filters.kalman.kalman>`,
    so the examples provided there may serve as
    inspiration for using `ekf`.

    """

    def __init__(
        self,
        X: dict,
        P0: np.ndarray,
        F: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
        G: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        P_jitter: Optional[float] = None,
    ):
        # F and H are necessary also for ekf because they are required to the ricatti.
        # yes but the can be delivered at each call in the immediate linearized form.

        if F is None:
            F = np.eye(P0.shape[0])

        if H is None:
            H = np.zeros(P0.shape[0])

        super().__init__(X, F, H, P0=P0, G=G, Q=Q, R=R, P_jitter=P_jitter)
        self._ekf = True

    def predict(
        self,
        F: Optional[np.ndarray] = None,
        fx: Optional[np.ndarray] = None,
        dt=None,  # type: ignore
        u: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
    ):
        """
        Predicts the next state of the system based on the current state
        and an optional nonlinear state transition function.

        Parameters
        ----------
        F : np.ndarray, optional
            The state transition Jacobian matrix. If not provided, the
            previously set `F` matrix is used.
        fx : np.ndarray, optional
            Nonlinear state transition function derivative. If specified,
            this value is used for updating the state with nonlinear dynamics.
        dt : float, optional
            Time step duration. Must be provided if `fx` is specified.
        u : np.ndarray, optional
            Control input vector, affecting the state based on the `G` matrix.
        Q : np.ndarray, optional
            Process noise covariance matrix, representing uncertainty in
            the model during prediction.

        Raises
        ------
        TypeError
            If `fx` is provided without a corresponding `dt` value.

        Examples
        --------
        The examples in this section are intended to
        demonstrate the usage of the `ekf` class and specifically the `predict` method.
        However, they are not limited to nonlinear dynamics.
        For detailed usage that highlights the properties of nonlinear dynamics,
        refer to the :mod:`filters <c4dynamics.filters>` module introduction.



        Import required packages:

        .. code::

          >>> from c4dynamics.filters import ekf



        Plain `predict` step
        (predict in steady-state mode where the process variance matrix
        remains constant
        and is provided once to initialize the filter):

        .. code::

          >>> _ekf = ekf({'x': 0}, P0 = 0.5**2, F = 1, H = 1, Q = 0.05, R = 200)
          >>> print(_ekf)
          [ x ]
          >>> _ekf.X          # doctest: +NUMPY_FORMAT
          [0]
          >>> _ekf.P          # doctest: +NUMPY_FORMAT
          [[0.25]]
          >>> _ekf.predict()
          >>> _ekf.X          # doctest: +NUMPY_FORMAT
          [0]
          >>> _ekf.P          # doctest: +NUMPY_FORMAT
          [[0.3]]


        Predict with control input:

        .. code::

          >>> _ekf = ekf({'x': 0}, P0 = 0.5**2, F = 1, G = 150, H = 1, R = 200, Q = 0.05)
          >>> _ekf.X      # doctest: +NUMPY_FORMAT
          [0]
          >>> _ekf.P         # doctest: +NUMPY_FORMAT
          [[0.25]]
          >>> _ekf.predict(u = 1)
          >>> _ekf.X   # doctest: +NUMPY_FORMAT
          [150]
          >>> _ekf.P  # doctest: +NUMPY_FORMAT
          [[0.3]]



        Predict with updated process noise covariance matrix:

        .. code::

          >>> _ekf = ekf({'x': 0}, P0 = 0.5**2, F = 1, G = 150, H = 1, R = 200, Q = 0.05)
          >>> _ekf.X   # doctest: +NUMPY_FORMAT
          [0]
          >>> _ekf.P  # doctest: +NUMPY_FORMAT
          [[0.25]]
          >>> _ekf.predict(u = 1, Q = 0.01)
          >>> _ekf.X  # doctest: +NUMPY_FORMAT
          [150]
          >>> _ekf.P  # doctest: +NUMPY_FORMAT
          [[0.26]]




        """

        if fx is not None:
            if dt is None:
                raise TypeError("For nonlinear derivatives inpout (fx), dt must be provided.")
            self.X += np.atleast_1d(fx).ravel() * dt
            self._nonlinearF = True

        if F is not None:
            # "if F" is not enough because F is an array and
            # the truth value of an array with more than one
            # element is ambiguous.
            self.F = np.atleast_2d(F)

        super().predict(u=u, Q=Q)

        self._nonlinearF = True

    def update(
        self,
        z: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
        hx: Optional[np.ndarray] = None,
        innov: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        gate: Optional[float] = None,
    ):
        """
        Updates the state estimate based on the latest measurement, using an
        optional nonlinear measurement function.

        Parameters
        ----------
        z : np.ndarray, optional
            Measurement vector, representing observed values from the system.
            Required unless `innov` is provided directly.
        H : np.ndarray, optional
            Measurement Jacobian matrix. If provided, it overrides the
            previously set `H` matrix for this update step.
        hx : np.ndarray, optional
            Nonlinear measurement function output, h(x). If provided, it is
            used in place of the linear ``H @ X`` to form the innovation
            ``z - hx``. Ignored if `innov` is provided directly.
        innov : np.ndarray, optional
            The innovation itself, overriding ``z - hx``. Use this for a
            residual that isn't a plain subtraction, e.g. an angle-wrapped
            measurement. `z` is not required when `innov` is given.
        R : np.ndarray, optional
            Measurement noise covariance matrix, representing the uncertainty
            in the measurements.
        gate : float, optional
            Chi-squared NIS gating threshold; see
            :meth:`kalman.update <c4dynamics.filters.kalman.kalman.update>`.
            If the innovation fails the gate, the update is rejected and
            `update` returns `None`.


        Examples
        --------
        The examples in this section are intended to
        demonstrate the usage of the `ekf` class and specifically the `update` method.
        However, they are not limited to nonlinear dynamics.
        For detailed usage that highlights the properties of nonlinear dynamics,
        refer to the :mod:`filters <c4dynamics.filters>` module introduction.


        Import required packages:

        .. code::

          >>> from c4dynamics.filters import ekf



        Plain update step:

        .. code::

          >>> _ekf = ekf({'x': 0}, P0 = 0.5**2, F = 1, H = 1, Q = 0.05, R = 200)
          >>> print(_ekf)
          [ x ]
          >>> _ekf.X   # doctest: +NUMPY_FORMAT
          [0]
          >>> _ekf.P                # doctest: +NUMPY_FORMAT
          [[0.25]]
          >>> _ekf.update(z = 100)  # returns Kalman gain   # doctest: +NUMPY_FORMAT
          [[0.001...]]
          >>> _ekf.X                # doctest: +NUMPY_FORMAT
          [0.124...]
          >>> _ekf.P                # doctest: +NUMPY_FORMAT
          [[0.249...]]



        Update with modified measurement noise covariance matrix:

        .. code::

          >>> _ekf = ekf({'x': 0}, P0 = 0.5**2, F = 1, G = 150, H = 1, R = 200, Q = 0.05)
          >>> _ekf.X   # doctest: +NUMPY_FORMAT
          [0]
          >>> _ekf.P  # doctest: +NUMPY_FORMAT
          [[0.25]]
          >>> K = _ekf.update(z = 150, R = 0)
          >>> K   # doctest: +NUMPY_FORMAT
          [[1]]
          >>> _ekf.X  # doctest: +NUMPY_FORMAT
          [150]
          >>> _ekf.P  # doctest: +NUMPY_FORMAT
          [[0]]

        """
        if H is not None:
            self.H = np.atleast_2d(H)

        return super().update(z=z, R=R, hx=hx, innov=innov, gate=gate)


if __name__ == "__main__":

    from c4dynamics import rundoctests

    rundoctests(sys.modules[__name__])
