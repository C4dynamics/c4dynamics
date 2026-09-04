# import sys
from typing import Optional

# sys.path.append(".")
import c4dynamics as c4d


class lowpass(c4d.state):
    """
    A first-order low-pass filter for smoothing signals,
    supporting both discrete and continuous systems.

    Parameters
    ----------
    alpha : float, optional
        Smoothing factor for a discrete system. Must be in the range (0, 1). Defaults to None.
    dt : float, optional
        Time step for a continuous system. Must be positive. Defaults to None.
    tau : float, optional
        Time constant for a continuous system. Must be positive. Defaults to None.
    y0 : float, optional
        Initial value of the state. Defaults to 0.

    Raises
    ------
    ValueError
        If neither `alpha` nor both `dt` and `tau` are provided.
        If `alpha` is out of the range (0, 1) for a discrete system.
        If `dt` or `tau` is non-positive for a continuous system.

    Notes
    -----
    - For a continuous system, `dt` and `tau` are required, and `alpha` is calculated as `dt / tau`.
    - For a discrete system, `alpha` alone is required and directly specifies the smoothing factor.

    Example
    -------

    .. code::

      >>> filter_continuous = lowpass(dt=0.01, tau=0.1, y0=0)
      >>> filter_discrete = lowpass(alpha=0.5, y0=1)
      >>> filter_continuous.sample(1.0) # doctest: +ELLIPSIS
      0.09...
      >>> filter_discrete.sample(2.0)
      1.5
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        dt: Optional[float] = None,
        tau: Optional[float] = None,
        y0: float = 0,
    ) -> None:
        # Initialize alpha based on the provided parameters
        if dt is not None and tau is not None:
            if dt <= 0 or tau <= 0:
                raise ValueError("For a continuous system, `dt` and `tau` must be positive.")
            self.alpha = dt / tau
        elif alpha is not None:
            if not (0 < alpha < 1):
                raise ValueError("For a discrete system, `alpha` must be in the range (0, 1).")
            self.alpha = alpha
        else:
            raise ValueError(
                "Provide either `alpha` for a discrete system or both "
                "`dt` and `tau` for a continuous system."
            )

        self.y = y0  # Initial state value

    def sample(self, x: float) -> float:
        """
        Applies the low-pass filter to the input value and returns the filtered output.

        Parameters
        ----------
        x : float
            Input value to be filtered.

        Returns
        -------
        float
            The filtered output value after applying the low-pass filter.

        Notes
        -----
        - For a continuous system: `y'(t) = -y(t) / tau + x(t) / tau`
        - For a discrete system: `y[k] = (1 - alpha) * y[k-1] + alpha * x[k]`
        - The filter's state (`self.y`) is updated in place.

        Example
        -------

        .. code::

          >>> lp_filter = lowpass(alpha=0.5)
          >>> lp_filter.sample(2.0)
          1.0
          >>> lp_filter.sample(3.0)
          2.0
        """
        # Update the filter's state
        self.y = (1 - self.alpha) * self.y + self.alpha * x
        return self.y


if __name__ == "__main__":
    import contextlib
    import doctest
    import os

    from c4dynamics import IgnoreOutputChecker, cprint

    # Register the custom OutputChecker
    doctest.OutputChecker = IgnoreOutputChecker

    tofile = False
    optionflags = doctest.FAIL_FAST

    if tofile:
        with open("tests/_out/output.txt", "w") as f, \
             contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            result = doctest.testmod(optionflags=optionflags)
    else:
        result = doctest.testmod(optionflags=optionflags)

    if result.failed == 0:
        cprint(os.path.basename(__file__) + ": all tests passed!", "g")
    else:
        print(f"{result.failed} test(s) failed.")
