import sys

# sys.path.append(".")
import time

_tic = 0.0


def tic():
    """Starts stopwatch timer.

    Inspired by `MATLAB's` tic toc, `tic()` records the current time to start
    measuring elapsed time.
    When used in conjunction with `toc()` serves as a stopwatch
    timer to measure the time interval between two events.

    Returns
    -------
    out : float
        The recorded start time.

    Examples
    --------

    .. code::

      >>> import c4dynamics as c4d
      >>> import numpy as np


    .. code::

      >>> N = 10000
      >>> tic()   # doctest: +IGNORE_OUTPUT
      >>> a = np.ones((1, 3))
      >>> for i in range(N - 1):
      ...     a = np.concatenate((a, np.ones((1, 3))))
      >>> t1 = toc() # doctest: +IGNORE_OUTPUT
      >>> c4d.cprint('numpy concat: ' + str(1000 * t1) + ' ms', 'r') # doctest: +IGNORE_OUTPUT
      numpy concat: 40.0 ms

    .. code::

      >>> tic() # doctest: +IGNORE_OUTPUT
      >>> a = np.zeros((N, 3))
      >>> for i in range(N):
      ...     a[i, :] = np.ones((1, 3))
      >>> t2 = toc() # doctest: +IGNORE_OUTPUT
      >>> c4d.cprint('numpy predefined: ' + str(1000 * t2) + ' ms', 'g') # doctest: +IGNORE_OUTPUT
      numpy predefined: 3.0 ms

    .. code::

      >>> tic()# doctest: +IGNORE_OUTPUT
      >>> a = []
      >>> for i in range(N):
      ...     a.append([1, 1, 1])
      >>> a = np.array(a)
      >>> t3 = toc()# doctest: +IGNORE_OUTPUT
      >>> c4d.cprint('list to numpy: ' + str(1000 * t3) + ' ms', 'y') # doctest: +IGNORE_OUTPUT
      list to numpy: 0.0 ms
    """

    global _tic
    _tic = time.time()
    return _tic


def toc(show=True, minutes=False):
    """Stops the stopwatch timer and reads the elapsed time.

    Measures the elapsed time since the last call to `tic()` and prints the result in seconds.


    Returns
    -------
    out : float
        Elapsed time in seconds.

    Examples
    --------

    .. code::

      >>> import c4dynamics as c4d
      >>> import numpy as np


    .. code::

      >>> N = 10000
      >>> tic() # doctest: +IGNORE_OUTPUT
      >>> a = np.ones((1, 3))
      >>> for i in range(N - 1):
      ...     a = np.concatenate((a, np.ones((1, 3))))
      >>> t1 = toc() # doctest: +IGNORE_OUTPUT
      >>> c4d.cprint('numpy concat: ' + str(1000 * t1) + ' ms', 'r') # doctest: +IGNORE_OUTPUT
      numpy concat: 31.0 ms

    .. code::

      >>> tic() # doctest: +IGNORE_OUTPUT
      >>> a = np.zeros((N, 3))
      >>> for i in range(N):
      ...     a[i, :] = np.ones((1, 3))
      >>> t2 = toc() # doctest: +IGNORE_OUTPUT
      >>> c4d.cprint('numpy predefined: ' + str(1000 * t2) + ' ms', 'g') # doctest: +IGNORE_OUTPUT
      numpy predefined: 15.0 ms

    .. code::

      >>> tic() # doctest: +IGNORE_OUTPUT
      >>> a = []
      >>> for i in range(N):
      ...     a.append([1, 1, 1])
      >>> a = np.array(a)
      >>> t3 = toc() # doctest: +IGNORE_OUTPUT
      >>> c4d.cprint('list to numpy: ' + str(1000 * t3) + ' ms', 'y') # doctest: +IGNORE_OUTPUT
      list to numpy: 0.0 ms
    """

    # global _tic
    dt = time.time() - _tic
    units = "sec"
    if minutes:
        dt /= 60
        units = "min"
    if show:
        print(f"{dt:.3f}{units}")
    return dt


# tic():

#     Records the current time to start measuring elapsed time.

#     Returns:
#     float: The recorded start time.

# toc():

#     Measures the elapsed time since the last call to `tic()` and prints the result in seconds.

#     Returns:
#     float: The elapsed time in seconds.

if __name__ == "__main__":

    # import doctest, contextlib, os
    # from c4dynamics import IgnoreOutputChecker, cprint

    # # Register the custom OutputChecker
    # doctest.OutputChecker = IgnoreOutputChecker

    # tofile = False
    # optionflags = doctest.FAIL_FAST

    # if tofile:
    #   with open(os.path.join('tests', '_out', 'output.txt'), 'w') as f:
    #     with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
    #       result = doctest.testmod(optionflags = optionflags)
    # else:
    #   result = doctest.testmod(optionflags = optionflags)

    # if result.failed == 0:
    #   cprint(os.path.basename(__file__) + ": all tests passed!", 'g')
    # else:
    #   print(f"{result.failed}")
    from c4dynamics import rundoctests

    rundoctests(sys.modules[__name__])
