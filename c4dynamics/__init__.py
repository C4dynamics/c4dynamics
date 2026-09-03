"""

C4DYNAMICS
==========

c4dynamics provides
  1. State objects as fundamental data structure for dynamic systems.
  2. Internal systems and 3rd party integrated libraries.
  3. Fast algorithmic operations over objects and systems.


How to use the documentation
----------------------------
Documentation is currently availble through examples,
readme pages, and inline comments.


Available subpackages
---------------------
sensors
  Models of EO and EM sensors.
detectors
  Objects detection models.
filters
  Kalman and lowpass filters.
eqm
  Runge Kutta solvers for integrating the equations of motion.
rotmat
  Rotation matrices and rotational operations.
"""

import os
import sys
import doctest
import warnings
import subprocess
import numpy as np

#
# body objects
##
from . import rotmat as rotmat
from .states.lib.pixelpoint import pixelpoint as pixelpoint
from .states.lib.datapoint import datapoint as datapoint
from .states.lib.datapoint import create as create
from .states.state import state as state

# rotmat is required to importing rigidbody:
from .states.lib.rigidbody import rigidbody as rigidbody  # rotmat is required to import rigidbody.

#
# routines
##
from . import eqm as eqm

#
# utils
##
from .utils.const import *  # noqa: F403
from .utils.math import *  # noqa: F403
from .utils.gen_gif import gif as gif
from .utils.cprint import cprint as cprint
from .utils.plottools import plotdefaults as plotdefaults
from .utils.plottools import _figdef as _figdef
from .utils.plottools import _legdef as _legdef
from .utils import tictoc as tictoc
from .utils.tictoc import tic as tic
from .utils.tictoc import toc as toc
from .utils._struct import struct as struct
from .utils.idx2keys import idx2keys as idx2keys
from . import datasets as datasets

#
# sensors
##
from . import sensors as sensors
from . import filters as filters
from . import detectors as detectors

#
# reinforcement learning
##
from . import envs as envs  # noqa: F401

#
# version
##
__version__ = "2.4.0"  # update also in pyproject.toml


#
# some convinient mirroring
##
j = os.path.join


"""
WARNINGS
"""


class c4warn(UserWarning):
    pass


# customize the warning messages:
YELLOW = "\033[93m"
RESET = "\033[0m"  # Reset color to default


# Override showwarning to globally apply custom formatting
def show_warning(message, category, filename, lineno, file=None, line=None):

    if issubclass(category, c4warn):
        # Apply formatting for c4warn warnings

        # FIXME suppressing is absolutely not working.
        message1 = str(message) + "\n"
        message2 = "To suppress c4dynamics' warnings, run: import warnings," \
                   "import c4dynamics as c4d, warnings.simplefilter('ignore', c4d.c4warn)\n"

        print(f"\n{YELLOW}{message1}{RESET}{message2} (File: {filename}, Line: {lineno})")
    else:
        # For other warnings, use the default behavior
        print(f"{category.__name__}: {message} (File: {filename}, Line: {lineno})")


warnings.showwarning = show_warning

"""
TESTING
"""


class IgnoreOutputChecker(doctest.OutputChecker):
    from typing import Union

    IGNORE_OUTPUT = doctest.register_optionflag("IGNORE_OUTPUT")  # 2048
    NUMPY_FORMAT = doctest.register_optionflag("NUMPY_FORMAT")  # 4096

    def check_output(self, want, got, optionflags):

        # If the IGNORE_OUTPUT flag is set, always return True
        if optionflags & self.IGNORE_OUTPUT:
            return True

        # If NUMPY_FORMAT flag is set, compare NumPy arrays with formatting tolerance
        if optionflags & self.NUMPY_FORMAT:

            want = self._convert_to_array(want)
            got = self._convert_to_array(got)

            if want is not None and got is not None:

                abs_tol = 1e-3
                rel_tol = 1e-3

                if False:

                    # Calculate element-wise absolute and relative differences
                    # if diff < abs (for small values) OR diff/want < rel (for large values)
                    np.abs(want - got) < abs_tol
                    np.abs((want - got) / np.where(want != 0, want, np.inf)) < rel_tol

                return np.allclose(want, got, atol=abs_tol, rtol=rel_tol)

        # Otherwise, fall back to the original behavior
        return super().check_output(want, got, optionflags)  # type: ignore

    def _convert_to_array(self, text):

        import re

        """Attempt to convert text to a NumPy array for comparison."""
        try:

            if "," not in text:
                text = re.sub(r"\s+", ",", text.strip())

            # Remove 'np.type'
            text = re.sub(r"np\.\w+", "", text)
            # Remove 'array(' and strip leading/trailing whitespace characters
            text = re.sub(r"(array\()", "", text).strip()
            # Remove brackets and parantheses
            text = re.sub(r"[\[\]\(\)]", "", text)
            # remove ellipsis
            text = re.sub(r"\.\.\.", "", text)
            # Convert to NumPy array
            return np.fromstring(text, sep=",")

        except ValueError:
            print(
                "\n \033[31m DOCSTRING ERROR: Could not convert to an array \033[0m",
                file=sys.stderr,
            )
            return None  # Return None if conversion fails

    # Filter out tests for the animate method by overriding the testmod.


def testmod_filtering(module, filter_functions=[], **kwargs):

    tests = []

    for test in doctest.DocTestFinder().find(module):
        filtfunc = False
        for func in filter_functions:
            if func == test.name:
                filtfunc = True
                break
        if filtfunc:
            continue
        tests.append(test)

    runner = doctest.DocTestRunner(**kwargs)
    for test in tests:
        runner.run(test)
    return runner.summarize()


def rundoctests(module, exclude_functions=[]):
    import doctest
    import contextlib
    import os
    from c4dynamics import IgnoreOutputChecker  # , cprint

    # from matplotlib import pyplot as plt
    from pathlib import Path

    tofile = False
    # pltbe = plt.get_backend()
    # plt.switch_backend("tkAgg")

    # Register the custom OutputChecker
    doctest.OutputChecker = IgnoreOutputChecker

    optionflags = doctest.FAIL_FAST
    np.set_printoptions(legacy="1.25")

    if tofile:
        with open(os.path.join("tests", "_out", "output.txt"), "w") as f:
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                # result = doctest.testmod(optionflags = optionflags)
                result = testmod_filtering(module, exclude_functions, optionflags=optionflags)
    else:
        # result = doctest.testmod(optionflags = optionflags)
        result = testmod_filtering(module, exclude_functions, optionflags=optionflags)

    if result.failed == 0:
        cprint(Path(module.__file__).parts[-1] + ": all tests passed!", "g")
    else:
        print(f"{result.failed}")

    # plt.switch_backend(pltbe)


"""
ROOT FOLDER
"""


# just find the package root folder:
def c4dir(dir, addpath=""):
    # dirname and basename are supplamentary:
    # c:\dropbox\c4dynamics\text.txt
    # dirname: c:\dropbox\c4dynamics
    # basename: text.txt

    inc4d = os.path.basename(dir) == "c4dynamics"
    hasc4d = any(f == "c4dynamics" for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f)))

    if inc4d and hasc4d:
        addpath += ""
        return addpath

    addpath += "..\\"
    return c4dir(os.path.dirname(dir), addpath)


"""
KEYWORDS
"""
#
# TODO BUG FIXME HACK NOTE XXX
#
# TODO IMPROVMEMNT
#
# BUG LOGICAL FAILURE PROBABLY COMES WITH XXX
#     Highlights the presence of a bug or an issue.
# FIXME NOT SEVERE BUT A BETTER IDEA IS TO DO SO
#       Indicates that there is a problem or bug that needs to be fixed.
# HACK I KNOW ITS NOT BEST SOLUTION TREAT IF U HAVE SPARE TIME
#      Suggests that a workaround or temporary solution has been
#      implemented and should be revisited.
# NOTE MORE IMPORTANT THAN A CASUAL COMMENT
#      Provides additional information or context about the code.
# XXX TREAT THIS BEFORE OTHERS
#     Used to highlight something that is problematic, needs attention,
#     or should be addressed later.


# FIXME
# dependencies: for time limits requirementx.txt currently
# install all. actually maybe 90% of the users use only numpy
# and pyplot so it's a good practice to offer another full-req.txt
# file and add an import check for those not necessary:


def ensure_package(package_name):
    try:
        __import__(package_name)
    except ImportError:
        # Check if the user is in a conda environment
        in_conda_env = os.environ.get("CONDA_PREFIX") is not None
        manager = "conda" if in_conda_env else "pip"

        user_input = input(
            f"{package_name} is required but not installed. "
            f"Would you like to install it with {manager}? (y/n): "
        )

        if user_input.lower() == "y":
            if manager == "conda":
                subprocess.check_call(["conda", "install", package_name, "-y"])
            else:
                subprocess.check_call(["pip", "install", package_name])
