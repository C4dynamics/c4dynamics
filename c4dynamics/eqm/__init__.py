# import sys
import os

# sys.path.append(".")

from c4dynamics.eqm.derivs import eqm3 as eqm3
from c4dynamics.eqm.derivs import eqm6 as eqm6
from c4dynamics.eqm.integrate import int3 as int3
from c4dynamics.eqm.integrate import int6 as int6

if __name__ == "__main__":

    import doctest
    import contextlib
    from c4dynamics import IgnoreOutputChecker, cprint

    # Register the custom OutputChecker
    doctest.OutputChecker = IgnoreOutputChecker

    tofile = False
    optionflags = doctest.FAIL_FAST

    if tofile:
        with open(os.path.join("tests", "_out", "output.txt"), "w") as f, \
             contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            result = doctest.testmod(optionflags=optionflags)
    else:
        result = doctest.testmod(optionflags=optionflags)

    if result.failed == 0:
        cprint(os.path.basename(__file__) + ": all tests passed!", "g")
    else:
        print(f"{result.failed}")
