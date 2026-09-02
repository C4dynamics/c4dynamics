"""

`c4dynamics` provides controller implementations for common dynamic-system
control problems.


.. list-table::
  :header-rows: 0

  * - :func:`quad_pid.dynamics <c4dynamics.controllers.quad_pid.dynamics>`
    - 12-state quadcopter rigid-body dynamics
  * - :func:`quad_pid.InitializeControllers <c4dynamics.controllers.quad_pid.InitializeControllers>`
    - instantiate the quadcopter cascade-PID loops + allocator
  * - :class:`quad_pid.OuterPositionPID <c4dynamics.controllers.quad_pid.OuterPositionPID>`
    - quadcopter position loop  (50 Hz)
  * - :class:`quad_pid.MiddleAttitudePID <c4dynamics.controllers.quad_pid.MiddleAttitudePID>`
    - quadcopter attitude loop (100 Hz)
  * - :class:`quad_pid.InnerRatePID <c4dynamics.controllers.quad_pid.InnerRatePID>`
    - quadcopter body-rate loop (200 Hz)
  * - :class:`quad_pid.ControlAllocator <c4dynamics.controllers.quad_pid.ControlAllocator>`
    - quadcopter torques -> rotor speeds allocator


"""

import sys

from c4dynamics.controllers.quad_pid import dynamics as dynamics  # noqa: F401
from c4dynamics.controllers.quad_pid import position_reference as position_reference  # noqa: F401
from c4dynamics.controllers.quad_pid import velocity_reference as velocity_reference  # noqa: F401
from c4dynamics.controllers.quad_pid import InitializeControllers as InitializeControllers  # noqa: F401
from c4dynamics.controllers.quad_pid import OuterPositionPID as OuterPositionPID  # noqa: F401
from c4dynamics.controllers.quad_pid import MiddleAttitudePID as MiddleAttitudePID  # noqa: F401
from c4dynamics.controllers.quad_pid import InnerRatePID as InnerRatePID  # noqa: F401
from c4dynamics.controllers.quad_pid import ControlAllocator as ControlAllocator  # noqa: F401
from c4dynamics.controllers.quad_pid import run_fig8_pid as run_fig8_pid  # noqa: F401
from c4dynamics.controllers.quad_pid import plot_results as plot_results  # noqa: F401
from c4dynamics.controllers.quad_pid import compute_metrics as compute_metrics  # noqa: F401
from c4dynamics.controllers import quad_pid as quad_pid  # noqa: F401


if __name__ == "__main__":

    from c4dynamics import rundoctests

    rundoctests(sys.modules[__name__])
