"""

`c4dynamics` provides reference physical-parameter models for the vehicles
used across its examples -- mass, geometry, and other fixed physical
properties. This module holds configuration data only, no dynamics and no
control logic (see :mod:`c4dynamics.controllers` for that).


.. list-table::
  :header-rows: 0

  * - :func:`quad.default_quad_config <c4dynamics.models.quad.default_quad_config>`
    - reference quadcopter physical parameters


"""

import sys

from c4dynamics.models.quad import default_quad_config as default_quad_config
from c4dynamics.models import quad as quad

if __name__ == "__main__":

    from c4dynamics import rundoctests

    rundoctests(sys.modules[__name__])
