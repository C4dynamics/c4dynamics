"""

.. list-table::
  :header-rows: 0

  * - :class:`State <c4dynamics.states.state.state>`
    - The state class
  * - :mod:`States Library <c4dynamics.states.lib>`
    - A collection of predefined states



"""

# TODO how come the overall title is not word-capatilized and the smaller are.


import sys

# sys.path.append(".")
from c4dynamics.states.lib import datapoint as datapoint
from c4dynamics.states.lib import pixelpoint as pixelpoint
from c4dynamics.states.lib import rigidbody as rigidbody

if __name__ == "__main__":

    from c4dynamics import rundoctests

    rundoctests(sys.modules[__name__])
