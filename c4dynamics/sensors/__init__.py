"""

`c4dynamics` provides sensor models to simulate real world applications.
The models include the functionality and the errors model
of electro-optic, lasers, electro-magnetic, and navigation devices.


.. list-table::
  :header-rows: 0

  * - :class:`seeker <c4dynamics.sensors.seeker.seeker>`
    - Direction detector
  * - :class:`radar <c4dynamics.sensors.radar.radar>`
    - Range-direction detector
  * - :class:`gps <c4dynamics.sensors.navigation.gps>`
    - Inertial position receiver
  * - :class:`imu <c4dynamics.sensors.navigation.imu>`
    - Gyroscope + accelerometer
  * - :class:`magnetometer <c4dynamics.sensors.navigation.magnetometer>`
    - Heading (yaw) sensor




"""

#
# i think maybe to include also detectors in this module
# and rename it to something like perception \ source \ input
# measures \
#

import sys

# sys.path.append(".")

from c4dynamics.sensors.radar import radar as radar  # noqa: F401
from c4dynamics.sensors.lineofsight import lineofsight as lineofsight  # noqa: F401
from c4dynamics.sensors.seeker import seeker as seeker  # noqa: F401
from c4dynamics.sensors.navigation import gps as gps  # noqa: F401
from c4dynamics.sensors.navigation import imu as imu  # noqa: F401
from c4dynamics.sensors.navigation import magnetometer as magnetometer  # noqa: F401

# Background Material
# -------------------

# Introduction
# ^^^^^^^^^^^^
# '''

# seekers:
#   matter:
# 	    radar
# 	    laser
# 	    optic
#   functionallity:
#       altitude radar
#       lineofsight seeker

# navigation sensors (c4dynamics.sensors.navigation):
# 	imu
# 		accelerometers
# 		roll gyro
# 		rate gyro
# 	gps
# 	magnetometer
# 	lidar  (not yet implemented)


if __name__ == "__main__":

    from c4dynamics import rundoctests

    rundoctests(sys.modules[__name__])
