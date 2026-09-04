# spacecraft dyanmics and control

# Rotating Reference Frame
# ^^^^^^^^^^^^^^^^^^^^^^^^

# A vector resolved in a given reference frame is said to be
# **expressed** in that frame (sometimes said **referred to**).

# The rate of change of a vector, as viewed by an observer fixed to and moving
# with a given reference frame,
# is said to be **relative to** or **with respect to** that reference frame.

# It's important to note here that the rate of change of a vector must be
# relative to an inertial reference frame,
# but it can be expressed in any reference frame.


import sys

# sys.path.append(".")

from c4dynamics.rotmat.rotmat import rotx as rotx
from c4dynamics.rotmat.rotmat import roty as roty
from c4dynamics.rotmat.rotmat import rotz as rotz
from c4dynamics.rotmat.rotmat import dcm321 as dcm321
from c4dynamics.rotmat.rotmat import dcm321euler as dcm321euler
from c4dynamics.rotmat.animate import animate as animate

if __name__ == "__main__":

    from c4dynamics import rundoctests

    rundoctests(sys.modules[__name__])
