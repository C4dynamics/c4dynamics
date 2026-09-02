"""

c4dynamics provides filters to estimate full state of a system from noisy measurements.
The filters include the functionality and the errors model of electro-optic, lasers,
and electro-magnetic devices.


.. list-table::
  :header-rows: 0

  * - :class:`kalman <c4dynamics.filters.kalman.kalman>`
    - Kalman filter
  * - :class:`ekf <c4dynamics.filters.ekf.ekf>`
    - Extended Kalman filter
  * - :class:`lpf <c4dynamics.filters.lowpass.lowpass>`
    - Lowpass filter


"""

import sys

# sys.path.append(".")

from c4dynamics.filters.kalman import kalman as kalman  # noqa: F401
from c4dynamics.filters.ekf import ekf as ekf  # noqa: F401
from c4dynamics.filters.lowpass import lowpass as lowpass  # noqa: F401


# NOTE
# the line:
# "Note that the divider of R is :math:`dt_{measure}` rather than simply :math:`dt`
#   because often times the sampling-time of the measures is different than
#   the sampling-time that uses to integrate the dynamics.
#   However, when the measure times and the integration times are equal,
#   then :math:`dt_{measure} = dt`."
# required clarification. i think i took it from simon. not sure.
# anyway it's weird. as for a continuous system with Q = R
# it's gone have balanced weights.
# however the translation of it to discrete matrices with Q = Q*dt, R = R/dt
# violates the balance.
# I think it's realy wrong. in pg 232 (247) he says explicitly:
# now let us think about measurement noise. suppose we have a discrete-time
# measurement of a constant x every T seconds. The measurement times are tk = k*T (k=1,2,..).
# ..
# the error covariance at time tk is independent of the sample time T if: R = Rc/T.
# where Rc is some constant.
# this implis that
#     lim(R, T->0) = Rc * delta(t)
# where delta(t) is the constinuous time impulse function.
# this estabilshes the equivalence between white meausrement noise
# in discrete time and continuous time. the effects of white measuremnt noise in discrete time
# and continuous time are the same if:
# vk ~ (0, R)
# v(t) ~ (0, Rc)
# i  think i should simply say that c4d kamlans are discrete kalmans.
# only that the user can provdie also cont. matrices.
# and now that i think about that its become more clear to me
# that what i should do is only suggest discrete filter and also
# provide util for covnerting cont to discr system and then the user
# provides the disc systems.
# i rather think that best thing is to separate the covariance matrices from
# discretization and just present it as given for the final system.
# or to add a remark and say that if also the covarinace matrices are given for
# cont system then this the way to discretize it.
# or add a note that in practice the noise of the discrete system
# should be know ampricialy or by data sheet.
# or to add that in practice the sensors are taking measurements in discrete times.
#
# another problem arises:
# the kalman is implemented as discrete-time system.
# if the user provides system matrices A,B,C in the continuous-time domain,
# then i ask also the time-step parameter and convert them to the discrete-time
# form and then calculate the filter equations.
# the problem that if the user provide continuous-time matrices,
# probably he also provide his noise covariances Q and R in the continuous time domain,
# which means that the noise descriptions do not match the discrete form of my filter.
# what do u suggest to do?
# gpt:
# convert q by yourself according to:
#     Qd = A^-1 * (e^AT - I) * Qc
# R is in anyway sampled in disc times.
# alternatively change the model to discrete inputs only.
#
#
# FIXME
# the example of the kalman filter must be fixed as there's no reason to
# divide R here by dt becuase it's the vairance of the radar in the given time step!!
# see figures in w.doc.

# """
# Franklin, G.F., Powell,D.J., and Workman, M.L., Digital Control of Dynamic Systems
# ch 9
# 9.4.2 the discrete kf:
# w(t) and v(t) have no time correlation.
# E(w*w^T)=Rw=Q
# E(v*v^T)=Rv=R

# 9.4.4. noise matrices and discerete equivalents.
# the process uncertainty acts on the continuous portion of the system.




# i have a cont system sampled with a
# discrete samples camera. let's say the sensor errors with its algo are
# sig_camera in both position and bounding box.
# i want to show an example where i give the camera and the process the
# same weight and i run them in a steady state mode.
# the model in const velocity model.
# then i say i want to overcome an error in the linearity and extend the
# uncertainty of the process with still continuous modeling of the process.
# ** remark: how at all can kalman designers
# introduce the uncertainty in the noise? after all kalman
# restrains that factor to be a white noise with mean 0 and
# im not sure the model uncertainty behaves in that way.
# ** any way in the next example i want to show
# that same results could be achieved by using discrete matrices.
# """

if __name__ == "__main__":

    from c4dynamics import rundoctests

    rundoctests(sys.modules[__name__])
