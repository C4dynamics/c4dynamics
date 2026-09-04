import sys

import numpy as np
from scipy.special import erfinv

# sys.path.append(".")
import c4dynamics as c4d

__doc__ = """
Math functions aliasing.


For convenience, C4dynamics aliases some of
NumPy's trigonometric functions
with slight variations in some of them.

All c4dynamics math functions reside directly in the c4dynamics namespace.
For example, if c4dynamics is imported as c4d, then the sin() function is given by c4d.sin().



.. code::

    import numpy as np
    import c4dynamics as c4d


.. data:: sin

    ``c4d.sin = np.sin``

.. data:: sind

    ``c4d.sind = lambda n: np.sin(n * c4d.d2r)``

.. data:: cos

    ``c4d.cos = np.cos``

.. data:: cosd

    ``c4d.cosd = lambda n: np.cos(n * c4d.d2r)``

.. data:: tan

    ``c4d.tan = np.tan``

.. data:: tand

    ``c4d.tand = lambda n: np.tan(n * c4d.d2r)``


.. data:: asin

    ``c4d.asin = np.arcsin``

.. data:: asind

    ``c4d.asind = lambda n: np.arcsin(n) * c4d.r2d``

.. data:: acos

    ``c4d.acos = np.arccos``

.. data:: acosd

    ``c4d.acosd = lambda n: np.arccos(n) * c4d.r2d``

.. data:: atan

    ``c4d.atan = np.arctan``

.. data:: atan2

    ``c4d.atan2 = np.arctan2``

.. data:: atan2d

    ``c4d.atan2d = lambda y, x: np.arctan2(y, x) * c4d.r2d``


.. data:: sqrt

    ``c4d.sqrt = np.sqrt``

.. data:: norm

    ``c4d.norm = np.linalg.norm``


"""
sin = np.sin


# sind = lambda n: np.sin(n * c4d.d2r)
def sind(n):
    return np.sin(n * c4d.d2r)


cos = np.cos


# cosd = lambda n: np.cos(n * c4d.d2r)
def cosd(n):
    return np.cos(n * c4d.d2r)


tan = np.tan


# tand = lambda n: np.tan(n * c4d.d2r)
def tand(n):
    return np.tan(n * c4d.d2r)


asin = np.arcsin


# asind = lambda n: np.arcsin(n) * c4d.r2d
def asind(n):
    return np.arcsin(n) * c4d.r2d


acos = np.arccos


# acosd = lambda n: np.arccos(n) * c4d.r2d
def acosd(n):
    return np.arccos(n) * c4d.r2d


atan = np.arctan


# atand = lambda n: np.arctan(n) * c4d.r2d
def atand(n):
    return np.arctan(n) * c4d.r2d


atan2 = np.arctan2  # atan2(y, x)


# atan2d = lambda y, x: np.arctan2(y, x) * c4d.r2d  # atan2(y, x)
def atan2d(y, x):
    return np.arctan2(y, x) * c4d.r2d


sqrt = np.sqrt
norm = np.linalg.norm


# mrandn preserves matlab normal distributed numbers generation
# XXX it doesnt preserve anything. just a suggested implementation to make in both sides.
# no more it also doesnt generate normal distribution. see also the test test_mrandn
# currently disabled.
# mrandn = lambda n=1: np.sqrt(2) * erfinv(2 * np.random.rand(n) - 1)
def mrandn(n=1):
    return np.sqrt(2) * erfinv(2 * np.random.rand(n) - 1)


if __name__ == "__main__":

    from c4dynamics import rundoctests

    rundoctests(sys.modules[__name__])
