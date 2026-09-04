import sys

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

sys.path.append(".")

available = {f.name for f in font_manager.fontManager.ttflist}
if "Times New Roman" in available:
    mpl.rcParams["font.family"] = "Times New Roman"
else:
    mpl.rcParams["font.family"] = "serif"



# pragma: no cover
def _figdef():
    factorsize = 4
    aspectratio = 1080 / 1920
    return plt.subplots(
        1,
        1,
        dpi=200,
        figsize=(factorsize, factorsize * aspectratio),
        gridspec_kw={"left": 0.15, "right": 0.9, "top": 0.9, "bottom": 0.2},
    )


# pragma: no cover
def _legdef():
    return {"fontsize": 4, "facecolor": None}


def plotdefaults(ax, title, xlabel="", ylabel="", fontsize=8, ilines=None):
    """

    Setting default properties on a matplotlib axis.

    Parameters
    ----------

    ax : matplotlib.axes.Axes
        The matplotlib axis on which to set the properties.

    title : str
        Plot title.

    xlabel : str
        The label for the x-axis.

    ylabel : str
        The label for the y-axis.

    fontsize : int, optional
        The font size for the title, x-axis label, y-axis label, and tick labels. Default is 14.


    Example
    -------

    .. code::

      >>> import c4dynamics as c4d
      >>> f16 = c4d.rigidbody()
      >>> dt = .01
      >>> for t in np.arange(0, 9, dt):
      ...   if t < 3:
      ...     f16.phi += dt * 180 / 9 * c4d.d2r
      ...   elif t < 6:
      ...     f16.phi += dt * 180 / 6 * c4d.d2r
      ...   else:
      ...     f16.phi += dt * 180 / 3  * c4d.d2r
      ...   f16.store(t)
      >>> ax = plt.subplot()
      >>> ax.plot(*f16.data('phi', c4d.r2d), 'm', linewidth = 2) # doctest: +IGNORE_OUTPUT
      >>> c4d.plotdefaults(ax, '$\\varphi$', 'Time', 'deg', fontsize = 18)


    .. figure:: /_examples/utils/plotdefaults.png

    """

    #
    # axis
    ##
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.grid(alpha=0.5)
    ax.tick_params(axis="both", labelsize=fontsize)

    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.yaxis.get_major_formatter().set_scientific(False)


if __name__ == "__main__":

    from c4dynamics import rundoctests
    rundoctests(sys.modules[__name__])

