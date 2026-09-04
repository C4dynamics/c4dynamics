# optional vision modules
try:
    import imageio
    import natsort
except ImportError:
    # vision modules are unavailable
    pass

import os
import sys
import warnings

# sys.path.append(".")
import c4dynamics as c4d


def gif(dirname, gif_name, duration=None):
    """

    Gif creator.

    `gif` creates a Gif from a directory containing image files.

    Parameters
    ----------

    dirname : str
        The path to the directory containing image files.

    gif_name : str
        The desired name of the output GIF file.

    duration : float, optional
        The duration (in seconds) for each frame of the GIF.
        If None, default duration is used.


    Example
    -------

    Let's generate images of an F16 aircraft in different orientations and create a gif from them.

    1. Prepare trajectory to animate.


    Import required packages:

    .. code::

      >>> import c4dynamics as c4d
      >>> from IPython.display import Image
      >>> import numpy as np
      >>> import os


    Settings and initial conditions:

    .. code::

      >>> f16 = c4d.rigidbody()
      >>> dt = 0.01


    Main loop:

    .. code::

      >>> for t in np.arange(0, 9, dt):
      ...  # in 3 seconds make 180 deg:
      ...  if t < 3:
      ...    f16.psi += dt * 180 * c4d.d2r / 3
      ...  elif t < 6:
      ...    f16.theta += dt * 180 * c4d.d2r / 3
      ...  else:
      ...    f16.phi -= dt * 180 * c4d.d2r / 3
      ...  f16.store(t)


    2. Animate and save image files.

    (Use c4dynamics' :mod:`datasets <c4dynamics.datasets>` to fetch F16 3D model.)

    .. code::

      >>> f16path = c4d.datasets.d3_model('f16')
      Fetched successfully
      >>> x0 = [90 * c4d.d2r, 0, 180 * c4d.d2r]
      >>> outfol = os.path.join('tests', '_out', 'f16b')
      >>> f16.animate(f16path, savedir = outfol, angle0 = x0, modelcolor = [0, 0, 0],
      ...       cbackground = [230 / 255, 230 / 255, 255 / 255])


    3. Export images as gif.

    .. code::

      >>> gifname = 'f16_monochrome_gif.gif'
      >>> c4d.gif(outfol, gifname, duration = 1)
      >>> Image(filename = os.path.join(outfol, gifname))  # doctest: +IGNORE_OUTPUT

    .. figure:: /_examples/animate/f16_monochrome_gif.gif

    """
    dirfiles = natsort.natsorted(os.listdir(dirname))
    imgfiles = [
        f for f in dirfiles if f.lower().endswith((".png", ".jpg", ".jpeg", "bmp", ".tiff"))
    ]

    if not imgfiles:
        warnings.warn(
            f"""No image files in {dirfiles}."
             " ('.png', '.jpg', '.jpeg', 'bmp', '.tiff', are supported). """,
            c4d.c4warn,
        )
        return

    fps = 60  # 60hz (60frames per second) -> 16.6msec
    if duration is None:
        interval = 1
    else:
        # FIXME sth here is wrong. see the test test_specified_fps_effect currently disabled.
        interval = max(int(len(imgfiles) / (duration * fps)), 1)

    images = []

    for i in range(0, len(imgfiles), interval):
        images.append(imageio.v2.imread(os.path.join(dirname, imgfiles[i])))

    gif_name = gif_name if gif_name.endswith(".gif") else gif_name + ".gif"
    imageio.mimsave(os.path.join(dirname, gif_name), images, loop=0, duration=int(1000 * 1 / fps))


if __name__ == "__main__":

    from c4dynamics import rundoctests

    try:
        import open3d as o3d  # noqa: F401

        rundoctests(sys.modules[__name__])
    except ImportError:
        rundoctests(sys.modules[__name__], ["__main__.gif"])
