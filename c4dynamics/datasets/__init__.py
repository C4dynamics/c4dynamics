"""

Usage of Datasets
=================

C4dynamics dataset functions can be simply called as follows:
:code:`c4dynamics.datasets.module(file)`,
where ``module`` and ``file`` define the dataset.
The available modules and files are detailed on the corresponding pages.
This downloads the dataset file over the network once, saves it to the cache,
and returns the path to the file.




Dataset retrieval and storage
=============================

The YOLOv3 weights file can be found at the official YOLO
site: `Joseph Redmon <https://pjreddie.com/darknet/yolo/>`_.
Oher dataset files are available in the
C4dynamics GitHub repository under
`datasets <https://github.com/C4dynamics/C4dynamics/blob/main/datasets/>`_


C4dynamics.datasets uses
`Pooch <https://www.fatiando.org/pooch/latest/>`_,
a Python package designed to simplify fetching data files.
`Pooch` retrieves the necessary dataset files
from these repositories when the dataset function is called.


A registry file of all datasets provides a mapping of filenames
to their SHA256 hashes and repository URLs
Pooch uses this registry to manage and verify downloads when the function is called.
After downloading the dataset once, the files are saved
in the system cache directory under ``'c4data'``.


Dataset cache locations may vary on different platforms.

For Windows::

    'C:\\Users\\<user>\\AppData\\Local\\c4data'

For macOS::

    '~/Library/Caches/c4data'

For Linux and other Unix-like platforms::

    '~/.cache/c4data'  # or the value of the XDG_CACHE_HOME env var, if defined


In environments with constrained network connectivity for various security
reasons or on systems without continuous internet connections,
one may manually
load the cache of the datasets by placing the contents of the dataset repo in
the above mentioned cache directory to avoid fetching dataset errors without
the internet connectivity.

"""

import sys

# sys.path.append(".")

# from c4dynamics.datasets._manager import (
#     sha256,
#     image,
#     video,
#     nn_model,
#     d3_model,
#     download_all,
#     clear_cache,
# )
from c4dynamics.datasets._manager import sha256 as sha256
from c4dynamics.datasets._manager import image as image
from c4dynamics.datasets._manager import video as video
from c4dynamics.datasets._manager import nn_model as nn_model
from c4dynamics.datasets._manager import d3_model as d3_model
from c4dynamics.datasets._manager import download_all as download_all
from c4dynamics.datasets._manager import clear_cache as clear_cache

if __name__ == "__main__":

    # import doctest, contextlib
    # from c4dynamics import IgnoreOutputChecker, cprint

    # # Register the custom OutputChecker
    # doctest.OutputChecker = IgnoreOutputChecker

    # tofile = False
    # optionflags = doctest.FAIL_FAST

    # if tofile:
    #   with open(os.path.join('tests', '_out', 'output.txt'), 'w') as f:
    #     with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
    #       result = doctest.testmod(optionflags = optionflags)
    # else:
    #   result = doctest.testmod(optionflags = optionflags)

    # if result.failed == 0:
    #   cprint(os.path.basename(__file__) + ": all tests passed!", 'g')
    # else:
    #   print(f"{result.failed}")
    from c4dynamics import rundoctests

    rundoctests(sys.modules[__name__])
