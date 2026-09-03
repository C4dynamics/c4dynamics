from setuptools import setup
setup()

# # from pathlib import Path
# import codecs
# import os

# # import re

# package = "c4dynamics"
# here = os.path.abspath(os.path.dirname(__file__))

# with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
#     long_description = "\n" + fh.read()


# # import c4dynamics

# # VERSION = c4dynamics.__version__
# VERSION = "2.4.1"  # change also in __init__.py and pyproject.toml
# print(VERSION)
# # print("did u remember to upgrade the version number??")

# DESCRIPTION = "Python framework for state-space modeling."
# LONG_DESCRIPTION = "Tsipor (bird) Dynamics (c4dynamics) is The Python framework for state-space" \
#                         " modeling and algorithm development."


# setup(
#     name=package,
#     version=VERSION,
#     author="c4dynamics",
#     author_email="zivmeri@gmail.com",
#     description=DESCRIPTION,
#     long_description_content_type="text/markdown",
#     long_description=long_description,  # LONG_DESCRIPTION   #
#     packages=find_packages(),
#     include_package_data=True,
#     install_requires=[
#         "matplotlib>=3.9.2",
#         "numpy>=1.26.0",
#         "pooch>=1.8.0",
#         "scipy>=1.13.0",
#     ],
#     extras_require={
#         "vision": [
#             "imageio>=2.37.0",
#             "natsort>=8.3.1",
#             "opencv-python>=4.11.0.86,<5",
#         ],
#         "dev": [
#             "coverage>=7.0.0",
#             "imageio>=2.37.0",
#             "natsort>=8.3.1",
#             "opencv-python>=4.11.0.86,<5",
#             "nbsphinx>=0.9.3",
#             "Sphinx>=8.1.3",
#             "sphinx-book-theme>=1.1.4",
#             "sphinx_design>=0.6.1",
#         ],
#     },
#     python_requires=">=3.8",  # update also in run-tests.yml, readme.md,
#     # pyproject.yaml, setup_guide.ipynb
# )


# print("always make a tag version with a pip upload!")
