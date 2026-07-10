# from IPython.display import display_html
# display_html("<script>Jupyter.notebook.kernel.restart()</script>", raw = True)

import re
import os
import sys

# import importlib

"""

required packages:

pip install sphinx
pip install sphinx_design
pip install sphinx_book_theme

https://github.com/numpy/numpy/issues/22408
Too light color for the label of "dev" version selection button
The label of "dev" version selection button is colored light gray in the dark mode and the button itself is colored orange. The difference in brightness between those two colors is too small to read.
In the light mode, the label color is black and it looks clear.

ss 2022-10-08 15 55 00

This is because pydata-sphinx-theme defines color for #version_switcher_button, whose value is #c9d1d9 in the dark mode although it is #333333 in the light mode. numpy.css's definition only overwrites the background color for "dev".

H3-level section heading colors
I think H3-level section heading colors is a bit too dark, but this may be a matter of opinion:

ss 2022-10-08 15 54 25

The cause is the same as that of the issue with the body text color; the color of h3 is overwritten in numpy.css.




Each introduction page on your GitHub page, website,
and documentation serves different purposes and targets
different audiences. Here's a breakdown of the intention and focus for each:

https://chatgpt.com/share/671776d9-c600-8002-bf9e-c9345a1fa792


### 1. **GitHub Page (Repository README)**
**Purpose**: To introduce your project to developers and contributors.
**Focus**: Provide a concise overview of the project, its purpose, and how to get started.
**What to include**:
   - **Project Name** and **Brief Description**: Explain the problem your project solves or the features it offers.
   - **Key Features**: Highlight what makes your project unique or useful.
   - **Installation Instructions**: How someone can clone the repository and run the project locally.
   - **Usage**: Basic examples of how to use the project or API.
   - **Contributing Guidelines**: Link to guidelines for developers who want to contribute.
   - **Links to Documentation**: Provide a link to the full documentation page.
   - **License**: The open-source license under which your project is distributed.
readme examples:
  https://github.com/pandas-dev/pandas


### 2. **Website (General Audience)**
**Purpose**: To introduce your product or project to a broad, often non-technical audience.
**Focus**: Showcase the benefits and value of your project in a way that appeals to potential users, stakeholders, or customers.
**What to include**:
   - **Clear Value Proposition**: A headline that clearly explains what the product or project does and why it matters.
   - **Key Features and Benefits**: Highlight the primary benefits and advantages, focusing on user outcomes.
   - **Visuals or Demos**: Include images, videos, or demos that show the product in action.
   - **Call to Action**: Direct visitors to take the next step, whether it's signing up, downloading, or contacting you for more information.
   - **Testimonials or Success Stories**: If applicable, add feedback or use cases to demonstrate impact.
   - **Links to GitHub/Documentation**: For technical users, offer easy access to the repository or full documentation.

### 3. **Documentation Page (Technical Audience)**
**Purpose**: To provide comprehensive and detailed instructions on how to use and integrate the product or project.
**Focus**: Make it easy for users to navigate and understand how to effectively use the system.
**What to include**:
   - **Introduction to the Product/Project**: A technical overview that explains the functionality, components, and intended use.
   - **Quick Start Guide**: A step-by-step guide to get the user up and running as quickly as possible.
   - **Detailed API/Function Documentation**: Exhaustive documentation for developers that includes code examples, parameters, and return values.
   - **Usage Scenarios or Tutorials**: Provide use cases, examples, or tutorials that demonstrate how to use different features.
   - **Troubleshooting/FAQ**: Address common issues users might face.
   - **Versioning Information**: Ensure users know which version they are reading about, and provide clear instructions for navigating between versions if necessary.

Each introduction should be tailored to the specific audience, with clear and relevant information that guides them toward the next step, whether that’s using, contributing to, or learning more about your project.



missdistance.png
ballistic_trajectory.png
drifting_car_snapshot.png
mpc_diagram_thumbnail.png
fig8_quad.png

are images that not generated during the build because they
are defined only in a css block (not genuine sphinx) then they
need to be postioned in the _static folder



"""
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# beutiful docs:
# https://ds4sd.github.io/docling/reference/document_converter/


project = "C4DYNAMICS"
copyright = "2023, c4dynamics"
author = "c4dynamics"
# release   = '0.0.40'
# # Check if the module is loaded
# if 'c4dynamics' in sys.modules:
#     print('\033[91m' + 'reimport c4dynamics' + '\033[0m')
#     del sys.modules['c4dynamics']

# sys.path.append('.')
# sys.path.append(os.path.join('..', '..'))
sys.path.insert(0, os.path.abspath(".."))  # adjust '..' if conf.py is in docs/

# sys.path.append('./c4dynamics')
# sys.path.append(os.path.join(os.getcwd(), 'c4dynamics'))
# print('\033[91m' + os.getcwd() + '\033[0m')

import c4dynamics as c4d

# import c4dynamics.states.state
# from c4dynamics.states.state import state.X0
# import import__all__ext

c4d.cprint("successfully imported c4dynamics", "y")
# c4d.cprint('successfully imported import__all__ext', 'y')


# Reload the module
# importlib.reload(importlib.import_module(project.lower()))

# The short X.Y version (including .devxxx, rcX, b1 suffixes if present)
version = re.sub(r"(\d+\.\d+)\.\d+(.*)", r"\1\2", c4d.__version__)
version = re.sub(r"(\.dev\d+).*?$", r"\1", version)
# The full version, including alpha/beta/rc tags.
release = c4d.__version__
print("%s %s" % (version, release))


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_design",
    "nbsphinx",
    # , 'import__all__ext'
    # , 'myst_nb'
    # , 'sphinxcontrib.bibtex'
]

# bibtex_bibfiles = ['references.bib']
# jupyter_execute_notebooks = 'off'


# verbose debug
traceback_show = True
autodoc_default_flags = ["members"]
autosummary_generate = True
# Otherwise, the Return parameter list looks different from the Parameters list
napoleon_use_rtype = False
# napoleon_google_docstring = True
# napoleon_numpy_docstring = False
napoleon_type_aliases = {"Args": "Arguments", "Parameters": "Arguments"}

"""
  https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html


  Docstring Sections
  All of the following section headers are supported:

    Args (alias of Parameters)
    Arguments (alias of Parameters)
    Attention
    Attributes
    Caution
    Danger
    Error
    Example
    Examples
    Hint
    Important
    Keyword Args (alias of Keyword Arguments)
    Keyword Arguments
    Methods
    Note
    Notes
    Other Parameters
    Parameters
    Return (alias of Returns)
    Returns
    Raise (alias of Raises)
    Raises
    References
    See Also
    Tip
    Todo
    Warning
    Warnings (alias of Warning)
    Warn (alias of Warns)
    Warns
    Yield (alias of Yields)
    Yields

  # Napoleon settings
    napoleon_google_docstring = True
    napoleon_numpy_docstring = True
    napoleon_include_init_with_doc = False
    napoleon_include_private_with_doc = False
    napoleon_include_special_with_doc = True
    napoleon_use_admonition_for_examples = False
    napoleon_use_admonition_for_notes = False
    napoleon_use_admonition_for_references = False
    napoleon_use_ivar = False
    napoleon_use_param = True
    napoleon_use_rtype = True
    napoleon_preprocess_types = False
    napoleon_type_aliases = None
    napoleon_attr_annotations = True
"""


# maps functions with a class name that is indistinguishable when case is
# ignore to another filename


autosummary_filename_map = {  # not sure this is necessary anymore
    "c4dynamics.datapoint.x": "c4dynamics.datapoint.x_var",
    #  'c4dynamics.datapoint.X': 'datapoint.x_property',
}

autodoc_default_options = {"ignore-module-all": True}


# Sphinx project configuration
templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
source_suffix = ".rst"
# The encoding of source files.
source_encoding = "utf-8"
master_doc = "index"
pygments_style = "default"
add_function_parentheses = False


suppress_warnings = [
    "misc.highlighting_failure",
]


# nbsphinx_allow_errors = True
# nbsphinx_execute = 'always'  # or 'auto' if you only want to execute if outputs are missing
# nbsphinx_execute_arguments = [
#     '--TagRemovePreprocessor.enabled=True',
#     '--TagRemovePreprocessor.remove_cell_tags=noexec'
# ]
# nbsphinx_execute = 'never'


# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------


# HTML output configuration
# -----------------------------------------------------------------------------
# html_title = f'{project} <span class="project-version">{version}</span>'
html_title = "%s v%s" % (project, version)
html_short_title = project
# Don't use the logo since it gets in the way of the project name and is
# repeated in the front page.
# html_logo = "_static/pooch-logo.png"
"""
  html_static_path
  ----------------
  A list of paths that contain custom static files.
  Relative paths are taken as relative to the configuration directory (conf.py base).

  The content of these folders are copied to the output's _static directory
  after the theme's static files,
  so a file named default.css will overwrite the theme's default.css.

  Then no matter what static folder the image is located, for example
  source/_architecture/image.png
  the image is copied to _static
  then the reference to it should be with respct to _static:
  _static/image.png


  configuration directory
  -----------------------
  The directory containing conf.py.

"""
html_static_path = ["_papers", "_architecture", "_static"]
html_favicon = "_static/c4dlogo.svg"
html_last_updated_fmt = "%b %d, %Y"
# html_copy_source = True
# CSS files are relative to the static path
html_css_files = ["style.css"]
html_extra_path = []
html_show_sourcelink = False
html_show_sphinx = True
html_show_copyright = True
# html_logo = "_static/c4dlogo.svg"


html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": f"https://github.com/{author.lower()}/{project.lower()}",
    "repository_branch": "main",
    "path_to_docs": "docs",
    # "launch_buttons": {
    #   "binderhub_url": "https://mybinder.org",
    #   "notebook_interface": "jupyterlab",
    # }
    # "use_edit_page_button": True,
    # "use_issues_button": True,
    # "use_repository_button": True,
    # "use_download_button": True,
    "home_page_in_toc": True,
    "analytics": {
        "google_analytics_id": "G-LKRZFXZKQP",
    }
}
html_js_files = [
    "custom.js",
]

"""
procedure for version.
1 checkout main revision
2 branch it
3 tag it
4 .. check gpt wahts first. push and merge or merge and

USE CASES
1. see pretty example for jupyter demo here https://cocalc.com/share/public_paths/7557a5ac1c870f1ec8f01271959b16b49df9d087/07-Kalman-Filter-Math.ipynb
2. beatuiful user guide session: https://genesis-world.readthedocs.io/en/latest/

example cheatsheet:
https://www.doabledanny.com/static/ab88214c9082dc96bb7d53b90fc6981b/5620d/dark_mode.png




AI for docstrings
=================
12 in seeker i commented some parts that included toctrees and description
    of arguments in construction etc.
17 add section explains how the user adds methods (not just params and vars. )
19 protcet the _data array to prevent modification from outside.
20 maybe add an option to introduce parameters as dict in the constructor.







Omit Articles: In titles, headings, labels, lists, and brief technical descriptions.
Include Articles: In full sentences, abstracts, summaries, and more narrative forms of writing
https://chatgpt.com/share/799febb0-c875-41be-99f7-c5555741d387





"""
