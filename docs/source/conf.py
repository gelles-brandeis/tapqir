# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

# Configuration file for the Sphinx documentation builder.

#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "tapqir"
copyright = "2021, Yerdos Ordabayev"
author = "Yerdos Ordabayev"

# The full version, including alpha/beta/rc tags
from tapqir import __version__

release = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "cliff.sphinxext",
    "sphinx.ext.viewcode",
    "sphinx_panels",
    "sphinx_gallery.load_style",
    "sphinx.ext.autosummary",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Generate the API documentation when building
autosummary_generate = True

intersphinx_mapping = dict(
    ipython=("https://ipython.readthedocs.io/en/stable/", None),
    matplotlib=("https://matplotlib.org/", None),
    numpy=("https://numpy.org/doc/stable/", None),
    pandas=("https://pandas.pydata.org/docs/", None),
    python=("https://docs.python.org/3", None),
    scipy=("https://docs.scipy.org/doc/scipy/reference/", None),
    torch=("https://pytorch.org/docs/master/", None),
    pyro=("http://docs.pyro.ai/en/stable/", None),
)

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/gelles-brandeis/tapqir",
            "icon": "fab fa-github-square",
        },
    ],
    "show_prev_next": False,
    "use_edit_page_button": True,
}

# Edit this Page link.
html_context = {
    # "github_url": "https://github.com", # or your GitHub Enterprise interprise
    "github_user": "gelles-brandeis",
    "github_repo": "tapqir",
    "github_version": "latest",
    "doc_path": "docs/source/",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/sphinx_gallery.css"]
html_logo = "_static/logo.png"

autodoc_default_options = {"member-order": "bysource"}

autoprogram_cliff_application = "tapqir"
