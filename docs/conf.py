# flake8: noqa
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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------
import os
import sys

import gymnasium_robotics

project = "Gymnasium-Robotics"
copyright = "2023 Farama Foundation"
author = "Farama Foundation"

# The full version, including alpha/beta/rc tags
release = gymnasium_robotics.__version__

sys.path.insert(0, os.path.abspath(".."))

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_github_changelog",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Napoleon settings
napoleon_use_ivar = True
napoleon_use_admonition_for_references = True
# See https://github.com/sphinx-doc/sphinx/issues/9119
napoleon_custom_sections = [("Returns", "params_style")]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = "Gymnasium-Robotics Documentation"
html_baseurl = "https://robotics.farama.org/"
html_copy_source = False
html_favicon = "_static/img/robotics-favicon.png"
html_theme_options = {
    "light_logo": "img/robotics.png",
    "dark_logo": "img/robotics-white.png",
    "image": "img/robotics-github.png",
    "description": "Gymnasium-Robotics is a collection of robotics simulation environments for Reinforcement Learning",
    "gtag": "G-M37ZN7PPMN",
    "versioning": True,
    "source_repository": "https://github.com/Farama-Foundation/Gymnasium-Robotics/",
    "source_branch": "master",
    "source_directory": "docs/",
}

html_static_path = ["_static"]
html_css_files = []

# -- Generate Changelog -------------------------------------------------

sphinx_github_changelog_token = os.environ.get("SPHINX_GITHUB_CHANGELOG_TOKEN")
