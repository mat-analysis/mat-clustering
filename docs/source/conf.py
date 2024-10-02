# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mat-clustering'
copyright = '2024, Tarlis Tortelli Portela and Yuri Santos'
author = 'Tarlis Tortelli Portela and Yuri Santos'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',  # For Google style and NumPy style docstrings
    'sphinx.ext.viewcode',  # Include links to the source code
    'sphinx_rtd_theme',
    'myst_parser',
]

# Optional: If your source code is outside the docs directory
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

templates_path = ['_templates']
exclude_patterns = []

autosummary_generate = True  # Automatically generate summary pages


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
