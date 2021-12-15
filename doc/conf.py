# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import datetime
import inspect
import os
from os.path import relpath, dirname
import re
import sys

from orix import __author__, __version__

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath("."))
sys.path.append("../")

project = "orix"
copyright = f"2018-{str(datetime.now().year)}, {__author__}"
author = __author__
release = __version__


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinxcontrib.bibtex",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx_gallery.load_style",
    "nbsphinx",
]

# Create links to references within orix's documentation to these packages
intersphinx_mapping = {
    "dask": ("https://docs.dask.org/en/latest", None),
    "diffpy.structure": ("https://www.diffpy.org/diffpy.structure", None),
    "h5py": ("http://docs.h5py.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org", None),
    "numpy": ("https://docs.scipy.org/doc/numpy", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Syntax highlighting
pygments_style = "friendly"

# Logo
html_logo = "_static/img/orix_logo.png"
html_favicon = "_static/img/orix_logo.png"

# nbsphinx configuration
# Taken from nbsphinx' own nbsphinx configuration file, with slight
# modification to point nbviewer and Binder to the GitHub master links
# when the documentation is launched from a orix version with "dev" in
# the version
if "dev" in __version__:
    release_version = "master"
else:
    release_version = "v" + __version__
# This is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = (
    r"""
{% set docname = 'doc/' + env.doc2path(env.docname, base=None) %}

.. raw:: html

    <style>a:hover { text-decoration: underline; }</style>

    <div class="admonition note">
      This page was generated from
      <a class="reference external" href="https://github.com/pyxem/orix/blob/"""
    + f"{release_version}"
    + r"""/{{ docname|e }}">{{ docname|e }}</a>.
      Interactive online version:
      <span style="white-space: nowrap;"><a href="https://mybinder.org/v2/gh/pyxem/orix/"""
    + f"{release_version}"
    + r"""?filepath={{ docname|e }}"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>.</span>
      <script>
        if (document.location.host) {
          $(document.currentScript).replaceWith(
            '<a class="reference external" ' +
            'href="https://nbviewer.jupyter.org/url' +
            (window.location.protocol == 'https:' ? 's/' : '/') +
            window.location.host +
            window.location.pathname.slice(0, -4) +
            'ipynb">View in <em>nbviewer</em></a>.'
          );
        }
      </script>
    </div>

.. raw:: latex

    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
"""
)
# https://nbsphinx.readthedocs.io/en/0.8.0/never-execute.html
nbsphinx_execute = "auto"  # auto, always, never
nbsphinx_execute_arguments = [
    "--InlineBackend.rc=figure.facecolor='w'",
    "--InlineBackend.rc=font.size=15",
]

# sphinxcontrib-bibtex configuration
bibtex_bibfiles = ["bibliography.bib"]


def linkcode_resolve(domain, info):
    """Determine the URL corresponding to Python object.

    This is taken from SciPy's conf.py:
    https://github.com/scipy/scipy/blob/master/doc/source/conf.py.
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    startdir = os.path.abspath(os.path.join(dirname(__file__), ".."))
    fn = relpath(fn, start=startdir).replace(os.path.sep, "/")

    if fn.startswith("orix/"):
        m = re.match(r"^.*dev0\+([a-f0-9]+)$", __version__)
        pre_link = "https://github.com/pyxem/orix/blob/"
        if m:
            return pre_link + "%s/%s%s" % (m.group(1), fn, linespec)
        elif "dev" in __version__:
            return pre_link + "master/%s%s" % (fn, linespec)
        else:
            return pre_link + "v%s/%s%s" % (__version__, fn, linespec)
    else:
        return None
