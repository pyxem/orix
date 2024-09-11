# Configuration file for the Sphinx documentation app.
# See the documentation for a full list of configuration options:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import datetime
import inspect
import os
from os.path import dirname, relpath
import re
import sys

from numpydoc.docscrape_sphinx import SphinxDocString

import orix
from orix import data

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath("."))
sys.path.append("../")

project = "orix"
author = "orix developers"
copyright = f"2018-{str(datetime.now().year)}, {author}"
release = orix.__version__

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "matplotlib.sphinxext.plot_directive",
    "nbsphinx",
    "sphinxcontrib.bibtex",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.imgconverter",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx_codeautolink",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "numpydoc",  # Must be loaded after autodoc
]

# Create links to references within orix's documentation to these packages
intersphinx_mapping = {
    "black": ("https://black.readthedocs.io/en/stable", None),
    "coverage": ("https://coverage.readthedocs.io/en/latest", None),
    "dask": ("https://docs.dask.org/en/latest", None),
    "defdap": ("https://defdap.readthedocs.io/en/latest", None),
    "diffpy.structure": ("https://www.diffpy.org/diffpy.structure", None),
    "diffsims": ("https://diffsims.readthedocs.io/en/latest", None),
    "h5py": ("https://docs.h5py.org/en/stable", None),
    "kikuchipy": ("https://kikuchipy.org/en/latest", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "nbsphinx": ("https://nbsphinx.readthedocs.io/en/latest", None),
    "nbval": ("https://nbval.readthedocs.io/en/latest", None),
    "numba": ("https://numba.readthedocs.io/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "numpydoc": ("https://numpydoc.readthedocs.io/en/latest", None),
    "pooch": ("https://www.fatiando.org/pooch/latest", None),
    "pytest": ("https://docs.pytest.org/en/stable", None),
    "python": ("https://docs.python.org/3", None),
    "pyxem": ("https://pyxem.readthedocs.io/en/latest", None),
    "readthedocs": ("https://docs.readthedocs.io/en/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
    "sphinx-gallery": ("https://sphinx-gallery.github.io/stable", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Suppress warnings from Sphinx regarding "duplicate source files":
    # https://github.com/executablebooks/MyST-NB/issues/363#issuecomment-1682540222
    "examples/*/*.ipynb",
    "examples/*/*.py",
]

# HTML theming: pydata-sphinx-theme
# https://pydata-sphinx-theme.readthedocs.io
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/pyxem/orix",
    "header_links_before_dropdown": 6,
    "logo": {"alt_text": project, "text": project},
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "use_edit_page_button": True,
}
html_context = {
    "github_user": "pyxem",
    "github_repo": "orix",
    "github_version": "develop",
    "doc_path": "doc",
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Syntax highlighting
pygments_style = "friendly"

# Logo
html_logo = "_static/img/orix_logo.png"
html_favicon = "_static/img/orix_logo.png"

# -- nbsphinx
# https://nbsphinx.readthedocs.io
# Taken from nbsphinx' own nbsphinx configuration file, with slight
# modifications to point nbviewer and Binder to the GitHub develop
# branch links when the documentation is launched from a kikuchipy
# version with "dev" in the version
if "dev" in release:
    release_version = "develop"
else:
    release_version = "v" + release
# This is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = (
    r"""
{% set docname = 'doc/' + env.doc2path(env.docname, base=None)[:-6] + '.ipynb' %}

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
nbsphinx_allow_errors = True
nbsphinx_execute_arguments = [
    "--InlineBackend.rc=figure.facecolor='w'",
    "--InlineBackend.rc=font.size=15",
]

# -- sphinxcontrib-bibtex
# https://sphinxcontrib-bibtex.readthedocs.io
bibtex_bibfiles = ["user/bibliography.bib"]
bibtex_reference_style = "author_year"

# -- sphinx-codeautolink
codeautolink_custom_blocks = {
    "python3": None,
    "pycon3": "sphinx_codeautolink.clean_pycon",
}


def linkcode_resolve(domain, info):
    """Determine the URL corresponding to Python object.
    This is taken from SciPy's conf.py:
    https://github.com/scipy/scipy/blob/develop/doc/source/conf.py.
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

    # Use the original function object if it is wrapped.
    obj = getattr(obj, "__wrapped__", obj)

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

    startdir = os.path.abspath(os.path.join(dirname(orix.__file__), ".."))
    fn = relpath(fn, start=startdir).replace(os.path.sep, "/")

    if fn.startswith("orix/"):
        m = re.match(r"^.*dev0\+([a-f\d]+)$", release)
        pre_link = "https://github.com/pyxem/orix/blob/"
        if m:
            return pre_link + "%s/%s%s" % (m.group(1), fn, linespec)
        elif "dev" in release:
            return pre_link + "develop/%s%s" % (fn, linespec)
        else:
            return pre_link + "v%s/%s%s" % (release, fn, linespec)
    else:
        return None


# -- Copy button customization (taken from PyVista)
# Exclude traditional Python prompts from the copied code
copybutton_prompt_text = r">>> ?|\.\.\. "
copybutton_prompt_is_regexp = True


# -- sphinx.ext.autodoc
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
autosummary_ignore_module_all = False
autosummary_imported_members = True
autodoc_typehints_format = "short"
autodoc_default_options = {
    "show-inheritance": True,
}


# -- numpydoc
# https://numpydoc.readthedocs.io
numpydoc_show_class_members = False
numpydoc_use_plots = True
numpydoc_xref_param_type = True
# fmt: off
numpydoc_validation_checks = {
    "all",   # All but the following:
    "ES01",  # Not all docstrings need an extend summary
    "EX01",  # Examples: Will eventually enforce
    "GL01",  # Contradicts numpydoc examples
    "GL02",  # Appears to be broken?
    "GL07",  # Appears to be broken?
    "GL08",  # Methods can be documented in super class
    "PR01",  # Parameters can be documented in super class
    "PR02",  # Properties with setters might have docstrings w/"Returns"
    "PR04",  # Doesn't seem to work with type hints?
    "RT01",  # Abstract classes might not have return sections
    "SA01",  # Not all docstrings need a "See Also"
    "SA04",  # "See Also" section does not need descriptions
    "SS06",  # Not possible to make all summaries one line
    "YD01",  # Yields: No plan to enforce
}
# fmt: on


# -- matplotlib.sphinxext.plot_directive
# https://matplotlib.org/stable/api/sphinxext_plot_directive_api.html
plot_formats = ["png"]
plot_html_show_source_link = False
plot_html_show_formats = False
plot_include_source = True


def _str_examples(self):
    examples_str = "\n".join(self["Examples"])
    if (
        self.use_plots
        and (
            re.search(r"\b(.plot)\b", examples_str)
            or re.search(r"\b(.plot_map)\b", examples_str)
            or re.search(r"\b(.imshow)\b", examples_str)
        )
        and "plot::" not in examples_str
    ):
        out = []
        out += self._str_header("Examples")
        out += [".. plot::", ""]
        out += self._str_indent(self["Examples"])
        out += [""]
        return out
    else:
        return self._str_section("Examples")


SphinxDocString._str_examples = _str_examples


# -- Sphinx-Gallery
# https://sphinx-gallery.github.io
sphinx_gallery_conf = {
    "backreferences_dir": "reference/generated",
    "doc_module": ("orix",),
    "examples_dirs": "../examples",
    "filename_pattern": "^((?!sgskip).)*$",
    "gallery_dirs": "examples",
    "reference_url": {"orix": None},
    "run_stale_examples": False,
    "show_memory": True,
}
autosummary_generate = True


# Download example datasets prior to building the docs
print("[orix] Downloading example datasets (if not found in the cache)")
_ = data.sdss_ferrite_austenite(allow_download=True)
_ = data.sdss_austenite(allow_download=True)
_ = data.ti_orientations(allow_download=True)


def skip_member(app, what, name, obj, skip, options):
    """Exclude objects not defined within orix from the API reference.

    This ensures inherited members from Matplotlib axes extensions are
    excluded from the reference. We could exclude inherited members
    all together in the Sphinx Jinja2 template. But, this would mean
    classes such as Rotation would have to "overwrite" all methods and
    properties from Quaternion for these members to be listed in the
    API reference of Rotation. Instead, we allow inherited members, but
    try our best to skip members coming from outside orix here.
    """
    if what in ["attribute", "property"]:
        obj_module = inspect.getmodule(getattr(obj, "fget", None))
    else:
        obj_module = inspect.getmodule(obj)
    return "orix" not in getattr(obj_module, "__name__", [])


def setup(app):
    app.connect("autodoc-skip-member", skip_member)
