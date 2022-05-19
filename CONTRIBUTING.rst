============
Contributing
============

``orix`` is a community maintained project. We welcome contributions in the form of bug
reports, feature requests, code, documentation, and more. These guidelines provide
resources on how best to contribute.

For new users, checking out the `GitHub guides <https://guides.github.com>`_ are
recommended.

.. tip::
    This guide can look intimidating to people who want to contribute, but have limited
    experience with tools like ``git``, ``pytest``, and ``sphinx``. The shortest route
    to start contributing is to create a GitHub account and explain what you want to do
    `in an issue <https://github.com/pyxem/orix/issues/new>`_.

Questions, comments, and feedback
=================================

Have any questions, comments, suggestions for improvements, or any other
inquiries regarding the project? Feel free to
`open an issue <https://github.com/pyxem/orix/issues>`_ or
`make a pull request <https://github.com/pyxem/orix/pulls>`_ in our GitHub repository.

.. _set-up-a-development-installation:

Set up a development installation
=================================

You need a `fork
<https://docs.github.com/en/get-started/quickstart/contributing-to-projects#about-forking>`_
of the `repository <https://github.com/pyxem/orix>`_ in order to make changes to orix.

Make a local copy of your forked repository and change directories::

    git clone https://github.com/your-username/orix.git
    cd orix

Set the ``upstream`` remote to the main orix repository::

    git remote add upstream https://github.com/pyxem/orix.git

We recommend installing in a `conda environment
<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
with the `Miniconda distribution <https://docs.conda.io/en/latest/miniconda.html>`_::

    conda create --name orix-dev python=3.10
    conda activate orix-dev

Then, install the required dependencies while making the development version available
globally (in the ``conda`` environment)::

    pip install --editable .[dev]

This installs all necessary development dependencies, including those for running tests
and building documentation.

Code style
==========

The code making up orix is formatted closely following the `Style Guide for Python Code
<https://www.python.org/dev/peps/pep-0008/>`_ with `The Black Code style
<https://black.readthedocs.io/en/stable/the_black_code_style/index.html>`_. We use
`pre-commit <https://pre-commit.com>`_ to run ``black`` automatically prior to each
local commit. Please install it in your environment::

    pre-commit install

Next time you commit some code, your code will be formatted inplace according
to the default black configuration.

Note that ``black`` won't format `docstrings
<https://www.python.org/dev/peps/pep-0257/>`_. We follow the `numpydoc
<https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_
standard.

Comment lines should preferably be limited to 72 characters.

Package imports should be structured into three blocks with blank lines between them
(descending order): standard library (like ``os`` and ``typing``), third party packages
(like ``numpy`` and ``matplotlib``) and finally ``orix`` imports.

Make changes
============

If you want to add a new feature, branch from the ``develop`` branch, and when you want
to fix a bug, branch from ``main`` instead.

To create a new feature branch that tracks the upstream development branch::

    git checkout develop -b your-awesome-feature-name upstream/develop

When you've made some changes you can view them with::

    git status

Add and commit your created, modified or deleted files::

    git add my-file-or-directory
    git commit -s -m "An explanatory commit message"

The ``-s`` flag makes sure that you sign your commit with your `GitHub-registered email
<https://github.com/settings/emails>`_ as the author. You can set this up following
`this GitHub guide
<https://help.github.com/en/github/setting-up-and-managing-your-github-user-account/setting-your-commit-email-address>`_.

Keep your branch up-to-date
===========================

If you are adding a new feature, make sure to merge ``develop`` into your feature
branch. If you are fixing a bug, merge ``main`` into your bug fix branch instead.

To update a feature branch, switch to the ``develop`` branch::

    git checkout develop

Fetch changes from the upstream branch and update ``develop``::

    git pull upstream develop --tags

Update your feature branch::

    git checkout your-awesome-feature-name
    git merge develop

Share your changes
==================

Update your remote branch::

    git push -u origin your-awesome-feature-name

You can then make a `pull request
<https://guides.github.com/activities/forking/#making-a-pull-request>`_ to orix's
``develop`` branch for new features and ``main`` branch for bug fixes. Good job!

Build and write documentation
=============================

We use `Sphinx <https://www.sphinx-doc.org/en/master/>`_ for documenting functionality.
Install necessary dependencies to build the documentation::

    pip install --editable .[doc]

.. note::

    The user guide notebooks require some small datasets to be downloaded via the
    :mod:`orix.data` module upon building the documentation. See the section on the
    :ref:`data module <adding-data-to-data-module>` for more details.

Then, build the documentation from the ``doc`` directory::

    cd doc
    make html

The documentation's HTML pages are built in the ``doc/build/html`` directory from files
in the `reStructuredText (reST)
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_
plaintext markup language. They should be accessible in the browser by typing
``file:///your-absolute/path/to/orix/doc/build/html/index.html`` in the address bar.

Tips for writing Jupyter Notebooks that are meant to be converted to reST text
files by `nbsphinx <https://nbsphinx.readthedocs.io/en/latest/>`_:

- Notebooks (with the ``.ipynb`` file extension) are ignored by git (listed in the
  ``.gitignore`` file). The ``-f`` `git flag
  <https://git-scm.com/docs/git-add#Documentation/git-add.txt--f>`_ must be added to
  ``git add -f notebook.ipynb`` in order to update an existing notebook or add a new
  one. Notebooks are ignored by git in general to avoid non-documentation changes to
  notebooks, like cell IDs, being pushed unnecessarily.
- All notebooks should have a Markdown (MD) cell with this message at the top,
  "This notebook is part of the ``orix`` documentation https://orix.readthedocs.io.
  Links to the documentation won't work from the notebook.", and have
  ``"nbsphinx": "hidden"`` in the cell metadata so that the message is not visible when
  displayed in the documentation.
- Use ``ax[0].imshow(...);`` to silence ``matplotlib`` output if a ``matplotlib``
  command is the last line in a cell.
- Refer to our API reference with this general MD
  ``[Vector3d.zvector()](reference.rst#orix.vector.Vector3d.zvector)``. Remember to add
  the parentheses ``()`` if the reference points to a function or method.
- Reference external APIs via standard MD like
  ``[Lattice](https://www.diffpy.org/diffpy.structure/mod_lattice.html#diffpy.structure.lattice.Lattice)``.
- The Sphinx gallery thumbnail used for a notebook is set by adding the
  ``nbsphinx-thumbnail`` tag to a code cell with an image output. The notebook must be
  added to the gallery in the relevant topic within the user guide to be included in the
  documentation pages.
- The ``furo`` Sphinx theme displays the documentation in a light or dark theme,
  depending on the browser/OS setting. It is important to make sure the documentation is
  readable with both themes. This means for example displaying all figures with a white
  background for axes labels and ticks and figure titles etc. to be readable.
- Whenever the documentation is built (locally or on the Read the Docs server),
  ``nbsphinx`` only runs the notebooks *without* any cell output stored. It is
  recommended that notebooks are stored without cell output, so that functionality
  within them are run and tested to ensure continued compatibility with code changes.
  Cell output should only be stored in notebooks which are too computationally intensive
  for the Read the Docs server to handle, which has a limit of 15 minutes and 3 GB of
  memory per `documentation build <https://docs.readthedocs.io/en/stable/builds.html>`_.

In general, we run all notebooks every time the documentation is built with Sphinx, to
ensure that all notebooks are compatible with the current API at all times. This is
important! For computationally expensive notebooks however, we store the cell outputs so
the documentation doesn't take too long to build, either by us locally or the Read The
Docs GitHub action. To check that the notebooks with stored cell outputs are compatible
with the current API, we run a scheduled GitHub Action every Monday morning which checks
that the notebooks run OK and that they produce the same output now as when they were
last executed. We use `nbval <https://nbval.readthedocs.io>`_ for this.

The user guide notebooks can be run interactively in the browser with the help of
`Binder <https://mybinder.readthedocs.io>`_. When creating a server from the orix source
code, Binder installs the packages listed in the `environment.yml` configuration file,
which must include all `doc` dependencies in `setup.py` necessary to run the notebooks.

Deprecations
============

We attempt to adhere to semantic versioning as best we can. This means that as little,
ideally no, functionality should break between minor releases. Deprecation warnings are
raised whenever possible and feasible for functions/methods/properties/arguments, so
that users get a heads-up one (minor) release before something is removed or changes,
with a possible alternative to be used.

The decorator should be placed right above the object signature to be deprecated::

    @deprecate(since=0.8, removal=0.9, alternative="bar")
    def foo(self, n):
        return n + 1

    @property
    @deprecate(since=0.9, removal=0.10, alternative="another", object_type="property")
    def this_property(self):
        return 2

Run and write tests
===================

All functionality in orix is tested with `pytest <https://docs.pytest.org>`_. The tests
reside in a ``tests`` module. Tests are short methods that call functions in ``orix``
and compare resulting output values with known answers. Install necessary dependencies
to run the tests::

   pip install --editable .[tests]

Some useful `fixtures <https://docs.pytest.org/en/latest/fixture.html>`_ are available
in the ``conftest.py`` file.

.. note::

    Some :mod:`orix.data` module tests check that data not part of the package
    distribution can be downloaded from the web, thus downloading some small datasets to
    your local cache. See the section on the
    :ref:`data module <adding-data-to-data-module>` for more details.

To run the tests::

   pytest --cov --pyargs orix

The ``--cov`` flag makes `coverage.py <https://coverage.readthedocs.io/en/latest/>`_
prints a nice report in the terminal. For an even nicer presentation, you can use
``coverage.py`` directly::

   coverage html

Then, you can open the created ``htmlcov/index.html`` in the browser and inspect the
coverage in more detail.

.. _adding-data-to-data-module:

Adding data to the data module
==============================

Test data for user guides and tests are included in the :mod:`orix.data` module via the
`pooch <https://www.fatiando.org/pooch/latest>`_ Python library. These are listed in a
file registry (``orix.data._registry.py``) with their file verification string (hash,
SHA256, obtained with e.g. ``sha256sum <file>``) and location, the latter potentially
not within the package but from the `orix-data <https://github.com/pyxem/orix-data>`_
repository or elsewhere, since some files are considered too large to include in the
package.

If a required dataset isn't in the package, but is in the registry, it can be downloaded
from the repository when the user passes ``allow_download=True`` to e.g.
``sdss_austenite()``. The dataset is then downloaded to a local cache, in the location
returned from ``pooch.os_cache("orix")``. The location can be overwritten with a global
``ORIX_DATA_DIR`` variable locally, e.g. by setting export ``ORIX_DATA_DIR=~/orix_data``
in ``~/.bashrc``. Pooch handles downloading, caching, version control, file verification
(against hash) etc. If we have updated the file hash, pooch will re-download it. If the
file is available in the cache, it can be loaded as the other files in the data module.

With every new version of orix, a new directory of data sets with the version name is
added to the cache directory. Any old directories are not deleted automatically, and
should then be deleted manually if desired.

Continuous integration (CI)
===========================

We use `GitHub Actions <https://github.com/pyxem/orix/actions>`_ to ensure that
orix can be installed on Windows, macOS and Linux (Ubuntu). After a successful
installation, the CI server runs the tests. After the tests return no errors, code
coverage is reported to `Coveralls
<https://coveralls.io/github/pyxem/orix?branch=develop>`_.
