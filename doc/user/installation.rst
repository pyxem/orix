============
Installation
============

orix can be installed with `pip <https://pypi.org/project/orix/>`__,
`conda <https://anaconda.org/conda-forge/orix>`__ or from source, and supports Python
>= 3.7. All alternatives are available on Windows, macOS and Linux.

.. _install-with-pip:

With pip
========

orix is availabe from the Python Package Index (PyPI), and can therefore be installed
with `pip <https://pip.pypa.io/en/stable>`__. To install, run the following::

    pip install orix

To update orix to the latest release::

    pip install --upgrade orix

To install a specific version of orix (say version 0.8.1)::

    pip install orix==0.8.1

.. _optional-dependencies:

With Anaconda
=============

To install with Anaconda, we recommend you install it in a `conda environment
<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__
with the `Miniconda distribution <https://docs.conda.io/en/latest/miniconda.html>`__.
To create an environment and activate it, run the following::

   conda create --name orix-env python=3.9
   conda activate orix-env

If you prefer a graphical interface to manage packages and environments, you can install
the `Anaconda distribution <https://docs.continuum.io/anaconda>`__ instead.

To install::

    conda install orix --channel conda-forge

To update orix to the latest release::

    conda update orix

To install a specific version of orix (say version 0.8.1)::

    conda install orix==0.8.1 -c conda-forge

.. _install-from-source:

From source
===========

The source code is hosted on `GitHub <https://github.com/pyxem/orix>`__. One way to
install orix from source is to clone the repository from `GitHub
<https://github.com/pyxem/orix>`__, and install with ``pip``::

    git clone https://github.com/pyxem/orix.git
    cd orix
    pip install --editable .

The source can also be downloaded as tarballs or zip archives via links like
`https://github.com/pyxem/orix/archive/v<major.minor.patch>/orix-<major.minor.patch>.tar.gz`_,
where the version ``<major.minor.patch>`` can be e.g. ``0.10.2``, and ``tar.gz`` can be
exchanged with ``zip``.

See the :ref:`contributing guide <setting-up-a-development-installation>` for how to set
up a development installation and keep it up to date.

.. _https://github.com/pyxem/orix/archive/v<major.minor.patch>/orix-<major.minor.patch>.tar.gz: https://github.com/pyxem/orix/archive/v<major.minor.patch>/orix-<major.minor.patch>.tar.gz