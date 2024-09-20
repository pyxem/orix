============
Installation
============

orix can be installed with `pip <https://pypi.org/project/orix/>`__,
`conda <https://anaconda.org/conda-forge/orix>`__ or from source, and supports Python
>= 3.10.
All alternatives are available on Windows, macOS and Linux.

.. _install-with-pip:

With pip
========

orix is availabe from the Python Package Index (PyPI), and can therefore be installed
with `pip <https://pip.pypa.io/en/stable>`__.
To install all of orix's functionality, do::

    pip install orix[all]

To install only the strictly required dependencies with limited functionality, do::

    pip install orix

See :ref:`dependencies` for the base and optional dependencies and alternatives for how
to install these.

To update orix to the latest release::

    pip install --upgrade orix

To install a specific version of orix (say version 0.12.1)::

    pip install orix==0.12.1

.. _install-with-anaconda:

With Anaconda
=============

To install with Anaconda, we recommend you install it in a `conda environment
<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__
with the `Miniconda distribution <https://docs.conda.io/en/latest/miniconda.html>`__.
To create an environment and activate it, run the following::

   conda create --name orix-env python=3.12
   conda activate orix-env

If you prefer a graphical interface to manage packages and environments, you can install
the `Anaconda distribution <https://docs.continuum.io/anaconda>`__ instead.

To install all of orix's functionality, do::

    conda install orix --channel conda-forge

To install only the strictly required dependencies with limited functionality, do::

    conda install orix-base -c conda-forge

See :ref:`dependencies` for the base and optional dependencies and alternatives for how
to install these.

To update orix to the latest release::

    conda update orix

To install a specific version of orix (say version 0.12.1)::

    conda install orix==0.12.1 -c conda-forge

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


.. _dependencies:

Dependencies
============

orix builds on the great work and effort of many people.
This is a list of core package dependencies:

================================================ ================================================
Package                                          Purpose
================================================ ================================================
:doc:`dask<dask:index>`                          Out-of-memory processing of data larger than RAM
:doc:`diffpy.structure <diffpy.structure:index>` Handling of crystal structures
:doc:`h5py <h5py:index>`                         Read/write of HDF5 files
:doc:`matplotlib <matplotlib:index>`             Visualization
`matplotlib-scalebar`_                           Scale bar for crystal map plots
:doc:`numba <numba:index>`                       CPU acceleration
:doc:`numpy <numpy:index>`                       Handling of N-dimensional arrays
:doc:`pooch <pooch:api/index>`                   Downloading and caching of datasets
:doc:`scipy <scipy:index>`                       Optimization algorithms, filtering and more
`tqdm <https://tqdm.github.io/>`__               Progressbars
================================================ ================================================

.. _matplotlib-scalebar: https://github.com/ppinard/matplotlib-scalebar

Some functionality requires optional dependencies:

=================== ===========================================
Package             Purpose                                    
=================== ===========================================
`numpy-quaternion`_ Faster quaternion and vector multiplication
=================== ===========================================

.. _numpy-quaternion: https://quaternion.readthedocs.io/en/stable/

Optional dependencies can be installed either with ``pip install orix[all]`` or by
installing each dependency separately, such as ``pip install orix numpy-quaternion``.
