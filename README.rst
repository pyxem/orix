|logo| orix
===========

.. |logo| image:: https://raw.githubusercontent.com/pyxem/orix/develop/doc/_static/img/orix_logo.png
  :width: 50

orix is an open-source Python library for analysing orientations and crystal symmetry.

The package defines objects and functions for the analysis of orientations represented
as quaternions and 3D rotation vectors, accounting for crystal symmetry.
Functionality builds primarily on `NumPy <https://www.numpy.org>`_ and `Matplotlib
<https://matplotlib.org>`_.
Initiation of the package was inspired by `MTEX <https://mtex-toolbox.github.io>`_.

orix is released under the GPL v3 license.

.. |pypi_version| image:: https://img.shields.io/pypi/v/orix.svg?style=flat
   :target: https://pypi.python.org/pypi/orix

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/orix.svg?logo=conda-forge&logoColor=white
   :target: https://anaconda.org/conda-forge/orix

.. |build_status| image:: https://github.com/pyxem/orix/workflows/build/badge.svg
   :target: https://github.com/pyxem/orix/actions/workflows/build.yml

.. |python| image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/

.. |Coveralls| image:: https://coveralls.io/repos/github/pyxem/orix/badge.svg?branch=develop
   :target: https://coveralls.io/github/pyxem/orix?branch=develop

.. |pypi_downloads| image:: https://img.shields.io/pypi/dm/orix.svg?label=PyPI%20downloads
   :target: https://pypi.org/project/orix/

.. |conda_downloads| image:: https://img.shields.io/conda/dn/conda-forge/orix.svg?label=Conda%20downloads
   :target: https://anaconda.org/conda-forge/orix

.. |doi| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3459662.svg
   :target: https://doi.org/10.5281/zenodo.3459662

.. |GPLv3| image:: https://img.shields.io/github/license/pyxem/orix
   :target: https://opensource.org/license/GPL-3.0

.. |GH-discuss| image:: https://img.shields.io/badge/GitHub-Discussions-green?logo=github
   :target: https://github.com/pyxem/orix/discussions

.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/pyxem/orix/HEAD

.. |docs| image:: https://readthedocs.org/projects/orix/badge/?version=latest
   :target: https://orix.readthedocs.io/en/latest

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

+----------------------+------------------------------------------------+
| Deployment           | |pypi_version| |conda|                         |
+----------------------+------------------------------------------------+
| Build status         | |build_status| |docs| |python|                 |
+----------------------+------------------------------------------------+
| Metrics              | |Coveralls|                                    |
+----------------------+------------------------------------------------+
| Activity             | |pypi_downloads| |conda_downloads|             |
+----------------------+------------------------------------------------+
| Citation             | |doi|                                          |
+----------------------+------------------------------------------------+
| License              | |GPLv3|                                        |
+----------------------+------------------------------------------------+
| Community            | |GH-discuss|                                   |
+----------------------+------------------------------------------------+
| Formatter            | |black|                                        |
+----------------------+------------------------------------------------+

Documentation
-------------

Refer to the `documentation <https://orix.readthedocs.io>`__ for detailed installation
instructions, a user guide, and the `changelog
<https://orix.readthedocs.io/en/latest/changelog.html>`_.

Installation
------------

orix can be installed with ``pip``::

    pip install orix

or ``conda``::

    conda install orix -c conda-forge

The source code is hosted in `GitHub <https://github.com/pyxem/orix>`_, and can also be
downloaded from `PyPI <https://pypi.org/project/orix>`_ and
`Anaconda <https://anaconda.org/conda-forge/orix>`_.

Further details are available in the
`installation guide <https://orix.readthedocs.io/en/latest/user/installation.html>`_.

Citing orix
-----------

If analysis using orix forms a part of published work please cite the paper (`journal
<https://doi.org/10.1107/S1600576720011103>`_, `arXiv
<https://arxiv.org/abs/2001.02716>`_) and `the software
<https://doi.org/10.5281/zenodo.3459662>`_.
