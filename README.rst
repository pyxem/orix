|build_status|_ |Coveralls|_ |docs|_ |pypi_version|_  |downloads|_ |black|_ |doi|_

.. |build_status| image:: https://github.com/pyxem/orix/workflows/build/badge.svg
.. _build_status: https://github.com/pyxem/orix/actions

.. |Coveralls| image:: https://coveralls.io/repos/github/pyxem/orix/badge.svg?branch=master
.. _Coveralls: https://coveralls.io/github/pyxem/orix?branch=master

.. |docs| image:: https://readthedocs.org/projects/orix/badge/?version=latest
.. _docs: https://orix.readthedocs.io/en/latest

.. |pypi_version| image:: http://img.shields.io/pypi/v/orix.svg?style=flat
.. _pypi_version: https://pypi.python.org/pypi/orix

.. |downloads| image:: https://anaconda.org/conda-forge/orix/badges/downloads.svg
.. _downloads: https://anaconda.org/conda-forge/orix

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _black: https://github.com/psf/black

.. |doi| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3459662.svg
.. _doi: https://doi.org/10.5281/zenodo.3459662

orix is an open-source python library for analysing orientations and crystal symmetry.

The package defines objects and functions for the analysis of orientations represented
as quaternions or 3D rotation vectors accounting for crystal symmetry. Functionality
builds primarily on top of `numpy <http://www.numpy.org/>`__ and
`matplotlib <https://matplotlib.org/>`__ and is heavily inspired by the
`MATLAB <https://www.mathworks.com/products/matlab.html>`__ package
`MTEX <http://mtex-toolbox.github.io/>`__.

If analysis using orix forms a part of published work please cite the manuscript at the
following `link <https://onlinelibrary.wiley.com/iucr/doi/10.1107/S1600576720011103>`_.
You can also find demos in the `orix-demos <https://github.com/pyxem/orix-demos>`_ repo.

orix is released under the GPL v3 license.