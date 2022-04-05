.. raw:: html

    <p>
      <h1>
        <a href="https://orix.readthedocs.io"><img valign="middle" src="https://raw.githubusercontent.com/pyxem/orix/master/doc/_static/img/orix_logo.png" width="50" alt="orix logo"/></a>
        orix
      </h1>
    </p>

|binder|_ |build_status|_ |Coveralls|_ |docs|_ |pypi_version|_  |downloads|_ |black|_ |doi|_

.. |binder| image:: https://mybinder.org/badge_logo.svg
.. _binder: https://mybinder.org/v2/gh/pyxem/orix/HEAD

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
builds primarily on `numpy <http://www.numpy.org/>`_ and `matplotlib
<https://matplotlib.org/>`_ and is heavily inspired by the MATLAB package `MTEX
<http://mtex-toolbox.github.io/>`_.

If analysis using orix forms a part of published work please cite the paper (`journal
<https://doi.org/10.1107/S1600576720011103>`_, `arXiv
<https://arxiv.org/abs/2001.02716>`_).

`Documentation <https://orix.readthedocs.io>`_ is hosted on Read the Docs with a
complete reference for functions and classes, as well as a user guide in the form of
Jupyter notebooks. These notebooks can be inspected statically on the web page or via
`nbviewer <https://nbviewer.org/github/pyxem/orix/tree/master/doc>`_,
interactively in the browser by clicking the Binder link above and navigating to the
``doc/`` directory, or on your own computer by downloading them and the corresponding
data. We hope you find them useful!

orix is released under the GPL v3 license.
