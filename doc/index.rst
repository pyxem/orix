============================
orix |release| documentation
============================

orix is an open-source Python library for analysing orientations and crystal symmetry.

The package defines objects and functions for the analysis of orientations accounting
for crystal symmetry.
Functionality builds primarily on `NumPy <https://www.numpy.org>`_ and `Matplotlib
<https://matplotlib.org>`_.
Initiation of the package was inspired by `MTEX <https://mtex-toolbox.github.io>`_.

.. toctree::
    :hidden:
    :titlesonly:

    user/index.rst
    reference/index.rst
    dev/index.rst
    changelog.rst

Installation
============

orix can be installed with `pip <https://pypi.org/project/orix>`__ or `conda
<https://anaconda.org/conda-forge/orix>`__:

.. tab-set::

    .. tab-item:: pip

        .. code-block:: bash

            pip install orix

    .. tab-item:: conda

        .. code-block:: bash

            conda install orix -c conda-forge

Further details are available in the :doc:`installation guide <user/installation>`.

Learning resources
==================

.. See: https://sphinx-design.readthedocs.io/en/furo-theme/grids.html
.. grid:: 2
    :gutter: 2

    .. grid-item-card::
        :link: tutorials/index
        :link-type: doc

        :octicon:`book;2em;sd-text-info` Tutorials
        ^^^

        In-depth guides for using orix.

    .. grid-item-card::
        :link: examples/index
        :link-type: doc

        :octicon:`zap;2em;sd-text-info` Examples
        ^^^

        Short recipies to common tasks using orix.

    .. grid-item-card::
        :link: reference/index
        :link-type: doc

        :octicon:`code;2em;sd-text-info` API reference
        ^^^

        Descriptions of all functions, modules, and objects in orix.

    .. grid-item-card::
        :link: dev/index
        :link-type: doc

        :octicon:`people;2em;sd-text-info` Contributing
        ^^^

        orix is a community project maintained for and by its users. There are many ways
        you can help!

Citing orix
===========

If analysis using orix forms a part of published work please cite the paper (`journal
<https://doi.org/10.1107/S1600576720011103>`_, `arXiv
<https://arxiv.org/abs/2001.02716>`_) and the package itself via the Zenodo DOI:
https://doi.org/10.5281/zenodo.3459662.

orix is released under the GPL v3 license.
