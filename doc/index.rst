====
orix
====

orix is an open-source Python library for analysing orientations and crystal symmetry.

The package defines objects and functions for the analysis of orientations accounting
for crystal symmetry. Functionality builds primarily on `NumPy <https://www.numpy.org>`_
and `Matplotlib <https://matplotlib.org>`_, and is heavily inspired by the MATLAB
package `MTEX <https://mtex-toolbox.github.io>`_.

.. toctree::
    :caption: Learning resources
    :hidden:

    examples/index.rst
    tutorials/index.rst
    Reference <reference/index.rst>
    related_projects.rst
    bibliography.rst

.. toctree::
    :caption: Help & development
    :hidden:

    installation.rst
    changelog.rst
    contributing.rst
    License <https://github.com/pyxem/orix/blob/develop/LICENSE>

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

Further details are available in the :doc:`installation guide <installation>`.

Learning resources
==================

.. See: https://sphinx-design.readthedocs.io/en/furo-theme/grids.html
.. grid:: 2 3 3 3
    :gutter: 2

    .. grid-item-card:: Examples
        :img-top: _static/img/colormap_banners/banner0.png
        :link: examples/index
        :link-type: doc

        Short recipies to common tasks using orix.

    .. grid-item-card:: Tutorials
        :img-top: _static/img/colormap_banners/banner1.png
        :link: tutorials/index
        :link-type: doc

        In-depth guides for using orix.

    .. grid-item-card:: Reference
        :img-top: _static/img/colormap_banners/banner2.png
        :link: reference/index
        :link-type: doc

        Descriptions of functions, modules, and objects in orix.

Contributing
============

orix is a community project maintained for and by its users. There are many ways you can
help!

- Help other users in `our GitHub discussions
  <https://github.com/pyxem/orix/discussions>`__
- report a bug or request a feature `on GitHub
  <https://github.com/pyxem/orix/issues>`__
- or improve the :doc:`documentation and code <contributing>`

Citing orix
===========

If analysis using orix forms a part of published work please cite the paper (`journal
<https://doi.org/10.1107/S1600576720011103>`_, `arXiv
<https://arxiv.org/abs/2001.02716>`_) and the package itself via the Zenodo DOI:
https://doi.org/10.5281/zenodo.3459662.

orix is released under the GPL v3 license.

Work using orix
---------------

* P. Harrison, X. Zhou, S. M. Das, P. Lhuissier, C. H. Liebscher, M. Herbig, W. Ludwig,
  E. F. Rauch: Reconstructing Dual-Phase Nanometer Scale Grains within a Pearlitic Steel
  Tip in 3D through 4D-Scanning Precession Electron Diffraction Tomography and Automated
  Crystal Orientation Mapping, *Ultramicroscopy* 113536 (2022) (`journal
  <https://doi.org/10.1016/j.ultramic.2022.113536>`__).
* N. Cautaerts, P. Crout, H. W. Ånes, E. Prestat, J. Jeong, G. Dehm, C. H. Liebscher:
  Free, flexible and fast: Orientation mapping using the multi-core and GPU-accelerated
  template matching capabilities in the python-based open source 4D-STEM analysis
  toolbox Pyxem, *Ultramicroscopy* 113517 (2022) (`journal
  <https://doi.org/10.1016/j.ultramic.2022.113517>`__, `arXiv
  <https://arxiv.org/abs/2111.07347>`__).
* D. N. Johnstone, B. H. Martineau, P. Crout, P. A. Midgley, A. S. Eggeman:
  Density-based clustering of crystal (mis)orientations and the orix Python library,
  *Journal of Applied Crystallography* 53(5) (2020) (`journal
  <https://doi.org/10.1107/S1600576720011103>`__, `arXiv
  <https://arxiv.org/abs/2001.02716>`__).
