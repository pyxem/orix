.. include:: ../README.rst

Work using orix
---------------

* D. N. Johnstone, B. H. Martineau, P. Crout, P. A. Midgley, A. S. Eggeman:
  Density-based clustering of crystal (mis)orientations and the orix Python library,
  Journal of Applied Crystallography 53(5) (2020), (`journal
  <https://doi.org/10.1107/S1600576720011103>`_, `arXiv
  <https://arxiv.org/abs/2001.02716>`_).

.. toctree::
    :hidden:
    :caption: Getting started

    installation.rst


.. _user-guide:

User guide
----------

Crystal geometry
~~~~~~~~~~~~~~~~

Conventions for the unit cell, symmetry operations and relevant reference frames, and
how to operate with vectors in the crystal and sample reference frames.

.. nbgallery::
    :caption: User guide

    crystal_reference_frame.ipynb
    crystal_directions.ipynb
    point_groups.ipynb
    inverse_pole_figures.ipynb

Orientations
~~~~~~~~~~~~

.. nbgallery::

    uniform_sampling_of_orientation_space.ipynb

Vectors
~~~~~~~

.. nbgallery::

    stereographic_projection.ipynb

Clustering of (mis)orientations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. nbgallery::

    clustering_across_fundamental_region_boundaries.ipynb
    clustering_orientations.ipynb
    clustering_misorientations.ipynb

Crystallographic maps
~~~~~~~~~~~~~~~~~~~~~

.. nbgallery::

    crystal_map.ipynb

.. toctree::
    :hidden:
    :caption: Help & reference

    reference.rst
    bibliography.rst
    changelog.rst
    contributing.rst
    related_projects.rst
