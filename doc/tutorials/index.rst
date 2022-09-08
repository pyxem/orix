=========
Tutorials
=========

This page contains more in-depth guides for using orix. It is broken up into
sections covering specific topics.

For shorter examples, see our :doc:`/examples/index`. For descriptions of
the functions, modules, and objects in orix, see the :doc:`/reference/index`.

The tutorials are live and available on MyBinder: |Binder|

.. |Binder| image:: https://static.mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/pyxem/orix/develop?filepath=doc/tutorials

Crystal geometry
================

These tutorials cover conventions for the unit cell, symmetry operations and relevant
reference frames, and also show how to operate with vectors in the crystal and sample
reference frames.

.. nbgallery::

    crystal_reference_frame
    point_groups
    crystal_directions
    inverse_pole_figures

Orientations
============

This tutorial covers how to sample orientation space of the proper point groups.

.. nbgallery::

    uniform_sampling_of_orientation_space

Vectors
=======

These tutorials cover how to visualize 3D vectors in the stereographic projection,
sample unit vectors and quantify and visualize the distribution of crystal poles in the
sample reference frame.

.. nbgallery::

    stereographic_projection
    s2_sampling
    pole_density_function

Clustering of (mis)orientations
===============================

These tutorials cover how to segment points in a crystal map into clusters based on the
(mis)orientation distances. This functionality was why orix was initially written.

.. nbgallery::

    clustering_across_fundamental_region_boundaries
    clustering_orientations
    clustering_misorientations

Crystal maps
============

This tutorial covers how to access and operate with orientations and other data in a
crystallographic map (typically acquired in a diffraction experiment)

.. nbgallery::

    crystal_map
