=============
API reference
=============

This reference manual details the public classes, modules and functions in orix as
generated from their docstrings. Many of the docstrings contain examples.

.. caution::

    orix is in an alpha stage, and there will likely be breaking changes with each
    release.

....

quaternion
==========

.. automodule:: orix.quaternion

Quaternion
----------

.. autoclass:: orix.quaternion.Quaternion
    :members:
    :show-inheritance:

Rotation
--------

.. automodule:: orix.quaternion.rotation

.. autoclass:: orix.quaternion.rotation.Rotation
    :members:
    :undoc-members:
    :show-inheritance:

Misorientation and Orientation
------------------------------

.. automodule:: orix.quaternion.orientation

.. autoclass:: orix.quaternion.orientation.Misorientation
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: orix.quaternion.orientation.Orientation
    :members:
    :undoc-members:
    :show-inheritance:

OrientationRegion
-----------------

.. automodule:: orix.quaternion.orientation_region

.. autoclass:: orix.quaternion.orientation_region.OrientationRegion
    :members:
    :undoc-members:
    :show-inheritance:

Symmetry
--------

.. automodule:: orix.quaternion.symmetry

.. autoclass:: orix.quaternion.symmetry.Symmetry
    :members:
    :undoc-members:
    :show-inheritance:

....

vector
======

.. automodule:: orix.vector

Vector3d
--------

.. autoclass:: orix.vector.Vector3d
    :members:
    :undoc-members:
    :show-inheritance:

NeoEuler
--------

.. automodule:: orix.vector.neo_euler

.. autoclass:: orix.vector.neo_euler.NeoEuler
    :members:
    :undoc-members:
    :show-inheritance:

AxAngle
~~~~~~~

.. autoclass:: orix.vector.neo_euler.AxAngle
    :members:
    :undoc-members:
    :show-inheritance:

Homochoric
~~~~~~~~~~

.. autoclass:: orix.vector.neo_euler.Homochoric
    :members:
    :undoc-members:
    :show-inheritance:

Rodrigues
~~~~~~~~~

.. autoclass:: orix.vector.neo_euler.Rodrigues
    :members:
    :undoc-members:
    :show-inheritance:

SphericalRegion
---------------

.. automodule:: orix.vector.spherical_region

.. autoclass:: orix.vector.spherical_region.SphericalRegion
    :members:
    :undoc-members:
    :show-inheritance:

....

scalar
======

.. automodule:: orix.scalar

.. autoclass:: orix.scalar.Scalar
    :members:
    :undoc-members:
    :show-inheritance:

....

sampling
========

Generators
----------

.. automodule:: orix.sampling.sample_generators
    :members:
    :undoc-members:
    :show-inheritance:

Utilities
---------

.. automodule:: orix.sampling.sampling_utils
    :members:
    :undoc-members:
    :show-inheritance:

....

crystal_map
===========

.. automodule:: orix.crystal_map

Phase
-----

.. autoclass:: orix.crystal_map.phase_list.Phase
    :members:
    :undoc-members:
    :show-inheritance:

PhaseList
---------

.. autoclass:: orix.crystal_map.phase_list.PhaseList
    :members:
    :undoc-members:
    :show-inheritance:

CrystalMap
----------

.. autoclass:: orix.crystal_map.crystal_map.CrystalMap
    :members:
    :undoc-members:
    :show-inheritance:

CrystalMapProperties
--------------------

.. autoclass:: orix.crystal_map.crystal_map.CrystalMapProperties
    :members:
    :undoc-members:
    :show-inheritance:

....

io
==

.. automodule:: orix.io
    :members:
    :undoc-members:

ANG
---

.. automodule:: orix.io.plugins.ang
    :members:
    :undoc-members:
    :show-inheritance:

EMsoft h5ebsd
-------------

.. automodule:: orix.io.plugins.emsoft_h5ebsd
    :members:
    :undoc-members:
    :show-inheritance:

orix HDF5
---------

.. automodule:: orix.io.plugins.orix_hdf5
    :members:
    :undoc-members:
    :show-inheritance:

....

plot
====

RotationPlot
------------

.. autoclass:: orix.plot.rotation_plot.RotationPlot
    :members:
    :undoc-members:
    :show-inheritance:

RodriguesPlot
-------------

.. autoclass:: orix.plot.rotation_plot.RodriguesPlot
    :members:
    :undoc-members:
    :show-inheritance:

AxAnglePlot
-----------

.. autoclass:: orix.plot.rotation_plot.AxAnglePlot
    :members:
    :undoc-members:
    :show-inheritance:

CrystalMapPlot
--------------

.. automodule:: orix.plot.crystal_map_plot

.. autoclass:: orix.plot.crystal_map_plot.CrystalMapPlot
    :members:
    :undoc-members:
    :show-inheritance:
