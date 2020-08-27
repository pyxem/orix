=============
API reference
=============

This reference manual details the public classes, modules and functions in orix as
generated from their docstrings. Many of the docstrings contain examples.

.. caution::

    orix is in an alpha stage, and there will likely be breaking changes with each
    release.

....

Object3d
========

.. automodule:: orix.base
    :members:

....

quaternion
==========

.. automodule:: orix.quaternion

Quaternion
----------

.. automodule:: orix.quaternion.quaternion
    :members:
    :show-inheritance:

Rotation
--------

.. automodule:: orix.quaternion.rotation
    :members:
    :show-inheritance:

Misorientation and Orientation
------------------------------

.. automodule:: orix.quaternion.orientation
    :members:
    :show-inheritance:

OrientationRegion
-----------------

.. automodule:: orix.quaternion.orientation_region
    :members:
    :show-inheritance:

Symmetry
--------

.. automodule:: orix.quaternion.symmetry
    :members:
    :show-inheritance:

....

vector
======

.. automodule:: orix.vector

Vector3d
--------

.. automodule:: orix.vector.vector3d

.. autoclass:: orix.vector.vector3d.Vector3d
    :members:
    :show-inheritance:

NeoEuler
--------

.. automodule:: orix.vector.neo_euler

.. autoclass:: orix.vector.neo_euler.NeoEuler
    :members:
    :show-inheritance:

AxAngle
~~~~~~~

.. autoclass:: orix.vector.neo_euler.AxAngle
    :members:
    :show-inheritance:

Homochoric
~~~~~~~~~~

.. autoclass:: orix.vector.neo_euler.Homochoric
    :members:
    :show-inheritance:

Rodrigues
~~~~~~~~~

.. autoclass:: orix.vector.neo_euler.Rodrigues
    :members:
    :show-inheritance:

SphericalRegion
---------------

.. automodule:: orix.vector.spherical_region
    :members:
    :show-inheritance:

....

scalar
======

.. automodule:: orix.scalar
    :members:
    :show-inheritance:

....

sampling
========

Generators
----------

.. automodule:: orix.sampling.sample_generators
    :members:
    :show-inheritance:

Utilities
---------

.. automodule:: orix.sampling.sampling_utils
    :members:
    :show-inheritance:

....

crystal_map
===========

.. automodule:: orix.crystal_map

Phase
-----

.. autoclass:: orix.crystal_map.phase_list.Phase
    :members:
    :show-inheritance:

    .. automethod:: __init__

PhaseList
---------

.. autoclass:: orix.crystal_map.phase_list.PhaseList
    :members:
    :show-inheritance:

    .. automethod:: __init__

CrystalMap
----------

.. autoclass:: orix.crystal_map.crystal_map.CrystalMap
    :members:
    :show-inheritance:

    .. automethod:: __init__

CrystalMapProperties
--------------------

.. autoclass:: orix.crystal_map.crystal_map.CrystalMapProperties
    :members:
    :show-inheritance:

....

io
==

.. automodule:: orix.io
    :members:

ANG
---

.. automodule:: orix.io.plugins.ang
    :members:
    :show-inheritance:

EMsoft h5ebsd
-------------

.. automodule:: orix.io.plugins.emsoft_h5ebsd
    :members:
    :show-inheritance:

orix HDF5
---------

.. automodule:: orix.io.plugins.orix_hdf5
    :members:
    :show-inheritance:

....

plot
====

.. automodule:: orix.plot

RotationPlot
------------

.. autoclass:: orix.plot.rotation_plot.RotationPlot
    :members:
    :show-inheritance:

RodriguesPlot
-------------

.. autoclass:: orix.plot.rotation_plot.RodriguesPlot
    :members:
    :show-inheritance:

AxAnglePlot
-----------

.. autoclass:: orix.plot.rotation_plot.AxAnglePlot
    :members:
    :show-inheritance:

CrystalMapPlot
--------------

.. automodule:: orix.plot.crystal_map_plot

.. autoclass:: orix.plot.crystal_map_plot.CrystalMapPlot
    :members:
    :show-inheritance:
