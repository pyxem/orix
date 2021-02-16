=============
API reference
=============

This reference manual details the public modules, classes, and functions in orix, as
generated from their docstrings. Many of the docstrings contain examples, however, see
the user guide for how to use orix.

.. caution::

    orix is in an alpha stage, so there will be breaking changes with each release.

.. module:: orix

The list of top modules:

.. autosummary::
    base
    crystal_map
    io
    plot
    projections
    quaternion
    sampling
    scalar
    vector

....

base
====
.. automodule:: orix.base
    :members:
    :show-inheritance:

....

crystal_map
===========
.. automodule:: orix.crystal_map
.. currentmodule:: orix.crystal_map
.. autosummary::
    CrystalMap
    CrystalMapProperties
    Phase
    PhaseList

Phase
-----
.. currentmodule:: orix.crystal_map.Phase
.. autosummary::
    deepcopy
.. autoclass:: orix.crystal_map.Phase
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

PhaseList
---------
.. currentmodule:: orix.crystal_map.PhaseList
.. autosummary::
    add
    deepcopy
    id_from_name
    sort_by_id
.. autoclass:: orix.crystal_map.PhaseList
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

CrystalMap
----------
.. currentmodule:: orix.crystal_map.CrystalMap
.. autosummary::
    deepcopy
    get_map_data
.. autoclass:: orix.crystal_map.CrystalMap
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

CrystalMapProperties
--------------------
.. currentmodule:: orix.crystal_map.CrystalMapProperties
.. autoclass:: orix.crystal_map.CrystalMapProperties
    :members:
    :undoc-members:
    :show-inheritance:

....

io
===
.. currentmodule:: orix.io
.. autosummary::
    load
    loadang
    loadctf
    plugins
    save

.. automodule:: orix.io
    :members:

plugins
-------
.. automodule:: orix.io.plugins
.. currentmodule:: orix.io.plugins
.. autosummary::
    ang
    emsoft_h5ebsd
    orix_hdf5

ang
~~~
.. automodule:: orix.io.plugins.ang
    :members:
    :undoc-members:
    :show-inheritance:

emsoft_h5ebsd
~~~~~~~~~~~~~
.. automodule:: orix.io.plugins.emsoft_h5ebsd
    :members:
    :undoc-members:
    :show-inheritance:

orix_hdf5
~~~~~~~~~
.. automodule:: orix.io.plugins.orix_hdf5
    :members:
    :undoc-members:
    :show-inheritance:

....

plot
====
.. automodule:: orix.plot
.. currentmodule:: orix.plot
.. autosummary::
    AxAnglePlot
    CrystalMapPlot
    RodriguesPlot
    RotationPlot
    StereographicPlot

AxAnglePlot
-----------
.. currentmodule:: orix.plot.AxAnglePlot
.. autoclass:: orix.plot.AxAnglePlot
    :members:
    :undoc-members:
    :show-inheritance:

CrystalMapPlot
--------------
.. currentmodule:: orix.plot.CrystalMapPlot
.. autosummary::
    add_colorbar
    add_overlay
    add_scalebar
    plot_map
    remove_padding
.. autoclass:: orix.plot.CrystalMapPlot
    :members:
    :undoc-members:
    :show-inheritance:

RodriguesPlot
-------------
.. currentmodule:: orix.plot.RodriguesPlot
.. autoclass:: orix.plot.RodriguesPlot
    :members:
    :undoc-members:
    :show-inheritance:

RotationPlot
------------
.. currentmodule:: orix.plot.RotationPlot
.. autoclass:: orix.plot.RotationPlot
    :members:
    :undoc-members:
    :show-inheritance:

StereographicPlot
-----------------
.. currentmodule:: orix.plot.StereographicPlot
.. autosummary::
    azimuth_grid
    polar_grid
    show_hemisphere_label
    set_labels
    symmetry_marker
.. autoclass:: orix.plot.StereographicPlot
    :show-inheritance:
    :members: azimuth_grid, hemisphere, name, polar_grid, pole, set_labels, show_hemisphere_label, symmetry_marker

....

projections
===========
.. automodule:: orix.projections
.. currentmodule:: orix.projections
.. autosummary::
    StereographicProjection
    InverseStereographicProjection

StereographicProjection
-----------------------
.. currentmodule:: orix.projections.StereographicProjection
.. autosummary::
    spherical2xy
    spherical2xy_split
    vector2xy
    vector2xy_split
.. autoclass:: orix.projections.StereographicProjection
    :members:
    :undoc-members:

    .. automethod:: __init__

InverseStereographicProjection
------------------------------
.. currentmodule:: orix.projections.InverseStereographicProjection
.. autosummary::
    xy2spherical
    xy2vector
.. autoclass:: orix.projections.InverseStereographicProjection
    :members:
    :undoc-members:

    .. automethod:: __init__

....

quaternion
==========
.. automodule:: orix.quaternion
.. currentmodule:: orix.quaternion
.. autosummary::
    Orientation
    OrientationRegion
    Misorientation
    Quaternion
    Rotation
    Symmetry

Orientation and Misorientation
------------------------------
.. automodule:: orix.quaternion.orientation
.. autoclass:: orix.quaternion.Orientation
    :show-inheritance:
    :members:
    :undoc-members:
.. autoclass:: orix.quaternion.Misorientation
    :show-inheritance:
    :members:
    :undoc-members:

OrientationRegion
-----------------
.. autoclass:: orix.quaternion.OrientationRegion
    :show-inheritance:
    :members:
    :undoc-members:

Quaternion
----------
.. autoclass:: orix.quaternion.Quaternion
    :show-inheritance:
    :members:
    :undoc-members:

Rotation
--------
.. autoclass:: orix.quaternion.Rotation
    :show-inheritance:
    :members:
    :undoc-members:

Symmetry
--------
.. autoclass:: orix.quaternion.Symmetry
    :show-inheritance:
    :members:
    :undoc-members:

....

sampling
========

.. automodule:: orix.sampling
    :members:
    :show-inheritance:

....

scalar
======

.. automodule:: orix.scalar
    :members:
    :show-inheritance:

....

vector
======

.. automodule:: orix.vector
    :members:
    :show-inheritance:

.. Object3d
.. ========
..
.. .. automodule:: orix.base
..     :members:
..
.. ....
..
.. quaternion
.. ==========
..
.. .. automodule:: orix.quaternion
..
.. Quaternion
.. ----------
..
.. .. automodule:: orix.quaternion.quaternion
..     :members:
..     :show-inheritance:
..
.. Rotation
.. --------
..
.. .. automodule:: orix.quaternion.rotation
..     :members:
..     :show-inheritance:
..
.. Misorientation and Orientation
.. ------------------------------
..
.. .. automodule:: orix.quaternion.orientation
..     :members:
..     :show-inheritance:
..
.. OrientationRegion
.. -----------------
..
.. .. automodule:: orix.quaternion.orientation_region
..     :members:
..     :show-inheritance:
..
.. Symmetry
.. --------
..
.. .. automodule:: orix.quaternion.symmetry
..     :members:
..     :show-inheritance:
..
.. ....
..
.. vector
.. ======
..
.. .. automodule:: orix.vector
..
.. Vector3d
.. --------
..
.. .. automodule:: orix.vector.vector3d
..
.. .. autoclass:: orix.vector.vector3d.Vector3d
..     :members:
..     :show-inheritance:
..
.. NeoEuler
.. --------
..
.. .. automodule:: orix.vector.neo_euler
..
.. .. autoclass:: orix.vector.neo_euler.NeoEuler
..     :members:
..     :show-inheritance:
..
.. AxAngle
.. ~~~~~~~
..
.. .. autoclass:: orix.vector.neo_euler.AxAngle
..     :members:
..     :show-inheritance:
..
.. Homochoric
.. ~~~~~~~~~~
..
.. .. autoclass:: orix.vector.neo_euler.Homochoric
..     :members:
..     :show-inheritance:
..
.. Rodrigues
.. ~~~~~~~~~
..
.. .. autoclass:: orix.vector.neo_euler.Rodrigues
..     :members:
..     :show-inheritance:
..
.. SphericalRegion
.. ---------------
..
.. .. automodule:: orix.vector.spherical_region
..     :members:
..     :show-inheritance:
..
.. ....
..
.. scalar
.. ======
..
.. .. automodule:: orix.scalar
..     :members:
..     :show-inheritance:
..
.. ....
..
.. sampling
.. ========
..
.. Generators
.. ----------
..
.. .. automodule:: orix.sampling.sample_generators
..     :members:
..     :show-inheritance:
..
.. Utilities
.. ---------
..
.. .. automodule:: orix.sampling.sampling_utils
..     :members:
..     :show-inheritance:
..
.. ....
..
.. crystal_map
.. ===========
..
.. .. automodule:: orix.crystal_map
..
.. Phase
.. -----
..
.. .. autoclass:: orix.crystal_map.phase_list.Phase
..     :members:
..     :show-inheritance:
..
..     .. automethod:: __init__
..
.. PhaseList
.. ---------
..
.. .. autoclass:: orix.crystal_map.phase_list.PhaseList
..     :members:
..     :show-inheritance:
..
..     .. automethod:: __init__
..
.. CrystalMap
.. ----------
..
.. .. autoclass:: orix.crystal_map.crystal_map.CrystalMap
..     :members:
..     :show-inheritance:
..
..     .. automethod:: __init__
..
.. CrystalMapProperties
.. --------------------
..
.. .. autoclass:: orix.crystal_map.crystal_map.CrystalMapProperties
..     :members:
..     :show-inheritance:
..
.. ....
..
.. io
.. ==
..
.. .. automodule:: orix.io
..     :members:
..
.. ANG
.. ---
..
.. .. automodule:: orix.io.plugins.ang
..     :members:
..     :show-inheritance:
..
.. EMsoft h5ebsd
.. -------------
..
.. .. automodule:: orix.io.plugins.emsoft_h5ebsd
..     :members:
..     :show-inheritance:
..
.. orix HDF5
.. ---------
..
.. .. automodule:: orix.io.plugins.orix_hdf5
..     :members:
..     :show-inheritance:
..
.. ....
..
.. plot
.. ====
..
.. .. automodule:: orix.plot
..
.. RotationPlot
.. ------------
..
.. .. autoclass:: orix.plot.rotation_plot.RotationPlot
..     :members:
..     :show-inheritance:
..
.. RodriguesPlot
.. -------------
..
.. .. autoclass:: orix.plot.rotation_plot.RodriguesPlot
..     :members:
..     :show-inheritance:
..
.. AxAnglePlot
.. -----------
..
.. .. autoclass:: orix.plot.rotation_plot.AxAnglePlot
..     :members:
..     :show-inheritance:
..
.. CrystalMapPlot
.. --------------
..
.. .. automodule:: orix.plot.crystal_map_plot
..
.. .. autoclass:: orix.plot.crystal_map_plot.CrystalMapPlot
..     :members:
..     :show-inheritance:
..
