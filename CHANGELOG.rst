=========
Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_, and
this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Unreleased
==========

2021-12-21 - version 0.8.0
==========================

Added
-----
- `FundamentalSector` class of vector normals describing a fundamental sector in the
  stereographic projection, typically the inverse pole figure of a `Symmetry`.
- `Symmetry.fundamental_sector` attribute with a `FundamentalSector` for that symmetry.
- `StereographicPlot.restrict_to_sector()` to restrict the stereographic projection to
  a sector, typically the inverse pole figure of a `Symmetry`.
- `StereographicPlot.stereographic_grid()` to control the azimuth and polar grid lines.
- Sampling of vectors in UV mesh on a unit sphere (*S2*).
- `ndim` attribute to Object3d and derived classes which returns number of navigation
  dimensions.
- Setting the symmetry of a (Mis)Orientation via a `symmetry.setter`.
- Projection of vectors into the fundamental sector (inverse pole figure) of a symmetry.
- Plotting of orientations within an inverse pole figure given a Laue symmetry and
  sample direction.
- Inverse pole figure colouring of orientations given a Laue symmetry and sample
  direction.
- `from_axes_angles()` method to `Rotation` and `Orientation` as a shortcut to
  `from_neo_euler()` for axis/angle pairs.
- `Orientation` based classes now accept a `symmetry` argument upon initialisation.
- Euler angle colouring of orientations given a proper point group symmetry.
- Simple unit cell orientation plotting with `plot_unit_cell` for `Orientation`
  instances.

Changed
-------
- `StereographicPlot` doesn't use Matplotlib's `transforms` framework anymore, and
  (X, Y) replaces (azimuth, polar) as internal coordinates.
- Renamed `Symmetry` method `fundamental_sector()` to `fundamental_zone()`.
- `Orientation` class methods `from_euler`, `from_matrix`, and `from_neo_euler` no longer 
  return the smallest angle orientation when a `symmetry` is given.
- `CrystalMap.orientations` no longer returns smallest angle orientation.
- The methods `flatten`, `reshape`, and `squeeze` have been overridden in
  `Misorientation` based classes to maintain the initial symmetry of the returned
  instance.
- `Rotation.to_euler()` returns angles in the ranges (0, 2 pi), (0, pi), and (0, 2 pi).
- `CrystalMap.get_map_data()` doesn't round values by default anymore. Passing
  `decimals=3` retains the old behaviour.
- `CrystalMap.plot()` doesn't override the Matplotlib status bar by default anymore.
  Passing `override_status_bar=True` retains the old behaviour.

Deprecated
----------
- The `data_dim` attribute of Object3d and all derived classes is deprecated from 0.8
  and will be removed in 0.9. Use `ndim` instead.
- Setting (Mis)Orientation symmetry via `set_symmetry()` is deprecated in 0.8, in favour of
  setting it directly via a `symmetry.setter`, and will be removed in 0.9. Use
  `map_into_symmetry_reduced_zone()` instead.
 
Removed
-------
- `StereographicPlot` methods `azimuth_grid()` and `polar_grid()`.
  Use `stereographic_grid()` instead.
- `from_euler()` no longer accepts "Krakow_Hielscher" as a convention, use "MTEX" instead.

Fixed
-----

- `CrystalMap.get_map_data()` can return an array of shape (3,) if there are that many
  points in the map.
- Reading of point groups with "-" sign, like -43m, from EMsoft h5ebsd files.


2021-09-07 - version 0.7.0
==========================

Added
-----
- Memory-efficient calculation of a misorientation angle (geodesic distance) matrix
  between orientations using Dask.
- Symmetry reduced calculations of dot products between orientations.
- Two notebooks on clustering of orientations (not misorientations) across fundamental
  region boundaries are added to the user guide from the orix-demos repository.
- Convenience method `Misorientation.scatter()` (and subclasses) to plot orientations in
  either axis-angle or Rodrigues fundamental zone.
- Method `Object3d.get_random_sample()`, inherited by all 3D objects, returning a new
  flattened instance with elements drawn randomly from the original instance.
- Add `transpose()` method to all 3D classes to transpose navigation dimensions.
- Reading of a `CrystalMap` from orientation data in Bruker's HDF5 file format.
- Uniform sampling of orientation space using cubochoric sampling.

Changed
-------
- `to_euler()` changed internally, "Krakow_Hielscher" deprecated, use "MTEX" instead.
- Default orientation space sampling method from "haar_euler" to "cubochoric".

2021-05-23 - version 0.6.0
==========================

Added
-----
- Python 3.9 support.
- User guide with Jupyter notebooks as part of the Read the Docs documentation.
- `CrystalMap.plot()` method for easy plotting of phases, properties etc.
- .ang file writer for CrystalMap objects (via `orix.io.save()`).
- `Miller` class, inheriting functionality from the `Vector3d` class, to handle
  operations with direct lattice vectors (uvw/UVTW) and reciprocal lattice vectors
  (hkl/hkil).
- `Vector3d.scatter()` and `Vector3d.draw_circle()` methods to show unit vectors and
  great/small circles in stereographic projection.
- Stereographic plot using Matplotlib's `transforms` framework for plotting vectors,
  great/small circles, and symmetry elements.
- `projections` module for projecting vectors to various coordinates, including
  stereographic coordinates.
- `CrystalMap.empty()` class method to create empty map of a given shape with identity
  rotations.
- Sampling of *SO(3)* now provided via two methods (up from the one in previous
  versions).
- Warning when trying to create rotations from large Euler angles.
- Passing symmetry when initializing an `Orientation`.
- Overloaded division for `Vector3d` (left hand side) by numbers and suitably shaped
  array-like objects.

Changed
-------
- Names of spherical coordinates for the `Vector3d` class, `phi` to `azimuth`, `theta`
  to `polar`, and `r` to `radial`. Similar changes to to/from_polar parameter names.
- `CrystalMap.get_map_data()` tries to respect input data type, other minor
  improvements.
- Continuous integration migrated from Travis CI to GitHub Actions.

Fixed
-----
- Symmetry is preserved when creating a misorientation from orientations or when
  inverting orientations.
- Reading of properties (scores etc.) from EMsoft h5ebsd files with certain map shapes.
- Reading of crystal symmetry from EMsoft h5ebsd dot product files in CrystalMap plugin.

2020-11-03 - version 0.5.1
==========================

Added
-----
- This project now keeps a Changelog.
- Testing for Python 3.8 on macOS.

Fixed
-----
- `CrystalMap` properties allow arrays with number of dimensions greater than 2.
- .ang file reader now recognises phase IDs defined in the header.
- EMsoft file reader reads unrefined Euler angles correctly.
