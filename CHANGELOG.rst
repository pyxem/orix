=========
Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_, and
this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Unreleased
==========


2021-09-07 - version 0.7.0
==========================

Added
-----
- Memory-efficient calculation of a misorientation angle (geodesic distance) matrix
  between orientations using Dask.
- Symmetry reduced calculations of dot products between orientations.
- Two notebooks on clustering of orientations (not misorientations) across fundamental
  region boundaries are added to the user guide from the orix-demos repository.
- Convenience method Misorientation.scatter() (and subclasses) to plot orientations in either axis-angle or
  Rodrigues fundamental zone, without having to set it up via Matplotlib.
- Method Object3d.get_random_sample(), inherited by all 3D objects, returning a new
  flattened instance with elements drawn randomly from the original instance.
- Add transpose method to all 3D classes to transpose navigation dimensions.
- Reading of a crystal map from orientation data in Bruker's HDF5 file format.
- Uniform sampling of orientation space using cubochoric sampling.

Changed
-------
- to_euler() changed internally, "Krakow_Hielscher" deprecated, use "MTEX" instead.
- Default orientation space sampling method from "haar_euler" to "cubochoric".

2021-05-23 - version 0.6.0
==========================

Added
-----
- Python 3.9 support.
- User guide with Jupyter notebooks as part of the Read the Docs documentation
- CrystalMap.plot() method for easy plotting of phases, properties etc.
- .ang file writer for CrystalMap objects (via orix.io.save())
- Miller class, inheriting functionality from the Vector3d class, to handle operations with direct lattice vectors (uvw/UVTW) and reciprocal lattice vectors (hkl/hkil).
- Vector3d.scatter() and Vector3d.draw_circle() methods to show unit vectors and
  great/small circles in stereographic projection
- Stereographic projection using Matplotlib's projections framework for plotting
  vectors, great/small circles, and symmetry elements
- orix.projections module for projecting vectors to various coordinates, including
  stereographic coordinates
- CrystalMap.empty() class method to create empty map of a given shape with identity
  rotations.
- sampling of SO3 now provided via two methods (up from the one in previous versions)
- Warning when trying to create rotations from large Euler angles
- Passing symmetry when initializing an Orientation.
- Overloaded division for Vector3d (left hand side) by numbers and suitably shaped
  array-like objects

Changed
-------
- Names of spherical coordinates for the Vector3d class, "phi" to "azimuth", "theta" to
  "polar", and "r" to "radial". Similar changes to to/from_polar parameter names.
- CrystalMap.get_map_data() tries to respect input data type, other minor improvements
- Continuous integration migrated from Travis CI to GitHub Actions

Fixed
-----
- Symmetry is preserved when creating a misorientation from orientations or when
  inverting orientations
- Reading of properties (scores etc.) from EMsoft h5ebsd files with certain map shapes
- Reading of crystal symmetry from EMsoft h5ebsd dot product files in CrystalMap plugin

2020-11-03 - version 0.5.1
==========================

Added
-----
- This project now keeps a Changelog
- Testing for Py3.8 on OSX

Fixed
-----
- CrystalMap properties allow arrays with number of dimensions greater than 2
- .ang file reader now recognises phase IDs defined in the header
- EMsoft file reader reads unrefined Euler angles correctly
