=========
Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_, and
this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Unreleased
==========

Added
-----
- Miller class, inherinting functionality from the Vector3d class, to handle operations
  with direct lattice vectors (uvw/UVTW) and reciprocal lattice vectors (hkl/hkil).
- Vector3d.scatter() and Vector3d.draw_circle() methods to show unit vectors and
  great/small circles in stereographic projection
- User guide with Jupyter notebooks as part of the Read the Docs documentation
- Stereographic projection using Matplotlib's projections framework for plotting
  vectors, great/small circles, and symmetry elements
- orix.projections module for projecting vectors to various coordinates, including
  stereographic coordinates
- .ang file writer for CrystalMap objects (via orix.io.save())
- Overloaded division for Vector3d (left hand side) by numbers and suitably shaped
  array-like objects

Changed
-------
- Names of spherical coordinates for the Vector3d class, "phi" to "azimuth", "theta" to
  "polar", and "r" to "radial". Similar changes to to/from_polar parameter names.
- CrystalMap.get_map_data() tries to respect input data type, other minor improvements
- Continuous integration migrated from Travis CI to GitHub Actions

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
