=========
Changelog
=========

All user facing changes to this project are documented in this file. The format is based
on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`__, and this project tries
its best to adhere to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`__.

2024-09-20 - version 0.13.1
===========================

Added
-----
- Support for Python 3.12.

Changed
-------
- numpy-quaternion is now an optional dependency and will not be installed with ``pip``
  unless ``pip install orix[all]`` is used.

Removed
-------
- Support for Python 3.8 and 3.9.

Fixed
-----
- ``Phase.from_cif()`` still gives a valid phase even though the space group could not
  be read.

2024-09-03 - version 0.13.0
===========================

Added
-----
- We can now read 2D crystal maps from Channel Text Files (CTFs) using ``io.load()``.

Changed
-------
- Phase names in crystal maps read from .ang files with ``io.load()`` now prefer to use
  the abbreviated "Formula" instead of "MaterialName" in the file header.

Removed
-------
- Removed deprecated ``from_neo_euler()`` method for ``Quaternion`` and its subclasses.
- Removed deprecated argument ``convention`` in ``from_euler()`` and ``to_euler()``
  methods for ``Quaternion`` and its subclasses. Use ``direction`` instead. Passing
  ``convention`` will now raise an error.

Deprecated
----------
- ``loadang()`` and ``loadctf()`` are deprecated and will be removed in the next minor
  release. Please use ``io.load()`` instead.

2024-04-21 - version 0.12.1
===========================

Fixed
-----
- ``ax2qu`` and ``Quaternion.from_axes_angles()`` would raise if the input arrays were
  broadcastable but the final dimension was ``1``. This has been fixed.
- ``Phase.from_cif()`` now correctly adjusts atom positions when forcing
  ``Phase.structure.lattice.base`` to use the crystal axes alignment ``e1 || a``,
  ``e3 || c*``. This bug was introduced in 0.12.0.

2024-04-13 - version 0.12.0
===========================

Added
-----
- ``Vector3d.from_path_ends()`` class method to get vectors between two vectors.
- Convenience function ``plot.format_labels()`` to get nicely formatted vector labels to
  use when plotting vectors.
- Two offsets in the stereographic coordinates (X, Y) can be given to
  ``StereographicPlot.text()`` to offset text coordinates.
- Explicit support for Python 3.11.
- Creating quaternions from neo-eulerian vectors via new class methods
  ``from_rodrigues()`` and ``from_homochoric()``, replacing the now deprecated
  ``from_neo_euler()``. ``from_rodrigues()`` accepts an angle parameter to allow passing
  Rodrigues-Frank vectors.
- Creating neo-eulerian vectors from quaternions via new methods ``to_axes_angles()``,
  ``to_rodrigues()`` and ``to_homochoric()``. Rodrigues-Frank vectors can be returned
  from ``to_rodrigues()`` by passing ``frank=True``.
- ``inv()`` method for ``Quaternion``, ``Rotation``, ``Orientation``, and
  ``Misorientation``. For the three first, its behavior is identical to the inversion
  operator ``~``. For misorientations, it inverts the direction of the transformation.
  Convenient for chaining operations.
- The ``random()`` methods of ``Orientation`` and ``Misorientation`` now accept
  ``symmetry``. A ``random()`` method is also added to ``Vector3d`` and ``Miller``, the
  latter accepting a ``phase``.
- Function ``orix.sampling.get_sample_reduced_fundamental()`` for sampling rotations
  that rotate the Z-vector (0, 0, 1) onto the fundamental sector of the Laue group of a
  given ``Symmetry``.

Changed
-------
- The ``convention`` parameter in ``from_euler()`` and ``to_euler()`` will be removed in
  the next minor release, 0.13, instead of release 1.0 as previously stated.
- Allow passing a tuple of integers to ``reshape()`` methods of 3D objects.
- ``random()`` methods no longer accept a list as a valid shape: pass a tuple instead.
- Increase minimal version of Matplotlib to >= 3.5.

Removed
-------
- Support for Python 3.7.

Deprecated
----------
- Creating quaternions from neo-eulerian vectors via ``from_neo_euler()`` is deprecated
  and will be removed in v0.13. Use the existing ``from_axes_angles()`` and the new
  ``from_rodrigues()`` and ``from_homochoric()`` instead.

Fixed
-----
- Transparency of polar stereographic grid lines can now be controlled by Matplotlib's
  ``grid.alpha``, just like the azimuth grid lines.
- Previously, ``Phase`` did not adjust atom positions when forcing
  ``Phase.structure.lattice.base`` to use the crystal axes alignment ``e1 || a``,
  ``e3 || c*``. This is now fixed.

2023-03-14 - version 0.11.1
===========================

Fixed
-----
- Initialization of a crystal map with a phase list with fewer phases than in the phase
  ID array given returns a map with a new phase list with correct phase IDs.

2023-02-09 - version 0.11.0
===========================

Added
-----
- Creation of one or more ``Quaternion`` (or instances of inheriting classes) from one
  or more SciPy ``Rotation``.
- Creation of one ``Quaternion`` or ``Rotation`` by aligning sets of vectors in two
  reference frames, one ``Orientation`` by aligning sets of sample vectors and crystal
  vectors, and one ``Misorientation`` by aligning two sets of crystal vectors in two
  different crystals.
- ``row`` and ``col`` properties to ``CrystalMap`` giving the row and column coordinate
  of each map point given by ``CrystalMap.shape``.
- ``Rotation`` class methods ``from_neo_euler()``, ``from_axes_angles()``,
  ``from_euler()``, ``from_matrix()``, ``random()`` and ``identity()`` and methods
  ``to_euler()`` and ``to_matrix()`` are now available from the ``Quaternion`` class as
  well.
- ``StereographicPlot.restrict_to_sector()`` allows two new parameters to control the
  amount of padding (in degrees in stereographic projection) and whether to show the
  sector edges. Keyword arguments can also be passed on to Matplotlib's ``PathPatch()``.
- Option to pass degrees to the ``Quaternion`` methods ``from_axes_angles()``,
  ``from_euler()`` and ``to_euler()`` by passing ``degrees=True``.
- Option to get degrees from all ``angle_with()`` and ``angle_with_outer()`` methods
  by passing ``degrees=True``.
- Option to pass degrees to the ``(Mis)Orientation`` method ``get_distance_matrix()``
  by passing ``degrees=True``.
- Option to pass degrees to the ``Vector3d`` methods ``from_polar()`` and ``to_polar()``
  by passing ``degrees=True``.
- Option to get spherical coordinates from
  ``InverseStereographicProjection.xy2spherical()`` in degrees or pass them as degrees
  to ``StereographicProjection`` methods ``spherical2xy()`` and ``spherical2xy_split()``
  by passing ``degrees=True``.


Changed
-------
- Bumped minimal version of ``diffpy.structure >= 3.0.2``.
- Only ASTAR .ang files return crystal maps with ``"nm"`` as scan unit.

Removed
-------
- Parameter ``z`` when creating a ``CrystalMap`` and the ``z`` and ``dz`` attributes of
  the class were deprecated in 0.10.1 and are now removed.
- Passing ``shape`` or ``step_sizes`` with three values to
  ``create_coordinate_arrays()`` was depreacted in 0.10. and will now raise an error.
- Parameter ``depth`` (and ``axes``) in ``CrystalMapPlot.plot_map()`` was depreacted in
  0.10.1 and will now raise an error if passed.
- The ``z`` and ``dz`` datasets are not present in new orix HDF5 files. They are not
  read if present in older files.

Fixed
-----
- Reading of EDAX TSL .ang files with ten columns should now work.

2022-10-25 - version 0.10.2
===========================

Fixed
-----
- ``Miller.symmetrise(unique=True)`` returns the correct number of symmetrically
  equivalent but unique vectors, by rounding to 10 instead of 12 decimals prior to
  finding the unique vectors with NumPy.

Changed
-------
- Unique rotations and vectors are now found by rounding to 10 instead of 12 decimals.

2022-10-03 - version 0.10.1
===========================

Deprecated
----------
- Parameter ``z`` when creating a ``CrystalMap`` and the ``z`` and ``dz`` attributes of
  the class are deprecated and will be removed in 0.11.0. Support for 3D crystal maps is
  minimal and brittle, and it was therefore decided to remove it altogether.
- Passing ``shape`` or ``step_sizes`` with three values to ``create_coordinate_arrays()``
  is depreacted and will raise an error in 0.11.0. See the previous point for the reason.
- Parameter ``depth`` in ``CrystalMapPlot.plot_map()`` is depreacted and will be removed
  in 0.11.0. See the top point for the reason.

Fixed
-----
- ``StereographicPlot.scatter()`` now accepts both ``c``/``color`` and ``s``/``sizes``
  to set the color and sizes of scatter points, in line with
  ``matplotlib.axes.Axes.scatter()``.
- Indexing/slicing into an already indexed/sliced ``CrystalMap`` now correctly returns
  the index/slice according to ``CrystalMap.shape`` and not the original shape of the
  un-sliced map.

2022-09-22 - version 0.10.0
===========================

Added
-----
- Support for type hints has been introduced and a section on this topic has been added
  to the contributing guide.
- ``Vector3d.pole_density_function()`` has been implemented which allows for calculation
  of the Pole Density Function (PDF) and quantification of poles in the stereographic
  projection.
- Seven methods for sampling unit vectors from regular grids on *S2* via
  ``orix.sampling.sample_S2()``.
- Calculation of the Inverse Pole Density Function (IPDF), ie. pole density in the
  crystal point group fundamental sector, through 
  ``InversePoleFigurePlot.pole_density_function()``.
- The ``orix.measure`` module has been introduced. The ``measure`` module is related to
  quantification of orientation and vector data.
- Plotting the IPF color key on a created ``InversePoleFigurePlot`` is now possible with
  ``plot_ipf_color_key()``.
- Examples gallery to documentation.

Changed
-------
- Moved part of documentation showing plotting of Wulff net and symmetry markers from
  the tutorials to examples.
- Renamed user guide notebooks to tutorials in documentation.
- Reference frame labels of stereographic projection of ``Symmetry.plot()`` from (a, b)
  to (e1, e2), signifying the standard Cartesian reference frame attached to a crystal.
- Tighten distribution of random orientation clusters in tutorial showing clustering
  across fundamental region boundaries, to avoid clustering sometimes giving two
  clusters instead of three.

Removed
-------
- Support for Python 3.6 has been removed. The minimum supported version in ``orix`` is
  now Python 3.7.
- ``Object3d.check()``, ``Quaternion.check_quaternion()`` and
  ``Vector3d.check_vector()``, as these methods were not used internally.
- Deprecated method ``distance()`` of ``Misorientation`` and ``Orientation`` classes,
  use ``get_distance_matrix()`` instead.

Fixed
-----
- Plotting of unit cells works with Matplotlib v3.6, at the expense of a warning raised
  with earlier versions.

2022-05-16 - version 0.9.0
==========================

Added
-----
- Dask computation of ``Quaternion`` and ``Rotation`` ``outer()`` methods through
  addition of a ``lazy`` parameter. This is useful to reduce memory usage when working
  with large arrays.
- Dask implementation of the ``Quaternion`` - ``Vector3d`` outer product.
- Point group ``Symmetry`` elements can now be viewed in the stereographic projection
  using ``Symmetry.plot()``. The notebook point_groups.ipynb has been added to the
  documentation.
- Add ``reproject`` argument to ``Vector3d.scatter()`` which reprojects vectors located
  on the hidden hemisphere to the visible hemisphere.
- ``reproject`` argument to ``Vector3d.draw_circle()`` which reprojects parts of
  circle(s) on the other hemisphere to the current hemisphere.
- ``Rotation`` objects can now be checked for equality. Equality is determined by
  comparing their shape, data, and whether the rotations are improper.
- ``angle_with_outer()`` has been added to both  ``Rotation`` and ``Orientation``
  classes which computes the misorientation angle between every ``Rotation`` in the two
  sets of rotations. In the case of ``Orientation.angle_with_outer()``, this is the
  symmetry reduced misorientation.
- Notebook on clustering of misorientations across fundamental region boundaries moved
  from the orix-demos repository to the user guide.
- ``orix.data`` module with test data used in the user guide and tests.
- ``Misorientation.get_distance_matrix()`` for memory-efficient calculation of a
  misorientation angle (geodesic distance) matrix between misorientations using Dask.
- Clarification of crystal axes alignment in documentation.
- Creation of a ``Phase`` instance from a CIF file.

Changed
-------
- ``from_euler()`` method of ``Rotation``-based classes now interprets angles in Bunge
  convention by default, ie. ``direction="lab2crystal"``. The returned ``Rotation`` from
  this function may be inverted from prior releases and users are advised to check their
  code.
- The ``direction`` parameter in ``from_euler()`` methods, in addition to
  ``"lab2crystal"`` (now default) and ``"crystal2lab"``, now also accepts a convenience
  argument ``"mtex"`` which is consistent with the ``"crystal2lab"`` direction in
  `MTEX <https://mtex-toolbox.github.io/MTEXvsBungeConvention.html>`_.
- ``S4`` (-4) ``Symmetry`` has been corrected.
- Organized user guide documentation into topics.

Deprecated
----------
- The ``convention`` parameter in ``from_euler()`` and ``to_euler()`` methods has been
  deprecated, in favour of ``direction`` in the former. This parameter will be removed
  in release 1.0.
- ``Misorientation.distance()`` in favour of ``Misorientation.get_distance_matrix()``.

Fixed
-----
- Fixed bug in ``sample_S2_uv_mesh()`` and removed duplicate vectors at poles.
- The results from ``Orientation.dot_outer()`` are now returned as
  ``self.shape + other.shape``, which is consistent with ``Rotation.dot_outer()``.
- Writing of property arrays in .ang writer from masked CrystalMap.

Removed
-------
- ``orix.scalar.Scalar`` class has been removed and the data held by ``Scalar`` is now
  returned directly as a ``numpy.ndarray``.
- The deprecation of function ``(Mis)Orientation.set_symmetry()`` and property
  ``Object3d.data_dim`` has expired and have been removed.

2022-02-21 - version 0.8.2
==========================

Changed
-------
- ``orix.quaternion.Quaternion`` now relies on `numpy-quaternion
  <https://quaternion.readthedocs.io/en/latest/>`_ for quaternion conjugation,
  quaternion-quaternion and quaternion-vector multiplication, and quaternion-quaternion
  and quaternion-vector outer products.
- Rounding in functions, e.g. ``Object3d.unique()`` and ``Rotation.unique()``, is now
  set consistently at 12 dp.

Fixed
-----
- ``Miller.in_fundamental_sector()`` doesn't raise errors.
- ``Miller.unique()`` now correctly returns unique vectors due to implemented rounding.

2022-02-14 - version 0.8.1
==========================

Added
-----
- Python 3.10 support.
- Option to pass figure initialization keyword arguments to Matplotlib via plotting
  methods.

Fixed
-----
- ``Orientation`` disorientation angles and dot products returned from ``angle_with()``
  and ``dot()`` and ``dot_outer()``, which now calculates the misorientation as
  ``other * ~self``. Disorientation angles ``(o2 - o1).angle`` and ``o1.angle_with(o2)``
  are now the same.
- The inverse indices returned from ``Rotation.unique()`` now correctly recreate the
  original ``Rotation`` instance.
- Handling of property arrays in .ang writer with multiple values per map point.
- ``CrystalMap``'s handling of a mask of which points are in the data.

2021-12-21 - version 0.8.0
==========================

Added
-----
- ``FundamentalSector`` class of vector normals describing a fundamental sector in the
  stereographic projection, typically the inverse pole figure of a ``Symmetry``.
- ``Symmetry.fundamental_sector`` attribute with a ``FundamentalSector`` for that
  symmetry.
- ``StereographicPlot.restrict_to_sector()`` to restrict the stereographic projection to
  a sector, typically the inverse pole figure of a ``Symmetry``.
- ``StereographicPlot.stereographic_grid()`` to control the azimuth and polar grid
  lines.
- Sampling of vectors in UV mesh on a unit sphere (*S2*).
- ``ndim`` attribute to Object3d and derived classes which returns number of navigation
  dimensions.
- Setting the symmetry of a (Mis)Orientation via a ``symmetry.setter``.
- Projection of vectors into the fundamental sector (inverse pole figure) of a symmetry.
- Plotting of orientations within an inverse pole figure given a Laue symmetry and
  sample direction.
- Inverse pole figure colouring of orientations given a Laue symmetry and sample
  direction.
- ``from_axes_angles()`` method to ``Rotation`` and ``Orientation`` as a shortcut to
  ``from_neo_euler()`` for axis/angle pairs.
- ``Orientation`` based classes now accept a ``symmetry`` argument upon initialisation.
- Euler angle colouring of orientations given a proper point group symmetry.
- Simple unit cell orientation plotting with ``plot_unit_cell`` for ``Orientation``
  instances.

Changed
-------
- ``StereographicPlot`` doesn't use Matplotlib's ``transforms`` framework anymore, and
  (X, Y) replaces (azimuth, polar) as internal coordinates.
- Renamed ``Symmetry`` method ``fundamental_sector()`` to ``fundamental_zone()``.
- ``Orientation`` class methods ``from_euler``, ``from_matrix``, and ``from_neo_euler``
  no longer  return the smallest angle orientation when a ``symmetry`` is given.
- ``CrystalMap.orientations`` no longer returns smallest angle orientation.
- The methods ``flatten``, ``reshape``, and ``squeeze`` have been overridden in
  ``Misorientation`` based classes to maintain the initial symmetry of the returned
  instance.
- ``Rotation.to_euler()`` returns angles in the ranges (0, 2 pi), (0, pi), and
  (0, 2 pi).
- ``CrystalMap.get_map_data()`` doesn't round values by default anymore. Passing
  ``decimals=3`` retains the old behaviour.
- ``CrystalMap.plot()`` doesn't override the Matplotlib status bar by default anymore.
  Passing ``override_status_bar=True`` retains the old behaviour.

Deprecated
----------
- The ``data_dim`` attribute of Object3d and all derived classes is deprecated from 0.8
  and will be removed in 0.9. Use ``ndim`` instead.
- Setting (Mis)Orientation symmetry via ``set_symmetry()`` is deprecated in 0.8, in
  favour of setting it directly via a ``symmetry.setter``, and will be removed in 0.9.
  Use ``map_into_symmetry_reduced_zone()`` instead.
 
Removed
-------
- ``StereographicPlot`` methods ``azimuth_grid()`` and ``polar_grid()``.
  Use ``stereographic_grid()`` instead.
- ``from_euler()`` no longer accepts ``"Krakow_Hielscher"`` as a convention, use
  ``"MTEX"`` instead.

Fixed
-----
- ``CrystalMap.get_map_data()`` can return an array of shape (3,) if there are that many
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
- Convenience method ``Misorientation.scatter()`` (and subclasses) to plot orientations
  in either axis-angle or Rodrigues fundamental zone.
- Method ``Object3d.get_random_sample()``, inherited by all 3D objects, returning a new
  flattened instance with elements drawn randomly from the original instance.
- Add ``transpose()`` method to all 3D classes to transpose navigation dimensions.
- Reading of a ``CrystalMap`` from orientation data in Bruker's HDF5 file format.
- Uniform sampling of orientation space using cubochoric sampling.

Changed
-------
- ``to_euler()`` changed internally, "Krakow_Hielscher" deprecated, use "MTEX" instead.
- Default orientation space sampling method from "haar_euler" to "cubochoric".

2021-05-23 - version 0.6.0
==========================

Added
-----
- Python 3.9 support.
- User guide with Jupyter notebooks as part of the Read the Docs documentation.
- ``CrystalMap.plot()`` method for easy plotting of phases, properties etc.
- .ang file writer for CrystalMap objects (via ``orix.io.save()``).
- ``Miller`` class, inheriting functionality from the ``Vector3d`` class, to handle
  operations with direct lattice vectors (uvw/UVTW) and reciprocal lattice vectors
  (hkl/hkil).
- ``Vector3d.scatter()`` and ``Vector3d.draw_circle()`` methods to show unit vectors and
  great/small circles in stereographic projection.
- Stereographic plot using Matplotlib's ``transforms`` framework for plotting vectors,
  great/small circles, and symmetry elements.
- ``projections`` module for projecting vectors to various coordinates, including
  stereographic coordinates.
- ``CrystalMap.empty()`` class method to create empty map of a given shape with identity
  rotations.
- Sampling of *SO(3)* now provided via two methods (up from the one in previous
  versions).
- Warning when trying to create rotations from large Euler angles.
- Passing symmetry when initializing an ``Orientation``.
- Overloaded division for ``Vector3d`` (left hand side) by numbers and suitably shaped
  array-like objects.

Changed
-------
- Names of spherical coordinates for the ``Vector3d`` class, ``phi`` to ``azimuth``,
  ``theta`` to ``polar``, and ``r`` to ``radial``. Similar changes to to/from_polar
  parameter names.
- ``CrystalMap.get_map_data()`` tries to respect input data type, other minor
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
- ``CrystalMap`` properties allow arrays with number of dimensions greater than 2.
- .ang file reader now recognises phase IDs defined in the header.
- EMsoft file reader reads unrefined Euler angles correctly.
