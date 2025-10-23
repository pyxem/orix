#
# Copyright 2018-2025 the orix developers
#
# This file is part of orix.
#
# orix is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# orix is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with orix. If not, see <http://www.gnu.org/licenses/>.
#

from __future__ import annotations

import copy
from typing import Any

import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
import numpy as np

from orix.crystal_map._phase import Phase
from orix.crystal_map._phase_list import PhaseList
from orix.crystal_map.crystal_map_properties import CrystalMapProperties
from orix.plot._util.color import get_named_matplotlib_colors
from orix.quaternion.orientation import Orientation
from orix.quaternion.rotation import Rotation


class CrystalMap:
    """Crystallographic map of orientations, crystal phases and key
    properties associated with every spatial coordinate in a 1D or 2D.

    Parameters
    ----------
    rotations
        Rotation in each point. Must be passed with all spatial
        dimensions in the first array axis (flattened). May contain
        multiple rotations per point, included in the second array
        axes. Crystal map data size is set equal to the first array
        axis' size.
    phase_id
        Phase ID of each pixel. IDs equal to ``-1`` are considered not
        indexed. If not given, all points are considered to belong to
        one phase with ID ``0``.
    x
        Map x coordinate of each data point. If not given, the map is
        assumed to be 1D, and it is set to an array of increasing
        integers from 0 to the length of the ``phase_id`` array.
    y
        Map y coordinate of each data point. If not given, the map is
        assumed to be 1D, and it is set to ``None``.
    z
        Map z coordinate of each data point. If not given, the map is
        assumed to be 1D, and it is set to ``None``.
    phase_list
        A list of phases in the data with their with names, space
        groups, point groups, and structures. The order in which the
        phases appear in the list is important, as it is this, and not
        the phases' IDs, that is used to link the phases to the
        input ``phase_id`` if the IDs aren't exactly the same as in
        ``phase_id``. If not given, a phase list with as many phases as
        there are unique phase IDs in ``phase_id`` is created.
    prop
        Dictionary of properties of each data point.
    scan_unit
        Length unit of the data. Default is ``"px"``.
    is_in_data
        Array of booleans signifying whether a point is in the data.

    See Also
    --------
    create_coordinate_arrays
    :mod:`~orix.data`
    :func:`~orix.io.load`

    Notes
    -----
    Data is stored as 1D arrays and reshaped when necessary.

    Examples
    --------
    Constructing a crystal map from scratch, with two rows and three
    columns and containing Austenite and Ferrite orientations

    >>> from diffpy.structure import Atom, Lattice, Structure
    >>> from orix.crystal_map import create_coordinate_arrays, CrystalMap, PhaseList
    >>> from orix.quaternion import Rotation
    >>> coords, n = create_coordinate_arrays(shape=(2, 3))
    >>> structures = [
    ...     Structure(
    ...         title="austenite",
    ...         atoms=[Atom("fe", [0] * 3)],
    ...         lattice=Lattice(0.360, 0.360, 0.360, 90, 90, 90)
    ...     ),
    ...     Structure(
    ...         title="ferrite",
    ...         atoms=[Atom("fe", [0] * 3)],
    ...         lattice=Lattice(0.287, 0.287, 0.287, 90, 90, 90)
    ...     )
    ... ]
    >>> xmap = CrystalMap(
    ...     rotations=Rotation.from_axes_angles([0, 0, 1], np.linspace(0, np.pi, n)),
    ...     phase_id=np.array([0, 0, 1, 1, 0, 1]),
    ...     x=coords["x"],
    ...     y=coords["y"],
    ...     phase_list=PhaseList(space_groups=[225, 229], structures=structures),
    ...     prop={"score": np.random.random(n)},
    ...     scan_unit="nm",
    ... )
    >>> xmap
    Phase  Orientations       Name  Space group  Point group  Proper point group       Color
        0     3 (50.0%)  austenite        Fm-3m         m-3m                 432    tab:blue
        1     3 (50.0%)    ferrite        Im-3m         m-3m                 432  tab:orange
    Properties: score
    Scan unit: nm

    Data in a crystal map can be selected in multiple ways. Let's
    demonstrate this on a dual phase dataset available in the
    :mod:`~orix.data` module

    >>> from orix import data
    >>> xmap = data.sdss_ferrite_austenite(allow_download=True)
    >>> xmap
    Phase   Orientations       Name  Space group  Point group  Proper point group       Color
        1   5657 (48.4%)  austenite         None          432                 432    tab:blue
        2   6043 (51.6%)    ferrite         None          432                 432  tab:orange
    Properties: iq, dp
    Scan unit: um
    >>> xmap.shape
    (100, 117)

    Selecting based on coordinates, passing ranges (slices), integers or
    both

    >>> xmap2 = xmap[20:40, 50:60]
    >>> xmap2
    Phase  Orientations       Name  Space group  Point group  Proper point group       Color
        1   148 (74.0%)  austenite         None          432                 432    tab:blue
        2    52 (26.0%)    ferrite         None          432                 432  tab:orange
    Properties: iq, dp
    Scan unit: um
    >>> xmap2.shape
    (20, 10)
    >>> xmap2 = xmap[20:40, 3]
    >>> xmap2
    Phase  Orientations       Name  Space group  Point group  Proper point group       Color
        1    16 (80.0%)  austenite         None          432                 432    tab:blue
        2     4 (20.0%)    ferrite         None          432                 432  tab:orange
    Properties: iq, dp
    Scan unit: um
    >>> xmap2.shape
    (20, 1)

    Note that 1-dimensions are NOT removed

    >>> xmap2 = xmap[10, 10]
    >>> xmap2
    Phase  Orientations     Name  Space group  Point group  Proper point group       Color
        2    1 (100.0%)  ferrite         None          432                 432  tab:orange
    Properties: iq, dp
    Scan unit: um
    >>> xmap2.shape
    (1, 1)

    Select by phase name(s)

    >>> xmap2 = xmap["austenite"]
    >>> xmap2
    Phase  Orientations       Name  Space group  Point group  Proper point group     Color
        1  5657 (100.0%)  austenite         None          432                 432  tab:blue
    Properties: iq, dp
    Scan unit: um
    >>> xmap2.shape
    (100, 117)
    >>> xmap["austenite", "ferrite"]
    Phase  Orientations       Name  Space group  Point group  Proper point group       Color
        1  5657 (48.4%)  austenite         None          432                 432    tab:blue
        2  6043 (51.6%)    ferrite         None          432                 432  tab:orange
    Properties: iq, dp
    Scan unit: um

    Select by indexed and not indexed data

    >>> xmap["indexed"]
    Phase  Orientations       Name  Space group  Point group  Proper point group       Color
        1  5657 (48.4%)  austenite         None          432                 432    tab:blue
        2  6043 (51.6%)    ferrite         None          432                 432  tab:orange
    Properties: iq, dp
    Scan unit: um
    >>> xmap["not_indexed"]
    No data.

    Select with a boolean array (possibly chained)

    >>> xmap[xmap.dp > 0.81]
    Phase  Orientations       Name  Space group  Point group  Proper point group       Color
        1  4092 (44.8%)  austenite         None          432                 432    tab:blue
        2  5035 (55.2%)    ferrite         None          432                 432  tab:orange
    Properties: iq, dp
    Scan unit: um
    >>> xmap[(xmap.iq > np.mean(xmap.iq)) & (xmap.phase_id == 1)]
    Phase  Orientations       Name  Space group  Point group  Proper point group     Color
        1  1890 (100.0%)  austenite         None          432                 432  tab:blue
    Properties: iq, dp
    Scan unit: um
    """

    def __init__(
        self,
        rotations: "Rotation | CrystalMap",
        phase_id: np.ndarray | None = None,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        z: np.ndarray | None = None,
        phase_list: PhaseList | None = None,
        prop: dict | None = None,
        scan_unit: str | None = "px",
        is_in_data: np.ndarray | None = None,
        indices: np.ndarray | None = None,
        spacing: np.ndarray | None = None,
        origin: np.ndarray | None = None,
        indexing_order: str = "zyx",
    ) -> None:
        if isinstance(rotations, CrystalMap):
            return CrystalMap.__init__(
                self,
                rotations.rotations,
                rotations.phase_id,
                rotations.x,
                rotations.y,
                rotations.z,
                rotations.phases,
                rotations.prop,
                rotations.scan_unit,
                rotations.is_in_data,
            )

        # Set data size and conventions
        if x is not None:
            data_size = x.size
        else:
            data_size = rotations.shape[0]
        if indexing_order in ["xyz", "zyx"]:
            self._indexing_order = indexing_order
        else:
            raise ValueError(
                f"indexing_oder must be 'xyz' or 'zyz'm not {indexing_order}"
            )

        # Set rotations
        if not isinstance(rotations, Rotation):
            raise ValueError(
                f"rotations must be of type {Rotation}, not {type(rotations)}."
            )
        if rotations.size == data_size:
            self._rotations = rotations.flatten()
        elif rotations.shape[0] == data_size:
            self._rotations = rotations
        else:
            raise ValueError(
                "'rotations' has a shape of {}. Either the ".format(rotations.shape)
                + "total size or the size of the first indicies of 'rotations' must"
                + "match the size of the CrystalMap, {}".format(data_size)
            )

        # Set phase IDs
        if phase_id is None:  # Assume single phase data
            phase_id = np.zeros(data_size)
        phase_id = phase_id.astype(int)
        self._phase_id = phase_id

        # Set data point IDs
        point_id = np.arange(data_size)
        self._id = point_id

        # Set spatial coordinates
        if indices is not None:
            self._set_grid_from_indices(indices, spacing, origin)
        else:
            self._set_grid_from_coords(x, y, z)
        if x is None and y is None and z is None:
            x = np.arange(data_size)  # TODO: I think this might have weird gap cases...
        self._x = x
        self._y = y
        self._z = z

        # Create phase list
        # Sorted in ascending order
        unique_phase_ids = np.unique(phase_id)
        include_not_indexed = False
        if unique_phase_ids[0] == -1:
            include_not_indexed = True
            unique_phase_ids = unique_phase_ids[1:]
        # Also sorted in ascending order
        if phase_list is None:
            self._phases = PhaseList(ids=unique_phase_ids)
        else:
            phase_list = phase_list.deepcopy()
            phase_ids = phase_list.ids
            n_different = len(phase_ids) - len(unique_phase_ids)
            if n_different > 0:
                # Remove superfluous phases by removing the phases whose
                # ID is not in the ID array, in descending list order
                for i in phase_ids[::-1]:
                    if i not in unique_phase_ids:
                        del phase_list[i]
                        n_different -= 1
                    if n_different == 0:
                        break
            elif n_different < 0:
                # Create new phase list adding the missing phases with
                # default initial values (but unique colors)
                phase_dict = {}
                all_colors, _ = get_named_matplotlib_colors()
                all_colors = list(all_colors.keys())
                all_unique_colors = np.delete(
                    all_colors, np.isin(all_colors, phase_list.colors)
                )
                ci = 0
                for i in unique_phase_ids:
                    if i in phase_ids:
                        phase_dict[i] = phase_list[i]
                    else:
                        phase_dict[i] = Phase(color=all_unique_colors[ci])
                        ci += 1
                phase_list = PhaseList(phase_dict)
            # Ensure phase list IDs correspond to IDs in phase_id array
            new_ids = list(unique_phase_ids.astype(int))
            phase_list._dict = dict(zip(new_ids, phase_list._dict.values()))
            self._phases = phase_list

        # Set whether measurements are indexed
        is_indexed = np.ones(data_size, dtype=bool)
        is_indexed[phase_id == -1] = False

        # Add "not_indexed" to phase list and ensure not indexed points
        # have correct phase ID
        if include_not_indexed:
            self.phases.add_not_indexed()
            self._phase_id[~is_indexed] = -1

        # Set array with True for points in data
        if is_in_data is None:
            is_in_data = np.ones(data_size, dtype=bool)
        self.is_in_data = is_in_data

        # Set scan unit
        self.scan_unit = scan_unit

        # Set properties
        if prop is None:
            prop = {}
        self._prop = CrystalMapProperties(prop, id=point_id)

        # Set original data shape (needed if data shape changes in
        # __getitem__())
        self._original_shape = self._data_shape_from_coordinates(only_is_in_data=False)

    def _set_grid_from_indices(
        self, indices: np.ndarray, spacing: np.ndarray, origin: np.ndarray
    ):
        """Sets the values for _layer, _row, _column, _dx, _dy, and _dz based on
        integer indices values"""
        indices = np.atleast_2d(indices)
        if len(indices.shape) != 2:
            ValueError("indices must be interpretable as a two-dimensional array")
        if not np.issubdtype(indices.dtype, np.integer):
            ValueError("indices must be an array of integers")
        dims = indices.shape[0]
        if not np.isin(dims, (1, 2, 3)):
            ValueError("indices must have a shape of (d, N), where 'd' is 1, 2, or 3")
        size = indices.shape[1]
        if size != self._rotations.shape[0]:
            ValueError("There must be the same number of indices as there are pixels")

        if spacing is None:
            spacing = np.ones(dims, dtype=np.float32)
        spacing = np.atleast_1d(spacing).flatten()
        if spacing.size != dims:
            ValueError(
                "Spacing should have {} values, not {}".format(dims, spacing.size)
            )

        if origin is None:
            spacing = np.zeros(dims, dtype=np.float32)
        spacing = np.atleast_1d(spacing).flatten()
        if spacing.size != dims:
            ValueError(
                "origin should have {} values, not {}".format(dims, spacing.size)
            )

        if self._indexing_order == "xyz":
            indices = indices[::-1, :]
            origin = origin[::-1]

        # Assign data AFTER all checks and calculations are completed.
        self._layer = indices[-2] if dims > 2 else None
        self._row = indices[-1] if dims > 1 else None
        self._column = indices[0]

        self._dz = spacing[-2] if dims > 2 else 0
        self._dy = spacing[-1] if dims > 1 else 0
        self._dx = spacing[0]

        self._zmin = origin[-2] if dims > 2 else 0
        self._ymin = origin[-1] if dims > 1 else 0
        self._xmin = origin[0]

        return

    def _set_grid_from_coords(self, x, y, z):
        """Sets the values for _layer, _row, _column, _dx, _dy, and _dz based on
        xyz spatial coordinates"""
        # reminder: Default numpy conventions imply zyx (layer/row/column) ordering

        if y is None and z is not None:
            ValueError("y cannot be None if z is not None")
        if x is None and y is not None:
            ValueError("x cannot be None if y is not None")
        if x is None and z is not None:
            ValueError("x cannot be None if z is not None")

        if z is None:
            dz = 0
            zmin = 0
            layer = None
        elif not np.issubdtype(z.dtype, np.number):
            ValueError("z must be interpretable as a 1d array of floats or ints")
        else:
            z = np.atleast_1d(z).flatten()
            dz = _step_size_from_coordinates(z)
            zmin = np.min(z)
            layer = np.around((z - zmin) / dz, 0).astype(int)

        if y is None:
            dy = 0
            ymin = 0
            row = None
        elif not np.issubdtype(y.dtype, np.number):
            ValueError("y must be interpretable as a 1d array of floats or ints")
        else:
            y = np.atleast_1d(y).flatten()
            dy = _step_size_from_coordinates(y)
            ymin = np.min(y)
            row = np.around((y - ymin) / dy, 0).astype(int)

        if x is None:
            x = np.arange(self._rotations.shape[0], dtype=int)
        elif not np.issubdtype(x.dtype, np.number):
            ValueError("x must be interpretable as a 1d array of floats or ints")
        x = np.atleast_1d(x).flatten()
        dx = _step_size_from_coordinates(x)
        xmin = np.min(x)
        column = np.around((x - xmin) / dx, 0).astype(int)

        # Assign data AFTER all checks and calculations are completed.
        self._layer = layer
        self._row = row
        self._column = column

        self._dz = dz
        self._dy = dy
        self._dx = dx

        self._zmin = zmin
        self._ymin = ymin
        self._xmin = xmin

        return

    @property
    def id(self) -> np.ndarray:
        """Return the ID of points in data."""
        return self._id[self.is_in_data]

    @property
    def size(self) -> int:
        """Return the total number of points in data."""
        return np.count_nonzero(self.is_in_data)

    @property
    def shape(self) -> tuple:
        """Return the shape of points in data."""
        nx = None if self.column is None else np.max(self.column) - np.min(self.column)
        ny = None if self.row is None else np.max(self.row) - np.min(self.row)
        nz = None if self.layer is None else np.max(self.layer) - np.min(self.layer)
        if self._indexing_order == "xyz":
            all_n = [nx, ny, nz]
        else:
            all_n = [nz, ny, nx]
        return tuple(int(n + 1) for n in all_n if n is not None)

    @property
    def ndim(self) -> int:
        """Return the number of data dimensions of points in data."""
        if self._column is None:
            return 0
        elif self._row is None:
            return 1
        elif self._layer is None:
            return 2
        else:
            return 3

    @property
    def x(self) -> np.ndarray | None:
        """Return the x coordinates of points in data."""
        if self._column is None:
            return
        else:
            return (self._column[self.is_in_data] * self._dx) + self._xmin

    @property
    def y(self) -> np.ndarray | None:
        """Return the y coordinates of points in data."""
        if self._row is None:
            return
        else:
            return (self._row[self.is_in_data] * self._dy) + self._ymin

    @property
    def z(self) -> np.ndarray | None:
        """Return the z coordinates of points in data."""
        if self._layer is None:
            return
        else:
            return (self._layer[self.is_in_data] * self._dz) + self._zmin

    @property
    def dx(self) -> float:
        """Return the x coordinate step size."""
        return self._dx

    @dx.setter
    def dx(self, dx: float | int):
        if self.column is None:
            ValueError("dx cannot be set when column is None")
        dx = np.asanyarray(dx).flatten()[0]
        if not np.isin(type(dx), np.number):
            ValueError("dx must be interpretable as an int or float")
        self._dx = dx

    @property
    def dy(self) -> float:
        """Return the y coordinate step size."""
        return self._dy

    @dy.setter
    def dy(self, dy: float | int):
        if self.row is None:
            ValueError("dy cannot be set when row is None")
        dy = np.asanyarray(dy).flatten()[0]
        if not np.isin(type(dy), np.number):
            ValueError("dy must be interpretable as an int or float")
        self._dy = dy

    @property
    def dz(self) -> float:
        """Return the z coordinate step size."""
        return self._dz

    @dz.setter
    def dz(self, dz: float | int):
        if self.layer is None:
            ValueError("dz cannot be set when row is None")
        dz = np.asanyarray(dz).flatten()[0]
        if not np.isin(type(dz), np.number):
            ValueError("dz must be interpretable as an int or float")
        self._dz = dz

    @property
    def column(self) -> np.ndarray | None:
        """Returns the column (x-axis) indice for each point in the CrystalMap.

        alias for 'col', for convenience"""
        return self._column

    @property
    def col(self) -> np.ndarray | None:
        """Returns the column (x-axis) indice for each point in the CrystalMap."""
        # TODO: re-add example
        return self._column

    @property
    def row(self) -> np.ndarray | None:
        """Returns the row (y-axis) indice for each point in the CrystalMap."""
        # TODO: re-add example
        return self._row

    @property
    def layer(self) -> np.ndarray | None:
        """Returns the layer (z-axis) indice for each point in the CrystalMap."""
        # TODO: re-add example
        return self._layer

    @property
    def phase_id(self) -> np.ndarray:
        """Return or set the phase IDs of points in data.

        Parameters
        ----------
        value : numpy.ndarray or int
            Phase ID of points in data.
        """
        return self._phase_id[self.is_in_data]

    @phase_id.setter
    def phase_id(self, value: np.ndarray | int) -> None:
        """Set phase ID of points in data."""
        self._phase_id[self.is_in_data] = value
        if value == -1 and "not_indexed" not in self.phases.names:
            self.phases.add_not_indexed()

    @property
    def phases(self) -> PhaseList:
        """Return or set the list of phases.

        Parameters
        ----------
        value : PhaseList
            Phase list with at least as many phases as unique phase IDs
            in :attr:`phase_id`.

        Raises
        ------
        ValueError
            If there are fewer phases in the list than unique phase IDs.
        """
        return self._phases

    @phases.setter
    def phases(self, value: PhaseList) -> None:
        """Set the list of phases."""
        if np.unique(self.phase_id).size > value.size:
            raise ValueError(
                "There must be at least as many phases as there are unique phase IDs"
            )
        else:
            self._phases = value

    @property
    def phases_in_data(self) -> PhaseList:
        """Return the list of phases in data.

        See Also
        --------
        phases

        Notes
        -----
        Can be useful when there are phases in :attr:`phases` which are
        not in the data.
        """
        unique_ids = np.unique(self.phase_id)
        phase_list = self.phases[np.intersect1d(unique_ids, self.phases.ids)]
        if isinstance(phase_list, Phase):  # One phase in data
            # Get phase ID so it carries over to the new `PhaseList`
            # instance
            phase = phase_list  # Since it's actually a single phase
            phase_id = self.phases.id_from_name(phase.name)
            return PhaseList(phases=phase, ids=phase_id)
        else:  # Multiple phases in data
            return phase_list

    @property
    def rotations(self) -> Rotation:
        """Return the rotations in the data."""
        return self._rotations[self.is_in_data]

    @property
    def rotations_per_point(self) -> int:
        """Return the number of rotations per data point in data."""
        return self.rotations.size // self.is_indexed.size

    @property
    def rotations_shape(self) -> tuple:
        """Return the shape of :attr:`rotations`.

        Notes
        -----
        Map shape and possible multiple rotations per point are
        accounted for. 1-dimensions are squeezed out.
        """
        return tuple(i for i in self.shape + (self.rotations_per_point,) if i != 1)

    @property
    def orientations(self) -> Orientation:
        """Return orientations (rotations respecting symmetry), in data.

        Raises
        ------
        ValueError
            When the (potentially sliced map) has more than one phase in
            the data.

        Examples
        --------
        >>> from orix import data
        >>> xmap = data.sdss_ferrite_austenite(allow_download=True)
        >>> xmap
        Phase   Orientations       Name  Space group  Point group  Proper point group       Color
            1   5657 (48.4%)  austenite         None          432                 432    tab:blue
            2   6043 (51.6%)    ferrite         None          432                 432  tab:orange
        Properties: iq, dp
        Scan unit: um
        >>> xmap["austenite"].orientations
        Orientation (5657,) 432
        [[ 0.8686  0.3569 -0.2749 -0.2064]
         [ 0.8681  0.3581 -0.2744 -0.2068]
         [ 0.8684  0.3578 -0.2751 -0.2052]
         ...
         [ 0.9639  0.022   0.0754 -0.2545]
         [ 0.8854  0.3337 -0.2385  0.2187]
         [ 0.885   0.3341 -0.2391  0.2193]]
        """
        phases = self.phases_in_data
        if phases.size == 1:
            # Extract top matching rotations per point, if more than one
            if self.rotations_per_point > 1:
                rotations = self.rotations[:, 0]
            else:
                rotations = self.rotations
            # Point group can be None, so it cannot be passed upon
            # initialization to Orientation but has to be set afterwards
            # to trigger the checks
            orientations = Orientation(rotations)
            orientations.symmetry = phases[:].point_group
            return orientations
        else:
            raise ValueError(
                f"Data has the phases {phases.names}, however, you are executing a "
                "command that only permits one phase."
            )

    @property
    def is_indexed(self) -> np.ndarray:
        """Return whether points in data are indexed."""
        return self.phase_id != -1

    @property
    def all_indexed(self) -> np.ndarray:
        """Return whether all points in data are indexed."""
        return np.count_nonzero(self.is_indexed) == self.is_indexed.size

    @property
    def prop(self) -> CrystalMapProperties:
        """Return the data properties in each data point."""
        self._prop.is_in_data = self.is_in_data
        self._prop.id = self.id
        return self._prop

    @property
    def _coordinates(self) -> dict[str, np.ndarray | None]:
        """Return the coordinates of points in the data."""
        return {"z": self.z, "y": self.y, "x": self.x}

    @property
    def _all_coordinates(self) -> dict[str, np.ndarray | None]:
        """Return the coordinates of all points."""
        return {"z": self._z, "y": self._y, "x": self._x}

    @property
    def _step_sizes(self) -> dict[str, float | None]:
        """Return the step sizes of dimensions in the data."""
        return {"z": self.dz, "y": self.dy, "x": self.dx}

    @property
    def _coordinate_axes(self) -> dict:
        """Return which data axis corresponds to which coordinate."""
        present_coordinates = [k for k, v in self._coordinates.items() if v is not None]
        return {i: coord for i, coord in zip(range(self.ndim), present_coordinates)}

    def __getattr__(self, item) -> Any:
        """Return an attribute in the :attr:`prop` dictionary directly
        from the ``CrystalMap`` instance.

        Called when the default attribute access fails with an
        ``AttributeError``.
        """
        if item in self.__getattribute__("_prop"):
            # Calls CrystalMapProperties.__getitem__()
            return self.prop[item]
        else:
            return object.__getattribute__(self, item)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set a class instance attribute."""
        if hasattr(self, "_prop") and name in self._prop:
            # Calls CrystalMapProperties.__setitem__()
            self.prop[name] = value
        else:
            return object.__setattr__(self, name, value)

    def __getitem__(self, key: str | slice | tuple | int | np.ndarray) -> CrystalMap:
        """return a subset of the CrystalMap instance.

        See the docstring of ``__init__()`` for examples.

        Parameters
        ----------
        key
            If ``str``, it must be a valid phase or ``"not_indexed"`` or
            ``"indexed"``. If ``slice`` or ``tuple``, it must be within
            the map shape. If ``int``, it must be a valid
            :attr:`self.id`. If boolean array, it must be of map shape.
        """
        # Create an empty boolean mask.
        data_to_keep = np.zeros(self.size, dtype=bool)

        if isinstance(key, (str, slice, int)):
            key = (key,)  # make non-iterable inputs iterable.

        # determine what method is being used for masking out data.
        if isinstance(key, tuple) and np.all(type(x) is str for x in key):
            # This is a list of strings referencing phases (or lack thereof)
            for k in key:
                for phase_id, phase in self.phases:
                    if k == phase.name:
                        data_to_keep[self.phase_id == phase_id] = True
                    elif k.lower() == "indexed":
                        # Add all indexed phases to data
                        data_to_keep[self.phase_id != -1] = True
                    elif k.lower() == "not_indexed":
                        data_to_keep[self.phase_id == -1] = True
                    else:
                        raise Warning("phase {} was not found in self.phases".format(k))

        elif isinstance(key, np.ndarray) and key.dtype == np.bool_:
            # Boolean numpy array.
            if key.shape == self.shape:  # mask on coordinates
                if self._indexing_order == "zyx":
                    data_to_keep = key.flatten("C")
                else:
                    data_to_keep = key.flatten("F")
            elif key.size == self.rotations.shape[0]:  # mask on all data
                data_to_keep = key
            elif key.size == np.count_nonzero(self.is_in_data):  # mask on included data
                data_to_keep[self.is_in_data] = key
            else:
                ValueError(
                    "boolean arrays must be either the same size or shape as "
                    + "the CrystalMap to allow for numpy-like masking"
                )

        elif np.all(isinstance(x, (slice, int)) for x in key):
            # Numpy-like slicing.
            data_to_keep = np.ones(self.size, dtype=bool)
            slices = [slice(None, None, None)] * self.ndim
            for i, k in enumerate(key):
                slices[i] = k
            if self._indexing_order == "xyz":
                slices = slices[::-1]
            for axis, choice in zip((self.layer, self.row, self.col), slices):
                if isinstance(choice, int):
                    data_to_keep[axis != choice] = False
                else:
                    if choice.stop is not None:
                        data_to_keep[axis > choice.stop] = False
                    if choice.start is not None:
                        data_to_keep[axis <= choice.start] = False
                        axis = axis - choice.start
                    if choice.step is not None:
                        data_to_keep[axis % choice.start != 0] = False
        else:
            ValueError("'key was not recognized as phase names, slices, or indices'")

        # apply existing data mask if applicable
        data_to_keep[~self.is_in_data] = False

        # Return a new instance of just the desired subset of data
        # TODO: create new map
        return

    def __repr__(self) -> str:
        """Return a nice representation of the data."""
        if self.size == 0:
            return "No data."

        phases = self.phases_in_data
        phase_ids = self.phase_id

        # Ensure attributes set to None are treated OK
        names = ["None" if not name else name for name in phases.names]
        sg_names = ["None" if not i else i.short_name for i in phases.space_groups]
        pg_names = ["None" if not i else i.name for i in phases.point_groups]
        ppg_names = [
            "None" if not i else i.proper_subgroup.name for i in phases.point_groups
        ]

        # Determine column widths
        unique_phases = np.unique(phase_ids)
        p_sizes = [np.where(phase_ids == i)[0].size for i in unique_phases]
        id_len = 5
        ori_len = max(max([len(str(p_size)) for p_size in p_sizes]) + 9, 12)
        name_len = max(max([len(n) for n in names]), 4)
        sg_len = max(max([len(i) for i in sg_names]), 11)
        pg_len = max(max([len(i) for i in pg_names]), 11)
        ppg_len = max(max([len(i) for i in ppg_names]), 18)
        col_len = max(max([len(i) for i in phases.colors]), 5)

        # Column alignment
        align = ">"  # right ">" or left "<"

        # Header (note the two-space spacing)
        representation = (
            "{:{align}{width}}  ".format("Phase", width=id_len, align=align)
            + "{:{align}{width}}  ".format("Orientations", width=ori_len, align=align)
            + "{:{align}{width}}  ".format("Name", width=name_len, align=align)
            + "{:{align}{width}}  ".format("Space group", width=sg_len, align=align)
            + "{:{align}{width}}  ".format("Point group", width=pg_len, align=align)
            + "{:{align}{width}}  ".format(
                "Proper point group", width=ppg_len, align=align
            )
            + "{:{align}{width}}\n".format("Color", width=col_len, align=align)
        )

        # Overview of data for each phase
        for i, phase_id in enumerate(unique_phases.astype(int)):
            p_size = np.where(phase_ids == phase_id)[0].size
            p_fraction = 100 * p_size / self.size
            ori_str = f"{p_size} ({p_fraction:.1f}%)"
            representation += (
                f"{phase_id:{align}{id_len}}  "
                + f"{ori_str:{align}{ori_len}}  "
                + f"{names[i]:{align}{name_len}}  "
                + f"{sg_names[i]:{align}{sg_len}}  "
                + f"{pg_names[i]:{align}{pg_len}}  "
                + f"{ppg_names[i]:{align}{ppg_len}}  "
                + f"{phases.colors[i]:{align}{col_len}}\n"
            )

        # Properties and spatial coordinates
        props = []
        for k in self.prop.keys():
            props.append(k)
        representation += "Properties: " + ", ".join(props) + "\n"

        # Scan unit
        representation += f"Scan unit: {self.scan_unit}"

        return representation

    def deepcopy(self) -> CrystalMap:
        """Return a deep copy using :func:`copy.deepcopy` function."""
        return copy.deepcopy(self)

    @classmethod
    def empty(
        cls,
        shape: int | tuple[int, int] | tuple[int] | None = None,
        step_sizes: float | tuple[float] | tuple[float, float] | None = None,
    ) -> CrystalMap:
        """Return a crystal map of a given 2D shape and step sizes with
        identity rotations.

        Parameters
        ----------
        shape
            Map shape. Default is a 2D map of shape (5, 10), i.e. with
            five rows and ten columns.
        step_sizes
            Map step sizes. If not given, it is set to 1 px in each map
            direction given by ``shape``.

        Returns
        -------
        xmap
            Crystal map.
        """
        d, n = create_coordinate_arrays(shape, step_sizes)
        d["rotations"] = Rotation.identity((n,))
        return cls(**d)

    def get_map_data(
        self,
        item: str | np.ndarray,
        decimals: int | None = None,
        fill_value: int | float | None = np.nan,
    ) -> np.ndarray:
        """Return an array of a class instance attribute, with values
        equal to ``False`` in :attr:`self.is_in_data` set to
        ``fill_value``, of map data shape.

        Parameters
        ----------
        item
            Name of the class instance attribute or a
            :class:`numpy.ndarray`.
        decimals
            Number of decimals to round data point values to. If not
            given, no rounding is done.
        fill_value
            Value to fill points not in the data with. Default is
            :class:`numpy.nan`.

        Returns
        -------
        output_array
            Array of the class instance attribute with points not in
            data set to ``fill_value``, of float data type.

        Notes
        -----
        Rotations and orientations should be accessed via
        :attr:`rotations` and :attr:`orientations`.

        If ``item`` is ``"orientations"`` or ``"rotations"`` and there
        are multiple rotations per point, only the first rotation is
        used. Rotations are returned as Euler angles.

        Examples
        --------
        >>> from orix import data
        >>> xmap = data.sdss_ferrite_austenite(allow_download=True)
        >>> xmap
        Phase   Orientations       Name  Space group  Point group  Proper point group       Color
            1   5657 (48.4%)  austenite         None          432                 432    tab:blue
            2   6043 (51.6%)    ferrite         None          432                 432  tab:orange
        Properties: iq, dp
        Scan unit: um
        >>> xmap.shape
        (100, 117)

        Get a 2D map in the correct shape of any attribute, ready for
        plotting

        >>> xmap.iq.shape
        (11700,)
        >>> iq = xmap.get_map_data("iq")
        >>> iq.shape
        (100, 117)
        """
        # Get full map shape
        map_shape = self._original_shape

        # Declare array of correct shape, accounting for RGB
        # TODO: Better account for `item.shape`, e.g. quaternions
        #  (item.shape[-1] == 4) in a more general way than here (not
        #  more if/else)!
        map_size = np.prod(map_shape)
        if isinstance(item, np.ndarray):
            array = np.empty(map_size, dtype=item.dtype)
            if item.shape[-1] == 3 and map_size > 3:  # Assume RGB
                map_shape += (3,)
                array = np.column_stack((array,) * 3)
        elif item in ["orientations", "rotations"]:  # Definitely RGB
            array = np.empty(map_size, dtype=np.float64)
            map_shape += (3,)
            array = np.column_stack((array,) * 3)
        else:
            array = np.empty(map_size, dtype=np.float64)

        # Enter non-masked values into array
        if isinstance(item, np.ndarray):
            # TODO: Account for 2D map with more than one value per point
            array[self.is_in_data] = item
        elif item in ["orientations", "rotations"]:
            if item == "rotations":
                # Use only the top matching rotation per point
                if self.rotations_per_point > 1:
                    rotations = self.rotations[:, 0]
                else:
                    rotations = self.rotations
                array[self.is_in_data] = rotations.to_euler()
            else:  # item == "orientations"
                # Fill in orientations per phase
                for i, _ in self.phases_in_data:
                    phase_mask = (self._phase_id == i) * self.is_in_data
                    phase_mask_in_data = self.phase_id == i
                    array[phase_mask] = self[phase_mask_in_data].orientations.to_euler()
        else:  # String
            data = self.__getattr__(item)
            if data is None:
                raise ValueError(f"{item} is {data}.")
            else:
                # TODO: Account for 2D map with more than one value per point
                array[self.is_in_data] = data
                array = array.astype(data.dtype)

        # Slice and reshape array
        slices = self._data_slices_from_coordinates()
        reshaped_array = array.reshape(map_shape)
        sliced_array = reshaped_array[slices]

        # Reshape and slice mask with points not in data
        if array.shape[-1] == 3 and map_size > 3:  # RGB
            not_in_data = np.dstack((~self.is_in_data,) * 3)
        else:  # Scalar
            not_in_data = ~self.is_in_data
        not_in_data = not_in_data.reshape(map_shape)[slices]

        # Fill points not in data with the fill value
        if not_in_data.any():
            if fill_value is None or fill_value is np.nan:
                sliced_array = sliced_array.astype(np.float64)
            sliced_array[not_in_data] = fill_value

        # Round values
        if decimals is not None:
            output_array = np.round(sliced_array, decimals=decimals)
        else:  # np.issubdtype(array.dtype, np.bool_):
            output_array = sliced_array

        return output_array

    def plot(
        self,
        value: np.ndarray | str | None = None,
        overlay: str | np.ndarray | None = None,
        scalebar: bool | None = None,
        scalebar_properties: dict | None = None,
        legend: bool = True,
        legend_properties: dict | None = None,
        colorbar: bool = False,
        colorbar_label: str | None = None,
        colorbar_properties: dict | None = None,
        remove_padding: bool = False,
        return_figure: bool = False,
        axis: int | None = None,
        layer: int | None = None,
        figure_kwargs: dict | None = None,
        **kwargs,
    ) -> mfigure.Figure | None:
        r"""Plot a 2D map with any crystallographic map property as map
        values.

        Wraps :meth:`matplotlib.axes.Axes.imshow`: see that method for
        relevant keyword arguments.

        Parameters
        ----------
        value
            An array or an attribute string to plot. If not given, a
            phase map is plotted.
        overlay
            Name of map property or a property array to use in the
            alpha (RGBA) channel. The property range is adjusted for
            maximum contrast. Not used if not given.
        scalebar
            Whether to add a scalebar along the horizontal map
            dimension. If not given, a scalebar is added if
            :mod:`matplotlib-scalebar` is installed.
        scalebar_properties
            Keyword arguments passed to
            :class:`matplotlib_scalebar.scalebar.ScaleBar`.
        legend
            Whether to add a legend to the plot. This is only
            implemented for a phase plot (in which case default is
            ``True``).
        legend_properties
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.legend`.
        colorbar
            Whether to add an opinionated colorbar (default is
            ``False``).
        colorbar_label
            Label of colorbar.
        colorbar_properties
            Keyword arguments passed to
            :meth:`orix.plot.CrystalMapPlot.add_colorbar`.
        remove_padding
            Whether to remove white padding around figure (default is
            ``False``).
        return_figure
            Whether to return the figure (default is ``False``).
        axis
            For 3D xmap, axis on which to plot 2D slice.
        layer
            For 3D xmap, layer on defined axis to plot 2D slice.
        figure_kwargs
            Keyword arguments passed to
            :func:`matplotlib.pyplot.subplots`.
        **kwargs
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.imshow`.

        Returns
        -------
        fig
            The created figure, returned if ``return_figure=True``.

        See Also
        --------
        matplotlib.axes.Axes.imshow
        orix.plot.CrystalMapPlot.plot_map
        orix.plot.CrystalMapPlot.add_scalebar
        orix.plot.CrystalMapPlot.add_overlay
        orix.plot.CrystalMapPlot.add_colorbar

        Examples
        --------
        >>> from orix import data
        >>> xmap = data.sdss_ferrite_austenite(allow_download=True)
        >>> xmap
        Phase   Orientations       Name  Space group  Point group  Proper point group       Color
            1   5657 (48.4%)  austenite         None          432                 432    tab:blue
            2   6043 (51.6%)    ferrite         None          432                 432  tab:orange
        Properties: iq, dp
        Scan unit: um

        Plot phase map

        >>> xmap.plot()

        Remove padding and return the figure (e.g. to be saved)

        >>> fig = xmap.plot(remove_padding=True, return_figure=True)

        Plot a dot product (similarity score) map

        >>> xmap.plot("dp", colorbar=True, colorbar_label="Dot product", cmap="gray")
        """
        # Register "plot_map" projection with Matplotlib
        import orix.plot.crystal_map_plot

        if figure_kwargs is None:
            figure_kwargs = {}

        fig, ax = plt.subplots(subplot_kw={"projection": "plot_map"}, **figure_kwargs)
        ax.plot_map(
            self,
            value=value,
            scalebar=scalebar,
            scalebar_properties=scalebar_properties,
            legend=legend,
            legend_properties=legend_properties,
            axis=axis,
            layer=layer,
            **kwargs,
        )

        if overlay is not None:
            ax.add_overlay(self, overlay)

        if remove_padding:
            ax.remove_padding()

        if colorbar:
            if colorbar_properties is None:
                colorbar_properties = dict()
            ax.add_colorbar(label=colorbar_label, **colorbar_properties)

        if return_figure:
            return fig

    def _xmap_slice_from_axis(self, axis: int, layer: int) -> "CrystalMap":
        """Returns a 2D slice of a CrystalMap object along a given axis.

        Parameters
        ----------
        axis
            For 3D xmap, axis on which to plot 2D slice.
        layer
            For 3D xmap, layer on defined axis to plot 2D slice.

        Returns
        -------
        CrystalMap
            2D CrystalMap slice.
        """
        return self[(slice(None),) * (axis % self.ndim) + (slice(layer, layer + 1),)]

    def _data_slices_from_coordinates(self, only_is_in_data: bool = True) -> tuple:
        """Return a slices defining the current data extent in all
        directions.

        Parameters
        ----------
        only_is_in_data
            Whether to determine slices of points in data or all points.
            Default is ``True``.

        Returns
        -------
        slices
            Data slice in each existing direction in (y, x) order.
        """
        if only_is_in_data:
            coordinates = self._coordinates
        else:
            coordinates = self._all_coordinates
        slices = _data_slices_from_coordinates(coordinates, self._step_sizes)
        return slices

    def _data_shape_from_coordinates(self, only_is_in_data: bool = True) -> tuple:
        """Return data shape based upon coordinate arrays.

        Parameters
        ----------
        only_is_in_data
            Whether to determine shape of points in data or all points.
            Default is ``True``.

        Returns
        -------
        data_shape
            Shape of data in each existing direction in (y, x) order.
        """
        data_shape = []
        for dim_slice in self._data_slices_from_coordinates(only_is_in_data):
            data_shape.append(dim_slice.stop - dim_slice.start)
        return tuple(data_shape)


def _data_slices_from_coordinates(
    coords: dict[str, np.ndarray], steps: dict[str, float] | None = None
) -> tuple[slice]:
    """Return a list of slices defining the current data extent in all
    directions.

    Parameters
    ----------
    coords
        Dictionary with coordinate arrays.
    steps
        Dictionary with step sizes in each direction. If not given, they
        are computed from *coords*.

    Returns
    -------
    slices
        Data slice in each direction.
    """
    if steps is None:
        steps = {
            "x": _step_size_from_coordinates(coords["x"]),
            "y": _step_size_from_coordinates(coords["y"]),
            "z": _step_size_from_coordinates(coords["z"]),
        }
    slices = []
    for coords, step in zip(coords.values(), steps.values()):
        if coords is not None and step != 0:
            c_min, c_max = np.min(coords), np.max(coords)
            i_min = int(np.around(c_min / step))
            i_max = int(np.around((c_max / step) + 1))
            slices.append(slice(i_min, i_max))
    slices = tuple(slices)
    return slices


def _step_size_from_coordinates(coordinates: np.ndarray) -> float:
    """Return step size in input *coordinates* array.

    Parameters
    ----------
    coordinates
        Linear coordinate array.

    Returns
    -------
    step_size
        Step size in *coordinates* array.
    """
    unique = np.sort(np.unique(coordinates))
    if unique.size != 1:
        deltas, counts = np.unique(unique[1:] - unique[:-1], return_counts=True)
        step_size = deltas[np.argmax(counts)]
    else:
        step_size = 0
    return step_size


def create_coordinate_arrays(
    shape: tuple[int] | tuple[int, int] | None = None,
    step_sizes: tuple[float] | tuple[float, float] | None = None,
) -> tuple[dict, int]:
    """Return flattened coordinate arrays from a given map shape and
    step sizes, suitable for initializing a
    :class:`~orix.crystal_map.CrystalMap`.

    Arrays for 1D or 2D maps can be returned.

    Parameters
    ----------
    shape
        Map shape. Default is a 2D map of shape (5, 10) with five rows
        and ten columns.
    step_sizes
        Map step sizes. If not given, it is set to 1 px in each map
        direction given by *shape*.

    Returns
    -------
    d
        Dictionary with keys ``"x"`` and ``"y"``, depending on the
        length of *shape*, with coordinate arrays.
    map_size
        Number of map points.

    Examples
    --------
    >>> from orix.crystal_map import create_coordinate_arrays
    >>> create_coordinate_arrays((2, 3))
    ({'x': array([0, 1, 2, 0, 1, 2]), 'y': array([0, 0, 0, 1, 1, 1])}, 6)
    >>> create_coordinate_arrays((3, 2))
    ({'x': array([0, 1, 0, 1, 0, 1]), 'y': array([0, 0, 1, 1, 2, 2])}, 6)
    >>> create_coordinate_arrays((2, 3), (1.5, 1.5))
    ({'x': array([0. , 1.5, 3. , 0. , 1.5, 3. ]), 'y': array([0. , 0. , 0. , 1.5, 1.5, 1.5])}, 6)
    """
    if not shape:
        shape = (5, 10)
    ndim = len(shape)
    if not step_sizes:
        step_sizes = (1,) * ndim

    if ndim == 3 or len(step_sizes) > 2:
        raise ValueError("Can only create coordinate arrays for 2D maps")

    # Set up as if a 2D map is to be returned
    dy, dx = (1,) * (2 - ndim) + step_sizes
    ny, nx = (1,) * (2 - ndim) + shape
    d = dict()

    # Add coordinate arrays depending on the number of map dimensions
    d["x"] = np.tile(np.arange(nx) * dx, ny).flatten()
    map_size = nx
    if ndim > 1:
        d["y"] = np.sort(np.tile(np.arange(ny) * dy, nx)).flatten()
        map_size *= ny

    return d, map_size
