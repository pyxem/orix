# -*- coding: utf-8 -*-
# Copyright 2018-2020 The pyXem developers
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with orix.  If not, see <http://www.gnu.org/licenses/>.

import copy
import logging

import numpy as np

from orix.quaternion.rotation import Rotation
from orix.quaternion.orientation import Orientation
from .phase_list import PhaseList
from .crystal_map_properties import CrystalMapProperties

_log = logging.getLogger(__name__)


class CrystalMap:
    """Crystallographic map of rotations, crystal phases and key properties
    associated with every spatial coordinate in a 1D, 2D or 3D space.

    All properties are stored as 1D arrays, and reshaped when necessary.

    Attributes
    ----------
    all_indexed : bool
        Whether all data points are indexed.
    dx, dy, dz : float
        Step sizes in x, y and z directions.
    id : int
        Data point ID.
    is_indexed : numpy.ndarray
        Boolean array with True for indexed data points.
    ndim : int
        Number of map dimensions.
    orientations : orix.quaternion.orientation.Orientation
        Orientation(s) of each data point.
    phase_id : numpy.ndarray
        Phase ID of each data point.
    phases : orix.crystal_map.PhaseList
        List of phases with their IDs, names, crystal symmetries and
        colors (possibly more than are in the data).
    phases_in_data : orix.crystal_map.PhaseList
        List of phases in the data, with their IDs, names, crystal
        symmetries and colors.
    prop : dict
        Dictionary of numpy arrays of key properties of each data point.
    rotations : orix.quaternion.rotation.Rotation
        Rotations of each data point.
    rotations_per_point : int
        Number of rotations per data point.
    scan_unit : str
        Length unit of data, default is 'px'.
    shape : tuple
        Shape of data in points (pixels).
    size : int
        Number of data points.
    x, y, z : float
        Coordinates in each data direction.

    Methods
    -------
    deepcopy()
        Return a deep copy using :py:func:`~copy.deepcopy` function.
    get_map_data(item, decimals=3, fill_value=None)
        Return an array of a class instance attribute, with masked values
        set to `fill_value`, of map shape.
    """

    def __init__(
        self,
        rotations,
        phase_id=None,
        x=None,
        y=None,
        z=None,
        phase_name=None,
        symmetry=None,
        prop=None,
    ):
        """
        Parameters
        ----------
        rotations : orix.quaternion.rotation.Rotation
            Rotation of each data point. Must contain only one spatial
            dimension in the first array axis. May contain multiple
            rotations per point, included in the second array axes. Crystal
            map data size is set equal to the first array axis' size.
        phase_id : numpy.ndarray, optional
            Phase ID of each pixel. IDs equal to -1 are considered not
            indexed. If ``None`` is passed (default), all points are
            considered to belong to one phase with ID 1.
        x : numpy.ndarray, optional
            Map x coordinate of each data point. If ``None`` is passed,
            the map is assumed to be 1D, and it is set to an array of
            increasing integers from 0 to the length of the `phase_id`
            array.
        y : numpy.ndarray, optional
            Map y coordinate of each data point. If ``None`` is passed,
            the map is assumed to be 1D, and it is set to ``None``.
        z : numpy.ndarray, optional
            Map z coordinate of each data point. If ``None`` is passed, the
            map is assumed to be 2D or 1D, and it is set to ``None``.
        phase_name : str or list of str, optional
            Name of phases.
        symmetry : str or list of str, optional
            Point group of crystal symmetries of phases in the map.
        prop : dict of numpy.ndarray, optional
            Dictionary of properties of each data point.
        """

        # Set rotations
        if not isinstance(rotations, Rotation):
            raise ValueError(
                f"rotations must be of type {Rotation}, not {type(rotations)}."
            )
        # Underscores are used to enable getting attributes with masks
        self._rotations = rotations

        # Set data size
        data_size = rotations.shape[0]

        # Set phase ID
        if phase_id is None:
            phase_id = np.ones(data_size)
        phase_id = phase_id.astype(int)
        self._phase_id = phase_id

        # Set data point ID
        point_id = np.arange(data_size)
        self._id = point_id

        # Set spatial coordinates
        # TODO: Enable setting these via a specimen_reference_frame attribute
        if x is None:
            x = np.arange(data_size)
        self._x = x
        self._y = y
        self._z = z

        # Set step sizes
        # TODO: Enable updating these if the spatial coordinate arrays are updated
        # TODO: Can be done via CrystalMap._step_size_from_coordinates()
        self.dx = self._step_size_from_coordinates(x)
        self.dy = self._step_size_from_coordinates(y)
        self.dz = self._step_size_from_coordinates(z)

        # Create phase list
        unique_phase_ids = np.unique(phase_id)  # Sorted in ascending order
        include_not_indexed = False
        if unique_phase_ids[0] == -1:
            include_not_indexed = True
            unique_phase_ids = unique_phase_ids[1:]
        self.phases = PhaseList(
            names=phase_name, symmetries=symmetry, phase_ids=unique_phase_ids,
        )

        # Set whether measurements are indexed
        is_indexed = np.ones(data_size, dtype=bool)
        is_indexed[np.where(phase_id == -1)] = False

        # Add 'not_indexed' to phase list and ensure not indexed points have the correct
        # phase ID
        if include_not_indexed:
            self.phases.add_not_indexed()
            self._phase_id[~is_indexed] = -1

        # Set array with True for points in data
        self.is_in_data = np.ones(data_size, dtype=bool)

        # Set scan unit
        self.scan_unit = "px"

        # Set properties
        self._prop = CrystalMapProperties(prop, id=point_id)

        # Set original data shape (needed if data shape changes in__getitem__())
        self._original_shape = self._data_shape_from_coordinates()

    @property
    def id(self):
        """Return ID of points in the data."""
        return self._id[self.is_in_data]

    @property
    def size(self):
        """Return total number of points in the data."""
        return np.sum(self.is_in_data)

    @property
    def shape(self):
        """Return shape of points in the data."""
        return self._data_shape_from_coordinates()

    @property
    def ndim(self):
        """Return number of data dimensions of points in the data."""
        return len(self.shape)

    @property
    def x(self):
        """Return x coordinates of points in data."""
        if self._x is None:
            # Will never go here as the implementation is now, however might be the
            # case when enabling setting a specimen reference frame
            return self._x
        else:
            return self._x[self.is_in_data]

    @property
    def y(self):
        """Return y coordinates (possibly None) of points in the data."""
        if self._y is None:
            return self._y
        else:
            return self._y[self.is_in_data]

    @property
    def z(self):
        """Return z coordinates (possibly None) of points in the data."""
        if self._z is None:
            return self._z
        else:
            return self._z[self.is_in_data]

    @property
    def phase_id(self):
        """Return phase IDs of points in the data."""
        return self._phase_id[self.is_in_data]

    @phase_id.setter
    def phase_id(self, value):
        """Set phase ID of points in the data."""
        self._phase_id[self.is_in_data] = value
        if value == -1 and "not_indexed" not in self.phases.names:
            self.phases.add_not_indexed()

    @property
    def phases_in_data(self):
        """Return a list of phases in the data.

        This is needed because it can be useful to have phases not in the
        data but in `self.phases`.
        """
        unique_ids = np.unique(self.phase_id)
        return self.phases[np.intersect1d(unique_ids, self.phases.phase_ids)]

    @property
    def rotations(self):
        """Return a Rotation object of rotations in the data."""
        return self._rotations[self.is_in_data]

    @property
    def rotations_per_point(self):
        """Return number of rotations per data point."""
        return self.rotations.size // self.is_indexed.size

    @property
    def orientations(self):
        """Return an Orientation object of orientations in data."""
        phases = self.phases_in_data
        if phases.size == 1:
            # Extract top matching rotations per point, if more than one match
            if self.rotations_per_point > 1:
                rotations = self.rotations[:, 0]
            else:
                rotations = self.rotations
            # Get phase from phase ID
            phase = phases[phases.phase_ids[0]]
            return Orientation(rotations).set_symmetry(phase.symmetry)
        else:
            raise ValueError(
                f"Map contains the phases {phases.names}, however, you are"
                " executing a command that only permits one phase."
            )

    @property
    def is_indexed(self):
        """Return whether points in data are indexed."""
        return self.phase_id != -1

    @property
    def all_indexed(self):
        """Return whether all points in data are indexed."""
        return np.count_nonzero(self.is_indexed) == self.is_indexed.size

    @property
    def prop(self):
        """Return a :class:`~orix.crystal_map.CrystalMapProperties`
        dictionary with data properties.
        """
        self._prop.is_in_data = self.is_in_data
        self._prop.id = self.id
        return self._prop

    @property
    def _coordinates(self):
        """Return a dictionary of coordinates of points in the data."""
        # TODO: Make this "dynamic"/dependable when enabling specimen reference frame
        return {"z": self.z, "y": self.y, "x": self.x}

    @property
    def _step_sizes(self):
        """Return a dictionary of step sizes of dimensions in the data."""
        # TODO: Make this "dynamic"/dependable when enabling specimen reference frame
        return {"z": self.dz, "y": self.dy, "x": self.dx}

    def __getattr__(self, item):
        """Get an attribute in the `prop` dictionary directly from the
        CrystalMap object.

        Called when the default attribute access fails with an
        AttributeError.
        """
        if item in self.__getattribute__("_prop"):
            return self.prop[item]  # Calls CrystalMapProperties.__getitem__()
        else:
            return object.__getattribute__(self, item)

    def __setattr__(self, name, value):
        """Set a class instance attribute."""
        if hasattr(self, "_prop") and name in self._prop:
            self.prop[name] = value  # Calls CrystalMapProperties.__setitem__()
        else:
            return object.__setattr__(self, name, value)

    def __getitem__(self, key):
        """Return a masked copy of the CrystalMap object.

        Parameters
        ----------
        key : str, slice or boolean numpy.ndarray
            If str, it must be a valid phase or "not_indexed" or "indexed".
            If slice, it must be within the map shape. If boolean array, it
            must be of map shape.

        Examples
        --------
        A CrystalMap object can be indexed in multiple ways...

        >>> cm
        Phase   Orientations       Name  Symmetry       Color
            1   5657 (48.4%)  austenite       432    tab:blue
            2   6043 (51.6%)    ferrite       432  tab:orange
        Properties: iq, dp
        >>> cm.shape
        (100, 117)

        ... by slicing

        >>> cm2 = cm[20:40, 50:60]
        >>> cm2
        Phase   Orientations       Name  Symmetry       Color
            1    148 (74.0%)  austenite       432    tab:blue
            2     52 (26.0%)    ferrite       432  tab:orange
        Properties: iq, dp
        Scan unit: um
        >>> cm2.shape
        (20, 10)

        ... by phase name(s)

        >>> cm2 = cm["austenite"]
        Phase   Orientations       Name  Symmetry     Color
            1  5657 (100.0%)  austenite       432  tab:blue
        Properties: iq, dp
        Scan unit: um
        >>> cm2.shape
        (100, 117)
        >>> cm["austenite", "ferrite"]
        Phase   Orientations       Name  Symmetry       Color
            1   5657 (48.4%)  austenite       432    tab:blue
            2   6043 (51.6%)    ferrite       432  tab:orange
        Properties: iq, dp
        Scan unit: um

        ... by "indexed" and "not_indexed"

        >>> cm["indexed"]
        Phase   Orientations       Name  Symmetry       Color
            1   5657 (48.4%)  austenite       432    tab:blue
            2   6043 (51.6%)    ferrite       432  tab:orange
        Properties: iq, dp
        Scan unit: um
        >>> cm["not_indexed"]
        No data.

        ... or by boolean arrays ((chained) conditional(s))

        >>> cm[cm.dp > 0.81]
        Phase   Orientations       Name  Symmetry       Color
            1   4092 (44.8%)  austenite       432    tab:blue
            2   5035 (55.2%)    ferrite       432  tab:orange
        Properties: iq, dp
        Scan unit: um
        >>> cm[(cm.iq > np.mean(cm.iq)) & (cm.phase_id == 1)]
        Phase   Orientations       Name  Symmetry     Color
            1  1890 (100.0%)  austenite       432  tab:blue
        Properties: iq, dp
        Scan unit: um
        """
        # Initiate a mask to be added to the returned copy of the CrystalMap object, to
        # ensure that only the unmasked values are in the data of the copy (True in
        # `is_in_data`). First, no points are in the data, but are added if they satisfy
        # the condition in the input key (phase string, boolean or slice).
        is_in_data = np.zeros(self.size, dtype=bool)

        # The original object might already have set some points to not be in the data.
        # If so, `is_in_data` is used to update the original `is_in_data`. Since
        # `new_is_in_data` is not initiated for all key types, we declare it here and
        # check for it later.
        new_is_in_data = None

        # Override mask values
        if isinstance(key, str) or (isinstance(key, tuple) and isinstance(key[0], str)):
            # From phase string(s)
            if not isinstance(key, tuple):  # Make single string iterable
                key = (key,)
            for k in key:
                for phase_id, phase in self.phases:
                    if k == phase.name:
                        is_in_data[self.phase_id == phase_id] = True
                    elif k.lower() == "indexed":
                        # Add all indexed phases to the data
                        is_in_data[self.phase_id != -1] = True
        elif isinstance(key, np.ndarray) and key.dtype == np.bool_:
            # From boolean numpy array
            is_in_data = key
        elif isinstance(key, slice) or (
            isinstance(key, tuple) and any([(isinstance(i, slice)) for i in key])
        ):
            # From slice(s)
            if isinstance(key, slice):  # Make iterable if single slice
                key = (key,)

            slices = list([0] * self.ndim)
            for i, k in enumerate(key):
                slices[i] = k

            new_is_in_data = np.zeros(self.shape, dtype=bool)  # > 1D
            new_is_in_data[tuple(slices)] = True
            # Note that although all points within slice(s) was sought, points within
            # the slice(s) which are already removed from the data are still kept out by
            # this boolean multiplication
            new_is_in_data = new_is_in_data.flatten() * self.is_in_data

        # Insert the mask into a mask with the full map shape, if not done already
        if new_is_in_data is None:
            new_is_in_data = np.zeros_like(self.is_in_data, dtype=bool)  # 1D
            new_is_in_data[self.id] = is_in_data

        # Return a copy with all attributes shallow except for the mask
        new_map = copy.copy(self)
        new_map.is_in_data = new_is_in_data

        _log.debug(f"getitem: Return a shallow copy with updated is_in_data")
        return new_map

    def __repr__(self):
        """Print a nice representation of the data."""
        if self.size == 0:
            return "No data."

        phases = self.phases_in_data
        phase_ids = self.phase_id

        # Ensure attributes set to None are treated OK
        names = ["None" if not name else name for name in phases.names]
        symmetry_names = ["None" if not sym else sym.name for sym in phases.symmetries]

        # Determine column widths
        unique_phases = np.unique(phase_ids)
        p_sizes = [np.where(phase_ids == i)[0].size for i in unique_phases]
        id_len = 5
        ori_len = max(max([len(str(p_size)) for p_size in p_sizes]) + 9, 13)
        name_len = max(max([len(n) for n in names]), 5)
        sym_len = max(max([len(sn) for sn in symmetry_names]), 8)
        col_len = max(max([len(i) for i in phases.colors]), 6)

        # Column alignment
        align = ">"  # left ">" or right "<"

        # Header (note the two-space spacing)
        representation = (
            "{:{align}{width}}  ".format("Phase", width=id_len, align=align)
            + "{:{align}{width}}  ".format("Orientations", width=ori_len, align=align)
            + "{:{align}{width}}  ".format("Name", width=name_len, align=align)
            + "{:{align}{width}}  ".format("Symmetry", width=sym_len, align=align)
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
                + f"{symmetry_names[i]:{align}{sym_len}}  "
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

    def get_map_data(self, item, decimals=3, fill_value=None):
        """Return an array of a class instance attribute, with values equal
        to ``False`` in ``self.is_in_data``set to `fill_value`, of map
        shape.

        Parameters
        ----------
        item : str or numpy.ndarray
            Name of the class instance attribute or a numpy.ndarray.
        decimals : int, optional
            How many decimals to round data point values to (default is 3).
        fill_value : None, optional
            Value to fill points not in the data with. If ``None``
            (default), np.nan is used.

        Returns
        -------
        output_array : numpy.ndarray
            Array of the class instance attribute with points not in data
            set to `fill_value`, of float data type.
        """
        # Get full map shape
        map_shape = self._original_shape

        # Declare array of correct shape, accounting for RGB
        array = np.zeros(np.prod(map_shape))
        if isinstance(item, np.ndarray):
            if item.shape[-1] == 3:  # Assume RGB
                map_shape += (3,)
                array = np.column_stack((array,) * 3)
        elif item in ["orientations", "rotations"]:  # Definitely RGB
            map_shape += (3,)
            array = np.column_stack((array,) * 3)

        # Enter non-masked values into array
        if isinstance(item, np.ndarray):
            array[self.is_in_data] = item
        elif item in ["orientations", "rotations"]:
            # Use only the top matching rotation per point
            if self.rotations_per_point > 1:
                rotations = self._rotations[:, 0]
            else:
                rotations = self._rotations

            # Fill in orientations or rotations per phase
            # TODO: Consider whether orientations should be calculated upon loading
            for i, phase in self.phases_in_data:
                phase_mask = self.phase_id == i
                data_point_id = self[phase_mask]._id
                if item == "orientations":
                    array[data_point_id] = (
                        Orientation(rotations[data_point_id])
                        .set_symmetry(phase.symmetry)
                        .to_euler()
                    )
                else:  # item == "rotations"
                    array[data_point_id] = rotations[data_point_id].to_euler()
        else:  # String
            item = self.__getattr__(item)
            array[self.is_in_data] = item

        # Round values
        rounded_array = np.round(array, decimals=decimals)

        # Slice and reshape array
        slices = self._data_slices_from_coordinates()
        reshaped_array = rounded_array.reshape(map_shape)
        output_array = reshaped_array[slices]

        # Reshape and slice mask with points *not* in data
        if array.shape[-1] == 3:  # RGB
            not_in_data = np.dstack((~self.is_in_data,) * 3)
        else:  # Scalar
            not_in_data = ~self.is_in_data
        not_in_data = not_in_data.reshape(map_shape)[slices]

        output_array[not_in_data] = fill_value

        return output_array

    def deepcopy(self):
        """Return a deep copy using :func:`copy.deepcopy` function."""
        return copy.deepcopy(self)

    @staticmethod
    def _step_size_from_coordinates(coordinates):
        """Return step size in input `coordinates` array.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Linear coordinate array.

        Returns
        -------
        step_size : float
            Step size in `coordinates` array.
        """
        unique_sorted = np.sort(np.unique(coordinates))
        step_size = 0
        if unique_sorted.size != 1:
            step_size = unique_sorted[1] - unique_sorted[0]
        return step_size

    def _data_slices_from_coordinates(self):
        """Return a tuple of slices defining the current data extent in all
        directions.

        Returns
        -------
        slices : tuple of slices
            Data slice in each existing dimension, in (z, y, x) order. If
            data is not masked, the tuple is empty.
        """
        slices = []

        # Loop over dimension coordinates and step sizes
        for coordinates, step in zip(
            self._coordinates.values(), self._step_sizes.values()
        ):
            if coordinates is not None:
                slices.append(
                    slice(
                        int(np.min(coordinates) / step),
                        int(1 + np.max(coordinates) // step),
                    )
                )

        return tuple(slices)

    def _data_shape_from_coordinates(self):
        """Return data shape based upon coordinate arrays.

        Returns
        -------
        data_shape : tuple of ints
            Shape of data in all existing dimensions, in (z, y, x) order.
        """
        data_shape = []
        for dim_slice in self._data_slices_from_coordinates():
            data_shape.append(dim_slice.stop - dim_slice.start)
        return tuple(data_shape)
