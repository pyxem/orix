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
from numbers import Number

import numpy as np

from orix.quaternion.rotation import Rotation
from orix.quaternion.orientation import Orientation
from .phase_list import PhaseList


class CrystalMap:
    """Crystallographic map of rotations, crystal phases and key
     properties associated with every spatial coordinate in a 1D, 2D or 3D
     space.

    Crystal phases and key properties are stored in the map shape, while
    rotations are stored as a :class:`orix.quaternion.rotation.Rotation`
    object with one spatial dimension.

    Attributes
    ----------
    all_indexed : bool
        Whether all pixels are indexed.
    indexed : numpy.ndarray
        Boolean array with True for indexed pixels.
    ndim : int
        Number of map dimensions.
    orientations : orix.quaternion.orientation.Orientation
        Orientations of each pixel. Always 1D.
    phase_id_map : numpy.ndarray
        Phase ID of each pixel as imported.
    phases : orix.crystal_map.PhaseList
        List of phases with their IDs, names, crystal symmetries and
        colors (possibly more than are in the map).
    phases_in_map : orix.crystal_map.PhaseList
        List of phases in the map, with their IDs, names, crystal
        symmetries and colors.
    prop : dict
        Dictionary of numpy arrays of quality metrics or other properties
        of each pixel.
    rotations : orix.quaternion.rotation.Rotation
        Rotations of each pixel. Always 1D.
    scan_unit : str
        Length unit of map, default is 'px'.
    shape : tuple
        Shape of map in pixels.
    size : int
        Number of pixels in map.
    step_sizes : numpy.ndarray of floats
        An array of the step size in each map direction.

    Methods
    -------
    deepcopy()
        Return a deep copy using :py:func:`~copy.deepcopy` function.

    """

    def __init__(
            self,
            rotations,
            phase_id_map,
            phase_name=None,
            symmetry=None,
            prop=None,
            indexed=None,
            step_sizes=None,
    ):
        """
        Parameters
        ----------
        rotations : orix.quaternion.rotation.Rotation
            Rotation of each pixel.
        phase_id_map : numpy.ndarray
            Phase ID of each pixel. The map shape is set to this array's
            shape.
        phase_name : str or list of str, optional
            Name of phases.
        symmetry : str or list of str, optional
            Point group of crystal symmetries of phases in the map.
        prop : dict of numpy.ndarray, optional
            Dictionary of properties of each pixel.
        indexed : numpy.ndarray
            Boolean array with True for indexed pixels.
        step_sizes : float or iterable of floats, optional
            Step sizes in each map direction.
        """

        # Set rotations (always 1D, needed for masking)
        if not isinstance(rotations, Rotation):
            raise ValueError(
                f"rotations must be of type {Rotation}, not {type(rotations)}.")
        self._rotations = rotations.reshape(np.prod(rotations.shape))

        # Set phase ID
        self._phase_id = phase_id_map.astype(int)
        map_shape = phase_id_map.shape
        map_ndim = phase_id_map.ndim

        # Create phase list
        unique_phase_ids = np.unique(phase_id_map).astype(int)
        self.phases = PhaseList(
            names=phase_name,
            symmetries=symmetry,
            phase_ids=unique_phase_ids,
        )

        # Set step sizes
        if step_sizes is None:
            self._step_sizes = np.ones(map_ndim)
        elif isinstance(step_sizes, Number):
            # Assume same step size in all directions
            self._step_sizes = np.ones(map_ndim) * step_sizes
        else:
            self._step_sizes = np.array(step_sizes)

        # Set whether pixels are indexed
        if indexed is None:
            self._indexed = np.ones(map_shape, dtype=bool)
        else:
            self._indexed = indexed

        # Set scan unit
        self._scan_unit = 'px'

        # Set properties (calling prop.setter ensures reshaping to map shape)
        self.prop = {}
        if isinstance(prop, dict):
            self.prop = prop

    @property
    def phase_id(self):
        """Return numpy.ndarray of the phase ID of each map pixel."""
        if self.all_indexed is False:
            return np.ma.masked_array(self._phase_id, mask=~self._indexed)
        else:
            return self._phase_id

    @property
    def phases_in_map(self):
        """Return a list of phases in the map."""
        # Since self.phases might contain phases not in the map
        return self.phases[
            np.intersect1d(np.unique(self._phase_id), self.phases.phase_ids)
        ]

    @property
    def rotations(self):
        """Return a Rotation object, which is always 1D.

        Must always be 1D because of possible masked elements in indexed
        attribute.
        """
        return self._rotations[self.indexed.ravel()]

    @property
    def orientations(self):
        """Return an Orientation object, which is always 1D."""
        phases = self.phases_in_map
        if phases.size == 1:
            phase = phases[int(phases.phase_ids[0])]
            return Orientation(self.rotations).set_symmetry(phase.symmetry)
        else:
            raise ValueError(
                f"Map contains the phases {phases.names}, however, you are"
                " executing a command that only permits one phase."
            )

    @property
    def scan_unit(self):
        """Return scan unit as a string."""
        return self._scan_unit

    @scan_unit.setter
    def scan_unit(self, value):
        """Set scan unit as a string."""
        self._scan_unit = str(value)

    @property
    def prop(self):
        """Return a dict of properties of each pixel."""
        return self._prop

    @prop.setter
    def prop(self, value):
        """Add a dict of properties of each pixel."""
        reshaped_values = {}
        for k, v in value.items():
            if not self.all_indexed:
                v = np.ma.masked_array(v.ravel(), mask=~self.indexed.ravel())
            reshaped_values[k] = v.reshape(self.shape)
        self._prop = reshaped_values

    @property
    def step_sizes(self):
        """Return pixel step size in each direction in scan units."""
        return self._step_sizes

    @step_sizes.setter
    def step_sizes(self, value):
        """Set pixel step size in each direction in scan units."""
        if isinstance(value, Number):  # Same step size in all directions
            value = np.ones(self.ndim) * value
        self._step_sizes = np.array(value)

    @property
    def size(self):
        """Return total number of map pixels."""
        if not self.all_indexed:
            return self.phase_id[self.indexed].size
        else:
            return self.phase_id.size

    @property
    def shape(self):
        """Return shape of map in pixels."""
        return self.phase_id.shape

    @property
    def ndim(self):
        """Return number of map dimensions."""
        return len(self.shape)

    @property
    def indexed(self):
        """Return boolean numpy.ndarray with indexed pixels set to True."""
        return self._indexed

    @indexed.setter
    def indexed(self, value):
        """Set boolean numpy.ndarray with indexed pixels set to True."""
        self._indexed = np.array(value.reshape(self.shape), dtype=bool)

    @property
    def all_indexed(self):
        """Return whether all map pixels are indexed."""
        return np.count_nonzero(self._indexed) == self._indexed.size

    def __getattr__(self, item):
        """Return class attribute or property if the attribute (`item`) is
        in the `prop` dictionary.

        The property array is masked if the crystal map is masked.
        """
        prop = self._prop
        if item in prop:
            if self.all_indexed is False:
                return np.ma.masked_array(prop[item], mask=~self.indexed)
            else:
                return prop[item]
        else:  # Python's default behaviour when looking up attributes
            return self.__getattribute__(item)

    def __getitem__(self, key):
        """Return a new CrystalMap object.

        Parameters
        ----------
        key : str, slice or boolean numpy.ndarray
            If str, it must be a valid phase. If slice, it must be within
            the map shape. If boolean array, it must be of map shape.

        Examples
        --------
        A CrystalMap object can be indexed in multiple ways...

        >>> cm
        Phase  Orientations   Name       Symmetry  Color
        1      5657 (48.4%)   austenite  432       tab:blue
        2      6043 (51.6%)   ferrite    432       tab:orange
        Properties: iq, ci, fit
        Scan unit: um
        >>> cm.shape
        (100, 117)

        ... by slicing

        >>> cm2 = cm[20:40, 50:60]
        >>> cm2
        Phase  Orientations   Name       Symmetry  Color
        1      148 (74.0%)    austenite  432       lime
        2      52 (26.0%)     ferrite    432       r
        Properties: iq, ci, fit, ci_times_iq
        Scan unit: um
        >>> cm2.shape
        (20, 10)

        ... by phase name(s)

        >>> cm2 = cm["austenite"]
        Phase  Orientations   Name       Symmetry  Color
        1      5657 (100.0%)  austenite  432       tab:blue
        Properties: iq, ci, fit
        Scan unit: um
        >>> cm2.shape
        (100, 117)
        >>> cm["austenite", "ferrite"]
        Phase  Orientations   Name       Symmetry  Color
        1      5657 (48.4%)   austenite  432       tab:blue
        2      6043 (51.6%)   ferrite    432       tab:orange
        Properties: iq, ci, fit
        Scan unit: um

        ... or by boolean arrays ((chained) conditional(s))

        >>> cm[cm.ci > 0.81]
        Phase  Orientations   Name       Symmetry  Color
        1      4092 (44.8%)   austenite  432       tab:blue
        2      5035 (55.2%)   ferrite    432       tab:orange
        Properties: iq, ci, fit
        Scan unit: um
        >>> cm[(cm.iq > np.mean(cm.iq)) & (cm.phase_id == 1)]
        Phase  Orientations   Name       Symmetry  Color
        1      1890 (100.0%)  austenite  432       tab:blue
        Properties: iq, ci, fit
        Scan unit: um
        """

        # Get map shape and ndim
        map_shape = self.shape
        map_ndim = self.ndim

        # Set up necessary mask and slices
        mask = np.zeros(map_shape, dtype=bool)
        slices = list([0] * map_ndim)  # Ensure list to avoid potential errors

        if (
                isinstance(key, str) or
                (isinstance(key, tuple) and isinstance(key[0], str))
        ):
            # Get data from phase(s)
            if not isinstance(key, tuple):  # Make single string iterable
                key = (key,)
            for k in key:
                for phase_id, phase in self.phases_in_map:
                    if k == phase.name:
                        mask[self.phase_id == phase_id] = True
        elif isinstance(key, np.ndarray) and key.dtype == np.bool_:
            # Get data from boolean array
            key = key.reshape(map_shape)
            mask = key
        elif (
                isinstance(key, slice)
                or (
                        isinstance(key, tuple)
                        and any([(isinstance(i, slice)) for i in key])
                )
        ):
            # Get data from slice(s)
            if isinstance(key, slice):
                key = (key,)
            for i, k in enumerate(key):
                slices[i] = k
            mask[tuple(slices)] = True

        # Create slices if not created already
        if slices == list([0] * map_ndim):
            for i in range(map_ndim):
                dim_to_collapse = map_ndim - i - 1
                collapsed_dim = np.sum(mask, axis=dim_to_collapse)
                non_zero = np.nonzero(collapsed_dim)
                slices[i] = slice(np.min(non_zero), np.max(non_zero) + 1, None)
        slices = tuple(slices)

        # Keep all rotations within slice. Pixels within slice where mask is
        # False is masked out when calling self.rotations later
        r = self._rotations.reshape(*map_shape)  # Same shape as mask
        within_slice = np.zeros_like(mask)
        within_slice[slices] = True
        new_r = r[within_slice].flatten()  # 1D again

        # Create new crystal map
        new_map = CrystalMap(
            rotations=new_r,
            phase_id_map=self.phase_id[slices],
            prop={name: array[slices] for name, array in self.prop.items()},
            indexed=mask[slices],
            step_sizes=self.step_sizes,  # TODO: Slice when dimensions are lost
        )
        new_map.scan_unit = self.scan_unit

        # Get new phase list
        new_phase_ids = np.unique(self.phase_id[mask])
        new_phase_list = self.phases[new_phase_ids]
        new_map.phases = new_phase_list

        return new_map

    def __repr__(self):
        phases = self.phases
        phase_ids = self.phase_id[self.indexed]

        # Ensure attributes set to None are treated OK
        names = ['None' if not n else n for n in phases.names]
        symmetry_names = [
            'None' if not s else s.name for s in phases.symmetries]

        # Determine column widths
        unique_phases = np.unique(phase_ids)
        p_sizes = [np.where(phase_ids == i)[0].size for i in unique_phases]
        id_len = 5
        ori_len = max(max([len(str(p_size)) for p_size in p_sizes]) + 9, 13)
        name_len = max(max([len(n) for n in names]), 5)
        sym_len = max(max([len(sn) for sn in symmetry_names]), 8)
        col_len = max(max([len(i) for i in phases.colors]), 6)

        # Header (note the two-space spacing)
        representation = (
                "{:<{width}}  ".format("Phase", width=id_len)
                + "{:<{width}}  ".format("Orientations", width=ori_len)
                + "{:<{width}}  ".format("Name", width=name_len)
                + "{:<{width}}  ".format("Symmetry", width=sym_len)
                + "{:<{width}}\n".format("Color", width=col_len)
        )

        # Overview of data for each phase
        for i, phase_id in enumerate(unique_phases.astype(int)):
            p_size = np.where(phase_ids == phase_id)[0].size
            p_fraction = 100 * p_size / self._phase_id[self._indexed].size
            ori_str = f"{p_size} ({p_fraction:.1f}%)"
            representation += (
                    f"{phase_id:<{id_len}}  "
                    + f"{ori_str:<{ori_len}}  "
                    + f"{names[i]:<{name_len}}  "
                    + f"{symmetry_names[i]:<{sym_len}}  "
                    + f"{phases.colors[i]:<{col_len}}\n"
            )

        # Properties and spatial coordinates
        props = []
        for k in self._prop.keys():
            props.append(k)
        representation += "Properties: " + ", ".join(props) + "\n"

        # Scan unit
        representation += f"Scan unit: {self._scan_unit}"

        return representation

    def deepcopy(self):
        """Return a deep copy using :py:func:`~copy.deepcopy` function."""
        return copy.deepcopy(self)
