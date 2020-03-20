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
import matplotlib.patches as mpatches

from orix.quaternion.rotation import Rotation
from orix.quaternion.orientation import Orientation
from .phase_list import PhaseList
from orix.plot import _plot_crystal_map


class CrystalMap:
    """Crystallographic map storing rotations, phases and pixel properties.

    Phases and pixel properties are stored in the map shape, while
    rotations are always stored as a 1D array.

    This class is inspired by the EBSD class available in MTEX
    [Bachmann2010]_.

    Attributes
    ----------
    all_indexed : bool
        Whether all pixels are indexed.
    dx : numpy.ndarray
        Step sizes in each map direction.
    indexed : numpy.ndarray
        Boolean array with True for indexed pixels.
    ndim : int
        Number of map dimensions.
    orientations : orix.quaternion.orientation.Orientation
        Orientations of each pixel. Always 1D.
    phase_id : numpy.ndarray
        Phase ID of each pixel as imported.
    phases : orix.crystal_map.PhaseList
        List of phases with their IDs, names, crystal symmetries and
        colors (possibly more than are in the map).
    phases_in_map : orix.crystal_map.PhaseList
        List of phases in the map, with their IDs, names, crystal
        symmetries and colors.
    pixel_id : numpy.ndarray
        ID of each map pixel.
    prop : dict
        Dictionary of numpy arrays of quality metrics or other properties
        of each pixel.
    rotations : orix.quaternion.rotation.Rotation
        Rotations of each pixel. Always 1D.
    scan_unit : str
        Length unit of map, default is 'um'.
    shape : tuple
        Shape of map in pixels.
    size : int
        Number of pixels in map.

    Methods
    -------
    deepcopy()
        Return a deep copy using :py:func:`~copy.deepcopy` function.
    plot_prop(
        prop, colorbar=True, scalebar=True, padding=False, **kwargs)
        Plot of a map property.
    plot_phase(
        overlay=None, legend=True, scalebar=True, padding=False,
        **kwargs)
        Return and plot map phases.

    References
    ----------
    .. [Bachmann2010] F. Bachmann, R. Hielscher, H. Schaeben, "Texture\
        Analysis with MTEX – Free and Open Source Software Toolbox," Solid
        State Phenomena 160, 63–68, 2010.

    """

    def __init__(
            self,
            rotations=None,
            phase_id=None,
            phase_name=None,
            symmetry=None,
            prop=None,
            indexed=None,
            dx=None,
    ):
        """
        Parameters
        ----------
        rotations : numpy.ndarray, optional
            Rotation of each pixel.
        phase_id : numpy.ndarray, optional
            Phase ID of each pixel.
        phase_name : str or list of str, optional
            Name of phases.
        symmetry : str or list of str, optional
            Point group of crystal symmetries of phases in the map.
        prop : dict of numpy.ndarray, optional
            Dictionary of quality metrics or other properties of each
            pixel.
        indexed : numpy.ndarray
            Boolean array with True for indexed pixels.
        dx : numpy.ndarray, optional
            Step sizes in each map direction.
        """

        if phase_id is None:
            phase_id = np.arange(rotations.size, dtype=int)

        # Set phase ID
        self._phase_id = phase_id.astype(int)

        # Set map size, shape and number of dimensions
        map_shape = phase_id.shape
        map_ndim = phase_id.ndim

        # Set rotations (always 1D, needed for masking)
        if rotations is not None and not isinstance(rotations, Rotation):
            try:
                rotations = Rotation.from_euler(rotations)
            except (IndexError, TypeError):
                # TODO: Update error message with more detailed restrictions
                raise ValueError(
                    f"Rotations '{rotations}' must be of type numpy.ndarray."
                )
        self._rotations = rotations.reshape(np.prod(rotations.shape))

        # Set step sizes
        if dx is None:
            self._dx = np.ones(map_ndim)
        elif isinstance(dx, Number):
            # Assume same step size in all directions
            self._dx = np.ones(map_ndim) * dx
        elif len(dx) != map_ndim:
            raise ValueError(
                f"{dx} must have same number of entries as number of map "
                f"dimensions {map_ndim}"
            )
        else:
            self._dx = dx

        # Set whether pixels are indexed
        if indexed is None:
            self._indexed = np.ones(map_shape, dtype=bool)
        else:
            self._indexed = indexed

        # Create phase list
        phase_ids = np.unique(phase_id).astype(int)
        self._phases = PhaseList(
            names=phase_name,
            symmetries=symmetry,
            phase_ids=phase_ids,
        )

        # Set scan unit
        self._scan_unit = 'um'

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
    def phases(self):
        """Return a list of phases in the map (and potentially more)."""
        return self._phases

    @property
    def phases_in_map(self):
        """Return a list of phases in the map."""
        # Since self.phases might contain phases not in the map
        return self._phases[
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
        phases = self._phases
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
        e = (
            f"{value} must be a dict with strings as keys and numpy.ndarrays as"
            f" values of same size {self.size} as the map."
        )
        if not isinstance(value, dict):
            raise ValueError(e)

        reshaped_values = {}
        map_size = self.size
        map_shape = self.shape
        for k, v in value.items():
            if not self.all_indexed:
                v = np.ma.masked_array(v.ravel(), mask=~self.indexed.ravel())
                v_size = v.compressed().size
            else:
                v_size = v.size
            if isinstance(k, str) and v_size == map_size:
                reshaped_values[k] = v.reshape(map_shape)
            else:
                raise ValueError(e)

        self._prop = reshaped_values

    @property
    def pixel_id(self):
        """Return map pixel IDs as a numpy.ndarray of same shape as map."""
        pixel_id = np.arange(self.size).reshape(self.shape)
        if self.all_indexed is False:
            return np.ma.masked_array(pixel_id, mask=~self.indexed)
        else:
            return pixel_id

    @property
    def dx(self):
        """Return pixel step size in each direction in scan units."""
        return self._dx

    @dx.setter
    def dx(self, value):
        """Set pixel step size in each direction in scan units."""
        ndim = self.ndim
        if isinstance(value, Number):
            # Assume same step size in all directions
            self._dx = np.zeros(ndim) * value
        elif len(value) != ndim:
            raise ValueError(
                f"{value} must have same number of entries as number of map "
                f"dimensions {ndim}."
            )
        else:
            self._dx = np.array(value)

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
        if (
                isinstance(value, np.ndarray)
                and value.dtype == np.bool_
                and value.size == self.size
        ):
            # Reshape if not same shape as map
            self._indexed = value.reshape(self.shape)
        else:
            raise ValueError(
                f"{value} must be a boolean array of same size as the map, "
                f"{self.size}."
            )

    @property
    def all_indexed(self):
        """Return whether all map pixels are indexed."""
        return np.count_nonzero(self._indexed) == self._indexed.size

    def __getattr__(self, item):
        """Return class attribute or prop if item is in prop dict."""
        prop = self.__getattribute__('_prop')
        if item in prop:
            if self.all_indexed is False:
                return np.ma.masked_array(prop[item], mask=~self.indexed)
            else:
                return prop[item]
        else:  # Default behaviour
            return self.__getattribute__(item)

    def __getitem__(self, key):
        # Get map shape, size and ndim once
        map_shape = self.shape
        map_size = self.size
        map_ndim = self.ndim

        # Set up necessary slices
        slices = list([0] * map_ndim)  # Ensure list to avoid potential errors

        # Create mask
        mask = np.zeros(map_shape, dtype=bool)

        if (
                isinstance(key, str) or
                (isinstance(key, tuple) and isinstance(key[0], str))
        ):
            # Get data from phase(s)
            if not isinstance(key, tuple):  # Make single string iterable
                key = (key,)

            for k in key:
                match = False
                for phase_id, phase in self.phases_in_map:
                    if k == phase.name:
                        mask[self.phase_id == phase_id] = True
                        match = True
                if match is False:
                    raise IndexError(
                        f"{k} is not among the available phases "
                        f"{self.phases_in_map.names} in the map."
                    )
        elif isinstance(key, np.ndarray) and key.dtype == np.bool_:
            # Get data from conditional(s)
            if key.size == map_size:
                key = key.reshape(map_shape)
            else:
                raise IndexError(
                    f"{key} must be of the same size as the map ({map_size}), "
                    f"but is instead {key.size}."
                )
            if np.count_nonzero(key) == 0:
                raise IndexError(f"Indexing condition match no map pixels.")
            mask = key
        elif (
                isinstance(key, slice)
                or (
                        isinstance(key, tuple)
                        and any([(isinstance(i, slice)) for i in key])
                )
        ):
            # Get data from slice(s)
            if isinstance(key, tuple):
                n_slices = len(key)
                if n_slices > map_ndim:
                    raise IndexError(
                        f"Cannot slice {n_slices} dimensions when the map has "
                        f"only {map_ndim}."
                    )

            # Overwrite entries in slices list
            if isinstance(key, slice):
                key = (key,)
            for i, k in enumerate(key):
                slices[i] = k
            slices = tuple(slices)

            # Slice mask
            mask[slices] = True
        else:
            raise IndexError(
                f"{key} must be a valid str, slice or boolean array."
            )

        # Create slices if not created already
        if slices == list([0] * map_ndim):
            for i in range(map_ndim):
                dim_to_collapse = map_ndim - i - 1
                collapsed_dim = np.sum(mask, axis=dim_to_collapse)
                non_zero = np.nonzero(collapsed_dim)
                slices[i] = slice(np.min(non_zero), np.max(non_zero) + 1, None)
            slices = tuple(slices)

        # Create new crystal map
        new_map = CrystalMap(
            rotations=self._rotations,  # Is sliced when calling self.rotations
            phase_id=self.phase_id[slices],
            prop={name: array[slices] for name, array in self.prop.items()},
            indexed=mask[slices],
            dx=self.dx,  # TODO: Slice when dimensions are lost
        )

        # Get new phase list
        new_phase_ids = np.unique(self.phase_id[mask])
        new_phase_list = self.phases[new_phase_ids]
        new_map._phases = new_phase_list

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

    def plot_phase(
            self,
            overlay=None,
            legend=True,
            scalebar=True,
            padding=True,
            **kwargs,
    ):
        """Return and plot map phases.

        Parameters
        ----------
        overlay : str, optional
            Map property to use as alpha value. The property is adjusted
            for maximum contrast.
        legend : bool, optional
            Whether to display a legend with phases in the map (default is
            ``True``).
        scalebar : bool, optional
            Whether to add a scalebar (default is ``True``) along the last
            map dimension.
        padding : bool, optional
            Whether to show white padding (default is ``True``). Setting
            this to false removes all white padding outside of plot,
            including pixel coordinate ticks.
        **kwargs :
            Optional keyword arguments passed to
            :meth:`matplotlib.pyplot.imshow`.

        Returns
        -------
        phase : numpy.ndarray
            Phase array as passed to :meth:`matplotlib.pyplot.imshow` with
            colors and potential alpha value if a valid `overlay` argument
            was passed.
        fig : matplotlib.figure.Figure
            Top level container for all plot elements.
        ax : matplotlib.axes.Axes
            Axes object returned by :meth:`matplotlib.pyplot.subplots`.
        im : matplotlib.image.AxesImage
            Image object returned by :meth:`matplotlib.axes.Axes.imshow`.

        Examples
        --------
        >>> cm
        Phase  Orientations   Name       Symmetry  Color
        1      5657 (48.4%)   austenite  432       tab:blue
        2      6043 (51.6%)   ferrite    432       tab:orange
        Properties: iq, ci, fit
        Scan unit: um
        >>> phase, fig, ax, im = cm.plot_phase(overlay='ci')
        """

        # Create 1D phase array and add RGB channels to each map pixel
        n_channels = 3
        phase = np.ones((np.prod(self.shape), n_channels))

        # Color each map pixel with corresponding phase color RGB tuple
        phase_id = self.phase_id.ravel()
        for i, color in zip(np.unique(phase_id), self.phases.colors_rgb):
            mask = phase_id == i
            phase[mask] = phase[mask] * color

        # Scale RGB values with gray scale from property
        if overlay:
            if overlay not in self.prop.keys():
                raise ValueError(
                    f"{overlay} is not among available properties "
                    f"{list(self.prop.keys())}."
                )
            else:
                prop = self.prop[overlay].ravel()

                # Scale prop to [0, 1] to maximize image contrast
                prop_min = prop.min()
                prop = (prop - prop_min) / (prop.max() - prop_min)
                for i in range(n_channels):
                    phase[:, i] *= prop

        # Create legend patches with phase color and name
        patches = None
        if legend:
            patches = [
                mpatches.Patch(color=p.color_rgb, label=p.name)
                for _, p in self.phases_in_map
            ]

        # Reshape phase map to 2D + RGB channels
        phase = phase.reshape(self.shape + (n_channels,))

        # Set non-indexed points to white (or None for black)
        phase[~self.indexed] = (1, 1, 1)

        fig, ax, im = _plot_crystal_map(
            crystal_map=self,
            data=phase,
            legend_patches=patches,
            scalebar=scalebar,
            padding=padding,
            **kwargs,
        )

        # Should find some way to mute other than cm.plot_phase();
        return phase, fig, ax, im

    def plot_prop(
            self,
            prop,
            colorbar=True,
            scalebar=True,
            padding=True,
            **kwargs,
    ):
        """Plot of a map property.

        Parameters
        ----------
        prop : str
            The property in `prop` to plot.
        colorbar : bool, optional
            Whether to add a colorbar (default is ``True``).
        scalebar : bool, optional
            Whether to add a scalebar (default is ``True``) along the last
            map dimension.
        padding : bool, optional
            Whether to show white padding (default is ``True``). Setting
            this to ``False`` removes all white padding outside of plot,
            including pixel coordinate ticks.
        **kwargs :
            Optional keyword arguments passed to
            :meth:`matplotlib.pyplot.imshow`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Top level container for all plot elements.
        ax : matplotlib.axes.Axes
            Axes object returned by :meth:`matplotlib.pyplot.subplots`.
        im : matplotlib.image.AxesImage
            Image object returned by :meth:`matplotlib.axes.Axes.imshow`.
        prop_to_plot : numpy.ndarray
            Property array as passed to :meth:`matplotlib.pyplot.imshow`.

        Examples
        --------
        >>> cm
        Phase  Orientations   Name       Symmetry  Color
        1      5657 (48.4%)   austenite  432       tab:blue
        2      6043 (51.6%)   ferrite    432       tab:orange
        Properties: iq, ci, fit
        Scan unit: um
        >>> data, fig, ax, im = cm.plot_prop('ci')
        """

        if prop not in self.prop:
            raise ValueError(
                f"{prop} is not among available properties "
                f"{list(self.prop.keys())}."
            )
        else:
            prop_to_plot = self.prop[prop]

        fig, ax, im = _plot_crystal_map(
            crystal_map=self,
            data=prop_to_plot,
            scalebar=scalebar,
            cmap=kwargs.pop('cmap', 'gray'),
            padding=padding,
            **kwargs,
        )

        # Colorbar
        if colorbar:
            fig.colorbar(im, ax=ax)

        # Should find some way to mute other than cm.plot_prop('ci');
        return prop_to_plot, fig, ax, im
