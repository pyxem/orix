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

import warnings

import numpy as np
import matplotlib
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.axes import Axes
from matplotlib import projections
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

rcParams = matplotlib.rcParams


class CrystalMapPlot(Axes):
    """A base class for plotting of CrystalMap objects."""

    name = None
    _property_name = None

    def _plot_crystal_map(
            self,
            crystal_map,
            patches=None,
            scalebar=True,
            scalebar_properties=None,
            legend_properties=None,
            **kwargs,
    ):
        if scalebar:
            self.add_scalebar(crystal_map, scalebar_properties)

        if patches:
            d = {
                "borderpad": 0.3,
                "handlelength": 0.75,
                "handletextpad": 0.3,
                "framealpha": 0.6,
                "prop": fm.FontProperties(size=11),
            }
            if legend_properties is None:
                legend_properties = {}
            [legend_properties.setdefault(k, v) for k, v in d.items()]
            self.legend(handles=patches, **legend_properties)

        data = kwargs.pop("X")  # Extract for code clarity
        im = super().imshow(X=data, **kwargs)

        im = self._override_status_bar(im, crystal_map)

        return im

    def _override_status_bar(self, image, crystal_map):
        """Display coordinates and Euler angles per pixel in status bar.

         This is done by overriding
         :meth:`matplotlib.images.AxesImage.get_cursor_data`,
         :meth:`matplotlib.images.AxesImage.format_cursor_data` and
         :meth:`matplotlib.axes.Axes.format_coord`.

        Parameters
        ----------
        image : matplotlib.images.AxesImage
            Image object.
        crystal_map : orix.crystal_map.CrystalMap
            Crystal map object to obtain necessary data from.

        Returns
        -------
        image : matplotlib.images.AxesImage
            Image object where the above mentioned methods are overridden.
        """
        # Get rotations in radians, ensuring proper masking of pixels
        r = crystal_map._rotations.to_euler()
        r = np.round(r, decimals=3)
        r[~crystal_map.indexed.ravel()] = None

        n_rows, n_cols = crystal_map.shape

        im_data = image.get_array()

        def status_bar_data(event):
            col = int(event.xdata + 0.5)
            row = int(event.ydata + 0.5)
            return row, col, r[col + (row * n_cols)], im_data[row, col]

        # Set width of x and y fields in plotting status bar
        x_width = len(str(n_cols - 1))
        y_width = len(str(n_rows - 1))

        if self.name == "property_map":
            def format_status_bar_data(data):
                return (
                    f"(y,x):({data[0]:{y_width}},{data[1]:{x_width}})"
                    f" {self._property_name}:{data[3]}"
                    f" rot:({data[2][0]:5},{data[2][1]:5},{data[2][2]:5})"
                )
        else:
            def format_status_bar_data(data):
                return (
                    f"(y,x):({data[0]:{y_width}},{data[1]:{x_width}})"
                    f" rot:({data[2][0]:5},{data[2][1]:5},{data[2][2]:5})"
                )

        # Override
        image.get_cursor_data = status_bar_data
        image.format_cursor_data = format_status_bar_data
        self.axes.format_coord = lambda x, y: ""

        return image

    def add_scalebar(self, crystal_map, scalebar_properties):
        """Add a scalebar to the axes object via `AnchoredSizeBar`.

        To find an appropriate scalebar width, this snippet from MTEX
        [Bachmann2010]_ written by Eric Payton and Philippe Pinard is used:
        https://github.com/mtex-toolbox/mtex/blob/b8fc167d06d453a2b3e212b1ac383acbf85a5a27/plotting/scaleBar.m,

        Parameters
        ----------
        crystal_map : orix.crystal_map.CrystalMap
            Crystal map object to obtain necessary data from.
        scalebar_properties : dict
            Keyword arguments passed to
            :func:`mpl_toolkits.axes_grid1.anchored_artists.AnchoredSizeBar`.

        Examples
        --------
        >>> cm
        Phase  Orientations   Name       Symmetry  Color
        1      5657 (48.4%)   austenite  432       tab:blue
        2      6043 (51.6%)   ferrite    432       tab:orange
        Properties: iq, ci, fit
        Scan unit: um
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(projection="phase_map")
        >>> im = ax.phase_map(cm, scalebar=False)
        >>> ax.add_scalebar()
        """
        map_width = crystal_map.shape[-1]
        step_size = crystal_map.step_sizes[-1]
        scan_unit = crystal_map.scan_unit

        # Initial scalebar width should be approximately 1/10 of map width
        scalebar_width = 0.1 * (map_width * step_size)

        # Ensure a suitable number is used, e.g. going from 1000 nm to 1 um
        scalebar_width, scan_unit, factor = convert_unit(
            scalebar_width, scan_unit)

        # This snippet for finding a suitable scalebar width is taken from MTEX:
        # https://github.com/mtex-toolbox/mtex/blob/b8fc167d06d453a2b3e212b1ac383acbf85a5a27/plotting/scaleBar.m,
        # written by Eric Payton and Philippe Pinard.
        # We want a round, not too high number without decimals
        good_values = np.array(
            [1, 2, 5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 200, 500, 750],
            dtype=int,
        )
        # Find good data closest to initial scalebar width
        difference = abs(scalebar_width - good_values)
        good_value_idx = np.where(difference == difference.min())[0][0]
        scalebar_width = good_values[good_value_idx]

        # Scale width by factor from above conversion (usually factor = 1.0)
        scalebar_width = scalebar_width * factor
        scalebar_width_px = scalebar_width / step_size

        # Allow for a potential decimal in scalebar number if something didn't
        # go as planned
        if scalebar_width.is_integer():
            scalebar_width = int(scalebar_width)
        else:
            warnings.warn(f"Scalebar width {scalebar_width} is not an integer.")

        if scan_unit == 'um':
            scan_unit = "\u03BC" + "m"

        # Set up arguments to AnchoredSizeBar() if not already present in kwargs
        d = {
            "loc": 3,
            "pad": 0.2,
            "sep": 3,
            "borderpad": 0.5,
            "size_vertical": 1,
            "fontproperties": fm.FontProperties(size=11),
        }
        if scalebar_properties is None:
            scalebar_properties = {}
        [scalebar_properties.setdefault(k, v) for k, v in d.items()]

        # Create scalebar
        bar = AnchoredSizeBar(
            transform=self.axes.transData,
            size=scalebar_width_px,
            label=str(scalebar_width) + ' ' + scan_unit,
            **scalebar_properties,
        )
        bar.patch.set_alpha(0.6)

        self.axes.add_artist(bar)

    def remove_padding(self):
        """Remove all white padding outside of the figure.

        Examples
        --------
        >>> cm
        Phase  Orientations   Name       Symmetry  Color
        1      5657 (48.4%)   austenite  432       tab:blue
        2      6043 (51.6%)   ferrite    432       tab:orange
        Properties: iq, ci, fit
        Scan unit: um
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(projection="phase_map")
        >>> ax.phase_map(cm)
        >>> ax.remove_padding()
        """
        self.set_axis_off()
        self.margins(0, 0)

        # Tune subplot layout
        colorbar = self.images[0].colorbar
        if colorbar is not None:
            right = self.figure.subplotpars.right
        else:
            right = 1
        self.figure.subplots_adjust(top=1, bottom=0, right=right, left=0)


class PhaseMap(CrystalMapPlot):

    name = "phase_map"

    def phase_map(
            self,
            crystal_map,
            scalebar=True,
            scalebar_properties=None,
            legend_properties=None,
            **kwargs,
    ):
        """Plot a map of phases with a scalebar and legend.

        Wraps :meth:`matplotlib.axes.Axes.imshow`, see that method for
        relevant keyword arguments.

        Parameters
        ----------
        crystal_map : orix.crystal_map.CrystalMap
            Crystal map object to obtain data to plot from.
        scalebar : bool, optional
            Whether to add a scalebar (default is ``True``) along the last
            map dimension.
        scalebar_properties : dict
            Dictionary of keyword arguments passed to
            :func:`mpl_toolkits.axes_grid1.anchored_artists.AnchoredSizeBar`.
        legend_properties : dict
            Dictionary of keyword arguments passed to
            :meth:`matplotlib.axes.legend`.
        kwargs :
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.imshow`.

        Returns
        -------
        im : matplotlib.image.AxesImage
            Image object.

        See Also
        --------
        matplotlib.axes.Axes.imshow
        orix.plot.crystal_map_plot.CrystalMapPlot._add_scalebar
        """

        # Create data array
        n_channels = 3  # RGB
        map_shape = crystal_map.shape
        data = np.ones((np.prod(map_shape), n_channels))

        # Color each map pixel with corresponding phase color RGB tuple
        phase_id = crystal_map.phase_id.ravel()
        for i, color in zip(np.unique(phase_id), crystal_map.phases.colors_rgb):
            mask = phase_id == i
            data[mask] = data[mask] * color

        # Reshape to 2D array
        data = data.reshape(map_shape + (n_channels,))

        # Legend
        patches = []
        for _, p in crystal_map.phases_in_map:
            patches.append(mpatches.Patch(color=p.color_rgb, label=p.name))

        return super()._plot_crystal_map(
            crystal_map,
            patches=patches,
            scalebar=scalebar,
            scalebar_properties=scalebar_properties,
            legend_properties=legend_properties,
            X=data,
            **kwargs
        )

    def add_overlay(self, crystal_map, property_name):
        """Use a crystal map property as gray scale values of a phase map.

        The property's range is adjusted to [0, 1] for maximum contrast.

        Parameters
        ----------
        crystal_map : orix.crystal_map.CrystalMap
            Crystal map object to obtain necessary data from.
        property_name : str
            Name of map property to scale phase array with. The property
            range is adjusted for maximum contrast.

        Examples
        --------
        >>> cm
        Phase  Orientations   Name       Symmetry  Color
        1      5657 (48.4%)   austenite  432       tab:blue
        2      6043 (51.6%)   ferrite    432       tab:orange
        Properties: iq, ci, fit
        Scan unit: um
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(projection="phase_map")
        >>> im = ax.phase_map(cm)
        >>> ax.add_overlay(cm, "ci")
       """
        im = self.images[0]
        data = im.get_array()

        properties = crystal_map.prop
        if property_name not in properties.keys():
            raise ValueError(
                f"{property_name} is not among available properties "
                f"{list(properties.keys())}."
            )
        prop = properties[property_name]

        # Scale prop to [0, 1] to maximize image contrast
        prop_min = prop.min()
        prop = (prop - prop_min) / (prop.max() - prop_min)

        n_channels = 3
        for i in range(n_channels):
            data[:, :, i] *= prop

        # Set non-indexed points to white (or None for black)
        data[~crystal_map.indexed] = (1, 1, 1)

        data = data.reshape(crystal_map.shape + (n_channels,))
        im.set_data(data)


class PropertyMap(CrystalMapPlot):

    name = "property_map"

    def property_map(
            self,
            crystal_map,
            property_name,
            scalebar=True,
            scalebar_properties=None,
            **kwargs,
    ):
        """Plot a map of a property with a scalebar.

        Wraps :meth:`matplotlib.axes.Axes.imshow`, see that method for
        relevant keyword arguments.

        Parameters
        ----------
        crystal_map : orix.crystal_map.CrystalMap
            Crystal map object to obtain data to plot from.
        property_name : str
            Name of map property to plot.
        scalebar : bool, optional
            Whether to add a scalebar (default is ``True``) along the last
            map dimension.
        scalebar_properties : dict
            Dictionary of keyword arguments passed to
            :func:`mpl_toolkits.axes_grid1.anchored_artists.AnchoredSizeBar`.
        kwargs :
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.imshow` and
            :meth:`~orix.plot.crystal_map_plot.CrystalMapPlot._add_scalebar`.

        Returns
        -------
        im : matplotlib.image.AxesImage
            Image object.

        See Also
        --------
        matplotlib.axes.Axes.imshow
        orix.plot.crystal_map_plot.CrystalMapPlot._add_scalebar

        Examples
        --------
        >>> cm
        Phase  Orientations   Name       Symmetry  Color
        1      5657 (48.4%)   austenite  432       tab:blue
        2      6043 (51.6%)   ferrite    432       tab:orange
        Properties: iq, ci, fit
        Scan unit: um
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(projection="property_map")
        >>> im = ax.property_map(cm, "ci")
        """
        properties = crystal_map.prop
        if property_name not in properties.keys():
            raise ValueError(
                f"{property_name} is not among available properties "
                f"{list(properties.keys())}."
            )
        data = properties[property_name]

        self._property_name = property_name

        return super()._plot_crystal_map(
            crystal_map,
            scalebar=scalebar,
            scalebar_properties=scalebar_properties,
            X=data,
            cmap=kwargs.pop("cmap", "gray"),
            **kwargs,
        )

    def add_colorbar(self, title=None, **kwargs):
        """Add an opinionated colorbar to the figure.

        Parameters
        ----------
        title : str, optional
            Colorbar title, default is ``None``.
        kwargs :
            Keyword arguments passed to
            :meth:`mpl_toolkits.axes_grid1.make_axes_locatable.append_axes`.

        Examples
        --------
        >>> cm
        Phase  Orientations   Name       Symmetry  Color
        1      5657 (48.4%)   austenite  432       tab:blue
        2      6043 (51.6%)   ferrite    432       tab:orange
        Properties: iq, ci, fit
        Scan unit: um
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(projection="property_map")
        >>> im = ax.property_map(cm, "ci")
        >>> ax.add_colorbar("Confidence index")

        If the default options are not as desired, a colorbar can be added
        and modified using, instead of the above line:

        >>> cbar = fig.colorbar(im, ax=ax)
        >>> cbar.ax.set_ylabel(title="ci", rotation=0)
        """

        # Keyword arguments
        d = {"position": "right", "size": "5%", "pad": 0.1}
        [kwargs.setdefault(k, v) for k, v in d.items()]

        # Add colorbar
        divider = make_axes_locatable(self)
        cax = divider.append_axes(**kwargs)
        cbar = self.figure.colorbar(self.images[0], cax=cax)

        # Set title with padding
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(title)


projections.register_projection(PhaseMap)
projections.register_projection(PropertyMap)


def convert_unit(value, unit):
    """Return the data with a suitable, not too large, unit.

    This algorithm is taken directly from MTEX [Bachmann2010]_
    https://github.com/mtex-toolbox/mtex/blob/a74545383160610796b9525eedf50a241800ffae/plotting/plotting_tools/switchUnit.m,
    written by Ralf Hielscher.

    Parameters
    ----------
    value : float
        The data to convert.
    unit : str
        The data unit, e.g. um.

    Returns
    -------
    new_value : float
        The input data converted to the suitable unit.
    new_unit : str
        A more suitable unit than the input.
    factor : float
        Factor to multiple `new_value` with to get the input data.

    Examples
    --------
    >>> convert_unit(17.55 * 1e3, 'nm')
    17.55 um 999.9999999999999
    >>> convert_unit(17.55 * 1e-3, 'mm')
    17.55 um 0.001

    """

    # If unit is 'px', we assume 'um', and revert unit in the end
    unit_is_px = False
    if unit == 'px':
        unit = 'um'
        unit_is_px = True

    # Create lookup-table with units and power
    lookup_table = []
    letters = 'yzafpnum kMGTPEZY'
    new_unit_idx = None
    for i, letter in enumerate(letters):
        # Ensure 'm' is entered correctly
        current_unit = (letter + 'm').strip(' ')
        lookup_table.append((current_unit, 10 ** (3 * i - 24)))
        if unit == current_unit:
            new_unit_idx = i

    # Find the lookup-table index of the most suitable unit
    value_in_metres = value * lookup_table[new_unit_idx][1]
    power_of_value = np.floor(np.log10(value_in_metres))
    suitable_unit_idx = int(np.floor(power_of_value / 3) + 8)

    # Calculate new data, unit and the conversion factor
    new_value = value_in_metres / lookup_table[suitable_unit_idx][1]
    new_unit = lookup_table[suitable_unit_idx][0]
    factor = lookup_table[suitable_unit_idx][1] / lookup_table[new_unit_idx][1]

    if unit_is_px:
        new_unit = 'px'

    return new_value, new_unit, factor
