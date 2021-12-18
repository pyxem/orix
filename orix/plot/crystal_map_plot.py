# -*- coding: utf-8 -*-
# Copyright 2018-2021 the orix developers
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

import matplotlib.patches as mpatches
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.projections import register_projection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar import dimension, scalebar
import numpy as np

from orix.scalar import Scalar
from orix.vector import Vector3d


class CrystalMapPlot(Axes):
    """2D plotting of :class:`~orix.crystal_map.crystal_map.CrystalMap`
    objects.
    """

    name = "plot_map"
    _data_axes = None
    _data_slices = None
    _data_shape = None
    colorbar = None
    scalebar = None

    def plot_map(
        self,
        crystal_map,
        value=None,
        scalebar=True,
        scalebar_properties=None,
        legend=True,
        legend_properties=None,
        axes=None,
        depth=None,
        override_status_bar=False,
        **kwargs,
    ) -> AxesImage:
        r"""Plot a 2D map with any CrystalMap attribute as map values.

        Wraps :meth:`matplotlib.axes.Axes.imshow`, see that method for
        relevant keyword arguments.

        Parameters
        ----------
        crystal_map : orix.crystal_map.CrystalMap
            Crystal map object to obtain data to plot from.
        value : numpy.ndarray, optional
            Attribute array to plot. If value is None (default), a phase
            map is plotted.
        scalebar : bool, optional
            Whether to add a scalebar (default is True) along the
            horizontal map dimension.
        scalebar_properties : dict
            Dictionary of keyword arguments passed to
            :func:`mpl_toolkits.axes_grid1.anchored_artists.AnchoredSizeBar`.
        legend : bool, optional
            Whether to add a legend to the plot. This is only
            implemented for a phase plot (in which case default is
            True).
        legend_properties : dict
            Dictionary of keyword arguments passed to
            :meth:`matplotlib.axes.legend`.
        axes : tuple of ints, optional
            Which data axes to plot if data has more than two
            dimensions. The index of data to plot in the final dimension
            is determined by `depth`. If None (default), data along the
            two last axes is plotted.
        depth : int, optional
            Which layer along the third axis to plot if data has more
            than two dimensions. If None (default), data in the first
            index (layer) is plotted.
        override_status_bar : bool, optional
            Whether to display Euler angles and any overlay values in
            the status bar when hovering over the map (default is
            False).
        kwargs
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.imshow`.

        Returns
        -------
        im : matplotlib.image.AxesImage
            Image object, to be used further to get data from etc.

        See Also
        --------
        matplotlib.axes.Axes.imshow
        orix.plot.CrystalMapPlot.add_scalebar
        orix.plot.CrystalMapPlot.add_overlay
        orix.plot.CrystalMapPlot.add_colorbar

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from orix import plot
        >>> from orix.io import load

        Import a crystal map

        >>> xmap = load("/some/directory/data.ang")

        Plot a phase map

        >>> fig = plt.figure()  # Get figure
        >>> ax = fig.add_subplot(projection="plot_map")  # Get axes
        >>> im = ax.plot_map(xmap)  # Get image

        Add an overlay

        >>> ax.add_overlay(xmap, xmap.iq)

        Plot an arbitrary map property, also changing scalebar location

        >>> ax = plt.subplot(projection="plot_map")
        >>> ax.plot_map(
        ...     xmap, xmap.dp, cmap="cividis", scalebar_properties={"loc": 4}
        ... )

        Add a colorbar

        >>> cbar = ax.add_colorbar("Dot product")  # Get colorbar

        Plot orientation angle in degrees of one phase

        >>> xmap2 = xmap["austenite"]
        >>> austenite_angles = xmap2.orientations.angle.data * 180 / np.pi
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(projection="plot_map")
        >>> im = ax.plot_map(xmap2, austenite_angles)
        >>> ax.add_colorbar("Orientation angle [$^{\circ}$]")

        Remove all figure and axes padding

        >>> ax.remove_padding()

        Write annotated figure to file

        >>> fig.savefig(
        ...     "/some/directory/image.png",
        ...     pad_inches=0,
        ...     bbox_inches="tight"
        ... )

        Write un-annotated image to file

        >>> plt.imsave("/some/directory/image2.png", im.get_array())
        """
        self._set_plot_shape(crystal_map=crystal_map, axes=axes, depth=depth)

        patches = None
        if value is None:  # Phase map
            # Color each map pixel with corresponding phase color RGB tuple
            phase_id = crystal_map.get_map_data("phase_id")
            phase_id = phase_id[self._data_slices]
            unique_phase_ids = np.unique(phase_id[~np.isnan(phase_id)])
            data = np.ones(phase_id.shape + (3,))
            for i, color in zip(
                unique_phase_ids, crystal_map.phases_in_data.colors_rgb
            ):
                mask = phase_id == int(i)
                data[mask] = data[mask] * color

            # Add legend patches to plot
            patches = []
            for _, p in crystal_map.phases_in_data:
                patches.append(mpatches.Patch(color=p.color_rgb, label=p.name))
        else:  # Create masked array of correct shape
            if isinstance(value, (Scalar, Vector3d)):
                value = value.data
            data = crystal_map.get_map_data(value)
            data = data[self._data_slices]

        # Squeeze 1-dimensions
        data = np.squeeze(data)

        # Legend
        if legend and isinstance(patches, list):
            if legend_properties is None:
                legend_properties = {}
            self._add_legend(patches, **legend_properties)

        # Scalebar
        if scalebar:
            if scalebar_properties is None:
                scalebar_properties = {}
            _ = self.add_scalebar(crystal_map, **scalebar_properties)

        im = self.imshow(X=data, **kwargs)
        if override_status_bar:
            im = self._override_status_bar(im, crystal_map)

        return im

    def add_scalebar(self, crystal_map, **kwargs):
        """Add a scalebar to the axes instance and return it.

        The scalebar is also available as an attribute :attr:`scalebar`.

        Parameters
        ----------
        crystal_map : orix.crystal_map.CrystalMap
            Crystal map instance to obtain necessary data from.
        kwargs
            Keyword arguments passed to
            :class:`matplotlib_scalebar.scalebar.ScaleBar`.

        Returns
        -------
        bar : matplotlib_scalebar.scalebar.ScaleBar
            Scalebar.

        Examples
        --------
        >>> xmap
        Phase  Orientations   Name       Symmetry  Color
        1      5657 (48.4%)   austenite  432       tab:blue
        2      6043 (51.6%)   ferrite    432       tab:orange
        Properties: iq, dp
        Scan unit: um

        Create a phase map without a scale bar and add it afterwards

        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(projection="plot_map")
        >>> im = ax.plot_map(xmap, scalebar=False)
        >>> sbar = ax.add_scalebar(xmap, location=4, frameon=False)
        """
        # Get whether z, y or x
        last_axis = crystal_map.ndim - 1
        horizontal = crystal_map._coordinate_axes[last_axis]

        # Set a reasonable unit dimension
        scan_unit = crystal_map.scan_unit
        if scan_unit == "px":
            dim = "pixel-length"
        elif scan_unit[-1] == "m":
            dim = "si-length"  # Default
        else:
            dim = dimension._Dimension(scan_unit)

        # Set up arguments to AnchoredSizeBar() if not already present in kwargs
        d = dict(
            pad=0.2,
            sep=3,
            border_pad=0.5,
            location="lower left",
            box_alpha=0.6,
            dimension=dim,
        )
        [kwargs.setdefault(k, v) for k, v in d.items()]

        # Create scalebar
        bar = scalebar.ScaleBar(
            dx=crystal_map._step_sizes[horizontal],
            units=crystal_map.scan_unit,
            **kwargs,
        )
        self.axes.add_artist(bar)
        self.scalebar = bar

        return bar

    def add_overlay(self, crystal_map, item):
        """Use a crystal map property as gray scale values of a phase
        map.

        The property's range is adjusted to [0, 1] for maximum contrast.

        Parameters
        ----------
        crystal_map : orix.crystal_map.CrystalMap
            Crystal map object to obtain necessary data from.
        item : str
            Name of map property to scale phase array with. The property
            range is adjusted for maximum contrast.

        Examples
        --------
        >>> xmap
        Phase  Orientations   Name       Symmetry  Color
        1      5657 (48.4%)   austenite  432       tab:blue
        2      6043 (51.6%)   ferrite    432       tab:orange
        Properties: iq, dp
        Scan unit: um

        Plot a phase map with a map property as overlay

        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(projection="plot_map")
        >>> im = ax.plot_map(xmap)
        >>> ax.add_overlay(xmap, xmap.dp)

        """
        image = self.images[0]
        image_data = image.get_array()

        if image_data.ndim < 3:
            # Adding overlay to a scalar plot (should this be allowed?)
            image_data = image.to_rgba(image_data)[:, :, :3]  # No alpha

        # Scale prop to [0, 1] to maximize image contrast
        overlay = crystal_map.get_map_data(item)
        overlay_min = np.nanmin(overlay)
        rescaled_overlay = (overlay - overlay_min) / (np.nanmax(overlay) - overlay_min)

        n_channels = 3
        for i in range(n_channels):
            image_data[:, :, i] *= rescaled_overlay

        image.set_data(image_data)

    def add_colorbar(self, label=None, **kwargs):
        """Add an opinionated colorbar to the figure and return it.

        The colorbar is also available as an attribute :attr:`colorbar`.

        Parameters
        ----------
        label : str, optional
            Colorbar title, default is ``None``.
        kwargs
            Keyword arguments passed to
            :meth:`mpl_toolkits.axes_grid1.make_axes_locatable.append_axes`.

        Returns
        -------
        cbar : matplotlib.colorbar
            Colorbar.

        Examples
        --------
        >>> xmap
        Phase  Orientations   Name       Symmetry  Color
        1      5657 (48.4%)   austenite  432       tab:blue
        2      6043 (51.6%)   ferrite    432       tab:orange
        Properties: iq, dp
        Scan unit: um

        Plot a map property and add a colorbar

        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(projection="plot_map")
        >>> im = ax.plot_map(xmap, xmap.dp, cmap="inferno")
        >>> cbar = ax.add_colorbar("Dot product")

        If the default options are not satisfactory, the colorbar can be
        updated

        >>> cbar.ax.set_ylabel(ylabel="dp", rotation=90)
        """
        # Keyword arguments
        d = {"position": "right", "size": "5%", "pad": 0.1}
        [kwargs.setdefault(k, v) for k, v in d.items()]

        # Add colorbar
        divider = make_axes_locatable(self)
        cax = divider.append_axes(**kwargs)
        cbar = self.figure.colorbar(self.images[0], cax=cax)

        # Set label with padding
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(label, rotation=270)

        self.colorbar = cbar

        return cbar

    def remove_padding(self):
        """Remove all white padding outside of the figure.

        Examples
        --------
        >>> xmap
        Phase  Orientations   Name       Symmetry  Color
        1      5657 (48.4%)   austenite  432       tab:blue
        2      6043 (51.6%)   ferrite    432       tab:orange
        Properties: iq, dp
        Scan unit: um

        Remove all figure and axes padding of a phase map

        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(projection="plot_map")
        >>> ax.plot_map(xmap)
        >>> ax.remove_padding()
        """
        self.set_axis_off()
        self.margins(0, 0)

        # Tune subplot layout
        cbar = self.images[0].colorbar
        if cbar is not None:
            right = self.figure.subplotpars.right
        else:
            right = 1
        self.figure.subplots_adjust(top=1, bottom=0, right=right, left=0)

    def _set_plot_shape(self, crystal_map, axes=None, depth=None):
        """Set `CrystalMapPlot` attributes describing which data axes to
        plot.

        Parameters
        ----------
        crystal_map : orix.crystal_map.CrystalMap
            Map to determine plotting axes and slices from.
        axes : list of ints, optional
            Data axes to plot. If ``None``, the last two data axes are
            plotted (default).
        depth : int, optional
            Which data layer to plot along the final axis not in `axes` if
            data is 3D. If ``None``, this is set to zero, i.e. the first
            layer (default).
        """
        ndim = crystal_map.ndim

        # Get data axes to plot
        if axes is None:
            axes = [ndim - 2, ndim - 1]
        axes = list(axes)
        axes.sort()
        self._data_axes = axes[:2]  # Can only plot two axes!

        if depth is None:  # Plot first layer
            depth = 0

        # Get data slices to plot
        slices = []
        data_shape = []
        for data_axis, axis_size in zip(
            crystal_map._coordinate_axes.keys(), crystal_map._original_shape
        ):
            data_slice = slice(depth, depth + 1, None)
            for plot_axis in self._data_axes:
                if data_axis == plot_axis:
                    data_slice = slice(None, None, None)
                    data_shape.append(axis_size)
            slices.append(data_slice)
        self._data_slices = tuple(slices)
        self._data_shape = tuple(data_shape)

    def _add_legend(self, patches, **kwargs):
        """Add a legend to the axes object.

        Parameters
        ----------
        patches : list of matplotlib.patches.Patch
            Patches with color code and name.
        kwargs
            Keyword arguments passed to :meth:`matplotlib.axes.legend`.
        """
        d = {
            "borderpad": 0.3,
            "handlelength": 0.75,
            "handletextpad": 0.3,
            "framealpha": 0.6,
        }
        [kwargs.setdefault(k, v) for k, v in d.items()]
        self.legend(handles=patches, **kwargs)

    def _override_status_bar(self, image, crystal_map):
        """Display coordinates, a property value (if scalar values are
        plotted), and Euler angles (in radians) per data point in the
        status bar.

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
        # Get data shape to plot
        n_rows, n_cols = self._data_shape

        # Get rotations, ensuring correct masking
        # TODO: Show orientations in Euler angles (computationally
        #  intensive...)
        r = crystal_map.get_map_data("rotations", decimals=3)
        r = r[self._data_slices].squeeze()

        # Get image data, overwriting potentially masked regions set to 0.0
        image_data = image.get_array()  # numpy.masked.MaskedArray
        # Force float because np.nan is a float
        image_data = image_data.astype("float")
        image_data[image_data.mask] = np.nan

        def status_bar_data(event):
            col = int(event.xdata + 0.5)
            row = int(event.ydata + 0.5)
            return row, col, r[row, col], image_data[row, col]

        # Set width of status bar fields
        x_width = len(str(n_cols - 1))
        y_width = len(str(n_rows - 1))
        scalar_width = len(str(np.nanmax(image_data)))

        # Override
        image.get_cursor_data = status_bar_data
        self.axes.format_coord = lambda x, y: ""

        def format_status_bar_data_rgb(data):
            """Status bar format for RGB plots."""
            return (
                f"(y,x):({data[0]:{y_width}},{data[1]:{x_width}})"
                f" rot:({data[2][0]:5},{data[2][1]:5},{data[2][2]:5})"
            )

        def format_status_bar_data_scalar(data):
            """Status bar format for scalar plots."""
            return (
                f"(y,x):({data[0]:{y_width}},{data[1]:{x_width}})"
                f" val:{data[3]:{scalar_width}}"
                f" rot:({data[2][0]:5},{data[2][1]:5},{data[2][2]:5})"
            )

        # Pick status bar format and override this as well
        if image_data.ndim > 2 and image_data.shape[-1] == 3:
            image.format_cursor_data = format_status_bar_data_rgb
        else:
            image.format_cursor_data = format_status_bar_data_scalar

        return image


register_projection(CrystalMapPlot)
