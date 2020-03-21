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
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm


def _plot_crystal_map(
        crystal_map,
        data,
        legend_patches=None,
        scalebar=True,
        padding=True,
        **kwargs,
):
    """Plot crystal map data.

    Parameters
    ----------
    crystal_map : orix.crystal_map.CrystalMap
        CrystalMap object, needed to get map shape, scan unit and
        horizontal step size.
    data : numpy.ndarray
        Map data to plot. A 2D array with potential RGB(A) channels.
    legend_patches : list of matplotlib.patches.Patch, optional
        Legend patches, e.g. phase colors and names. If ``None`` is passed
        (default), no legend is added.
    scalebar : bool, optional
        Whether to display a scalebar (default is ``True``).
    padding : bool, optional
        Whether to show white padding (default is ``True``). Setting this
        to ``False`` removes all white padding outside of plot, including
        pixel coordinate ticks.
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
    """

    cmap = kwargs.pop('cmap', None)
    interpolation = kwargs.pop('interpolation', 'none')

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap=cmap, interpolation=interpolation, **kwargs)

    # Remove white padding around plot, including pixel coordinate ticks
    if not padding:
        ax.set(xticks=[], yticks=[])
        fig.gca().set_axis_off()
        fig.subplots_adjust(
            top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        ax.margins(0, 0)
        fig.gca().xaxis.set_major_locator(plt.NullLocator())
        fig.gca().yaxis.set_major_locator(plt.NullLocator())

    # Legend
    if legend_patches:
        ax = _add_legend(axes=ax, patches=legend_patches)

    # Scalebar
    if scalebar:
        ax = _add_scalebar(
            axes=ax,
            map_width=crystal_map.shape[1],
            scan_unit=crystal_map.scan_unit,
            step_size=crystal_map.step_sizes[-1],
        )

    return fig, ax, im


def _add_legend(axes, patches):
    """Add a legend with patches to axes.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        Axes object to add legend to.
    patches : list of matplotlib.patches.Patch
        Legend patches, e.g. phase colors and names.
    **kwargs :
        Optional keyword arguments passed to
        :meth:`matplotlib.axes.Axes.legend`.

    Returns
    -------
    axes : matplotlib.axes.Axes
        Axes object with legend.
    """
    axes.legend(
        handles=patches,
        borderpad=0.3,
        handlelength=0.75,
        handletextpad=0.3,
        framealpha=0.6,
    )
    return axes


def _add_scalebar(axes, map_width, scan_unit, step_size):
    """Add a scalebar via `AnchoredSizeBar` to axes.

    To find an appropriate scalebar width, this snippet from
    MTEX [Bachmann2010]_, written by Eric Payton and Philippe Pinard, is
    used: https://github.com/mtex-toolbox/mtex/blob/b8fc167d06d453a2b3e212b1ac383acbf85a5a27/plotting/scaleBar.m,

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        Axes object to add scalebar to.
    map_width : int
        Map width in pixels.
    scan_unit : str
        Map scan unit, e.g. 'um' or 'nm'.
    step_size : float
        Map step size in last map dimension.

    Returns
    -------
    axes : matplotlib.axes.Axes
        Axes object with scalebar.
    """

    # Initial scalebar width should be approximately 1/10 of map width
    scalebar_width = 0.1 * (map_width * step_size)

    # Ensure a suitable number is used, e.g. going from 1000 nm to 1 um
    scalebar_width, scan_unit, factor = convert_unit(scalebar_width, scan_unit)

    # This snippet for finding a suitable scalebar width is taken from
    # MTEX: https://github.com/mtex-toolbox/mtex/blob/b8fc167d06d453a2b3e212b1ac383acbf85a5a27/plotting/scaleBar.m,
    # written by Eric Payton and Philippe Pinard.
    # We want a round, not too high number without decimals
    good_values = np.array(
        [1, 2, 5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 200, 500, 750],
        dtype=int,
    )
    # Find good value closest to initial scalebar width
    difference = abs(scalebar_width - good_values)
    good_value_idx = np.where(difference == difference.min())[0][0]
    scalebar_width = good_values[good_value_idx]

    # Scale width by factor from above conversion (usually factor = 1.0)
    scalebar_width = scalebar_width * factor
    scalebar_width_px = scalebar_width / step_size

    # Allow for a potential decimal in scalebar number if something didn't go
    # as planned
    if scalebar_width.is_integer():
        scalebar_width = int(scalebar_width)
    else:
        warnings.warn(f"Scalebar width {scalebar_width} is not an integer.")

    if scan_unit == 'um':
        scan_unit = "\u03BC" + "m"

    # Create scalebar
    fontprops = fm.FontProperties(size=11)
    bar = AnchoredSizeBar(
        transform=axes.transData,
        size=scalebar_width_px,
        label=str(scalebar_width) + ' ' + scan_unit,
        loc=3,
        pad=0.2,
        sep=3,
        borderpad=0.3,
        frameon=True,
        size_vertical=1,
        color="black",
        fontproperties=fontprops,
    )
    bar.patch.set_alpha(0.6)

    # Add scalebar to axes
    axes.add_artist(bar)

    return axes


def convert_unit(value, unit):
    """Return the value with a suitable, not too large, unit.

    This algorithm is taken directly from MTEX [Bachmann2010]_
    https://github.com/mtex-toolbox/mtex/blob/a74545383160610796b9525eedf50a241800ffae/plotting/plotting_tools/switchUnit.m,
    written by Ralf Hielscher.

    Parameters
    ----------
    value : float
        The value to convert.
    unit : str
        The value unit, e.g. um.

    Returns
    -------
    new_value : float
        The input value converted to the suitable unit.
    new_unit : str
        A more suitable unit than the input.
    factor : float
        Factor to multiple `new_value` with to get the input value.

    Examples
    --------
    >>> convert_unit(17.55 * 1e3, 'nm')
    17.55 um 999.9999999999999
    >>> convert_unit(17.55 * 1e-3, 'mm')
    17.55 um 0.001

    """

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

    # Calculate new value, unit and the conversion factor
    new_value = value_in_metres / lookup_table[suitable_unit_idx][1]
    new_unit = lookup_table[suitable_unit_idx][0]
    factor = lookup_table[suitable_unit_idx][1] / lookup_table[new_unit_idx][1]

    return new_value, new_unit, factor
