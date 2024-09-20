# Copyright 2018-2024 the orix developers
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

from typing import Optional, Tuple, Union

from matplotlib.figure import Figure
import numpy as np
from scipy.interpolate import griddata

from orix.plot.direction_color_keys import DirectionColorKey
from orix.plot.direction_color_keys._util import (
    polar_coordinates_in_sector,
    rgb_from_polar_coordinates,
)
from orix.projections import StereographicProjection
from orix.quaternion import Symmetry
from orix.sampling import sample_S2
from orix.vector import Vector3d


class DirectionColorKeyTSL(DirectionColorKey):
    """Assign colors to (crystal) directions rotated by crystal
    orientations and projected into an inverse pole figure, according to
    the Laue symmetry of the crystal.

    This is based on the TSL color key implemented in MTEX.
    """

    def __init__(self, symmetry: Symmetry) -> None:
        """Create an inverse pole figure (IPF) color key to color
        crystal directions according to a Laue symmetry's fundamental
        sector (IPF).

        Parameters
        ----------
        symmetry
            (Laue) symmetry of the crystal. If a non-Laue symmetry
            is given, the Laue symmetry of that symmetry will be used.
        """
        laue_symmetry = symmetry.laue
        super().__init__(laue_symmetry)

    def direction2color(self, direction: Vector3d) -> np.ndarray:
        """Return an RGB color per orientation given a Laue symmetry
        and a sample direction.

        Plot the inverse pole figure color key with :meth:`plot`.

        Parameters
        ----------
        direction
            Directions to color.

        Returns
        -------
        rgb
            Color array of shape ``direction.shape + (3,)``.
        """
        laue_group = self.symmetry
        h = direction.in_fundamental_sector(laue_group)
        azimuth, polar = polar_coordinates_in_sector(laue_group.fundamental_sector, h)
        polar = 0.5 + polar / 2
        return rgb_from_polar_coordinates(azimuth, polar)

    def _create_rgba_grid(
        self, alpha: float = 1.0, return_extent: bool = False
    ) -> Union[
        np.ndarray,
        Tuple[np.ndarray, Tuple[Tuple[float, float], Tuple[float, float]]],
    ]:
        """Create the 2d colormap used to represent crystal directions.

        Parameters
        ----------
        alpha
            Transparency value for plot.
        return_extent
            If ``True`` a tuple of tuples ``(min, max)`` representing
            the extent of the fundamental sector in the stereographic
            projection for both ``x`` and ``y`` is also returned.
            Default is ``False``.

        Returns
        -------
        rgba_grid
            Colormap values with RGBA channels.
        extent
            Tuple of tuples ``(min, max)`` representing the extent of
            the fundamental sector in the stereographic projection for
            both ``x`` and ``y``. Returned if ``return_extent=True``.
        """
        laue_group = self.symmetry
        sector = laue_group.fundamental_sector

        v = sample_S2(2)
        v2 = Vector3d(np.row_stack((v[v <= sector].data, sector.edges.data)))

        rgb = self.direction2color(v2)
        r, g, b = rgb.T

        x, y = StereographicProjection().vector2xy(v2)
        # Round, otherwise `scipy.interpolate.griddata` is too slow
        x = x.round(11)
        y = y.round(11)
        yx = np.column_stack((y, x))

        y_min = np.min(y)
        y_max = np.max(y)
        x_min = np.min(x)
        x_max = np.max(x)
        grid_step = 0.001
        grid_yx = np.mgrid[y_min:y_max:grid_step, x_min:x_max:grid_step]

        griddata_kwargs = dict(points=yx, xi=tuple(grid_yx), method="cubic")
        r2 = griddata(values=r, **griddata_kwargs)
        g2 = griddata(values=g, **griddata_kwargs)
        b2 = griddata(values=b, **griddata_kwargs)
        a2 = np.full_like(r2, alpha)
        # create RGBA image and clip to ensure [0..1] range
        rgba_grid = np.dstack((r2, g2, b2, a2)).clip(0, 1)
        # some values are NaN as they are not within fundamental zone
        NaN_values = np.isnan(r2)
        rgba_grid[NaN_values] = 1  # set to valid values
        # make invalid points transparent
        rgba_grid[NaN_values, -1] = 0
        rgba_grid = rgba_grid[::-1]

        if return_extent:
            return rgba_grid, ((x_min, x_max), (y_min, y_max))
        else:
            return rgba_grid

    def plot(self, return_figure: bool = False) -> Optional[Figure]:
        """Plot the inverse pole figure color key.

        Parameters
        ----------
        return_figure
            Whether to return the figure. Default is ``False``.

        Returns
        -------
        figure
            Color key figure, returned if ``return_figure=True``.
        """
        from orix.plot.inverse_pole_figure_plot import _setup_inverse_pole_figure_plot

        fig, axes = _setup_inverse_pole_figure_plot(self.symmetry)
        ax = axes[0]
        ax.plot_ipf_color_key()

        if return_figure:
            return fig
