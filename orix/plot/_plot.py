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

import matplotlib.projections as mprojections

from .crystal_map_plot import CrystalMapPlot
from .rotation_plot import AxAnglePlot, HomochoricPlot, RodriguesPlot
from .stereographic_plot import StereographicPlot

# Inverse pole figure plot class must be imported below stereographic
# plot class, since the former imports the latter
# isort: off
from .inverse_pole_figure_plot import InversePoleFigurePlot

# isort: on


def register_projections() -> None:
    """Register custom Matplotlib projections.

    This must be called when using one of our custom projections in
    Matplotlib.

    See Also
    --------
    :class:`~orix.plot.CrystalMapPlot`
    :class:`~orix.plot.InversePoleFigurePlot`
    :class:`~orix.plot.AxAnglePlot`
    :class:`~orix.plot.RodriguesPlot`
    :class:`~orix.plot.HomochoricPlot`
    :class:`~orix.plot.StereographicPlot`

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from orix.plot import register_projections
    >>> fig = plt.figure()

    May fail

    >>> ax = fig.add_subplot(projection="stereographic")  # doctest: +SKIP

    Works

    >>> register_projections()
    >>> ax = fig.add_subplot(projection="stereographic")
    """
    projections = [
        AxAnglePlot,
        CrystalMapPlot,
        HomochoricPlot,
        InversePoleFigurePlot,
        RodriguesPlot,
        StereographicPlot,
    ]
    for proj in projections:
        mprojections.register_projection(proj)
