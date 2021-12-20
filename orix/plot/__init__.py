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

"""Extensions of Matplotlib's projections framework for plotting
:class:`~orix.vector.Vector3d`, :class:`~orix.quaternion.Rotation`,
:class:`~orix.quaternion.Orientation`,
:class:`~orix.quaternion.Misorientation`, and
:class:`~orix.crystal_map.CrystalMap`.

Example of usage::

    >>> import matplotlib.pyplot as plt
    >>> from orix import plot, vector
    >>> fig, ax = plt.subplots(subplot_kw=dict(projections="stereographic"))
    >>> ax.scatter(vector.Vector3d([[0, 0, 1], [1, 0, 1]]))
"""

from orix.plot.crystal_map_plot import CrystalMapPlot
from orix.plot.direction_color_keys import DirectionColorKeyTSL
from orix.plot.orientation_color_keys import EulerColorKey, IPFColorKeyTSL
from orix.plot.rotation_plot import AxAnglePlot, RodriguesPlot, RotationPlot
from orix.plot.stereographic_plot import StereographicPlot

# Must be imported below StereographicPlot since it imports it
from orix.plot.inverse_pole_figure_plot import InversePoleFigurePlot


# Lists what will be imported when calling "from orix.plot import *"
__all__ = [
    "AxAnglePlot",
    "CrystalMapPlot",
    "DirectionColorKeyTSL",
    "EulerColorKey",
    "InversePoleFigurePlot",
    "IPFColorKeyTSL",
    "RodriguesPlot",
    "RotationPlot",
    "StereographicPlot",
]
