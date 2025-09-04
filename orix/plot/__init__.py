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

"""Extensions of Matplotlib's projections framework for plotting
:class:`~orix.vector.Vector3d`, :class:`~orix.quaternion.Rotation`,
:class:`~orix.quaternion.Orientation`,
:class:`~orix.quaternion.Misorientation`, and
:class:`~orix.crystal_map.CrystalMap`.

NOTE: While lazy loading is preferred, the following six classes
are explicitly imported in order to populate matplotlib.projections
"""
import lazy_loader

from orix.plot.crystal_map_plot import CrystalMapPlot
from orix.plot.rotation_plot import AxAnglePlot, RodriguesPlot, RotationPlot
from orix.plot.stereographic_plot import StereographicPlot

# Must be imported below StereographicPlot since it imports it
from orix.plot.inverse_pole_figure_plot import InversePoleFigurePlot  # isort: skip


# Imports from stub file (see contributor guide for details)
__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

del lazy_loader
