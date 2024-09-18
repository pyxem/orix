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

from typing import Optional

from matplotlib.figure import Figure
import numpy as np

from orix.plot.direction_color_keys import DirectionColorKeyTSL
from orix.plot.orientation_color_keys import IPFColorKey
from orix.quaternion import Orientation, Symmetry
from orix.vector.vector3d import Vector3d


class IPFColorKeyTSL(IPFColorKey):
    """Assign colors to crystal directions rotated by crystal
    orientations and projected into an inverse pole figure, according to
    the Laue symmetry of the crystal.

    This is based on the TSL color key implemented in MTEX.
    """

    def __init__(
        self, symmetry: Symmetry, direction: Optional[Vector3d] = None
    ) -> None:
        """Create an inverse pole figure (IPF) color key to color
        orientations according a sample direction and a Laue symmetry's
        fundamental sector (IPF).

        Parameters
        ----------
        symmetry : orix.quaternion.Symmetry
            (Laue) symmetry of the crystal. If a non-Laue symmetry
            is given, the Laue symmetry of that symmetry will be used.
        direction : orix.vector.Vector3d, optional
            Sample direction. If not given, sample Z direction (out of
            plane) is used.
        """
        super().__init__(symmetry.laue, direction=direction)

    @property
    def direction_color_key(self) -> DirectionColorKeyTSL:
        return DirectionColorKeyTSL(self.symmetry)

    def orientation2color(self, orientation: Orientation) -> np.ndarray:
        """Return an RGB color per orientation given a Laue symmetry
        and a sample direction.

        Plot the inverse pole figure color key with :meth:`plot`.

        Parameters
        ----------
        orientation
            Orientations to color.

        Returns
        -------
        rgb
            Color array of shape ``orientation.shape + (3,)``.
        """
        # TODO: Take crystal axes into account, by using Miller instead
        # of Vector3d
        m = orientation * self.direction
        rgb = self.direction_color_key.direction2color(m)
        return rgb

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
        return self.direction_color_key.plot(return_figure=return_figure)
