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

import matplotlib.figure as mfigure
import numpy as np

from orix.plot.direction_color_keys.direction_color_key_tsl import (
    DirectionColorKeyTSL,
)
from orix.plot.orientation_color_keys.ipf_color_key import IPFColorKey
from orix.quaternion.orientation import Orientation
from orix.quaternion.symmetry import Symmetry
from orix.vector.vector3d import Vector3d


class IPFColorKeyTSL(IPFColorKey):
    """Assign colors to crystal directions rotated by crystal
    orientations and projected into an inverse pole figure, according to
    the Laue symmetry of the crystal.

    This is based on the TSL color key implemented in MTEX.
    """

    def __init__(
        self, symmetry: Symmetry, direction: Vector3d | str | None = None
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
        if type(direction) is str:
            direction = {
                "x": Vector3d.xvector(),
                "y": Vector3d.yvector(),
                "z": Vector3d.zvector(),
            }.get(direction.lower(), "fail")
            if direction == "fail":
                raise IOError(
                    "'direction' must be 'x', 'y','z', or a Vector3D instance"
                )

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

    def plot(self, return_figure: bool = False) -> mfigure.Figure | None:
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
