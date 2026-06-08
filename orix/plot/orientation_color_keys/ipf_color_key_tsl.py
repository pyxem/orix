#
# Copyright 2018-2026 the orix developers
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

from typing import TYPE_CHECKING, Literal, overload

import numpy as np

from orix.plot.direction_color_keys.direction_color_key_tsl import (
    DirectionColorKeyTSL,
)
from orix.plot.orientation_color_keys.ipf_color_key import (
    VALID_SAMPLE_DIRECTION,
    IPFColorKey,
)
from orix.quaternion.orientation import Orientation
from orix.quaternion.symmetry import Symmetry
from orix.vector.miller import Miller

if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.figure as mfigure


class IPFColorKeyTSL(IPFColorKey):
    r"""Assigns colors to orientations in a symmetry-aware manner.

    Parameters
    ----------
    symmetry
        Crystal symmetry used to define the colormap. If symmetry
        is not a Laue Symmetry, the equivalent Laue symmetry will
        be used instead.
    direction
        Sample direction. If not given, sample Z direction (out of
        plane) is used.

    Notes
    -----
    IPF color keys work by defining symmetry-specific color maps that
    convert every orientation into a color. Images such as EBSD maps can
    then be colored based on the crystal vector aligned with a queried
    sample direction. For the simple case where the sample direction is
    aligned with the viewing direction, this is akin to asking "what
    color is the crystal vector pointing out of the screen at me."

    The TSL colormaps map only the 11 Laue classes, as they can all be
    described as weighted combinations of red, green, and blue. For
    details on this map as well as options for less limiting alternative
    mappings, refer to :cite:`nolze2016orientation`.
    """

    def __init__(
        self, symmetry: Symmetry, direction: VALID_SAMPLE_DIRECTION | None = None
    ) -> None:
        # Symmetry converted to Laue by calling overwritten parsing
        # method
        super().__init__(symmetry=symmetry, direction=direction)

    # -------------------------- Properties -------------------------- #

    @property
    def direction_color_key(self) -> DirectionColorKeyTSL:
        """Return the direction color key of the IPF color key."""
        return DirectionColorKeyTSL(self.symmetry)

    # ------------------------ Public methods ------------------------ #

    def orientation2color(self, orientation: Orientation) -> np.ndarray:
        """Return an RGB color per orientation given a Laue symmetry
        and a sample direction.

        Parameters
        ----------
        orientation
            Orientations to color.

        Returns
        -------
        rgb
            Color array of *orientation* shape + (3,).

        See Also
        --------
        :meth:`plot`
        """
        # v_crystal = g_s2c * v_sample
        v_crys: Miller = orientation * self.direction

        rgb = self.direction_color_key.direction2color(v_crys)

        return rgb

    @overload
    def plot(
        self, return_figure: Literal[False] = False
    ) -> None: ...  # pragma: no cover

    @overload
    def plot(
        self, return_figure: Literal[True] = True
    ) -> "mfigure.Figure": ...  # pragma: no cover

    def plot(self, return_figure: bool = False) -> "mfigure.Figure | None":
        """Plot the inverse pole figure color key.

        Parameters
        ----------
        return_figure
            Whether to return the figure. Default is False.

        Returns
        -------
        figure
            Color key figure, returned if *return_figure* is True.
        """
        return self.direction_color_key.plot(return_figure=return_figure)

    # ----------------------- Private methods ------------------------ #

    def _parse_symmetry_or_raise(self, symmetry: Symmetry) -> Symmetry:
        symmetry = super()._parse_symmetry_or_raise(symmetry)
        return symmetry.laue
