# -*- coding: utf-8 -*-
# Copyright 2019-2021 The kikuchipy developers
#
# This file is part of kikuchipy.
#
# kikuchipy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# kikuchipy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with kikuchipy. If not, see <http://www.gnu.org/licenses/>.

import numpy as np

from orix.vector import Vector3d


class StereographicProjection2:
    """Get stereographic coordinates from other representations."""

    def vector2xy(self, v):
        """(x, y, z) to (X, Y)."""
        zenith, azimuth, radial = self.vector2spherical(v)

        # Map zenith to upper hemishphere
        zenith = np.where(zenith < np.pi / 2, zenith, np.pi - zenith)

        # Stereographic projection
        rho = radial * np.tan(zenith / 2)

        x = rho * np.cos(azimuth)
        y = rho * np.sin(azimuth)

        return x, y

    def spherical2xy(self, azimuth, polar):
        """(azimuth, polar) via unit vector to (X, Y)."""
        v = Vector3d.from_polar(theta=polar, phi=azimuth, r=1)
        return self.vector2xy(v)


class InverseStereographicProjection2:
    """Get other representations from stereographic coordinates."""

    @staticmethod
    def xy2vector(x, y):
        """(X, Y) to (x, y, z) unit vector."""
        polar = 2 * np.arctan(np.sqrt(x ** 2 + y ** 2))
        azimuth = np.arctan2(y, x)
        return Vector3d.from_polar(theta=polar, phi=azimuth, r=1)

    def xy2spherical(self, x, y):
        """(X, Y) to (azimuthal, polar)."""
        v = self.xy2vector(x=x, y=y)
        azimuth = v.phi
        polar = v.theta
        return azimuth, polar
