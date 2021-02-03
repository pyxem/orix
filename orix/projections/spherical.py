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

from orix.vector import SphericalRegion


class InverseSphericalProjection:
    """Get spherical coordinates from various representations.
    """

    def __init__(self, region=[0, 0, 1], antipodal=True):
        self.region = SphericalRegion(region)
        self.antipodal = antipodal

    def from_vector(self, v):
        v_inside = v[v < self.region]  # Restrict to plottable domain
        theta, phi, r = v_inside.to_polar()
        return theta.data, phi.data, r.data

    def xy2spherical(self, x, y):
        theta = 2 * np.arctan(np.sqrt(x ** 2 + y ** 2))
        phi = np.arctan2(y, x)
        return theta, phi

    def __repr__(self):
        region = self.region.data[0].tolist()
        return (
            f"{self.__class__.__name__}, region={region}, "
            f"antipodal={self.antipodal}"
        )
