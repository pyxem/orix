# -*- coding: utf-8 -*-
# Copyright 2018-2019 The pyXem developers
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

import numpy as np
from orix.vector import Vector3d
from orix.grid.s1grid import S1Grid


class S2Grid:

    theta_grid = None  # type: S1Grid
    rho_grid = None  # type: S1Grid
    points = None  # type: Vector3d

    def __init__(self, theta_grid: S1Grid, rho_grid: S1Grid):
        self.theta_grid = theta_grid
        self.rho_grid = rho_grid
        theta = np.tile(theta_grid.points, rho_grid.points.shape)
        rho = rho_grid.points
        v = Vector3d.from_polar(theta, rho)
        self.points = v
