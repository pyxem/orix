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

"""Generation of equispaced spherical grids in *SO(2)*."""

import numpy as np

from orix.vector import Vector3d


def uniform_SO2_sample(resolution):
    r"""Vectors of a UV mesh on a unit sphere *SO(2)*.

    The mesh vertices are defined by the parametrization

    .. math::
        x = \sin(u)\cos(v)
        y = \sin(u)\sin(v)
        z = \cos(u)

    Parameters
    ----------
    resolution : float
        Maximum angle between nearest neighbour grid points, in degrees.
        The resolution of :math:`u` and :math:`v` are rounded up to get
        an integer number of equispaced polar and azimuthal grid lines.

    Returns
    -------
    Vector3d
    """
    steps_azimuth = int(np.ceil(360 / resolution))
    steps_polar = int(np.ceil(180 / resolution)) + 1
    azimuth = np.linspace(0, np.pi, num=steps_azimuth, endpoint=True)
    polar = np.linspace(0, 2 * np.pi, num=steps_polar, endpoint=False)
    azimuth_prod, polar_prod = np.meshgrid(azimuth, polar)
    return Vector3d.from_polar(azimuth=azimuth_prod.ravel(), polar=polar_prod.ravel())
