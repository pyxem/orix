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

"""Private tools for coloring crystal directions.

These are adopted from MTEX. See e.g.
https://github.com/mtex-toolbox/mtex/blob/develop/geometry/@sphericalRegion/polarCoordinates.m.
"""

import matplotlib.colors as mcolors
import numpy as np

from orix.vector import Vector3d


def hsl2hsv(hue, saturation, lightness):
    l2 = 2 * lightness
    s2 = np.where(2 * l2 <= 1, saturation * l2, saturation * (2 - l2))
    saturation = (2 * s2) / (l2 + s2)
    saturation[np.isnan(saturation)] = 0
    value = (l2 + s2) / 2
    return hue, saturation, value


def polar2rgb(azimuth, polar):
    angle = np.mod(azimuth / (2 * np.pi), 1)
    radius = polar / np.pi

    # Compute RGB values from angle and radius
    gray_value = 1
    lightness = (radius - 0.5) * gray_value + 0.5
    saturation = (
        gray_value * (1 - np.abs(2 * radius - 1)) / (1 - np.abs(2 * lightness - 1))
    )
    saturation[np.isnan(saturation)] = 0

    h, s, v = hsl2hsv(angle, saturation, lightness)
    hsv = np.column_stack([h, s, v])

    return mcolors.hsv_to_rgb(hsv)


def calc_angle(center, rx, v):
    rx2 = (rx - rx.dot(center) * center).unit
    ry = center.cross(rx2).unit
    dv = (v - center).unit
    rho = np.mod(np.arctan2(ry.dot(dv).data, rx2.dot(dv).data), 2 * np.pi)
    rho[np.isnan(rho)] = 0
    return rho


def polar_coordinates(sector, v, center, ref):
    v = Vector3d(v).unit
    vxcenter = v.cross(center).unit

    if np.count_nonzero(sector.dot(center).data) == 0:
        r = center.angle_with(v).data / np.pi
    else:
        r = np.ones(v.size) * np.inf
        for normal in sector:
            bc = vxcenter.cross(normal).unit
            d = v.angle_with(-bc).data / center.angle_with(-bc).data
            r = np.minimum(r, d)  # Element-wise minimum

    vz = Vector3d.zvector()
    if np.allclose(center.data, vz.data):
        rx = ref - center
    else:
        rx = vz - center

    azimuth = calc_angle(center=center, rx=rx, v=v)

    return azimuth, r
