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

These functions are adopted from MTEX.
"""

import matplotlib.colors as mcolors
import numpy as np

from orix.quaternion import Rotation
from orix.vector import Vector3d


def polar_coordinates_in_sector(sector, v):
    r"""Calculate the polar coordinates of crystal direction(s)
    :class:`Vector3d` relative to the (bary)center of a
    :class:`FundamentalSector` of a Laue group.

    Parameters
    ----------
    sector : FundamentalSector
        A fundamental sector of a Laue group, described by a set of
        normal vectors, a center, and vertices.
    v : Vector3d
        Crystal direction(s) to get polar coordinates for.

    Returns
    -------
    azimuth : np.ndarray
        Azimuthal polar coordinate(s).
    polar : np.ndarray
        Polar polar coordinate(s).

    Notes
    -----
    This procedure is adopted from MTEX' :code:`polarCoordinates`
    function, which implements the coloring described in section 2.4 in
    :cite:`nolze2016orientation` (see Fig. 4 in that reference).

    The azimuthal coordinate is the angle to the barycenter relative to
    some fixed vertex of the sector, here chosen as the north pole
    [001]. The polar coordinate is the distance to the barycenter. The
    barycenter is the center of the fundamental sector.
    """
    center = sector.center.unit
    n_vertices = sector.vertices.size
    v = v.unit

    # Azimuthal coordinate
    if n_vertices == 0:
        # Point group Ci (-1) has no vertices
        rx = Vector3d.xvector() - center
    else:
        rx = Vector3d.zvector() - center  # North pole to sector center
    azimuth = _calculate_azimuth(center, rx, v)
    if n_vertices != 0:
        azimuth = _correct_azimuth(azimuth, sector, rx)

    # Polar coordinate
    if np.count_nonzero(sector.dot(center).data) == 0:
        polar = center.angle_with(v).data / np.pi
    else:
        # Normal to plane containing sector center and crystal direction
        v_center_normal = v.cross(center).unit
        polar = np.full(v.shape, np.inf)
        for normal in sector:
            boundary_points = v_center_normal.cross(normal).unit
            # Some boundary points are zero vectors
            with np.errstate(invalid="ignore"):
                distances_polar = (-v).angle_with(boundary_points).data
                distances_polar /= (-center).angle_with(boundary_points).data
            distances_polar[np.isnan(distances_polar)] = 1
            polar = np.minimum(polar, distances_polar)  # Element-wise

    return azimuth, polar


def rgb_from_polar_coordinates(azimuth, polar):
    """Calculate RGB colors from polar coordinates.

    Parameters
    ----------
    azimuth : np.ndarray
        Azimuthal coordinate(s).
    polar : np.ndarray
        Polar coordinate(s).

    Returns
    -------
    rgb : np.ndarray
        Color(s).
    """
    angle = np.mod(azimuth / (2 * np.pi), 1)
    h, s, v = hsl_to_hsv(angle, 1, polar)
    return mcolors.hsv_to_rgb(np.stack((h, s, v), axis=-1))


def hsl_to_hsv(hue, saturation, lightness):
    """Convert color described by HSL (hue, saturation and lightness) to
    HSV (hue, saturation and value).

    Adapted from MTEX' function :code:`hsl2hsv`.

    Parameters
    ----------
    hue : np.ndarray or float
        Hue(s). Not changed by the function, but included in input and
        output for convenience.
    saturation : np.ndarray or float
        Saturation value(s).
    lightness : np.ndarray or float
        Lightness value(s).

    Returns
    -------
    hue : np.ndarray or float
        The same copy of hue(s) as input.
    saturation2 : np.ndarray or float
        Adjusted saturation value(s).
    value : np.ndarray or float
        Value(s).
    """
    l2 = 2 * lightness
    s2 = saturation * np.where(l2 <= 1, l2, 2 - l2)
    saturation2 = (2 * s2) / (l2 + s2)
    saturation2[np.isnan(saturation2)] = 0
    value = (l2 + s2) / 2
    return hue, saturation2, value


def _calculate_azimuth(center, rx, v):
    """Calculate the azimuthal coordinate of the polar coordinates of a
    fundamental sector (inverse pole figure).

    Taken from MTEX' :code:`calcAngle` function.

    Parameters
    ----------
    center : ~orix.vector.Vector3d
    rx : ~orix.vector.Vector3d
    v : ~orix.vector.Vector3d

    Returns
    -------
    azimuth : numpy.ndarray
    """
    rx = (rx - rx.dot(center) * center).unit  # Orthogonal to center
    ry = center.cross(rx).unit  # Perpendicular to rx
    distances_azimuthal = (v - center).unit
    azimuth = np.arctan2(
        ry.dot(distances_azimuthal).data, rx.dot(distances_azimuthal).data
    )
    azimuth = np.mod(azimuth, 2 * np.pi)
    azimuth[np.isnan(azimuth)] = 0
    return azimuth


def _correct_azimuth(azimuth, sector, rx):
    """Correct the azimuthal coordinate of the polar coordinates of a
    fundamental sector (inverse pole figure) when vertices are not
    an equal distance from each other.

    This ensures for example that the entirely blue part of the m-3m
    sector is in the [111] direction, and not in the direction
    corresponding to the sector position perpendicular to the
    [001]-[101] line (straight north from the barycenter).

    Taken from MTEX' :code:`correctAngle` function.

    Parameters
    ----------
    azimuth : numpy.ndarray
    sector : ~orix.vector.FundamentalSector
    rx : ~orix.vector.Vector3d

    Returns
    -------
    azimuth : numpy.ndarray
        Corrected azimuthal coordinates.
    """
    center = sector.center.unit
    vertices = sector.vertices.unit

    m = 1000
    azimuth2 = np.linspace(0, 2 * np.pi, m)
    rot = Rotation.from_axes_angles(center, azimuth2[:-1])
    normals = rot * rx.cross(center).unit

    # Distance between center and all boundary points
    cross_outer = Vector3d.zero((sector.size, m - 1))
    for i in range(sector.size):
        cross_outer[i] = sector[i].cross(normals)
    polar = np.min(cross_outer.angle_with(center).data, axis=0)

    if vertices.size == 3:
        angle = _calculate_azimuth(center, rx, vertices)
        azimuth3 = np.round(m * np.sort(angle) / (2 * np.pi)).astype(int)
        if azimuth3[0] < 10:
            azimuth3 = np.delete(azimuth3, 0)
        idx = np.arange(azimuth3[0])
        polar[idx] /= np.sum(polar[idx]) / 3
        idx = np.arange(azimuth3[0], azimuth3[1])
        polar[idx] /= np.sum(polar[idx]) / 3
        idx = np.arange(azimuth3[1], polar.size)
        polar[idx] /= np.sum(polar[idx]) / 3

    polar = 2 * np.pi * np.cumsum(np.append(np.zeros(1), polar / np.sum(polar)))
    azimuth = np.interp(azimuth, azimuth2, polar)

    return azimuth
