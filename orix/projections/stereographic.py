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

"""Stereographic projection of vectors."""

import numpy as np

from orix.vector import Vector3d


class StereographicProjection:
    """Get stereographic coordinates from other representations."""

    def __init__(self, pole=-1):
        self.pole = pole

    def vector2xy(self, v):
        """(x, y, z) to (X, Y)."""
        vx, vy, vz = v.xyz
        pole = self.pole
        x = -pole * vx / (vz - pole)
        y = -pole * vy / (vz - pole)
        return x, y

    def spherical2xy(self, azimuth, polar):
        """(azimuth, polar) to (X, Y)."""
        v = Vector3d.from_polar(theta=polar, phi=azimuth, r=1)
        return self.vector2xy(v)

    @classmethod
    def project_split(cls, v):
        """Convert vector3d to [X,Y] split by hemisphere"""
        v = Vector3d(v)
        return _get_stereographic_hemisphere_coords(v)

    @classmethod
    def project_split_spherical(cls, theta, phi):
        """Convert theta, phi to [X,Y] split by hemisphere"""
        return _get_stereographic_hemisphere_coords_spherical(theta, phi)


class InverseStereographicProjection:
    """Get other representations from stereographic coordinates."""

    def __init__(self, pole=-1):
        self.pole = pole

    def xy2vector(self, x, y):
        """Convert stereographic coordinates (X, Y) to unit vectors
        (x, y, z).

        Parameters
        ----------
        coordinates : (N, 2) np.ndarray
           (x, y) coordinates of the stereo plot
        pole : int
            1 or -1 to indicate the projection point being [0, 0, 1] or
            [0, 0, -1] respectively.

        Returns
        -------
        orix.vector.Vector3d
            Unit vectors corresponding to stereographic coordinates.

        Notes
        -----
        The 1 pole is usually used to project vectors with z < 0, and
        the -1 pole is usually used to project vectors with z > 0. This
        way, the coordinates end up in the unit circle, otherwise they
        end up outside.
        """
        denom = 1 + x ** 2 + y ** 2
        vx = 2 * x / denom
        vy = 2 * y / denom
        vz = -self.pole * (1 - x ** 2 - y ** 2) / denom
        return Vector3d(np.column_stack([vx, vy, vz]))

    def xy2spherical(self, x, y):
        """(X, Y) to (azimuth, polar)."""
        v = self.xy2vector(x=x, y=y)
        return v.phi, v.theta


def _get_stereographic_hemisphere_coords_spherical(theta, phi):
    """Convert spherical polar (theta) - azimuthal (phi) coordinates to
    stereographic split by hemispheres"""
    v = Vector3d.from_polar(theta, phi)
    return _get_stereographic_hemisphere_coords(v)


def _get_stereographic_hemisphere_coords(vector3d):
    """
    Converts Vector3d objects into stereographic coordinates

    The upper and lower hemisphere (z>=0 and z<0) are split so that all
    coordinates can be plot within a unit circle.

    Parameters
    ----------
    vector3d : orix.vector.vector3d.Vector3d
        The list of vectors to convert to stereographic coordinates

    Returns
    -------
    upper_coordinates : (N, 2) np.ndarray
       (x, y) stereo coordinates of the vectors in the upper hemisphere
    lower_coordinates : (N, 2) np.ndarray
        (x, y) stereo coordinates of the vectors in the lower hemisphere
    """
    vector3d = vector3d.unit
    vector3d_up = vector3d[vector3d.z >= 0]
    vector3d_down = vector3d[vector3d.z < 0]
    upper_coordinates = _get_stereographic_coordinates(vector3d_up, pole=-1)
    lower_coordinates = _get_stereographic_coordinates(vector3d_down, pole=1)
    return upper_coordinates, lower_coordinates


def _get_stereographic_coordinates(vector3d, pole=-1):
    """
    Converts Vector3d objects into stereographic coordinates

    Parameters
    ----------
    vector3d : orix.vector.vector3d.Vector3d
        The list of vectors to convert to stereographic coordinates
    pole : int
        1 or -1 to indicate the projection point being [0,0,1] or [0,0,-1]
        respectively.

    Returns
    -------
    coordinates : (N, 2) np.ndarray
       (x, y) coordinates of the stereo plot

    Notes
    -----
    The 1 pole is usually used to project vectors with z<0, and the -1 pole
    is usually used to project vectors with z>0. This way the coordinates
    end up in the unit circle, otherwise they end up outside.
    """
    if pole not in [1, -1]:
        raise ValueError("Pole must be 1 or -1")
    vector3d = vector3d.unit
    x = -pole * vector3d.x.data / (vector3d.z.data - pole)
    y = -pole * vector3d.y.data / (vector3d.z.data - pole)
    stereo_coords = np.vstack([x, y]).T
    return stereo_coords
