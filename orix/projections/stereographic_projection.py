# -*- coding: utf-8 -*-
# Copyright 2018-2020 the orix developers
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

"""
Stereographic projection of 3D vectors
"""

import numpy as np
from orix.vector import Vector3d


class StereographicProjection:
    """Project 3D vectors on the x-y plane stereographically"""
    @classmethod
    def project(cls, v, pole=-1):
        """Convert vector3d to [X,Y]"""
        v = Vector3d(v)
        return _get_stereographic_coordinates(v, pole)

    @classmethod
    def project_spherical(cls, theta, phi, pole=-1):
        """Convert theta, phi to [X,Y]"""
        return _get_stereographic_from_spherical(theta, phi, pole)

    @classmethod
    def project_split(cls, v):
        """Convert vector3d to [X,Y] split by hemisphere"""
        v = Vector3d(v)
        return _get_stereographic_hemisphere_coords(v)

    @classmethod
    def project_split_spherical(cls, theta, phi):
        """Convert theta, phi to [X,Y] split by hemisphere"""
        return _get_stereographic_hemisphere_coords_spherical(theta, phi)

    @classmethod
    def iproject(cls, xy, pole=-1):
        """Convert [X, Y] to Vector3d"""
        return _get_unitvectors_from_stereographic(xy, pole=pole)

    @classmethod
    def iproject_spherical(cls, xy, pole=-1):
        """Convert [X, Y] to theta, phi"""
        return _get_spherical_from_stereographic(xy, pole=pole)


def _get_stereographic_from_spherical(theta, phi, pole):
    """Convert spherical polar (theta) - azimuthal (phi) coordinates to
    stereographic"""
    v = Vector3d.from_polar(theta, phi)
    return _get_stereographic_coordinates(v, pole)


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
    x = -pole*vector3d.x.data/(vector3d.z.data-pole)
    y = -pole*vector3d.y.data/(vector3d.z.data-pole)
    stereo_coords = np.vstack([x, y]).T
    return stereo_coords


def _get_spherical_from_stereographic(xy, pole=-1):
    """
    Convert stereographic coordinates to theta, phi spherical coordinates
    """
    xyz = _get_unitvectors_from_stereographic(xy, pole=pole)
    theta, phi, _ = xyz.to_polar()
    return theta, phi


def _get_unitvectors_from_stereographic(xy, pole=-1):
    """
    Convert stereographic coordinates to 3D unit vectors.

    Parameters
    ----------
    coordinates : (N, 2) np.ndarray
       (x, y) coordinates of the stereo plot
    pole : int
        1 or -1 to indicate the projection point being [0,0,1] or [0,0,-1]
        respectively.

    Returns
    -------
    vector3d : orix.vector.vector3d.Vector3d
        The list of unit vectors corresponding to stereographic coordinates

    Notes
    -----
    The 1 pole is usually used to project vectors with z<0, and the -1 pole
    is usually used to project vectors with z>0. This way the coordinates
    end up in the unit circle, otherwise they end up outside.
    """
    if xy.ndim == 1:
        xy = xy.reshape((1, xy.shape[0]))
    x = xy[:, 0]
    y = xy[:, 1]
    zz = -pole*(1 - x**2 - y**2)/(1 + x**2 + y**2)
    xx = 2*x/(1 + x**2 + y**2)
    yy = 2*y/(1 + x**2 + y**2)
    xyz = np.vstack([xx, yy, zz]).T
    return Vector3d(xyz)
