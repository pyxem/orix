# Copyright 2018-2024 the orix developers
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

"""Stereographic projection of 3D vectors."""

from typing import Tuple, Union

import numpy as np

from orix.vector import SphericalRegion, Vector3d

_UPPER_HEMISPHERE = SphericalRegion([0, 0, 1])
_LOWER_HEMISPHERE = SphericalRegion([0, 0, -1])


class StereographicProjection:
    """Get stereographic coordinates (X, Y) from representations of
    vectors.

    Parameters
    ----------
    pole
        -1 or 1, where -1 (1) means the projection point is the
        lower (upper) pole [00-1] ([001]), i.e. only vectors with
        z > 0 (z < 0) are returned.
    """

    def __init__(self, pole: int = -1):
        """Initialize projection by setting whether the south pole (-1)
        or north pole (1) is the projection point.
        """
        self.pole = pole
        self.region = SphericalRegion([0, 0, pole * -1])

    def vector2xy(self, v: Vector3d) -> Tuple[np.ndarray, np.ndarray]:
        r"""Return stereographic coordinates (X, Y) of 3D unit vectors.

        Parameters
        ----------
        v
            If it is not a unit vector, it will be made into one.

        Returns
        -------
        x
            Stereographic coordinate X of shape same shape as the input
            vector shape. Only the vectors with :attr:`~Vector3d.z`
            coordinate positive (``pole`` = -1) or negative (``pole`` =
            1) are returned.
        y
            Stereographic coordinate Y of shape same shape as the input
            vector shape. Only the vectors with :math:`z` coordinate
            positive (`pole` = -1) or negative (`pole` = 1) are
            returned.

        Notes
        -----
        The stereographic coordinates :math:`(X, Y)` are calculated from
        the unit vectors' cartesian coordinates :math:`(x, y, z)` as

        .. math::
            (X, Y) = \left(\frac{-px}{z - p}, \frac{-py}{z - p}\right),

        where :math:`p` is either 1 (north pole as projection point) or
        -1 (south pole as projection point).
        """
        v = v[v <= self.region]
        return _vector2xy(v, pole=self.pole)

    def spherical2xy(
        self,
        azimuth: Union[np.ndarray, list, tuple, float],
        polar: Union[np.ndarray, list, tuple, float],
        degrees: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Return stereographic coordinates (X, Y) from 3D unit vectors
        created from spherical coordinates, azimuth :math:`\phi` and
        polar :math:`\theta`, defined as in the ISO 31-11 standard
        :cite:`weisstein2005spherical`.

        Parameters
        ----------
        azimuth
            Spherical azimuth coordinate.
        polar
            Spherical polar coordinate.
        degrees
            If ``True``, the coordinates are assumed to be in degrees.
            Default is ``False``.

        Returns
        -------
        x
            Stereographic coordinate X of shape same shape as the input
            vector shape. Only the vectors with :attr:`~Vector3d.z`
            coordinate positive (``pole`` = -1) or negative (``pole`` =
            1) are returned.
        y
            Stereographic coordinate Y of shape same shape as the input
            vector shape. Only the vectors with :attr:`~Vector3d.z`
            coordinate positive (``pole`` = -1) or negative (``pole`` =
            1) are returned.

        See Also
        --------
        vector2xy
        """
        v = Vector3d.from_polar(azimuth, polar, degrees=degrees)
        return self.vector2xy(v)

    @staticmethod
    def vector2xy_split(
        v: Vector3d,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return two sets of stereographic coordinates (X, Y) from 3D
        unit vectors: one set for vectors in the upper hemisphere, and
        one for the lower.

        Parameters
        ----------
        v
            If it is not a unit vector, it will be made into one.

        Returns
        -------
        x_upper
            Stereographic coordinate X of upper hemisphere vectors, of
            shape same shape as the input vector shape.
        y_upper
            Stereographic coordinate Y of upper hemisphere vectors, of
            shape same shape as the input vector shape.
        x_lower
            Stereographic coordinate X of lower hemisphere vectors, of
            shape same shape as the input vector shape.
        y_lower
            Stereographic coordinate Y of lower hemisphere vectors, of
            shape same shape as the input vector shape.

        See Also
        --------
        vector2xy
        """
        x_upper, y_upper = _vector2xy(v[v <= _UPPER_HEMISPHERE], pole=-1)
        x_lower, y_lower = _vector2xy(v[v <= _LOWER_HEMISPHERE], pole=1)
        return x_upper, y_upper, x_lower, y_lower

    def spherical2xy_split(
        self,
        azimuth: Union[np.ndarray, list, tuple, float],
        polar: Union[np.ndarray, list, tuple, float],
        degrees: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""Return two sets of stereographic coordinates (X, Y) from 3D
        unit vectors created from spherical coordinates, azimuth
        :math:`\phi` and polar :math:`\theta`, defined as in the
        ISO 31-11 standard :cite:`weisstein2005spherical`: one set for
        vectors in the upper hemisphere, and one for the lower.

        Parameters
        ----------
        azimuth
            Spherical azimuth coordinate.
        polar
            Spherical polar coordinate.
        degrees
            If ``True``, the coordinates are assumed to be in degrees.
            Default is ``False``.

        Returns
        -------
        x_upper
            Stereographic coordinate X of upper hemisphere vectors, of
            shape same shape as the input vector shape.
        y_upper
            Stereographic coordinate Y of upper hemisphere vectors, of
            shape same shape as the input vector shape.
        x_lower
            Stereographic coordinate X of lower hemisphere vectors, of
            shape same shape as the input vector shape.
        y_lower
            Stereographic coordinate Y of lower hemisphere vectors, of
            shape same shape as the input vector shape.

        See Also
        --------
        vector2xy
        """
        v = Vector3d.from_polar(azimuth, polar, degrees=degrees)
        return self.vector2xy_split(v)

    @property
    def inverse(self):
        """Return the corresponding inverse projection,
        :class:`InverseStereographicProjection`, with the same
        projection pole.
        """
        return InverseStereographicProjection(pole=self.pole)


def _vector2xy(v: Vector3d, pole: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return stereographic coordinates (X, Y) of 3D unit vectors.

    (X, Y) is both zero for vectors with z equal to the projection pole.

    Parameters
    ----------
    v
        If it is not a unit vector, it will be made into one.
    pole
        -1 or 1, where -1 (1) means the projection point of the
        stereographic transform is the lower (upper) pole [00-1]
        ([001]), i.e. only vectors with z > 0 (z < 0) are returned.

    Returns
    -------
    x
        Stereographic coordinate X of shape same shape as the input
        vector shape.
    y
        Stereographic coordinate Y of shape same shape as the input
        vector shape.

    See Also
    --------
    vector2xy
    """
    vx, vy, vz = v.unit.xyz
    # We explicitly say (X, Y) is zero when the denominator is zero
    denom = vz - pole
    not_zero = denom != 0
    x = np.divide(-pole * vx, denom, where=not_zero, out=np.zeros_like(vx))
    y = np.divide(-pole * vy, denom, where=not_zero, out=np.zeros_like(vy))
    return x, y


class InverseStereographicProjection:
    """Get 3D unit vectors or spherical coordinates from stereographic
    coordinates (X, Y).

    Parameters
    ----------
    pole
        -1 or 1, where -1 (1) means the projection point is the
        lower (upper) pole [00-1] ([001]), i.e. only vectors with
        z > 0 (z < 0) are returned.
    """

    def __init__(self, pole: int = -1):
        """Initialize inverse projection by setting whether the south
        pole (-1) or north pole (1) is the projection point.
        """
        self.pole = pole

    def xy2vector(
        self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]
    ) -> Vector3d:
        r"""Return 3D unit vectors from stereographic coordinates
        (X, Y).

        Parameters
        ----------
        x
            X coordinates.
        y
            Y coordinates.

        Returns
        -------
        vector
            Unit vectors corresponding to the stereographic coordinates.
            Whether the upper or lower hemisphere points are returned is
            controlled by ``pole`` (-1 = upper, 1 = lower).

        Notes
        -----
        The 3D unit vectors :math:`(x, y, z)` are calculated from the
        stereographic coordinates :math:`(X, Y)` as

        .. math::
            (x, y, z) = \left(
                \frac{2x}{1 + x^2 + y^2},
                \frac{2y}{1 + x^2 + y^2},
                \frac{-p(1 - x^2 - y^2)}{1 + x^2 + y^2}
            \right),

        where :math:`p` is either 1 (north pole as projection point) or
        -1 (south pole as projection point).
        """
        denom = 1 + x**2 + y**2
        vx = 2 * x / denom
        vy = 2 * y / denom
        vz = -self.pole * (1 - x**2 - y**2) / denom
        return Vector3d(np.column_stack([vx, vy, vz]))

    def xy2spherical(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        degrees: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Return spherical coordinates, azimuth :math:`phi` and
        polar :math:`theta`, defined as in the ISO 31-11 standard
        :cite:`weisstein2005spherical`, from stereographic coordinates
        (X, Y).

        Parameters
        ----------
        x
            X coordinates.
        y
            Y coordinates.
        degrees
            If ``True``, the given angles are returned in degrees.
            Default is ``False``.

        Returns
        -------
        azimuth
            Azimuth spherical coordinate corresponding to (X, Y).
            Whether the coordinates for the upper or lower hemisphere
            points are returned is controlled by ``pole`` (-1 = upper,
            1 = lower).
        theta
            Polar spherical coordinate corresponding to (X, Y). Whether
            the coordinates for the upper or lower hemisphere points are
            returned is controlled by ``pole`` (-1 = upper, 1 = lower).

        See Also
        --------
        xy2vector
        StereographicProjection.spherical2xy
        """
        v = self.xy2vector(x=x, y=y)
        azimuth, polar, _ = v.to_polar(degrees=degrees)
        return azimuth, polar
