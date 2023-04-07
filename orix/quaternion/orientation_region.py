# -*- coding: utf-8 -*-
# Copyright 2018-2023 the orix developers
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

"""An orientation region is some subset of the complete space of orientations.

The complete orientation space represents every possible orientation of an
object. The whole space is not always needed, for example if the orientation
of an object is constrained or (most commonly) if the object is symmetrical. In
this case, the space can be segmented using sets of Rotations representing
boundaries in the space. This is clearest in the Rodrigues parametrisation,
where the boundaries are planes, such as the example here: the asymmetric
domain of an adjusted 432 symmetry.

.. image:: _static/img/orientation-region-Oq.png
   :width: 300px
   :alt: Boundaries of an orientation region in Rodrigues space.
   :align: center

Rotations or orientations can be inside or outside of an orientation region.
"""

from __future__ import annotations

import itertools
from typing import Tuple

import numpy as np

from orix.quaternion import Quaternion
from orix.quaternion.rotation import Rotation
from orix.quaternion.symmetry import C1, Symmetry, get_distinguished_points
from orix.vector import Rodrigues

_EPSILON = 1e-9  # small number to avoid round off problems


def _get_large_cell_normals(s1, s2):
    dp = get_distinguished_points(s1, s2)
    normals = Rodrigues.zero(dp.shape + (2,))
    planes1 = dp.axis * np.tan(dp.angle / 4)
    planes2 = -dp.axis * np.tan(dp.angle / 4) ** -1
    planes2.data[np.isnan(planes2.data)] = 0
    normals[:, 0] = planes1
    normals[:, 1] = planes2
    normals = Rotation.from_neo_euler(normals).flatten().unique(antipodal=False)
    if not normals.size:
        return normals
    _, inv = normals.axis.unique(return_inverse=True)
    axes_unique = []
    angles_unique = []
    for i in np.unique(inv):
        n = normals[inv == i]
        axes_unique.append(n.axis.data[0])
        angles_unique.append(np.max(n.angle))
    normals = Rotation.from_axes_angles(np.array(axes_unique), angles_unique)
    return normals


def get_proper_groups(Gl: Symmetry, Gr: Symmetry) -> Tuple[Symmetry, Symmetry]:
    """Return the appropriate groups for the asymmetric domain
    calculation.

    Parameters
    ----------
    Gl
        First point group.
    Gr
        Second point group.

    Returns
    -------
    Gl
        First proper subgroup(s) or proper inversion subgroup(s), as
        appropriate.
    Gr
        Second proper subgroup(s) or proper inversion subgroup(s), as
        appropriate.

    Raises
    ------
    NotImplementedError
        If both groups are improper and neither contain an inversion,
        special consideration is needed which is not yet implemented in
        orix.
    """
    if Gl.is_proper and Gr.is_proper:
        return Gl, Gr
    elif Gl.is_proper and not Gr.is_proper:
        return Gl, Gr.proper_subgroup
    elif not Gl.is_proper and Gr.is_proper:
        return Gl.proper_subgroup, Gr
    else:
        if Gl.contains_inversion and Gr.contains_inversion:
            return Gl.proper_subgroup, Gr.proper_subgroup
        elif Gl.contains_inversion and not Gr.contains_inversion:
            return Gl.proper_subgroup, Gr.laue_proper_subgroup
        elif not Gl.contains_inversion and Gr.contains_inversion:
            return Gl.laue_proper_subgroup, Gr.proper_subgroup
        else:
            raise NotImplementedError(
                "Both groups are improper and do not contain inversion"
            )


class OrientationRegion(Rotation):
    """A set of :class:`~orix.quaternion.Rotation` which are the normals
    of an orientation region.
    """

    @classmethod
    def from_symmetry(cls, s1: Symmetry, s2: Symmetry = C1) -> OrientationRegion:
        """Return the set of unique (mis)orientations of a symmetrical
        object.

        Parameters
        ----------
        s1
            First symmetry.
        s2
            Second symmetry. Default is C1 (the identity).

        Returns
        -------
        region
            Orientation region.
        """
        s1, s2 = get_proper_groups(s1, s2)
        large_cell_normals = _get_large_cell_normals(s1, s2)
        disjoint = s1 & s2
        fz = disjoint.fundamental_zone()
        fz_normals = Rotation.from_axes_angles(fz, np.pi)
        normals = Rotation(np.concatenate([large_cell_normals.data, fz_normals.data]))
        region = cls(normals)
        vertices = region.vertices()
        if vertices.size > 0:
            region = region[np.any(np.isclose(region.dot_outer(vertices), 0), axis=1)]
        return region

    def vertices(self) -> Rotation:
        """Return the vertices of the orientation region.

        Returns
        -------
        rot
            Region vertices.
        """
        normal_combinations = list(itertools.combinations(self, 3))
        if len(normal_combinations) < 1:
            return Rotation.empty()
        c1, c2, c3 = zip(*normal_combinations)
        c1, c2, c3 = (
            Rotation.stack(c1).flatten(),
            Rotation.stack(c2).flatten(),
            Rotation.stack(c3).flatten(),
        )
        rot = Rotation.triple_cross(c1, c2, c3)
        rot = rot[~np.any(np.isnan(rot.data), axis=-1)]
        rot = rot[rot < self].unique()
        surface = np.any(np.isclose(rot.dot_outer(self), 0), axis=1)
        return rot[surface]

    def faces(self) -> list:
        """Return the faces of the orientation region.

        Returns
        -------
        faces
            List of sets of rotations, each set describing a face or the
            region.
        """
        normals = Rotation(self)
        vertices = self.vertices()
        faces = []
        for n in normals:
            faces.append(vertices[np.isclose(vertices.dot(n), 0)])
        faces = [f for f in faces if f.size > 2]
        return faces

    def __gt__(self, other: OrientationRegion) -> np.ndarray:
        """Overridden greater than method. Applying this to an
        Orientation will return only those orientations that lie within
        the orientation region.
        """
        c = Quaternion(self).dot_outer(Quaternion(other))
        inside = np.logical_or(
            np.all(np.greater_equal(c, -_EPSILON), axis=0),
            np.all(np.less_equal(c, +_EPSILON), axis=0),
        )
        return inside

    def get_plot_data(self) -> Rotation:
        """Return suitable rotations for the construction of a
        wireframe delineating the borders of the region.

        Returns
        -------
        g
            Rotations delineating the borders of the region.
        """
        from orix.vector import Vector3d

        # Get a grid of vector directions
        theta = np.linspace(0, 2 * np.pi - _EPSILON, 361)
        rho = np.linspace(0, np.pi - _EPSILON, 181)
        theta, rho = np.meshgrid(theta, rho)
        v = Vector3d.from_polar(rho, theta)

        # Get the norms of the cell vector normals
        n = Rodrigues.from_rotation(self).norm[:, np.newaxis, np.newaxis]
        if n.size == 0:
            return Rotation.from_axes_angles(v, np.pi)

        d = (-self.axis).dot_outer(v.unit)
        x = n * d
        with np.errstate(divide="ignore"):
            omega = 2 * np.arctan(np.where(x != 0, x**-1, np.pi))

        # Keep the smallest allowed angle
        omega[omega < 0] = np.pi
        omega = np.min(omega, axis=0)
        g = Rotation.from_axes_angles(v.unit, omega)

        return g
