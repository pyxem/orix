#
# Copyright 2018-2026 the orix developers
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with orix. If not, see <http://www.gnu.org/licenses/>.
#

from __future__ import annotations

import itertools

import numpy as np

from orix.quaternion.quaternion import Quaternion
from orix.quaternion.rotation import Rotation
from orix.quaternion.symmetry import C1, Symmetry, get_distinguished_points
from orix.utils import _constants
from orix.utils._deprecation import deprecated, deprecated_argument
from orix.vector.neo_euler import Rodrigues


def _get_large_cell_normals(
    start: Symmetry = C1,
    end: Symmetry = C1,
    distinguished_points: Rotation | None = None,
) -> Rotation:
    """Return rotations defining fundamental zone bounds due to
    symmetry.

    Given two symmetries, calculates every unique rotation equivalent
    to the identity, called "distinguished points". A Voronoi
    tesselation is then done to define the bounds within which all
    rotations are closer to identity than any distinguished points.

    Instead of calculating them, a set of distinguished points can
    also be added by the user, in which case the symmetry arguments
    will be ignored. This is useful for defining non-crystallographic
    orientation regions.

    Parameters
    ----------
    start
        Starting symmetry.
    end
        Ending symmetry.
    distinguished_points
        Set of rotations that are equivalent to identity. If given,
        *start* and *end* will be ignored, and the normals will be
        defined via tesselation of these points.

    Returns
    -------
    normals
        Rotation normals defining the fundamental zone bounds for the
        given symmetries.
    """
    if distinguished_points is None:
        distinguished_points = get_distinguished_points(s1=start, s2=end)

    # Remove points with a rotation angle of zero (the identity and its
    # antipodal), which would produce an infinite Rodrigues vector
    distinguished_points = distinguished_points[distinguished_points.angle > 0]

    if distinguished_points.size == 0:
        return Rotation.empty()

    normals = Rodrigues.zero(distinguished_points.shape + (2,))
    planes1 = distinguished_points.axis * np.tan(distinguished_points.angle / 4)
    planes2 = -distinguished_points.axis * np.tan(distinguished_points.angle / 4) ** -1
    planes2.data[np.isnan(planes2.data)] = 0
    normals[:, 0] = planes1
    normals[:, 1] = planes2
    normals = Rotation.from_rodrigues(normals).flatten().unique(antipodal=False)

    _, inv = normals.axis.unique(return_inverse=True)
    axes_unique = []
    angles_unique = []
    for i in np.unique(inv):
        n = normals[inv == i]
        axes_unique.append(n.axis.data[0])
        angles_unique.append(np.max(n.angle))
    normals = Rotation.from_axes_angles(axes_unique, angles_unique)

    return normals


# TODO: Remove once 0.16.0 is released
@deprecated(since="0.16.0", removal="0.17.0", alternative="get_asymmetric_groups")
def get_proper_groups(Gl: Symmetry, Gr: Symmetry) -> tuple[Symmetry, Symmetry]:
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
    return get_asymmetric_groups(start=Gl, end=Gr)


def get_asymmetric_groups(start: Symmetry, end: Symmetry) -> tuple[Symmetry, Symmetry]:
    """Return groups for defining a fundamental zone (orientation
    region) for (mis)orientations :cite:`morawiec2004orientations`.

    Parameters
    ----------
    start
        Initial point group, C1 for orientations.
    end
        Ending point group.

    Returns
    -------
    start
        Initial proper, inversion, or laue subgroup as appropriate.
    end
        Final proper, inversion, or laue subgroup as appropriate.

    See Also
    --------
    :meth:`~orix.quaternion.OrientationRegion.from_symmetry`

    Notes
    -----
    Parametrization of the fundamental zone follows section 6.3.1 in
    :cite:`morawiec2004orientations`. The output can be used with
    :meth:`~orix.quaternion.OrientationRegion.from_symmetry` to
    reproduce results from that textbook.

    Because this method intentionally omits misorientation symmetries
    where combinations of rotoinversions create an ambiguous
    definition of the fundamental zone, orix has instead adopted a
    method based on :cite:`krakow2017onthree` (see
    :meth:`~orix.quaternion.OrientationRegion.from_symmetry`). However,
    asymmetric domains are still used elsewhere, so this function is
    provided for convenience.
    """
    if start.is_proper and end.is_proper:
        return start, end
    elif start.is_proper and not end.is_proper:
        return start, end.proper_subgroup
    elif not start.is_proper and end.is_proper:
        return start.proper_subgroup, end
    else:
        if start.contains_inversion and end.contains_inversion:
            return start.proper_subgroup, end.proper_subgroup
        elif start.contains_inversion and not end.contains_inversion:
            return start.proper_subgroup, end.laue_proper_subgroup
        elif not start.contains_inversion and end.contains_inversion:
            return start.laue_proper_subgroup, end.proper_subgroup
        else:
            return start.laue_proper_subgroup, end.laue_proper_subgroup


class OrientationRegion(Rotation):
    """A subset of rotation space.

    The complete set of all possible rigid body rotations is called
    *SO(3)*. It can be thought of as half the quaternion unit sphere,
    the entirety of Rodrigues space, the set of all 3x3 matrices with a
    determinant of 1, or various other descriptors based on the
    application.

    Sometimes, this whole space is not needed, for example if the
    orientation of an object is constrained or (most commonly) if the
    object is symmetrical. In this case, the space can be segmented
    using sets of rotations representing boundaries in the space. This
    can be most easily visualized using Rodrigues space, where the
    boundaries become flat planes normal to the rodrigues vectors of
    those bounding rotations.

    .. image:: /_static/img/orientation-region-Oq.png
       :width: 300px
       :alt: Boundaries of an orientation region in Rodrigues space.
       :align: center

    Quaternions can then be quickly defined as inside or outside of
    these regions via a dot product operation.

    Notes
    -----
    Notably, these regions are only defined in *SO(3)*, which means they
    cannot account for improper operations. This is why
    :meth:`from_symmetry` calculates identical regions for point groups
    *432* and *m-3m* despite *m-3m* having twice as many distinguished
    points. This ends up being irrelevant for orientations since any
    improper operation that places a point within a fundamental zone
    always has a paired proper operation that returns an identical
    quaternion, but it can create confusion for misorientations with
    rotoinversions when we assume an orientation region can uniquely
    define a true fundamental zone.
    """

    # ------------------------ Dunder methods ------------------------ #

    def __gt__(self, other: OrientationRegion) -> np.ndarray:
        """Overridden greater than method.

        Applying this to a rotation will return only those that lie
        within the region. This operation does not account for
        inversion.
        """
        c = Quaternion(self).dot_outer(Quaternion(other))
        inside = np.logical_or(
            np.all(np.greater_equal(c, -_constants.eps9), axis=0),
            np.all(np.less_equal(c, _constants.eps9), axis=0),
        )
        return inside

    # ------------------------ Class methods ------------------------- #

    # TODO: Remove deprecations and handling once 0.16.0 is released
    @classmethod
    @deprecated_argument("s1", since="0.16.0", removal="0.17.0", alternative="start")
    @deprecated_argument("s2", since="0.16.0", removal="0.17.0", alternative="end")
    def from_symmetry(
        cls,
        start: Symmetry = C1,
        end: Symmetry = C1,
        s1=None,
        s2=None,
    ) -> OrientationRegion:
        """Return an orientation region for a given symmetry.

        Parameters
        ----------
        start
            Initial point group, C1 for passive orientations.
        end
            Ending point group.

        Returns
        -------
        region
            The orientation region.

        Notes
        -----
        These regions are identical to the fundamental zone (FZ) for all
        orientations and every misorientation where both symmetries are
        proper and/or centrosymmetric. For all other cases, it is still
        garunteed to inlude only one unique representation achievable
        through proper rotations.

        orix follows the FZ definitions described in
        :cite:`krakow2017onthree`, except when it comes to handling of
        improper symmetry elements. As described in section 3(b) of that
        paper, FZ boundaries created by improper rotations represent
        operations that cannot be achieved through rigid body rotations.
        As a result, it is often appropriate to ignore these elements, a
        practice orix also defaults to in operations such as orientation
        :meth:`~orix.quaternion.Orientation.reduce` and
        :meth:`~orix.quaternion.Orientation.mean`.

        However, in order to support the rare exceptions where improper
        FZ boundaries might be relevant (for example, grain boundary
        misorientation distributions between non-centrosymmetric
        crystals), orix allows defining FZs that include improper
        elements for all 1024 possible combinations of two symmetries.

        For 704 combinations, including all orientations and any
        misorientation with centrosymmetry, this is irrelevant as
        both methods exactly reduce to the same 121 cases described in
        :cite:`krakow2017onthree`. For the remaining 320 combinations
        where one or both point groups contain rotoinversions but are
        not centrosymmetric (for example, *6mm* --> *6mm*), there are
        always one and possibly two improper rotations that also map to
        the FZ but with unique quaternion values, as well as a possible
        unique pseudo-proper rotation only achievable through two
        rotoinversions. In these cases, orix will return the FZ bounding
        the unique proper and pseudo-proper representations.
        """
        if s1 is not None:
            start = s1
        if s2 is not None:
            end = s2

        # Step 1: fundamental zones are only defined for proper rotations.
        # add inversion centers where necessary to define as unique as
        # possible of a fundamental zone, then remove all improper operators.

        # If either symmetry is proper, any improper symmetries of the second
        # group will fall outside the shared fundamental zone.
        if not start.is_proper and not end.is_proper:
            # If both symmetries contain an inversion, all improper operators
            # will have a paired proper operator. If neither do, the proper
            # and improper rotations will form two identical but inverted
            # fundamental zones. Both cases produce one proper, two improper,
            # and one pseudo-proper fundamental zone, but the first case is
            # uninteresting because they have idencial quaternion
            # representations. The second case requires some special
            # consideration when using fundamental zones for averages or
            # reducing. Regardless though, for both cases, reducing both
            # symmetries to their proper subgroup gives the correct
            # fundamental zone.
            if start.contains_inversion != end.contains_inversion:
                # The remaining case is when only one of the two groups
                # contains an inversion. In this case, the combination of
                # an inversion and rotation creates a mirror that needs to
                # be added to the disjoint group. this is easiest done
                # by converting the inversion-less symmetry to it's laue
                # symmetry.
                if not start.contains_inversion:
                    start = start.laue
                if not end.contains_inversion:
                    end = end.laue
        # With mirrors from inversion/rotoinversion combinations accounted
        # for, remove all improper operators. The fundamental zone will now be
        # one of the 61 in Table 3 of krakow2017.
        start = start.proper_subgroup
        end = end.proper_subgroup

        # Step 2: define the bounding cells using the distinguished points.
        # This is equivalent to the voronoi tesselation described in Krakow,
        # but done in rodrigues space to take advantage of rectilinear planes.
        dp = get_distinguished_points(start, end)
        # These large cell normals are always one of the 15 from figure 5 of
        # krakow2017
        large_cell_normals = _get_large_cell_normals(dp)

        # Step 3: (only for misorientations) restrict the domain to the
        # fundamental sector of the pole figure of the shared symmetries.
        disjoint = start & end
        fz = disjoint.fundamental_zone()
        fz_normals = Rotation.from_axes_angles(fz, np.pi)
        normals = Rotation(np.concatenate([large_cell_normals.data, fz_normals.data]))
        region = cls(normals)
        vertices = region.vertices()
        if vertices.size:
            region = region[np.any(np.isclose(region.dot_outer(vertices), 0), axis=1)]

        return region

    # --------------------- Other public methods --------------------- #

    def vertices(self) -> Rotation:
        """Return the vertices of the orientation region.

        Returns
        -------
        rot
            Orientation region vertices.
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
        r = Rotation.triple_cross(c1, c2, c3)
        r = r[~np.any(np.isnan(r.data), axis=-1)]
        r = r[r < self].unique()
        surface = np.any(np.isclose(r.dot_outer(self), 0), axis=1)
        return r[surface]

    def faces(self) -> list[Rotation]:
        """Return the faces of the orientation region.

        Returns
        -------
        faces
            List of sets of rotations, each set describing a face of the
            region.
        """
        normals = Rotation(self)
        vertices = self.vertices()
        faces = []
        for n in normals:
            faces.append(vertices[np.isclose(vertices.dot(n), 0)])
        faces = [f for f in faces if f.size > 2]
        return faces

    def get_plot_data(self) -> Rotation:
        """Return suitable rotations for the construction of a wireframe
        delineating the borders of the region.

        Returns
        -------
        rot
            Rotations delineating the borders of the region.
        """
        from orix.vector import Vector3d

        # Get a grid of vector directions
        theta = np.linspace(0, 2 * np.pi - _constants.eps9, 361)
        rho = np.linspace(0, np.pi - _constants.eps9, 181)
        theta, rho = np.meshgrid(theta, rho)
        g = Vector3d.from_polar(rho, theta)

        # Get the cell vector normal norms
        if self.size == 0:
            return Rotation.from_axes_angles(g, np.pi)
        n = self.to_rodrigues().norm[:, np.newaxis, np.newaxis]

        d = (-self.axis).dot_outer(g.unit)
        x = n * d
        with np.errstate(divide="ignore"):
            omega = 2 * np.arctan(np.where(x != 0, x**-1, np.pi))

        # Keep the smallest allowed angle
        omega[omega < 0] = np.pi
        omega = np.min(omega, axis=0)
        r = Rotation.from_axes_angles(g.unit, omega)

        return r
