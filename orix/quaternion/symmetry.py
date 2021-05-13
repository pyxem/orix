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

"""Collections of transformations representing a symmetry group.

An object's symmetry can be characterized by the transformations relating
symmetrically-equivalent views on that object. Consider the following shape.

.. image:: /_static/img/triad-object.png
   :width: 200px
   :alt: Image of an object with three-fold symmetry.
   :align: center

This obviously has three-fold symmetry. If we rotated it by
:math:`\\frac{2}{3}\\pi` or :math:`\\frac{4}{3}\\pi`, the image would be unchanged.
These angles, as well as :math:`0`, or the identity, expressed as quaternions,
form a group. Applying any operation in the group to any other results in
another member of the group.

Symmetries can consist of rotations or inversions, expressed as
improper rotations. A mirror symmetry is equivalent to a 2-fold rotation
combined with inversion.
"""

from diffpy.structure.spacegroups import GetSpaceGroup
import numpy as np

from orix.quaternion.rotation import Rotation
from orix.vector import AxAngle, Vector3d


class Symmetry(Rotation):
    """The set of rotations comprising a point group."""

    name = ""

    def __repr__(self):
        cls = self.__class__.__name__
        shape = str(self.shape)
        data = np.array_str(self.data, precision=4, suppress_small=True)
        rep = "{} {}{pad}{}\n{}".format(
            cls, shape, self.name, data, pad=self.name and " "
        )
        return rep

    def __and__(self, other):
        return Symmetry.from_generators(
            *[g for g in self.subgroups if g in other.subgroups]
        )

    @property
    def order(self):
        """int : The number of elements of the group."""
        return self.size

    @property
    def is_proper(self):
        """bool : True if this group contains only proper rotations."""
        return np.all(np.equal(self.improper, 0))

    @property
    def subgroups(self):
        """list of Symmetry : the groups that are subgroups of this
        group.
        """
        return [g for g in _groups if g._tuples <= self._tuples]

    @property
    def proper_subgroups(self):
        """list of Symmetry : the proper groups that are subgroups of
        this group.
        """
        return [g for g in self.subgroups if g.is_proper]

    @property
    def proper_subgroup(self):
        """Symmetry : the largest proper group of this subgroup."""
        subgroups = self.proper_subgroups
        subgroups_sorted = sorted(subgroups, key=lambda g: g.order)
        return subgroups_sorted[-1]

    @property
    def laue(self):
        """Symmetry : this group plus inversion."""
        return Symmetry.from_generators(self, Ci)

    @property
    def laue_proper_subgroup(self):
        """Symmetry : the proper subgroup of this group plus inversion.
        """
        return self.laue.proper_subgroup

    @property
    def contains_inversion(self):
        """bool : True if this group contains inversion."""
        return Ci._tuples <= self._tuples

    @property
    def diads(self):
        axis_orders = self.get_axis_orders()
        diads = [ao for ao in axis_orders if axis_orders[ao] == 2]
        if len(diads) == 0:
            return Vector3d.empty()
        return Vector3d.stack(diads).flatten()

    @property
    def primary_axis(self):
        v001 = Vector3d([0, 0, 1]).unit
        v111 = Vector3d([1, 1, 1]).unit
        v100 = Vector3d([1, 0, 0]).unit
        v010 = Vector3d([0, 1, 0]).unit
        return dict(
            triclinic=Vector3d.empty(),
            monoclinic=v010,
            orthorhombic=v100,
            tetragonal=v001,
            trigonal=v111,
            hexagonal=v001,
            cubic=v100,
        )[self.system]

    @property
    def secondary_axis(self):
        v111 = Vector3d([1, 1, 1]).unit
        v100 = Vector3d([1, 0, 0]).unit
        v010 = Vector3d([0, 1, 0]).unit
        v_empty = Vector3d.empty()
        return dict(
            triclinic=v_empty,
            monoclinic=v_empty,
            orthorhombic=v100,
            tetragonal=v100,
            trigonal=v010,
            hexagonal=v100,
            cubic=v111,
        )[self.system]

    @property
    def tertiary_axis(self):
        v110 = Vector3d([1, 1, 0]).unit
        v1bar10 = Vector3d([1, -1, 0]).unit
        v120 = Vector3d([1, 2, 0]).unit
        v110 = Vector3d([1, 1, 0]).unit
        v_empty = Vector3d.empty()
        return dict(
            triclinic=v_empty,
            monoclinic=v_empty,
            orthorhombic=v110,
            tetragonal=v110,
            trigonal=v1bar10,
            hexagonal=v120,
            cubic=v110,
        )[self.system]

    @property
    def is_primary_axis_rotation(self):
        parallel_with_e3 = np.isclose(np.abs(self.axis.dot(self.primary_axis).data), 1)
        not_identity = self.angle.data != 0
        return parallel_with_e3 * not_identity

    @property
    def primary_axis_rotation(self):
        """Rotations with their axis parallel with e3, excluding the
        identity element.
        """
        return self[self.is_primary_axis_rotation]

    @property
    def primary_axis_order(self):
        """Order (multiplicity) of the primary rotation axis."""
        r = self.primary_axis_rotation
        if r.size == 0:
            return 1
        else:
            # Get smallest angle about this axis or axes
            smallest_angle = np.min(np.abs(r.angle.data))
            return int(2 * np.pi / smallest_angle)

    @property
    def system(self):
        name = self.name
        if name in ["1", "-1"]:
            return "triclinic"
        elif name in ["211", "121", "112", "2", "m11", "1m1", "11m", "m", "2/m"]:
            return "monoclinic"
        elif name in ["222", "mm2", "mmm"]:
            return "orthorhombic"
        elif name in ["4", "-4", "4/m", "422", "4mm", "-42m", "4/mmm"]:
            return "tetragonal"
        elif name in ["3", "-3", "321", "312", "32", "3m", "-3m"]:
            return "trigonal"
        elif name in ["6", "-6", "6/m", "622", "6mm", "-6m2", "6/mmm"]:
            return "hexagonal"
        elif name in ["23", "m-3", "432", "-43m", "m-3m"]:
            return "cubic"
        else:
            return None

    @property
    def max_euler_angles(self):
        pg = self.proper_subgroup
        angles = {
            "1": (360, 180, 360),
            "211": (360, 90, 360),
            "112": (360, 180, 180),
            "222": (360, 90, 180),
            "3": (360, 180, 120),
            "321": (360, 90, 120),
            "4": (360, 180, 90),
            "422": (360, 90, 90),
            "6": (360, 180, 60),
            "622": (360, 90, 60),
            # MTEX:
            #            "23": (360, 180, 180),
            #            "432": (360, 90, 90),
            "23": (360, 45, 180),
            "432": (360, 54.74, 90),
        }
        return np.deg2rad(angles[pg.name])

    @property
    def rotations_not_about_primary_axis(self):
        axis110 = Vector3d([1, 1, 0])
        axis111 = Vector3d([1, 1, 1])
        a1 = Vector3d.xvector()
        a2 = Vector3d.yvector()
        m = a1 - a2
        pg = self.proper_subgroup
        if pg.name in ["1", "211", "121"]:
            r = ~Rotation(pg)
        elif pg.name in ["112", "3", "4", "6"]:
            r = Rotation.identity()
        elif pg.name in ["222", "321", "422", "622"]:
            angles = 2 * np.pi / 2 * np.arange(0, 2)
            r = Rotation.from_neo_euler(AxAngle.from_axes_angles(a1, angles))
        elif pg.name == "312":
            angles = 2 * np.pi / 2 * np.arange(0, 2)
            r = Rotation.from_neo_euler(AxAngle.from_axes_angles(m, angles))
        elif pg.name == "23":
            angles1 = 2 * np.pi / 3 * np.arange(0, 3)
            angles2 = 2 * np.pi / 2 * np.arange(0, 2)
            r1 = Rotation.from_neo_euler(AxAngle.from_axes_angles(axis111, angles1))
            r2 = Rotation.from_neo_euler(AxAngle.from_axes_angles(a1, angles2))
            r = r1.outer(r2)
        elif pg.name == "432":
            angles1 = 2 * np.pi / 3 * np.arange(0, 3)
            angles2 = 2 * np.pi / 2 * np.arange(0, 2)
            r1 = Rotation.from_neo_euler(AxAngle.from_axes_angles(axis111, angles1))
            r2 = Rotation.from_neo_euler(AxAngle.from_axes_angles(axis110, angles2))
            r = r1.outer(r2)
        return r.flatten()

    #    @property
    #    def rotations_not_about_primary_axis2(self):
    #        pg = self.proper_subgroup
    #        dp = np.abs(pg.primary_axis.dot(pg.axis).data)
    #        about_primary_axis = np.logical_or(np.isclose(dp, 1), np.isclose(dp, 0))
    #        return Rotation(pg[~about_primary_axis])
    ##        r = Rotation(pg[~about_primary_axis])
    ##        r = r.outer(r).unique()
    ##        return r
    #
    #    @property
    #    def max_euler_alpha(self):
    #        return 2 * np.pi
    #
    #    @property
    #    def max_euler_beta(self):
    #        return np.pi / np.min([2, self.proper_subgroup.secondary_axis_order])
    #
    #    @property
    #    def max_euler_gamma(self):
    #        return 2 * np.pi / self.proper_subgroup.primary_axis_order
    #
    #    @property
    #    def is_primary_axis_rotation(self):
    #        parallel_with_e3 = np.isclose(np.abs(self.axis.dot(self.primary_axis).data), 1)
    #        not_identity = self.angle.data != 0
    #        return parallel_with_e3 * not_identity
    #
    #    @property
    #    def primary_axis_rotation(self):
    #        """Rotations with their axis parallel with e3, excluding the
    #        identity element.
    #        """
    #        return self[self.is_primary_axis_rotation]
    #
    #    @property
    #    def is_secondary_axis_rotation(self):
    #        vz = Vector3d.zvector()
    #        perpendicular_to_e3 = np.isclose(np.abs(self.axis.dot(vz).data), 0)
    #        not_identity = self.angle.data != 0
    #        return perpendicular_to_e3 * not_identity
    #
    #    @property
    #    def secondary_axis_rotation(self):
    #        """Rotations with their axis perpendicular to e3, excluding the
    #        identity element.
    #        """
    #        return self[self.is_secondary_axis_rotation]
    #
    #    @property
    #    def primary_axis_order(self):
    #        """Order (multiplicity) of the primary rotation axis."""
    #        r = self.primary_axis_rotation
    #        if r.size == 0:
    #            return 1
    #        else:
    #            # Get smallest angle about this axis or axes
    #            smallest_angle = np.min(np.abs(r.angle.data))
    #            return int(2 * np.pi / smallest_angle)

    #    @property
    #    def secondary_axis_order(self):
    #        """Order (multiplicity) of the secondary rotation axis."""
    #        r = self.secondary_axis_rotation
    #        if r.size == 0:
    #            return 1
    #        else:
    #            # Get smallest angle about this axis or axes
    #            smallest_angle = np.min(np.abs(r.angle.data))
    #            return int(2 * np.pi / smallest_angle)

    @property
    def _tuples(self):
        """Set of tuple : the differentiators of this group."""
        s = Rotation(self.flatten())
        tuples = set([tuple(d) for d in s._differentiators()])
        return tuples

    @classmethod
    def from_generators(cls, *generators):
        """Create a Symmetry from a minimum list of generating
        transformations.

        Parameters
        ----------
        generators : Rotation
            An arbitrary list of constituent transformations.

        Returns
        -------
        Symmetry

        Examples
        --------
        Combining a 180° rotation about [1, -1, 0] with a 4-fold
        rotoinversion axis along [0, 0, 1]

        >>> myC2 = Symmetry([(1, 0, 0, 0), (0, 0.75**0.5, -0.75**0.5, 0)])
        >>> myS4 = Symmetry([(1, 0, 0, 0), (0.5**0.5, 0, 0, 0.5**0.5)])
        >>> myS4.improper = [0, 1]
        >>> mySymmetry = Symmetry.from_generators(myC2, myS4)
        >>> mySymmetry
        Symmetry (8,)
        [[ 1.      0.      0.      0.    ]
         [ 0.      0.7071 -0.7071  0.    ]
         [ 0.7071  0.      0.      0.7071]
         [ 0.      0.     -1.      0.    ]
         [ 0.      1.      0.      0.    ]
         [-0.7071  0.      0.      0.7071]
         [ 0.      0.      0.      1.    ]
         [ 0.     -0.7071 -0.7071  0.    ]]
        """
        generator = cls((1, 0, 0, 0))
        for g in generators:
            generator = generator.outer(Symmetry(g)).unique()
        size = 1
        size_new = generator.size
        while size_new != size and size_new < 48:
            size = size_new
            generator = generator.outer(generator).unique()
            size_new = generator.size
        return generator

    def fundamental_sector(self):
        from orix.vector.neo_euler import AxAngle
        from orix.vector.spherical_region import SphericalRegion

        symmetry = self.antipodal
        symmetry = symmetry[symmetry.angle > 0]
        axes, order = symmetry.get_highest_order_axis()
        if order > 6:
            return Vector3d.empty()
        axis = Vector3d.zvector().get_nearest(axes, inclusive=True)
        r = Rotation.from_neo_euler(AxAngle.from_axes_angles(axis, 2 * np.pi / order))

        diads = symmetry.diads
        nearest_diad = axis.get_nearest(diads)
        if nearest_diad.size == 0:
            nearest_diad = axis.perpendicular

        n1 = axis.cross(nearest_diad).unit
        n2 = -(r * n1)
        next_diad = r * nearest_diad
        n = Vector3d.stack((n1, n2)).flatten()
        sr = SphericalRegion(n.unique())
        inside = symmetry[symmetry.axis < sr]
        if inside.size == 0:
            return sr
        axes, order = inside.get_highest_order_axis()
        axis = axis.get_nearest(axes)
        r = Rotation.from_neo_euler(AxAngle.from_axes_angles(axis, 2 * np.pi / order))
        nearest_diad = next_diad
        n1 = axis.cross(nearest_diad).unit
        n2 = -(r * n1)
        n = Vector3d(np.concatenate((n.data, n1.data, n2.data)))
        sr = SphericalRegion(n.unique())
        return sr

    def get_axis_orders(self):
        s = self[self.angle > 0]
        if s.size == 0:
            return {}
        return {
            Vector3d(a): b + 1
            for a, b in zip(*np.unique(s.axis.data, axis=0, return_counts=True))
        }

    def get_highest_order_axis(self):
        axis_orders = self.get_axis_orders()
        if len(axis_orders) == 0:
            return Vector3d.zvector(), np.infty
        highest_order = max(axis_orders.values())
        axes = Vector3d.stack(
            [ao for ao in axis_orders if axis_orders[ao] == highest_order]
        ).flatten()
        return axes, highest_order


# Triclinic
C1 = Symmetry((1, 0, 0, 0))
C1.name = "1"
Ci = Symmetry([(1, 0, 0, 0), (1, 0, 0, 0)])
Ci.improper = [0, 1]
Ci.name = "-1"

# Special generators
_mirror_xy = Symmetry([(1, 0, 0, 0), (0, 0.75 ** 0.5, -(0.75 ** 0.5), 0)])
_mirror_xy.improper = [0, 1]
_cubic = Symmetry([(1, 0, 0, 0), (0.5, 0.5, 0.5, 0.5)])

# 2-fold rotations
C2x = Symmetry([(1, 0, 0, 0), (0, 1, 0, 0)])
C2x.name = "211"
C2y = Symmetry([(1, 0, 0, 0), (0, 0, 1, 0)])
C2y.name = "121"
C2z = Symmetry([(1, 0, 0, 0), (0, 0, 0, 1)])
C2z.name = "112"
C2 = Symmetry(C2z)
C2.name = "2"

# Mirrors
Csx = Symmetry([(1, 0, 0, 0), (0, 1, 0, 0)])
Csx.improper = [0, 1]
Csx.name = "m11"
Csy = Symmetry([(1, 0, 0, 0), (0, 0, 1, 0)])
Csy.improper = [0, 1]
Csy.name = "1m1"
Csz = Symmetry([(1, 0, 0, 0), (0, 0, 0, 1)])
Csz.improper = [0, 1]
Csz.name = "11m"
Cs = Symmetry(Csz)
Cs.name = "m"

# Monoclinic
C2h = Symmetry.from_generators(C2, Cs)
C2h.name = "2/m"

# Orthorhombic
D2 = Symmetry.from_generators(C2z, C2x, C2y)
D2.name = "222"
C2v = Symmetry.from_generators(C2x, Csz)
C2v.name = "mm2"
D2h = Symmetry.from_generators(Csz, Csx, Csy)
D2h.name = "mmm"

# 4-fold rotations
C4x = Symmetry(
    [
        (1, 0, 0, 0),
        (0.5 ** 0.5, 0.5 ** 0.5, 0, 0),
        (0, 1, 0, 0),
        (-(0.5 ** 0.5), 0.5 ** 0.5, 0, 0),
    ]
)
C4y = Symmetry(
    [
        (1, 0, 0, 0),
        (0.5 ** 0.5, 0, 0.5 ** 0.5, 0),
        (0, 0, 1, 0),
        (-(0.5 ** 0.5), 0, 0.5 ** 0.5, 0),
    ]
)
C4z = Symmetry(
    [
        (1, 0, 0, 0),
        (0.5 ** 0.5, 0, 0, 0.5 ** 0.5),
        (0, 0, 0, 1),
        (-(0.5 ** 0.5), 0, 0, 0.5 ** 0.5),
    ]
)
C4 = Symmetry(C4z)
C4.name = "4"

# Tetragonal
S4 = Symmetry.from_generators(C2, Ci)
S4.name = "-4"
C4h = Symmetry.from_generators(C4, Cs)
C4h.name = "4/m"
D4 = Symmetry.from_generators(C4, C2x, C2y)
D4.name = "422"
C4v = Symmetry.from_generators(C4, Csx)
C4v.name = "4mm"
D2d = Symmetry.from_generators(D2, _mirror_xy)
D2d.name = "-42m"
D4h = Symmetry.from_generators(C4h, Csx, Csy)
D4h.name = "4/mmm"

# 3-fold rotations
C3x = Symmetry([(1, 0, 0, 0), (0.5, 0.75 ** 0.5, 0, 0), (-0.5, 0.75 ** 0.5, 0, 0)])
C3y = Symmetry([(1, 0, 0, 0), (0.5, 0, 0.75 ** 0.5, 0), (-0.5, 0, 0.75 ** 0.5, 0)])
C3z = Symmetry([(1, 0, 0, 0), (0.5, 0, 0, 0.75 ** 0.5), (-0.5, 0, 0, 0.75 ** 0.5)])
C3 = Symmetry(C3z)
C3.name = "3"

# Trigonal
S6 = Symmetry.from_generators(C3, Ci)
S6.name = "-3"
D3x = Symmetry.from_generators(C3, C2x)
D3x.name = "321"
D3y = Symmetry.from_generators(C3, C2y)
D3y.name = "312"
D3 = Symmetry(D3x)
D3.name = "32"
C3v = Symmetry.from_generators(C3, Csx)
C3v.name = "3m"
D3d = Symmetry.from_generators(S6, Csx)
D3d.name = "-3m"

# Hexagonal
C6 = Symmetry.from_generators(C3, C2)
C6.name = "6"
C3h = Symmetry.from_generators(C3, Cs)
C3h.name = "-6"
C6h = Symmetry.from_generators(C6, Cs)
C6h.name = "6/m"
D6 = Symmetry.from_generators(C6, C2x, C2y)
D6.name = "622"
C6v = Symmetry.from_generators(C6, Csx)
C6v.name = "6mm"
D3h = Symmetry.from_generators(C3, C2y, Csz)
D3h.name = "-6m2"
D6h = Symmetry.from_generators(D6, Csz)
D6h.name = "6/mmm"

# Cubic
T = Symmetry.from_generators(C2, _cubic)
T.name = "23"
Th = Symmetry.from_generators(T, Ci)
Th.name = "m-3"
O = Symmetry.from_generators(C4, _cubic, C2x)
O.name = "432"
Td = Symmetry.from_generators(T, _mirror_xy)
Td.name = "-43m"
Oh = Symmetry.from_generators(O, Ci)
Oh.name = "m-3m"


# Collections of groups for convenience
_groups = [
    # Triclinic
    C1,
    Ci,
    # Monoclinic
    C2x,
    C2y,
    C2z,
    Csx,
    Csy,
    Csz,
    C2h,
    # Orthorhombic
    D2,
    C2v,
    D2h,
    # Tetragonal
    C4,
    S4,
    C4h,
    D4,
    C4v,
    D2d,
    D4h,
    # Trigonal
    C3,
    S6,
    D3x,
    D3y,
    D3,
    C3v,
    D3d,
    # Hexagonal
    C6,
    C3h,
    C6h,
    D6,
    C6v,
    D3h,
    D6h,
    # Cubic
    T,
    Th,
    O,
    Td,
    Oh,
]
_proper_groups = [C1, C2, C2x, C2y, C2z, D2, C4, D4, C3, D3x, D3y, D3, C6, D6, T, O]
groups_dict = dict(zip([g.name for g in _groups], _groups))
proper_groups_dict = dict(zip([pg.name for pg in _proper_groups], _proper_groups))


def get_distinguished_points(s1, s2=C1):
    """Points symmetrically equivalent to identity with respect to `s1`
    and `s2`.

    Parameters
    ----------
    s1, s2 : Symmetry

    Returns
    -------
    Rotation
    """
    distinguished_points = s1.outer(s2).antipodal.unique(antipodal=False)
    return distinguished_points[distinguished_points.angle > 0]


spacegroup2pointgroup_dict = {
    "PG1": {"proper": C1, "improper": C1},
    "PG1bar": {"proper": C1, "improper": Ci},
    "PG2": {"proper": C2, "improper": C2},
    "PGm": {"proper": C2, "improper": Cs},
    "PG2/m": {"proper": C2, "improper": C2h},
    "PG222": {"proper": D2, "improper": D2},
    "PGmm2": {"proper": C2, "improper": C2v},
    "PGmmm": {"proper": D2, "improper": D2h},
    "PG4": {"proper": C4, "improper": C4},
    "PG4bar": {"proper": C4, "improper": S4},
    "PG4/m": {"proper": C4, "improper": C4h},
    "PG422": {"proper": D4, "improper": D4},
    "PG4mm": {"proper": C4, "improper": C4v},
    "PG4bar2m": {"proper": D4, "improper": D2d},
    "PG4barm2": {"proper": D4, "improper": D2d},
    "PG4/mmm": {"proper": D4, "improper": D4h},
    "PG3": {"proper": C3, "improper": C3},
    "PG3bar": {"proper": C3, "improper": S6},  # Improper also known as C3i
    "PG312": {"proper": D3, "improper": D3},
    "PG321": {"proper": D3, "improper": D3},
    "PG3m1": {"proper": C3, "improper": C3v},
    "PG31m": {"proper": C3, "improper": C3v},
    "PG3m": {"proper": C3, "improper": C3v},
    "PG3bar1m": {"proper": D3, "improper": D3d},
    "PG3barm1": {"proper": D3, "improper": D3d},
    "PG3barm": {"proper": D3, "improper": D3d},
    "PG6": {"proper": C6, "improper": C6},
    "PG6bar": {"proper": C6, "improper": C3h},
    "PG6/m": {"proper": C6, "improper": C6h},
    "PG622": {"proper": D6, "improper": D6},
    "PG6mm": {"proper": C6, "improper": C6v},
    "PG6barm2": {"proper": D6, "improper": D3h},
    "PG6bar2m": {"proper": D6, "improper": D3h},
    "PG6/mmm": {"proper": D6, "improper": D6h},
    "PG23": {"proper": T, "improper": T},
    "PGm3bar": {"proper": T, "improper": Th},
    "PG432": {"proper": O, "improper": O},
    "PG4bar3m": {"proper": T, "improper": Td},
    "PGm3barm": {"proper": O, "improper": Oh},
}


def get_point_group(space_group_number, proper=False):
    """Maps a space group number to its (proper) point group.

    Parameters
    ----------
    space_group_number : int
        Between 1 and 231.
    proper : bool, optional
        Whether to return the point group with proper rotations only
        (True), or just the point group (False). Default is False.

    Returns
    -------
    point_group : orix.quaternion.symmetry.Symmetry
        One of the 11 proper or 32 point groups.

    Examples
    --------
    >>> from orix.quaternion.symmetry import get_point_group
    >>> pgOh = get_point_group(225)
    >>> pgOh.name
    'm-3m'
    >>> pgO = get_point_group(225, proper=True)
    >>> pgO.name
    '432'
    """
    spg = GetSpaceGroup(space_group_number)
    pgn = spg.point_group_name
    if proper:
        return spacegroup2pointgroup_dict[pgn]["proper"]
    else:
        return spacegroup2pointgroup_dict[pgn]["improper"]


# Point group alias mapping. This is needed because in EDAX TSL OIM
# Analysis 7.2, e.g. point group 432 is entered as 43.
# Used when reading a phase's point group from an EDAX ANG file header
point_group_aliases = {
    "121": ["20",],
    "2/m": ["2"],
    "222": ["22",],
    "422": ["42"],
    "432": ["43",],
    "m-3m": ["m3m"],
}
