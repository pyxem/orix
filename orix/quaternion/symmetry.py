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
        data = np.array_str(self.data, precision=4, suppress_small=True)
        return f"{self.__class__.__name__} {self.shape} {self.name}\n{data}"

    def __and__(self, other):
        generators = [g for g in self.subgroups if g in other.subgroups]
        return Symmetry.from_generators(*generators)

    @property
    def order(self):
        """Number of elements of the group as :class:`int`."""
        return self.size

    @property
    def is_proper(self):
        """Whether this group contains only proper rotations as
        :class:`bool`.
        """
        return np.all(np.equal(self.improper, 0))

    @property
    def subgroups(self):
        """List groups that are subgroups of this group as a
        :class:`list` of :class:`Symmetry`.
        """
        return [g for g in _groups if g._tuples <= self._tuples]

    @property
    def proper_subgroups(self):
        """List of proper groups that are subgroups of this group as a
        :class:`list` of :class:`Symmetry`.
        """
        return [g for g in self.subgroups if g.is_proper]

    @property
    def proper_subgroup(self):
        """The largest proper group of this subgroup as a
        :class:`Symmetry`.
        """
        subgroups = self.proper_subgroups
        if len(subgroups) == 0:
            return Symmetry(self)
        else:
            subgroups_sorted = sorted(subgroups, key=lambda g: g.order)
            return subgroups_sorted[-1]

    @property
    def laue(self):
        """This group plus inversion as a :class:`Symmetry`."""
        laue = Symmetry.from_generators(self, Ci)
        laue.name = _get_laue_group_name(self.name)
        return laue

    @property
    def laue_proper_subgroup(self):
        """The proper subgroup of this group plus inversion as a
        :class:`Symmetry`.
        """
        return self.laue.proper_subgroup

    @property
    def contains_inversion(self):
        """Whether this group contains inversion as a :class:`bool`."""
        return Ci._tuples <= self._tuples

    @property
    def diads(self):
        """Diads of this symmetry as a set of
        :class:`~orix.vector.Vector3d`.
        """
        axis_orders = self.get_axis_orders()
        diads = [ao for ao in axis_orders if axis_orders[ao] == 2]
        if len(diads) == 0:
            return Vector3d.empty()
        else:
            return Vector3d.stack(diads).flatten()

    @property
    def euler_fundamental_region(self):
        r"""Fundamental Euler angle region of the proper subgroup.

        Returns
        -------
        region : tuple
            Maximum Euler angles :math:`(\phi_{1, max}, \Phi_{max},
            \phi_{2, max})` in degrees. No symmetry is assumed if the
            proper subgroup name is not recognized.
        """
        # fmt: off
        angles = {
              "1": (360, 180, 360),  # Triclinic
            "211": (360,  90, 360),  # Monoclinic
            "121": (360,  90, 360),
            "112": (360, 180, 180),
            "222": (360,  90, 180),  # Orthorhombic
              "4": (360, 180,  90),  # Tetragonal
            "422": (360,  90,  90),
              "3": (360, 180, 120),  # Trigonal
            "312": (360,  90, 120),
             "32": (360,  90, 120),
              "6": (360, 180,  60),  # Hexagonal
            "622": (360,  90,  60),
             "23": (360,  90, 180),  # Cubic
            "432": (360,  90,  90),
        }
        # fmt: on
        proper_subgroup_name = self.proper_subgroup.name
        if proper_subgroup_name in angles.keys():
            region = angles[proper_subgroup_name]
        else:
            region = angles["1"]
        return region

    @property
    def system(self):
        """Which of the seven crystal systems this symmetry belongs to.

        Returns
        -------
        str or None
            None is returned if the symmetry name is not recognized.
        """
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
    def _tuples(self):
        """Set of tuple : the differentiators of this group."""
        s = Rotation(self.flatten())
        tuples = set([tuple(d) for d in s._differentiators()])
        return tuples

    @property
    def fundamental_sector(self):
        """:class:`~orix.vector.FundamentalSector` describing the
        inverse pole figure given by the point group name.

        These sectors are taken from MTEX'
        :code:`crystalSymmetry.fundamentalSector`.
        """
        # Avoid circular import
        from orix.vector import FundamentalSector

        name = self.name
        vx = Vector3d.xvector()
        vy = Vector3d.yvector()
        vz = Vector3d.zvector()

        # Map everything on the northern hemisphere if there is an
        # inversion or some symmetry operation not parallel to Z
        if any(vz.angle_with(self.outer(vz)) > np.pi / 2):
            n = vz
        else:
            n = Vector3d.empty()

        # Region on the northern hemisphere depends just on the number
        # of symmetry operations
        if self.size > 1 + n.size:
            angle = 2 * np.pi * (1 + n.size) / self.size
            new_v = Vector3d.from_polar(
                azimuth=[np.pi / 2, angle - np.pi / 2], polar=[np.pi / 2, np.pi / 2]
            )
            n = Vector3d(np.vstack([n.data, new_v.data]))

        # We only set the center "by hand" for T (23), Th (m-3) and O
        # (432), since the UV S2 sampling isn't uniform enough to
        # produce the correct center according to MTEX
        center = None

        # Override normal(s) for some point groups
        if name == "-1":
            n = vz
        elif name in ["m11", "1m1", "11m"]:
            idx_min_angle = np.argmin(self.angle.data)
            n = self[idx_min_angle].axis
            if name == "m11":
                n = -n
        elif name == "mm2":
            n = self[self.improper].axis  # Mirror planes
            idx = n.angle_with(-vy) < np.pi / 4
            n[idx] = -n[idx]
        elif name in ["321", "312", "3m", "-3m", "6m2"]:
            n = n.rotate(angle=-np.pi / 6)
        elif name == "-42m":
            n = n.rotate(angle=-np.pi / 4)
        elif name == "23":
            n = Vector3d([[1, 1, 0], [1, -1, 0], [0, -1, 1], [0, 1, 1]])
            # Taken from MTEX
            center = Vector3d([0.707558, -0.000403, 0.706655])
        elif name in ["m-3", "432"]:
            n = Vector3d(np.vstack([vx.data, [0, -1, 1], [-1, 0, 1], vy.data, vz.data]))
            # Taken from MTEX
            center = Vector3d([0.349928, 0.348069, 0.869711])
        elif name == "-43m":
            n = Vector3d([[1, -1, 0], [1, 1, 0], [-1, 0, 1]])
        elif name == "m-3m":
            n = Vector3d(np.vstack([[1, -1, 0], [-1, 0, 1], vy.data]))

        fs = FundamentalSector(n).flatten().unique()
        fs._center = center

        return fs

    @property
    def _primary_axis_order(self):
        """Order of primary rotation axis for the proper subgroup.

        Used in to map Euler angles into the fundamental region in
        :meth:`~orix.quaternion.Orientation.in_euler_fundamental_region`.

         Returns
         -------
         int or None
            None is returned if the proper subgroup name is not
            recognized.
        """
        # TODO: Find this dynamically
        name = self.proper_subgroup.name
        if name in ["1", "211", "121"]:
            return 1
        elif name in ["112", "222", "23"]:
            return 2
        elif name in ["3", "312", "32"]:
            return 3
        elif name in ["4", "422", "432"]:
            return 4
        elif name in ["6", "622"]:
            return 6
        else:
            return None

    @property
    def _special_rotation(self):
        """Symmetry operations of the proper subgroup different from
        rotation about the c-axis.

        Used in to map Euler angles into the fundamental region in
        :meth:`~orix.quaternion.Orientation.in_euler_fundamental_region`.

        These sectors are taken from MTEX'
        :code:`Symmetry.rotation_special`.

        Returns
        -------
        rot : Rotation
            The identity rotation is returned if the proper subgroup
            name is not recognized.
        """

        def symmetry_axis(v, n):
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            return Rotation.from_neo_euler(AxAngle.from_axes_angles(v, angles))

        # Symmetry axes
        vx = Vector3d.xvector()
        mirror = Vector3d((1, -1, 0))
        axis110 = Vector3d((1, 1, 0))
        axis111 = Vector3d((1, 1, 1))

        name = self.proper_subgroup.name
        if name in ["1", "211", "121"]:
            # All proper operations
            rot = self[~self.improper]
        elif name in ["112", "3", "4", "6"]:
            # Identity
            rot = self[0]
        elif name in ["222", "422", "622", "32"]:
            # Two-fold rotation about a-axis perpendicular to c-axis
            rot = symmetry_axis(-vx, 2)
        elif name == "312":
            # Mirror plane perpendicular to c-axis?
            rot = symmetry_axis(-mirror, 2)
        elif name in ["23", "432"]:
            # Three-fold rotation about [111]
            rot = symmetry_axis(-axis111, 3)
            if name == "23":
                # Combined with two-fold rotation about a-axis
                rot = rot.outer(symmetry_axis(-vx, 2))
            else:
                # Combined with two-fold rotation about [110]
                rot = rot.outer(symmetry_axis(-axis110, 2))
        else:
            rot = Rotation.identity((1,))

        return rot.flatten()

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
            generator = generator.outer(cls(g)).unique()
        size = 1
        size_new = generator.size
        while size_new != size and size_new < 48:
            size = size_new
            generator = generator.outer(generator).unique()
            size_new = generator.size
        return generator

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

    def fundamental_zone(self):
        from orix.vector import AxAngle, SphericalRegion

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
# fmt: off
_groups = [
    # Schoenflies   Crystal system  International   Laue class  Proper point group
    C1,   #         Triclinic        1              -1          1
    Ci,   #         Triclinic       -1              -1          1
    C2x,  #         Monoclinic       211             2/m        211
    C2y,  #         Monoclinic       121             2/m        121
    C2z,  #         Monoclinic       112             2/m        112
    Csx,  #         Monoclinic       m11             2/m        1
    Csy,  #         Monoclinic       1m1             2/m        1
    Csz,  #         Monoclinic       11m             2/m        1
    C2h,  #         Monoclinic       2/m             2/m        112
    D2,   #         Orthorhombic     222             mmm        222
    C2v,  #         Orthorhombic     mm2             mmm        211
    D2h,  #         Orthorhombic     mmm             mmm        222
    C4,   #         Tetragonal       4               4/m        4
    S4,   #         Tetragonal      -4               4/m        112
    C4h,  #         Tetragonal       4/m             4/m        4
    D4,   #         Tetragonal       422             4/mmm      422
    C4v,  #         Tetragonal       4mm             4/mmm      4
    D2d,  #         Tetragonal      -42m             4/mmm      222
    D4h,  #         Tetragonal       4/mmm           4/mmm      422
    C3,   #         Trigonal         3              -3          3
    S6,   #         Trigonal        -3              -3          3
    D3x,  #         Trigonal         321            -3m         32
    D3y,  #         Trigonal         312            -3m         312
    D3,   #         Trigonal         32             -3m         32
    C3v,  #         Trigonal         3m             -3m         3
    D3d,  #         Trigonal        -3m             -3m         32
    C6,   #         Hexagonal        6               6/m        6
    C3h,  #         Hexagonal       -6               6/m        6
    C6h,  #         Hexagonal        6/m             6/m        622
    D6,   #         Hexagonal        622             6/mmm      622
    C6v,  #         Hexagonal        6mm             6/mmm      6
    D3h,  #         Hexagonal       -6m2             6/mmm      312
    D6h,  #         Hexagonal        6/mmm           6/mmm      622
    T,    #         Cubic            23              m-3        23
    Th,   #         Cubic            m-3             m-3        23
    O,    #         Cubic            432             m-3m       432
    Td,   #         Cubic           -43m             m-3m       23
    Oh,   #         Cubic            m-3m            m-3m       432
]
# fmt: on
_proper_groups = [C1, C2, C2x, C2y, C2z, D2, C4, D4, C3, D3x, D3y, D3, C6, D6, T, O]


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
    "121": ["20"],
    "2/m": ["2"],
    "222": ["22"],
    "422": ["42"],
    "432": ["43"],
    "m-3m": ["m3m"],
}


def _get_laue_group_name(name):
    if name in ["1", "-1"]:
        return "-1"
    elif name in ["2", "211", "121", "112", "m11", "1m1", "11m", "2/m"]:
        return "2/m"
    elif name in ["222", "mm2", "mmm"]:
        return "mmm"
    elif name in ["4", "-4", "4/m"]:
        return "4/m"
    elif name in ["422", "4mm", "-42m", "4/mmm"]:
        return "4/mmm"
    elif name in ["3", "-3"]:
        return "-3"
    elif name in ["321", "312", "32", "3m", "-3m"]:
        return "-3m"
    elif name in ["6", "-6", "6/m"]:
        return "6/m"
    elif name in ["6mm", "-6m2", "6/mmm"]:
        return "6/mmm"
    elif name in ["23", "m-3"]:
        return "m-3"
    elif name in ["432", "-43m", "m-3m"]:
        return "m-3m"
    else:
        return None
