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

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from diffpy.structure.spacegroups import GetSpaceGroup
import matplotlib.figure as mfigure
import numpy as np

from orix._util import deprecated
from orix.quaternion.rotation import Rotation
from orix.vector import Vector3d

if TYPE_CHECKING:  # pragma: no cover
    from orix.quaternion import Orientation
    from orix.vector import FundamentalSector


class Symmetry(Rotation):
    r"""The set of rotations comprising a point group.

    An object's symmetry can be characterized by the transformations
    relating symmetrically-equivalent views on that object. Consider
    the following shape.

    .. image:: /_static/img/triad-object.png
       :width: 200px
       :alt: Image of an object with three-fold symmetry.
       :align: center

    This obviously has three-fold symmetry. If we rotated it by
    :math:`\frac{2}{3}\pi` or :math:`\frac{4}{3}\pi`, the image
    would be unchanged. These angles, as well as :math:`0`, or the
    identity, expressed as quaternions, form a group. Applying any
    operation in the group to any other results in another member of the
    group.

    Symmetries can consist of rotations or inversions, expressed as
    improper rotations. A mirror symmetry is equivalent to a 2-fold
    rotation combined with inversion.
    """

    name = ""

    # -------------------------- Properties -------------------------- #

    @property
    def order(self) -> int:
        """Return the number of elements of the group."""
        return self.size

    @property
    def is_proper(self) -> bool:
        """Return whether this group contains only proper rotations."""
        return bool(np.all(np.equal(self.improper, 0)))

    @property
    def subgroups(self) -> list[Symmetry]:
        """Return the list groups that are subgroups of this group."""
        groups = get_point_groups("all")
        return [g for g in groups if g._tuples <= self._tuples]

    @property
    def proper_subgroups(self) -> list[Symmetry]:
        """Return the list of proper groups that are subgroups of this
        group.
        """
        return [g for g in self.subgroups if g.is_proper]

    @property
    def proper_subgroup(self) -> Symmetry:
        """Return the largest proper group of this subgroup."""
        subgroups = self.proper_subgroups
        if len(subgroups) == 0:
            return Symmetry(self)
        else:
            subgroups_sorted = sorted(subgroups, key=lambda g: g.order)
            return subgroups_sorted[-1]

    @property
    def laue(self) -> Symmetry:
        """Return this group plus inversion."""
        laue = Symmetry.from_generators(self, Ci)
        laue.name = _get_laue_group_name(self.name)
        return laue

    @property
    def laue_proper_subgroup(self) -> Symmetry:
        """Return the proper subgroup of this group plus inversion."""
        return self.laue.proper_subgroup

    @property
    def contains_inversion(self) -> bool:
        """Return whether this group contains inversion."""
        return Ci._tuples <= self._tuples

    @property
    def diads(self) -> Vector3d:
        """Return the diads of this symmetry."""
        axis_orders = self.get_axis_orders()
        diads = [ao for ao in axis_orders if axis_orders[ao] == 2]
        if len(diads) == 0:
            return Vector3d.empty()
        else:
            return Vector3d.stack(diads).flatten()

    @property
    def euler_fundamental_region(self) -> tuple:
        r"""Return the fundamental Euler angle region of the proper
        subgroup.

        Returns
        -------
        region
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
              "2": (360, 180, 180),
            "222": (360,  90, 180),  # Orthorhombic
              "4": (360, 180,  90),  # Tetragonal
            "422": (360,  90,  90),
              "3": (360, 180, 120),  # Trigonal
            "321": (360,  90, 120),
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
    def system(self) -> str | None:
        """Return which of the seven crystal systems this symmetry
        belongs to.

        Returns
        -------
        system
            ``None`` is returned if the symmetry name is not recognized.
        """
        # fmt: off
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
        # fmt: on

    @property
    def _tuples(self) -> set:
        """Return the differentiators of this group."""
        s = Rotation(self.flatten())
        tuples = set([tuple(d) for d in s._differentiators()])
        return tuples

    @property
    def fundamental_sector(self) -> "FundamentalSector":
        """Return the fundamental sector describing the inverse pole
        figure given by the point group name.

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
                azimuth=[np.pi / 2, angle - np.pi / 2],
                polar=[np.pi / 2, np.pi / 2],
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
            idx_min_angle = np.argmin(self.angle)
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
    def _primary_axis_order(self) -> int | None:
        """Return the order of primary rotation axis for the proper
        subgroup.

        Used in to map Euler angles into the fundamental region in
        :meth:`~orix.quaternion.Orientation.in_euler_fundamental_region`.

        Returns
        -------
        order
            ``None`` is returned if the proper subgroup name is not
            recognized.
        """
        # TODO: Find this dynamically
        name = self.proper_subgroup.name
        if name in ["1", "211", "121"]:
            return 1
        elif name in ["112", "222", "23", "2"]:
            return 2
        elif name in ["3", "312", "321", "32"]:
            return 3
        elif name in ["4", "422", "432"]:
            return 4
        elif name in ["6", "622"]:
            return 6
        else:
            return None

    @property
    def _special_rotation(self) -> Rotation:
        """Symmetry operations of the proper subgroup different from
        rotation about the c-axis.

        Used in to map Euler angles into the fundamental region in
        :meth:`~orix.quaternion.Orientation.in_euler_fundamental_region`.

        These sectors are taken from MTEX'
        :code:`Symmetry.rotation_special`.

        Returns
        -------
        rot
            The identity rotation is returned if the proper subgroup
            name is not recognized.
        """

        def symmetry_axis(v: Vector3d, n: int) -> Rotation:
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            return Rotation.from_axes_angles(v, angles)

        # Symmetry axes
        vx = Vector3d.xvector()
        mirror = Vector3d((1, -1, 0))
        axis110 = Vector3d((1, 1, 0))
        axis111 = Vector3d((1, 1, 1))

        name = self.proper_subgroup.name
        if name in ["1", "211", "121"]:
            # All proper operations
            rot = self[~self.improper]
        elif name in ["2", "112", "3", "4", "6"]:
            # Identity
            rot = self[0]
        elif name in ["222", "422", "622", "32", "321"]:
            # Two-fold rotation about a-axis perpendicular to c-axis
            rot = symmetry_axis(-vx, 2)
        elif name == "312":
            # Mirror plane perpendicular to c-axis
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

    # ------------------------ Dunder methods ------------------------ #

    def __repr__(self) -> str:
        data = np.array_str(self.data, precision=4, suppress_small=True)
        return f"{self.__class__.__name__} {self.shape} {self.name}\n{data}"

    def __and__(self, other: Symmetry) -> Symmetry:
        generators = [g for g in self.subgroups if g in other.subgroups]
        return Symmetry.from_generators(*generators)

    def __hash__(self) -> int:
        return hash(self.name.encode() + self.data.tobytes() + self.improper.tobytes())

    # ------------------------ Class methods ------------------------- #

    @classmethod
    def from_generators(cls, *generators: Rotation) -> Symmetry:
        """Create a Symmetry from a minimum list of generating
        transformations.

        Parameters
        ----------
        *generators
            An arbitrary list of constituent transformations.

        Returns
        -------
        sym

        Examples
        --------
        Combining a 180° rotation about [1, -1, 0] with a 4-fold
        rotoinversion axis along [0, 0, 1]

        >>> from orix.quaternion import Symmetry
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

    # --------------------- Other public methods --------------------- #

    def get_axis_orders(self) -> dict[Vector3d, int]:
        s = self[self.angle > 0]
        if s.size == 0:
            return {}
        return {
            Vector3d(a): b + 1
            for a, b in zip(*np.unique(s.axis.data, axis=0, return_counts=True))
        }

    def get_highest_order_axis(self) -> tuple[Vector3d, np.ndarray]:
        axis_orders = self.get_axis_orders()
        if len(axis_orders) == 0:
            return Vector3d.zvector(), np.inf
        highest_order = max(axis_orders.values())
        axes = Vector3d.stack(
            [ao for ao in axis_orders if axis_orders[ao] == highest_order]
        ).flatten()
        return axes, highest_order

    def fundamental_zone(self) -> Vector3d:
        from orix.vector import SphericalRegion

        symmetry = self.antipodal
        symmetry = symmetry[symmetry.angle > 0]
        axes, order = symmetry.get_highest_order_axis()
        if order > 6:
            return Vector3d.empty()
        axis = Vector3d.zvector().get_nearest(axes, inclusive=True)
        r = Rotation.from_axes_angles(axis, 2 * np.pi / order)

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
        r = Rotation.from_axes_angles(axis, 2 * np.pi / order)
        nearest_diad = next_diad
        n1 = axis.cross(nearest_diad).unit
        n2 = -(r * n1)
        n = Vector3d(np.concatenate((n.data, n1.data, n2.data)))
        sr = SphericalRegion(n.unique())
        return sr

    def plot(
        self,
        orientation: "Orientation | None" = None,
        reproject_scatter_kwargs: dict | None = None,
        **kwargs,
    ) -> mfigure.Figure | None:
        """Stereographic projection of symmetry operations.

        The upper hemisphere of the stereographic projection is shown.
        Vectors on the lower hemisphere are shown after reprojection
        onto the upper hemisphere.

        Parameters
        ----------
        orientation
            The symmetry operations are applied to this orientation
            before plotting. The default value uses an orientation
            optimized to show symmetry elements.
        reproject_scatter_kwargs
            Dictionary of keyword arguments for the reprojected scatter
            points which is passed to
            :meth:`~orix.plot.StereographicPlot.scatter`, which passes
            these on to :meth:`matplotlib.axes.Axes.scatter`. The
            default marker style for reprojected vectors is "+". Values
            used for vector(s) on the visible hemisphere are used unless
            another value is passed here.
        **kwargs
            Keyword arguments passed to
            :meth:`~orix.plot.StereographicPlot.scatter`, which passes
            these on to :meth:`matplotlib.axes.Axes.scatter`.

        Returns
        -------
        fig
            The created figure, returned if ``return_figure=True`` is
            passed as a keyword argument.
        """
        if orientation is None:
            # orientation chosen to mimic stereographic projections as
            # shown: http://xrayweb.chem.ou.edu/notes/symmetry.html
            orientation = Rotation.from_axes_angles((-1, 8, 1), np.deg2rad(65))
        if not isinstance(orientation, Rotation):
            raise TypeError("Orientation must be a Rotation instance.")
        orientation = self.outer(orientation)

        kwargs.setdefault("return_figure", False)
        return_figure = kwargs.pop("return_figure")

        if reproject_scatter_kwargs is None:
            reproject_scatter_kwargs = {}
        reproject_scatter_kwargs.setdefault("marker", "+")
        reproject_scatter_kwargs.setdefault("label", "lower")

        v = orientation * Vector3d.zvector()

        figure = v.scatter(
            return_figure=True,
            axes_labels=[r"$e_1$", r"$e_2$", None],
            label="upper",
            reproject=True,
            reproject_scatter_kwargs=reproject_scatter_kwargs,
            **kwargs,
        )
        # add symmetry name to figure title
        figure.suptitle(f"${self.name}$")

        if return_figure:
            return figure


# ---------------- Proceedural definitions of Point Groups ---------------- #
# NOTE: ORIX uses Schoenflies symbols to define point groups. This is partly
# because the notation is short and always starts with a letter (ie, they
# make convenient python variables), and partly because it helps limit
# accidental misinterpretation of of Herman-Mauguin symbols as space group
# numbers. For example. "222" could be interpreted as SG#222 == Pn-3n, or
# as PG'222'== D3.  there are similar examples with 2, 3, 4, 32, etc.

# Additionally, there are 43 crystallographically valid Schonflies group
# notations, but only 32 unique ones, meaning certain point groups have
# redundant representations in Schonflies notation(S4==C4i, Ci==S2, S6==C3i,
# and C2==D1, for example). The International Tables of Crystallography,
# Volume A, Section 12.1 defines the 32 standard representations, but a few of
# the commonly used redundant ones are given below for convenience.

# Finally, while there are 32 Point groups, ITOC names several additional
# projections for the non-centrosymmetric groups (ie, using x and/or y as the
# rotation axis instead of z). These are included below as well, following
# the ITOC naming convention (for example, a 2-fold cyclic rotation around
# the x axis instead of the z axis is called C2x.)

# For more details on how point groups can be generated, the following three
# resources lay out three different but equally valid approaches:
#    1)"Structure of Materials", De Graef et al, Section 9.2
#    2)"International Tables of Crystallography: Volume A" Section 12.1
#    3)"Crystallogrpahic Texture and Group Representations", Chi-Sing Man,Ch2

# Triclinic
C1 = Symmetry((1, 0, 0, 0))
C1.name = "1"
Ci = Symmetry([(1, 0, 0, 0), (-1, 0, 0, 0)])
Ci.improper = [0, 1]
Ci.name = "-1"
# include redundant point group S2 == Ci
S2 = Symmetry([(1, 0, 0, 0), (-1, 0, 0, 0)])
S2.improper = [0, 1]
S2.name = "-1"

# Special generators
_mirror_xy = Symmetry([(1, 0, 0, 0), (0, 0.75**0.5, -(0.75**0.5), 0)])
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
# included redundant point group D1 == C2
D1 = Symmetry(C2z)
D1.name = "2"

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
C2v = Symmetry.from_generators(C2z, Csx)
C2v.name = "mm2"
D2h = Symmetry.from_generators(Csz, Csx, Csy)
D2h.name = "mmm"

# 4-fold rotations
C4x = Symmetry(
    [
        (1, 0, 0, 0),
        (0.5**0.5, 0.5**0.5, 0, 0),
        (0, 1, 0, 0),
        ((0.5**0.5), -(0.5**0.5), 0, 0),
    ]
)
C4y = Symmetry(
    [
        (1, 0, 0, 0),
        (0.5**0.5, 0, 0.5**0.5, 0),
        (0, 0, 1, 0),
        ((0.5**0.5), -0, 0.5**0.5, 0),
    ]
)
C4z = Symmetry(
    [
        (1, 0, 0, 0),
        (0.5**0.5, 0, 0, 0.5**0.5),
        (0, 0, 0, 1),
        ((0.5**0.5), 0, 0, -(0.5**0.5)),
    ]
)
C4 = Symmetry(C4z)
C4.name = "4"

# Tetragonal
S4 = Symmetry(C4)
S4.improper = [0, 1, 0, 1]
S4.name = "-4"
# include redundant point group C4i == S4
C4i = Symmetry(C4)
C4i.improper = [0, 1, 0, 1]
C4i.name = "-4"
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
C3x = Symmetry([(1, 0, 0, 0), (0.5, 0.75**0.5, 0, 0), (0.5, -(0.75**0.5), 0, 0)])
C3y = Symmetry([(1, 0, 0, 0), (0.5, 0, 0.75**0.5, 0), (0.5, 0, -(0.75**0.5), 0)])
C3z = Symmetry([(1, 0, 0, 0), (0.5, 0, 0, 0.75**0.5), (0.5, 0, 0, -(0.75**0.5))])
C3 = Symmetry(C3z)
C3.name = "3"

# Trigonal
C3i = Symmetry.from_generators(C3, Ci)
C3i.name = "-3"
# include redundant point group S6==C3i
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


def get_point_groups(subset: str = "unique"):
    """
    returns different subsets of the 32 crystallographic point groups. By
    default, this returns all 32 in the order they appear in the
    International Tables of Crystallography (ITOC).

    Parameters
    ----------
    subset : str, optional
        the point group list to return. The options are as follows:
            "unique" (default):
                All 32 point groups in the order they appear in space groups.
                Thus, they are grouped by crystal system and laue class
            "all":
                All 32 points groups, plus common axis-specific permutations
                for non-centrosymmetric groups (ie, C2 plus C2x and C2y) for
                a total of 37 point group projections. These
                are given in the same order as ITOC Table 10.2.2
            "all_repeated":
                The 37 point group projections, plus the redundant Schonflies
                and Herman-Mauguin group names. For example, both Ci and S2
                are included, as well as D3 =="32" and D3x == "321". NOTE:
                this means several of the entries symmetrically identical.
            "proper":
                The 11 proper point groups given in the same order as ITOC
                table 10.2.2.
                same order as "unique", which in turn aligns with Table 3.1
                of ITOC
            "proper_all":
                The 11 proper point groups, plus axis-specific permutations.
            "laue":
                The point groups corresponding to the 11 Laue groups, using
                the same ordering and definitions as Table 3.1 of ITOC. These
                are equivalent to adding an inversion symmetry to each op
                the 11 proper point groups
            "procedural":
                The 32 point groups, but presented in the procedural ordering
                described in "Structure of Materials" and other books, where
                point groups are created from successive applications of
                symmetry elements to the Cyclic (C_n) and Dihedral (D_n)
                groups.

    Returns
    -------
    point groups: [Symmetry,]
        a list of point group symmetries

    Notes
    -------
    For convenience, the following table shows the 38 point groups contained
    in "all", of which the other lists are subsets of.


    +-------------+--------------+---------------+------------+------------+
    | Schoenflies | System       |     HM        | Laue Class | Proper PG  |
    +=============+==============+===============+============+============+
    | C1          | Triclinic    |      1        |    -1      |    1       |
    +-------------+--------------+---------------+------------+------------+
    | Ci          | Triclinic    |     -1        |    -1      |    1       |
    +-------------+--------------+---------------+------------+------------+
    | C2          | Monoclinic   |   2 or 112    |    2/m     |   112      |
    +-------------+--------------+---------------+------------+------------+
    | C2x         | Monoclinic   |     211       |    2/m     |   211      |
    +-------------+--------------+---------------+------------+------------+
    | C2y         | Monoclinic   |     121       |    2/m     |   121      |
    +-------------+--------------+---------------+------------+------------+
    | Cs          | Monoclinic   |   11m or m    |    2/m     |    1       |
    +-------------+--------------+---------------+------------+------------+
    | Csx         | Monoclinic   |     m11       |    2/m     |    1       |
    +-------------+--------------+---------------+------------+------------+
    | Csy         | Monoclinic   |     1m1       |    2/m     |    1       |
    +-------------+--------------+---------------+------------+------------+
    | C2h         | Monoclinic   |     2/m       |    2/m     |   112      |
    +-------------+--------------+---------------+------------+------------+
    | D2          | Orthorhombic |     222       |    mmm     |   222      |
    +-------------+--------------+---------------+------------+------------+
    | C2v         | Orthorhombic |     mm2       |    mmm     |   211      |
    +-------------+--------------+---------------+------------+------------+
    | D2h         | Orthorhombic |     mmm       |    mmm     |   222      |
    +-------------+--------------+---------------+------------+------------+
    | C4          | Tetragonal   |      4        |    4/m     |    4       |
    +-------------+--------------+---------------+------------+------------+
    | S4          | Tetragonal   |     -4        |    4/m     |   112      |
    +-------------+--------------+---------------+------------+------------+
    | C4h         | Tetragonal   |     4/m       |    4/m     |    4       |
    +-------------+--------------+---------------+------------+------------+
    | D4          | Tetragonal   |     422       |   4/mmm    |   422      |
    +-------------+--------------+---------------+------------+------------+
    | C4v         | Tetragonal   |     4mm       |   4/mmm    |    4       |
    +-------------+--------------+---------------+------------+------------+
    | D2d         | Tetragonal   |    -42m       |   4/mmm    |   222      |
    +-------------+--------------+---------------+------------+------------+
    | D4h         | Tetragonal   |    4/mmm      |   4/mmm    |   422      |
    +-------------+--------------+---------------+------------+------------+
    | C3          | Trigonal     |      3        |    -3      |    3       |
    +-------------+--------------+---------------+------------+------------+
    | C3i         | Trigonal     |     -3        |    -3      |    3       |
    +-------------+--------------+---------------+------------+------------+
    | D3          | Trigonal     |  32 or 321    |    -3m     |   32       |
    +-------------+--------------+---------------+------------+------------+
    | D3y         | Trigonal     |     312       |    -3m     |   312      |
    +-------------+--------------+---------------+------------+------------+
    | C3v         | Trigonal     |     3m        |    -3m     |    3       |
    +-------------+--------------+---------------+------------+------------+
    | D3d         | Trigonal     |    -3m        |    -3m     |   32       |
    +-------------+--------------+---------------+------------+------------+
    | C6          | Hexagonal    |      6        |    6/m     |    6       |
    +-------------+--------------+---------------+------------+------------+
    | C3h         | Hexagonal    |     -6        |    6/m     |    6       |
    +-------------+--------------+---------------+------------+------------+
    | C6h         | Hexagonal    |     6/m       |    6/m     |   622      |
    +-------------+--------------+---------------+------------+------------+
    | D6          | Hexagonal    |     622       |   6/mmm    |   622      |
    +-------------+--------------+---------------+------------+------------+
    | C6v         | Hexagonal    |     6mm       |   6/mmm    |    6       |
    +-------------+--------------+---------------+------------+------------+
    | D3h         | Hexagonal    |    -6m2       |   6/mmm    |   312      |
    +-------------+--------------+---------------+------------+------------+
    | D6h         | Hexagonal    |    6/mmm      |   6/mmm    |   622      |
    +-------------+--------------+---------------+------------+------------+
    | T           | Cubic        |     23        |    m-3     |   23       |
    +-------------+--------------+---------------+------------+------------+
    | Th          | Cubic        |     m-3       |    m-3     |   23       |
    +-------------+--------------+---------------+------------+------------+
    | O           | Cubic        |     432       |   m-3m     |   432      |
    +-------------+--------------+---------------+------------+------------+
    | Td          | Cubic        |    -43m       |   m-3m     |   23       |
    +-------------+--------------+---------------+------------+------------+
    | Oh          | Cubic        |     m-3m      |   m-3m     |   432      |
    +-------------+--------------+---------------+------------+------------+

    """
    subset = str(subset).lower()
    if subset == "all_repeated":
        return [
            # Triclinic
            C1,
            Ci,
            S2,  # redundant
            # Monoclinic
            C2,
            D1,  # redundant
            C2x,
            C2y,
            C2z,  # redundant
            Cs,
            Csx,
            Csy,
            Csz,  # redundant
            C2h,
            # Orthorhombic
            D2,
            C2v,
            D2h,
            # Tetragonal
            C4,
            S4,
            C4i,  # redundant
            C4h,
            D4,
            C4v,
            D2d,
            D4h,
            # Trigonal
            C3,
            C3i,
            S6,  # redundant
            D3,
            D3x,
            D3y,
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
            # cubic
            T,
            Th,
            O,
            Td,
            Oh,
        ]
    elif subset == "all":
        return [
            # Triclinic
            C1,
            Ci,
            # Monoclinic
            C2,
            C2x,
            C2y,
            Cs,
            Csx,
            Csy,
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
            C3i,
            D3,
            D3y,
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
            # cubic
            T,
            Th,
            O,
            Td,
            Oh,
        ]
    elif subset == "unique":
        return [
            # Triclinic
            C1,
            Ci,
            # Monoclinic
            C2,
            Cs,
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
            C3i,
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
            # cubic
            T,
            Th,
            O,
            Td,
            Oh,
        ]
    elif subset == "proper":
        return [
            # Triclinic
            C1,
            # Monoclinic
            C2,
            # Orthorhombic
            D2,
            D4,
            # Tetragonal
            C4,
            # Trigonal
            C3,
            D3,
            # Hexagonal
            C6,
            D6,
            # cubic
            T,
            O,
        ]
    elif subset == "laue":
        return [
            # Triclinic
            Ci,
            # Monoclinic
            C2h,
            # Orthorhombic
            D2h,
            D4h,
            # Tetragonal
            C4h,
            # Trigonal
            C3i,
            D3d,
            # Hexagonal
            C6h,
            D6h,
            # cubic
            Th,
            Oh,
        ]
    elif subset == "proper_all":
        return [
            # Triclinic
            C1,
            # Monoclinic
            C2,
            C2x,
            C2y,
            # Orthorhombic
            D2,
            # Tetragonal
            C4,
            # Trigonal
            C3,
            D3,
            D3x,
            D3y,
            # Hexagonal
            C6,
            D6,
            # cubic
            T,
            O,
        ]
    elif subset == "procedural":
        return [
            # Cyclic
            C1,
            C2,
            C3,
            C4,
            C6,
            # Dihedral
            D2,
            D3,
            D4,
            D6,
            # Cyclic plus inversion (\ba{n})
            Ci,
            Cs,
            C3i,
            S4,
            C3h,
            # Cyclic plus perpendicular mirrors (n/m)
            C2h,
            C4h,
            C6h,
            # Cyclic plus vertical mirrors (nm)
            C2v,
            C3v,
            C4v,
            C6v,
            # Dihedral plus diagonal mirrors (\bar{n} m)
            D3d,
            D2d,
            D3h,
            # Dihedral with vertical and perpendicular mirros (n/m m)
            D2h,
            D4h,
            D6h,
            # Combining cyclic (n1 n2)
            T,
            O,
            # combining cyclic and mirrors
            Th,
            Td,
            Oh,
        ]
    else:
        raise ValueError("{} is not a valid subset option".format(subset))


def get_distinguished_points(s1: Symmetry, s2: Symmetry = C1) -> Rotation:
    """Return points symmetrically equivalent to identity with respect
    to ``s1`` and ``s2``.

    Parameters
    ----------
    s1
        First symmetry.
    s2
        Second symmetry.

    Returns
    -------
    distinguished_points
        Distinguished points.
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


@deprecated(
    since="0.14",
    removal="0.15",
    alternative="symmetry.get_point_group_from_space_group",
)
def get_point_group(space_group_number: int, proper: bool = False) -> Symmetry:
    r"""
    This function has been renamed to
    `orix.quaternion.symmetry.get_point_group_from_space_group`.
    for clairity"""
    return get_point_group_from_space_group(space_group_number, proper)


def get_point_group_from_space_group(
    space_group_number: Union(int, str), proper: bool = False
) -> Symmetry:
    """
    Maps a space group number or name to a crystallographic point group.

    Parameters
    ----------
    space_group_number: int between 1-231,  or str
        If is an int(n) or str(int(n)) where n is between 1 and 231, it will
        return the point group of the nth space group, as defined by the
        International Tables of Crystallogrphy. Otherwise, it will be passed
        to diffpy's dictionary of space group names for interpretation.

        Thus, 222 and '222' will both return symmetry.Oh (ie "432", the point
        group of SG#222==Pn-3n), but 'P222' will return symmetry.D2
        (ie "222", the proper point group of SG#16=='P222').

    proper: bool
        Whether to return the point group with proper rotations only
        (``True``), or the full point group (``False``). Default is
        ``False``.

    Returns
    -------
    point_group
        One of the 11 proper or 32 point groups.

    Notes:
    ----------
    This function uses diffpy.structure.spacegroups to convert names to
    space group IDs, and has some allowances for spelling and spacing
    differences. Thus, variations like "Pm-3m", 221, "PM3m", and "Pn3n" all
    map to symmetry.Oh == 'm-3m'. To see a full list of all name options
    avaiable, use the following snippet:

    >>> import diffpy.structure.spacegroups as sg
    >>> sg._buildSGLookupTable()
    >>> sg._sg_lookup_table.keys()

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
_EDAX_POINT_GROUP_ALIASES = {
    "121": ["20"],
    "2/m": ["2"],
    "222": ["22"],
    "422": ["42"],
    "321": ["32"],
    "622": ["62"],
    "432": ["43"],
    "m-3m": ["m3m"],
}


def _get_laue_group_name(name: str) -> str | None:
    # search through all the point groups defined in orix for one with a
    # matching name.
    valid_name = False
    for g in get_point_groups("all_repeated"):
        if g.name == name:
            valid_name = True
            break
    if valid_name == False:
        raise ValueError("{} is not a valid point group name")
    # add an inversion to get the laue group.
    g_laue = _get_unique_symmetry_elements(g, Ci)
    # find a laue group with matching operators
    for laue in get_point_groups("laue"):
        # first check for length
        if g_laue.shape != laue.shape:
            continue
        # then check for identical operators (regardless of order)
        if np.min(g_laue.outer(laue).angle ** 2, 1).max() < 1e-4:
            if np.min(g_laue.outer(laue).angle ** 2, 0).max() < 1e-4:
                return laue.name


def _get_unique_symmetry_elements(
    sym1: Symmetry, sym2: Symmetry, check_subgroups: bool = False
) -> Symmetry:
    """Compute the unique symmetry elements between two symmetries,
    defined as ``sym1.outer(sym2).unique()``.

    To improve computation speed some checks are performed prior to
    explicit computation of the unique elements. If ``sym1 == sym2``
    then the unique elements are just the symmetries themselves, and so
    are returned. If ``sym2`` is a :attr:`Symmetry.subgroup`` of
    ``sym1`` then the unique symmetry elements will be identical to
    ``sym1``, in this case ``sym1`` is returned. This check is made if
    ``check_subgroups=True``. As the symmetry order matters, this may
    not be the case if ``sym1`` is a subgroup of ``sym2``, so this is
    not checked here.

    If no relationship is determined between the symmetries then the
    unique symmetry elements are explicitly computed, as described
    above.

    Parameters
    ----------
    sym1
    sym2
    check_subgroups
        Whether to check if ``sym2`` is a subgroup of ``sym1``. Default
        is ``False``.

    Returns
    -------
    unique
        The unique symmetry elements.
    """
    if sym1 == sym2:
        return sym1
    if check_subgroups:
        # test whether sym2 is a subgroup of sym1
        sym2_is_sg_sym1 = True if sym2 in sym1.subgroups else False
        if sym2_is_sg_sym1:
            return sym1
    # default to explicit computation of the unique symmetry elements
    return sym1.outer(sym2).unique()
