#
# Copyright 2019-2025 the orix developers
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

from copy import copy
from typing import TYPE_CHECKING, Literal, Union

from diffpy.structure.spacegroups import GetSpaceGroup
import matplotlib.pyplot as plt
import numpy as np

from orix._util import deprecated
from orix.quaternion.rotation import Rotation
from orix.vector.vector3d import Vector3d

if TYPE_CHECKING:  # pragma: no cover
    from orix.quaternion.orientation import Orientation
    from orix.vector.fundamental_sector import FundamentalSector


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
    _s_name = ""

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
        groups = PointGroups._pg_sets["permutations"]
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

    def _get_symmetry_elements(self) -> list:
        """Returns all the crystallographically unique axes and their associated
        symmetry elements (mirrors, rotations, rotoinversions, etc).

        Returns
        -------
        axes Vector3d
            Vector(s) that are each either parallel to an axis of rotation or
            perpendicular to a mirror plane, or both.
        is_mirror bool
            Indicates whether each primary_axis is or is not perpendicular to a
            mirror plane
        s_type str ("inversion", "rotoinversion", "rotation", or"none")
            Indicates whether the rotational symmetry associated with the axis
            contains an inversion, a rotoinversion, purely rotational, or just an
            1-fold axis perpendicular to a mirror.
        folds
            Indicats the order of the rotational symmetry (1,2,3,4, or 6)

        Notes
        -------
        This function does not return ALL the axes and angles,
        (that function would be `Symmetry.to_axes_angles`), nor does it return
        the minimum generating elements. Instead, it returns all the primary axes
        plus information about the rotations, inversions, and/or mirrors associated
        with each.
        """
        # create masks to sort out which elements are what.
        # proper elements
        p_mask = ~self.improper
        # mirror planes
        m_mask = (np.abs(np.abs(self.angle) - np.pi) < 1e-4) * self.improper
        # rotations (both proper and improper)
        r_mask = np.abs(self.angle) > 1e-4
        # rotoinversions
        roto_mask = r_mask * ~p_mask * ~m_mask
        # the inversion symmetry
        i_mask = (~r_mask) * self.improper
        if np.sum(i_mask) > 0:
            has_inversion = True
        else:
            has_inversion = False

        elements = []
        # Find the unique axis familes.
        axis_families = (self.axis.in_fundamental_sector(self)).unique()

        # iterate through each primary axis and find what elements to add
        for axis in axis_families:
            # mask out just axes elements in the fundamental sector to avoid repeats
            axis_mask = (
                np.sum(np.abs((self.axis.in_fundamental_sector(self) - axis).data), 1)
                < 1e-4
            )
            m_flag = False
            folds = 0
            s_type = "empty"
            # check for mirrors
            if np.any(m_mask * axis_mask):
                m_flag = True
            # check to see if there are only proper rotations
            if np.all(p_mask[axis_mask]):
                # This might just be the identity.
                if not np.any(r_mask * axis_mask):
                    elements.append(
                        (copy(axis), m_flag, 1, "none"),
                    )
                    continue
                min_ang = np.abs(self[r_mask * axis_mask].angle).min()
                folds = np.around(2 * np.pi / min_ang).astype(int)
                elements.append(
                    (copy(axis), m_flag, folds, "rotation"),
                )
                continue
            # Check if there is a rotation with an inversion
            elif has_inversion:
                # this might just be the 1-fold inversion center
                if not np.any(r_mask * axis_mask):
                    elements.append(
                        (copy(axis), m_flag, 1, "inversion"),
                    )
                    continue
                min_ang = np.abs(self[r_mask * axis_mask].angle).min()
                folds = np.around(2 * np.pi / min_ang).astype(int)
                elements.append(
                    (copy(axis), m_flag, folds, "inversion"),
                )
                continue
            # the only other important option is a rotoinversion
            elif np.any(roto_mask[axis_mask]):
                min_ang = np.abs(self[roto_mask * axis_mask].angle).min()
                folds = np.around(2 * np.pi / min_ang).astype(int)
                elements.append(
                    (copy(axis), m_flag, folds, "rotoinversion"),
                )
                continue
            # if it it not a rotational symmetry of any type, it's a mirror
            else:
                elements.append(
                    (copy(axis), m_flag, 1, "none"),
                )
        # Finally, 3-fold rotations around the 111 create <110> mirrors
        # not on the primary axes. These we can add by hand.
        if np.any(np.abs(Vector3d([1, 1, 1]).angle_with(self.axis)) < 1e-4):
            v = Vector3d([0, 1, 1]).in_fundamental_sector(self)
            elements.append((v, True, 1, "none"))
        # split the list of lists into 4 variables.
        axes = [x[0] for x in elements]
        is_mirror = [x[1] for x in elements]
        s_type = [x[3] for x in elements]
        folds = [x[2] for x in elements]
        return axes, is_mirror, s_type, folds

    def get_axis_orders(self) -> dict[Vector3d, int]:
        """Return a dictionary of every rotation axis and it's order (ie, folds)"""
        s = self[self.angle > 0]
        if s.size == 0:
            return {}
        return {
            Vector3d(a): b + 1
            for a, b in zip(*np.unique(s.axis.data, axis=0, return_counts=True))
        }

    def get_highest_order_axis(self) -> tuple[Vector3d, np.ndarray]:
        """Return the highest order rotational axis and it's order (ie, folds)"""
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
        asymetric_vector: Vector3d | None = None,
        show_name: bool = True,
        plt_axis: plt.Axes | None = None,
        return_figure: bool = True,
        marker_dict: dict = {},
        marker_size: float = 150.0,
        mirror_width: float = 2.0,
        asymetric_vector_dict: dict = {},
        asymmetric_vector_size: float = 10.0,
        itoc_style=True,
    ) -> plt.Figure | None:
        """Creates a stereographic projection of symmetry operations in the group.
        Can also plot symmetrically equivalent variations of orientations or vectors
        to demonstrate the effect of symmetry operations.

        Parameters
        ----------
        asymetric_vector
            A marker will be added at the stereographic projection of this vector,
            along with all it's symmetrically equivalent rotations. By default, no
            vector will be plotted, and only rotation and mirror markers will be added
            to the plot.
        show_name
            If True, add both the Schoenflies and Hermann-Mauguin names of the point
            group to the title.
        plt_axis
            The matplotlib.Axis object into which to add the stereographic plot.
            If None is passed, a new figure and axis will be generated.
        return_figure
            If True, return the figure containing the plotting axis.
        marker_dict
            A dictionary of arguments to modify how the symmetry markers are
            generated. The following options are the overwritable defaults:
                  1: 'black'   <-- 1-fold marker color
                  2: 'green'   <-- 2-fold marker color
                  3: 'red'     <-- 3-fold marker color
                  4: 'purple'  <-- 4-fold marker color
                  6: 'magenta' <-- 6-fold marker color
                  'm': 'blue'  <-- 1-fold marker color
        marker_size
            The size of the rotational makers to be added to the plot. This is
            equivalent to the argument "s" in matplotlib.scatter
        mirror_width
            The width of the line used to draw the mirror planes. This is
            equivalent to the argument "linewidth" in matplotlib.plot
        asymetric_vector_dict
            A dictionary of arguments to modify the asymetric_vector markers.
            The following options are the overwritable defaults:
                'upper_color': 'black' < -- Upper hemisphere marker color
                'lower_color': 'grey' < -- Lower hemisphere marker color
                'upper_marker': '+' < -- Upper hemisphere marker shape
                'lower_marker': 'o' < -- Lower hemisphere marker shape
        asymmetric_vector_size
            size of the markers used to plot the asymetric vector markers.
        itoc_style
            If True, the plot will follow the ITOC convention of not placing redundant
            inversion symbols on rotation markers perpendicular to the viewing
            direction.

        Returns
        -------
        fig
            The created figure, returned if ``return_figure=True`` is
            passed as a keyword argument.

        Notes
        -----

        If users wish to have more control over their plots, this function can be used
        to modify an existing plot, like so:

        >>> import matplotlib.pyplot as plt
        >>> pg_Oh = PointGroups.get('m-3m')
        >>> v = Vector3d.random(10)
        >>> v_symm = pg_Oh.outer(v).flatten()
        >>> fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "stereographic"})
        >>> pg_Oh.plot(plt_axis=ax, show_name = False)
        >>> ax.set_title("my cool custom title")
        >>> ax.scatter(v_symm)

        In this way, keword arguments related to the plot, the title, the scattered
        vector markers, and/or the symmetry markers can be individually altered as
        desired.

        """
        # import orix.plot so matplotlib knows what the stereographic projection is.
        import orix.plot

        # dictionary of default colors for the symmetry markers.
        colors = {
            1: "black",
            2: "green",
            3: "red",
            4: "purple",
            6: "magenta",
            "m": "blue",
        }
        # after resetting defaults, update color choices passed in via color_dict
        colors.update(marker_dict)
        # if the user did not pass in an axis, generate one
        if plt_axis is None:
            fig, plt_axis = plt.subplots(subplot_kw={"projection": "stereographic"})
        else:
            fig = plt_axis.get_figure()

        # add a default title if requested
        if show_name:
            plt_axis.set_title(self._s_name + "   ( " + self.name + " )")

        # determine the  symnmetry elements and plot them.
        elements = self._get_symmetry_elements()
        for v, m, t, f in zip(*elements):
            # plot each symmetrically equivalent mirror plane only once
            if m:
                for mv in (self * v).unique():
                    m_circ = mv.get_circle()
                    plt_axis.plot(m_circ, color=colors["m"], linewidth=mirror_width)
            # plot each symmetrically equivalent rotation element only once
            c = colors[f]
            if f > 1:
                for sv in (self * v).unique():
                    # ITOC doesn't plot inversion or rotoinversion markers for
                    # symmetry elements with axes perpendicular to the out-of plane
                    # direction, as the information is redundant.
                    z_ang = np.abs(sv.angle_with(Vector3d.zvector()))
                    is_perp = np.abs(z_ang - (np.pi / 2)) < 1e-4
                    if itoc_style and is_perp:
                        plt_axis.symmetry_marker(
                            sv, folds=f, s=marker_size, color=c, modifier=None
                        )
                    else:
                        plt_axis.symmetry_marker(
                            sv, folds=f, s=marker_size, color=c, modifier=t
                        )
            # if this is the primary axis and there is no rotation but an inversion
            # (ie, this is symmetry.Ci, the `-1` PG), add the appropriate marker.
            elif f == 1 and np.abs(v.angle_with(Vector3d.zvector())) < 1e-4:
                if t != "inversion":
                    continue
                for sv in (self * v).unique():
                    plt_axis.symmetry_marker(
                        sv, folds=f, s=marker_size, color=c, modifier=t
                    )

        # plot asymmetric markers if requested.
        if asymetric_vector is not None:
            v_symm = self.outer(asymetric_vector).flatten()
            vdict = {
                "upper_color": "black",
                "lower_color": "grey",
                "upper_marker": "+",
                "lower_marker": "o",
            }
            vdict.update(asymetric_vector_dict)
            mask = v_symm.z >= 0
            plt_axis.scatter(
                -1 * v_symm[~mask],
                marker=vdict["lower_marker"],
                c=vdict["lower_color"],
            )
            plt_axis.scatter(
                v_symm[mask],
                marker=vdict["upper_marker"],
                c=vdict["upper_color"],
            )

        # return the figure if requested
        if return_figure:
            return fig


# ---------------- Proceedural definitions of Point Groups ---------------- #
# NOTE: ORIX uses Schoenflies symbols to define point groups. This is partly
# because the notation is short and always starts with a letter (ie, they
# make convenient python variables), and partly because it helps limit
# accidental misinterpretation of Hermann-Mauguin symbols as space group
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
C1._s_name = "C1"
Ci = Symmetry([(1, 0, 0, 0), (-1, 0, 0, 0)])
Ci.improper = [0, 1]
Ci.name = "-1"
Ci._s_name = "Ci"
# include redundant point group S2 == Ci
S2 = Symmetry([(1, 0, 0, 0), (-1, 0, 0, 0)])
S2.improper = [0, 1]
S2.name = "-1"
S2._s_name = "S2"

# Special generators
_mirror_xy = Symmetry([(1, 0, 0, 0), (0, 0.75**0.5, -(0.75**0.5), 0)])
_mirror_xy.improper = [0, 1]
_cubic = Symmetry([(1, 0, 0, 0), (0.5, 0.5, 0.5, 0.5)])

# 2-fold rotations
C2x = Symmetry([(1, 0, 0, 0), (0, 1, 0, 0)])
C2x.name = "211"
C2x._s_name = "C2x"
C2y = Symmetry([(1, 0, 0, 0), (0, 0, 1, 0)])
C2y.name = "121"
C2y._s_name = "C2y"
C2z = Symmetry([(1, 0, 0, 0), (0, 0, 0, 1)])
C2z.name = "112"
C2z._s_name = "C2z"
C2 = Symmetry(C2z)
C2.name = "2"
C2._s_name = "C2"
# included redundant point group D1 == C2
D1 = Symmetry(C2z)
D1.name = "2"
D1._s_name = "D1"

# Mirrors
Csx = Symmetry([(1, 0, 0, 0), (0, 1, 0, 0)])
Csx.improper = [0, 1]
Csx.name = "m11"
Csx._s_name = "Csx"
Csy = Symmetry([(1, 0, 0, 0), (0, 0, 1, 0)])
Csy.improper = [0, 1]
Csy.name = "1m1"
Csy._s_name = "Csy"
Csz = Symmetry([(1, 0, 0, 0), (0, 0, 0, 1)])
Csz.improper = [0, 1]
Csz.name = "11m"
Csz._s_name = "Csz"
Cs = Symmetry(Csz)
Cs.name = "m"
Cs._s_name = "Cs"

# Monoclinic
C2h = Symmetry.from_generators(C2, Cs)
C2h.name = "2/m"
C2h._s_name = "C2h"

# Orthorhombic
D2 = Symmetry.from_generators(C2z, C2x, C2y)
D2.name = "222"
D2._s_name = "D2"
C2v = Symmetry.from_generators(C2z, Csx)
C2v.name = "mm2"
C2v._s_name = "C2v"
D2h = Symmetry.from_generators(Csz, Csx, Csy)
D2h.name = "mmm"
D2h._s_name = "D2h"

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
C4._s_name = "C4"

# Tetragonal
S4 = Symmetry(C4)
S4.improper = [0, 1, 0, 1]
S4.name = "-4"
S4._s_name = "S4"
# include redundant point group C4i == S4
C4i = Symmetry(C4)
C4i.improper = [0, 1, 0, 1]
C4i.name = "-4"
C4i._s_name = "C4i"
C4h = Symmetry.from_generators(C4, Cs)
C4h.name = "4/m"
C4h._s_name = "C4h"
D4 = Symmetry.from_generators(C4, C2x, C2y)
D4.name = "422"
D4._s_name = "D4"
C4v = Symmetry.from_generators(C4, Csx)
C4v.name = "4mm"
C4v._s_name = "C4v"
D2d = Symmetry.from_generators(D2, _mirror_xy)
D2d.name = "-42m"
D2d._s_name = "D2d"
D4h = Symmetry.from_generators(C4h, Csx, Csy)
D4h.name = "4/mmm"
D4h._s_name = "D4h"

# 3-fold rotations
C3x = Symmetry([(1, 0, 0, 0), (0.5, 0.75**0.5, 0, 0), (0.5, -(0.75**0.5), 0, 0)])
C3y = Symmetry([(1, 0, 0, 0), (0.5, 0, 0.75**0.5, 0), (0.5, 0, -(0.75**0.5), 0)])
C3z = Symmetry([(1, 0, 0, 0), (0.5, 0, 0, 0.75**0.5), (0.5, 0, 0, -(0.75**0.5))])
C3 = Symmetry(C3z)
C3.name = "3"
C3._s_name = "C3"

# Trigonal
C3i = Symmetry.from_generators(C3, Ci)
C3i.name = "-3"
C3i._s_name = "C3i"
# include redundant point group S6==C3i
S6 = Symmetry.from_generators(C3, Ci)
S6.name = "-3"
S6._s_name = "S6"
D3x = Symmetry.from_generators(C3, C2x)
D3x.name = "321"
D3x._s_name = "D3x"
D3y = Symmetry.from_generators(C3, C2y)
D3y.name = "312"
D3y._s_name = "D3y"
D3 = Symmetry(D3x)
D3.name = "32"
D3._s_name = "D3"
C3v = Symmetry.from_generators(C3, Csx)
C3v.name = "3m"
C3v._s_name = "C3v"
D3d = Symmetry.from_generators(S6, Csx)
D3d.name = "-3m"
D3d._s_name = "D3d"

# Hexagonal
C6 = Symmetry.from_generators(C3, C2)
C6.name = "6"
C6._s_name = "C6"
C3h = Symmetry.from_generators(C3, Cs)
C3h.name = "-6"
C3h._s_name = "C3h"
C6h = Symmetry.from_generators(C6, Cs)
C6h.name = "6/m"
C6h._s_name = "C6h"
D6 = Symmetry.from_generators(C6, C2x, C2y)
D6.name = "622"
D6._s_name = "D6"
C6v = Symmetry.from_generators(C6, Csx)
C6v.name = "6mm"
C6v._s_name = "C6v"
D3h = Symmetry.from_generators(C3, C2y, Csz)
D3h.name = "-6m2"
D3h._s_name = "-D3h"
D6h = Symmetry.from_generators(D6, Csz)
D6h.name = "6/mmm"
D6h._s_name = "D6h"

# Cubic
T = Symmetry.from_generators(C2, _cubic)
T.name = "23"
T._s_name = "T"
Th = Symmetry.from_generators(T, Ci)
Th.name = "m-3"
Th._s_name = "Th"
O = Symmetry.from_generators(C4, _cubic, C2x)
O.name = "432"
O._s_name = "O"
Td = Symmetry.from_generators(T, _mirror_xy)
Td.name = "-43m"
Td._s_name = "Td"
Oh = Symmetry.from_generators(O, Ci)
Oh.name = "m-3m"
Oh._s_name = "Oh"


class PointGroups(list):
    # make a lookup table of common subsets of Point Groups
    _pg_sets = {
        "permutations_repeated": [
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
        ],
        "permutations": [
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
        ],
        "groups": [
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
        ],
        "proper_groups": [
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
        ],
        "proper_permutations": [
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
        ],
        "laue": [
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
        ],
        "procedural": [
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
        ],
    }
    _subset_names = _pg_sets.keys()
    _point_group_names = dict(
        [(x.name, x) for x in _pg_sets["permutations_repeated"]]
    ).keys()

    _spacegroup2pointgroup_dict = {
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

    def __init__(self, symmetry_list: list = [C1]):
        """
        A list of symmetry operators with convenence functions for parsing entries
        and displaying information.

        This class is primarily intended to be called using PointGroups.subset(),
        or to return a single Symmetry object using PointGroups.get().

        Parameters
        ----------
        symmetry_list
            A list of orix.quaternion.symmetry.Symmetry objects, each representing
            a crystallographic class.
        """
        if not isinstance(symmetry_list, list):
            raise ValueError("'symmetry_list' must be a list of Symmetry objects")
        elif not np.all([isinstance(x, Symmetry) for x in symmetry_list]):
            raise ValueError("'symmetry_list' must be a list of Symmetry objects")
        else:
            self.extend(symmetry_list)

    def __repr__(self):
        str_data = (
            "| Name  | System      | HM     | Laue  | Proper |\n"
            + "=" * 48
            + "\n"
            + "\n".join(
                [
                    "| "
                    + "| ".join(
                        [
                            x._s_name.ljust(6),
                            x.system.ljust(13),
                            x.name.ljust(6),
                            x.laue.name.ljust(6),
                            x.proper_subgroup.name.ljust(7),
                        ]
                    )
                    + "|"
                    for x in self
                ]
            )
        )

        return str_data

    def get(name: Literal[PointGroups._point_group_names]):
        """
        Given a string or integer representation, this function will attempt to
        return an associated Symmetry object.

        This is done by first checking the labels defined in orix, which includes
        Hermann-Mauguin ('m3m' or '2', etc.) and Shoenflies ('C6' or D3h', etc.).

        If it cannot find a match in either list, it will attempt to look up the
        space group name using diffpy's GetSpaceGroup, and relate that back
        to a point group. this is equivalent to PointGroups.from_space_group(name)

         Parameters
        ----------
        name : string in PointGroups._point_group_names
            either the Hermann-Maugin or Shoenflies name for a crystallographic
            point gorup.

        Returns
        -------
        point_group
            an object of Class `Symmetry` representing the requested
            crystallographic point group.
        """
        # check the 'groups' list first, then 'permutations',
        # then 'permutations_repeated'.
        print(vars().keys())
        for subset in ["groups", "permutations", "permutations_repeated"]:
            pgs = PointGroups._pg_sets[subset]
            pg_dict = dict([(x.name, x) for x in pgs])
            if str(name).lower() in pg_dict.keys():
                return pg_dict[name.lower()]
            # repeat check with Shoenflies notation
            pg_dict_s = dict([(x._s_name.lower(), x) for x in pgs])
            if str(name).lower() in pg_dict_s.keys():
                return pg_dict_s[name.lower()]
        # If the name doesn't exist in orix, try diffpy
        try:
            return PointGroups.from_space_group(name)
        # If the name still cannot be found, return a ValueError
        except ValueError:
            raise ValueError(
                f"'name' must be one of {', '.join(map(str, pg_dict.keys()))},"
                + f" {', '.join(map(str, pg_dict_s.keys()))}, or must be a string or "
                + "integer recognized by diffpy.structure.spacegroups.GetSpaceGroup"
                + f". name = '{name}' is not a valid value."
            )

    def from_space_group(
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
            return PointGroups._spacegroup2pointgroup_dict[pgn]["proper"]
        else:
            return PointGroups._spacegroup2pointgroup_dict[pgn]["improper"]

    @classmethod
    def get_set(self, name: Literal[PointGroups._subset_names] = "groups"):
        """
        returns different subsets of the 32 crystallographic point groups. By
        default, this returns all 32 in the order they appear in the
        International Tables of Crystallography (ITOC).

        Parameters
        ----------
        subset : str, optional
            the point group list to return. The options are as follows:
                "groups" (default):
                    All 32 point groups in the order they appear in ITOC's space groups.
                    Thus, they are grouped by crystal system and Laue class
                "permutations":
                    All 32 points groups, plus common axis-specific permutations
                    for non-centrosymmetric groups (ie, C2 plus C2x and C2y) for
                    a total of 37 point group projections. These
                    are given in the same order as ITOC Table 10.2.2
                "permutations_repeated":
                    The 37 point group projections, plus the redundant Schonflies
                    and Hermann-Mauguin group names. For example, both Ci and S2
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
        point groups: PointGroups
            A PointGroup class containing the requested symmetries
        """
        pg_opts = self._pg_sets.keys()
        if name in pg_opts:
            return PointGroups(self._pg_sets[name])
        elif name.lower() in pg_opts:
            return PointGroups(self._pg_sets[name.lower()])
        # if the name doesn't exist, return a ValueError
        raise ValueError(
            f"'name' must be one of {', '.join(map(str, pg_opts))}, not '{name}'"
        )


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


@deprecated(
    since="0.14",
    removal="0.15",
    alternative="PointGroups.from_space_group",
)
def get_point_group(space_group_number: int, proper: bool = False) -> Symmetry:
    """
    This function has been moved to the PointGroups class
    """
    return PointGroups.from_space_group(space_group_number, proper)


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
    for g in PointGroups._pg_sets["permutations_repeated"]:
        if g.name == name:
            valid_name = True
            break
    if valid_name is False:
        raise ValueError(f"{name} is not a valid point group name")
    # if the matching point group as a Schoenflies name that ends in an x,y, or z,
    # it's a permutation of a point group. trade it for an unpermutated one.
    if np.isin(g._s_name[-1], ["x", "y", "z"]):
        s_name = g._s_name[:-1]
        for g in PointGroups._pg_sets["permutations_repeated"]:
            if g._s_name == s_name:
                break
    # add an inversion to get the laue group.
    g_laue = _get_unique_symmetry_elements(g, Ci)
    # find a laue group with matching operators
    for laue in PointGroups._pg_sets["laue"]:
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
