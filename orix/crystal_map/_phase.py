#
# Copyright 2018-2025 the orix developers
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

"""Phase class and private utilities used by it."""

from __future__ import annotations

import copy
from pathlib import Path
import warnings

import diffpy.structure as dst
from diffpy.structure.parsers import p_cif
from diffpy.structure.spacegroups import GetSpaceGroup, SpaceGroup
from diffpy.structure.symmetryutilities import ExpandAsymmetricUnit
import matplotlib.colors as mcolors
import numpy as np

from orix.plot._util.color import get_matplotlib_color
from orix.quaternion.symmetry import (
    _EDAX_POINT_GROUP_ALIASES,
    VALID_SYSTEMS,
    Symmetry,
    _groups,
    get_point_group,
)
from orix.vector.miller import Miller
from orix.vector.vector3d import Vector3d

# ======================== #
# Repeatedly used docstrings
# ======================== #
# DEVELOPER NOTE: as of python 3.14, due to an intentional choice by the python
# developers, classmethod docstrings cannot be f-strings, and cannot be
# edited within the method. Thus, only docstrings used repeatedly by properties
# or functions should be added here.
_xtal2cart_docstring = r"""There are multiple valid algorithms for converting between
        crystallographic and cartesian reference frames. ORIX uses
        the form defined in the International Tables for
        Crystallography, Volume A, section 1.5, which matches equation
        7.33 from Structure of Materials, second edition:

        .. math::
             \bf{e_1} = \frac{\bf{a_1}}{ |\bf{a_1}|} \quad\quad
             \bf{e_2} = \bf{e_3} \times \bf{e_1} \quad\quad             
             \bf{e_3} = \frac{\bf{a_3^{*}}}{ |\bf{a_3^{*}}|}

        Note the equations in both sources are using a different notation
        than what is used in orix to define coordinate axes. 
        :math:`\bf{a_{i}}` represents the :math:`(a, b, c)` direct lattice basis vectors,
        :math:`\bf{a_{i}^{*}}` represents the :math:`(a^{*}, b^{*}, c^{*})` reciprocal lattice basis vectors,
        and :math:`\bf{e_{i}}` represents the :math:`(x, y, z)` cartesion basis vectors.
        Additionally, :math:`| |` represents the magnitude (ie, length) of a vector.
        Notably, this sets:

            1) :math:`\bf{a_1}` parallel with :math:`\bf{e_1}`
            2) :math:`\bf{e_2}` perpendicular to :math:`\bf{a_3}` and :math:`\bf{a_1}`
            3) :math:`\bf{e_3}` perpendicular to :math:`\bf{a_1}` and :math:`\bf{a_2}`

        This is different from some conventions popular in x-ray
        diffraction references that choose to instead align :math:`\bf{a_3}`
        parallel to :math:`\bf{e_3}`. This makes no difference for
        cubic, tetragonal, and orthorhombic systems, but can lead to
        unintented results in all other systems. Users should
        take care to determine which convention was used when importing
        data from other sources.
        
        Regardless of choice, the reciprocal basis vectors :math:`{\bf{a^{*}}}`
        are then defined as 
        
        .. math::
            \bf{a_i} * \bf{a_j^{*}}=\delta_{ij}
        
        where :math:`\delta_{ij}` is the Kronecker delta. Details on how
        to solves these equations for every crystal system can be found
        in Structure of Materials, second edition, chapters 6 and 7."""


class Phase:
    """Symmetry and unit cell of a phase in a crystallographic map.

    The phase can be crystallographic or non-crystallographic, with the
    latter not having a crystal structure or symmetry set.

    Parameters
    ----------
    name
        The name to give the phase. If not given, a name will be inhereted
        from *structure*, if possible. If a phase is given, a copy of that
        phase is returned instead, and all further arguments are ignored.

    space_group
        Space group describing the symmetry of the phase. This can
        either be a number 1-230 corresponding to the 230 space groups
        defined in the International Tables for Crystallography, or a
        diffpy :class:`~diffpy.structure.spacegroup.SpaceGroup` object.
        If not given, the phase's space group is set to None.

    point_group
        Point group describing the non-translational symmetry of the
        phase. This can be an orix :class:`~orix.quaternion.symmetry.Symmetry`
        object or one of the known point
        group aliases (see the Notes section for details). It must be
        compatable with *space_group* if both are given. If not given,
        then either an appropriate point group will be derived from
        *space_group*, or set to None if *space_group* is also None.

    structure
        :class:`~diffpy.structure.structure.Structure` describing the
        lattice and atomic positions. If not given, a default
        :class:`~diffpy.structure.structure.Structure` is used. If
        a space group or point group has been defined, the structure
        will also include a :class:`~diffpy.structure.lattice.Lattice`
        with compatible default values.

    color
        The colour to use when plotting this phase. Default is blue.

    Examples
    --------
    The list of known point group aliases can be seen using the
    following command:
        >>> import orix.quaternion.symmetry as osm
        >>> pg_ aliases = [pg.name for pg in osm._groups]
        >>> print(pg_aliases)
    """

    def __init__(
        self,
        name: str | Phase | None = None,
        space_group: int | SpaceGroup | None = None,
        point_group: int | str | Symmetry | None = None,
        structure: dst.Structure | None = None,
        color: str | None = None,
    ) -> None:
        if isinstance(name, Phase):
            return Phase.__init__(
                self,
                name=name.name,
                space_group=name.space_group,
                point_group=name.point_group,
                structure=name.structure.copy(),
                color=name.color,
            )
        self.space_group = space_group  # Needs to be set before point group
        self.point_group = point_group
        self.color = color if color is not None else "tab:blue"

        if structure is None:
            pg = self.point_group
            if pg is not None and pg.system is not None:
                lat = default_lattice(pg.system)
            else:
                lat = dst.Lattice()
            structure = dst.Structure(lattice=lat)
        self.structure = structure

        if name is not None:
            self.name = name

    # ------------------------------ #
    #  Property Getters and Setters  #
    # ------------------------------ #
    @property
    def structure(self) -> dst.Structure:
        """The crystallographic unit cell.

        The unit cell is defined using the diffpy
        :class:`~diffpy.structure.Structure` class. For reciprocal
        space calculations, the structure must include a
        :class:`~diffpy.structure.Lattice`. For atomic distance
        calculations, the structure must include one or more
        :class:`~diffpy.structure.atom.Atom`.

        Notes
        -----
        %s
        """
        return self._structure

    @structure.setter
    def structure(self, value: dst.Structure) -> None:
        if not isinstance(value, dst.Structure):
            raise ValueError(f"{value} must be a diffpy.structure.Structure")

        # Ensure correct alignment
        old_matrix = value.lattice.base
        new_matrix = new_structure_matrix_from_alignment(old_matrix, x="a", z="c*")
        new_value = value.copy()

        # Ensure atom positions are expressed in the new basis
        new_value.placeInLattice(dst.Lattice(base=new_matrix))

        # Store old lattice for expand_asymmetric_unit
        self._diffpy_lattice = old_matrix
        if new_value.title == "" and hasattr(self, "_structure"):
            new_value.title = self.name

        self._structure = new_value

    structure.__doc__ %= _xtal2cart_docstring

    @property
    def name(self) -> str:
        """The name of the phase.

        Parameters
        ----------
        value : str
            Phase name.
        """
        return self.structure.title

    @name.setter
    def name(self, value: str) -> None:
        self.structure.title = str(value)

    @property
    def color(self) -> str:
        """Return or set the phase color name.

        Parameters
        ----------
        value : str
            A valid color string identifier recognized by
            :func:`matplotlib.colors.is_color_like`. If a valid alias is
            given, e.g. "g", the default name is used, e.g. "green".
        """
        return self._color

    @color.setter
    def color(self, value: str) -> None:
        """Set the phase color."""
        name, _ = get_matplotlib_color(value)
        self._color = name

    @property
    def color_rgb(self) -> tuple:
        """Return the phase color as RGB tuple."""
        return mcolors.to_rgb(self._color)

    @property
    def space_group(self) -> SpaceGroup | None:
        """The space group symmetry.

        Space groups are stored as diffpy :class:`~diffpy.spacegroup.SpaceGroup`
        objects, but can be set using an integer between 1 and 230.
        Note that changing the space group does not automatically update
        the point group or structure.
        """
        return self._space_group

    @space_group.setter
    def space_group(self, value: int | SpaceGroup | None) -> None:
        if isinstance(value, int):
            value = GetSpaceGroup(value)
        if not isinstance(value, SpaceGroup) and value is not None:
            raise ValueError(
                f"{value!r} must be of type {SpaceGroup}, an integer 1-230, or None"
            )
        # Overwrites any point group set before
        self._space_group: SpaceGroup | None = value

    @property
    def point_group(self) -> Symmetry | None:
        """The point group symmetry.

        Point Groups are stored as :class:`~orix.quaternion.symmetry.Symmetry`
        objects, but can be set using common shorthand names such as
        "m3m", "622", etc.
        """
        if self.space_group is not None:
            return get_point_group(self.space_group.number)
        else:
            return self._point_group

    @point_group.setter
    def point_group(self, value: int | str | Symmetry | None) -> None:
        if isinstance(value, int):
            value = str(value)
        if isinstance(value, str):
            for key, aliases in _EDAX_POINT_GROUP_ALIASES.items():
                if value in aliases:
                    value = key
                    break
            for point_group in _groups:
                if value == point_group.name:
                    value = point_group
                    break
        if not isinstance(value, Symmetry) and value is not None:
            raise ValueError(
                f"{value!r} must be of type {Symmetry}, the name of a valid point group"
                " as a string, or None"
            )
        else:
            if self.space_group is not None and value is not None:
                old_point_group_name = self.point_group.name
                if old_point_group_name != value.name:
                    warnings.warn(
                        "Setting space group to 'None', as current space group "
                        f"{self.space_group.short_name!r} is derived from current point"
                        f" group {old_point_group_name!r}"
                    )
                    self.space_group = None
            self._point_group = value

    @property
    def is_hexagonal(self) -> bool:
        """Return True for hexagonal and trigonal crystal structures."""
        return np.allclose(self.structure.lattice.abcABG()[3:], [90, 90, 120])

    @property
    def a_axis(self) -> Miller:
        """The :math:`a` axis of the crystal lattice.

        This is the vector describing the :math:`a` axis of the crystal
        lattice, expressed in the standard cartesian frame.

        Notes
        -----
        %s
        """
        return Miller(uvw=(1, 0, 0), phase=self)

    @property
    def b_axis(self) -> Miller:
        """The :math:`b` axis of the crystal lattice.

        This is the vector describing the :math:`b` axis of the crystal
        lattice, expressed in the standard cartesian frame.

        Notes
        -----
        %s
        """
        return Miller(uvw=(0, 1, 0), phase=self)

    @property
    def c_axis(self) -> Miller:
        """The :math:`c` axis of the crystal lattice.

        This is the vector describing the :math:`c` axis of the crystal
        lattice, expressed in the standard cartesian frame.

        Notes
        -----
        %s
        """
        return Miller(uvw=(0, 0, 1), phase=self)

    @property
    def ar_axis(self) -> Miller:
        """The :math:`a^{*}` axis of the reciprocal lattice.

        This is the vector describing the :math:`a^*` axis of the
        reciprocal lattice, expressed in the standard cartesian frame.

        Notes
        -----
        %s
        """
        return Miller(hkl=(1, 0, 0), phase=self)

    @property
    def br_axis(self) -> Miller:
        """The :math:`b^*` axis of the reciprocal lattice.

        This is the vector describing the :math:`b^*` axis of the
        reciprocal lattice, expressed in the standard cartesian frame.

        Notes
        -----
        %s
        """
        return Miller(hkl=(0, 1, 0), phase=self)

    @property
    def cr_axis(self) -> Miller:
        """The :math:`c^*` axis of the reciprocal lattice.

        This is the vector describing the :math:`c^*` axis of the
        reciprocal lattice, expressed in the standard cartesian frame.

        Notes
        -----
        %s
        """
        return Miller(hkl=(0, 0, 1), phase=self)

    # add repeated text to axis docstrings.
    a_axis.__doc__ %= _xtal2cart_docstring
    b_axis.__doc__ %= _xtal2cart_docstring
    c_axis.__doc__ %= _xtal2cart_docstring
    ar_axis.__doc__ %= _xtal2cart_docstring
    br_axis.__doc__ %= _xtal2cart_docstring
    cr_axis.__doc__ %= _xtal2cart_docstring

    def __repr__(self) -> str:
        if self.point_group is not None:
            pg_name = self.point_group.name
            ppg_name = self.point_group.proper_subgroup.name
        else:
            pg_name = self.point_group  # Should be None
            ppg_name = None
        if self.space_group is not None:
            sg_name = self.space_group.short_name
        else:
            sg_name = self.space_group  # Should be None
        return (
            f"<name: {self.name}. space group: {sg_name}. point group: {pg_name}. "
            f"proper point group: {ppg_name}. color: {self.color}>"
        )

    # ------------------------ #
    #  Class creation Methods  #
    # ------------------------ #
    @classmethod
    def triclinic(
        cls,
        name: str | None = None,
        space_group: int | SpaceGroup | None = None,
        point_group: int | str | Symmetry | None = None,
        color: str | None = None,
        a: float | None = None,
        b: float | None = None,
        c: float | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
    ) -> None:
        """Create a phase with a triclinic lattice and symmetry.

        Parameters
        ----------
            name
                The name to give the phase.

            space_group
                Space group describing the symmetry of the phase. This can
                either be the integers 1 or 2, corresponding to the two
                triclinic space group IDs, or a
                diffpy :class:`~diffpy.structure.spacegroup.SpaceGroup` object.
                If not given, the space group is set to SG#1 (P1).

            point_group
                Point group describing the non-translational symmetry of the
                phase. If not given, the point group is set to C1 ("1").

            color
                The colour to use when plotting this phase. Default is blue.

            a
                The unit cell length a (arbitrary units). Default is 1.

            b
                The unit cell length b (arbitrary units). Default is 1.

            c
                The unit cell length c (arbitrary units). Default is 1.

            alpha
                The angle (in degrees) between the b and c axes of the
                unit cell. Default is 90.

            beta
                The angle (in degrees) between the a and c axes of the
                unit cell. Default is 90.

            gamma
                The angle (in degrees) between the a and b axes of the
                unit cell. Default is 90.

        Returns
        -------
            phase
                A new triclinic phase.
        """
        if space_group is None and point_group is None:
            space_group = 1
        default = default_lattice("triclinic").abcABG()
        given = [a, b, c, alpha, beta, gamma]
        abcABG = [(j if j is not None else i) for i, j in zip(default, given)]
        lat = dst.Lattice(*abcABG)
        phase = cls(
            name=name,
            space_group=space_group,
            point_group=point_group,
            color=color,
            structure=dst.Structure(lattice=lat),
        )
        if phase.point_group.system != "triclinic":
            raise ValueError(f"{phase.point_group.name} is not a triclinic symmetry.")
        return phase

    @classmethod
    def monoclinic(
        cls,
        name: str | Phase | None = None,
        space_group: int | SpaceGroup | None = None,
        point_group: int | str | Symmetry | None = None,
        color: str | None = None,
        a: float = None,
        b: float = None,
        c: float = None,
        beta: float = None,
    ) -> None:
        """Create a phase with a monoclinic lattice and symmetry.

        Parameters
        ----------
            name
                The name to give the phase.

            space_group
                Space group describing the symmetry of the phase. This can
                either be an integer between 2 and 16 corresponding to one
                of the 13 monoclinic space group IDs, or a
                diffpy :class:`~diffpy.structure.spacegroup.SpaceGroup` object.
                If not given, the space group is set to SG#3 (P2).

            point_group
                Point group describing the non-translational symmetry of the
                phase. If not given, the point group is set to C2 ("2").

            color
                The colour to use when plotting this phase. Default is blue.

            a
                The unit cell length a (arbitrary units). Default is 1.

            b
                The unit cell length b (arbitrary units). Default is 1.

            c
                The unit cell length c (arbitrary units). Default is 1.

            beta
                The angle (in degrees) between the a and c axes of the
                unit cell. Default is 90.

        Returns
        -------
        phase
            A new  monoclinic phase.

        """
        if space_group is None and point_group is None:
            space_group = 3
        default = default_lattice("monoclinic").abcABG()
        given = [a, b, c, None, beta, None]
        abcABG = [(j if j is not None else i) for i, j in zip(default, given)]
        lat = dst.Lattice(*abcABG)
        phase = cls(
            name=name,
            space_group=space_group,
            point_group=point_group,
            color=color,
            structure=dst.Structure(lattice=lat),
        )
        if phase.point_group.system != "monoclinic":
            raise ValueError(f"{phase.point_group.name} is not a monoclinic symmetry.")
        return phase

    @classmethod
    def orthorhombic(
        cls,
        name: str | Phase | None = None,
        space_group: int | SpaceGroup | None = None,
        point_group: int | str | Symmetry | None = None,
        color: str | None = None,
        a: float = None,
        b: float = None,
        c: float = None,
    ) -> None:
        """Create a phase with an orthorhombic lattice and symmetry.

        Parameters
        ----------
            name
                The name to give the phase.

            space_group
                Space group describing the symmetry of the phase. This can
                either be an integer between 15 and 75 corresponding to one
                of the 59 orthorhombic space group IDs, or a
                diffpy :class:`~diffpy.structure.spacegroup.SpaceGroup` object.
                If not given, the space group is set to SG#15 (P222).

            point_group
                Point group describing the non-translational symmetry of the
                phase. If not given, the point group is set to D2 ("222").

            color
                The colour to use when plotting this phase. Default is blue.

            a
                The unit cell length a (arbitrary units). Default is 1.

            b
                The unit cell length b (arbitrary units). Default is 1.

            c
                The unit cell length c (arbitrary units). Default is 1.

        Returns
        -------
        phase
            A new orthorhombic phase.
        """
        if space_group is None and point_group is None:
            space_group = 16
        default = default_lattice("orthorhombic").abcABG()
        given = [a, b, c, None, None, None]
        abcABG = [(j if j is not None else i) for i, j in zip(default, given)]
        lat = dst.Lattice(*abcABG)
        phase = cls(
            name=name,
            space_group=space_group,
            point_group=point_group,
            color=color,
            structure=dst.Structure(lattice=lat),
        )
        if phase.point_group.system != "orthorhombic":
            raise ValueError(
                f"{phase.point_group.name} is not an orthorhombic symmetry."
            )
        return phase

    @classmethod
    def tetragonal(
        cls,
        name: str | Phase | None = None,
        space_group: int | SpaceGroup | None = None,
        point_group: int | str | Symmetry | None = None,
        color: str | None = None,
        a: float = None,
        c: float = None,
    ) -> None:
        """Create a phase with a tetragonal lattice and symmetry.

        Parameters
        ----------
            name
                The name to give the phase.

            space_group
                Space group describing the symmetry of the phase. This can
                either be an integer between 74 and 143 corresponding to one
                of the 68 tetragonal space group IDs, or a
                diffpy :class:`~diffpy.structure.spacegroup.SpaceGroup` object.
                If not given, the space group is set to SG#75 (P4).

            point_group
                Point group describing the non-translational symmetry of the
                phase. If not given, the point group is set to C4 ("4").

            color
                The colour to use when plotting this phase. Default is blue.

            a
                The unit cell length a (arbitrary units). Default is 1.

            c
                The unit cell length c (arbitrary units). Default is 1.

        Returns
        -------
        phase
            A new tetragonal phase.

        """
        if space_group is None and point_group is None:
            space_group = 75
        default = default_lattice("tetragonal").abcABG()
        given = [a, a, c, None, None, None]
        abcABG = [(j if j is not None else i) for i, j in zip(default, given)]
        lat = dst.Lattice(*abcABG)
        phase = cls(
            name=name,
            space_group=space_group,
            point_group=point_group,
            color=color,
            structure=dst.Structure(lattice=lat),
        )
        if phase.point_group.system != "tetragonal":
            raise ValueError(f"{phase.point_group.name} is not a tetragonal symmetry.")
        return phase

    @classmethod
    def trigonal(
        cls,
        name: str | Phase | None = None,
        space_group: int | SpaceGroup | None = None,
        point_group: int | str | Symmetry | None = None,
        color: str | None = None,
        a: float = None,
        alpha: float = None,
    ) -> None:
        """Create a phase with a trigonal (rhombohedral) lattice and symmetry.

        Parameters
        ----------
            name
                The name to give the phase.

            space_group
                Space group describing the symmetry of the phase. This can
                either be an integer between 142 and 168 corresponding to one
                of the 25 trigonal space group IDs, or a
                diffpy :class:`~diffpy.structure.spacegroup.SpaceGroup` object.
                If not given, the space group is set to SG#143 (P3).

            point_group
                Point group describing the non-translational symmetry of the
                phase. If not given, the point group is set to C3 ("3").

            color
                The colour to use when plotting this phase. Default is blue.

            a
                The unit cell length a (arbitrary units). Default is 1.

        Returns
        -------
        phase
            A new trigonal phase.

        Notes
        -----
        Trigonal symmetry requires a=b=c, alpha=beta=90, and gamma=120
        """
        if space_group is None and point_group is None:
            space_group = 143
        default = default_lattice("trigonal").abcABG()
        given = [a, a, a, alpha, alpha, alpha]
        abcABG = [(j if j is not None else i) for i, j in zip(default, given)]
        lat = dst.Lattice(*abcABG)
        phase = cls(
            name=name,
            space_group=space_group,
            point_group=point_group,
            color=color,
            structure=dst.Structure(lattice=lat),
        )
        if phase.point_group.system != "trigonal":
            raise ValueError(
                f"{phase.point_group.name} is not a trigonal (rhombohedral) symmetry."
            )
        return phase

    @classmethod
    def hexagonal(
        cls,
        name: str | Phase | None = None,
        space_group: int | SpaceGroup | None = None,
        point_group: int | str | Symmetry | None = None,
        color: str | None = None,
        a: float = None,
        c: float = None,
    ) -> None:
        """Create a phase with a hexagonal lattice and symmetry.


        Parameters
        ----------
            name
                The name to give the phase.

            space_group
                Space group describing the symmetry of the phase. This can
                either be an integer between 167 and 195 corresponding to one
                of the 27 hexagonal space group IDs, or a
                diffpy :class:`~diffpy.structure.spacegroup.SpaceGroup` object.
                If not given, the space group is set to SG#168 (P6).

            point_group
                Point group describing the non-translational symmetry of the
                phase. If not given, the point group is set to C6 ("6").

            color
                The colour to use when plotting this phase. Default is blue.

            a
                The unit cell length a (arbitrary units). Default is 1.

            c
                The unit cell length c (arbitrary units). Default is 1.

        Returns
        -------
        phase
            A new hexagonal phase.

        Notes
        -----
        Hexagonal symmetry requires a=b, alpha=beta=90, and gamma=120

        """
        if space_group is None and point_group is None:
            space_group = 168
        default = default_lattice("hexagonal").abcABG()
        given = [a, a, c, None, None, None]
        abcABG = [(j if j is not None else i) for i, j in zip(default, given)]
        lat = dst.Lattice(*abcABG)
        phase = cls(
            name=name,
            space_group=space_group,
            point_group=point_group,
            color=color,
            structure=dst.Structure(lattice=lat),
        )
        if phase.point_group.system != "hexagonal":
            raise ValueError(f"{phase.point_group.name} is not a hexagonal symmetry.")
        return phase

    @classmethod
    def cubic(
        cls,
        name: str | Phase | None = None,
        space_group: int | SpaceGroup | None = None,
        point_group: int | str | Symmetry | None = None,
        color: str | None = None,
        a: float = None,
    ) -> None:
        """Create a phase with a cubic lattice and symmetry.

        Parameters
        ----------
            name
                The name to give the Phase.

            space_group
                Space group describing the symmetry of the phase. This can
                either be an integer between 194 and 231 corresponding to one
                of the 36 cubic space group IDs, or a
                diffpy :class:`~diffpy.structure.spacegroup.SpaceGroup` object.
                If not given, the space group is set to SG#195 (P23).

            point_group
                Point group describing the non-translational symmetry of the
                phase. If not given, the point group is set to T ("23").

            color
                The colour to use when plotting this phase. Default is blue.

            a
                The unit cell length a (arbitrary units). Default is 1.

        Returns
        -------
        phase
            A new cubic phase.
        """
        if space_group is None and point_group is None:
            space_group = 195
        default = default_lattice("cubic").abcABG()
        given = [a, a, a, None, None, None]
        abcABG = [(j if j is not None else i) for i, j in zip(default, given)]
        lat = dst.Lattice(*abcABG)
        phase = cls(
            name=name,
            space_group=space_group,
            point_group=point_group,
            color=color,
            structure=dst.Structure(lattice=lat),
        )
        if phase.point_group.system != "cubic":
            raise ValueError(f"{phase.point_group.name} is not a cubic symmetry.")
        return phase

    @classmethod
    def from_cif(cls, filename: str | Path) -> Phase:
        r"""Create a Phase from a Crystallographic Information File (CIF).

        Parameters
        ----------
        filename
            Path to the \*.cif. The phase name is obtained from the file
            name.

        Returns
        -------
        phase
            New phase.

        Notes
        -----
        The file is read using :mod:`diffpy.structure` 's CIF file
        parser.

        See https://www.iucr.org/resources/cif for details on the CIF
        file format.
        """
        path = Path(filename)
        parser = p_cif.P_cif()
        name = path.stem
        structure = parser.parseFile(str(path))
        try:
            space_group = parser.spacegroup.number
        except AttributeError:  # pragma: no cover
            space_group = None
            warnings.warn(f"Could not read space group from CIF file {path!r}")
        return cls(name, space_group, structure=structure)

    def deepcopy(self) -> Phase:
        """Return a deep copy using :py:func:`~copy.deepcopy`."""
        return copy.deepcopy(self)

    def expand_asymmetric_unit(self) -> Phase:
        """Return a new phase with all symmetrically equivalent atoms.

        Returns
        -------
        expanded_phase
            New phase with the a :attr:`structure` with the unit cell
            filled with symmetrically equivalent atoms.

        Examples
        --------
        >>> from diffpy.structure import Atom, Lattice, Structure
        >>> import orix.crystal_map as ocm
        >>> atoms = [Atom("Si", xyz=(0, 0, 1))]
        >>> lattice = Lattice(4.04, 4.04, 4.04, 90, 90, 90)
        >>> structure = Structure(atoms = atoms,lattice=lattice)
        >>> phase = ocm.Phase(structure=structure, space_group=227)
        >>> phase.structure
        [Si   0.000000 0.000000 1.000000 1.0000]
        >>> expanded_phase = phase.expand_asymmetric_unit()
        >>> expanded_phase.structure
        [Si   0.000000 0.000000 0.000000 1.0000,
         Si   0.000000 0.500000 0.500000 1.0000,
         Si   0.500000 0.500000 0.000000 1.0000,
         Si   0.500000 0.000000 0.500000 1.0000,
         Si   0.750000 0.250000 0.750000 1.0000,
         Si   0.250000 0.250000 0.250000 1.0000,
         Si   0.250000 0.750000 0.750000 1.0000,
         Si   0.750000 0.750000 0.250000 1.0000]
        """
        if self.space_group is None:
            raise ValueError("Space group must be set")

        # Ensure atom positions are expressed in diffpy's convention
        diffpy_structure = self.structure.copy()
        diffpy_structure.placeInLattice(dst.Lattice(base=self._diffpy_lattice))
        xyz = diffpy_structure.xyz
        diffpy_structure.clear()

        eau = ExpandAsymmetricUnit(self.space_group, xyz)
        for atom, new_positions in zip(self.structure, eau.expandedpos):
            for pos in new_positions:
                new_atom = copy.deepcopy(atom)
                new_atom.xyz = pos
                # Only add new atom if not already present
                for present_atom in diffpy_structure:
                    if present_atom.element == new_atom.element and np.allclose(
                        present_atom.xyz, new_atom.xyz
                    ):
                        break
                else:
                    diffpy_structure.append(new_atom)

        # This handles conversion back to correct alignment
        expanded_phase = self.__class__(self)
        expanded_phase.structure = diffpy_structure

        return expanded_phase


def new_structure_matrix_from_alignment(
    old_matrix: np.ndarray,
    x: str | None = None,
    y: str | None = None,
    z: str | None = None,
) -> np.ndarray:
    """Return a new structure matrix given the old structure matrix and
    at least two aligned axes x, y, or z.

    The structure matrix defines the alignment of direct and reciprocal
    lattice base vectors with the cartesian reference frame of the
    crystal lattice defined by x, y, and z. x, y, and z are often termed
    :math:`e_i`.

    Parameters
    ----------
    old_matrix
        Old structure matrix, i.e. the 3x3 matrix of row base vectors
        expressed in Cartesian coordinates.
    x, y, z
        Which of the six axes "a", "b", "c", "a*", "b*", or "z*" are
        aligned with the base vectors of the cartesian crystal reference
        frame. At least two must be specified.

    Returns
    -------
    new_matrix
        New structure matrix according to the alignment.
    """
    if sum([i is None for i in [x, y, z]]) > 1:
        raise ValueError("At least two of x, y, z must be set.")

    # Old direct lattice base (row) vectors in Cartesian coordinates
    old_matrix = Vector3d(old_matrix)
    ad, bd, cd = old_matrix.unit

    # Old reciprocal lattice base vectors in cartesian coordinates
    ar = bd.cross(cd).unit
    br = cd.cross(ad).unit
    cr = ad.cross(bd).unit

    # New unit crystal base
    new_vectors = Vector3d.zero((3,))
    axes_mapping = {"a": ad, "b": bd, "c": cd, "a*": ar, "b*": br, "c*": cr}
    for i, al in enumerate([x, y, z]):
        if al in axes_mapping.keys():
            new_vectors[i] = axes_mapping[al]
    other_idx = {0: (1, 2), 1: (2, 0), 2: (0, 1)}
    for i in range(3):
        if np.isclose(new_vectors[i].norm, 0):
            other0, other1 = other_idx[i]
            new_vectors[i] = new_vectors[other0].cross(new_vectors[other1])

    # New crystal base
    new_matrix = new_vectors.dot(old_matrix.reshape(3, 1)).round(12)

    return new_matrix


def default_lattice(system: VALID_SYSTEMS) -> dst.Lattice:
    if system in ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "cubic"]:
        lat = dst.Lattice(1, 1, 1, 90, 90, 90)
    elif system in ["trigonal", "hexagonal"]:
        lat = dst.Lattice(1, 1, 1, 90, 90, 120)
    else:
        raise ValueError(f"Unknown crystal system {system!r}")
    return lat
