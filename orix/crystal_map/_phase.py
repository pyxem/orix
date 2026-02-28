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

from diffpy.structure import Lattice, Structure
from diffpy.structure.parsers import p_cif
from diffpy.structure.spacegroups import GetSpaceGroup, SpaceGroup
from diffpy.structure.symmetryutilities import ExpandAsymmetricUnit
import matplotlib.colors as mcolors
import numpy as np

from orix.plot._util.color import get_matplotlib_color
from orix.quaternion.symmetry import VALID_SYSTEMS, PointGroups, Symmetry
from orix.vector.miller import Miller
from orix.vector.vector3d import Vector3d


class Phase:
    """Symmetry and unit cell of a phase in a crystallographic map.

    The phase can be crystallographic or non-crystallographic, with the
    latter not having a crystal structure or symmetry set.

    Parameters
    ----------
    name
        Phase name. Overwrites the name in the *structure*. A phase can
        also be given, in which case a copy is returned and all other
        parameters are ignored.
    space_group
        Space group describing the symmetry operations resulting from
        associating the point group with a Bravais lattice, according
        to the International Tables for Crystallography. If not given,
        it is set to None.
    point_group
        Point group describing the symmetry operations of the phase's
        crystal structure, according to the International Tables for
        Crystallography. If neither this nor *space_group* is given, it
        is set to None. If not given but *space_group* is, it is derived
        from the space group. If both this and *space_group* is given,
        the space group must to be derived from adding translational
        symmetry to the point group.
    structure
        Unit cell with atoms and a lattice. If not given, a default
        :class:`~diffpy.structure.structure.Structure` compatible with
        the symmetry is used.
    color
        Phase color. If not given, it is set to the first default
        Matplotlib color "tab:blue".
    """

    def __init__(
        self,
        name: str | Phase | None = None,
        space_group: int | SpaceGroup | None = None,
        point_group: int | str | Symmetry | None = None,
        structure: Structure | None = None,
        color: str | None = None,
    ) -> None:
        if isinstance(name, Phase):
            return Phase.__init__(
                self,
                name.name,
                name.space_group,
                name.point_group,
                name.structure.copy(),
                name.color,
            )

        self.space_group = space_group  # Needs to be set before point group
        self.point_group = point_group

        self.color = color if color is not None else "tab:blue"

        if structure is None:
            pg = self.point_group
            if pg is not None and pg.system is not None:
                lat = default_lattice(pg.system)
            else:
                lat = Lattice()
            structure = Structure(lattice=lat)
        self.structure = structure

        if name is not None:
            self.name = name

    @property
    def structure(self) -> Structure:
        r"""Return or set the crystal structure containing a lattice
        (:class:`~diffpy.structure.lattice.Lattice`) and possibly many
        atoms (:class:`~diffpy.structure.atom.Atom`).

        Parameters
        ----------
        value : ~diffpy.structure.Structure
            Crystal structure. The cartesian reference frame of the
            crystal lattice is assumed to align :math:`a` with
            :math:`e_1` and :math:`c*` with :math:`e_3`. This alignment
            is assumed when transforming direct, reciprocal and
            cartesian vectors between these spaces.
        """
        return self._structure

    @structure.setter
    def structure(self, value: Structure) -> None:
        """Set the crystal structure."""
        if not isinstance(value, Structure):
            raise ValueError(f"{value} must be a diffpy.structure.Structure")

        # Ensure correct alignment
        old_matrix = value.lattice.base
        new_matrix = new_structure_matrix_from_alignment(old_matrix, x="a", z="c*")
        new_value = value.copy()

        # Ensure atom positions are expressed in the new basis
        new_value.placeInLattice(Lattice(base=new_matrix))

        # Store old lattice for expand_asymmetric_unit
        self._diffpy_lattice = old_matrix
        if new_value.title == "" and hasattr(self, "_structure"):
            new_value.title = self.name

        self._structure = new_value

    @property
    def name(self) -> str:
        """Return or set the phase name.

        Parameters
        ----------
        value : str
            Phase name.
        """
        return self.structure.title

    @name.setter
    def name(self, value: str) -> None:
        """Set the phase name."""
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
        """Return or set the space group.

        Parameters
        ----------
        value : int, SpaceGroup or None
            Space group. If an integer is passed, it must be between
            1-230.
        """
        return self._space_group

    @space_group.setter
    def space_group(self, value: int | SpaceGroup | None) -> None:
        """Set the space group."""
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
        """Return or set the point group.

        Parameters
        ----------
        value : int, str, Symmetry or None
            Point group.
        """
        if self.space_group is not None:
            return PointGroups.from_space_group(self.space_group.number)
        else:
            return self._point_group

    @point_group.setter
    def point_group(self, value: int | str | Symmetry | None) -> None:
        """Set the point group."""
        if isinstance(value, int):
            value = str(value)
        if isinstance(value, str):
            value = PointGroups.get(value)
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
        """Return whether the crystal structure is hexagonal/trigonal or
        not.
        """
        return np.allclose(self.structure.lattice.abcABG()[3:], [90, 90, 120])

    @property
    def a_axis(self) -> Miller:
        """Return the direct lattice vector :math:`a` in the cartesian
        reference frame of the crystal lattice :math:`e_i`.
        """
        return Miller(uvw=(1, 0, 0), phase=self)

    @property
    def b_axis(self) -> Miller:
        """Return the direct lattice vector :math:`b` in the cartesian
        reference frame of the crystal lattice :math:`e_i`.
        """
        return Miller(uvw=(0, 1, 0), phase=self)

    @property
    def c_axis(self) -> Miller:
        """Return the direct lattice vector :math:`c` in the cartesian
        reference frame of the crystal lattice :math:`e_i`.
        """
        return Miller(uvw=(0, 0, 1), phase=self)

    @property
    def ar_axis(self) -> Miller:
        """Return the reciprocal lattice vector :math:`a^{*}` in the
        cartesian reference frame of the crystal lattice :math:`e_i`.
        """
        return Miller(hkl=(1, 0, 0), phase=self)

    @property
    def br_axis(self) -> Miller:
        """Return the reciprocal lattice vector :math:`b^{*}` in the
        cartesian reference frame of the crystal lattice :math:`e_i`.
        """
        return Miller(hkl=(0, 1, 0), phase=self)

    @property
    def cr_axis(self) -> Miller:
        """Return the reciprocal lattice vector :math:`c^{*}` in the
        cartesian reference frame of the crystal lattice :math:`e_i`.
        """
        return Miller(hkl=(0, 0, 1), phase=self)

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

    @classmethod
    def from_cif(cls, filename: str | Path) -> Phase:
        r"""Return a new phase from a Crystallographic Information File
        (CIF).

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
        """Return a deep copy using :py:func:`~copy.deepcopy`
        function.
        """
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
        diffpy_structure.placeInLattice(Lattice(base=self._diffpy_lattice))
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


def default_lattice(system: VALID_SYSTEMS) -> Lattice:
    if system in [
        "triclinic",
        "monoclinic",
        "orthorhombic",
        "tetragonal",
        "cubic",
    ]:
        lat = Lattice()
    elif system in ["trigonal", "hexagonal"]:
        lat = Lattice(1, 1, 1, 90, 90, 120)
    else:
        raise ValueError(f"Unknown crystal system {system!r}")
    return lat
