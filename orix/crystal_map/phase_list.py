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

from collections import OrderedDict
import copy
from itertools import islice
from pathlib import Path
from typing import Generator
import warnings

from diffpy.structure import Structure
from diffpy.structure.parsers import p_cif
from diffpy.structure.spacegroups import GetSpaceGroup, SpaceGroup
import matplotlib.colors as mcolors
import numpy as np

from orix.quaternion.symmetry import (
    Symmetry,
    _groups,
    get_point_group,
    point_group_aliases,
)
from orix.vector import Miller, Vector3d

# All named Matplotlib colors (tableau and xkcd already lower case hex)
ALL_COLORS = mcolors.TABLEAU_COLORS
for k, v in {**mcolors.BASE_COLORS, **mcolors.CSS4_COLORS}.items():
    ALL_COLORS[k] = mcolors.to_hex(v)
ALL_COLORS.update(mcolors.XKCD_COLORS)


class Phase:
    """Name, symmetry, and color of a phase in a crystallographic map.

    Parameters
    ----------
    name
        Phase name. Overwrites the name in the ``structure`` object.
    space_group
        Space group describing the symmetry operations resulting from
        associating the point group with a Bravais lattice, according
        to the International Tables of Crystallography. If not given, it
        is set to ``None``.
    point_group
        Point group describing the symmetry operations of the phase's
        crystal structure, according to the International Tables of
        Crystallography. If not given and ``space_group`` is not given,
        it set to ``None``. If ``None`` is passed but ``space_group``
        is not ``None``, it is derived from the space group. If both
        ``point_group`` and ``space_group`` is not ``None``, the space
        group needs to be derived from the point group.
    structure
        Unit cell with atoms and a lattice. If not given, a default
        :class:`~diffpy.structure.structure.Structure` object is
        created.
    color
        Phase color. If not given, it is set to ``"tab:blue"`` (first
        among the default Matplotlib colors).

    Examples
    --------
    >>> from diffpy.structure import Atom, Lattice, Structure
    >>> from orix.crystal_map import Phase
    >>> p = Phase(
    ...     name="al",
    ...     space_group=225,
    ...     structure=Structure(
    ...         atoms=[Atom("al", [0, 0, 0])],
    ...         lattice=Lattice(0.405, 0.405, 0.405, 90, 90, 90)
    ...     )
    ... )
    >>> p
    <name: al. space group: Fm-3m. point group: m-3m. proper point group: 432. color: tab:blue>
    >>> p.structure
    [al   0.000000 0.000000 0.000000 1.0000]
    >>> p.structure.lattice
    Lattice(a=0.405, b=0.405, c=0.405, alpha=90, beta=90, gamma=90)
    """

    def __init__(
        self,
        name: str | None = None,
        space_group: int | SpaceGroup | None = None,
        point_group: int | str | Symmetry | None = None,
        structure: Structure | None = None,
        color: str | None = None,
    ) -> None:
        self.structure = structure if structure is not None else Structure()
        if name is not None:
            self.name = name
        self.space_group = space_group  # Needs to be set before point group
        self.point_group = point_group
        self.color = color if color is not None else "tab:blue"

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
        if isinstance(value, Structure):
            # Ensure correct alignment
            old_matrix = value.lattice.base
            new_matrix = _new_structure_matrix_from_alignment(old_matrix, x="a", z="c*")
            value = copy.deepcopy(value)
            # Ensure atom positions are expressed in the new basis
            old_xyz_cartn = value.xyz_cartn
            value.lattice.setLatBase(new_matrix)
            value.xyz_cartn = old_xyz_cartn
            if value.title == "" and hasattr(self, "_structure"):
                value.title = self.name
            self._structure = value
        else:
            raise ValueError(f"{value} must be a diffpy.structure.Structure object.")

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
        """Return or set the name of phase color.

        Parameters
        ----------
        value : str
            A valid color identifier. See
            :func:`matplotlib.colors.is_color_like`.
        """
        return self._color

    @color.setter
    def color(self, value: str) -> None:
        """Set the phase color."""
        value_hex = mcolors.to_hex(value)
        for name, color_hex in ALL_COLORS.items():
            if color_hex == value_hex:
                self._color = name
                break

    @property
    def color_rgb(self) -> tuple:
        """Return the phase color as RGB tuple."""
        return mcolors.to_rgb(self.color)

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
                f"'{value}' must be of type {SpaceGroup}, an integer 1-230, or None."
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
            return get_point_group(self.space_group.number)
        else:
            return self._point_group

    @point_group.setter
    def point_group(self, value: int | str | Symmetry | None) -> None:
        """Set the point group."""
        if isinstance(value, int):
            value = str(value)
        if isinstance(value, str):
            for correct, aliases in point_group_aliases.items():
                if value in aliases:
                    value = correct
                    break
            for point_group in _groups:
                if value == point_group.name:
                    value = point_group
                    break
        if not isinstance(value, Symmetry) and value is not None:
            raise ValueError(
                f"'{value}' must be of type {Symmetry}, the name of a valid point"
                " group as a string, or None."
            )
        else:
            if self.space_group is not None and value is not None:
                old_point_group_name = self.point_group.name
                if old_point_group_name != value.name:
                    warnings.warn(
                        "Setting space group to 'None', as current space group "
                        f"'{self.space_group.short_name}' is derived from current point"
                        f" group '{old_point_group_name}'."
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
        """Return a new phase from a CIF file using
        :mod:`diffpy.structure`'s CIF file parser.

        Parameters
        ----------
        filename
            Complete path to CIF file with ".cif" file ending. The phase
            name is obtained from the file name.

        Returns
        -------
        phase
            New phase.
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


class PhaseList:
    """A list of phases in a crystallographic map.

    Each phase in the list must have a unique phase id and name.

    Parameters
    ----------
    phases
        A list or dict of phases or a single phase. The other
        arguments are ignored if this is passed.
    names
        Phase names. Overwrites the names in the ``structure`` objects.
    space_groups
        Space groups.
    point_groups
        Point groups.
    colors
        Phase colors.
    ids
        Phase IDs.
    structures
        Unit cells with atoms and a lattice of each phase. If not given,
        a default :class:`diffpy.structure.Structure` is created for
        each phase.

    Examples
    --------
    >>> from diffpy.structure import Atom, Lattice, Structure
    >>> from orix.crystal_map import Phase, PhaseList
    >>> pl = PhaseList(
    ...     names=["al", "cu"],
    ...     space_groups=[225] * 2,
    ...     structures=[
    ...         Structure(
    ...             atoms=[Atom("al", [0] * 3)],
    ...             lattice=Lattice(0.405, 0.405, 0.405, 90, 90, 90)
    ...         ),
    ...         Structure(
    ...             atoms=[Atom("cu", [0] * 3)],
    ...             lattice=Lattice(0.361, 0.361, 0.361, 90, 90, 90)
    ...         ),
    ...     ]
    ... )
    >>> pl
    Id  Name  Space group  Point group  Proper point group       Color
     0    al        Fm-3m         m-3m                 432    tab:blue
     1    cu        Fm-3m         m-3m                 432  tab:orange
    >>> pl["al"].structure
    [al   0.000000 0.000000 0.000000 1.0000]

    A phase list can be indexed in multiple ways:

    Return a phase if only one phase matches the key

    >>> pl[0]  # Index with a single phase id
    <name: al. space group: Fm-3m. point group: m-3m. proper point group: 432. color: tab:blue>
    >>> pl["cu"]  # Index with a phase name
    <name: cu. space group: Fm-3m. point group: m-3m. proper point group: 432. color: tab:orange>
    >>> pl[:1]
    <name: al. space group: Fm-3m. point group: m-3m. proper point group: 432. color: tab:blue>

    Return a phase list

    >>> pl[0:]  # Index with slices
    Id  Name  Space group  Point group  Proper point group       Color
     0    al        Fm-3m         m-3m                 432    tab:blue
     1    cu        Fm-3m         m-3m                 432  tab:orange
    >>> pl["al", "cu"]  # Index with a tuple of phase names
    Id  Name  Space group  Point group  Proper point group       Color
     0    al        Fm-3m         m-3m                 432    tab:blue
     1    cu        Fm-3m         m-3m                 432  tab:orange
    >>> pl[0, 1]  # Index with a tuple of phase phase_ids
    Id  Name  Space group  Point group  Proper point group       Color
     0    al        Fm-3m         m-3m                 432    tab:blue
     1    cu        Fm-3m         m-3m                 432  tab:orange
    >>> pl[[0, 1]]  # Index with a list of phase_ids
    Id  Name  Space group  Point group  Proper point group       Color
     0    al        Fm-3m         m-3m                 432    tab:blue
     1    cu        Fm-3m         m-3m                 432  tab:orange
    """

    def __init__(
        self,
        phases: Phase | list[Phase] | dict[int, Phase] | None = None,
        names: str | list[str] | None = None,
        space_groups: int | SpaceGroup | list[int | SpaceGroup] | None = None,
        point_groups: str | int | Symmetry | list[str | int | Symmetry] | None = None,
        colors: str | list[str] | None = None,
        ids: int | list[int] | np.ndarray | None = None,
        structures: Structure | list[Structure] | None = None,
    ) -> None:
        """Create a new phase list."""
        d = {}
        if isinstance(phases, list):
            try:
                if isinstance(next(iter(phases)), Phase):
                    if ids is None:
                        ids = np.arange(len(phases))
                    d = dict(zip(ids, phases))
            except StopIteration:
                pass
        elif isinstance(phases, dict):
            try:
                if isinstance(next(iter(phases.values())), Phase):
                    d = phases
            except StopIteration:
                pass
        elif isinstance(phases, Phase):
            if ids is None:
                ids = 0
            d = {ids: phases}
        else:
            # Ensure possible single strings or single objects have
            # iterables of length 1
            if isinstance(names, str):
                names = list((names,))
            if isinstance(space_groups, (SpaceGroup, int)):
                space_groups = list((space_groups,))
            if isinstance(point_groups, (str, Symmetry, int)):
                point_groups = list((point_groups,))
            if isinstance(colors, (str, tuple)):
                colors = list((colors,))
            if isinstance(ids, int):
                ids = [ids]
            if isinstance(structures, Structure):
                structures = [structures]

            # Get the maximum number of entries in the input lists (also
            # handling the case where some lists are None)
            max_entries = max(
                [
                    len(i) if i is not None else 0
                    for i in [names, space_groups, point_groups, ids, structures]
                ]
            )

            if ids is None:
                ids = list(np.arange(max_entries))

            # Get first 2 * n entries in color list (for good measure)
            all_colors = list(islice(ALL_COLORS.keys(), 2 * max_entries))[::-1]

            # Create phase dictionary
            d = {}
            phase_id_iter = 0
            used_colors = []
            for i in range(max_entries):
                # Get name or None
                try:
                    name = names[i]
                except (IndexError, TypeError):
                    name = None

                # Get space group or None
                try:
                    space_group = space_groups[i]
                except (IndexError, TypeError):
                    space_group = None

                # Get point group or None
                try:
                    point_group = point_groups[i]
                except (IndexError, TypeError):
                    point_group = None

                # Get a color (always)
                try:
                    if colors[i] is not None:
                        color = colors[i]
                    else:
                        color = all_colors.pop()
                except (IndexError, TypeError):
                    color = all_colors.pop()
                while color in used_colors:
                    color = all_colors.pop()

                # Get a phase_id (always)
                try:
                    phase_id = ids[i]
                except IndexError:
                    phase_id = max(ids) + phase_id_iter + 1
                    phase_id_iter += 1

                # Get a structure or None
                try:
                    structure = structures[i]
                except (IndexError, TypeError):
                    structure = None

                d[phase_id] = Phase(
                    name=name,
                    space_group=space_group,
                    point_group=point_group,
                    color=color,
                    structure=structure,
                )

                # To ensure color aliases are added to `used_colors`
                used_colors.append(d[phase_id].color)

        # Finally create dictionary of phases
        self._dict = OrderedDict(sorted(d.items()))

    @property
    def names(self) -> list[str]:
        """Return the phases' names."""
        return [phase.name for _, phase in self]

    @property
    def space_groups(self) -> list[SpaceGroup]:
        """Return the phases' space groups."""
        return [phase.space_group for _, phase in self]

    @property
    def point_groups(self) -> list[Symmetry]:
        """Return the phases' point groups."""
        return [phase.point_group for _, phase in self]

    @property
    def colors(self) -> list[str]:
        """Return the phases' colors."""
        return [phase.color for _, phase in self]

    @property
    def colors_rgb(self) -> list[tuple]:
        """Return the phases' RGB color values."""
        return [phase.color_rgb for _, phase in self]

    @property
    def size(self) -> int:
        """Return the number of phases in the list."""
        return len(self._dict.items())

    @property
    def ids(self) -> list[int]:
        """Return the unique phase IDs in the list of phases."""
        return list(self._dict.keys())

    @property
    def structures(self) -> list[Structure]:
        """Return the phases' structures."""
        return [phase.structure for _, phase in self]

    def __getitem__(self, key) -> PhaseList | Phase:
        """Return a PhaseList or a Phase object, depending on the number
        of phases in the list matches the `key`.
        """
        # Make key iterable if it isn't already
        if not isinstance(key, (tuple, slice, list, np.ndarray)):
            key_iter = (key,)
        else:
            key_iter = key

        d = {}
        if isinstance(key_iter, str) or (
            isinstance(key_iter, tuple)
            and isinstance(key_iter[0], str)
            or (isinstance(key_iter, list) and isinstance(key_iter[0], str))
        ):
            # Use set to remove duplicates
            for key_name in list(set(key_iter)):
                for i, phase in self._dict.items():
                    if key_name == phase.name:
                        d[i] = phase
        elif isinstance(key_iter, (int, tuple, list, np.ndarray)):
            for i in list(set(key_iter)):  # Use set to remove duplicates
                d[i] = self._dict[i]
        elif isinstance(key_iter, slice):
            # Dicts cannot be sliced, hence this work-around
            id_arr_start = -1 if self.ids[0] == -1 else 0
            id_arr = np.arange(id_arr_start, max(self.ids) + 1)
            sliced_arr = id_arr[key_iter]
            ids_in_slice = [i for i in sliced_arr if i in self.ids]
            d = {i: self._dict[i] for i in ids_in_slice}

        # Raise KeyError if key is missing (not in the container), per
        # Python docs:
        # https://docs.python.org/3/reference/datamodel.html#object.__getitem__
        if d == {}:
            raise KeyError(f"{key} was not found in the phase list.")

        # Ensure integer phase IDs
        d = {int(i): p for i, p in d.items()}

        # Return a Phase object if only one phase matches the key
        if len(d) == 1:
            return [i for i in d.values()][0]
        else:
            return PhaseList(d)

    def __delitem__(self, key: int | str) -> None:
        """Delete a phase from the phase list.

        Parameters
        ----------
        key
            ID or name of a phase in the phase list.
        """
        if np.issubdtype(type(key), np.signedinteger):
            self._dict.pop(key)
        elif isinstance(key, str):
            matching_phase_id = None
            for phase_id, phase in self._dict.items():
                if key == phase.name:
                    matching_phase_id = phase_id
                    break
            if matching_phase_id is None:
                raise KeyError(f"{key} is not among the phase names {self.names}.")
            else:
                self._dict.pop(matching_phase_id)
        else:
            raise TypeError(f"{key} is an invalid phase ID or name.")

    def __iter__(self) -> Generator[tuple[int, Phase]]:
        """Return a tuple with phase ID and Phase object, in that order."""
        for phase_id, phase in self._dict.items():
            yield phase_id, phase

    def __repr__(self) -> str:
        if self.size == 0:
            return "No phases."

        # Ensure attributes set to None are treated OK
        names = ["None" if not i else i for i in self.names]
        sg_names = ["None" if not i else i.short_name for i in self.space_groups]
        pg_names = ["None" if not i else i.name for i in self.point_groups]
        ppg_names = [
            "None" if not i else i.proper_subgroup.name for i in self.point_groups
        ]

        # Determine column widths (allowing PhaseList to be empty)
        id_len = 2
        name_len = max(max([len(i) for i in names]), 4)
        sg_len = max(max([len(i) for i in sg_names]), 11)
        pg_len = max(max([len(i) for i in pg_names]), 11)
        ppg_len = max(max([len(i) for i in ppg_names]), 18)
        col_len = max(max([len(i) for i in self.colors]), 5)

        # Column alignment
        align = ">"  # right: ">", left: "<"

        # Header
        representation = (
            "{:{align}{width}}  ".format("Id", width=id_len, align=align)
            + "{:{align}{width}}  ".format("Name", width=name_len, align=align)
            + "{:{align}{width}}  ".format("Space group", width=sg_len, align=align)
            + "{:{align}{width}}  ".format("Point group", width=pg_len, align=align)
            + "{:{align}{width}}  ".format(
                "Proper point group", width=ppg_len, align=align
            )
            + "{:{align}{width}}".format("Color", width=col_len, align=align)
        )

        # Overview of each phase
        for i, phase_id in enumerate(self.ids):
            representation += (
                f"\n{phase_id:{align}{id_len}}  "
                + f"{names[i]:{align}{name_len}}  "
                + f"{sg_names[i]:{align}{sg_len}}  "
                + f"{pg_names[i]:{align}{pg_len}}  "
                + f"{ppg_names[i]:{align}{ppg_len}}  "
                + f"{self.colors[i]:{align}{col_len}}"
            )

        return representation

    def deepcopy(self) -> PhaseList:
        """Return a deep copy using :func:`copy.deepcopy` function."""
        return copy.deepcopy(self)

    def add_not_indexed(self) -> None:
        """Add a dummy phase to assign to not indexed data points.

        The phase, named ``"not_indexed"``, has a
        :attr:`Phase.point_group` equal to ``None``, and a white color
        when plotted.
        """
        self._dict[-1] = Phase(name="not_indexed", color="white")
        self.sort_by_id()

    def sort_by_id(self) -> None:
        """Sort the list according to phase ID in-place."""
        self._dict = OrderedDict(sorted(self._dict.items()))

    def id_from_name(self, name: str) -> int:
        """Return the phase ID from a phase name or raise an error if
        the phase is not in the list.

        Parameters
        ----------
        name
            Phase name.

        Returns
        -------
        phase_id
            Phase ID.
        """
        for phase_id, phase in self:
            if name == phase.name:
                return phase_id
        raise KeyError(f"'{name}' is not among the phase names {self.names}.")

    def add(self, value: Phase | list[Phase] | PhaseList) -> None:
        """Add phases to the end of a phase list in-place, incrementing
        the phase IDs.

        Parameters
        ----------
        value
            Phase(s) to add. If a ``PhaseList`` is added, the phase IDs
            in the old list are lost.

        Examples
        --------
        >>> from orix.crystal_map import Phase, PhaseList
        >>> pl = PhaseList(names=["a", "b"], space_groups=[10, 20])
        >>> pl.add(Phase("c", space_group=30))
        >>> pl.add([Phase("d", space_group=40), Phase("e")])
        >>> pl.add(PhaseList(names=["f", "g"], space_groups=[60, 70]))
        >>> pl
        Id  Name  Space group  Point group  Proper point group       Color
         0     a         P2/m          2/m                 112    tab:blue
         1     b        C2221          222                 222  tab:orange
         2     c         Pnc2          mm2                 211   tab:green
         3     d         Ama2          mm2                 211     tab:red
         4     e         None         None                None  tab:purple
         5     f         Pbcn          mmm                 222   tab:brown
         6     g         Fddd          mmm                 222    tab:pink
        """
        if isinstance(value, Phase):
            value = [value]
        if isinstance(value, PhaseList):
            value = [i for _, i in value]
        for phase in value:
            if phase.name in self.names:
                raise ValueError(
                    f"'{phase.name}' is already in the phase list {self.names}"
                )

            # Ensure a new color
            if phase.color in self.colors:
                for color_name in ALL_COLORS.keys():
                    if color_name not in self.colors:
                        phase.color = color_name
                        break

            # Increment the highest phase ID
            if self.ids:
                new_phase_id = max(self.ids) + 1
            else:  # `self.phase_ids` is an empty list
                new_phase_id = 0

            # Finally, add the phase to the list
            self._dict[new_phase_id] = phase


def _new_structure_matrix_from_alignment(
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
