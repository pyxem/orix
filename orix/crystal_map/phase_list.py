# -*- coding: utf-8 -*-
# Copyright 2018-2020 The pyXem developers
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

from collections import OrderedDict
import copy
from itertools import islice

from diffpy.structure import Structure
import matplotlib.colors as mcolors
import numpy as np

from orix.quaternion.symmetry import _groups, Symmetry

# All named Matplotlib colors (tableau and xkcd already lower case hex)
ALL_COLORS = mcolors.TABLEAU_COLORS
for k, v in {**mcolors.BASE_COLORS, **mcolors.CSS4_COLORS}.items():
    ALL_COLORS[k] = mcolors.to_hex(v)
ALL_COLORS.update(mcolors.XKCD_COLORS)

# Point group alias mapping
# Why is this needed? Well, e.g. in EDAX TSL OIM Analysis 7.2, point
# group 432 is entered as 43...
POINT_GROUP_ALIASES = {
    "432": "43",
}


class Phase:
    """Name, crystal symmetry, and color of a phase in a crystallographic
    map.

    Attributes
    ----------
    color : str
        Name of phase color in Matplotlib's list of named colors.
    color_rgb : tuple
        RGB values of phase color, obtained from the color name.
    name : str
        Phase name.
    structure : diffpy.structure.Structure
        Unit cell with atoms and lattice.
    symmetry : orix.quaternion.symmetry.Symmetry
        Crystal symmetries of the phase.

    Methods
    -------
    deepcopy()
        Return a deep copy using :py:func:`~copy.deepcopy` function.
    """

    def __init__(self, name=None, structure=None, symmetry=None, color=None):
        """
        Parameters
        ----------
        name : str, optional
            Phase name. Overwrites the name in the `structure` object.
        structure : diffpy.structure.Structure, optional
            Unit cell with atoms and a lattice. If None is passed
            (default), a default :class:`diffpy.structure.Structure`
            object is created.
        symmetry : str or orix.quaternion.symmetry.Symmetry, optional
            Point group of phase's crystal symmetry. If None is passed
            (default), it set to None.
        color : str, optional
            Phase color. If None is passed (default), it is set to
            'tab:blue' (first among the default Matplotlib colors).

        Examples
        --------
        >>> from diffpy.structure import Atom, Lattice, Structure
        >>> from orix.crystal_map import Phase
        >>> p = Phase(
        ...     name="al",
        ...     symmetry="m-3m",
        ...     structure=Structure(
        ...         atoms=[Atom("al", [0, 0, 0])],
        ...         lattice=Lattice(0.405, 0.405, 0.405, 90, 90, 90)
        ...     )
        ... )
        >>> p
        <name: al. symmetry: m-3m. color: tab:blue>
        >>> p.structure
        [al   0.000000 0.000000 0.000000 1.0000]
        >>> p.structure.lattice
        Lattice(a=0.405, b=0.405, c=0.405, alpha=90, beta=90, gamma=90)
        """
        self.structure = structure if structure is not None else Structure()
        if name is not None:
            self.name = name
        self.symmetry = symmetry
        self.color = color if color is not None else "tab:blue"

    @property
    def structure(self):
        """Phase unit cell."""
        return self._structure

    @structure.setter
    def structure(self, value):
        """Set phase structure."""
        if isinstance(value, Structure):
            if value.title == "" and hasattr(self, "_structure"):
                value.title = self.name
            self._structure = value
        else:
            raise ValueError(f"{value} must be a diffpy.structure.Structure object.")

    @property
    def name(self):
        """Phase name."""
        return self.structure.title

    @name.setter
    def name(self, value):
        """Set phase name as string."""
        self.structure.title = str(value)

    @property
    def color(self):
        """Name of phase color."""
        return self._color

    @color.setter
    def color(self, value):
        """Set phase color from something considered a valid color by
        :func:`matplotlib.colors.is_color_like`.
        """
        value_hex = mcolors.to_hex(value)
        for name, color_hex in ALL_COLORS.items():
            if color_hex == value_hex:
                self._color = name
                break

    @property
    def color_rgb(self):
        """Phase color as RGB tuple."""
        return mcolors.to_rgb(self.color)

    @property
    def symmetry(self):
        """Crystal symmetry of phase."""
        return self._symmetry

    @symmetry.setter
    def symmetry(self, value):
        """Set crystal symmetry of phase."""
        if isinstance(value, int):
            value = str(value)
        if isinstance(value, str):
            for correct, alias in POINT_GROUP_ALIASES.items():
                if value == alias:
                    value = correct
                    break
            for symmetry in _groups:
                if value.replace("-", "") == symmetry.name.replace("-", ""):
                    value = symmetry
                    break
        if not isinstance(value, Symmetry) and value is not None:
            raise ValueError(
                f"'{value}' must be of type {Symmetry}, the name of a valid point"
                " group as a string, or None."
            )
        else:
            self._symmetry = value

    def __repr__(self):
        if self.symmetry:
            symmetry_name = self.symmetry.name
        else:
            symmetry_name = self.symmetry  # Which should be None
        return (
            f"<name: {self.name}. symmetry: {symmetry_name}. color: " f"{self.color}>"
        )

    def deepcopy(self):
        """Return a deep copy using :py:func:`~copy.deepcopy` function."""
        return copy.deepcopy(self)


class PhaseList:
    """A list of phases in a crystallographic map.

    Each phase in the list must have a unique phase id and name.

    Attributes
    ----------
    colors : list of tuple
        List of tuples with three entries, RGB, defining phase colors.
    names : list of str
        List of phase names.
    ids : list of int
        List of unique phase indices in a crystallographic map as
        imported.
    size : int
        Number of phases in list.
    structures : list of diffpy.structure.Structure
        List of unit cells with atoms and the lattice of phase.
    symmetries : list of orix.quaternion.symmetry.Symmetry
        List of phase crystal symmetries.

    Methods
    -------
    add_not_indexed()
        Add a dummy phase to assign to not indexed data points.
    deepcopy()
        Return a deep copy using :py:func:`~copy.deepcopy` function.
    id_from_name(name):
        Get phase ID from phase name.
    sort_by_id()
        Sort list according to phase ID.
    """

    def __init__(
        self,
        phases=None,
        names=None,
        symmetries=None,
        colors=None,
        ids=None,
        structures=None,
    ):
        """
        Parameters
        ----------
        phases : orix.crystal_map.Phase, a list of orix.crystal_map.Phase\
                or a dictionary of orix.crystal_map.Phase, optional
            A list or dict of phases or a single phase. The other
            arguments are ignored if this is passed.
        names : str or list of str, optional
            Phase names. Overwrites the names in the `structure` objects.
        symmetries : str, int, orix.quaternion.symmetry.Symmetry or list\
                of str, int or orix.quaternion.symmetry.Symmetry, optional
            Point group symmetries.
        colors : str or list of str, optional
            Phase colors.
        ids : int, list of int or numpy.ndarray of int, optional
            Phase IDs.
        structures : diffpy.structure.Structure or list of
                diffpy.structure.Structure, optional
            Unit cells with atoms and a lattice of each phase. If None
            is passed (default), a default
            :class:`diffpy.structure.Structure` object is created for each
            phase.

        Examples
        --------
        >>> from diffpy.structure import Atom, Lattice, Structure
        >>> from orix.crystal_map import Phase, PhaseList
        >>> pl = PhaseList(
        ...     names=["al", "cu"],
        ...     symmetries=["m-3m"] * 2,
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
        Id  Name  Symmetry       Color
         0    al      m-3m    tab:blue
         1    cu      m-3m  tab:orange
        >>> pl["al"].structure
        [al   0.000000 0.000000 0.000000 1.0000]
        """
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
            if isinstance(symmetries, (str, Symmetry, int)):
                symmetries = list((symmetries,))
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
                    for i in [names, symmetries, ids, structures]
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

                # Get symmetry or None
                try:
                    symmetry = symmetries[i]
                except (IndexError, TypeError):
                    symmetry = None

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
                    name=name, symmetry=symmetry, color=color, structure=structure
                )

                # To ensure color aliases are added to `used_colors`
                used_colors.append(d[phase_id].color)

        # Finally create dictionary of phases
        self._dict = OrderedDict(sorted(d.items()))

    @property
    def names(self):
        """List of phase names in the list."""
        return [phase.name for _, phase in self]

    @property
    def symmetries(self):
        """List of crystal symmetries of phases in the list."""
        return [phase.symmetry for _, phase in self]

    @property
    def colors(self):
        """List of phase color names in the list."""
        return [phase.color for _, phase in self]

    @property
    def colors_rgb(self):
        """List of phase color RGB values in the list."""
        return [phase.color_rgb for _, phase in self]

    @property
    def size(self):
        """Number of phases in the list."""
        return len(self._dict.items())

    @property
    def ids(self):
        """Unique phase IDs in the list of phases."""
        return list(self._dict.keys())

    @property
    def structures(self):
        """List of phase structures."""
        return [phase.structure for _, phase in self]

    def __getitem__(self, key):
        """Return a PhaseList or a Phase object, depending on the number
        of phases in the list matches the `key`.

        Examples
        --------
        A PhaseList object can be indexed in multiple ways.

        >>> pl = PhaseList(names=['a', 'b'], symmetries=['1', '3'])
        >>> pl
        Id  Name  Symmetry  Color
        0   a     1         tab:blue
        1   b     3         tab:orange

        Return a Phase object if only one phase matches the key

        >>> pl[0]  # Index with a single phase id
        <name: a. symmetry: 1. color: tab:blue>
        >>> pl['b']  # Index with a phase name
        <name: b. symmetry: 3. color: tab:orange>
        >>> pl[:1]
        <name: b. symmetry: 3. color: tab:orange>

        Return a PhaseList object

        >>> pl[0:]  # Index with slices
        Id  Name  Symmetry  Color
        0   a     1         tab:blue
        1   b     3         tab:orange
        >>> pl['a', 'b']  # Index with a tuple of phase names
        Id  Name  Symmetry  Color
        0   a     1         tab:blue
        1   b     3         tab:orange
        >>> pl[0, 1]  # Index with a tuple of phase phase_ids
        Id  Name  Symmetry  Color
        0   a     1         tab:blue
        1   b     3         tab:orange
        >>> pl[[0, 1]]  # Index with a list of phase_ids
        Id  Name  Symmetry  Color
        0   a     1         tab:blue
        1   b     3         tab:orange
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

    def __setitem__(self, key, value):
        """Add a phase to the list with a name, symmetry and structure."""
        if key not in self.names:
            # Make sure the new phase gets a new color
            color_new = None
            for color_name in ALL_COLORS.keys():
                if color_name not in self.colors:
                    color_new = color_name
                    break

            # Create new ID
            if self.ids:
                new_phase_id = max(self.ids) + 1
            else:  # `self.phase_ids` is an empty list
                new_phase_id = 0

            self._dict[new_phase_id] = Phase(name=key, symmetry=value, color=color_new)
        else:
            raise ValueError(f"{key} is already in the phase list {self.names}.")

    def __delitem__(self, key):
        """Delete a phase from the phase list.

        Parameters
        ----------
        key : int or str
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

    def __iter__(self):
        """Return a tuple with phase ID and Phase object, in that order.
        """
        for phase_id, phase in self._dict.items():
            yield phase_id, phase

    def __repr__(self):
        if self.size == 0:
            return "No phases."

        # Ensure attributes set to None are treated OK
        names = ["None" if not i else i for i in self.names]
        symmetry_names = ["None" if not i else i.name for i in self.symmetries]

        # Determine column widths (allowing PhaseList to be empty)
        id_len = 2
        name_len = 4
        if names:
            name_len = max(max([len(i) for i in names]), name_len)
        sym_len = 8
        if symmetry_names:
            sym_len = max(max([len(i) for i in symmetry_names]), sym_len)
        col_len = 5
        if self.colors:
            col_len = max(max([len(i) for i in self.colors]), col_len)

        # Column alignment
        align = ">"  # right: ">", left: "<"

        # Header
        representation = (
            "{:{align}{width}}  ".format("Id", width=id_len, align=align)
            + "{:{align}{width}}  ".format("Name", width=name_len, align=align)
            + "{:{align}{width}}  ".format("Symmetry", width=sym_len, align=align)
            + "{:{align}{width}}".format("Color", width=col_len, align=align)
        )

        # Overview of each phase
        for i, phase_id in enumerate(self.ids):
            representation += (
                f"\n{phase_id:{align}{id_len}}  "
                + f"{names[i]:{align}{name_len}}  "
                + f"{symmetry_names[i]:{align}{sym_len}}  "
                + f"{self.colors[i]:{align}{col_len}}"
            )

        return representation

    def deepcopy(self):
        """Return a deep copy using :func:`copy.deepcopy` function."""
        return copy.deepcopy(self)

    def add_not_indexed(self):
        """Add a dummy phase to assign to not indexed data points.

        The phase, named "not_indexed", has a "symmetry" equal to None,
        and a white color when plotted.
        """
        self._dict[-1] = Phase(name="not_indexed", symmetry=None, color="white")
        self.sort_by_id()

    def sort_by_id(self):
        """Sort list according to phase ID."""
        self._dict = OrderedDict(sorted(self._dict.items()))

    def id_from_name(self, name):
        """Get phase ID from phase name.

        Parameters
        ----------
        name : str
            Phase name.
        """
        for phase_id, phase in self:
            if name == phase.name:
                return phase_id
        raise KeyError(f"'{name}' is not among the phase names {self.names}.")
