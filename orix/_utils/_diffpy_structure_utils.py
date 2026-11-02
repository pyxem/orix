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

"""Utilities for interfacing with diffpy.structure."""

from typing import Callable

from diffpy.structure import __version__
from diffpy.structure.lattice import Lattice
from diffpy.structure.parsers import p_cif
from diffpy.structure.spacegroups import SpaceGroup
from diffpy.structure.structure import Structure
from packaging.version import Version

DIFFPY_STRUCTURE_VERSION = Version(__version__)

# TODO: Remove these checks (and most likely this whole file) once
# 3.4.0 is the minimal supported version

if DIFFPY_STRUCTURE_VERSION >= Version("3.4.0"):
    from diffpy.structure import load_structure
    from diffpy.structure.spacegroups import get_space_group
else:
    from diffpy.structure import loadStructure as load_structure
    from diffpy.structure.spacegroups import GetSpaceGroup as get_space_group


def place_in_lattice(structure: Structure, lattice: Lattice) -> Structure:
    if DIFFPY_STRUCTURE_VERSION >= Version("3.4.0"):
        return structure.place_in_lattice(lattice)
    else:
        return structure.placeInLattice(lattice)


def get_cell_parms(lattice: Lattice) -> tuple[float, float, float, float, float, float]:
    if DIFFPY_STRUCTURE_VERSION >= Version("3.4.0"):
        return lattice.cell_parms()
    else:
        return lattice.abcABG()


def get_parser_and_structure_from_cif_file(fname: str) -> tuple[p_cif.P_cif, Structure]:
    parser = p_cif.P_cif()
    if DIFFPY_STRUCTURE_VERSION >= Version("3.4.0"):
        structure = parser.parse_file(fname)
    else:
        structure = parser.parseFile(fname)
    return parser, structure


get_space_group: Callable[[str | int], SpaceGroup]
# Simplified signature for our current use case
load_structure: Callable[[str], Structure]

__all__ = [
    "get_space_group",
    "get_parser_and_structure_from_cif_file",
    "load_structure",
    "place_in_lattice",
]
