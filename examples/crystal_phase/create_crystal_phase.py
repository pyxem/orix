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
r"""
====================
Create crystal phase
====================

This example shows various ways to create a crystal :class:`~orix.crystal_map.Phase`.

For alignment of the crystal axes with a Cartesian coordinate system, see the example on
:doc:`/examples/crystal_phase/crystal_reference_frame`.
"""

from diffpy.structure import Atom, Lattice, Structure

from orix.crystal_map import Phase

########################################################################################
# From a Crystallographic Information File (CIF) file.
#
# E.g. one for titanium from an online repository like the Americam Mineralogist
# Crystal Structure Database:
# https://rruff.geo.arizona.edu/AMS/download.php?id=13417.cif&down=text
# phase_ti = Phase.from_cif("ti.cif")
# print(phase_ti)

########################################################################################
# From a space group (note that the point group is derived)
phase_m3m = Phase(space_group=225)
print(phase_m3m)

########################################################################################
# From a point group (note that the space group is unknown since there are multiple
# options)
phase_432 = Phase(point_group="432")
print(phase_432)

########################################################################################
# Non-crystalline phase
phase_non = Phase()
print(phase_non)

########################################################################################
# Hexagonal alpha-titanium with a lattice and atoms
structure_ti = Structure(
    lattice=Lattice(4.5674, 4.5674, 2.8262, 90, 90, 120),
    atoms=[Atom("Ti", [0, 0, 0]), Atom("Ti", [1 / 3, 2 / 3, 1 / 2])],
)
print(structure_ti)

########################################################################################
phase_ti = Phase(space_group=191, structure=structure_ti)
print(phase_ti)
print(phase_ti.structure)
