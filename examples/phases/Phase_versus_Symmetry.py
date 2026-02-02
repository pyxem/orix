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

r"""
=====================
Phase versus Symmetry
=====================
"""

import diffpy.structure as dps

import numpy as np

import orix.crystal_map as ocm
import orix.quaternion as oqu
import orix.vector as ove

from orix.quaternion import symmetry

##############################################################################
# ORIX includes two different but related classes for describing crystallographic
# information, :class:`~orix.quaternion.symmetry.Symmetry`, and
# :class:`~orix.crystal_map.Phase`.
#
# A Symmetry object contains ONLY information on symmetrically equivalent transforms,
# and in most cases is a Laue and/or Point group. For example, the Symmetry of
# Alumina would be defined as:

Al2O3_sym = symmetry.D3d  # <-- Schoenflies notation for point group '-3m'
Al2O3_sym.plot()
print(Al2O3_sym)

##############################################################################
# On the other hand, a Phase object contains at minimum both the symmetry and
# the unit cell. Again, using Alumina as the example:

atoms = [
    dps.Atom("Al",[1/3,2/3,0.815]),
         dps.Atom("O",[0.361,1/3,0.583]),
         ]
lattice = dps.Lattice(0.481, 0.481, 1.391, 90, 90, 120)
structure = dps.Structure(atoms=atoms, lattice=lattice)
Al2O3_phase = ocm.Phase(name = "Alumina",
    space_group = 167,
    structure = structure,
    color = 'red',
).expand_asymmetric_unit()

unit_cell_figure = Al2O3_phase.plot_unit_cell(return_figure=True)
unit_cell_figure.suptitle(r"$Al_2O_3$ unit cell")
Al2O3_phase

##############################################################################
# Quaternion based transforms (orientation transforms, misorientation calculations,
# etc.) only require a Symmetry, whereas any calculation involving diffraction,
# distance calculations, Inverse Pole Figures, and/or Miller indicies require
# a Phase.
#
# Additionally, while it IS possible to define a phase without explicitly giving
# the cell parameters, this will cause ORIX to fill in default values for cell
# parameters based on the point group. This is simpler, has no effect on IPF coloring
# or orientation calculations, and allows for tracking information such as names and
# preferred plot color(hence why it is allowed), but it WILL cause incorrect Miller 
# calculations and IPF plotting.

lazy_Al2O3_phase = ocm.Phase(space_group=167)
correct_111 = ove.Miller(uvw=[1,1,1],phase=Al2O3_phase).xyz
incorrect_111 = ove.Miller(uvw=[1,1,1],phase=lazy_Al2O3_phase).xyz
print("Correct xyz for [111]:", np.stack(correct_111).flatten())
print("Incorrect xyz for [111]:", np.stack(incorrect_111).flatten())

