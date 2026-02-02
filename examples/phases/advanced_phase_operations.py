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
=========================
Advanced Phase operations
=========================

This example shows some additional uncommon but useful operations related to
:class:`~orix.crystal_map.Phase`.

"""

import orix.crystal_map as ocm
import orix.quaternion as oqu
import orix.vector as ove
import diffpy.structure as dps

# %%
# Directly accessing `diffpy.structure` operators
# -----------------------------------------------
#
# ORIX is primarily concerned with calculations related to point group symmetries,
# (i.e., no concern for translations). However, other projects using ORIX might
# want need this information to, for example, calculate extinction coefficients
# for a diffraction experiment. For those cases, the rotation matrices and
# translation vectors can be extracted from `Phase.space_group`.

phase = ocm.Phase(space_group=13)

for element in phase.space_group.symop_list:
    print("\nrotation:\n", element.R, "\ntranslation:", element.t)
    
# %%
# these can also be returned as ORIX objects.

rots = oqu.Quaternion.from_matrix([x.R for x in phase.space_group.symop_list])
vecs = ove.Vector3d([x.t for x in phase.space_group.symop_list])


A Phase, at minimum, contains both a symmetry and a unit cell. For more on how
# :class:`~orix.crystal_map.Phase` and :class:`~orix.quaternion.symmetry.Symmetry`
# differ, refer to :doc:`/examples/phases/Phase_versus_Symmetry`, but as a general
# rule, any calculation related to crystallographic vectors requires defining a
# Phase as opposed to only a Symmetry.
#
# The most basic method to define a Phase is using a point group name plus
# the unit cell parameters.

austenite_phase = ocm.Phase(
    name="Austenite",
    point_group="m3m",
    # a, b, c, alpha, beta, and gamma in nm and degrees for austenite
    cell_parameters=[0.36, 0.36, 0.36, 90, 90, 90],
)
print(austenite_phase)
austenite_phase.plot_unit_cell()

# %%
# However, for users needing more control or wishing to explicitly define atomic
# positions, occupancy, and other information, `diffpy.structure` can be imported
# and used to define a Structure, which is then be used to define the Phase.

import diffpy.structure as dps

ferrite_structure = dps.Structure(
    title="ferrite",
    # a, b, c, alpha, beta, and gamma in nm and degrees for ferrite
    lattice=dps.Lattice(0.287, 0.287, 0.287, 90, 90, 90),
    atoms=[dps.Atom("Fe", [1e-5, 1e-5, 1e-5])],
)

ferrite_phase = ocm.Phase(
    space_group=229, structure=ferrite_structure, color="black"
).expand_asymmetric_unit()
print(ferrite_phase)
ferrite_phase.plot_unit_cell()

# %%
# The structure can also be excluded, in which case a symmetry-aware default lattice
# is used. This can cause issues with vector calculations for non-cubic systems,
# and should be avoided when possible.
phase_non = ocm.Phase(name="blank phase", point_group="622")
print(phase_non._diffpy_lattice)

# %%
# As seen in the ferrite and autenite phases above, Phaces can be defined from
# either a space group or a point group. Space groups are defined using their
# space group number (refer to the following for details: http://img.chem.ucl.ac.uk/sgp/large/sgp.htm),
# which will cause the derived point group to be added automatically to the Phase.
phase_i3m = ocm.Phase(space_group=165)
print(phase_i3m)

# %% Alternately phases can be defined using a point group, in which case the
# space group is left as `None`, since it cannot be uniquely defined.
phase_432 = ocm.Phase(name="unknown", point_group="432")
print(phase_432)

# %%
# Regardless of choice, the required point group is saved as an :class:`~orix.quaternion.symmetry.Symmetry`
# instance, whereas the optional space group is a :class:`diffpy.structure.Spacegroup.spacegroupmod.SpaceGroup`
# instance.

print(phase_i3m.point_group)
print(phase_i3m.space_group)

# %%
# Phases can also be imported directly from a Crystallographic Information File (CIF) file.
#
# E.g. one for titanium from an online repository like the Americam Mineralogist
# Crystal Structure Database:
# https://rruff.geo.arizona.edu/AMS/download.php?id=13417.cif&down=text
# phase_ti = Phase.from_cif("ti.cif")
# print(phase_ti)

# %%
# Creating a PhaseList
# --------------------
#
# Since CrystalMap objects can contain multiple phases, it's often convenient to
# store sets of Phases as a in iteratable PhaseList object. This can be done
# by defining individual phases, or by defining the phases during creation
# of the list.

phases_from_list = ocm.PhaseList([ferrite_phase, austenite_phase, phase_432])
print(phases_from_list)

new_phases = ocm.PhaseList(
    names=["Alpha", "Beta", "Gamma"],
    space_groups=[75, 229, 225],
    colors=["red", "orange", "yellow"],
)
print(new_phases)

# %%
# These phases can then be referenced either by their index or their phase name
print(phases_from_list["Austenite"])
print(phases_from_list[0])
print(phases_from_list[:2])
print(phases_from_list["unknown", "ferrite"])

# %%
# Modifying Phases and PhaseLists
# -------------------------------
#
# The following Phase attributes can all be modified after initialization:
#    - name
#    - space_group
#    - point_group
#    - structure
#    - color
#
# Note though that overwriting `point_group` when `space_group` has already been
# defined will erase the original space group.

ferrite_phase.point_group = "422"
print(ferrite_phase)

# %%
# For the opposite case, altering the space group will overwrite
# the point group.
phase_i3m.space_group = 168
print(phase_i3m)

# %%
# This can also be done with a PhaseList using indexing to choose the Phase to
# alter.
phases_from_list[1].space_group = 229
print(phases_from_list)

# %%
# Phases can be added to a PhaseList after creation, either from another Phaselist or
# from a standalone and/or new Phase, as well as deketed. There is also a
# convenience function for adding a `not indexed` phase, as this often becomes
# relevant in experimental crystal maps (EBSD, HEDM, etc.).
phases_from_list.add(new_phases[0])
phases_from_list.add(ocm.Phase("sigma", point_group="4/mmm"))
print(phases_from_list)
del phases_from_list["unknown"]
print(phases_from_list)
phases_from_list.add_not_indexed()
print(phases_from_list)


# %%
# Shallow Copying Phases
# ----------------------
#
# PhaseLists generated from lists of Phases are shallow copies, meaning changes
# a Phase object will affect any PhaseList object created from it. This can be
# seen above in the ferrite phase of `phases_from_list`, which originally had a
# space group of `Im-3m`, but ended with a space group of `None`, reflecting
# the change made to to `ferrite_phase`.
