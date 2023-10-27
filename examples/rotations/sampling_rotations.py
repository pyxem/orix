"""
==================
Sampling rotations
==================

This example shows how to sample some phase object in Orix. We will
get both the zone axis and the reduced fundamental zone rotations for
the phase of interest.
"""
from diffpy.structure import Atom, Lattice, Structure
from orix.crystal_map import Phase
from orix.sampling import (
    get_sample_reduced_fundamental,
    get_sample_zone_axis,
)
from orix.vector import Vector3d

a = 5.431
latt = Lattice(a, a, a, 90, 90, 90)
atom_list = []
for coords in [[0, 0, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0]]:
    x, y, z = coords[0], coords[1], coords[2]
    atom_list.append(Atom(atype="Si", xyz=[x, y, z], lattice=latt))  # Motif part A
    atom_list.append(
        Atom(atype="Si", xyz=[x + 0.25, y + 0.25, z + 0.25], lattice=latt)
    )  # Motif part B
struct = Structure(atoms=atom_list, lattice=latt)
p = Phase(structure=struct, space_group=227)
reduced_fun = get_sample_reduced_fundamental(resolution=4, point_group=p.point_group)

vect_rot = (
    reduced_fun * Vector3d.zvector()
)  # get the vector representation of the rotations
vect_rot.scatter(grid=True)  # plot the stereographic projection of the rotations

# %%

zone_axis_rot, directions = get_sample_zone_axis(
    phase=p, density="7", return_directions=True
)  # get the zone axis rotations
zone_vect_rot = (
    zone_axis_rot * Vector3d.zvector()
)  # get the vector representation of the rotations
zone_vect_rot.scatter(
    grid=True, vector_labels=directions, text_kwargs={"size": 8, "rotation": 0}
)  # plot the stereographic projection of the rotations

# %%
