r"""
========================
Plot symmetry operations
========================

This example shows how stereographic projections with symmetry operators can be
automatically generated for the 32 crystallographic point groups.

The ordering follows the one given in section 9.2 of "Structures of Materials"
(DeGraef et.al, 2nd edition, 2012), starting with the cyclic groups, then the
dihedral groups, then those same groups plus inversion centers, then the successive
application of mirror planes and secondary rotational symmetries until all 32
groups are made.

The plots themselves as well as their labels follow the standards given in
Table 10.2.2 of the "International Tables of Crystallography, Volume A" (ITOC).
Both the nomenclature and marker styles thus differ slightly from some textbooks, as
there are some arbitrary convention choices in both Schoenflies notation and marker
styles.

Orix uses Schoenflies Notation (left label above each plot) for variable names since
they are short and always begin with a letter, but both Schoenflies and
Hermann-Mauguin (right label above each plot) names can be used to look up symmetry
groups using `PointGroups.get()`
"""

import matplotlib.pyplot as plt
import orix.plot
from orix.quaternion.symmetry import PointGroups
from orix.vector import Vector3d

# create a list of the 32 crystallographic point groups
point_groups = PointGroups.get_set("procedural")

# prepare the plots
fig, ax = plt.subplots(
    4, 8, subplot_kw={"projection": "stereographic"}, figsize=[14, 10]
)
ax = ax.flatten()

# create a vector to mirror over axes
v = Vector3d.from_polar(65, 80, degrees=True)
# Iterate through the 32 Point groups
for i, pg in enumerate(point_groups):
    pg.plot(asymetric_vector=v, plt_axis=ax[i], itoc_style=True)
