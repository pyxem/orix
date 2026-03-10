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
=======================
Crystal reference frame
=======================

This exampe shows how the crystal and sample reference frames are aligned for a
:class:`~orix.crystal_map.Phase`.
"""

from diffpy.structure import Lattice, Structure
import matplotlib.pyplot as plt
import numpy as np

from orix.crystal_map import Phase
from orix.quaternion import Rotation
from orix.vector import Miller

plt.rcParams.update(
    {
        "figure.figsize": (10, 5),
        "font.size": 20,
        "axes.grid": True,
        "lines.markersize": 10,
        "lines.linewidth": 3,
    }
)

########################################################################################
# Alignment and the structure matrix
# ----------------------------------
#
# The direct Bravais lattice is characterized by basis vectors :math:`(\mathbf{a}, \mathbf{b},
# \mathbf{c})` with unit cell edge lengths :math:(`a, b, c)` and angles :math:`(\alpha,
# \beta, \gamma)`.
# The reciprocal lattice has basis vectors given by
#
# .. math::
#   \mathbf{a^*} = \frac{\mathbf{b} \times \mathbf{c}}{\mathbf{a} \cdot (\mathbf{b} \times \mathbf{c})},\:\:
#   \mathbf{b^*} = \frac{\mathbf{c} \times \mathbf{a}}{\mathbf{a} \cdot (\mathbf{b} \times \mathbf{c})},\:\:
#   \mathbf{c^*} = \frac{\mathbf{a} \times \mathbf{b}}{\mathbf{a} \cdot (\mathbf{b} \times \mathbf{c})},
#
# with reciprocal lattice parameters :math:`(a^*, b^*, c^*)` and angles
# :math:`(\alpha^*, \beta^*, \gamma^*)`.
#
# Using these two crystallographic lattices, we can define a standard Cartesian
# (orthonormal) reference frame by the unit vectors :math:`(\mathbf{e_1}, \mathbf{e_2},
# \mathbf{e_3})`.
# In principle, the direct lattice reference frame can be oriented arbitrarily in the
# Cartesian reference frame. In orix we have chosen
#
# .. math::
#   \mathbf{e_1} ||\: \frac{\mathbf{a}}{a},\:\:
#   \mathbf{e_2} ||\: \mathbf{e_3} \times \mathbf{e_1},\:\:
#   \mathbf{e_3} ||\: \frac{\mathbf{c^*}}{c^*}.
#
# This alignment is used for example in :cite:`rowenhorst2015consistent` and
# :cite:`degraef2003introduction`, the latter which was the basis for the *EMsoft*
# Fortran suite of programs.
# Another common option is :math:`\mathbf{e_1} || \mathbf{a^*}/a^*, \mathbf{e_2} ||
# \mathbf{e_3} \times \mathbf{e_1}, \mathbf{e_3} || \mathbf{c}/c`, which is used for
# example in :cite:`britton2016tutorial` and the :mod:`diffpy.structure` Python package,
# which we'll come back to.
#
# In calculations, it is useful to describe the transformation of the Cartesian unit
# *row* vectors to the coordinates of the direct lattice vectors by the structure *row*
# matrix :math:`\mathbf{A}` (also called the crystal *base*).
# Given the chosen alignment of basis vectors with the Cartesian reference frame,
# :math:`\mathbf{A}` is defined as
#
# .. math::
#   \begin{equation}
#   \mathbf{A} =
#   \begin{pmatrix}
#   a & 0 & 0\\
#   b\cos\gamma & b\sin\gamma & 0\\
#   c\cos\beta & -c\frac{(\cos\beta\cos\gamma - \cos\alpha)}{\sin\gamma} & \frac{\mathrm{V}}{ab\sin\gamma}
#   \end{pmatrix},
#   \end{equation}
#
# where :math:`V` is the volume of the unit cell.
#
# In orix, we use the :class:`~diffpy.structure.lattice.Lattice` class to keep track of
# these properties internally.
#
# Let's create an hexagonal crystal with lattice parameters :math:`(a, b, c)` = (1.4,
# 1.4, 1.7) nm and :math:`(\alpha, \beta, \gamma) = (90^{\circ}, 90^{\circ},
# 120^{\circ})`
lat = Lattice(1.4, 1.4, 1.7, 90, 90, 120)
print(lat)

########################################################################################
# diffpy.structure stores the structure matrix in the
# :attr:`~diffpy.structure.lattice.Lattice.base` property
print(lat.base)

########################################################################################
# We see that diffpy.structure does not use the orix alignment mentioned above, since
# :math:`\mathbf{e1} \nparallel \mathbf{a} / a`.
# Instead, we see that :math:`\mathbf{e3} \parallel \mathbf{c} / c`, which is in line
# with the alternative alignment mentioned above.
#
# Thus, the alignment is updated internally whenever a
# :class:`~orix.crystal_map.Phase` is created, a class which brings together this
# crystal lattice and a point group :class:`~orix.quaternion.Symmetry` *S*
structure_hex = Structure(lattice=lat)
phase_hex = Phase(point_group="6/mmm", structure=structure_hex)
print(phase_hex.structure.lattice.base)

########################################################################################
# .. caution::
#
#   Using another alignment than the one described above has undefined behaviour in
#   orix.
#   Therefore, it is important that the structure matrix of a phase is not changed.
#
# .. note::
#
#   The lattice is included in a :class:`~diffpy.structure.structure.Structure` because
#   the latter class brings together a lattice and :class:`~diffpy.structure.atom.Atom`
#   s, which is useful when simulating diffraction.
#
# We can visualize the alignment of the direct and reciprocal lattice basis vectors with
# the Cartesian reference frame using the stereographic projection
t_hex = Miller(uvw=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], phase=phase_hex)
g_hex = Miller(hkl=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], phase=phase_hex)

fig = t_hex.scatter(
    c=["r", "g", "b"],
    marker="o",
    return_figure=True,
    axes_labels=["e1", "e2"],
    hemisphere="both",
)
g_hex.scatter(c=["r", "g", "b"], marker="x", s=300, figure=fig)

########################################################################################
# Alignment of symmetry operations
# --------------------------------
#
# To see which crystallographic axes the point group symmetry operations rotate about,
# we can add symmetry operations to the figure and show it again
R = Rotation.from_axes_angles([0, 1, 0], -65, degrees=True)
phase_hex.point_group.plot(figure=fig, orientation=R)

fig

########################################################################################
# Converting crystal vectors
# --------------------------
#
# The reference frame of the stereographic projection above is the Cartesian reference
# frame :math:`(\mathbf{e_1}, \mathbf{e_2}, \mathbf{e_3})`.
# The Cartesian coordinates :math:`(\mathbf{x}, \mathbf{y}, \mathbf{z})` of
# :math:`(\mathbf{a}, \mathbf{b}, \mathbf{c})` and :math:`(\mathbf{a^*}, \mathbf{b^*},
# \mathbf{c^*})` were found using :math:`\mathbf{A}` in the following conversions
#
# .. math::
#   \begin{align}
#   (x, y, z) &= [uvw] \cdot \mathbf{A},\\
#   (x, y, z) &= (hkl) \cdot (\mathbf{A}^{-1})^T.
#   \end{align}
#
# Let's compute the internal conversions directly and check for equality
A = phase_hex.structure.lattice.base
v1 = np.dot(t_hex.uvw, A)
print(np.allclose(t_hex.data, v1))

v2 = np.dot(g_hex.hkl, np.linalg.inv(A).T)
print(np.allclose(g_hex.data, v2.data))

plt.show()
