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
=====================================================
Symmetrically equivalent directions and plane normals
=====================================================

The symmetry operations :math:`s` of the point group symmetry assigned to a crystal
lattice can be applied to describe symmetrically equivalent crystal directions and
plane normals.
"""

# %%
import numpy as np

from orix.crystal_map import Phase
from orix.plot import format_labels, register_projections
from orix.vector import Miller

register_projections()

# %%
# All crystal systems are supported (triclinic, monoclinic, orthorhombic, trigonal,
# tetragonal, hexagonal, and cubic), but we'll use the cubic crystal as an example
# because of its high symmetry.

cubic = Phase(point_group="m-3m")
print(cubic, "\n", cubic.structure.lattice.abcABG())

# %%
# The directions :math:`\mathbf{t}` parallel to the crystal axes :math:`(\mathbf{a},
# \mathbf{b}, \mathbf{c})` given by :math:`[100]`, :math:`[\bar{1}00]`, :math:`[010]`,
# :math:`[0\bar{1}0]`, :math:`[001]`, and :math:`[00\bar{1}]` (:math:`\bar{1}` means
# "-1") are symmetrically equivalent, and can be obtained using
# :meth:`~orix.vector.Miller.symmetrise`.

t100 = Miller(uvw=[1, 0, 0], phase=cubic)
t100.symmetrise(unique=True)

# %%
# Without passing ``unique=True``, since the cubic crystal symmetry is described by 48
# symmetry operations :math:`s` (or elements), 48 directions :math:`\mathbf{t}` would
# have been returned.

# The six symmetrically equivalent directions, known as a family, may be expressed
# collectively as :math:`\left<100\right>`, the brackets implying all six permutations
# or variants of 1, 0, 0.
# This particular family is said to have a multiplicity of 6

t100.multiplicity

# %%
t6 = Miller(uvw=[[1, 0, 0], [1, 1, 0], [1, 1, 1]], phase=cubic)
t6

# %%
t6.multiplicity

# %%
# Let's plot the symmetrically equivalent directions from the direction families
# :math:`\left<100\right>`, :math:`\left<110\right>`, and :math:`\left<111\right>`
# impinging on the upper hemisphere.
# By also returning the indices of which family each symmetrically equivalent direction
# belongs to from :meth:`~orix.vector.Miller.symmetrise`, we can give a unique color per
# family.

t7, idx = t6.symmetrise(unique=True, return_index=True)
labels = format_labels(t7.uvw, ("[", "]"))

# Get an array with one color per family of vectors
colors = np.array([f"C{i}" for i in range(t6.size)])[idx]

t7.scatter(c=colors, vector_labels=labels, text_kwargs={"offset": (0, 0.02)})

# %%
# Similarly, symmetrically equivalent planes :math:`\mathbf{g} = (hkl)` can be
# collectively expressed as planes of the form :math:`\{hkl\}`.

g5 = Miller(hkl=[[1, 0, 0], [1, 1, 0], [1, 1, 1]], phase=cubic)
g5.multiplicity

# %%
g6, idx = g5.symmetrise(unique=True, return_index=True)

labels = format_labels(g6.hkl, ("(", ")"))
colors = np.array([f"C{i}" for i in range(g5.size)])[idx]

g6.scatter(c=colors, vector_labels=labels, text_kwargs={"offset": (0, 0.02)})

# %%
# We computed the angles between directions and plane normals earlier in this tutorial.
# In general, :meth:`~orix.vector.Miller.angle_with` does not consider symmetrically
# equivalent directions, unless ``use_symmetry=True`` is passed.
# Consider :math:`(100)` and :math:`(\bar{1}00)` and :math:`(111)` and
# :math:`(\bar{1}11)` in the stereographic plot above.

g7 = Miller(hkl=[[1, 0, 0], [1, 1, 1]], phase=cubic)
g8 = Miller(hkl=[[-1, 0, 0], [-1, 1, 1]], phase=cubic)

# %%
g7.angle_with(g8, degrees=True)

# %%
g7.angle_with(g8, use_symmetry=True, degrees=True)

# %%
# Thus, passing ``use_symmetry=True`` ensures that the smallest angles between
# :math:`\mathbf{g}_1` and the symmetrically equivalent directions to
# :math:`\mathbf{g}_2` are found.
