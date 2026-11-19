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
=====================================
Uniform sampling of orientation space
=====================================

This example compares cubochoric, Haar Euler, and quaternion sampling of
orientation space :math:`SO(3)` for the cubic proper point group
:math:`432`.

Use :func:`~orix.sampling.get_sample_fundamental` to sample the
fundamental zone of a proper point group. Pass the point group directly
or specify a space group. For example, space group 225
(:math:`Fm\bar{3}m`) gives the proper point group :math:`432`.
To sample a neighborhood around a rotation, use
:func:`~orix.sampling.get_sample_local`. In both functions,
``resolution`` is the desired characteristic spacing between rotations
in degrees.

.. list-table:: Proper crystallographic point groups
   :header-rows: 1

   * - Hermann–Mauguin
     - Schoenflies
   * - :math:`1`
     - :math:`C_1`
   * - :math:`2`
     - :math:`C_2`
   * - :math:`222`
     - :math:`D_2`
   * - :math:`4`
     - :math:`C_4`
   * - :math:`422`
     - :math:`D_4`
   * - :math:`3`
     - :math:`C_3`
   * - :math:`32`
     - :math:`D_3`
   * - :math:`6`
     - :math:`C_6`
   * - :math:`622`
     - :math:`D_6`
   * - :math:`23`
     - :math:`T`
   * - :math:`432`
     - :math:`O`
"""

import matplotlib.pyplot as plt
import numpy as np

import orix.quaternion as oqu
import orix.quaternion.symmetry as osy
import orix.sampling as osa
import orix.vector as ove

# %%
# Sample the cubic fundamental zone
# ---------------------------------
#
# Cubochoric sampling projects a uniform grid in a cube onto
# :math:`SO(3)`, including the identity rotation
# :cite:`singh2016orientation,rosca2014anew`. The number of steps
# :math:`N` along each semi-edge is set by ``semi_edge_steps`` or
# estimated from the resolution :math:`\langle\theta\rangle` in degrees.
#
# .. math::
#
#    N = \mathrm{round}\left(\frac{131.97049}{\langle\theta\rangle - 0.03732}\right).
#
# A resolution of 3 degrees gives 45 semi-edge steps.
#
# Haar Euler sampling uses grids uniform in :math:`\alpha`,
# :math:`\cos\beta`, and :math:`\gamma`. Quaternion sampling
# constructs unit quaternions from three uniform samples
# :cite:`lavalle2006planning`.
#
# Sample each method at 3 degrees. Draw 10,000 rotations from each grid
# without replacement so the plots use equal numbers of points.

point_group = osy.O
methods = ("cubochoric", "haar_euler", "quaternion")
colors = ("C0", "C1", "C2")
rng = np.random.default_rng(42)
samples = {}

for method in methods:
    rotations = osa.get_sample_fundamental(3, point_group=point_group, method=method)
    orientations = oqu.Orientation(rotations, symmetry=point_group)
    print(f"{method}: {orientations.size:,} orientations")
    samples[method] = orientations[rng.choice(orientations.size, 10_000, replace=False)]

# %%
# Compare the samples in Rodrigues space
# --------------------------------------
#
# The top and side views of the Rodrigues fundamental zone give a
# qualitative comparison of the sampled orientations.

fig = plt.figure(figsize=(11, 7))
wireframe_kwargs = dict(color="k", linewidth=1, alpha=0.2)

for column, (method, color) in enumerate(zip(methods, colors), start=1):
    for row, elevation in enumerate((90, 0)):
        samples[method].scatter(
            projection="rodrigues",
            figure=fig,
            position=(2, 3, row * 3 + column),
            wireframe_kwargs=wireframe_kwargs,
            c=color,
            s=4,
        )
        ax = fig.axes[-1]
        ax.view_init(elev=elevation, azim=0)
        if row == 0:
            ax.set_title(method)

fig.set_layout_engine("constrained")

# %%
# Compare the samples in inverse pole figures
# -------------------------------------------
#
# The inverse pole figures show the crystal directions corresponding
# to the sample X, Y, and Z axes.

directions = ove.Vector3d(np.eye(3))

for method, color in zip(methods, colors):
    fig = samples[method].scatter(
        projection="ipf",
        direction=directions,
        return_figure=True,
        figure_kwargs={"figsize": (8, 3), "layout": "constrained"},
        c=color,
        s=4,
    )
    fig.suptitle(method)
