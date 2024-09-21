r"""
================================================
Rotations mapping the fundamental sector on *S2*
================================================

This example shows how to sample rotations :math:`\mathbf{R}` that when rotating the
vector :math:`\mathbf{v_z} = (0, 0, 1)`, the resulting vectors cover the fundamental
sector of a given Laue class.

We show this by comparing the vectors we get by: 

1. Sampling rotations for *4/mmm* and then rotating :math:`\mathbf{v_z}`
2. Sampling all of *S2* but only keeping those within the corresponding fundamental
   sector.

Apart from the first rotation, all rotations have a Euler angle
:math:`\phi = 0^{\circ}`.
These "reduced" rotations can be useful in template matching of spot patterns from the
transmission electron microscope.
"""

import matplotlib.pyplot as plt
import numpy as np

from orix import plot, sampling
from orix.quaternion import symmetry
from orix.vector import Vector3d

# Sample rotations with an average misorientation
res = 2
pg = symmetry.D4h  # 4/mmm

R = sampling.get_sample_reduced_fundamental(res, point_group=pg)
print(np.allclose(R.to_euler()[1:, 0], 0))

########################################################################
# Get vectors within the fundamental sector following the two routes
v1 = R * Vector3d.zvector()

v2 = sampling.sample_S2(res)
v2 = v2[v2 <= pg.fundamental_sector]

# Only equivalent for the same S2 sampling method
print(np.allclose(v1.data, v2.data))
print(v1)
print(v2)

########################################################################
# Plot the vectors in the fundamental sector of Laue group 4/mmm
fig, (ax0, ax1) = plt.subplots(
    ncols=2, subplot_kw={"projection": "ipf", "symmetry": pg}, layout="tight"
)
ax0.scatter(v1, s=5)
ax1.scatter(v2, c="C1", s=5)
ax0.set_title("Rotated Z vectors", loc="left")
ax1.set_title("Directly sampled", loc="left")
_ = fig.suptitle("Vectors in the fundamental sector of 4/mmm", y=0.8)
