r"""
========================
Plot symmetry operations
========================

This example shows how to use the `from_path_ends` functions from
:class:`~orix.vector.Vector3d`, :class:`~orix.quaternions.Rotation`, and
:class:`~orix.quaternions.Orientation` to draw paths through thier
respective non-Euclidean spaces.
"""

import numpy as np
import matplotlib.pyplot as plt
from orix.quaternion import Orientation, Rotation
from orix.quaternion.symmetry import Oh, D3
from orix.vector import Vector3d


fig = plt.figure()

# plot a path in homochoric space with no symmetry
rot_path = Rotation(
    data=np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [1, 0, -1, 0],
            [1, 0, 0, -1],
            [1, 0, 0, -1],
        ]
    )
)
rotation_path = Rotation.from_path_ends(rot_path, closed=True)
# cast the rotation to a symmetry-less orientation for plotting purposes
Orientation(rotation_path).scatter(
    figure=fig, position=[2, 2, 1], marker=">", c=np.arange(700)
)

# plot a path in rodrigues space with m-3m (cubic) symmetry.
m3m_path = Orientation(
    data=np.array(
        [
            [1, 0, 0, 0],
            [2, 1, 0, 0],
            [3, 0, 1, 0],
            [4, 0, 0, 1],
            [5, 0, -1, 0],
            [6, 0, 0, -1],
            [7, 0, 0, -1],
            [8, 1, 0, 0],
            [9, 0, 1, 0],
            [10, 0, 0, 1],
            [11, 0, -1, 0],
            [12, 0, 0, -1],
            [13, 0, 0, -1],
        ]
    ),
    symmetry=Oh,
)
orientation_path = Orientation.from_path_ends(m3m_path.reduce(), closed=True).reduce()
orientation_path.scatter(figure=fig, position=[2, 2, 2], marker=">", c=np.arange(1300))

# plot a second path in rodrigues space with symmetry, but while also crossing a
# symmetry boundary
fiber_start = Rotation.identity(1)
fiber_middle = Rotation.from_axes_angles([1, 2, 3], np.pi)
fiber_end = Rotation.from_axes_angles([1, 2, 3], 2 * np.pi)
fiber_points = Orientation.stack([fiber_start, fiber_middle, fiber_end])
fiber_points.symmetry = Oh
fiber_path = Orientation.from_path_ends(fiber_points, closed=False).reduce()
fiber_path.scatter(figure=fig, position=[2, 2, 3], marker=">", c=np.arange(200))


# plot vectors
ax4 = plt.subplot(2, 2, 4, projection="stereographic")
vector_points = Vector3d(np.array([[-1,0,0],[0,1,0.1],[1,0,0.2],[0,-1,0.3],[-1,0,0.4]]))

vector_path = Vector3d.from_path_ends(vector_points,steps = 200)
ax4.scatter(vector_path, figure=fig, marker=">", c=np.arange(vector_path.size))
