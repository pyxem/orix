# -*- coding: utf-8 -*-
# Copyright 2018-2021 the orix developers
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with orix.  If not, see <http://www.gnu.org/licenses/>.

"""Uniform cubochoric sampling of *SO(3)* :cite:`singh2016orientation`.

This module and documentation is only relevant for orix developers, not
for users.

.. warning:
    This module is for internal use only.  Do not use it in your own
    code. We may change the API at any time with no warning.
"""

import numba as nb
import numpy as np

from orix.quaternion import Rotation
from orix.quaternion._conversions import (
    ax2qu_single,
    cu2ro_single,
    ro2ax_single,
)


def cubochoric_sampling(semi_edge_steps=None, resolution=None):
    r"""Uniform cubochoric sampling of *SO(3)*
    :cite:`singh2016orientation`.

    Parameters
    ----------
    semi_edge_steps : int, optional
        Number of grid points *N* along the semi-edge of the cubochoric
        cube. If not given, it will be calculated from `resolution`
        following Eq. (9) in :cite:`singh2016orientation`. For example,
        if an average disorientation of 1 degree is needed, this should
        be set to 137. This will result in $(2N + 1)^3$ = 20 796 875
        unique orientations.
    resolution : float, optional
        Average disorientation between resulting orientations. This must
        be given if `semi_edge_steps` is not.

    Returns
    -------
    ~orix.quaternion.Rotation
        Sampled rotations in *SO(3)*.

    Notes
    -----
    The cubochoric grid sampled is $S_{000}(N)$, which contains the
    identity rotation.
    """
    if semi_edge_steps is None:
        if resolution is None:
            raise ValueError("Either `semi_edge_steps` or `resolution` must be passed")
        else:
            semi_edge_steps = resolution_to_semi_edge_steps(resolution)
    quaternions = _cubochoric_sampling_loop(semi_edge_steps)
    return Rotation(quaternions)


@nb.jit(cache=True, nogil=True, nopython=True)
def resolution_to_semi_edge_steps(resolution):
    """Calculate the number of grid points *N* along the semi-edge of
    the cubochoric cube given an average disorientation between
    orientations :cite:`singh2016orientation`.

    Parameters
    ----------
    resolution : float

    Returns
    -------
    int
    """
    return int(np.round(131.97049 / (resolution - 0.03732)))


@nb.jit("float64[:, :](int64)", cache=True, nogil=True, nopython=True)
def _cubochoric_sampling_loop(semi_edge_steps):
    """See :func:`cubochoric_sampling`.

    If `semi_edge_steps` is 100, there will be (201, 201, 201) points
    sampled.
    """
    semi_edge_length = 0.5 * np.pi ** (2 / 3)
    step_size = semi_edge_length / semi_edge_steps
    n_points = (2 * semi_edge_steps + 1) ** 3
    rot = np.zeros((n_points, 4))

    xyz = np.zeros(3)
    step = 0
    for i in nb.prange(-semi_edge_steps + 1, semi_edge_steps + 1):
        xyz[0] = i * step_size
        for j in range(-semi_edge_steps + 1, semi_edge_steps + 1):
            xyz[1] = j * step_size
            for k in range(-semi_edge_steps + 1, semi_edge_steps + 1):
                xyz[2] = k * step_size

                # Discard the point and move to the next iteration if it
                # lies outside the cubochoric cube
                if np.max(np.abs(xyz)) > semi_edge_length:  # pragma: no cover
                    continue

                # Get quaternion via cubochoric coordinates -> Rodrigues
                # vector -> axis-angle pair
                rodrigues = cu2ro_single(xyz)
                axis_angle = ro2ax_single(rodrigues)
                rot[step] = ax2qu_single(axis_angle)

                step += 1

    return rot[:step]
