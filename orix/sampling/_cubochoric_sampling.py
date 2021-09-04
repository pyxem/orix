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

"""Cubochoric sampling of the Rodrigues Fundamental Zone of a point
group.
"""

import numba as nb
import numpy as np

from orix.quaternion import Quaternion


def cubochoric_sampling(n_cube_steps=None, resolution=None):
    if n_cube_steps is None:
        if resolution is not None:
            n_cube_steps = resolution_to_n_cube_steps(resolution)
        else:
            raise ValueError("Either `n_cube_steps` or `resolution` must be passed")
    quaternions = _cubochoric_sampling_loop(n_cube_steps)
    return Quaternion(quaternions)


@nb.jit(cache=True, nogil=True, nopython=True)
def resolution_to_n_cube_steps(resolution):
    return np.ceil(131.97049 / (resolution - 0.03732))


@nb.jit("int64(float64[:])", cache=True, nogil=True, nopython=True)
def get_pyramid_single(v):
    x, y, z = v
    x_abs, y_abs, z_abs = abs(x), abs(y), abs(z)
    if (x_abs <= z) and (y_abs <= z):
        return 1
    elif (x_abs <= -z) and (y_abs <= -z):
        return 2
    elif (z_abs <= x) and (y_abs <= x):
        return 3
    elif (z_abs <= -x) and (y_abs <= -x):
        return 4
    elif (x_abs <= y) and (z_abs <= y):
        return 5
    else:  # (x_abs <= -y) and (z_abs <= -y)
        return 6


@nb.jit("float64[:](float64[:])", cache=True, nogil=True, nopython=True)
def cu2ho_single(cu):
    if np.max(np.abs(cu)) > (np.pi ** (2 / 3) / 2) + 1e-8:
        return np.zeros(3)

    # Determine which pyramid pair the point lies in and copy
    # coordinates in correct order
    pyramid = get_pyramid_single(cu)
    if pyramid in [1, 2]:
        pass
    elif pyramid in [3, 4]:
        cu = np.roll(cu, -1)
    else:  # [5, 6]
        cu = np.roll(cu, 1)

    # Scale by the grid parameter ratio
    cu = cu * np.pi ** (1 / 6) / 6 ** (1 / 6)

    cu_abs = np.abs(cu)
    if np.max(cu_abs) == 0:
        ho = np.zeros(3)
    else:
        if np.max(cu_abs[:2]) == 0:
            ho = np.array([0, 0, np.sqrt(6 / np.pi) * cu[2]])
        else:
            x, y, z = cu
            prefactor = (
                (3 * np.pi / 4) ** (1 / 3)
                * 2 ** (1 / 4)
                / (np.pi ** (5 / 6) / 6 ** (1 / 6) / 2)
            )
            sqrt2 = np.sqrt(2)
            if np.abs(y) <= np.abs(x):
                q = (np.pi / 12) * y / x
                cosq = np.cos(q)
                sinq = np.sin(q)
                q = prefactor * x / np.sqrt(sqrt2 - cosq)
                t1 = (sqrt2 * cosq - 1) * q
                t2 = sqrt2 * sinq * q
            else:
                q = (np.pi / 12) * x / y
                cosq = np.cos(q)
                sinq = np.sin(q)
                q = prefactor * y / np.sqrt(sqrt2 - cosq)
                t1 = sqrt2 * sinq * q
                t2 = (sqrt2 * cosq - 1) * q
            c = t1 ** 2 + t2 ** 2
            s = np.pi * c / (24 * z ** 2)
            c = np.sqrt(np.pi) * c / np.sqrt(24) / z
            q = np.sqrt(1 - s)

            ho = np.array([t1 * q, t2 * q, np.sqrt(6 / np.pi) * z - c])

    if pyramid in [1, 2]:
        return ho
    elif pyramid in [3, 4]:
        return np.roll(ho, 1)
    else:  # pyramid in [5, 6]
        return np.roll(ho, -1)


@nb.jit("float64[:, :](float64[:, :])", cache=True, nogil=True, nopython=True)
def cu2ho(cu):
    ho = np.zeros_like(cu)
    for i in nb.prange(cu.shape[0]):
        ho[i] = cu2ho_single(cu[i])
    return ho


@nb.jit("float64[:](float64[:])", cache=True, nogil=True, nopython=True)
def ho2ax_single(ho):
    # Constants stolen directly from EMsoft
    # fmt: off
    tfit = np.array([
         0.9999999999999968,     -0.49999999999986866,     -0.025000000000632055,
        -0.003928571496460683,   -0.0008164666077062752,   -0.00019411896443261646,
        -0.00004985822229871769, -0.000014164962366386031, -1.9000248160936107e-6,
        -5.72184549898506e-6,     7.772149920658778e-6,    -0.00001053483452909705,
         9.528014229335313e-6,   -5.660288876265125e-6,     1.2844901692764126e-6,
         1.1255185726258763e-6,  -1.3834391419956455e-6,   7.513691751164847e-7,
        -2.401996891720091e-7,    4.386887017466388e-8,   -3.5917775353564864e-9
    ])
    # fmt: on
    ho_magnitude = np.sum(ho ** 2)
    if (ho_magnitude > -1e-8) and (ho_magnitude < 1e-8):
        ax = np.array([0, 0, 1, 0], dtype=np.float64)
    else:
        # Convert the magnitude to the rotation angle
        hom = ho_magnitude
        s = tfit[0] + tfit[1] * hom
        for i in nb.prange(2, 21):
            hom = hom * ho_magnitude
            s = s + tfit[i] * hom
        hon = ho / np.sqrt(ho_magnitude)
        s = 2 * np.arccos(s)
        if np.abs(s - np.pi) < 1e-6:
            ax = np.append(hon, np.pi)
        else:
            ax = np.append(hon, s)
    return ax


@nb.jit("float64[:, :](float64[:, :])", cache=True, nogil=True, nopython=True)
def ho2ax(ho):
    n_vectors = ho.shape[0]
    ax = np.zeros((n_vectors, 4), dtype=np.float64)
    for i in nb.prange(n_vectors):
        ax[i] = ho2ax_single(ho[i])
    return ax


@nb.jit("float64[:](float64[:])", cache=True, nogil=True, nopython=True)
def ax2ro_single(ax):
    ro = np.zeros(4, dtype=np.float64)
    angle = ax[3]
    if (angle > -1e-8) and (angle < 1e-8):
        ro[2] = 1
    else:
        ro[:3] = ax[:3]
        # Need to deal with the 180 degree case
        if np.abs(angle - np.pi) < 1e-7:
            ro[3] = np.inf
        else:
            ro[3] = np.tan(angle * 0.5)
    return ro


@nb.jit("float64[:, :](float64[:, :])", cache=True, nogil=True, nopython=True)
def ax2ro(ax):
    ro = np.zeros_like(ax)
    for i in nb.prange(ax.shape[0]):
        ro[i] = ax2ro_single(ax[i])
    return ro


@nb.jit("float64[:](float64[:])", cache=True, nogil=True, nopython=True)
def ho2ro_single(ho):
    return ax2ro_single(ho2ax_single(ho))


@nb.jit("float64[:, :](float64[:, :])", cache=True, nogil=True, nopython=True)
def ho2ro(ho):
    n_vectors = ho.shape[0]
    ro = np.zeros((n_vectors, 4), dtype=np.float64)
    for i in nb.prange(n_vectors):
        ro[i] = ho2ro_single(ho[i])
    return ro


@nb.jit("float64[:](float64[:])", cache=True, nogil=True, nopython=True)
def cu2ro_single(cu):
    if np.max(np.abs(cu)) == 0:
        return np.array([0, 0, 1, 0], dtype=np.float64)
    else:
        return ho2ro_single(cu2ho_single(cu))


@nb.jit("float64[:, :](float64[:, :])", cache=True, nogil=True, nopython=True)
def cu2ro(cu):
    n_vectors = cu.shape[0]
    ro = np.zeros((n_vectors, 4), dtype=np.float64)
    for i in nb.prange(n_vectors):
        ro[i] = cu2ro_single(cu[i])
    return ro


@nb.jit("float64[:](float64[:])", cache=True, nogil=True, nopython=True)
def ro2ax_single(ro):
    """Convert one Rodrigues vector to an axis-angle pair."""
    if (ro[3] > -1e-8) and (ro[3] < 1e-8):
        return np.array([0, 0, 1, 0], dtype=np.float64)
    elif np.isinf(ro[3]):
        return np.append(ro[:3], np.pi)
    else:
        norm = np.sqrt(np.sum(np.square(ro[:3]), axis=-1))
        return np.append(ro[:3] / norm, 2 * np.arctan(ro[3]))


@nb.jit("float64[:, :](float64[:, :])", cache=True, nogil=True, nopython=True)
def ro2ax(ro):
    """Convert many Rodrigues vectors to axis-angle pairs."""
    n_vectors = ro.shape[0]
    ax = np.zeros((n_vectors, 4), dtype=np.float64)
    for i in nb.prange(n_vectors):
        ax[i] = ro2ax_single(ro[i])
    return ax


@nb.jit("float64[:](float64[:])", cache=True, nogil=True, nopython=True)
def ax2qu_single(ax):
    """Convert one axis-angle pair to a quaternion."""
    if (ax[3] > -1e-8) and (ax[3] < 1e-8):
        return np.array([1, 0, 0, 0], dtype=np.float64)
    else:
        c = np.cos(ax[3] * 0.5)
        s = np.sin(ax[3] * 0.5)
        return np.append(c, ax[:3] * s)


@nb.jit("float64[:, :](float64[:, :])", cache=True, nogil=True, nopython=True)
def ax2qu(ax):
    """Convert many axis-angle pairs to quaternions."""
    n_vectors = ax.shape[0]
    qu = np.zeros((n_vectors, 4), dtype=np.float64)
    for i in nb.prange(n_vectors):
        qu[i] = ax2qu_single(ax[i])
    return qu


@nb.jit("float64[:, :](int64)", cache=True, nogil=True, nopython=True)
def _cubochoric_sampling_loop(n_cube_steps):
    cube_edge_length = 0.5 * np.pi ** (2 / 3)
    delta = cube_edge_length / n_cube_steps
    n_iterations = (2 * n_cube_steps + 1) ** 3
    quaternions = np.zeros((n_iterations, 4))

    v_cubochoric = np.zeros(3)
    step = 0
    for i in nb.prange(-n_cube_steps + 1, n_cube_steps + 1):
        v_cubochoric[0] = i * delta
        for j in range(-n_cube_steps + 1, n_cube_steps + 1):
            v_cubochoric[1] = j * delta
            for k in range(-n_cube_steps + 1, n_cube_steps + 1):
                v_cubochoric[2] = k * delta

                # Discard the point if it lies outside the cubochoric
                # cell
                if np.max(np.abs(v_cubochoric)) > cube_edge_length:
                    continue

                # Get quaternion via cubochoric coordinates -> Rodrigues
                # vector -> axis-angle pair
                rodrigues = cu2ro_single(v_cubochoric)
                axis_angle = ro2ax_single(rodrigues)
                quaternions[step] = ax2qu_single(axis_angle)

                step += 1

    return quaternions
