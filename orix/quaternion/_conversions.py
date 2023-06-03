# -*- coding: utf-8 -*-
# Copyright 2018-2023 the orix developers
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

"""Conversions of rotations between many common representations from
:cite:`rowenhorst2015consistent`, accelerated with Numba.

This module and documentation is only relevant for orix developers, not
for users.

.. warning:
    This module is for internal use only.  Do not use it in your own
    code. We may change the API at any time with no warning.
"""

import numba as nb
import numpy as np


@nb.jit("int64(float64[:])", cache=True, nogil=True, nopython=True)
def get_pyramid_single(xyz: np.ndarray) -> int:
    """Determine to which out of six pyramids in the cube a (x, y, z)
    coordinate belongs.

    Parameters
    ----------
    xyz
        1D array (x, y, z) of 64-bit floats.

    Returns
    -------
    pyramid
        Which pyramid ``xyz`` belongs to as a 64-bit integer.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    x, y, z = xyz
    x_abs, y_abs, z_abs = abs(x), abs(y), abs(z)
    if (x_abs <= z) and (y_abs <= z):  # Top
        return 1
    elif (x_abs <= -z) and (y_abs <= -z):  # Bottom
        return 2
    elif (z_abs <= x) and (y_abs <= x):  # Front
        return 3
    elif (z_abs <= -x) and (y_abs <= -x):  # Back
        return 4
    elif (x_abs <= y) and (z_abs <= y):  # Right
        return 5
    else:  # (x_abs <= -y) and (z_abs <= -y)  # Left
        return 6


@nb.jit("int64[:](float64[:, :])", cache=True, nogil=True, nopython=True)
def get_pyramid_2d(xyz: np.ndarray) -> np.ndarray:
    """Determine to which out of six pyramids in the cube a 2D array of
    (x, y, z) coordinates belongs.

    Parameters
    ----------
    xyz
        2D array of n (x, y, z) as 64-bit floats.

    Returns
    -------
    pyramids
        1D array of pyramids as 64-bit integers.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    n_scalars = xyz.shape[0]
    pyramids = np.zeros(n_scalars, dtype=np.int64)
    for i in nb.prange(n_scalars):
        pyramids[i] = get_pyramid_single(xyz[i])
    return pyramids


def get_pyramid(xyz: np.ndarray) -> np.ndarray:
    """n-dimensional wrapper for get_pyramid_2d, see the docstring of
    that function.
    """
    n_xyz = np.prod(xyz.shape[:-1])
    xyz2d = xyz.astype(np.float64).reshape(n_xyz, 3)
    pyramids = get_pyramid_2d(xyz2d).reshape(n_xyz)
    return pyramids


@nb.jit("float64[:](float64[:])", cache=True, nogil=True, nopython=True)
def cu2ho_single(cu: np.ndarray) -> np.ndarray:
    """Conversion from a single set of cubochoric coordinates to
    un-normalized homochoric coordinates :cite:`singh2016orientation`.

    Parameters
    ----------
    cu
        1D array of (x, y, z) as 64-bit floats.

    Returns
    -------
    ho
        1D array of (x, y, z) as 64-bit floats.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
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
            c = t1**2 + t2**2
            s = np.pi * c / (24 * z**2)
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
def cu2ho_2d(cu: np.ndarray) -> np.ndarray:
    """Conversion from multiple cubochoric coordinates to un-normalized
    homochoric coordinates :cite:`singh2016orientation`.

    Parameters
    ----------
    cu
        2D array of n (x, y, z) as 64-bit floats.

    Returns
    -------
    ho
        2D array of n (x, y, z) as 64-bit floats.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    ho = np.zeros_like(cu, dtype=np.float64)
    for i in nb.prange(cu.shape[0]):
        ho[i] = cu2ho_single(cu[i])
    return ho


def cu2ho(cu: np.ndarray) -> np.ndarray:
    """N-dimensional wrapper for cu2ho_2d, see the docstring of that
    function.
    """
    n_cu = np.prod(cu.shape[:-1])
    cu2d = cu.astype(np.float64).reshape(n_cu, 3)
    ho = cu2ho_2d(cu2d).reshape(cu.shape)
    return ho


@nb.jit("float64[:](float64[:])", cache=True, nogil=True, nopython=True)
def ho2ax_single(ho: np.ndarray) -> np.ndarray:
    """Conversion from a single set of homochoric coordinates to an
    un-normalized axis-angle pair :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    ho
        1D array of (x, y, z) as 64-bit floats.

    Returns
    -------
    ax
        1D array of (x, y, z, angle) as 64-bit floats.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    # Constants stolen directly from EMsoft
    # fmt: off
    fit_parameters = np.array([
         0.9999999999999968,     -0.49999999999986866,     -0.025000000000632055,
        -0.003928571496460683,   -0.0008164666077062752,   -0.00019411896443261646,
        -0.00004985822229871769, -0.000014164962366386031, -1.9000248160936107e-6,
        -5.72184549898506e-6,     7.772149920658778e-6,    -0.00001053483452909705,
         9.528014229335313e-6,   -5.660288876265125e-6,     1.2844901692764126e-6,
         1.1255185726258763e-6,  -1.3834391419956455e-6,   7.513691751164847e-7,
        -2.401996891720091e-7,    4.386887017466388e-8,   -3.5917775353564864e-9
    ])
    # fmt: on
    ho_magnitude = np.sum(ho**2)
    if (ho_magnitude > -1e-8) and (ho_magnitude < 1e-8):
        ax = np.array([0, 0, 1, 0], dtype=np.float64)
    else:
        # Convert the magnitude to the rotation angle
        hom = ho_magnitude
        s = fit_parameters[0] + fit_parameters[1] * hom
        for i in nb.prange(2, 21):
            hom = hom * ho_magnitude
            s = s + fit_parameters[i] * hom
        hon = ho / np.sqrt(ho_magnitude)
        s = 2 * np.arccos(s)
        if np.abs(s - np.pi) < 1e-8:  # pragma: no cover
            ax = np.append(hon, np.pi)
        else:
            ax = np.append(hon, s)
    return ax


@nb.jit("float64[:, :](float64[:, :])", cache=True, nogil=True, nopython=True)
def ho2ax_2d(ho: np.ndarray) -> np.ndarray:
    """Conversion from multiple homochoric coordinates to un-normalized
    axis-angle pairs :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    ho
        2D array of n (x, y, z) as 64-bit floats.

    Returns
    -------
    ax
        2D array of n (x, y, z, angle) as 64-bit floats.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    n_vectors = ho.shape[0]
    ax = np.zeros((n_vectors, 4), dtype=np.float64)
    for i in nb.prange(n_vectors):
        ax[i] = ho2ax_single(ho[i])
    return ax


def ho2ax(ho: np.ndarray) -> np.ndarray:
    """N-dimensional wrapper for ho2ax_2d, see the docstring of that
    function.
    """
    n_ho = np.prod(ho.shape[:-1])
    ho2d = ho.astype(np.float64).reshape(n_ho, 3)
    ho = ho2ax_2d(ho2d).reshape(ho.shape[:-1] + (4,))
    return ho


@nb.jit("float64[:](float64[:])", cache=True, nogil=True, nopython=True)
def ax2ro_single(ax: np.ndarray) -> np.ndarray:
    """Conversion from a single angle-axis pair to an un-normalized
    Rodrigues vector :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    ax
        1D array of (x, y, z, angle) as 64-bit floats.

    Returns
    -------
    ro
        1D array of (x, y, z, angle) as 64-bit floats.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
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
def ax2ro_2d(ax: np.ndarray) -> np.ndarray:
    """Conversion from multiple axis-angle pairs to un-normalized
    Rodrigues vectors :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    ax
        2D array of n (x, y, z, angle) as 64-bit floats.

    Returns
    -------
    ro
        2D array of n (x, y, z, angle) as 64-bit floats.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    ro = np.zeros_like(ax, dtype=np.float64)
    for i in nb.prange(ax.shape[0]):
        ro[i] = ax2ro_single(ax[i])
    return ro


def ax2ro(ax: np.ndarray) -> np.ndarray:
    """N-dimensional wrapper for ax2ro_2d, see the docstring of that
    function.
    """
    n_ax = np.prod(ax.shape[:-1])
    ax2d = ax.astype(np.float64).reshape(n_ax, 4)
    ro = ax2ro_2d(ax2d).reshape(ax.shape)
    return ro


@nb.jit("float64[:](float64[:])", cache=True, nogil=True, nopython=True)
def ro2ax_single(ro: np.ndarray) -> np.ndarray:
    """Conversion from a single Rodrigues vector to an un-normalized
    axis-angle pair :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    ro
        1D array of (x, y, z, angle) as 64-bit floats.

    Returns
    -------
    ax
        1D array of (x, y, z, angle) as 64-bit floats.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    if (ro[3] > -1e-8) and (ro[3] < 1e-8):
        return np.array([0, 0, 1, 0], dtype=np.float64)
    elif np.isinf(ro[3]):
        return np.append(ro[:3], np.pi)
    else:
        norm = np.sqrt(np.sum(np.square(ro[:3]), axis=-1))
        return np.append(ro[:3] / norm, 2 * np.arctan(ro[3]))


@nb.jit("float64[:, :](float64[:, :])", cache=True, nogil=True, nopython=True)
def ro2ax_2d(ro: np.ndarray) -> np.ndarray:
    """Conversion from multiple Rodrigues vectors to un-normalized
    axis-angle pairs :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    ro
        2D array of n (x, y, z, angle) as 64-bit floats.

    Returns
    -------
    ax
        2D array of n (x, y, z, angle) as 64-bit floats.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    n_vectors = ro.shape[0]
    ax = np.zeros((n_vectors, 4), dtype=np.float64)
    for i in nb.prange(n_vectors):
        ax[i] = ro2ax_single(ro[i])
    return ax


def ro2ax(ro: np.ndarray) -> np.ndarray:
    """N-dimensional wrapper for ro2ax_2d, see the docstring of that
    function.
    """
    n_ro = np.prod(ro.shape[:-1])
    ro2d = ro.astype(np.float64).reshape(n_ro, 4)
    ax = ro2ax_2d(ro2d).reshape(ro.shape)
    return ax


@nb.jit("float64[:](float64[:])", cache=True, nogil=True, nopython=True)
def ax2qu_single(ax: np.ndarray) -> np.ndarray:
    """Conversion from a single axis-angle pair to an un-normalized
    quaternion :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    ax
        1D array of (x, y, z, angle) as 64-bit floats.

    Returns
    -------
    qu
        1D array of (a, b, c, d) as 64-bit floats.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    if (ax[3] > -1e-8) and (ax[3] < 1e-8):
        return np.array([1, 0, 0, 0], dtype=np.float64)
    else:
        c = np.cos(ax[3] * 0.5)
        s = np.sin(ax[3] * 0.5)
        return np.append(c, ax[:3] * s)


@nb.jit("float64[:, :](float64[:, :])", cache=True, nogil=True, nopython=True)
def ax2qu_2d(ax: np.ndarray) -> np.ndarray:
    """Conversion from multiple axis-angle pairs to un-normalized
    quaternions :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    ax
        2D array of n (x, y, z, angle) as 64-bit floats.

    Returns
    -------
    qu
        2D array of n (a, b, c, d) as 64-bit floats.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    n_vectors = ax.shape[0]
    qu = np.zeros((n_vectors, 4), dtype=np.float64)
    for i in nb.prange(n_vectors):
        qu[i] = ax2qu_single(ax[i])
    return qu


def ax2qu(axes: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """N-dimensional wrapper for ax2qu_2d, see the docstring of that
    function for further details.

    Parameters
    ----------
    axes
        (...,3) dimensional numpy array of (x, y, z) vectors.

    angles
        numpy array of angles in radians.

    Returns
    -------
    qu
        2D array of n (a, b, c, d) as 64-bit floats.

    """
    # convert to numpy arrays of shape (...,3) and (...,1)
    axes = np.atleast_2d(axes)
    angles = np.atleast_1d(angles)
    if axes.shape[-1] != 3:
        raise ValueError("axes must be an array of shape (...,3)")
    if angles.shape[-1] != 1 or angles.shape == (1,):
        angles = angles.reshape(angles.shape + (1,))
    # get the shape of the data itself.
    ax_shape = axes.shape[:-1]
    ang_shape = angles.shape[:-1]
    # case of n-dimensional axis and single angle
    if ang_shape == (1,):
        angles = np.ones(ax_shape + (1,)) * angles
    # case of single axis and n-dimensional angle
    elif ax_shape == (1,):
        axes = np.ones(ang_shape + (3,)) * axes
    elif ax_shape != ang_shape:
        raise ValueError(
            """
        The dimensions of axes and angles are {} and {}, respectively.
        Either the dimensions must match, or one must be a singular value.
        """.format(
                axes.shape, angles.shape
            )
        )
    ax = np.concatenate([axes.data, angles], axis=-1)

    # convert the 'ax' array to the 2D array expected by ax2qu_2d
    n_ax = np.prod(ax.shape[:-1])
    ax2d = ax.astype(np.float64).reshape(n_ax, 4)
    # reshape the resulting quaternion to the original shape.
    qu = ax2qu_2d(ax2d).reshape(ax.shape)
    return qu


@nb.jit("float64[:](float64[:])", cache=True, nogil=True, nopython=True)
def ho2ro_single(ho: np.ndarray) -> np.ndarray:
    """Conversion from a single set of homochoric coordinates to an
    un-normalized Rodrigues vector :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    ho
        1D array of (x, y, z) as 64-bit floats.

    Returns
    -------
    ro
        1D array of (x, y, z, angle) as 64-bit floats.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    return ax2ro_single(ho2ax_single(ho))


@nb.jit("float64[:, :](float64[:, :])", cache=True, nogil=True, nopython=True)
def ho2ro_2d(ho: np.ndarray) -> np.ndarray:
    """Conversion from multiple homochoric coordinates to un-normalized
    Rodrigues vectors :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    ho
        2D array of n (x, y, z) as 64-bit floats.

    Returns
    -------
    ax
        2D array of n (x, y, z, angle) as 64-bit floats.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    n_vectors = ho.shape[0]
    ro = np.zeros((n_vectors, 4), dtype=np.float64)
    for i in nb.prange(n_vectors):
        ro[i] = ho2ro_single(ho[i])
    return ro


def ho2ro(ho: np.ndarray) -> np.ndarray:
    """N-dimensional wrapper for ho2ro_2d, see the docstring of that
    function.
    """
    n_ho = np.prod(ho.shape[:-1])
    ho2d = ho.astype(np.float64).reshape(n_ho, 3)
    ro = ho2ro_2d(ho2d).reshape(ho.shape[:-1] + (4,))
    return ro


@nb.jit("float64[:](float64[:])", cache=True, nogil=True, nopython=True)
def cu2ro_single(cu: np.ndarray) -> np.ndarray:
    """Conversion from a single set of cubochoric coordinates to an
    un-normalized Rodrigues vector :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    cu
        1D array of (x, y, z) as 64-bit floats.

    Returns
    -------
    ro
        1D array of (x, y, z, angle) as 64-bit floats.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    if np.max(np.abs(cu)) == 0:
        return np.array([0, 0, 1, 0], dtype=np.float64)
    else:
        return ho2ro_single(cu2ho_single(cu))


@nb.jit("float64[:, :](float64[:, :])", cache=True, nogil=True, nopython=True)
def cu2ro_2d(cu: np.ndarray) -> np.ndarray:
    """Conversion from multiple cubochoric coordinates to un-normalized
    Rodrigues vectors :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    cu
        2D array of n (x, y, z) as 64-bit floats.

    Returns
    -------
    ro
        2D array of n (x, y, z, angle) as 64-bit floats.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    n_vectors = cu.shape[0]
    ro = np.zeros((n_vectors, 4), dtype=np.float64)
    for i in nb.prange(n_vectors):
        ro[i] = cu2ro_single(cu[i])
    return ro


def cu2ro(cu: np.ndarray) -> np.ndarray:
    """N-dimensional wrapper for cu2ro_2d, see the docstring of that
    function.
    """
    n_cu = np.prod(cu.shape[:-1])
    cu2d = cu.astype(np.float64).reshape(n_cu, 3)
    ro = cu2ro_2d(cu2d).reshape(cu.shape[:-1] + (4,))
    return ro


@nb.jit("float64[:](float64[:])", cache=True, nogil=True, nopython=True)
def eu2qu_single(eu: np.ndarray) -> np.ndarray:
    """Convert three Euler angles (alpha, beta, gamma) to a unit
    quaternion.

    Parameters
    ----------
    eu
        1D array of (alpha, beta, gamma) Euler angles given in radians
        in the Bunge convention (ie, passive Z-X-Z) as 64-bit floats.

    Returns
    -------
    qu
        1D unit quaternion (a, b, c, d) as 64-bit floats.

    Notes
    -----
    Uses Eqs. A.5 & A.6 :cite:`rowenhorst2015consistent`.

    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    sigma = 0.5 * np.add(eu[0], eu[2])
    delta = 0.5 * np.subtract(eu[0], eu[2])
    c = np.cos(eu[1] / 2)
    s = np.sin(eu[1] / 2)

    qu = np.zeros(4, dtype=np.float64)
    qu[0] = np.array(c * np.cos(sigma), dtype=np.float64)
    qu[1] = np.array(-s * np.cos(delta), dtype=np.float64)
    qu[2] = np.array(-s * np.sin(delta), dtype=np.float64)
    qu[3] = np.array(-c * np.sin(sigma), dtype=np.float64)

    if qu[0] < 0:
        qu = -qu

    return qu


@nb.jit("float64[:, :](float64[:, :])", cache=True, nogil=True, nopython=True)
def eu2qu_2d(eu: np.ndarray) -> np.ndarray:
    """Conversion from multiple Euler angles (alpha, beta, gamma) to unit
    quaternions

    Parameters
    ----------
    eu
        2D array of n (alpha, beta, gamma) as 64-bit floats.

    Returns
    -------
    qu
        2D array of n (q0, q1, q2, q3) quaternions as 64-bit floats.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    n_vectors = eu.shape[0]
    qu = np.zeros((n_vectors, 4), dtype=np.float64)
    for i in nb.prange(n_vectors):
        qu[i] = eu2qu_single(eu[i])
    return qu


def eu2qu(eu: np.ndarray) -> np.ndarray:
    """N-dimensional wrapper for eu2qu_2d, see the docstring of that
    function.
    """
    n_eu = np.prod(eu.shape[:-1])
    eu2d = eu.astype(np.float64).reshape(n_eu, 3)
    qu = eu2qu_2d(eu2d).reshape(eu.shape[:-1] + (4,))
    return qu


@nb.jit("float64[:](float64[:, :])", cache=True, nogil=True, nopython=True)
def om2qu_single(om: np.ndarray) -> np.ndarray:
    """Convert a single (3,3) rotation matrix into a unit quaternion

    Parameters
    ----------
    ma
        (3, 3) rotation matrix stored as a 3D nupy array as 64-bit floats.

    Returns
    -------
    qu
        1D unit quaternion (a, b, c, d) as 64-bit floats.

    Notes
    -----
    Uses Eqs. A.11 :cite:`rowenhorst2015consistent`.

    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """

    q0_almost = 1 + om[0, 0] + om[1, 1] + om[2, 2]
    q1_almost = 1 + om[0, 0] - om[1, 1] - om[2, 2]
    q2_almost = 1 - om[0, 0] + om[1, 1] - om[2, 2]
    q3_almost = 1 - om[0, 0] - om[1, 1] + om[2, 2]

    qu = np.zeros(4, dtype=np.float64)
    eps = np.finfo(np.float64).eps

    if q0_almost < eps:
        qu[0] = 0
    else:
        qu[0] = 0.5 * np.sqrt(q0_almost)

    if q1_almost < eps:
        qu[1] = 0
    elif om[2, 1] < om[1, 2]:
        qu[1] = -0.5 * np.sqrt(q1_almost)
    else:
        qu[1] = 0.5 * np.sqrt(q1_almost)

    if q2_almost < eps:
        qu[2] = 0
    elif om[0, 2] < om[2, 0]:
        qu[2] = -0.5 * np.sqrt(q2_almost)
    else:
        qu[2] = 0.5 * np.sqrt(q2_almost)

    if q3_almost < eps:
        qu[3] = 0
    elif om[1, 0] < om[0, 1]:
        qu[3] = -0.5 * np.sqrt(q3_almost)
    else:
        qu[3] = 0.5 * np.sqrt(q3_almost)

    return qu


@nb.jit("float64[:, :](float64[:, :, :])", cache=True, nogil=True, nopython=True)
def om2qu_3d(om: np.ndarray) -> np.ndarray:
    """Conversion from multiple rotation matrices to unit quaternions

    Parameters
    ----------
    om
        3D array of n (3, 3) rotation matrices as 64-bit floats.

    Returns
    -------
    qu
        2D array of n (q0, q1, q2, q3) quaternions as 64-bit floats.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    n_vectors = om.shape[0]
    qu = np.zeros((n_vectors, 4), dtype=np.float64)
    for i in nb.prange(n_vectors):
        qu[i] = om2qu_single(om[i])
    return qu


def om2qu(om: np.ndarray) -> np.ndarray:
    """N-dimensional wrapper for om2qu_3d, see the docstring of that
    function.
    """
    if om.shape == (3, 3):
        n_om = 1
    else:
        n_om = np.prod(om.shape[:-2])
    om3d = om.astype(np.float64).reshape(n_om, 3, 3)
    qu = om2qu_3d(om3d).reshape(om.shape[:-2] + (4,))
    return qu
