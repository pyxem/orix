# Copyright 2018-2024 the orix developers
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
"""

from typing import Tuple

import numba as nb
import numpy as np

from orix import constants


@nb.njit("int64(float64[:])", cache=True, fastmath=True, nogil=True)
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


@nb.njit("int64[:](float64[:, :])", cache=True, fastmath=True, nogil=True)
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
    n = xyz.shape[0]
    pyramids = np.zeros(n, dtype=np.int64)
    for i in nb.prange(n):
        pyramids[i] = get_pyramid_single(xyz[i])
    return pyramids


def get_pyramid(xyz: np.ndarray) -> np.ndarray:
    """N-dimensional wrapper for get_pyramid_2d, see the docstring of
    that function.
    """
    xyz2d = xyz.astype(np.float64)
    xyz2d = xyz2d.reshape(-1, 3)

    pyramids = get_pyramid_2d(xyz2d).ravel()

    return pyramids


@nb.njit("float64[:](float64[:])", cache=True, fastmath=True, nogil=True)
def cu2ho_single(cu: np.ndarray) -> np.ndarray:
    """Convert a single set of cubochoric coordinates to un-normalized
    homochoric coordinates :cite:`singh2016orientation`.

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


@nb.njit("float64[:, :](float64[:, :])", cache=True, fastmath=True, nogil=True)
def cu2ho_2d(cu: np.ndarray) -> np.ndarray:
    """Convert multiple cubochoric coordinates to un-normalized
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
    cu2d = cu.astype(np.float64)
    cu2d = cu2d.reshape(-1, 3)

    ho = cu2ho_2d(cu2d)
    ho = ho.reshape(cu.shape)

    return ho


@nb.njit("float64[:](float64[:])", cache=True, fastmath=True, nogil=True)
def ho2ax_single(ho: np.ndarray) -> np.ndarray:
    """Convert a single set of homochoric coordinates to an
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
    # Constants copied from EMsoft
    # fmt: off
    fit_parameters = np.array([
         0.9999999999999968,     -0.49999999999986866,     -0.025000000000632055,
        -0.003928571496460683,   -0.0008164666077062752,   -0.00019411896443261646,
        -0.00004985822229871769, -0.000014164962366386031, -1.9000248160936107e-6,
        -5.72184549898506e-6,     7.772149920658778e-6,    -0.00001053483452909705,
         9.528014229335313e-6,   -5.660288876265125e-6,     1.2844901692764126e-6,
         1.1255185726258763e-6,  -1.3834391419956455e-6,    7.513691751164847e-7,
        -2.401996891720091e-7,    4.386887017466388e-8,    -3.5917775353564864e-9
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


@nb.njit("float64[:, :](float64[:, :])", cache=True, fastmath=True, nogil=True)
def ho2ax_2d(ho: np.ndarray) -> np.ndarray:
    """Convert multiple homochoric coordinates to un-normalized
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
    n = ho.shape[0]
    ax = np.zeros((n, 4), dtype=np.float64)
    for i in nb.prange(n):
        ax[i] = ho2ax_single(ho[i])
    return ax


def ho2ax(ho: np.ndarray) -> np.ndarray:
    """N-dimensional wrapper for ho2ax_2d, see the docstring of that
    function.
    """
    ho2d = ho.astype(np.float64)
    ho2d = ho2d.reshape(-1, 3)

    ax = ho2ax_2d(ho2d)
    ax = ax.reshape(ho.shape[:-1] + (4,))

    return ax


@nb.njit("float64[:](float64[:])", cache=True, fastmath=True, nogil=True)
def ax2ro_single(ax: np.ndarray) -> np.ndarray:
    """Convert a single angle-axis pair to an un-normalized Rodrigues
    vector :cite:`rowenhorst2015consistent`.

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
        # Need to deal with the 180 degree case. A cutoff of 0.001 will
        # give a maximum Rodrigues magnitude of approximately 2000.
        # Raising it higher can cause rounding errors during conversions
        # to other rotation representations. If there is ever a
        # situation where this error must be smaller, the accuracy of
        # the test values for Homochoric and axis/angle in the tests
        # must be increased to more than 4 significant figures.
        if np.abs(angle - np.pi) < 1e-3:
            ro[3] = np.inf
        else:
            ro[3] = np.tan(angle * 0.5)
    return ro


@nb.njit("float64[:, :](float64[:, :])", cache=True, fastmath=True, nogil=True)
def ax2ro_2d(ax: np.ndarray) -> np.ndarray:
    """Convert multiple axis-angle pairs to un-normalized Rodrigues
    vectors :cite:`rowenhorst2015consistent`.

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
    n = ax.shape[0]
    ro = np.zeros((n, 4), dtype=np.float64)
    for i in nb.prange(n):
        ro[i] = ax2ro_single(ax[i])
    return ro


def ax2ro(ax: np.ndarray) -> np.ndarray:
    """N-dimensional wrapper for ax2ro_2d, see the docstring of that
    function.
    """
    ax2d = ax.astype(np.float64)
    ax2d = ax2d.reshape(-1, 4)

    ro = ax2ro_2d(ax2d)
    ro = ro.reshape(ax.shape)

    return ro


@nb.njit("float64[:](float64[:])", cache=True, fastmath=True, nogil=True)
def ro2ax_single(ro: np.ndarray) -> np.ndarray:
    """Convert a single Rodrigues vector to an un-normalized axis-angle
    pair :cite:`rowenhorst2015consistent`.

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


@nb.njit("float64[:, :](float64[:, :])", cache=True, fastmath=True, nogil=True)
def ro2ax_2d(ro: np.ndarray) -> np.ndarray:
    """Convert multiple Rodrigues vectors to un-normalized axis-angle
    pairs :cite:`rowenhorst2015consistent`.

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
    n = ro.shape[0]
    ax = np.zeros((n, 4), dtype=np.float64)
    for i in nb.prange(n):
        ax[i] = ro2ax_single(ro[i])
    return ax


def ro2ax(ro: np.ndarray) -> np.ndarray:
    """N-dimensional wrapper for ro2ax_2d, see the docstring of that
    function.
    """
    ro2d = ro.astype(np.float64)
    ro2d = ro2d.reshape(-1, 4)

    ax = ro2ax_2d(ro2d)
    ax = ax.reshape(ro.shape)

    return ax


@nb.njit("float64[:](float64[:])", cache=True, fastmath=True, nogil=True)
def ax2qu_single(ax: np.ndarray) -> np.ndarray:
    """Convert a single axis-angle pair to a unit quaternion
    :cite:`rowenhorst2015consistent`.

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
        qu = np.append(c, ax[:3] * s)
        norm = np.sqrt(np.sum(np.square(qu)))
        qu = qu / norm
        return qu


@nb.njit("float64[:, :](float64[:, :])", cache=True, fastmath=True, nogil=True)
def ax2qu_2d(ax: np.ndarray) -> np.ndarray:
    """Convert multiple axis-angle pairs to unit quaternions
    :cite:`rowenhorst2015consistent`.

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
    n = ax.shape[0]
    qu = np.zeros((n, 4), dtype=np.float64)
    for i in nb.prange(n):
        qu[i] = ax2qu_single(ax[i])
    return qu


def ax2qu(axes: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """N-dimensional wrapper for ax2qu_2d, see the docstring of that
    function for further details.

    Parameters
    ----------
    axes
        N-dimensional array of (x, y, z) vectors with the final
        dimension equal to 3.
    angles
        Angles in radians.

    Returns
    -------
    qu
        2D array of n (a, b, c, d) as 64-bit floats.
    """
    axes = np.atleast_2d(axes)
    angles = np.atleast_1d(angles)

    if axes.shape[-1] != 3:
        raise ValueError("Final dimension of axes array must be 3.")
    if angles.shape[-1] != 1 or angles.shape == (1,) or axes.shape[:-1] == angles.shape:
        angles = angles.reshape(angles.shape + (1,))

    axes_shape = axes.shape[:-1]
    angles_shape = angles.shape[:-1]

    if angles_shape == (1,):
        # N-dimensional axis and single angle
        angles = np.ones(axes_shape + (1,)) * angles
    elif axes_shape == (1,):
        # Single axis and n-dimensional angle
        axes = np.ones(angles_shape + (3,)) * axes
    elif axes_shape != angles_shape:
        raise ValueError(
            f"The dimensions of axes {axes_shape} and angles {angles_shape} are "
            "incompatible. The dimensions must match or one must be a singular value."
        )

    axes_angles = np.concatenate([axes.data, angles], axis=-1)
    axes_angles_2d = axes_angles.reshape(-1, 4)
    axes_angles_2d = axes_angles_2d.astype(np.float64)

    qu = ax2qu_2d(axes_angles_2d)
    qu = qu.reshape(axes_angles.shape)

    return qu


@nb.njit("float64[:](float64[:])", cache=True, fastmath=True, nogil=True)
def qu2ax_single(qu: np.ndarray) -> np.ndarray:
    """Convert a single (un)normalized quaternion to a normalized
    axis-angle pair :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    qu
        1D array of (a, b, c, d) as 64-bit floats.

    Returns
    -------
    ax
        1D array of (x, y, z, angle) as 64-bit floats.

    Notes
    -----
    Uses Eq. A.16 in :cite:`rowenhorst2015consistent`.

    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    omega = 2 * np.arccos(qu[0])

    if omega < constants.eps9:
        return np.array([0, 0, 1, 0], dtype=np.float64)

    if np.abs(qu[0]) < constants.eps9:
        return np.array([qu[1], qu[2], qu[3], np.pi], dtype=np.float64)

    s = np.sqrt(np.sum(np.square(qu[1:])))
    if qu[0] <= 0:
        s = -s

    ax = np.array([qu[1], qu[2], qu[3], omega], dtype=np.float64)
    ax[:3] = ax[:3] / s

    return ax


@nb.njit("float64[:, :](float64[:, :])", cache=True, fastmath=True, nogil=True)
def qu2ax_2d(qu: np.ndarray) -> np.ndarray:
    """Convert multiple (un)normalized quaternions to normalized
    axis-angle pairs :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    qu
        2D array of n (a, b, c, d) as 64-bit floats.

    Returns
    -------
    ax
        2D array of n (x, y, z, angle) as 64-bit floats.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    n = qu.shape[0]
    ax = np.zeros((n, 4), dtype=np.float64)
    for i in nb.prange(n):
        ax[i] = qu2ax_single(qu[i])
    return ax


def qu2ax(qu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """N-dimensional wrapper for qu2ax_2d, see the docstring of that
    function for further details.

    Parameters
    ----------
    qu
        Quaternion(s) (a, b, c, d) with the final array dimension equal
        to 4.

    Returns
    -------
    axes
        Rotation axes of the same shape as the input array but with the
        final array dimension equal to 3.
    angles
        Rotation angles in radians of the same shape as the input array
        but with the final array dimension equal to 1.
    """
    qu_nd = np.atleast_2d(qu)

    if qu_nd.shape[-1] != 4:
        raise ValueError("Final dimension of quaternion array must be 4.")

    qu2d = qu.reshape(-1, 4)
    qu2d = qu2d.astype(np.float64)

    ax = qu2ax_2d(qu2d)
    ax = ax.reshape(qu.shape)

    axes = ax[..., :3]
    angles = ax[..., 3].reshape(ax.shape[:-1] + (1,))

    return axes, angles


@nb.njit("float64[:](float64[:])", cache=True, fastmath=True, nogil=True)
def ho2ro_single(ho: np.ndarray) -> np.ndarray:
    """Convert a single set of homochoric coordinates to an
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


@nb.njit("float64[:, :](float64[:, :])", cache=True, fastmath=True, nogil=True)
def ho2ro_2d(ho: np.ndarray) -> np.ndarray:
    """Convert multiple homochoric coordinates to un-normalized
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
    n = ho.shape[0]
    ro = np.zeros((n, 4), dtype=np.float64)
    for i in nb.prange(n):
        ro[i] = ho2ro_single(ho[i])
    return ro


def ho2ro(ho: np.ndarray) -> np.ndarray:
    """N-dimensional wrapper for ho2ro_2d, see the docstring of that
    function.
    """
    ho2d = ho.astype(np.float64)
    ho2d = ho2d.reshape(-1, 3)

    ro = ho2ro_2d(ho2d)
    ro = ro.reshape(ho.shape[:-1] + (4,))

    return ro


@nb.njit("float64[:](float64[:])", cache=True, fastmath=True, nogil=True)
def cu2ro_single(cu: np.ndarray) -> np.ndarray:
    """Convert a single set of cubochoric coordinates to an
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


@nb.njit("float64[:, :](float64[:, :])", cache=True, fastmath=True, nogil=True)
def cu2ro_2d(cu: np.ndarray) -> np.ndarray:
    """Convert multiple cubochoric coordinates to un-normalized
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
    n = cu.shape[0]
    ro = np.zeros((n, 4), dtype=np.float64)
    for i in nb.prange(n):
        ro[i] = cu2ro_single(cu[i])
    return ro


def cu2ro(cu: np.ndarray) -> np.ndarray:
    """N-dimensional wrapper for cu2ro_2d, see the docstring of that
    function.
    """
    cu2d = cu.astype(np.float64)
    cu2d = cu2d.reshape(-1, 3)

    ro = cu2ro_2d(cu2d)
    ro = ro.reshape(cu.shape[:-1] + (4,))

    return ro


@nb.njit("float64[:](float64[:])", cache=True, fastmath=True, nogil=True)
def eu2qu_single(eu: np.ndarray) -> np.ndarray:
    """Convert three Euler angles (alpha, beta, gamma) to a unit
    quaternion :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    eu
        1D array of (alpha, beta, gamma) Euler angles given in radians
        in the Bunge convention (i.e., passive Z-X-Z) as 64-bit floats.

    Returns
    -------
    qu
        1D unit quaternion (a, b, c, d) as 64-bit floats.

    Notes
    -----
    Uses Eqs. A.5 & A.6 in :cite:`rowenhorst2015consistent`.

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


@nb.njit("float64[:, :](float64[:, :])", cache=True, fastmath=True, nogil=True)
def eu2qu_2d(eu: np.ndarray) -> np.ndarray:
    """Convert multiple Euler angles (alpha, beta, gamma) to unit
    quaternions.

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
    n = eu.shape[0]
    qu = np.zeros((n, 4), dtype=np.float64)
    for i in nb.prange(n):
        qu[i] = eu2qu_single(eu[i])
    return qu


def eu2qu(eu: np.ndarray) -> np.ndarray:
    """N-dimensional wrapper for eu2qu_2d, see the docstring of that
    function.
    """
    eu2d = eu.astype(np.float64)
    eu2d = eu2d.reshape(-1, 3)

    qu = eu2qu_2d(eu2d)
    qu = qu.reshape(eu.shape[:-1] + (4,))

    return qu


@nb.njit("float64[:](float64[:, :])", cache=True, fastmath=True, nogil=True)
def om2qu_single(om: np.ndarray) -> np.ndarray:
    """Convert a single (3, 3) rotation matrix to a unit quaternion
    :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    om
        (3, 3) rotation matrix as an array of 64-bit floats.

    Returns
    -------
    qu
        1D unit quaternion (a, b, c, d) as 64-bit floats.

    Notes
    -----
    Uses Eq. A.11 in :cite:`rowenhorst2015consistent`.

    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    a_almost = 1 + om[0, 0] + om[1, 1] + om[2, 2]
    b_almost = 1 + om[0, 0] - om[1, 1] - om[2, 2]
    c_almost = 1 - om[0, 0] + om[1, 1] - om[2, 2]
    d_almost = 1 - om[0, 0] - om[1, 1] + om[2, 2]

    qu = np.zeros(4, dtype=np.float64)

    if a_almost < constants.eps9:
        qu[0] = 0
    else:
        qu[0] = 0.5 * np.sqrt(a_almost)

    if b_almost < constants.eps9:
        qu[1] = 0
    elif om[2, 1] < om[1, 2]:
        qu[1] = -0.5 * np.sqrt(b_almost)
    else:
        qu[1] = 0.5 * np.sqrt(b_almost)

    if c_almost < constants.eps9:
        qu[2] = 0
    elif om[0, 2] < om[2, 0]:
        qu[2] = -0.5 * np.sqrt(c_almost)
    else:
        qu[2] = 0.5 * np.sqrt(c_almost)

    if d_almost < constants.eps9:
        qu[3] = 0
    elif om[1, 0] < om[0, 1]:
        qu[3] = -0.5 * np.sqrt(d_almost)
    else:
        qu[3] = 0.5 * np.sqrt(d_almost)

    norm = np.sqrt(np.sum(np.square(qu)))
    qu = qu / norm

    return qu


@nb.njit("float64[:, :](float64[:, :, :])", cache=True, fastmath=True, nogil=True)
def om2qu_3d(om: np.ndarray) -> np.ndarray:
    """Convert multiple rotation matrices to unit quaternions
    :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    om
        3D array of n (3, 3) rotation matrices as 64-bit floats.

    Returns
    -------
    qu
        2D array of n (a, b, c, d) quaternions as 64-bit floats.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    n = om.shape[0]
    qu = np.zeros((n, 4), dtype=np.float64)
    for i in nb.prange(n):
        qu[i] = om2qu_single(om[i])
    return qu


def om2qu(om: np.ndarray) -> np.ndarray:
    """N-dimensional wrapper for om2qu_3d, see the docstring of that
    function for further details.
    """
    om3d = om.reshape((-1, 3, 3))
    om3d = om3d.astype(np.float64)

    qu = om2qu_3d(om3d)
    qu = qu.reshape(om.shape[:-2] + (4,))

    return qu


@nb.njit("float64[:](float64[:])", cache=True, fastmath=True, nogil=True)
def qu2eu_single(qu: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion to three Euler angles
    :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    qu
        Unit quaternion (a, b, c, d).

    Return
    ------
    eu
        Euler angles (alpha, beta, gamma) in radians in the Bunge
        convention (i.e., passive Z-X-Z).

    Notes
    -----
    Uses Eq. A.14 in :cite:`rowenhorst2015consistent`.

    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    eu = np.zeros(3, dtype=np.float64)

    # Following Eq. A.14, compute q_ad, q_bc, and chi, assuming P=1
    q_ad = (qu[0] * qu[0]) + (qu[3] * qu[3])
    q_bc = (qu[1] * qu[1]) + (qu[2] * qu[2])
    chi = np.sqrt(q_ad * q_bc)

    if chi < constants.eps9:
        if q_bc < constants.eps9:
            a = -2 * qu[0] * qu[3]
            b = qu[0] * qu[0] - qu[3] * qu[3]
        else:
            a = -2 * qu[1] * qu[2]
            b = qu[1] * qu[1] - qu[2] * qu[2]
            eu[1] = np.pi
        eu[0] = np.arctan2(a, b)
        return np.mod(eu, np.pi * 2)

    eu_0a = (qu[1] * qu[3] - qu[0] * qu[2]) / chi
    eu_0b = (-qu[0] * qu[1] - qu[2] * qu[3]) / chi
    eu_2a = (qu[0] * qu[2] + qu[1] * qu[3]) / chi
    eu_2b = (qu[2] * qu[3] - qu[0] * qu[1]) / chi

    eu[0] = np.arctan2(eu_0a, eu_0b)
    eu[1] = np.arctan2(2 * chi, q_ad - q_bc)
    eu[2] = np.arctan2(eu_2a, eu_2b)

    eu[np.abs(eu) < constants.eps9] = 0

    return np.mod(eu, np.pi * 2)


@nb.njit("float64[:, :](float64[:, :])", cache=True, fastmath=True, nogil=True)
def qu2eu_2d(qu: np.ndarray) -> np.ndarray:
    """Convert multiple unit quaternions to Euler angles.

    Parameters
    ----------
    qu
        2D array of n (a, b, c, d) quaternions as 64-bit floats.

    Returns
    -------
    eu
        2D array of n (alpha, beta, gamma) as 64-bit floats.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    n = qu.shape[0]
    eu = np.zeros((n, 3), dtype=np.float64)
    for i in nb.prange(n):
        eu[i] = qu2eu_single(qu[i])
    return eu


def qu2eu(qu: np.ndarray) -> np.ndarray:
    """N-dimensional wrapper for qu2eu_2d, see the docstring of that
    function.
    """
    qu2d = qu.reshape(-1, 4)
    qu2d = qu2d.astype(np.float64)

    eu = qu2eu_2d(qu2d)
    eu = eu.reshape(qu.shape[:-1] + (3,))

    return eu


@nb.njit("float64[:, :](float64[:])", cache=True, fastmath=True, nogil=True)
def qu2om_single(qu: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion to an orthogonal rotation matrix
     :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    qu
        1D unit quaternion (a, b, c, d) as 64-bit floats.

    Returns
    -------
    om
        (3, 3) rotation matrix as an array of 64-bit floats.

    Notes
    -----
    Uses Eq. A.15 :cite:`rowenhorst2015consistent`.

    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    om = np.zeros((3, 3), dtype=np.float64)

    bb = qu[1] ** 2
    cc = qu[2] ** 2
    dd = qu[3] ** 2
    qq = qu[0] ** 2 - (bb + cc + dd)  # q_mean in Eq. A.15

    bc = qu[1] * qu[2]
    ad = qu[0] * qu[3]
    bd = qu[1] * qu[3]
    ac = qu[0] * qu[2]
    cd = qu[2] * qu[3]
    ab = qu[0] * qu[1]
    om[0, 0] = qq + 2 * bb
    om[0, 1] = 2 * (bc - ad)
    om[0, 2] = 2 * (bd + ac)
    om[1, 0] = 2 * (bc + ad)
    om[1, 1] = qq + 2 * cc
    om[1, 2] = 2 * (cd - ab)
    om[2, 0] = 2 * (bd - ac)
    om[2, 1] = 2 * (cd + ab)
    om[2, 2] = qq + 2 * dd

    return om


@nb.njit("float64[:, :, :](float64[:, :])", cache=True, fastmath=True, nogil=True)
def qu2om_2d(qu: np.ndarray) -> np.ndarray:
    """Convert multiple unit quaternions to orthogonal rotation
    matrices.

    Parameters
    ----------
    qu
        2D array of n (q0, q1, q2, q3) quaternions as 64-bit floats.

    Returns
    -------
    om
        3D array of n (3, 3) rotation matrices as 64-bit floats.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    n = qu.shape[0]
    om = np.zeros((n, 3, 3), dtype=np.float64)
    for i in nb.prange(n):
        om[i, :, :] = qu2om_single(qu[i])
    return om


def qu2om(qu: np.ndarray) -> np.ndarray:
    """N-dimensional wrapper for om2qu_3d, see the docstring of that
    function.
    """
    qu2d = qu.reshape(-1, 4)
    qu2d = qu2d.astype(np.float64)

    om = qu2om_2d(qu2d)
    om = om.reshape(qu.shape[:-1] + (3, 3))

    return om


@nb.njit("float64[:](float64[:])", cache=True, fastmath=True, nogil=True)
def qu2ho_single(qu: np.ndarray) -> np.ndarray:
    """Convert a single (un)normalized quaternion to a normalized
    homochoric vector :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    qu
        1D array of (a, b, c, d) as 64-bit floats.

    Returns
    -------
    ho
        1D array of (x, y, z) as 64-bit floats.

    Notes
    -----
    Uses Eq. A.25 in :cite:`rowenhorst2015consistent`.

    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    omega = 2 * np.arccos(qu[0])

    if omega < constants.eps9:
        return np.zeros(3, dtype=np.float64)

    s = np.sqrt(np.sum(np.square(qu[1:])))
    n = qu[1:] / s
    f = 3 * (omega - np.sin(omega)) / 4
    ho = n * f ** (1 / 3)

    return ho


@nb.njit("float64[:, :](float64[:, :])", cache=True, fastmath=True, nogil=True)
def qu2ho_2d(qu: np.ndarray) -> np.ndarray:
    """Convert multiple (un)normalized quaternions to normalized
    homochoric vectors :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    qu
        2D array of n (a, b, c, d) as 64-bit floats.

    Returns
    -------
    ho
        2D array of n (x, y, z) as 64-bit floats.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    n = qu.shape[0]
    ho = np.zeros((n, 3), dtype=np.float64)
    for i in nb.prange(n):
        ho[i] = qu2ho_single(qu[i])
    return ho


def qu2ho(qu: np.ndarray) -> np.ndarray:
    """N-dimensional wrapper for qu2ho_2d, see the docstring of that
    function for further details.

    Parameters
    ----------
    qu
        Quaternion(s) (a, b, c, d) with the final array dimension equal
        to 4.

    Returns
    -------
    ho
        Homochoric vectors (x, y, z).
    """
    qu_nd = np.atleast_2d(qu)

    if qu_nd.shape[-1] != 4:
        raise ValueError("Final dimension of quaternion array must be 4.")

    qu2d = qu.reshape(-1, 4)
    qu2d = qu2d.astype(np.float64)

    ho = qu2ho_2d(qu2d)
    ho = ho.reshape(qu.shape[:-1] + (3,))

    return ho
