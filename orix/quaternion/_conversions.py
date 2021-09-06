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

"""Conversions of orientations between many common representations from
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
def get_pyramid_single(xyz):
    """Determine to which out of six pyramids in the cube a (x, y, z)
    coordinate belongs.

    Parameters
    ----------
    xyz : numpy.ndarray
        1D array (x, y, z) of 64-bit floats.

    Returns
    -------
    pyramid : int
        Which pyramid `xyz` belongs to as a 64-bit integer.

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


@nb.jit("float64[:](float64[:])", cache=True, nogil=True, nopython=True)
def cu2ho_single(cu):
    """Conversion from a single set of cubochoric coordinates to
    un-normalized homochoric coordinates :cite:`singh2016orientation`.

    Parameters
    ----------
    cu : numpy.ndarray
        1D array of (x, y, z) as 64-bit floats.

    Returns
    -------
    ho : numpy.ndarray
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
    """Conversion from multiple cubochoric coordinates to un-normalized
    homochoric coordinates :cite:`singh2016orientation`.

    Parameters
    ----------
    cu : numpy.ndarray
        2D array of n (x, y, z) as 64-bit floats.

    Returns
    -------
    ho : numpy.ndarray
        2D array of n (x, y, z) as 64-bit floats.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    ho = np.zeros_like(cu)
    for i in nb.prange(cu.shape[0]):
        ho[i] = cu2ho_single(cu[i])
    return ho


@nb.jit("float64[:](float64[:])", cache=True, nogil=True, nopython=True)
def ho2ax_single(ho):
    """Conversion from a single set of homochoric coordinates to an
    un-normalized axis-angle pair :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    ho : numpy.ndarray
        1D array of (x, y, z) as 64-bit floats.

    Returns
    -------
    ax : numpy.ndarray
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
    ho_magnitude = np.sum(ho ** 2)
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
def ho2ax(ho):
    """Conversion from multiple homochoric coordinates to un-normalized
    axis-angle pairs :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    ho : numpy.ndarray
        2D array of n (x, y, z) as 64-bit floats.

    Returns
    -------
    ho : numpy.ndarray
        2D array of n (x, y, z) as 64-bit floats.

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


@nb.jit("float64[:](float64[:])", cache=True, nogil=True, nopython=True)
def ax2ro_single(ax):
    """Conversion from a single angle-axis pair to an un-normalized
    Rodrigues vector :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    ax : numpy.ndarray
        1D array of (x, y, z, angle) as 64-bit floats.

    Returns
    -------
    ro : numpy.ndarray
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
def ax2ro(ax):
    """Conversion from multiple axis-angle pairs to un-normalized
    Rodrigues vectors :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    ax : numpy.ndarray
        2D array of n (x, y, z, angle) as 64-bit floats.

    Returns
    -------
    ro : numpy.ndarray
        2D array of n (x, y, z, angle) as 64-bit floats.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    ro = np.zeros_like(ax)
    for i in nb.prange(ax.shape[0]):
        ro[i] = ax2ro_single(ax[i])
    return ro


@nb.jit("float64[:](float64[:])", cache=True, nogil=True, nopython=True)
def ro2ax_single(ro):
    """Conversion from a single Rodrigues vector to an un-normalized
    axis-angle pair :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    ro : numpy.ndarray
        1D array of (x, y, z, angle) as 64-bit floats.

    Returns
    -------
    ax : numpy.ndarray
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
def ro2ax(ro):
    """Conversion from multiple Rodrigues vectors to un-normalized
    axis-angle pairs :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    ro : numpy.ndarray
        2D array of n (x, y, z, angle) as 64-bit floats.

    Returns
    -------
    ax : numpy.ndarray
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


@nb.jit("float64[:](float64[:])", cache=True, nogil=True, nopython=True)
def ax2qu_single(ax):
    """Conversion from a single axis-angle pair to an un-normalized
    quaternion :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    ax : numpy.ndarray
        1D array of (x, y, z, angle) as 64-bit floats.

    Returns
    -------
    qu : numpy.ndarray
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
def ax2qu(ax):
    """Conversion from multiple axis-angle pairs to un-normalized
    quaternions :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    ax : numpy.ndarray
        2D array of n (x, y, z, angle) as 64-bit floats.

    Returns
    -------
    qu : numpy.ndarray
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


@nb.jit("float64[:](float64[:])", cache=True, nogil=True, nopython=True)
def ho2ro_single(ho):
    """Conversion from a single set of homochoric coordinates to an
    un-normalized Rodrigues vector :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    ho : numpy.ndarray
        1D array of (x, y, z) as 64-bit floats.

    Returns
    -------
    ro : numpy.ndarray
        1D array of (x, y, z, angle) as 64-bit floats.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    return ax2ro_single(ho2ax_single(ho))


@nb.jit("float64[:, :](float64[:, :])", cache=True, nogil=True, nopython=True)
def ho2ro(ho):
    """Conversion from multiple homochoric coordinates to un-normalized
    Rodrigues vectors :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    ho : numpy.ndarray
        2D array of n (x, y, z) as 64-bit floats.

    Returns
    -------
    ax : numpy.ndarray
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


@nb.jit("float64[:](float64[:])", cache=True, nogil=True, nopython=True)
def cu2ro_single(cu):
    """Conversion from a single set of cubochoric coordinates to an
    un-normalized Rodrigues vector :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    cu : numpy.ndarray
        1D array of (x, y, z) as 64-bit floats.

    Returns
    -------
    ro : numpy.ndarray
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
def cu2ro(cu):
    """Conversion from multiple cubochoric coordinates to un-normalized
    Rodrigues vectors :cite:`rowenhorst2015consistent`.

    Parameters
    ----------
    cu : numpy.ndarray
        2D array of n (x, y, z) as 64-bit floats.

    Returns
    -------
    ro : numpy.ndarray
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


@nb.jit("float64[:](float64, float64, float64)", cache=True, nogil=True, nopython=True)
def eu2qu_single(alpha, beta, gamma):
    """Convert three Euler angles (alpha, beta, gamma) to a unit
    quaternion.

    Parameters
    ----------
    alpha, beta, gamma : float
        Euler angles in the Bunge convention in radians as 64-bit
        floats.

    Returns
    -------
    qu : numpy.ndarray
        1D unit quaternion (a, b, c, d) as 64-bit floats.

    Notes
    -----
    Uses Eqs. A.5 & A.6 :cite:`rowenhorst2015consistent`.

    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    sigma = 0.5 * np.add(alpha, gamma)
    delta = 0.5 * np.subtract(alpha, gamma)
    c = np.cos(beta / 2)
    s = np.sin(beta / 2)

    qu = np.zeros(4, dtype=np.float64)
    qu[0] = c * np.cos(sigma)
    qu[1] = -s * np.cos(delta)
    qu[2] = -s * np.sin(delta)
    qu[3] = -c * np.sin(sigma)

    if qu[0] < 0:
        qu = -qu

    return qu
