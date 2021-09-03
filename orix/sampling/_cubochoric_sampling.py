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

import numpy as np


BP = np.array(
    [
        0,
        1,
        0.577350269189626,
        0.414213562373095,
        0,
        0.267949192431123,
        0,
        0.198912367379658,
        0,
        0.158384440324536,
        0,
        0.131652497587396,
    ]
)


def is_inside_cubic_region(ro, symmetry="octahedral"):
    if np.isinf(ro[3]):
        return False
    r = ro[:3] * ro[3]
    if symmetry.lower() == "octahedral":
        c1 = np.max(abs(r)) - BP[3] <= 1e-8
    else:
        c1 = True
    c2 = np.sum(abs(r)) - 1 <= 1e-8
    return c1 * c2


def cu2ro(xyz):
    """Convert cubochoric to Rodrigues."""
    if np.max(abs(xyz)) == 0:
        return np.array([0, 0, 1, 0])
    else:
        return ho2ro(cu2ho(xyz))


def cu2ho(xyz):
    """Map from 3D cubic grid to 3D ball, LambertCubeToBall."""
    if np.max(abs(xyz)) > (np.pi ** (2 / 3) / 2) + 1e-8:
        return np.array([0] * 3)

    # Determine which pyramid pair the point lies in and copy coordinates
    # in correct order (see paper)
    p = get_pyramid(xyz)
    if p in [1, 2]:
        xyz2 = xyz
    elif p in [3, 4]:
        xyz2 = np.roll(xyz, -1)
    elif p in [5, 6]:
        xyz2 = np.roll(xyz, 1)
    else:
        raise ValueError

    # Scale by the grid parameter ratio
    xyz3 = xyz2 * np.pi ** (1 / 6) / 6 ** (1 / 6)
    x, y, z = xyz3

    # Transform to the sphere grid via the curved square, and intercept
    # the zero point
    if np.max(abs(xyz3)) == 0:
        lam_xyz = np.array([0, 0, 0])
    else:
        # Intercept all points along the z-axis
        if np.max(abs(xyz3[:2])) == 0:
            lam_xyz = np.array([0, 0, np.sqrt(6 / np.pi) * z])
        else:  # This is a general grid point
            prek = (
                (3 * np.pi / 4) ** (1 / 3)
                * 2 ** (1 / 4)
                / (np.pi ** (5 / 6) / 6 ** (1 / 6) / 2)
            )
            sqrt2 = np.sqrt(2)
            if abs(y) <= abs(x):
                q = (np.pi / 12) * y / x
                c = np.cos(q)
                s = np.sin(q)
                q = prek * x / np.sqrt(sqrt2 - c)
                T1 = (sqrt2 * c - 1) * q
                T2 = sqrt2 * s * q
            else:
                q = (np.pi / 12) * x / y
                c = np.cos(q)
                s = np.sin(q)
                q = prek * y / np.sqrt(sqrt2 - c)
                T1 = sqrt2 * s * q
                T2 = (sqrt2 * c - 1) * q
            # Transform to the sphere grid (inverse Lambert)
            c = T1 ** 2 + T2 ** 2
            s = np.pi * c / (24 * z ** 2)
            c = np.sqrt(np.pi) * c / np.sqrt(24) / z
            q = np.sqrt(1 - s)
            lam_xyz = np.array([T1 * q, T2 * q, np.sqrt(6 / np.pi) * z - c])

    # Reverse the coordinates back to the regular order according to the original
    # pyramid number
    if p in [1, 2]:
        return lam_xyz  # Homochoric
    elif p in [3, 4]:
        return np.roll(lam_xyz, 1)
    elif p in [5, 6]:
        return np.roll(lam_xyz, -1)


def get_pyramid(xyz):
    """Determine to which pyramid a point in a cubic grid belongs."""
    x, y, z = xyz
    if (abs(x) <= z) and (abs(y) <= z):
        return 1
    elif (abs(x) <= -z) and (abs(y) <= -z):
        return 2
    elif (abs(z) <= x) and (abs(y) <= x):
        return 3
    elif (abs(z) <= -x) and (abs(y) <= -x):
        return 4
    elif (abs(x) <= y) and (abs(z) <= y):
        return 5
    elif (abs(x) <= -y) and (abs(z) <= -y):
        return 6


tfit = np.array(
    [
        0.9999999999999968,
        -0.49999999999986866,
        -0.025000000000632055,
        -0.003928571496460683,
        -0.0008164666077062752,
        -0.00019411896443261646,
        -0.00004985822229871769,
        -0.000014164962366386031,
        -1.9000248160936107e-6,
        -5.72184549898506e-6,
        7.772149920658778e-6,
        -0.00001053483452909705,
        9.528014229335313e-6,
        -5.660288876265125e-6,
        1.2844901692764126e-6,
        1.1255185726258763e-6,
        -1.3834391419956455e-6,
        7.513691751164847e-7,
        -2.401996891720091e-7,
        4.386887017466388e-8,
        -3.5917775353564864e-9,
    ]
)


def ho2ro(ho):
    """Homochoric to Rodrigues."""
    return ax2ro(ho2ax(ho))


def ho2ax(ho):
    """Homochoric to axis-angle pair, ho2ax."""
    # Normalize ho and store the magnitude
    homag = np.sum(ho ** 2)
    if np.allclose(homag, 0, atol=1e-8):
        return np.array([0, 0, 1, 0])
    else:
        # Convert the magnitude to the rotation angle
        # s = tfit[0] + tfit[1] * homag + np.sum(tfit[2:] * homag**np.arange(2, 21))
        hom = homag
        s = tfit[0] + tfit[1] * hom
        for i in range(2, 21):
            hom *= homag
            s += tfit[i] * hom

        hon = ho / np.sqrt(homag)
        s = 2 * np.arccos(s)
        if abs(s - np.pi) < 1e-6:
            return np.append(hon, np.pi)
        else:
            return np.append(hon, s)


def ax2ro(ax):
    """Axis-angle pair to Rodrigues, ax2ro."""
    ro = np.zeros(4)
    if np.allclose(ax[3], 0, atol=1 - 8):
        ro[2] = 1
        return ro

    ro[:3] = ax[:3]
    # Need to deal with the 180 degree case
    if abs(ax[3] - np.pi) < 1e-7:
        ro[3] = np.inf
    else:
        ro[3] = np.tan(ax[3] * 0.5)

    return ro


def ro2eu(ro):
    """Convert a Rodrigues vector to a Euler angle triplet."""
    eu = om2eu(ro2om(ro))
    pivals = np.arange(1, 5) * np.pi / 2
    for i in range(3):
        if abs(eu[i] < 1e-12):
            eu[i] = 0
        for j in range(4):
            if abs(eu[i] - pivals[j]) < 1e-12:
                eu[i] = pivals[j]
    return eu


def ro2om(ro):
    """Convert a Rodrigues vector to an orientation matrix."""
    return ax2om(ro2ax(ro))


def om2eu(om):
    """Convert an orientation matrix to a Euler angle triplet."""
    if np.allclose(abs(om[2, 2]), 1):
        # We arbitrarily assign the entire angle to phi1
        if np.allclose(om[2, 2], 1):
            phi1 = np.arctan2(om[0, 1], om[0, 0])
            Phi = 0
            phi2 = 0
        else:
            phi1 = -np.arctan2(-om[0, 1], om[0, 0])
            Phi = np.pi
            phi2 = 0
    else:
        zeta = 1 / np.sqrt(1 - om[2, 2] ** 2)
        phi1 = np.arctan2(om[2, 0] * zeta, -om[2, 1] * zeta)
        Phi = np.arccos(om[2, 2])
        phi2 = np.arctan2(om[0, 2] * zeta, om[1, 2] * zeta)

    # REduce Euler angles to definition ranges (and positive values only)
    if phi1 < 0:
        phi1 = np.mod(phi1 + 100 * np.pi, 2 * np.pi)
    if Phi < 0:
        Phi = np.mod(Phi + 100 * np.pi, np.pi)
    if phi2 < 0:
        phi2 = np.mod(phi2 + 100 * np.pi, 2 * np.pi)
    return np.array([phi1, Phi, phi2])


def ro2ax(ro):
    """Convert a Rodrigues vector to an axis-angle pair."""
    if np.round(ro[3], 8) == 0:
        return np.array([0, 0, 1, 0])
    if np.isinf(ro[3]):
        return np.append(ro[:3], np.pi)
    else:
        vec_norm = np.sqrt(np.sum(np.square(ro[:3]), axis=-1))
        return np.append(ro[:3] / vec_norm, 2 * np.arctan(ro[3]))


def ax2om(ax):
    """Convert an axis-angle pair to an orientation matrix."""
    c_ax3 = np.cos(ax[3])
    s_ax3 = np.sin(ax[3])
    omc = 1 - c_ax3
    om = np.eye(3) * (ax[:3] ** 2 * omc + c_ax3)

    q1 = omc * ax[0] * ax[1]
    om[0, 1] = q1 + s_ax3 * ax[2]
    om[1, 0] = q1 - s_ax3 * ax[2]

    q2 = omc * ax[1] * ax[2]
    om[1, 2] = q2 + s_ax3 * ax[0]
    om[2, 1] = q2 - s_ax3 * ax[0]

    q3 = omc * ax[2] * ax[0]
    om[2, 0] = q3 + s_ax3 * ax[1]
    om[0, 2] = q3 - s_ax3 * ax[1]

    return om.T  # Transpose back when writing to file!


def ax2qu(ax):
    """Convert an axis-angle pair to a quaternion."""
    if np.round(ax[3], 8) == 0:
        return np.array([1, 0, 0, 0])
    else:
        c = np.cos(ax[3] * 0.5)
        s = np.sin(ax[3] * 0.5)
        return np.append(c, ax[:3] * s)


def ho2qu(ho):
    """Convert a Homochoric vector to a quaternion."""
    return ax2qu(ho2ax(ho))


def _cubochoric_sampling(nsteps):
    sedge = 0.5 * np.pi ** (2 / 3)
    delta = sedge / nsteps
    n = 0
    this_xyz = np.zeros(3)
    n_iterations = (2 * nsteps + 1) ** 3
    ori = np.zeros((n_iterations, 4), dtype=np.float64)
    for i in range(-nsteps + 1, nsteps + 1):
        this_xyz[0] = i * delta
        for j in range(-nsteps + 1, nsteps + 1):
            this_xyz[1] = j * delta
            for k in range(-nsteps + 1, nsteps + 1):
                this_xyz[2] = k * delta
                # Make sure that the point lies inside the cubochoric cell
                if np.max(abs(this_xyz)) > sedge:
                    continue
                # Get Rodrigues vector from Cubochoric coordinates
                this_ro = cu2ro(this_xyz)
                # Does the vector lie inside the fundamental zone?
                # Only implemented for cubic zone
                if is_inside_cubic_region(this_ro):
                    ori[n] = this_ro
                    n += 1
    return ori[:n]


# Rodrigues
# np.savetxt(
#    "/home/hakon/kode/orix_test_gridding/pg32_n100_orix_rod.txt",
#    ori_to_keep,
#    fmt="%17.9f",
# )
