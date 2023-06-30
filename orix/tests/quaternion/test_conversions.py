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

import numpy as np
import pytest

from orix.quaternion import Orientation, Quaternion, Rotation
from orix.quaternion.symmetry import C1, Oh
from orix.quaternion._conversions import (
    ax2qu,
    ax2qu_2d,
    ax2qu_single,
    ax2ro,
    ax2ro_2d,
    ax2ro_single,
    cu2ho,
    cu2ho_2d,
    cu2ho_single,
    cu2ro,
    cu2ro_2d,
    cu2ro_single,
    eu2qu,
    eu2qu_2d,
    eu2qu_single,
    get_pyramid,
    get_pyramid_2d,
    get_pyramid_single,
    ho2ax,
    ho2ax_2d,
    ho2ax_single,
    ho2ro,
    ho2ro_2d,
    ho2ro_single,
    om2qu,
    om2qu_3d,
    om2qu_single,
    qu2ax,
    qu2ax_2d,
    qu2ax_single,
    qu2eu,
    qu2eu_2d,
    qu2eu_single,
    qu2om,
    qu2om_2d,
    qu2om_single,
    ro2ax,
    ro2ax_2d,
    ro2ax_single,
)


# NOTE to future test writers on unittest data:
# All the data below can be recreated using 3Drotations, which is available
# at https://github.com/marcdegraef/3Drotations/blob/master/src/python
# 3Drotations is an expanded implementation of the rotation conversions
# laid out in 2015 Rowenhorst et et. al., written by a subset of the
# original authors.
# Note, however, that orix differs from 3Drotations in its handling of some
# edge cases. Software using 3Drotations (for example, Dream3D abd EBSDLib),
# handle rounding and other corrections after converting, whereas Orix
# accounts for them during. The first three angles in each set are tests of
# these edge cases, and will therefore differ between orix and 3Drotations.
# For all other angles, the datasets can be recreated using a variation of:
#   np.around(rotlib.qu2{insert_new_representation_here}(qu),4)
# this consistently gives results with 4 decimal of accuracy.
@pytest.fixture
def cubochoric_coordinates():
    return np.array(
        [
            [np.pi ** (2 / 3) / 2 + 1e-7, 1, 1],
            [0, 0, 0],
            [1.0725, 0, 0],
            [0, 0, 1],
            [0.1, 0.1, 0.2],
            [0.1, 0.1, -0.2],
            [0.5, 0.2, 0.1],
            [-0.5, -0.2, 0.1],
            [0.2, 0.5, 0.1],
            [0.2, -0.5, 0.1],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def homochoric_vectors():
    # np.around(rotlib.qu2ho(qu),4)
    return np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [1.3307, 0, 0],
            [0, 0, 1.2407],
            [0.0785, 0.0785, 0.2219],
            [0.0785, 0.0785, -0.2219],
            [0.5879, 0.1801, 0.0827],
            [-0.5879, -0.1801, 0.0827],
            [0.1801, 0.5879, 0.0827],
            [0.1801, -0.5879, 0.0827],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def axis_angle_pairs():
    # np.around(rotlib.qu2ax(qu),4)
    return np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [1, 0, 0, np.pi],
            [0, 0, 1, 2.8418],
            [0.3164, 0.3164, 0.8943, 0.4983],
            [0.3164, 0.3164, -0.8943, 0.4983],
            [0.9476, 0.2903, 0.1333, 1.2749],
            [-0.9476, -0.2903, 0.1333, 1.2749],
            [0.2903, 0.9476, 0.1333, 1.2749],
            [0.2903, -0.9476, 0.1333, 1.2749],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def rodrigues_vectors():
    # np.around(rotlib.qu2ro(qu),4)
    return np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [1, 0, 0, np.inf],
            [0, 0, 1, 6.6212],
            [0.3164, 0.3164, 0.8943, 0.2544],
            [0.3164, 0.3164, -0.8943, 0.2544],
            [0.9476, 0.2903, 0.1333, 0.7406],
            [-0.9476, -0.2903, 0.1333, 0.7406],
            [0.2903, 0.9476, 0.1333, 0.7406],
            [0.2903, -0.9476, 0.1333, 0.7406],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def orthogonal_matrices():
    # np.around(rotlib.qu2om(qu),4)
    return np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            [
                [-0.9554, -0.2953, 0.0000],
                [0.2953, -0.9554, 0.0000],
                [0.0000, 0.0000, 1.0000],
            ],
            [
                [0.8906, -0.4152, 0.1856],
                [0.4396, 0.8906, -0.1168],
                [-0.1168, 0.1856, 0.9757],
            ],
            [
                [0.8906, 0.4396, 0.1168],
                [-0.4152, 0.8906, -0.1856],
                [-0.1856, 0.1168, 0.9757],
            ],
            [
                [0.9277, 0.0675, 0.3672],
                [0.3224, 0.3512, -0.879],
                [-0.1883, 0.9339, 0.3041],
            ],
            [
                [0.9277, 0.0675, -0.3672],
                [0.3224, 0.3512, 0.879],
                [0.1883, -0.9339, 0.3041],
            ],
            [
                [0.3512, 0.0675, 0.9339],
                [0.3224, 0.9277, -0.1883],
                [-0.879, 0.3672, 0.3041],
            ],
            [
                [0.3512, -0.3224, -0.879],
                [-0.0675, 0.9277, -0.3672],
                [0.9339, 0.1883, 0.3041],
            ],
        ]
    )


@pytest.fixture
def quaternions_conversions():
    return np.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0.1493, 0, 0, 0.9888],
            [0.9691, 0.0780, 0.0780, 0.2205],
            [0.9691, 0.0780, 0.0780, -0.2205],
            [0.8036, 0.5640, 0.1728, 0.0793],
            [0.8036, -0.5640, -0.1728, 0.0793],
            [0.8036, 0.1728, 0.5640, 0.0793],
            [0.8036, 0.1728, -0.5640, 0.0793],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def euler_angles():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 3.1416, 0.0],
            [3.4413, 0.0, 0.0],
            [3.7033, 0.2211, 2.1325],
            [4.1507, 0.2211, 2.5799],
            [3.3405, 1.2618, 2.7459],
            [0.1989, 1.2618, 5.8875],
            [4.3167, 1.2618, 1.7697],
            [1.7697, 1.2618, 4.3167],
        ],
        dtype=np.float64,
    )


class TestRotationConversions:
    """Tests of conversions between rotation representations.

    Functions are accelerated with Numba, so both the Numba and Python
    versions of the function are tested.
    """

    def test_eu2qu2eu(self, quaternions_conversions, euler_angles):
        qu_64 = quaternions_conversions
        eu_64 = euler_angles
        qu_32 = qu_64.astype(np.float32)
        eu_32 = eu_64.astype(np.float32)
        # single
        for qu, eu in zip(qu_64, eu_64):
            assert np.allclose(qu2eu_single.py_func(qu), eu, atol=1e-4)
            assert np.allclose(eu2qu_single.py_func(eu), qu, atol=1e-4)
        # 2d
        assert np.allclose(qu2eu_2d.py_func(qu_64), eu_64, atol=1e-4)
        assert np.allclose(eu2qu_2d.py_func(eu_64), qu_64, atol=1e-4)
        # nd and jit
        assert np.allclose(eu2qu(eu_64), qu_64, atol=1e-4)
        assert np.allclose(qu2eu(qu_64), eu_64, atol=1e-4)
        # nd_float32
        assert np.allclose(qu2eu(qu_32), eu_32, atol=1e-4)
        assert np.allclose(eu2qu(eu_32), qu_32, atol=1e-4)
        # Symmetry-preserving
        ori = Orientation.from_euler(eu_64[-1], Oh)
        assert np.all(ori._symmetry[0] == C1)
        assert np.all(ori._symmetry[1] == Oh)
        assert np.allclose(ori.data[0], qu_64[-1], atol=1e-4)

    def test_get_pyramid(self, cubochoric_coordinates):
        """Cubochoric coordinates situated in expected pyramid."""
        cu_64 = cubochoric_coordinates
        pyramid = [3, 1, 3, 1, 1, 2, 3, 4, 5, 6]
        # single
        for xyz, p in zip(cu_64, pyramid):
            assert get_pyramid_single.py_func(xyz) == p
        # 2d
        assert all(get_pyramid_2d.py_func(cu_64) == pyramid)
        # nd
        assert all(get_pyramid(cu_64) == pyramid)

    def test_cu2ho(self, cubochoric_coordinates, homochoric_vectors):
        # single
        for cu, ho in zip(cubochoric_coordinates, homochoric_vectors):
            assert np.allclose(cu2ho_single.py_func(cu), ho, atol=1e-4)
        # 2d
        assert np.allclose(
            cu2ho_2d.py_func(cubochoric_coordinates), homochoric_vectors, atol=1e-4
        )
        # nd
        assert np.allclose(cu2ho(cubochoric_coordinates), homochoric_vectors, atol=1e-4)
        # nd_float32
        cu_32 = cubochoric_coordinates.astype(np.float32)
        ho_32 = homochoric_vectors.astype(np.float32)
        assert np.allclose(cu2ho(cu_32), ho_32, atol=1e-4)

    def test_ho2ax(self, homochoric_vectors, axis_angle_pairs):
        # single
        for ho, ax in zip(homochoric_vectors, axis_angle_pairs):
            assert np.allclose(ho2ax_single.py_func(ho), ax, atol=1e-4)
        # 2d
        assert np.allclose(
            ho2ax_2d.py_func(homochoric_vectors), axis_angle_pairs, atol=1e-4
        )
        # nd
        assert np.allclose(ho2ax(homochoric_vectors), axis_angle_pairs, atol=1e-4)
        # nd_float32
        axang_32 = axis_angle_pairs.astype(np.float32)
        ho_32 = homochoric_vectors.astype(np.float32)
        assert np.allclose(ho2ax(ho_32), axang_32, atol=1e-4)

    def test_ax2ro2ax(
        self, axis_angle_pairs, rodrigues_vectors, quaternions_conversions
    ):
        ax_64 = axis_angle_pairs
        ro_64 = rodrigues_vectors
        ax_32 = ax_64.astype(np.float32)
        ro_32 = ro_64.astype(np.float32)
        # single
        for ax, ro in zip(ax_64, ro_64):
            assert np.allclose(ax2ro_single.py_func(ax), ro, atol=1e-4)
            assert np.allclose(ro2ax_single.py_func(ro), ax, atol=1e-4)
        # 2d
        assert np.isinf(ax2ro_single.py_func(np.array([0, 0, 0, np.pi]))[3])
        assert np.allclose(ax2ro_2d.py_func(ax_64), ro_64, atol=1e-4)
        assert np.allclose(ro2ax_2d.py_func(ro_64), ax_64, atol=1e-4)
        # nd
        assert np.allclose(ax2ro(ax_64), ro_64, atol=1e-4)
        assert np.allclose(ro2ax(ro_64), ax_64, atol=1e-4)
        # nd_float32
        assert np.allclose(ax2ro(ax_32), ro_32, atol=1e-4)
        assert np.allclose(ro2ax(ro_32), ax_32, atol=1e-4)
        # Test Quaternion class implementations
        quat1 = Quaternion(quaternions_conversions)
        quat2 = Quaternion.from_rodrigues_frank(ro_64)
        assert np.allclose(quat1.data, quat2.data, atol=1e-4)
        rf_from_quat = quat1.to_rodrigues_frank()
        assert np.allclose(rf_from_quat[4:], ro_64[4:], atol=1e-4)
        assert Quaternion.from_rodrigues_frank([]).size == 0
        # Test warnings for Quaternion class
        with pytest.raises(UserWarning, match="179.99"):
            Quaternion.from_rodrigues([1e15, 1e15, 1e10])
        with pytest.raises(UserWarning, match="Maximum"):
            Quaternion.from_rodrigues([0, 0, 1e-16])
        with pytest.raises(ValueError, match="rodrigues_frank"):
            Quaternion.from_rodrigues([1, 2, 3, 4])
        with pytest.raises(ValueError, match="instead"):
            Quaternion.from_rodrigues_frank([1, 2, 3])

    def test_ax2qu2ax(self, axis_angle_pairs, quaternions_conversions):
        ax_64 = axis_angle_pairs
        qu_64 = quaternions_conversions
        ax_32 = ax_64.astype(np.float32)
        qu_32 = qu_64.astype(np.float32)
        axis_32 = ax_32[:, :3]
        ang_32 = ax_32[:, 3]
        # single
        for ax, qu in zip(ax_64, qu_64):
            assert np.allclose(ax2qu_single.py_func(ax), qu, atol=1e-4)
            assert np.allclose(qu2ax_single.py_func(qu), ax, atol=2e-4)
        # test conversion of souther hemisphere quaternion
        southern_qu = np.array([-1, 1, 1, 0], dtype=np.float64)
        south_ax = ax2qu_single(southern_qu)
        assert np.allclose(south_ax, np.array([1, 0, 0, 0]), atol=1e-4)
        # 2d
        assert np.allclose(ax2qu_2d.py_func(ax_64), qu_64, atol=1e-4)
        assert np.allclose(qu2ax_2d.py_func(qu_64), ax_64, atol=2e-4)
        # nd
        assert np.allclose(ax2qu(ax_64[:, :3], ax_64[:, 3]), qu_64, atol=1e-4)
        axang = np.hstack(qu2ax(qu_64))
        assert np.allclose(axang, ax_64, atol=2e-4)
        # nd_float32
        assert np.allclose(ax2qu(axis_32, ang_32), qu_32, atol=1e-4)
        axang_32 = np.hstack(qu2ax(qu_32))
        assert np.allclose(axang_32, ax_32, atol=2e-4)
        # make sure bad data causes the expected errors
        with pytest.raises(ValueError, match="(...,3)"):
            ax2qu(axis_32.T, ang_32)
        with pytest.raises(ValueError, match="(6, 3)"):
            ax2qu(axis_32[:6,], ang_32[:8])
        with pytest.raises(ValueError, match="(...,4)"):
            qu2ax(qu_64[:, :3])
        # Check Quaternion and Orientation class features
        ori = Orientation.from_axes_angles(axis_32, ang_32, Oh)
        assert np.all(ori._symmetry[0] == C1)
        assert np.all(ori._symmetry[1] == Oh)
        degrees = Quaternion(qu_64).to_axes_angles(degrees=True)[1]
        assert np.allclose(np.deg2rad(degrees), ax_64[:, 3], atol=4)

    def test_ho2ro(self, homochoric_vectors, rodrigues_vectors):
        # single
        for ho, ro in zip(homochoric_vectors, rodrigues_vectors):
            assert np.allclose(ho2ro_single.py_func(ho), ro, atol=1e-4)
        # 2d
        assert np.allclose(
            ho2ro_2d.py_func(homochoric_vectors), rodrigues_vectors, atol=1e-4
        )
        # nd
        assert np.allclose(ho2ro(homochoric_vectors), rodrigues_vectors, atol=1e-4)
        # nd_float32
        ho_32 = homochoric_vectors.astype(np.float32)
        rod_32 = rodrigues_vectors.astype(np.float32)
        assert np.allclose(ho2ro(ho_32), rod_32, atol=1e-4)

    def test_cu2ro(self, cubochoric_coordinates, rodrigues_vectors):
        # single
        for cu, ro in zip(cubochoric_coordinates, rodrigues_vectors):
            assert np.allclose(cu2ro_single.py_func(cu), ro, atol=1e-4)
        # 2d
        assert np.allclose(
            cu2ro_2d.py_func(cubochoric_coordinates), rodrigues_vectors, atol=1e-4
        )
        # nd
        assert np.allclose(cu2ro(cubochoric_coordinates), rodrigues_vectors, atol=1e-4)
        # nd_float32
        cub_32 = cubochoric_coordinates.astype(np.float32)
        rod_32 = rodrigues_vectors.astype(np.float32)
        assert np.allclose(cu2ro(cub_32), rod_32, atol=1e-4)

    def test_om2qu2om(self, orthogonal_matrices, quaternions_conversions):
        # checks both om2qu and qu2om simultaneously
        om_64 = orthogonal_matrices
        qu_64 = quaternions_conversions
        om_32 = om_64.astype(np.float32)
        qu_32 = qu_64.astype(np.float32)
        # single
        for om, qu in zip(om_64, qu_64):
            assert np.allclose(om2qu_single.py_func(om), qu, atol=1e-4)
            assert np.allclose(qu2om_single.py_func(qu), om, atol=1e-4)
        # test edge cases where q[0] is at or near zero
        rot_pi_111 = np.array([[-1, 2, 2], [2, -1, 2], [2, 2, -1]]) / 3
        qu_from_om = om2qu_single.py_func(rot_pi_111)
        qu_actual = np.array([0, np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3)])
        om_from_qu = qu2om_single.py_func(qu_actual)
        assert np.allclose(qu_from_om, qu_actual)
        assert np.allclose(om_from_qu, rot_pi_111)
        # 2d
        assert np.allclose(om2qu_3d.py_func(om_64), qu_64, atol=1e-4)
        assert np.allclose(om_64, qu2om_2d.py_func(qu_64), atol=1e-4)
        # nd
        assert np.allclose(om2qu(om_64), qu_64, atol=1e-4)
        assert np.allclose(om_64, qu2om(qu_64), atol=1e-4)
        # nd_float32
        assert np.allclose(om2qu(om_32), qu_32, atol=1e-4)
        assert np.allclose(om_32, qu2om(qu_32), atol=1e-4)
        # Quaternion.from_matrix input checks
        quat = Quaternion.from_matrix(om_64).data
        assert np.allclose(quat, qu_64, atol=1e-4)
        with pytest.raises(ValueError, match="(3, 3)"):
            Quaternion.from_matrix(om_64[:, :, :2])
        with pytest.raises(ValueError, match="(3, 3)"):
            Quaternion.from_matrix(om_64[:, :2, :])

    def test_qu2om(self, orthogonal_matrices, quaternions_conversions):
        # single
        for om, qu in zip(orthogonal_matrices, quaternions_conversions):
            assert np.allclose(om2qu_single.py_func(om), qu, atol=1e-4)
        # test edge cases where q[0] is at or near zero
        rot_pi_111 = np.array([[-1, 2, 2], [2, -1, 2], [2, 2, -1]]) / 3
        qu_from_om = om2qu_single.py_func(rot_pi_111)
        qu_actual = np.array([0, np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3)])
        assert np.allclose(qu_from_om, qu_actual)
        # 2d
        assert np.allclose(
            om2qu_3d.py_func(orthogonal_matrices), quaternions_conversions, atol=1e-4
        )
        # nd
        assert np.allclose(
            om2qu(orthogonal_matrices), quaternions_conversions, atol=1e-4
        )
        # nd_float32
        om_32 = orthogonal_matrices.astype(np.float32)
        qu_32 = quaternions_conversions.astype(np.float32)
        assert np.allclose(om2qu(om_32), qu_32, atol=1e-4)

    def test_quaternion_shortcuts(
        self, rodrigues_vectors, homochoric_vectors, quaternions_conversions
    ):
        """These functions should evenutally be added to _conversions.py,
        but are currently done with shortcut functions in the Quaternion
        Class that use Quaternion.axis and Quaternion.angle."""
        quat = Quaternion(quaternions_conversions[3:])
        quat_hom = quat.to_homochoric().data
        quat_rod = quat.to_rodrigues().data
        hom_3 = homochoric_vectors[3:]
        rod_3 = rodrigues_vectors[3:, :3] * rodrigues_vectors[3:, 3:]
        assert np.allclose(hom_3, quat_hom, atol=1e-4)
        assert np.allclose(rod_3, quat_rod, atol=2e-3)
