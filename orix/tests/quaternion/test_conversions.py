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

from orix.quaternion import Rotation
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
    ro2ax,
    ro2ax_2d,
    ro2ax_single,
)


@pytest.fixture
def cubochoric_coordinates():
    return np.array(
        [
            [np.pi ** (2 / 3) / 2 + 1e-7, 1, 1],
            [0, 0, 0],
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
    return np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
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
    return np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 1, 0],
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
    return np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 1, 0],
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
    return np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [
                [-0.95541, -0.29525, 0.0],
                [0.29525, -0.95541, 0.0],
                [0.0, 0.0, 1.0],
            ],
            [
                [0.89058, -0.41522, 0.18558],
                [0.43956, 0.89058, -0.11678],
                [-0.11678, 0.18558, 0.97566],
            ],
            [
                [0.89058, 0.43956, 0.11678],
                [-0.41522, 0.89058, -0.18558],
                [-0.18558, 0.11678, 0.97566],
            ],
            [
                [0.92770, 0.06746, 0.36716],
                [0.32236, 0.35124, -0.87903],
                [-0.1882, 0.93385, 0.30410],
            ],
            [
                [0.92770, 0.06746, -0.36716],
                [0.32236, 0.35124, 0.87903],
                [0.18827, -0.93385, 0.30410],
            ],
            [
                [0.35124, 0.06746, 0.93385],
                [0.32236, 0.92770, -0.1882],
                [-0.87903, 0.36716, 0.30410],
            ],
            [
                [0.35124, -0.32236, -0.87903],
                [-0.06746, 0.92770, -0.36716],
                [0.93385, 0.18827, 0.30410],
            ],
        ]
    )


@pytest.fixture
def quaternions_conversions():
    return np.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
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


class TestRotationConversions:
    """Tests of conversions between rotation representations.

    Functions are accelerated with Numba, so both the Numba and Python
    versions of the function are tested.
    """

    def test_eu2qu(self):
        eulers = np.arange(30).reshape(10, 3) / 30
        eulers[0] = [1, 2, 3]  # Covers case where q0 < 0
        rots_np = Rotation.from_euler(eulers).data
        # single
        assert np.allclose(rots_np[0], eu2qu_single.py_func(eulers[0]))
        # 2d
        assert np.allclose(rots_np, eu2qu_2d.py_func(eulers), atol=1e-4)
        # nd
        assert np.allclose(rots_np, eu2qu(eulers), atol=1e-4)

    def test_get_pyramid(self, cubochoric_coordinates):
        """Cubochoric coordinates situated in expected pyramid."""
        pyramid = [3, 1, 1, 1, 2, 3, 4, 5, 6]
        # single
        for xyz, p in zip(cubochoric_coordinates, pyramid):
            assert get_pyramid_single.py_func(xyz) == p
        # 2d
        assert all(get_pyramid_2d.py_func(cubochoric_coordinates) == pyramid)
        # nd
        assert all(get_pyramid(cubochoric_coordinates) == pyramid)

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

    def test_ax2ro(self, axis_angle_pairs, rodrigues_vectors):
        # single
        for ax, ro in zip(axis_angle_pairs, rodrigues_vectors):
            assert np.allclose(ax2ro_single.py_func(ax), ro, atol=1e-4)
        # 2d
        assert np.isinf(ax2ro_single.py_func(np.array([0, 0, 0, np.pi]))[3])
        assert np.allclose(
            ax2ro_2d.py_func(axis_angle_pairs), rodrigues_vectors, atol=1e-4
        )
        # nd
        assert np.allclose(ax2ro(axis_angle_pairs), rodrigues_vectors, atol=1e-4)
        # nd_float32
        axang_32 = axis_angle_pairs.astype(np.float32)
        rod_32 = rodrigues_vectors.astype(np.float32)
        assert np.allclose(ax2ro(axang_32), rod_32, atol=1e-4)

    def test_ro2ax(self, rodrigues_vectors, axis_angle_pairs):
        # single
        for ro, ax in zip(rodrigues_vectors, axis_angle_pairs):
            assert np.allclose(ro2ax_single.py_func(ro), ax, atol=1e-4)
        assert ro2ax_single.py_func(np.array([0, 0, 0, np.inf]))[3] == np.pi
        # 2d
        assert np.allclose(
            ro2ax_2d.py_func(rodrigues_vectors), axis_angle_pairs, atol=1e-4
        )
        # nd
        assert np.allclose(ro2ax(rodrigues_vectors), axis_angle_pairs, atol=1e-4)
        # nd_float32
        axang_32 = axis_angle_pairs.astype(np.float32)
        rod_32 = rodrigues_vectors.astype(np.float32)
        assert np.allclose(ro2ax(rod_32), axang_32, atol=1e-4)

    def test_ax2qu(self, axis_angle_pairs, quaternions_conversions):
        # single
        for ax, qu in zip(axis_angle_pairs, quaternions_conversions):
            assert np.allclose(ax2qu_single.py_func(ax), qu, atol=1e-4)
        # 2d
        assert np.allclose(
            ax2qu_2d.py_func(axis_angle_pairs), quaternions_conversions, atol=1e-4
        )
        # nd
        qu = ax2qu(axis_angle_pairs[:, :3], axis_angle_pairs[:, 3])
        assert np.allclose(qu, quaternions_conversions, atol=1e-4)
        # nd_float32
        ax_32 = axis_angle_pairs[:, :3].astype(np.float32)
        ang_32 = axis_angle_pairs[:, 3].astype(np.float32)
        qu_32 = quaternions_conversions.astype(np.float32)
        assert np.allclose(ax2qu(ax_32, ang_32), qu_32, atol=1e-4)

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

    def test_om2qu(self, orthogonal_matrices, quaternions_conversions):
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
