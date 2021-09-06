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

import numpy as np
import pytest

from orix.quaternion import Rotation
from orix.quaternion._conversions import (
    ax2qu_single,
    ax2qu,
    ax2ro_single,
    ax2ro,
    cu2ho_single,
    cu2ho,
    cu2ro_single,
    cu2ro,
    eu2qu_single,
    ho2ax_single,
    ho2ax,
    ho2ro_single,
    ho2ro,
    get_pyramid_single,
    ro2ax_single,
    ro2ax,
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
    def test_rotation_from_euler(self):
        euler = np.array([1, 2, 3])
        rot_np = Rotation.from_euler(euler).data
        assert np.allclose(rot_np, eu2qu_single.py_func(*euler))

    def test_get_pyramid_single(self, cubochoric_coordinates):
        pyramid = [3, 1, 1, 1, 2, 3, 4, 5, 6]
        for xyz, p in zip(cubochoric_coordinates, pyramid):
            assert get_pyramid_single.py_func(xyz) == p

    def test_cu2ho_single(self, cubochoric_coordinates, homochoric_vectors):
        for cu, ho in zip(cubochoric_coordinates, homochoric_vectors):
            assert np.allclose(cu2ho_single.py_func(cu), ho, atol=1e-4)

    def test_cu2ho(self, cubochoric_coordinates, homochoric_vectors):
        assert np.allclose(
            cu2ho.py_func(cubochoric_coordinates), homochoric_vectors, atol=1e-4
        )

    def test_ho2ax_single(self, homochoric_vectors, axis_angle_pairs):
        for ho, ax in zip(homochoric_vectors, axis_angle_pairs):
            assert np.allclose(ho2ax_single.py_func(ho), ax, atol=1e-4)

    def test_ho2ax(self, homochoric_vectors, axis_angle_pairs):
        assert np.allclose(
            ho2ax.py_func(homochoric_vectors), axis_angle_pairs, atol=1e-4
        )

    def test_ax2ro_single(self, axis_angle_pairs, rodrigues_vectors):
        for ax, ro in zip(axis_angle_pairs, rodrigues_vectors):
            assert np.allclose(ax2ro_single.py_func(ax), ro, atol=1e-4)
        assert np.isinf(ax2ro_single.py_func(np.array([0, 0, 0, np.pi]))[3])

    def test_ax2ro(self, axis_angle_pairs, rodrigues_vectors):
        assert np.allclose(
            ax2ro.py_func(axis_angle_pairs), rodrigues_vectors, atol=1e-4
        )

    def test_ro2ax_single(self, rodrigues_vectors, axis_angle_pairs):
        for ro, ax in zip(rodrigues_vectors, axis_angle_pairs):
            assert np.allclose(ro2ax_single.py_func(ro), ax, atol=1e-4)
        assert ro2ax_single.py_func(np.array([0, 0, 0, np.inf]))[3] == np.pi

    def test_ro2ax(self, rodrigues_vectors, axis_angle_pairs):
        assert np.allclose(
            ro2ax.py_func(rodrigues_vectors), axis_angle_pairs, atol=1e-4
        )

    def test_ax2qu_single(self, axis_angle_pairs, quaternions_conversions):
        for ax, qu in zip(axis_angle_pairs, quaternions_conversions):
            assert np.allclose(ax2qu_single.py_func(ax), qu, atol=1e-4)

    def test_ax2qu(self, axis_angle_pairs, quaternions_conversions):
        assert np.allclose(
            ax2qu.py_func(axis_angle_pairs), quaternions_conversions, atol=1e-4
        )

    def test_ho2ro_single(self, homochoric_vectors, rodrigues_vectors):
        for ho, ro in zip(homochoric_vectors, rodrigues_vectors):
            assert np.allclose(ho2ro_single.py_func(ho), ro, atol=1e-4)

    def test_ho2ro(self, homochoric_vectors, rodrigues_vectors):
        assert np.allclose(
            ho2ro.py_func(homochoric_vectors), rodrigues_vectors, atol=1e-4
        )

    def test_cu2ro_single(self, cubochoric_coordinates, rodrigues_vectors):
        for cu, ro in zip(cubochoric_coordinates, rodrigues_vectors):
            assert np.allclose(cu2ro_single.py_func(cu), ro, atol=1e-4)

    def test_cu2ro(self, cubochoric_coordinates, rodrigues_vectors):
        assert np.allclose(
            cu2ro.py_func(cubochoric_coordinates), rodrigues_vectors, atol=1e-4
        )
