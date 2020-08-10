# -*- coding: utf-8 -*-
# Copyright 2018-2020 the orix developers
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

import pytest
import numpy as np
import itertools

from orix.vector.neo_euler import AxAngle
from orix.vector import Vector3d
from orix.quaternion.rotation import Rotation


axes = [
    (1, 0, 0),
    (1, 1, 0),
    (2, -1, 0),
    (0, 2, 1),
    (-1, -1, -1),
]

angles = [
    -2 * np.pi,
    -5 * np.pi / 6,
    -np.pi / 3,
    0,
    np.pi / 12,
    np.pi / 3,
    3 * np.pi / 4,
    2 * np.pi,
    np.pi / 7,
]

axangles = [np.array(angle) * Vector3d(axis).unit for axis in axes for angle in angles]
axangles += [
    np.array(angle) * Vector3d(axis).unit
    for axis in itertools.combinations_with_replacement(axes, 2)
    for angle in itertools.combinations_with_replacement(angles, 2)
]


@pytest.fixture(params=axangles[:100])
def axangle(request):
    return AxAngle(request.param.data)


def test_angle(axangle):
    assert np.allclose(axangle.angle.data, axangle.norm.data)


def test_axis(axangle):
    assert axangle.axis.shape == axangle.shape


@pytest.mark.parametrize(
    "axis, angle, expected_axis",
    [
        ((2, 1, 1), np.pi / 4, (0.816496, 0.408248, 0.408248)),
        (Vector3d((2, 0, 0)), -2 * np.pi, (-1, 0, 0)),
    ],
)
def test_from_axes_angles(axis, angle, expected_axis):
    ax = AxAngle.from_axes_angles(axis, angle)
    assert np.allclose(ax.axis.data, expected_axis)
    assert np.allclose(ax.angle.data, abs(angle))


@pytest.mark.parametrize(
    "rotation, expected",
    [(Rotation([1, 0, 0, 0]), [0, 0, 0]), (Rotation([0, 1, 0, 0]), [np.pi, 0, 0])],
)
def test_from_rotation(rotation, expected):
    axangle = AxAngle.from_rotation(rotation)
    assert np.allclose(axangle.data, expected)
