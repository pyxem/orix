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

import pytest
import numpy as np

from orix.vector.neo_euler import Rodrigues, Homochoric
from orix.quaternion.rotation import Rotation


# Rodrigues


@pytest.mark.parametrize(
    "rotation, expected",
    [
        (Rotation([1, 0, 0, 0]), [0, 0, 0]),
        (Rotation([0.9239, 0.2209, 0.2209, 0.2209]), [0.2391, 0.2391, 0.2391]),
    ],
)
def test_from_rotation(rotation, expected):
    rodrigues = Rodrigues.from_rotation(rotation)
    assert np.allclose(rodrigues.data, expected, atol=1e-4)


@pytest.mark.parametrize(
    "rodrigues, expected",
    [(Rodrigues([0.2391, 0.2391, 0.2391]), np.pi / 4)],
)
def test_angle(rodrigues, expected):
    angle = rodrigues.angle
    assert np.allclose(angle.data, expected, atol=1e-3)


# Homochoric


@pytest.mark.parametrize(
    "rotation", [Rotation([1, 0, 0, 0]), Rotation([0.9239, 0.2209, 0.2209, 0.2209])]
)
def test_homochoric_from_rotation(rotation):
    _ = Homochoric.from_rotation(rotation)


@pytest.mark.parametrize(
    "rotation", [Rotation([1, 0, 0, 0]), Rotation([0.9239, 0.2209, 0.2209, 0.2209])]
)
@pytest.mark.xfail(strict=True, reason=AttributeError)
def test_homochoric_angle(rotation):
    h = Homochoric.from_rotation(rotation)
    _ = h.angle
