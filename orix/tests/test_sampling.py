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

from orix.quaternion.rotation import Rotation
from orix.quaternion.symmetry import C2, C6, D6, get_point_group
from orix.sampling.sampling_utils import uniform_SO3_sample
from orix.sampling.sample_generators import get_sample_fundamental, get_sample_local


@pytest.fixture(scope="session")
def sample():
    return uniform_SO3_sample(2)


@pytest.fixture(scope="session")
def fr():
    """ fixed rotation """
    r = Rotation([0.5, 0.5, 0, 0])
    return r


def test_uniform_SO3_sample_regions(sample, fr):
    """ Checks that different regions have the same density"""
    around_zero = sample[sample.a > 0.9]
    moved = fr * sample
    elsewhere = moved[sample.a > 0.9]
    # extra line simplifies the stacktrack
    x, y = around_zero.size, elsewhere.size
    assert np.isclose(x, y, rtol=0.025)


def test_uniform_SO3_sample_resolution(sample):
    """ Checks that doubling resolution doubles density (8-fold counts) """
    lower = uniform_SO3_sample(4)
    x, y = lower.size * 8, sample.size
    assert np.isclose(x, y, rtol=0.025)


@pytest.mark.parametrize("big,small", [(77, 52), (48, 37)])
def test_get_sample_local_width(big, small):
    """ Checks that width follows the expected trend (X - Sin(X)) """
    resolution = np.pi

    z = get_sample_local(resolution=resolution, grid_width=small)

    assert np.all(z.angle_with(Rotation([1, 0, 0, 0])) < np.deg2rad(small))
    assert np.any(
        z.angle_with(Rotation([1, 0, 0, 0])) > np.deg2rad(small - 1.5 * resolution)
    )

    x_size = z.size
    assert x_size > 0
    y_size = get_sample_local(resolution=np.pi, grid_width=big).size
    x_v = np.deg2rad(small) - np.sin(np.deg2rad(small))
    y_v = np.deg2rad(big) - np.sin(np.deg2rad(big))
    exp = y_size / x_size
    theory = y_v / x_v

    # resolution/width is high, so we must be generous on tolerance
    assert np.isclose(exp, theory, rtol=0.2)


@pytest.mark.parametrize("width", [60, 33])
def test_get_sample_local_center(fr, width):
    """ Checks that the center argument works as expected """
    resolution = 8
    x = get_sample_local(resolution=resolution, center=fr, grid_width=width)
    assert np.all((x.angle_with(fr) < np.deg2rad(width)))
    # makes sure some of our rotations are inner the outer region
    assert np.any(x.angle_with(fr) > np.deg2rad(width - resolution * 1.5))


@pytest.fixture(scope="session")
def C6_sample():
    return get_sample_fundamental(4, point_group=C6)


def test_get_sample_fundamental_zone_order(C6_sample):
    """ Cross check point counts to group order terms """
    D6_sample = get_sample_fundamental(4, point_group=D6)
    ratio = C6_sample.size / D6_sample.size
    assert np.isclose(ratio, 2, rtol=0.025)


def test_get_sample_fundamental_space_group(C6_sample):
    """ Going via the space_group route """
    # assert that space group #3 is has pg C2
    assert C2 == get_point_group(3, proper=True)
    C2_sample = get_sample_fundamental(4, space_group=3)
    ratio = C2_sample.size / C6_sample.size
    assert np.isclose(ratio, 3, rtol=0.025)
