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


def test_get_sample_local_width(fr):
    """ Checks that doubling the width 8 folds the number of points """
    x = get_sample_local(np.pi, fr, 15).size * 8
    y = get_sample_local(np.pi, fr, 30).size
    assert np.isclose(x, y, rtol=0.025)


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
