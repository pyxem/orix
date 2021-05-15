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

from orix.quaternion.rotation import Rotation
from orix.quaternion.symmetry import C2, C6, D6, get_point_group
from orix.sampling.SO3_sampling import uniform_SO3_sample
from orix.sampling.sample_generators import get_sample_fundamental, get_sample_local


@pytest.fixture(scope="session")
def sample():
    return uniform_SO3_sample(2)


@pytest.fixture(scope="session")
def fr():
    """ fixed rotation """
    r = Rotation([0.5, 0.5, 0, 0])
    return r

class TestUniformSO3():
    def test_old_method(self):
        _ = uniform_SO3_sample(10,method='quaternion')
        assert _.size > 0

    def test_default_method(self):
        _ = uniform_SO3_sample(10)
        assert _.size > 0

    def test_uniform_SO3_sample_regions(self,sample, fr):
        """ Checks that different regions have the same density"""
        around_zero = sample[sample.a > 0.9]
        moved = fr * sample
        elsewhere = moved[sample.a > 0.9]
        # extra line simplifies the stacktrack
        x, y = around_zero.size, elsewhere.size
        assert np.isclose(x, y, rtol=0.025)

    def test_uniform_SO3_sample_resolution(self,sample):
        """ Checks that doubling resolution doubles density (8-fold counts) """
        lower = uniform_SO3_sample(4)
        x, y = lower.size * 8, sample.size
        assert np.isclose(x, y, rtol=0.025)

class TestGetSampleLocal():
    @pytest.mark.parametrize("big,small,method", [(180, 90, 'quaternion'), (180, 90, 'harr_euler')])
    def test_get_sample_local_width(self, big, small,method):
        """ Checks that size follows the expected trend (X - Sin(X))
        With 180 and 90 we expect: pi and pi/2 - 1
        """

        resolution = 4
        x_size = get_sample_local(resolution=resolution, grid_width=small,method=method).size
        y_size = get_sample_local(resolution=resolution, grid_width=big,method=method).size
        x_v = np.deg2rad(small) - np.sin(np.deg2rad(small))
        y_v = np.deg2rad(big) - np.sin(np.deg2rad(big))
        exp = y_size / x_size
        theory = y_v / x_v

        assert x_size > 0 # if this fails exp will be nan
        assert np.isclose(exp, theory, atol=0.2)

class TestSamplingFundamentalSector():
    @pytest.fixture(scope="session")
    def C6_sample(self):
        return get_sample_fundamental(4, point_group=C6)

    def test_get_sample_fundamental_zone_order(self,C6_sample):
        """ Cross check point counts to group order terms """
        D6_sample = get_sample_fundamental(4, point_group=D6)
        ratio = C6_sample.size / D6_sample.size
        assert np.isclose(ratio, 2, atol=0.2)


    def test_get_sample_fundamental_space_group(self,C6_sample):
        """ Going via the space_group route """
        # assert that space group #3 is has pg C2
        assert C2 == get_point_group(3, proper=True)
        C2_sample = get_sample_fundamental(4, space_group=3)
        ratio = C2_sample.size / C6_sample.size
        assert np.isclose(ratio, 3, atol=0.2)
