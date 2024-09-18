# Copyright 2018-2024 the orix developers
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
from orix.quaternion.symmetry import C2, C4, C6, D6, Oh, get_point_group
from orix.sampling import (
    get_sample_fundamental,
    get_sample_local,
    get_sample_reduced_fundamental,
    uniform_SO3_sample,
)
from orix.sampling.SO3_sampling import _resolution_to_num_steps
from orix.sampling._polyhedral_sampling import (
    _get_angles_between_nn_gridpoints,
    _get_first_nearest_neighbors,
    _get_max_grid_angle,
    _get_start_and_end_index,
)
from orix.vector import Vector3d


@pytest.fixture(scope="session")
def sample():
    return uniform_SO3_sample(2, method="haar_euler")


@pytest.fixture(scope="session")
def fixed_rotation():
    """A fixed rotation."""
    return Rotation([0.5, 0.5, 0, 0])


class TestPolyhedralSamplingUtils:
    @pytest.mark.parametrize(
        "number_of_steps, include_start, include_end, positive_and_negative, expected",
        [
            (5, False, False, False, (1, 5)),
            (5, True, False, False, (0, 5)),
            (6, False, True, False, (1, 7)),
            (7, True, True, True, (-7, 8)),
        ],
    )
    def test_get_start_and_end_index(
        self,
        number_of_steps,
        include_start,
        include_end,
        positive_and_negative,
        expected,
    ):
        assert (
            _get_start_and_end_index(
                number_of_steps=number_of_steps,
                include_start=include_start,
                include_end=include_end,
                positive_and_negative=positive_and_negative,
            )
            == expected
        )

    def test_first_nearest_neighbors(self):
        grid = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 1],
            ]
        )
        fnn = np.array(
            [
                [1, 0, 1],
                [0, 1, 1],
                [0, 1, 0],
                [1, 0, 0],
            ]
        )
        angles = np.array([45, 45, 45, 45])
        fnn_test = _get_first_nearest_neighbors(grid)
        angles_test = _get_angles_between_nn_gridpoints(grid)
        assert np.allclose(fnn, fnn_test)
        assert np.allclose(angles, angles_test)
        assert abs(_get_max_grid_angle(grid) - 45.0) < 1e-7


class TestSamplingUtils:
    def test_resolution_to_num_steps(self):
        n_steps_odd = _resolution_to_num_steps(1, odd_only=True)
        assert isinstance(n_steps_odd, int)
        assert n_steps_odd == 361
        # Note that 360/1.002 = 359.28 rounds to 359
        assert _resolution_to_num_steps(1.002, even_only=True) == 360


class TestUniformSO3:
    def test_sample_size(self):
        r1 = uniform_SO3_sample(10, method="haar_euler")
        assert r1.size > 0
        r2 = uniform_SO3_sample(10, method="quaternion")
        assert r2.size > 0

    def test_uniform_SO3_sample_regions(self, sample, fixed_rotation):
        """Checks that different regions have the same density."""
        around_zero = sample[sample.a > 0.9]
        moved = fixed_rotation * sample
        elsewhere = moved[sample.a > 0.9]
        # Extra line simplifies the stack trace
        x, y = around_zero.size, elsewhere.size
        assert np.isclose(x, y, rtol=0.025)

    def test_uniform_SO3_sample_resolution(self, sample):
        """Checks that doubling resolution doubles density (8-fold
        counts).
        """
        lower = uniform_SO3_sample(4, method="haar_euler")
        x, y = lower.size * 8, sample.size
        assert np.isclose(x, y, rtol=0.025)


class TestGetSampleLocal:
    @pytest.mark.parametrize(
        "big, small, method", [(180, 90, "quaternion"), (180, 90, "haar_euler")]
    )
    def test_get_sample_local_width(self, big, small, method):
        """Checks that size follows the expected trend (X - Sin(X)).
        With 180 and 90 we expect: pi and pi/2 - 1.
        """
        res = 4  # Degrees
        x_size = get_sample_local(resolution=res, grid_width=small, method=method).size
        y_size = get_sample_local(resolution=res, grid_width=big, method=method).size
        x_v = np.deg2rad(small) - np.sin(np.deg2rad(small))
        y_v = np.deg2rad(big) - np.sin(np.deg2rad(big))
        exp = y_size / x_size
        theory = y_v / x_v

        assert x_size > 0  # If this fails exp will be NaN
        assert np.isclose(exp, theory, atol=0.2)

    def test_get_sample_local_center(self, fixed_rotation):
        # fixed rotation takes us 30 degrees from origin
        x = get_sample_local(
            resolution=3, grid_width=20, center=fixed_rotation, method="haar_euler"
        )
        # a == cos(omega/2)
        assert np.all(x.a < np.cos(np.deg2rad(5)))


class TestSampleFundamental:
    @pytest.fixture(scope="session")
    def C6_sample(self):
        return get_sample_fundamental(resolution=4, point_group=C6, method="haar_euler")

    def test_get_sample_fundamental_zone_order(self, C6_sample):
        """Cross check point counts to group order terms."""
        D6_sample = get_sample_fundamental(4, point_group=D6, method="haar_euler")
        ratio = C6_sample.size / D6_sample.size
        assert np.isclose(ratio, 2, atol=0.2)

    def test_get_sample_fundamental_space_group(self, C6_sample):
        """Going via the `space_group` route."""
        # Assert that space group #3 is has pg C2
        assert C2 == get_point_group(3, proper=True)
        C2_sample = get_sample_fundamental(4, space_group=3, method="haar_euler")
        ratio = C2_sample.size / C6_sample.size
        assert np.isclose(ratio, 3, atol=0.2)

    def test_get_sample_reduced_fundamental(self):
        vz = Vector3d.zvector()

        R_C1 = get_sample_reduced_fundamental(resolution=4)
        v_C1 = R_C1 * vz
        assert np.allclose(v_C1.mean().data, [0, 0, 0])

        R_C4 = get_sample_reduced_fundamental(resolution=4, point_group=C4)
        v_C4 = R_C4 * vz
        assert np.all(v_C4 <= C4.fundamental_sector)

        R_C6 = get_sample_reduced_fundamental(resolution=4, point_group=C6)
        v_C6 = R_C6 * vz
        assert np.all(v_C6 <= C6.fundamental_sector)

        R_Oh = get_sample_reduced_fundamental(point_group=Oh)
        v_Oh = R_Oh * vz
        assert np.all(v_Oh <= Oh.fundamental_sector)

        # Some rotations have a phi1 Euler angle of multiples of pi,
        # presumably due to rounding errors
        phi1_C1 = R_C1.to_euler()[:, 0].round(7)
        assert np.allclose(np.unique(phi1_C1), 0, atol=1e-7)
        phi1_C4 = R_C4.to_euler()[:, 0].round(7)
        assert np.allclose(np.unique(phi1_C4), [0, np.pi / 2], atol=1e-7)
        phi1_C6 = R_C6.to_euler()[:, 0].round(7)
        assert np.allclose(np.unique(phi1_C6), [0, np.pi / 2], atol=1e-7)
        phi1_Oh = R_Oh.to_euler()[:, 0].round(7)
        assert np.allclose(np.unique(phi1_Oh), [0, np.pi / 2], atol=1e-7)

        R_Oh2 = get_sample_reduced_fundamental(point_group=Oh, method="icosahedral")
        assert R_Oh.size > R_Oh2.size
