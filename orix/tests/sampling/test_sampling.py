# -*- coding: utf-8 -*-
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

from diffpy.structure import Atom, Lattice, Structure
import numpy as np
import pytest

from orix.crystal_map import Phase
from orix.quaternion import Rotation
from orix.quaternion.symmetry import C1, C2, C4, C6, D6, get_point_group
from orix.sampling import (
    get_sample_fundamental,
    get_sample_local,
    get_sample_reduced_fundamental,
    get_sample_zone_axis,
    uniform_SO3_sample,
)
from orix.sampling.SO3_sampling import _resolution_to_num_steps
from orix.sampling._polyhedral_sampling import (
    _get_angles_between_nn_gridpoints,
    _get_first_nearest_neighbors,
    _get_max_grid_angle,
    _get_start_and_end_index,
)


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

    @pytest.fixture(scope="session")
    def phase(self):
        a = 5.431
        latt = Lattice(a, a, a, 90, 90, 90)
        atom_list = []
        for coords in [[0, 0, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0]]:
            x, y, z = coords[0], coords[1], coords[2]
            atom_list.append(
                Atom(atype="Si", xyz=[x, y, z], lattice=latt)
            )  # Motif part A
            atom_list.append(
                Atom(atype="Si", xyz=[x + 0.25, y + 0.25, z + 0.25], lattice=latt)
            )  # Motif part B
        struct = Structure(atoms=atom_list, lattice=latt)
        p = Phase(structure=struct, space_group=227)
        return p

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
        rotations = get_sample_reduced_fundamental(resolution=4)
        rotations2 = get_sample_reduced_fundamental(resolution=4, point_group=C2)
        rotations4 = get_sample_reduced_fundamental(resolution=4, point_group=C4)
        rotations6 = get_sample_reduced_fundamental(resolution=4, point_group=C4)

        assert (
            np.abs(rotations.size / rotations2.size) - 2 < 0.1
        )  # about 2 times more rotations
        assert (
            np.abs(rotations2.size / rotations4.size) - 2 < 0.1
        )  # about 2 times more rotations
        assert (
            np.abs(rotations.size / rotations6.size) - 6 < 0.1
        )  # about 6 times more rotations

    def test_get_sample_reduced_fundamental_phase(self, phase):
        rotations = get_sample_reduced_fundamental(resolution=4, phase=phase)
        rotations2 = get_sample_reduced_fundamental(
            resolution=4, point_group=phase.point_group
        )
        np.testing.assert_allclose(rotations.data, rotations2.data)

    def test_get_sample_fundamental_phase(self, phase):
        rotations = get_sample_fundamental(resolution=4, phase=phase)
        rotations2 = get_sample_fundamental(resolution=4, point_group=phase.point_group)
        np.testing.assert_allclose(rotations.data, rotations2.data)

    @pytest.mark.parametrize("density", ("3", "7", "5"))
    @pytest.mark.parametrize("get_directions", (True, False))
    def test_get_zone_axis(self, density, get_directions, phase):
        if density == "5":
            with pytest.raises(ValueError):
                get_sample_zone_axis(phase=phase, density=density)
        else:
            if get_directions:
                rot, _ = get_sample_zone_axis(
                    phase=phase, density=density, return_directions=True
                )
            else:
                rot = get_sample_zone_axis(phase=phase, density=density)
            assert isinstance(rot, Rotation)
