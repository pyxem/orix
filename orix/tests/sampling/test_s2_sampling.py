# -*- coding: utf-8 -*-
# Copyright 2018-2022 the orix developers
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

from orix import sampling


class TestS2Sampling:
    @pytest.mark.parametrize("resolution, size", [(3, 7082), (4, 3962)])
    def test_uv_mesh(self, resolution, size):
        v1 = sampling.sample_S2_uv_mesh(resolution)
        assert v1.size == size
        assert np.allclose(v1.mean().data, [0, 0, 0])
        # test correct number of polar levels (z)
        expected_num_polar_z = int(np.ceil(180 / resolution)) + 1
        z_rounded = v1.z.round(6)
        assert np.unique(z_rounded).size == expected_num_polar_z
        # test no duplicate azimuthal values, get a central elevation
        z_unique = np.unique(z_rounded)
        expected_num_azimuth = int(np.ceil(360 / resolution))
        v1_elev = v1[np.isclose(z_rounded, z_unique[z_unique.size // 2])]
        assert np.unique(np.arctan2(v1_elev.y, v1_elev.x)).size == expected_num_azimuth
        # check only one pole at north and south pole
        assert np.isclose(v1.z, 1).sum() == 1
        assert np.isclose(v1.z, -1).sum() == 1

    @pytest.mark.parametrize(
        "grid_type, resolution, size",
        [
            ("spherified_corner", 3, 8666),
            ("normalized", 4, 5402),
            ("spherified_edge", 5, 1946),
        ],
    )
    def test_cube_mesh(self, grid_type, resolution, size):
        v1 = sampling.sample_S2_cube_mesh(resolution, grid_type=grid_type)
        assert v1.size == size
        assert np.allclose(v1.mean().data, [0, 0, 0])
        assert v1.unique().size == size

    def test_cube_mesh_raises(self):
        with pytest.raises(ValueError, match="The `grid_type` hexagonal"):
            _ = sampling.sample_S2_cube_mesh(2, "hexagonal")

    def test_random_sphere_mesh(self):
        grid = sampling.sample_S2_random_mesh(1).data
        assert grid.shape[0] == 10313
        assert grid.shape[1] == 3

    def test_seed_for_random_sphere_mesh(self):
        grid_7 = sampling.sample_S2_random_mesh(resolution=3, seed=7)
        grid_7_again = sampling.sample_S2_random_mesh(resolution=3, seed=7)
        grid_8 = sampling.sample_S2_random_mesh(resolution=3, seed=8)
        assert np.allclose(grid_7.data, grid_7_again.data)
        assert not np.allclose(grid_7.data, grid_8.data)

    def test_hexagonal_mesh(self):
        grid = sampling.sample_S2_hexagonal_mesh(5).data
        assert grid.shape == (3458, 3)

    def test_icosahedral_grid(self):
        grid = sampling.sample_S2_icosahedral_mesh(10).data
        assert grid.shape[0] == 642
        assert grid.shape[1] == 3
        np.testing.assert_almost_equal(np.sum(grid), 0)
        grid_unique = np.unique(grid, axis=0)
        assert grid.shape[0] == grid_unique.shape[0]
