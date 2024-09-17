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

from orix.projections import StereographicProjection
from orix.vector import Vector3d


@pytest.fixture
def vectors():
    return Vector3d(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -1],
            [-1, 0, 0],
            [0, -1, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [0, -1, -1],
            [-1, 0, -1],
            [-1, -1, 0],
            [0, -1, 1],
            [-1, 0, 1],
            [-1, 1, 0],
            [0, 1, -1],
            [1, 0, -1],
            [1, -1, 0],
            [1, 1, 1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
            [-1, -1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, -1],
        ]
    ).unit


class TestStereographicProjection:
    def test_simple_vector2xy(self):
        v = Vector3d([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        x_desired = [1, 0, -1, 0, 0]
        y_desired = [0, 1, 0, -1, 0]

        sp_up = StereographicProjection()
        x_up, y_up = sp_up.vector2xy(v)
        assert x_up.size == y_up.size == 5
        assert np.allclose(x_up, x_desired)
        assert np.allclose(y_up, y_desired)

        sp_down = StereographicProjection(pole=1)
        x_down, y_down = sp_down.vector2xy(v)
        assert x_down.size == y_down.size == 4

        assert np.allclose(x_down, x_desired[:-1])
        assert np.allclose(y_down, y_desired[:-1])

    def test_vector2xy(self, vectors):
        sp_up = StereographicProjection()
        x_up, y_up, x_down, y_down = sp_up.vector2xy_split(v=vectors)
        x1, y1 = sp_up.vector2xy(v=vectors)
        sp_down = StereographicProjection(pole=1)
        x2, y2 = sp_down.vector2xy(v=vectors)

        assert np.allclose(x1, x_up)
        assert np.allclose(y1, y_up)
        assert np.allclose(x2, x_down)
        assert np.allclose(y2, y_down)

    def test_spherical2xy(self, vectors):
        sp_up = StereographicProjection()
        azimuth = vectors.azimuth
        polar = vectors.polar
        x_up, y_up, x_down, y_down = sp_up.spherical2xy_split(azimuth, polar)
        x1, y1 = sp_up.spherical2xy(azimuth, polar)
        sp_down = StereographicProjection(pole=1)
        x2, y2 = sp_down.spherical2xy(azimuth, polar)

        assert np.allclose(x1, x_up)
        assert np.allclose(y1, y_up)
        assert np.allclose(x2, x_down)
        assert np.allclose(y2, y_down)

        # Degrees
        azimuth_deg = np.rad2deg(azimuth)
        polar_deg = np.rad2deg(polar)
        x_up3, y_up3, x_down3, y_down3 = sp_up.spherical2xy_split(
            azimuth_deg, polar_deg, degrees=True
        )
        x3, y3 = sp_up.spherical2xy(azimuth_deg, polar_deg, degrees=True)
        sp_down = StereographicProjection(pole=1)
        x4, y4 = sp_down.spherical2xy(azimuth_deg, polar_deg, degrees=True)

        assert np.allclose(x_up3, x3)
        assert np.allclose(y_up3, y3)
        assert np.allclose(x4, x_down3)
        assert np.allclose(y4, y_down3)

    def test_project_loop_xy(self, vectors):
        is_up = vectors.z >= 0
        v_up_in = vectors[is_up]
        v_down_in = vectors[~is_up]

        sp_up = StereographicProjection()
        sp_down = StereographicProjection(pole=1)

        x_up, y_up = sp_up.vector2xy(v_up_in)
        x_down, y_down = sp_down.vector2xy(v_down_in)
        v_up_out = sp_up.inverse.xy2vector(x_up, y_up)
        v_down_out = sp_down.inverse.xy2vector(x_down, y_down)

        assert np.allclose(v_up_in.data, v_up_out.data)
        assert np.allclose(v_down_in.data, v_down_out.data)

    def test_project_loop_spherical(self, vectors):
        is_up = vectors.z >= 0
        v_up = vectors[is_up]
        v_down = vectors[~is_up]
        azimuth_up_in = v_up.azimuth
        polar_up_in = v_up.polar
        azimuth_down_in = v_down.azimuth
        polar_down_in = v_down.polar

        sp_up = StereographicProjection()
        sp_down = StereographicProjection(pole=1)

        x_up, y_up = sp_up.spherical2xy(azimuth_up_in, polar_up_in)
        x_down, y_down = sp_down.spherical2xy(azimuth_down_in, polar_down_in)
        azimuth_up_out, polar_up_out = sp_up.inverse.xy2spherical(x_up, y_up)
        azimuth_down_out, polar_down_out = sp_down.inverse.xy2spherical(x_down, y_down)

        assert np.allclose(azimuth_up_in, azimuth_up_out)
        assert np.allclose(polar_up_in, polar_up_out)
        assert np.allclose(azimuth_down_in, azimuth_down_out)
        assert np.allclose(polar_down_in, polar_down_out)

        # Degrees
        azimuth_up_deg, polar_up_deg = sp_up.inverse.xy2spherical(
            x_up, y_up, degrees=True
        )
        assert np.allclose(np.rad2deg(azimuth_up_out), azimuth_up_deg)
        assert np.allclose(np.rad2deg(polar_up_out), polar_up_deg)
