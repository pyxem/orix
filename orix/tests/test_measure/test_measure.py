#
# Copyright 2018-2025 the orix developers
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with orix. If not, see <http://www.gnu.org/licenses/>.
#

from copy import deepcopy

import numpy as np
import pytest

from orix.measure import pole_density_function
from orix.quaternion import symmetry
from orix.vector import Vector3d
from orix.measure.pole_density_function import _cube_gnom_coordinates
from orix.sampling.S2_sampling import sample_S2_equiangle_cube_mesh_face_centers
from orix.sampling.S2_sampling import _sample_S2_uv_mesh_coordinates, sample_S2_random_mesh

@pytest.fixture(
    params=[
        symmetry.D2h,
        symmetry.S6,
        symmetry.D3d,
        symmetry.C4h,
        symmetry.D4h,
        symmetry.C6h,
        symmetry.D6h,
        symmetry.Th,
        symmetry.Oh,
    ]
)
def point_groups(request):
    return request.param


class TestMeasurePoleDensityFunction:

    def test_output_format(self):
        v = sample_S2_random_mesh(1, seed=954)
        hist1, (x1, y1) = pole_density_function(v)
        assert hist1.shape[0] + 1 == x1.shape[0] == y1.shape[0]
        assert hist1.shape[1] + 1 == x1.shape[1] == y1.shape[1]
        assert isinstance(hist1, np.ma.MaskedArray)
        assert hist1.mask.sum() == 0

        hist2, (x2, y2) = pole_density_function(v, symmetry=symmetry.C6)
        assert hist2.shape[0] + 1 == x2.shape[0] == y2.shape[0]
        assert hist2.shape[1] + 1 == x2.shape[1] == y2.shape[1]
        assert hist1.shape == hist2.shape
        assert x1.shape == x2.shape
        assert y1.shape == y2.shape
        assert isinstance(hist2, np.ma.MaskedArray)
        assert hist2.mask.sum() > 0

    @pytest.mark.parametrize("resolution", [0.5, 1.0])
    def test_pole_density_function_mrd_norm(self, point_groups, resolution):
        pg = point_groups
        v = sample_S2_random_mesh(1.0, seed=230)

        # Make plot grid
        _, polar_coords = _sample_S2_uv_mesh_coordinates(
            resolution,
            hemisphere='upper',
            azimuth_endpoint=True,
        )
        polar_coords = polar_coords[:-1] + np.diff(polar_coords) / 2
        solid_angle = np.abs(np.sin(polar_coords))[np.newaxis, :]

        for sigma in [2*resolution, 5*resolution]:
            hist, _ = pole_density_function(v, symmetry=pg, resolution=resolution, sigma=sigma)
            mean_value = np.sum(solid_angle*hist) / np.sum(solid_angle*~hist.mask)
            print(mean_value)
            assert np.allclose(mean_value, 1.0, rtol=0.01)

    def test_pole_density_function_log(self):
        # v = Vector3d.random(11_234)
        v = sample_S2_random_mesh(1.0, seed=230)

        hist1, _ = pole_density_function(v, log=False)
        hist2, _ = pole_density_function(v, log=True)
        assert not np.allclose(hist1, hist2)

    def test_pole_density_function_sigma(self):
        # v = Vector3d.random(11_234)
        v = sample_S2_random_mesh(1.0, seed=230)

        hist1, _ = pole_density_function(v, sigma=2.5)
        hist2, _ = pole_density_function(v, sigma=5)
        assert not np.allclose(hist1, hist2)

    def test_pole_density_function_weights(self):
        # v = Vector3d.random(11_234)
        v = sample_S2_random_mesh(1.0, seed=230)
        v.z[v.z < 0] *= -1

        hist0, _ = pole_density_function(v, weights=None)
        weights1 = np.ones(v.shape[0])
        hist1, _ = pole_density_function(v, weights=weights1)
        assert np.allclose(hist0, hist1)

        weights2 = 2 * np.ones(v.shape[0])
        hist2, _ = pole_density_function(v, weights=weights2)
        # the same because MRD normalizes by average
        assert np.allclose(hist0, hist2)

        #TDOD: Ask about expected behaviour for mrd=False
        # hist0_counts, _ = pole_density_function(v, weights=None, mrd=False)
        # hist2_counts, _ = pole_density_function(v, weights=weights2, mrd=False)
        # not the same because hist values are not normalized
        # assert not np.allclose(hist0_counts, hist2_counts)

        # non-uniform weights
        weights2[54] *= 1.01
        hist2_1, _ = pole_density_function(v, weights=weights2)
        assert not np.allclose(hist0, hist2_1)

    def test_PDF_IPDF_equivalence(self):
        v = Vector3d.random(100_000)

        hist_pdf, _ = pole_density_function(v, weights=None)
        hist_ipdf, _ = pole_density_function(v, weights=None, symmetry=symmetry.C1)

        # in testing this test passes at tolerance of 1% for 100_000
        # vectors, but raise tolerance to 2% to ensure pass
        assert np.allclose(hist_pdf, hist_ipdf, atol=0.02)

    def test_pole_density_function_empty_vector_raises(self):
        v = Vector3d.empty()
        assert not v.size

        with pytest.raises(
            ValueError
        ):
            pole_density_function(v)


class TestGnomCubeRoutines:

    def test_corner_edge_assignment(self):
        """ Make sure we get useable results for corner-cases.
        """
        corners = Vector3d(
            [[1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],]
            )

        edges = Vector3d(
            [[0, 1, 1],
            [0, 1, -1],
            [0, -1, 1],
            [0, -1, -1],
            [1, 0, 1],
            [1, 0, -1],
            [-1, 0, 1],
            [-1, 0, -1],
            [1, 1, 0],
            [1, -1, 0],
            [-1, 1, 0],
            [-1, -1, 0],]
            )

        c_index, c_coordinates = _cube_gnom_coordinates(corners)
        e_index, e_coordinates = _cube_gnom_coordinates(edges)

        assert np.all(c_index == [0, 5, 0, 3, 2, 1, 4, 1,])
        assert np.all(e_index == [2, 5, 4, 3, 0, 5, 4, 1, 0, 3, 2, 1,])

    def test_grid_correct_mapping(self):
        """ Make sure grids get assigned to the correct faces and coordinates.
        """
        faces_grid = sample_S2_equiangle_cube_mesh_face_centers(15)
        index_faces, corrdinates_faces = _cube_gnom_coordinates(faces_grid)

        exp_coords = np.array([-0.65449847, -0.39269908, -0.13089969, 0.13089969, 0.39269908, 0.65449847])
        for face_index in range(6):
            assert np.all(index_faces[face_index] == face_index)
            assert np.allclose(corrdinates_faces[0, face_index], exp_coords[:, np.newaxis])
            assert np.allclose(corrdinates_faces[1, face_index], exp_coords[np.newaxis, :])

    def test_blurring_kernel(self):
        """ Check that the smoothing gives us roughly the correct width.
        """
        vectors =Vector3d(np.array([0.0, 0, 1]))
        for resolution in [0.25, 0.5, 1.0]:
            for s in [5, 10, 20]:
                sigma = resolution * s
                hist, _ = pole_density_function(vectors, sigma=sigma, resolution=resolution)
                assert hist[0, 0] / hist[0, s] > 2.3
                assert hist[0, 0] / hist[0, s] < 3.0
