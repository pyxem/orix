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

from copy import deepcopy

import numpy as np
import pytest

from orix.measure import pole_density_function
from orix.quaternion import symmetry
from orix.vector import Vector3d


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
    def test_pole_density_function(self):
        v = Vector3d.random(10_000)

        hist1, (x1, y1) = pole_density_function(v)
        assert hist1.shape[0] + 1 == x1.shape[0] == y1.shape[0]
        assert hist1.shape[1] + 1 == x1.shape[1] == y1.shape[1]
        assert np.allclose(hist1.mean(), 1, rtol=1e-3)
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

        assert np.allclose(hist1.mean(), hist2.mean())

    def test_pole_density_function_symmetry(self, point_groups):
        pg = point_groups
        v = Vector3d.random(10_000)

        hist, _ = pole_density_function(v, symmetry=pg, mrd=False)
        assert np.allclose(hist.sum(), v.size, rtol=0.01)

    def test_pole_density_function_hemisphere(self):
        v = Vector3d.random(11_234)

        hist1_upper, _ = pole_density_function(v, hemisphere="upper")
        assert np.allclose(hist1_upper.mean(), 1)

        hist1_lower, _ = pole_density_function(v, hemisphere="lower")
        assert np.allclose(hist1_lower.mean(), 1)

        hist2_upper, _ = pole_density_function(v, hemisphere="upper", mrd=False)
        hist2_lower, _ = pole_density_function(v, hemisphere="lower", mrd=False)
        assert np.allclose(hist2_upper.sum() + hist2_lower.sum(), v.size)

    @pytest.mark.parametrize("n", [10, 1000, 10_000, 12_546])
    def test_pole_density_function_values(self, n):
        # vectors only on upper hemisphere
        v = Vector3d.random(n)
        v2 = deepcopy(v)
        v2.z[v2.z < 0] *= -1

        hist1, _ = pole_density_function(v2, mrd=False)
        assert np.allclose(hist1.sum(), n, atol=0.1)

        hist2, _ = pole_density_function(v, symmetry=symmetry.Th, mrd=False)
        assert np.allclose(hist2.sum(), n, rtol=0.1)

        hist3, _ = pole_density_function(v2, symmetry=symmetry.Th, mrd=False)
        assert np.allclose(hist3.sum(), n, rtol=0.1)

    def test_pole_density_function_log(self):
        v = Vector3d.random(11_234)

        hist1, _ = pole_density_function(v, log=False)
        hist2, _ = pole_density_function(v, log=True)
        assert not np.allclose(hist1, hist2)

    def test_pole_density_function_sigma(self):
        v = Vector3d.random(11_234)

        hist1, _ = pole_density_function(v, sigma=2.5)
        hist2, _ = pole_density_function(v, sigma=5)
        assert not np.allclose(hist1, hist2)

    def test_pole_density_function_weights(self):
        v = Vector3d.random(11_234)
        v.z[v.z < 0] *= -1

        hist0, _ = pole_density_function(v, weights=None)
        weights1 = np.ones(v.shape[0])
        hist1, _ = pole_density_function(v, weights=weights1)
        assert np.allclose(hist0, hist1)

        weights2 = 2 * np.ones(v.shape[0])
        hist2, _ = pole_density_function(v, weights=weights2)
        # the same because MRD normalizes by average
        assert np.allclose(hist0, hist2)

        hist0_counts, _ = pole_density_function(v, weights=None, mrd=False)
        hist2_counts, _ = pole_density_function(v, weights=weights2, mrd=False)
        # not the same because hist values are not normalized
        assert not np.allclose(hist0_counts, hist2_counts)

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
            ValueError, match="`azimuth` and `polar` angles have 0 size"
        ):
            pole_density_function(v)
