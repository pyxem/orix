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

import numpy as np
import pytest

from orix.quaternion.rotation import Rotation
from orix.quaternion.symmetry import C1, C2, D2, C4, D4, C3, D3, C6, D6, T, O
from orix.sampling import get_sample_fundamental, get_sample_local
from orix.sampling._cubochoric_sampling import (
    cubochoric_sampling,
    resolution_to_semi_edge_steps,
    _cubochoric_sampling_loop,
)


class TestCubochoricSampling:
    def test_cubochoric_sampling_fundamental(self):
        kwargs = dict(semi_edge_steps=10, method="cubochoric")
        rot_identity = Rotation.identity().data

        rot_pg1 = get_sample_fundamental(point_group=C1, **kwargs)
        assert rot_pg1.size == 8000
        assert np.any(rot_pg1.data == rot_identity)

        rot_pg2 = get_sample_fundamental(point_group=C2, **kwargs)
        assert rot_pg2.size == 4026
        assert np.any(rot_pg2.data == rot_identity)

        rot_pg222 = get_sample_fundamental(point_group=D2, **kwargs)
        assert rot_pg222.size == 1983
        assert np.any(rot_pg222.data == rot_identity)

        rot_pg4 = get_sample_fundamental(point_group=C4, **kwargs)
        assert rot_pg4.size == 2076
        assert np.any(rot_pg4.data == rot_identity)

        rot_pg422 = get_sample_fundamental(point_group=D4, **kwargs)
        assert rot_pg422.size == 925
        assert np.any(rot_pg422.data == rot_identity)

        rot_pg3 = get_sample_fundamental(point_group=C3, **kwargs)
        assert rot_pg3.size == 2710
        assert np.any(rot_pg3.data == rot_identity)

        rot_pg32 = get_sample_fundamental(point_group=D3, **kwargs)
        assert rot_pg32.size == 1337
        assert np.any(rot_pg32.data == rot_identity)

        rot_pg6 = get_sample_fundamental(point_group=C6, **kwargs)
        assert rot_pg6.size == 1298
        assert np.any(rot_pg6.data == rot_identity)

        rot_pg622 = get_sample_fundamental(point_group=D6, **kwargs)
        assert rot_pg622.size == 611
        assert np.any(rot_pg622.data == rot_identity)

        rot_pg23 = get_sample_fundamental(point_group=T, **kwargs)
        assert rot_pg23.size == 703
        assert np.any(rot_pg23.data == rot_identity)

        rot_pg432 = get_sample_fundamental(point_group=O, **kwargs)
        assert rot_pg432.size == 361
        assert np.any(rot_pg432.data == rot_identity)

    def test_cubochoric_sampling_resolution(self):
        kwargs = dict(point_group=C4, method="cubochoric")
        rot_res = get_sample_fundamental(resolution=6.9, **kwargs)
        rot_steps = get_sample_fundamental(semi_edge_steps=19, **kwargs)

        assert np.allclose(rot_res.data, rot_steps.data)

    def test_get_sample_local_center_cubochoric(self):
        # Fixed rotation takes us 30 degrees from origin
        center = Rotation([0.5, 0.5, 0, 0])
        rot = get_sample_local(
            resolution=4, grid_width=20, center=center, method="cubochoric"
        )
        assert np.all(rot.a < np.cos(np.deg2rad(5)))

    def test_resolution_to_semi_edge_steps(self):
        resolutions = np.arange(1, 11)
        semi_edge_steps = np.array([137, 67, 45, 33, 27, 22, 19, 17, 15, 13])
        for resolution, steps in zip(resolutions, semi_edge_steps):
            assert resolution_to_semi_edge_steps.py_func(resolution) == steps

    def test_cubochoric_sampling_loop(self):
        semi_edge_steps = 10
        quat_arr = _cubochoric_sampling_loop.py_func(semi_edge_steps)
        assert quat_arr.shape[0] == (2 * semi_edge_steps + 1) ** 3 - 1261

    def test_cubochoric_sampling_raises(self):
        with pytest.raises(ValueError, match="Either `semi_edge_steps` or "):
            _ = cubochoric_sampling()
