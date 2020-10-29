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

import numpy as np
import pytest

from orix.crystal_map.crystal_map_properties import CrystalMapProperties


class TestCrystalMapProperties:
    @pytest.mark.parametrize(
        "dictionary, id, is_in_data",
        [
            ({}, np.arange(10), np.ones(10, dtype=bool)),
            ({"iq": np.arange(10), "dp": np.zeros(10)}, np.arange(10), None,),
            ({}, np.arange(5), np.array([1, 1, 0, 1, 1], dtype=bool)),
        ],
    )
    def test_init_properties(self, dictionary, id, is_in_data):
        props = CrystalMapProperties(
            dictionary=dictionary, id=id, is_in_data=is_in_data
        )

        assert props == dictionary
        assert np.allclose(props.id, id)
        if is_in_data is None:
            is_in_data = np.ones(id.size, dtype=bool)
        assert np.allclose(props.is_in_data, is_in_data)

    def test_set_item(self):
        map_size = 10
        is_in_data = np.ones(map_size, dtype=bool)
        is_in_data[5] = False
        d = {"iq": np.arange(map_size)}
        props = CrystalMapProperties(d, id=np.arange(map_size), is_in_data=is_in_data)

        # Set array with an array
        n_in_data = map_size - len(np.where(~is_in_data)[0])
        props["iq"] = np.arange(n_in_data) + 1
        expected_array = np.array([1, 2, 3, 4, 5, 5, 6, 7, 8, 9])
        assert np.allclose(props.get("iq"), expected_array)

        # Set array with one value (broadcasting works)
        props["iq"] = 2
        expected_array2 = np.ones(map_size) * 2
        expected_array2[5] = expected_array[5]
        assert np.allclose(props.get("iq"), expected_array2)

    def test_set_item_nd(self):
        map_size = 10
        d = {"iq": np.arange(map_size)}
        props = CrystalMapProperties(d, id=np.arange(map_size))

        # 2D
        prop_2d = np.arange(map_size * 2).reshape((10, 2))
        props["prop_2d"] = prop_2d
        assert np.allclose(props["prop_2d"], prop_2d)

        # 3D
        prop_3d = np.arange(map_size * 4).reshape((10, 2, 2))
        props["prop_3d"] = prop_3d
        assert np.allclose(props["prop_3d"], prop_3d)

        with pytest.raises(IndexError, match="boolean index did not match indexed"):
            props["prop_3d_wrong"] = np.random.random(40).reshape((2, 10, 2))

        # Update 2D array, accounting for in data values
        is_in_data = np.ones(map_size, dtype=bool)
        is_in_data[5] = False
        props.is_in_data = is_in_data
        new_prop_2d = np.arange(map_size * 2).reshape((map_size, 2))
        props["prop_2d"] = new_prop_2d[is_in_data]
        np.allclose(props["prop_2d"], new_prop_2d[is_in_data])

    def test_set_item_error(self):
        map_size = 10
        is_in_data = np.ones(map_size, dtype=bool)
        id_to_change = 3
        is_in_data[id_to_change] = False

        d = {"iq": np.arange(map_size)}
        props = CrystalMapProperties(d, id=np.arange(map_size), is_in_data=is_in_data)

        # Set array with an array
        with pytest.raises(ValueError, match="shape mismatch: value array of shape"):
            props["iq"] = np.arange(map_size) + 1

        # Set new 2D array
        props.is_in_data[id_to_change] = True
        with pytest.raises(IndexError, match="boolean index did not match indexed"):
            new_shape = (10 // 2, 10 // 5)
            props["dp"] = np.arange(map_size).reshape(new_shape)
