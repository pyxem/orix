# -*- coding: utf-8 -*-
# Copyright 2018-2020 The pyXem developers
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

from orix.crystal_map import CrystalMap
from orix.quaternion.orientation import Orientation
from orix.quaternion.rotation import Rotation

# Note that many parts of the CrystalMap() class are tested while testing IO and the
# Phase() and PhaseList() classes


class TestCrystalMapInit:
    def test_minimal_init(self, rotations):
        map_size = 2

        assert isinstance(rotations, Rotation)

        cm = CrystalMap(rotations=rotations)

        assert cm.size == map_size
        assert cm.shape == (map_size,)
        assert cm.ndim
        assert np.allclose(cm.id, np.arange(map_size))
        assert isinstance(cm.rotations, Rotation)
        assert np.allclose(cm.rotations.data, rotations.data)

    @pytest.mark.parametrize("rotation_format", ["array", "list"])
    def test_init_with_invalid_rotations(self, rotations, rotation_format):
        with pytest.raises(ValueError):
            if rotation_format == "array":
                _ = CrystalMap(rotations=rotations.data)
            else:  # rotation_format == "list"
                _ = CrystalMap(rotations=list(rotations.data))

    @pytest.mark.parametrize(
        (
            "crystal_map_input, expected_shape, expected_step_sizes, "
            "expected_rotations_per_point"
        ),
        [
            (((1, 5, 5), (0, 1.5, 1.5), 3), (5, 5), {"z": 0, "y": 1.5, "x": 1.5}, 3),
            (((1, 1, 10), (0, 0.1, 0.1), 1), (10,), {"z": 0, "y": 0, "x": 0.1}, 1),
            (((2, 10, 1), (0, 1e-3, 0), 2), (10,), {"z": 0, "y": 1e-3, "x": 0}, 2),
            (((10, 1, 1), (0, 0.1, 0.1), 1), (), {"z": 0, "y": 0, "x": 0}, 1),
        ],
        indirect=["crystal_map_input"],
    )
    def test_init_with_different_coordinate_arrays(
        self,
        crystal_map_input,
        expected_shape,
        expected_step_sizes,
        expected_rotations_per_point,
    ):
        z = crystal_map_input["z"]
        y = crystal_map_input["y"]
        x = crystal_map_input["x"]
        cm = CrystalMap(**crystal_map_input)

        assert cm.shape == expected_shape
        assert cm._step_sizes == expected_step_sizes
        assert cm.rotations_per_point == expected_rotations_per_point

        if cm.shape == ():
            assert cm._coordinates == {"z": None, "y": None, "x": None}
        else:
            for actual_coords, expected_coords, expected_step_size in zip(
                cm._coordinates.values(), [z, y, x], expected_step_sizes.values()
            ):
                if expected_step_size == 0:
                    assert actual_coords is None
                else:
                    assert np.allclose(actual_coords, expected_coords)

    @pytest.mark.parametrize(
        "crystal_map_input",
        [((2, 10, 5), (1, 1, 1), 1)],
        indirect=["crystal_map_input"],
    )
    def test_init_without_x_coordinates(self, crystal_map_input):
        r = crystal_map_input["rotations"]
        z = crystal_map_input["z"]
        y = crystal_map_input["y"]
        cm = CrystalMap(rotations=r, x=None, y=y, z=z)

        assert np.allclose(cm.x, np.arange(cm.size))
        assert np.allclose(cm.y, y)
        assert np.allclose(cm.z, z)

    def test_init_map_with_props(self, crystal_map_input):
        x = crystal_map_input["x"]
        props = {"iq": np.arange(x.size)}
        cm = CrystalMap(prop=props, **crystal_map_input)

        assert cm.prop == props
        assert np.allclose(cm.prop.id, cm.id)
        assert np.allclose(cm.prop.is_in_data, cm.is_in_data)

    @pytest.mark.parametrize("unique_phase_ids", [[0, 1], [1, -1]])
    def test_init_with_phase_id(self, crystal_map_input, unique_phase_ids):
        x = crystal_map_input["x"]

        # Create phase ID array, ensuring all are present
        phase_id = np.random.choice(unique_phase_ids, x.size)
        for i, unique_id in enumerate(unique_phase_ids):
            phase_id[i] = unique_id

        cm = CrystalMap(phase_id=phase_id, **crystal_map_input)

        assert np.allclose(cm.phase_id, phase_id)
        # Test all_indexed
        if -1 in unique_phase_ids:
            assert not cm.all_indexed
            assert -1 in cm.phases.phase_ids
        else:
            assert cm.all_indexed
            assert -1 not in cm.phases.phase_ids


class TestCrystalMapGetItem:
    @pytest.mark.parametrize(
        "crystal_map_input, slice_tuple, expected_shape",
        [
            (((5, 5, 5), (1, 1, 1), 1), slice(None, None, None), (5, 5, 5)),
            (
                ((5, 5, 5), (1, 1, 1), 1),
                (slice(1, 2, None), slice(None, None, None),),
                (5, 5),
            ),
            (
                ((2, 5, 5), (1, 1, 1), 1),
                (slice(0, 2, None), slice(None, None, None), slice(1, 4, None),),
                (2, 5, 3),
            ),
        ],
        indirect=["crystal_map_input"],
    )
    def test_get_by_slice(self, crystal_map_input, slice_tuple, expected_shape):
        cm = CrystalMap(**crystal_map_input)

        cm2 = cm[slice_tuple]
        assert cm2.shape == expected_shape

    def test_get_by_phase_name(self, crystal_map_input, phase_list):
        x = crystal_map_input["x"]

        # Create phase ID array, ensuring all are present
        phase_ids = np.random.choice(phase_list.phase_ids, x.size)
        for i, unique_id in enumerate(phase_list.phase_ids):
            phase_ids[i] = unique_id

        # Get number of points with each phase ID
        n_points_per_phase = {}
        for phase_i, phase in phase_list:
            n_points_per_phase[phase.name] = len(np.where(phase_ids == phase_i)[0])

        cm = CrystalMap(phase_id=phase_ids, **crystal_map_input)
        cm.phases = phase_list

        for (_, phase), (expected_phase_name, n_points) in zip(
            phase_list, n_points_per_phase.items()
        ):
            cm2 = cm[phase.name]

            assert cm2.size == n_points
            assert cm2.phases_in_data.names == [expected_phase_name]

    def test_get_by_indexed_not_indexed(self, crystal_map):
        cm = crystal_map

        # Set some points to not_indexed
        cm[2:4].phase_id = -1

        indexed = cm["indexed"]
        not_indexed = cm["not_indexed"]

        assert indexed.size + not_indexed.size == cm.size
        assert np.allclose(np.unique(not_indexed.phase_id), np.array([-1]))
        assert np.allclose(np.unique(indexed.phase_id), np.array([1]))

    def test_get_by_condition(self, crystal_map):
        cm = crystal_map

        cm.prop["dp"] = np.arange(cm.size)

        n_points = 2
        assert cm.ndim == 2  # Test code assumption
        cm[0, :n_points].dp = -1
        cm2 = cm[cm.dp < 0]

        assert cm2.size == n_points
        assert np.sum(cm2.dp) == -2

    def test_get_by_multiple_conditions(self, crystal_map, phase_list):
        cm = crystal_map

        assert phase_list.phase_ids == [0, 1, 2]  # Test code assumption

        cm.phases = phase_list
        cm.prop["dp"] = np.arange(cm.size)
        a_phase_id = phase_list.phase_ids[0]
        cm[cm.dp > 3].phase_id = a_phase_id

        condition1 = cm.dp > 3
        condition2 = cm.phase_id == a_phase_id
        cm2 = cm[condition1 & condition2]
        assert cm2.size == np.sum(condition1 * condition2)
        assert np.allclose(cm2.is_in_data, condition1 * condition2)

    @pytest.mark.parametrize("point_id, raises", [(0, False), (1, False), (1000, True)])
    def test_get_by_integer(self, crystal_map, point_id, raises):
        if raises:
            with pytest.raises(IndexError, match=f"{point_id} is out of bounds for"):
                _ = crystal_map[point_id]
        else:
            point = crystal_map[point_id]
            expected_point = crystal_map[crystal_map.id == point_id]

            assert np.allclose(point.rotations.data, expected_point.rotations.data)
            assert point._coordinates == expected_point._coordinates


class TestCrystalMapSetAttributes:
    def test_set_scan_unit(self, crystal_map):
        cm = crystal_map
        assert cm.scan_unit == "px"

        micron = "um"
        cm.scan_unit = micron
        assert cm.scan_unit == micron

    @pytest.mark.parametrize("set_phase_id", [1, -1])
    def test_set_phase_ids(self, crystal_map, set_phase_id):
        cm = crystal_map

        phase_ids = cm.phase_id
        condition = cm.x > 1.5
        cm[condition].phase_id = set_phase_id
        phase_ids[condition] = set_phase_id

        assert np.allclose(cm.phase_id, phase_ids)

        if set_phase_id == -1:
            assert "not_indexed" in cm.phases.names

    @pytest.mark.parametrize("set_phase_id, index_error", [(-1, False), (2, True)])
    def test_set_phase_id_with_unknown_id(self, crystal_map, set_phase_id, index_error):
        cm = crystal_map

        condition = cm.x > 1.5
        phase_ids = cm.phases.phase_ids  # Get before adding a new phase

        if index_error:
            with pytest.raises(IndexError, match="list index out of range"):
                # `set_phase_id` ID is not in `self.phases.phase_ids`
                cm[condition].phase_id = set_phase_id
                _ = cm.__repr__()

            # Add unknown ID to phase list to fix `self.__repr__()`
            cm.phases["a"] = 432
        else:
            cm[condition].phase_id = set_phase_id

        _ = cm.__repr__()

        new_phase_ids = phase_ids + [set_phase_id]
        new_phase_ids.sort()
        assert cm.phases.phase_ids == new_phase_ids

    def test_phases_in_data(self, crystal_map, phase_list):
        cm = crystal_map
        cm.phases = phase_list

        assert cm.phases_in_data.names != cm.phases.names

        ids_not_in_data = np.setdiff1d(
            np.array(cm.phases.phase_ids), np.array(cm.phases_in_data.phase_ids)
        )
        condition1 = cm.x > 1.5
        condition2 = cm.y > 1.5
        for new_id, condition in zip(ids_not_in_data, [condition1, condition2]):
            cm[condition].phase_id = new_id

        assert cm.phases_in_data.names == cm.phases.names


class TestCrystalMapOrientations:
    def test_orientations(self, crystal_map_input, phase_list):
        x = crystal_map_input["x"]

        # Create phase ID array, ensuring all are present
        phase_ids = np.random.choice(phase_list.phase_ids, x.size)
        for i, unique_id in enumerate(phase_list.phase_ids):
            phase_ids[i] = unique_id

        cm = CrystalMap(phase_id=phase_ids, **crystal_map_input)

        # Set phases and make sure all are in data
        cm.phases = phase_list
        assert cm.phases_in_data.names == cm.phases.names

        # Ensure all points have a valid orientation
        orientations_size = 0
        for phase_id in phase_list.phase_ids:
            o = cm[cm.phase_id == phase_id].orientations
            assert isinstance(o, Orientation)
            orientations_size += o.size

        assert orientations_size == cm.size

    def test_getting_orientations_none_symmetry_raises(self, crystal_map_input):
        cm = CrystalMap(**crystal_map_input)

        assert cm.phases.symmetries == [None]

        with pytest.raises(TypeError, match="'NoneType' object is not iterable"):
            _ = cm.orientations

    def test_getting_orientations_multiple_phases_raises(self, crystal_map, phase_list):
        cm = crystal_map

        cm.phases = phase_list
        cm[cm.x > 1.5].phase_id = 2

        with pytest.raises(ValueError, match="Data has the phases "):
            _ = cm.orientations

    @pytest.mark.parametrize(
        "crystal_map_input, rotations_per_point",
        [(((1, 5, 5), (0, 1.5, 1.5), 3), 3)],
        indirect=["crystal_map_input"],
    )
    def test_multiple_orientations_per_point(
        self, crystal_map_input, rotations_per_point
    ):
        cm = CrystalMap(**crystal_map_input)

        cm.phases[1].symmetry = "m-3m"

        assert cm.rotations_per_point == rotations_per_point
        assert cm.orientations.size == cm.size


class TestCrystalMapProp:
    def test_add_new_crystal_map_property(self, crystal_map):
        cm = crystal_map

        prop_name = "iq"
        prop_values = np.arange(cm.size)
        cm.prop[prop_name] = prop_values

        assert np.allclose(cm.prop.get(prop_name), prop_values)
        assert np.allclose(cm.prop[prop_name], prop_values)

    def test_overwrite_crystal_map_property_values(self, crystal_map):
        cm = crystal_map

        prop_name = "iq"
        prop_values = np.arange(cm.size)
        cm.prop[prop_name] = prop_values

        assert np.allclose(cm.prop[prop_name], prop_values)

        new_prop_value = -1
        cm.__setattr__(prop_name, new_prop_value)

        assert np.allclose(cm.prop[prop_name], np.ones(cm.size) * new_prop_value)


class TestCrystalMapMasking:
    def test_getitem_with_masking(self, crystal_map_input):
        x = crystal_map_input["x"]
        props = {"iq": np.arange(x.size)}
        cm = CrystalMap(prop=props, **crystal_map_input)

        cm2 = cm[cm.iq > 1]

        assert np.allclose(cm2.prop.id, cm2.id)
        assert np.allclose(cm2.prop.is_in_data, cm2.is_in_data)


class TestCrystalMapGetMapData:
    @pytest.mark.parametrize(
        "crystal_map_input, to_get, expected_array",
        [
            (
                ((1, 4, 4), (0, 0.5, 1), 2),
                "x",
                np.array([0, 1, 2, 3] * 4).reshape((4, 4)),
            ),
            (
                ((1, 4, 4), (0, 0.5, 1), 2),
                "y",
                np.array([[i * 0.5] * 4 for i in range(4)]),  # [0, 0, 0, 0, 0.5, ...]
            ),
            (
                ((2, 4, 4), (0.28, 0.5, 1), 2),
                "z",
                np.stack((np.zeros((4, 4)), np.ones((4, 4)) * 0.28), axis=0),
            ),
        ],
        indirect=["crystal_map_input"],
    )
    def test_get_coordinate_array(self, crystal_map_input, to_get, expected_array):
        cm = CrystalMap(**crystal_map_input)

        # Get via string
        data_via_string = cm.get_map_data(to_get)
        assert np.allclose(data_via_string, expected_array)

        # Get via numpy array
        if to_get == "x":
            data_via_array = cm.get_map_data(cm.x)
        elif to_get == "y":
            data_via_array = cm.get_map_data(cm.y)
        else:  # to_get == "z"
            data_via_array = cm.get_map_data(cm.z)
        assert np.allclose(data_via_array, expected_array)

        # Make sure they are the same
        assert np.allclose(data_via_array, data_via_string)

    def test_get_none_raises(self, crystal_map):
        item = "z"
        with pytest.raises(ValueError, match=f"{item} is None."):
            _ = crystal_map.get_map_data(item)

    def test_get_property_array(self, crystal_map):
        cm = crystal_map

        expected_array = np.arange(cm.size)
        prop_name = "iq"
        cm.prop[prop_name] = expected_array

        iq = cm.get_map_data(prop_name)

        assert np.allclose(iq, expected_array.reshape(cm.shape))

    @pytest.mark.parametrize(
        "crystal_map_input",
        [((1, 3, 2), (0, 1, 1), 3), ((2, 1, 2), (1, 1, 1), 1),],
        indirect=["crystal_map_input"],
    )
    def test_get_orientations_array(self, crystal_map_input, phase_list):
        cm = CrystalMap(**crystal_map_input)

        cm[3].phase_id = 2
        # Test code assumption
        assert np.allclose(np.unique(cm.phase_id), np.array([1, 2]))
        cm.phases = phase_list

        # Get all with string
        o = cm.get_map_data("orientations")

        # Get per phase with string
        o1 = cm[cm.phase_id == 1].get_map_data("orientations")
        o2 = cm[cm.phase_id == 2].get_map_data("orientations")

        # Merge the two orientation arrays
        o3 = o1
        mask = ~np.isnan(o2)
        o3[mask] = o2[mask]

        assert np.allclose(o, o3, atol=1e-3)

        # Do calculations "manually"
        data_shape = (cm.size, 3)
        array = np.zeros(data_shape)

        if cm.rotations_per_point > 1:
            rotations = cm.rotations[:, 0]
        else:
            rotations = cm.rotations

        for i, phase in cm.phases_in_data:
            phase_mask = cm._phase_id == i
            phase_mask_in_data = cm.phase_id == i
            array[phase_mask] = (
                Orientation(rotations[phase_mask_in_data])
                .set_symmetry(phase.symmetry)
                .to_euler()
            )

        assert np.allclose(o, array.reshape(o.shape), atol=1e-3)

    @pytest.mark.parametrize(
        "crystal_map_input",
        [((1, 2, 2), (0, 1, 1), 2), ((3, 2, 2), (1, 1, 1), 1),],
        indirect=["crystal_map_input"],
    )
    def test_get_rotations_array(self, crystal_map_input):
        cm = CrystalMap(**crystal_map_input)

        # Get with string
        r = cm.get_map_data("rotations")

        new_shape = cm.rotations_shape + (3,)
        expected_array = cm.rotations.to_euler().reshape(*new_shape)
        assert np.allclose(r, expected_array, atol=1e-3)

        # Get with array (RGB)
        new_shape2 = (cm.size, 3)
        r2 = cm.get_map_data(r.reshape(*new_shape2))
        assert np.allclose(r2, expected_array, atol=1e-3)


class TestCrystalMapRepresentation:
    def test_representation(self, crystal_map, phase_list):
        cm = crystal_map
        cm.phases = phase_list
        cm.scan_unit = "nm"

        assert phase_list.phase_ids == [0, 1, 2]  # Test code assumption

        cm[0].phase_id = phase_list.phase_ids[0]
        cm[1].phase_id = phase_list.phase_ids[2]

        cm.prop["iq"] = np.arange(cm.size)

        assert cm[cm.phase_id == -1].__repr__() == "No data."

        print(cm.__repr__())

        assert cm.__repr__() == (
            "Phase   Orientations   Name  Symmetry   Color\n"
            "    0       1 (8.3%)      a      m-3m       r\n"
            "    1     10 (83.3%)      b       432       g\n"
            "    2       1 (8.3%)      c         3       b\n"
            "Properties: iq\n"
            "Scan unit: nm"
        )


class TestCrystalMapCopying:
    def test_shallowcopy_crystal_map(self, crystal_map_input):
        map_size = crystal_map_input["x"].size
        phase_ids = np.arange(map_size)

        cm = CrystalMap(phase_id=phase_ids, **crystal_map_input)

        cm2 = cm[:]  # Everything except `is_in_data` is shallow copied
        cm3 = cm  # These are the same objects (of course)

        assert np.may_share_memory(cm2._phase_id, cm._phase_id)
        assert np.may_share_memory(cm2._rotations.data, cm._rotations.data)

        cm[5].phase_id = -2
        assert np.allclose(cm2.phase_id, cm.phase_id)
        assert np.allclose(cm3.phase_id, cm.phase_id)

        # The user is strictly speaking only supposed to change this via __getitem__()
        cm.is_in_data[2] = False
        assert cm2.size != cm.size
        assert cm3.size == cm.size
        assert np.allclose(cm2.is_in_data, cm.is_in_data) is False
        assert np.may_share_memory(cm3.is_in_data, cm.is_in_data)

    def test_deepcopy_crystal_map(self, crystal_map_input):
        map_size = crystal_map_input["x"].size
        phase_ids = np.arange(map_size)

        cm = CrystalMap(phase_id=phase_ids, **crystal_map_input)

        cm2 = cm.deepcopy()

        cm[5].phase_id = -2
        assert np.allclose(cm2.phase_id, cm.phase_id) is False
        assert np.may_share_memory(cm2._phase_id, cm._phase_id) is False
