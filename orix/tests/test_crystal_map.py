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

import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
import pytest

from orix.crystal_map import CrystalMap, Phase, PhaseList, create_coordinate_arrays
from orix.crystal_map.crystal_map import _data_slices_from_coordinates
from orix.plot import CrystalMapPlot
from orix.quaternion import Orientation, Rotation
from orix.quaternion.symmetry import C2, C3, C4, O

# Note that many parts of the CrystalMap() class are tested while
# testing IO and the Phase() and PhaseList() classes


class TestCrystalMap:
    def test_minimal_init(self, rotations):
        map_size = 2

        assert isinstance(rotations, Rotation)

        xmap = CrystalMap(rotations=rotations)

        assert np.allclose(xmap.x, np.arange(map_size))
        assert xmap.size == map_size
        assert xmap.shape == (map_size,)
        assert xmap.ndim
        assert np.allclose(xmap.id, np.arange(map_size))
        assert isinstance(xmap.rotations, Rotation)
        assert np.allclose(xmap.rotations.data, rotations.data)

    def test_init_with_invalid_rotations(self, rotations):
        with pytest.raises(ValueError):
            _ = CrystalMap(rotations=rotations.data)

    @pytest.mark.parametrize(
        (
            "crystal_map_input, expected_shape, expected_size, expected_step_sizes, "
            "expected_rotations_per_point, expected_coords_nan"
        ),
        [
            (
                ((5, 5), (1.5, 1.5), 3, [0]),
                (5, 5),
                25,
                {"y": 1.5, "x": 1.5},
                3,
                (False, False),
            ),
            (
                ((1, 10), (0.1, 0.1), 1, [0]),
                (10,),
                10,
                {"y": 0, "x": 0.1},
                1,
                (True, False),
            ),
            (
                ((10, 1), (1e-3, 0), 2, [0]),
                (10,),
                10,
                {"y": 1e-3, "x": 0},
                2,
                (False, True),
            ),
        ],
        indirect=["crystal_map_input"],
    )
    def test_init_with_different_coordinate_arrays(
        self,
        crystal_map_input,
        expected_shape,
        expected_size,
        expected_step_sizes,
        expected_rotations_per_point,
        expected_coords_nan,
    ):
        xmap = CrystalMap(**crystal_map_input)
        coordinate_arrays = [crystal_map_input["y"], crystal_map_input["x"]]

        assert xmap.shape == expected_shape
        assert xmap.size == expected_size
        assert xmap._step_sizes == expected_step_sizes
        assert xmap.rotations_per_point == expected_rotations_per_point

        for actual_coords, expected_coords, expected_nan in zip(
            xmap._coordinates.values(), coordinate_arrays, expected_coords_nan
        ):
            if actual_coords is None:
                assert expected_nan
            else:
                assert np.allclose(actual_coords, expected_coords)

    def test_init_map_with_props(self, crystal_map_input):
        x = crystal_map_input["x"]
        props = {"iq": np.arange(x.size)}
        xmap = CrystalMap(prop=props, **crystal_map_input)

        assert xmap.prop == props
        assert np.allclose(xmap.prop.id, xmap.id)
        assert np.allclose(xmap.prop.is_in_data, xmap.is_in_data)

    @pytest.mark.parametrize(
        "crystal_map_input",
        [
            ((4, 3), (1.5, 1.5), 1, [0, 1]),
            ((4, 3), (1.5, 1.5), 1, [1, -1]),
        ],
        indirect=["crystal_map_input"],
    )
    def test_init_with_phase_id(self, crystal_map_input):
        phase_id = crystal_map_input["phase_id"]
        xmap = CrystalMap(**crystal_map_input)

        assert np.allclose(xmap.phase_id, phase_id)
        # Test all_indexed
        if -1 in np.unique(phase_id):
            assert not xmap.all_indexed
            assert -1 in xmap.phases.ids
        else:
            assert xmap.all_indexed
            assert -1 not in xmap.phases.ids

    @pytest.mark.parametrize(
        "crystal_map_input, expected_presence",
        [
            (((4, 3), (1.5, 1.5), 1, [0, 1, 2]), [True, True, True]),
            (((4, 3), (1.5, 1.5), 1, [0, 1, 2, 3]), [True, True, True, True]),
            (((4, 3), (1.5, 1.5), 1, [2, 42]), [True, False]),
        ],
        indirect=["crystal_map_input"],
    )
    def test_init_with_phase_list(self, crystal_map_input, expected_presence):
        point_groups = [C2, C3, C4]
        phase_list = PhaseList(point_groups=point_groups)
        xmap = CrystalMap(phase_list=phase_list, **crystal_map_input)

        n_point_groups = len(point_groups)
        n_phase_ids = len(xmap.phases.ids)
        n_different = n_point_groups - n_phase_ids
        if n_different < 0:
            point_groups += [None] * abs(n_different)

        for p1, p2, expected in zip(
            xmap.phases.point_groups, point_groups, expected_presence
        ):
            assert (p1 == p2) == expected

        unique_phase_ids = list(np.unique(crystal_map_input["phase_id"]).astype(int))
        assert xmap.phases.ids == unique_phase_ids

    def test_init_with_sparse_phase_list(self):
        """Initialize a map with a sparse phase list only containing one
        of the phase IDs passed in the phase list.
        """
        pl = PhaseList(ids=[1], names=["b"], space_groups=[225])
        phase_id = np.array([0, 0, 1, 1, 1, 3])
        xmap = CrystalMap(
            rotations=Rotation.identity(phase_id.size),
            is_in_data=phase_id == 1,
            phase_list=pl,
            phase_id=phase_id,
        )
        for i, name, color, sg_name in zip(
            [0, 1, 3],
            ["", "b", ""],
            ["tab:orange", "tab:blue", "tab:green"],
            [None, "Fm-3m", None],
        ):
            assert xmap.phases[i].name == name
            assert xmap.phases[i].color == color
            if sg_name is None:
                assert xmap.phases[i].space_group is None
            else:
                assert xmap.phases[i].space_group.short_name == sg_name

    def test_init_with_single_point_group(self, crystal_map_input):
        point_group = O
        phase_list = PhaseList(point_groups=point_group)
        xmap = CrystalMap(phase_list=phase_list, **crystal_map_input)
        assert np.allclose(xmap.phases.point_groups[0].data, point_group.data)

    @pytest.mark.parametrize(
        "crystal_map_input, phase_names, phase_ids, desired_phase_names",
        [
            (((7, 4), (1, 1), 1, [0]), ["a", "b", "c"], [0, 1, 2], ["a"]),
            (((7, 4), (1, 1), 1, [0, 1]), ["a", "b", "c"], [0, 2, 1], ["a", "c"]),
            (((7, 4), (1, 1), 1, [0, 2]), ["a", "b", "c"], [0, 2, 1], ["a", "b"]),
            (((7, 4), (1, 1), 1, [3]), ["a", "b", "c"], [0, 2, 1], ["a"]),
        ],
        indirect=["crystal_map_input"],
    )
    def test_init_with_too_many_phases(
        self, crystal_map_input, phase_names, phase_ids, desired_phase_names
    ):
        """More phases than phase IDs."""
        phase_list = PhaseList(names=phase_names, ids=phase_ids)
        xmap = CrystalMap(phase_list=phase_list, **crystal_map_input)

        assert xmap.phases.names == desired_phase_names

    @pytest.mark.parametrize(
        "shape, step_sizes, desired_coordinates, desired_step_sizes",
        [
            (
                (5, 5),
                None,
                dict(
                    y=np.sort(np.tile(np.arange(5) * 1, 5)).flatten(),
                    x=np.tile(np.arange(5), 5).flatten(),
                ),
                dict(y=1, x=1),
            ),
            (
                (2, 3),
                (3, 4),
                dict(
                    y=np.sort(np.tile(np.arange(2) * 3, 3)).flatten(),
                    x=np.tile(np.arange(3) * 4, 2).flatten(),
                ),
                dict(y=3, x=4),
            ),
            (
                None,  # Default (5, 10)
                None,  # Default (1, 1)
                dict(
                    y=np.sort(np.tile(np.arange(5), 10)).flatten(),
                    x=np.tile(np.arange(10), 5).flatten(),
                ),
                dict(y=1, x=1),
            ),
            (
                (5,),
                (1,),
                dict(y=None, x=np.tile(np.arange(5), 1).flatten()),
                dict(y=0, x=1),
            ),
        ],
    )
    def test_class_method_empty(
        self, shape, step_sizes, desired_coordinates, desired_step_sizes
    ):
        xmap = CrystalMap.empty(shape=shape, step_sizes=step_sizes)
        assert xmap.scan_unit == "px"
        if shape is None:
            shape = (5, 10)  # Default
        desired_size = np.prod(shape)
        assert np.allclose(
            xmap.rotations.data,
            np.array([1, 0, 0, 0] * desired_size).reshape(desired_size, 4),
        )
        assert xmap.shape == shape
        for i in ["y", "x"]:
            assert xmap._step_sizes[i] == desired_step_sizes[i]
            coords = xmap._coordinates[i]
            desired_coords = desired_coordinates[i]
            if coords is None:
                assert coords == desired_coords
            else:
                assert np.allclose(coords, desired_coords)

    def test_is_in_data(self, crystal_map_input):
        """Ensure `CrystalMap._original_shape` is correct when set upon
        initialization of a `CrystalMap` where some points are not in
        the data.
        """
        map_shape = (4, 3)
        is_in_data = np.ones(map_shape, dtype=bool)

        # All points in data
        xmap = CrystalMap(is_in_data=is_in_data.flatten(), **crystal_map_input)
        assert xmap.shape == xmap._original_shape == map_shape
        assert np.allclose(xmap.get_map_data("phase_id"), np.zeros(is_in_data.shape))

        # First row not in data
        is_in_data1 = is_in_data.copy()
        is_in_data1[0, :] = False
        xmap1 = CrystalMap(is_in_data=is_in_data1.flatten(), **crystal_map_input)
        new_shape1 = (map_shape[0] - 1, map_shape[1])
        assert xmap1.shape == new_shape1
        assert xmap1._original_shape == map_shape
        assert np.allclose(xmap1.get_map_data("phase_id"), np.zeros(new_shape1))

        # Second row not in data
        is_in_data2 = is_in_data.copy()
        is_in_data2[1, :] = False
        xmap2 = CrystalMap(is_in_data=is_in_data2.flatten(), **crystal_map_input)
        assert xmap2.shape == xmap2._original_shape == map_shape
        desired_phase_id = np.zeros(is_in_data.shape)
        desired_phase_id[1, :] = np.nan
        assert np.allclose(
            xmap2.get_map_data("phase_id"), desired_phase_id, equal_nan=True
        )

        # Last column not in data
        is_in_data3 = is_in_data.copy()
        is_in_data3[:, -1] = False
        xmap3 = CrystalMap(is_in_data=is_in_data3.flatten(), **crystal_map_input)
        new_shape3 = (map_shape[0], map_shape[1] - 1)
        assert xmap3.shape == new_shape3
        assert xmap3._original_shape == map_shape
        assert np.allclose(xmap3.get_map_data("phase_id"), np.zeros(new_shape3))

    def test_set_phases(self, phase_list):
        assert phase_list.size == 3
        xmap = CrystalMap.empty((2, 3))

        # OK
        xmap._phase_id = np.array([0, 1, 2, 0, 1, 2])
        xmap.phases = phase_list

        # Not OK
        xmap._phase_id = np.array([0, 1, 2, 1, 2, 3])
        with pytest.raises(ValueError, match="There must be at least as many phases "):
            xmap.phases = phase_list


class TestCrystalMapGetItem:
    @pytest.mark.parametrize(
        "crystal_map_input, slice_tuple, expected_shape",
        [
            (((5, 5), (1, 1), 1, [0]), slice(None, None, None), (5, 5)),
            (
                ((5, 5), (1, 1), 1, [0]),
                (slice(1, 2, None), slice(None, None, None)),
                (1, 5),
            ),
            (
                ((5, 5), (1, 1), 1, [0]),
                (slice(None, None, None), slice(1, 4, None)),
                (5, 3),
            ),
            (
                ((10, 10), (0.5, 0.1), 2, [0]),
                (slice(5, 10, None), slice(None, None, None)),
                (5, 10),
            ),
            (
                ((10, 10), (0.5, 0.1), 2, [0]),
                (slice(2, 4, None), slice(None, 3, None)),
                (2, 3),
            ),
        ],
        indirect=["crystal_map_input"],
    )
    def test_get_by_slice(self, crystal_map_input, slice_tuple, expected_shape):
        xmap = CrystalMap(**crystal_map_input)

        xmap2 = xmap[slice_tuple]
        assert xmap2.shape == expected_shape

    def test_get_by_phase_name(self, crystal_map_input, phase_list):
        x = crystal_map_input["x"]

        # Create phase ID array, ensuring all are present
        phase_ids = np.random.choice(phase_list.ids, x.size)
        for i, unique_id in enumerate(phase_list.ids):
            phase_ids[i] = unique_id

        # Get number of points with each phase ID
        n_points_per_phase = {}
        for phase_i, phase in phase_list:
            n_points_per_phase[phase.name] = len(np.where(phase_ids == phase_i)[0])

        crystal_map_input.pop("phase_id")
        xmap = CrystalMap(phase_id=phase_ids, **crystal_map_input)
        xmap.phases = phase_list

        for (_, phase), (expected_phase_name, n_points) in zip(
            phase_list, n_points_per_phase.items()
        ):
            xmap2 = xmap[phase.name]

            assert xmap2.size == n_points
            assert xmap2.phases_in_data.names == [expected_phase_name]

    def test_get_by_indexed_not_indexed(self, crystal_map):
        xmap = crystal_map

        # Set some points to not_indexed
        xmap[2:4].phase_id = -1

        indexed = xmap["indexed"]
        not_indexed = xmap["not_indexed"]

        assert indexed.size + not_indexed.size == xmap.size
        assert np.allclose(np.unique(not_indexed.phase_id), np.array([-1]))
        assert np.allclose(np.unique(indexed.phase_id), np.array([0]))

    @pytest.mark.parametrize(
        "crystal_map_input",
        [((4, 3), (1, 1), 1, [0])],
        indirect=["crystal_map_input"],
    )
    def test_get_by_condition(self, crystal_map_input):
        xmap = CrystalMap(**crystal_map_input)

        xmap.prop["dp"] = np.arange(xmap.size)

        n_points = 2
        assert xmap.shape == (4, 3)  # Test code assumption
        xmap[0, :n_points].dp = -1
        xmap2 = xmap[xmap.dp < 0]

        assert xmap2.size == n_points
        assert np.sum(xmap2.dp) == -n_points

    def test_get_by_multiple_conditions(self, crystal_map, phase_list):
        xmap = crystal_map

        assert phase_list.ids == [0, 1, 2]  # Test code assumption

        xmap.phases = phase_list
        xmap.prop["dp"] = np.arange(xmap.size)
        a_phase_id = phase_list.ids[0]
        xmap[xmap.dp > 3].phase_id = a_phase_id

        condition1 = xmap.dp > 3
        condition2 = xmap.phase_id == a_phase_id
        xmap2 = xmap[condition1 & condition2]
        assert xmap2.size == np.sum(condition1 * condition2)
        assert np.allclose(xmap2.is_in_data, condition1 * condition2)

    @pytest.mark.parametrize(
        "crystal_map_input, integer_slices, expected_id, raises",
        [
            (((4, 4), (1, 1), 1, [0]), (0, 2), 2, False),
            (((4, 4), (1, 1), 3, [0]), (2, 0), 8, False),
            (((4, 4), (1, 1), 3, [0]), (2, 3), 11, False),
            (((4, 1), (1, 0), 2, [0]), 1, 1, False),
            (((4, 4), (1, 1), 3, [0]), (1000, 0, 0), None, True),
        ],
        indirect=["crystal_map_input"],
    )
    def test_get_by_integer(
        self, crystal_map_input, integer_slices, expected_id, raises
    ):
        # This also tests `phase_id`
        xmap = CrystalMap(**crystal_map_input)
        if raises:
            with pytest.raises(IndexError):
                _ = xmap[integer_slices]
        else:
            point = xmap[integer_slices]
            expected_point = xmap[xmap.id == expected_id]

            assert np.allclose(point.rotations.data, expected_point.rotations.data)
            assert point._coordinates == expected_point._coordinates

    def test_get_twice(self):
        """Indexing into a CrystalMap cropped twice returns the desired
        part of the map.
        """
        xmap = CrystalMap.empty()
        xmap2 = xmap[1:4, 4:9]

        assert np.allclose(xmap.id, np.arange(xmap.size))
        # fmt: off
        assert np.allclose(
            xmap2.id,
            [
                14, 15, 16, 17, 18,
                24, 25, 26, 27, 28,
                34, 35, 36, 37, 38,
            ]
        )
        # fmt: on
        assert np.allclose(xmap2[1:, 1:4].id, [25, 26, 27, 35, 36, 37])
        assert np.allclose(xmap2[1, 2].id, 26)
        assert np.allclose(xmap2[1, 3].id, 27)
        assert np.allclose(xmap2[2, 2].id, 36)

    def test_row_col(self):
        """Map points are assigned correct row and column coordinates."""
        xmap1 = CrystalMap.empty()
        xmap1.phases[0].name = "a"
        xmap1.phases.add(Phase("b"))
        xmap1[1:4, 5:9].phase_id = 1
        xmap1[1, 1].phase_id = 1

        rows, cols = np.indices(xmap1.shape)
        assert np.allclose(xmap1.row, rows.flatten())
        assert np.allclose(xmap1.col, cols.flatten())

        # Sliced (weird formatting to increase readability of 2D map)
        # fmt: off
        assert np.allclose(
            xmap1["a"].row,
            np.array([
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1,    1, 1, 1,             1,
                2, 2, 2, 2, 2,             2,
                3, 3, 3, 3, 3,             3,
                4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
            ])
        )
        assert np.allclose(
            xmap1["b"].row,
            np.array([

                   0,          0, 0, 0, 0,
                               1, 1, 1, 1,
                               2, 2, 2, 2,

            ])
        )
        assert np.allclose(
            xmap1["a"].col,
            np.array([
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                0,    2, 3, 4,             9,
                0, 1, 2, 3, 4,             9,
                0, 1, 2, 3, 4,             9,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            ])
        )
        assert np.allclose(
            xmap1["b"].col,
            np.array([

                   0,          4, 5, 6, 7,
                               4, 5, 6, 7,
                               4, 5, 6, 7,

            ])
        )
        # fmt: on

        # 1D without 1-dimension
        xmap2 = CrystalMap.empty((5,))
        assert np.allclose(xmap2.row, np.zeros(xmap2.size))
        assert np.allclose(xmap2.col, np.arange(xmap2.size))

        # 1D with 1-dimension
        xmap3 = CrystalMap.empty((1, 5))
        assert np.allclose(xmap3.row, np.zeros(xmap3.size))
        assert np.allclose(xmap3.col, np.arange(xmap3.size))
        xmap4 = CrystalMap.empty((5, 1))
        assert np.allclose(xmap4.row, np.arange(xmap4.size))
        assert np.allclose(xmap4.col, np.zeros(xmap4.size))


class TestCrystalMapSetAttributes:
    def test_set_scan_unit(self, crystal_map):
        xmap = crystal_map
        assert xmap.scan_unit == "px"

        micron = "um"
        xmap.scan_unit = micron
        assert xmap.scan_unit == micron

    @pytest.mark.parametrize("set_phase_id", [1, -1])
    def test_set_phase_ids(self, crystal_map, set_phase_id):
        xmap = crystal_map

        phase_ids = xmap.phase_id
        condition = xmap.x > 1.5
        xmap[condition].phase_id = set_phase_id
        phase_ids[condition] = set_phase_id

        assert np.allclose(xmap.phase_id, phase_ids)

        if set_phase_id == -1:
            assert "not_indexed" in xmap.phases.names

    def test_set_phase_ids_raises(self, crystal_map):
        with pytest.raises(ValueError, match="NumPy boolean array indexing assignment"):
            crystal_map[1, 1].phase_id = -1 * np.ones(10)

    @pytest.mark.parametrize("set_phase_id, index_error", [(-1, False), (1, True)])
    def test_set_phase_id_with_unknown_id(self, crystal_map, set_phase_id, index_error):
        xmap = crystal_map

        condition = xmap.x > 1.5
        phase_ids = xmap.phases.ids  # Get before adding a new phase

        if index_error:
            with pytest.raises(IndexError, match="list index out of range"):
                # `set_phase_id` ID is not in `self.phases.phase_ids`
                xmap[condition].phase_id = set_phase_id
                _ = repr(xmap)

            # Add unknown ID to phase list to fix `repr(self)`
            xmap.phases.add(Phase("a", point_group=432))  # Add phase with ID 1
        else:
            xmap[condition].phase_id = set_phase_id

        _ = repr(xmap)

        new_phase_ids = phase_ids + [set_phase_id]
        new_phase_ids.sort()
        assert xmap.phases.ids == new_phase_ids

    def test_phases_in_data(self, crystal_map, phase_list):
        xmap = crystal_map
        xmap.phases = phase_list

        assert xmap.phases_in_data.names != xmap.phases.names

        ids_not_in_data = np.setdiff1d(
            np.array(xmap.phases.ids), np.array(xmap.phases_in_data.ids)
        )
        condition1 = xmap.x > 1.5
        condition2 = xmap.y > 1.5
        for new_id, condition in zip(ids_not_in_data, [condition1, condition2]):
            xmap[condition].phase_id = new_id

        assert xmap.phases_in_data.names == xmap.phases.names


class TestCrystalMapOrientations:
    def test_orientations(self, crystal_map_input, phase_list):
        x = crystal_map_input["x"]

        # Create phase ID array, ensuring all are present
        phase_ids = np.random.choice(phase_list.ids, x.size)
        for i, unique_id in enumerate(phase_list.ids):
            phase_ids[i] = unique_id

        crystal_map_input.pop("phase_id")
        xmap = CrystalMap(phase_id=phase_ids, **crystal_map_input)

        # Set phases and make sure all are in data
        xmap.phases = phase_list
        assert xmap.phases_in_data.names == xmap.phases.names

        # Ensure all points have a valid orientation
        orientations_size = 0
        for phase_id in phase_list.ids:
            o = xmap[xmap.phase_id == phase_id].orientations
            assert isinstance(o, Orientation)
            orientations_size += o.size

        assert orientations_size == xmap.size

    @pytest.mark.parametrize(
        "point_group, rotation, expected_orientation",
        [
            (C2, [(0.6088, 0, 0, 0.7934)], [(-0.7934, 0, 0, 0.6088)]),
            (C3, [(0.6088, 0, 0, 0.7934)], [(-0.9914, 0, 0, 0.1305)]),
            (C4, [(0.6088, 0, 0, 0.7934)], [(-0.9914, 0, 0, -0.1305)]),
            (O, [(0.6088, 0, 0, 0.7934)], [(-0.9914, 0, 0, -0.1305)]),
        ],
    )
    def test_orientations_symmetry(self, point_group, rotation, expected_orientation):
        r = Rotation(rotation)
        xmap = CrystalMap(rotations=r, phase_id=np.array([0]))
        xmap.phases = PhaseList(Phase("a", point_group=point_group))

        o = xmap.orientations
        o = o.map_into_symmetry_reduced_zone()

        o1 = Orientation(r)
        o1.symmetry = point_group
        o1 = o1.map_into_symmetry_reduced_zone()

        assert np.allclose(o.data, o1.data, atol=1e-3)
        assert np.allclose(o.data, expected_orientation, atol=1e-3)

    def test_orientations_none_symmetry_raises(self, crystal_map_input):
        xmap = CrystalMap(**crystal_map_input)
        assert xmap.phases[:].point_group is None
        with pytest.raises(TypeError, match="Value must be an instance of"):
            _ = xmap.orientations

    def test_orientations_multiple_phases_raises(self, crystal_map, phase_list):
        xmap = crystal_map

        xmap.phases = phase_list
        xmap[xmap.x > 1.5].phase_id = 2

        with pytest.raises(ValueError, match="Data has the phases "):
            _ = xmap.orientations

    @pytest.mark.parametrize(
        "crystal_map_input, rotations_per_point",
        [(((5, 5), (1.5, 1.5), 3, [0]), 3)],
        indirect=["crystal_map_input"],
    )
    def test_multiple_orientations_per_point(
        self, crystal_map_input, rotations_per_point
    ):
        xmap = CrystalMap(**crystal_map_input)

        assert xmap.phases.ids == [0]  # Test code assumption
        xmap.phases[0].point_group = "m-3m"

        assert xmap.rotations_per_point == rotations_per_point
        assert xmap.orientations.size == xmap.size


class TestCrystalMapProp:
    def test_add_new_crystal_map_property(self, crystal_map):
        xmap = crystal_map

        prop_name = "iq"
        prop_values = np.arange(xmap.size)
        xmap.prop[prop_name] = prop_values

        assert np.allclose(xmap.prop.get(prop_name), prop_values)
        assert np.allclose(xmap.prop[prop_name], prop_values)

    def test_overwrite_crystal_map_property_values(self, crystal_map):
        xmap = crystal_map

        prop_name = "iq"
        prop_values = np.arange(xmap.size)
        xmap.prop[prop_name] = prop_values

        assert np.allclose(xmap.prop[prop_name], prop_values)

        new_prop_value = -1
        xmap.__setattr__(prop_name, new_prop_value)

        assert np.allclose(xmap.prop[prop_name], np.ones(xmap.size) * new_prop_value)


class TestCrystalMapMasking:
    def test_getitem_with_masking(self, crystal_map_input):
        x = crystal_map_input["x"]
        props = {"iq": np.arange(x.size)}
        xmap = CrystalMap(prop=props, **crystal_map_input)

        xmap2 = xmap[xmap.iq > 1]

        assert np.allclose(xmap2.prop.id, xmap2.id)
        assert np.allclose(xmap2.prop.is_in_data, xmap2.is_in_data)


class TestCrystalMapGetMapData:
    @pytest.mark.parametrize(
        "crystal_map_input, to_get, expected_array",
        [
            (
                ((4, 4), (0.5, 1), 2, [0]),
                "x",
                np.array([0, 1, 2, 3] * 4).reshape(4, 4),
            ),
            (
                ((4, 4), (0.5, 1), 2, [0]),
                "y",
                np.array([[i * 0.5] * 4 for i in range(4)]),  # [0, 0, 0, 0, 0.5, ...]
            ),
        ],
        indirect=["crystal_map_input"],
    )
    def test_get_coordinate_array(self, crystal_map_input, to_get, expected_array):
        xmap = CrystalMap(**crystal_map_input)

        # Get via string
        data_via_string = xmap.get_map_data(to_get)
        assert np.allclose(data_via_string, expected_array)

        # Get via numpy array
        if to_get == "x":
            data_via_array = xmap.get_map_data(xmap.x)
        else:
            data_via_array = xmap.get_map_data(xmap.y)
        assert np.allclose(data_via_array, expected_array)

        # Make sure they are the same
        assert np.allclose(data_via_array, data_via_string)

    def test_get_property_array(self, crystal_map):
        xmap = crystal_map

        expected_array = np.arange(xmap.size)
        prop_name = "iq"
        xmap.prop[prop_name] = expected_array

        iq = xmap.get_map_data(prop_name)

        assert np.allclose(iq, expected_array.reshape(xmap.shape))

    @pytest.mark.parametrize(
        "crystal_map_input",
        [((3, 2), (1, 1), 3, [0]), ((3, 2), (1, 1), 1, [0])],
        indirect=["crystal_map_input"],
    )
    def test_get_orientations_array(self, crystal_map_input, phase_list):
        xmap = CrystalMap(**crystal_map_input)

        xmap[:2, 0].phase_id = 1
        # Test code assumption
        id1 = 0
        id2 = 1
        assert np.allclose(np.unique(xmap.phase_id), np.array([id1, id2]))
        xmap.phases = phase_list

        # Get all with string
        o = xmap.get_map_data("orientations")

        # Get per phase with string
        xmap1 = xmap[xmap.phase_id == id1]
        xmap2 = xmap[xmap.phase_id == id2]
        o1 = xmap1.get_map_data("orientations")
        o2 = xmap2.get_map_data("orientations")

        expected_o1 = xmap1.orientations.to_euler()
        expected_shape = expected_o1.shape
        assert np.allclose(
            o1[~np.isnan(o1)].reshape(expected_shape), expected_o1, atol=1e-3
        )

        expected_o2 = xmap2.orientations.to_euler()
        expected_shape = expected_o2.shape
        assert np.allclose(
            o2[~np.isnan(o2)].reshape(expected_shape), expected_o2, atol=1e-3
        )

        # Do calculations "manually"
        data_shape = (xmap.size, 3)
        array = np.zeros(data_shape)

        if xmap.rotations_per_point > 1:
            rotations = xmap.rotations[:, 0]
        else:
            rotations = xmap.rotations

        for i, phase in xmap.phases_in_data:
            phase_mask = xmap._phase_id == i
            phase_mask_in_data = xmap.phase_id == i
            oi = Orientation(rotations[phase_mask_in_data])
            oi.symmetry = phase.point_group
            array[phase_mask] = oi.to_euler()

        assert np.allclose(o, array.reshape(o.shape), atol=1e-3)

    @pytest.mark.parametrize(
        "crystal_map_input",
        [((2, 2), (1, 1), 2, [0]), ((2, 2), (1, 1), 1, [0])],
        indirect=["crystal_map_input"],
    )
    def test_get_rotations_array(self, crystal_map_input):
        xmap = CrystalMap(**crystal_map_input)

        # Get with string
        r = xmap.get_map_data("rotations")

        new_shape = xmap.shape + (3,)
        if xmap.rotations_per_point > 1:
            expected_array = xmap.rotations[:, 0].to_euler().reshape(new_shape)
        else:
            expected_array = xmap.rotations.to_euler().reshape(new_shape)
        assert np.allclose(r, expected_array, atol=1e-3)

        # Get with array (RGB)
        new_shape2 = (xmap.size, 3)
        r2 = xmap.get_map_data(r.reshape(new_shape2))
        assert np.allclose(r2, expected_array, atol=1e-3)

    @pytest.mark.parametrize(
        "crystal_map_input",
        [((9, 3), (1.5, 1.5), 2, [0]), ((10, 5), (0.1, 0.1), 3, [0])],
        indirect=["crystal_map_input"],
    )
    def test_get_phase_id_array_from_3d_data(self, crystal_map_input):
        xmap = CrystalMap(**crystal_map_input)
        _ = xmap.get_map_data(xmap.phase_id)

    @pytest.mark.parametrize(
        "crystal_map_input, to_get",
        [
            (((1, 3), (1, 1), 1, [0]), "y"),
            (((3, 1), (1, 1), 1, [0]), "x"),
        ],
        indirect=["crystal_map_input"],
    )
    def test_get_unknown_string_raises(self, crystal_map_input, to_get):
        xmap = CrystalMap(**crystal_map_input)
        with pytest.raises(ValueError, match=f"{to_get} is None."):
            _ = xmap.get_map_data(to_get)

    def test_get_boolean_array(self, crystal_map):
        xmap = crystal_map

        assert np.issubdtype(xmap.get_map_data("is_indexed").dtype, bool)
        assert np.issubdtype(xmap.get_map_data(xmap.is_in_data).dtype, bool)

    @pytest.mark.parametrize("dtype_in", [np.uint8, int, np.float32, float, bool])
    def test_preserve_dtype(self, crystal_map, dtype_in):
        xmap = crystal_map
        prop_name = "new_prop"
        xmap.prop[prop_name] = np.ones(xmap.size, dtype=dtype_in)

        assert xmap.get_map_data(prop_name).dtype == dtype_in

    @pytest.mark.parametrize("dtype_in", [bool, int])
    def test_not_preserve_dtype(self, crystal_map, dtype_in):
        xmap = crystal_map
        prop_name = "new_prop"
        xmap.prop[prop_name] = np.ones(xmap.size, dtype=dtype_in)
        xmap.is_in_data[0] = False

        assert not xmap.is_in_data.all()
        assert xmap.get_map_data(prop_name, fill_value=None).dtype == float

    def test_get_map_data(self, crystal_map):
        # Test decimals paramter
        crystal_map.prop["iq"] = np.random.random(crystal_map.size)
        iq1 = crystal_map.get_map_data("iq").flatten()
        assert np.allclose(iq1, crystal_map.iq)
        iq2 = crystal_map.get_map_data("iq", decimals=3).flatten()
        assert not np.allclose(iq2, crystal_map.iq)
        assert np.allclose(iq2, crystal_map.iq, atol=1e-3)

        # Test https://github.com/pyxem/orix/issues/220
        xmap2 = CrystalMap.empty((3,))
        _ = xmap2.get_map_data("x")
        _ = xmap2.get_map_data(xmap2.x)


class TestCrystalMapRepresentation:
    def test_representation(self, crystal_map, phase_list):
        xmap = crystal_map
        xmap.phases = phase_list
        xmap.scan_unit = "nm"

        # Test code assumptions
        assert phase_list.ids == [0, 1, 2]
        assert xmap.shape == (4, 3)

        xmap[0, 1].phase_id = phase_list.ids[1]
        xmap[1, 1].phase_id = phase_list.ids[2]

        xmap.prop["iq"] = np.arange(xmap.size)

        assert repr(xmap[xmap.phase_id == -1]) == "No data."

        assert repr(xmap) == (
            "Phase  Orientations  Name  Space group  Point group  Proper point group  "
            "Color\n"
            "    0    10 (83.3%)     a        Im-3m         m-3m                 432  "
            "    r\n"
            "    1      1 (8.3%)     b         P432          432                 432  "
            "    g\n"
            "    2      1 (8.3%)     c           P3            3                   3  "
            "    b\n"
            "Properties: iq\n"
            "Scan unit: nm"
        )


class TestCrystalMapCopying:
    def test_shallowcopy_crystal_map(self, crystal_map):
        xmap2 = crystal_map[:]  # Everything except `is_in_data` is shallow copied
        xmap3 = crystal_map  # These are the same objects (of course)

        assert np.may_share_memory(xmap2._phase_id, crystal_map._phase_id)
        assert np.may_share_memory(xmap2._rotations.data, crystal_map._rotations.data)

        crystal_map[3, 0].phase_id = -2
        assert np.allclose(xmap2.phase_id, crystal_map.phase_id)
        assert np.allclose(xmap3.phase_id, crystal_map.phase_id)

        # The user is strictly speaking only supposed to change this via __getitem__()
        crystal_map.is_in_data[2] = False
        assert xmap2.size != crystal_map.size
        assert xmap3.size == crystal_map.size
        assert np.allclose(xmap2.is_in_data, crystal_map.is_in_data) is False
        assert np.may_share_memory(xmap3.is_in_data, crystal_map.is_in_data)

    def test_deepcopy_crystal_map(self, crystal_map):
        xmap2 = crystal_map.deepcopy()

        crystal_map[3, 0].phase_id = -2
        assert np.allclose(xmap2.phase_id, crystal_map.phase_id) is False
        assert np.may_share_memory(xmap2._phase_id, crystal_map._phase_id) is False


class TestCrystalMapShape:
    @pytest.mark.parametrize(
        "crystal_map_input, expected_slices",
        [
            (
                ((10, 30), (0.1, 0.1), 2, [0]),
                (slice(0, 10, None), slice(0, 30, None)),
            ),
            (
                ((13, 27), (0.7, 0.9), 3, [0]),
                (slice(0, 13, None), slice(0, 27, None)),
            ),
            (
                ((4, 3), (1.5, 1.5), 1, [0]),
                (slice(0, 4, None), slice(0, 3, None)),
            ),
            # Testing rounding 15 / 2 = 7.5 and 45 / 2 = 22.5
            (
                ((15, 45), (2, 2), 1, [0]),
                (slice(0, 15, None), slice(0, 45, None)),
            ),
        ],
        indirect=["crystal_map_input"],
    )
    def test_data_slices_from_coordinates(self, crystal_map_input, expected_slices):
        xmap = CrystalMap(**crystal_map_input)
        assert xmap._data_slices_from_coordinates() == expected_slices

    @pytest.mark.parametrize(
        "crystal_map_input, slices, expected_size, expected_shape, expected_slices",
        [
            # Slice 2D data with an index in all axes
            (
                ((3, 4), (1, 1), 1, [0]),
                (2, 3),
                1,
                (1, 1),
                (slice(2, 3, None), slice(3, 4, None)),
            ),
            # Slice 2D data with an index in only one axis
            (
                ((3, 4), (0.1, 0.1), 1, [0]),
                (1,),
                4,
                (1, 4),
                (slice(1, 2, None), slice(0, 4, None)),
            ),
        ],
        indirect=["crystal_map_input"],
    )
    def test_data_slice_from_coordinates_masked(
        self, crystal_map_input, slices, expected_size, expected_shape, expected_slices
    ):
        xmap = CrystalMap(**crystal_map_input)

        # Mask data
        xmap2 = xmap[slices]

        assert xmap2.size == expected_size
        assert xmap2.shape == expected_shape
        assert xmap2._data_slices_from_coordinates() == expected_slices

    @pytest.mark.parametrize(
        "crystal_map_input, expected_shape",
        [
            (((10, 30), (0.1, 0.1), 2, [0]), (10, 30)),
            (((13, 27), (0.7, 0.9), 3, [0]), (13, 27)),
            (((4, 3), (1.5, 1.5), 1, [0]), (4, 3)),
            (((15, 45), (2, 2), 2, [0]), (15, 45)),
        ],
        indirect=["crystal_map_input"],
    )
    def test_shape_from_coordinates(self, crystal_map_input, expected_shape):
        xmap = CrystalMap(**crystal_map_input)
        assert xmap.shape == expected_shape

    @pytest.mark.parametrize(
        "crystal_map_input, expected_shape",
        [
            (((10, 30), (0.1, 0.1), 2, [0]), (10, 30, 2)),
            (((13, 27), (0.7, 0.9), 3, [0]), (13, 27, 3)),
            (((4, 3), (1.5, 1.5), 5, [0]), (4, 3, 5)),
            (((15, 45), (2, 2), 2, [0]), (15, 45, 2)),
        ],
        indirect=["crystal_map_input"],
    )
    def test_rotation_shape(self, crystal_map_input, expected_shape):
        xmap = CrystalMap(**crystal_map_input)
        assert xmap.rotations_shape == expected_shape

    @pytest.mark.parametrize(
        "crystal_map_input, expected_coordinate_axes",
        [
            (((10, 30), (0.1, 0.1), 1, [0]), {0: "y", 1: "x"}),
            (((13, 27), (0.7, 0.9), 1, [0]), {0: "y", 1: "x"}),
            (((13, 27), (1.5, 1.5), 1, [0]), {0: "y", 1: "x"}),
            (((13, 1), (2, 1), 2, [0]), {0: "y"}),
            (((1, 13), (0, 2), 1, [0]), {0: "x"}),
        ],
        indirect=["crystal_map_input"],
    )
    def test_coordinate_axes(self, crystal_map_input, expected_coordinate_axes):
        xmap = CrystalMap(**crystal_map_input)
        assert xmap._coordinate_axes == expected_coordinate_axes

    def test_data_slices_from_coordinates_no_steps(self):
        d, _ = create_coordinate_arrays((3, 4), step_sizes=(0.1, 0.2))
        slices = _data_slices_from_coordinates(d)
        assert slices == (slice(0, 4, None), slice(0, 3, None))


class TestCrystalMapPlotMethod:
    def test_plot(self, crystal_map):
        xmap = crystal_map
        fig1 = xmap.plot(return_figure=True)
        assert isinstance(fig1.axes[0], CrystalMapPlot)

        sbar1 = fig1.axes[0].artists[0]
        assert isinstance(sbar1, ScaleBar)
        assert sbar1.dimension.base_units == xmap.scan_unit
        assert sbar1.location == 3  # "lower left"

        prop = np.arange(xmap.size)
        location = "upper right"
        fig2 = xmap.plot(
            return_figure=True,
            remove_padding=True,
            overlay=prop,
            scalebar_properties=dict(location=location),
        )
        ax2 = fig2.axes[0]

        # One effect of "removing padding"
        assert ax2._xmargin == 0
        assert ax2._ymargin == 0

        sbar2 = fig2.axes[0].artists[0]
        assert sbar2.location == 1  # "upper right"

        plt.close("all")


class TestCreateCoordinateArrays:
    def test_create_coordinate_arrays_raises(self):
        with pytest.raises(ValueError, match="Can only create coordinate arrays for "):
            _ = create_coordinate_arrays((1, 2, 3))
