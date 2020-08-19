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

from contextlib import contextmanager
from collections import OrderedDict
from io import StringIO
from numbers import Number
import os
import sys

from diffpy.structure import Lattice, Structure
from diffpy.structure.spacegroups import GetSpaceGroup
from h5py import File
import pytest
import numpy as np

from orix import __version__ as orix_version
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.io import (
    load,
    save,
    loadang,
    loadctf,
    _plugin_from_footprints,
    _overwrite_or_not,
)
from orix.io.plugins.ang import (
    _get_header,
    _get_phases_from_header,
    _get_vendor_columns,
)
from orix.io.plugins.orix_hdf5 import (
    hdf5group2dict,
    dict2crystalmap,
    dict2phaselist,
    dict2phase,
    dict2structure,
    dict2lattice,
    dict2atom,
    dict2hdf5group,
    crystalmap2dict,
    phaselist2dict,
    phase2dict,
    structure2dict,
    lattice2dict,
    atom2dict,
)
from orix.io.plugins import ang, emsoft_h5ebsd, orix_hdf5
from orix.quaternion.rotation import Rotation

from orix.tests.conftest import (
    ANGFILE_TSL_HEADER,
    ANGFILE_ASTAR_HEADER,
    ANGFILE_EMSOFT_HEADER,
)

plugin_list = [ang, emsoft_h5ebsd, orix_hdf5]


@contextmanager
def replace_stdin(target):
    orig = sys.stdin
    sys.stdin = target
    yield
    sys.stdin = orig


def assert_dictionaries_are_equal(input_dict, output_dict):
    for key in output_dict.keys():
        output_value = output_dict[key]
        input_value = input_dict[key]
        if isinstance(output_value, (dict, OrderedDict)):
            assert_dictionaries_are_equal(input_value, output_value)
        else:
            if isinstance(output_value, (np.ndarray, Number)):
                assert np.allclose(input_value, output_value)
            elif isinstance(output_value, Rotation):
                assert np.allclose(input_value.to_euler(), output_value.to_euler())
            elif isinstance(output_value, Phase):
                assert_dictionaries_are_equal(
                    input_value.__dict__, output_value.__dict__
                )
            elif isinstance(output_value, PhaseList):
                assert_dictionaries_are_equal(input_value._dict, output_value._dict)
            elif isinstance(output_value, Structure):
                assert np.allclose(output_value.xyz, input_value.xyz)
                assert str(output_value.element) == str(input_value.element)
                assert np.allclose(output_value.occupancy, input_value.occupancy)
            else:
                assert input_value == output_value


class TestGeneralIO:
    def test_load_no_filename_match(self):
        fname = "what_is_hip.ang"
        with pytest.raises(IOError, match=f"No filename matches '{fname}'."):
            _ = load(fname)

    @pytest.mark.parametrize("temp_file_path", ["ctf"], indirect=["temp_file_path"])
    def test_load_unsupported_format(self, temp_file_path):
        np.savetxt(temp_file_path, X=np.random.rand(100, 8))
        with pytest.raises(IOError, match=f"Could not read "):
            _ = load(temp_file_path)

    @pytest.mark.parametrize(
        "top_group, expected_plugin",
        [("Scan 1", emsoft_h5ebsd), ("crystal_map", orix_hdf5), ("Scan 2", None)],
    )
    def test_plugin_from_footprints(self, temp_file_path, top_group, expected_plugin):
        with File(temp_file_path, mode="w") as f:
            f.create_group(top_group)
            assert (
                _plugin_from_footprints(
                    temp_file_path, plugins=[emsoft_h5ebsd, orix_hdf5]
                )
                is expected_plugin
            )

    def test_overwrite_or_not(self, crystal_map, temp_file_path):
        save(temp_file_path, crystal_map)
        with pytest.warns(UserWarning, match="Not overwriting, since your terminal "):
            _overwrite_or_not(temp_file_path)

    @pytest.mark.parametrize(
        "answer, expected", [("y", True), ("n", False), ("m", None)]
    )
    def test_overwrite_or_not_input(
        self, crystal_map, temp_file_path, answer, expected
    ):
        save(temp_file_path, crystal_map)
        if answer == "m":
            with replace_stdin(StringIO(answer)):
                with pytest.raises(EOFError):
                    _overwrite_or_not(temp_file_path)
        else:
            with replace_stdin(StringIO(answer)):
                assert _overwrite_or_not(temp_file_path) is expected

    @pytest.mark.parametrize("temp_file_path", ["angs", "hdf4", "h6"])
    def test_save_unsupported_raises(self, temp_file_path, crystal_map):
        _, ext = os.path.splitext(temp_file_path)
        with pytest.raises(IOError, match=f"'{ext}' does not correspond to any "):
            save(temp_file_path, crystal_map)

    def test_save_overwrite_raises(self, temp_file_path, crystal_map):
        with pytest.raises(ValueError, match="`overwrite` parameter can only be "):
            save(temp_file_path, crystal_map, overwrite=1)

    @pytest.mark.parametrize(
        "overwrite, expected_phase_name", [(True, "hepp"), (False, "")]
    )
    def test_save_overwrite(
        self, temp_file_path, crystal_map, overwrite, expected_phase_name
    ):
        assert crystal_map.phases[0].name == ""
        save(temp_file_path, crystal_map)
        assert os.path.isfile(temp_file_path) is True

        crystal_map.phases[0].name = "hepp"
        save(temp_file_path, crystal_map, overwrite=overwrite)

        crystal_map2 = load(temp_file_path)
        assert crystal_map2.phases[0].name == expected_phase_name


@pytest.mark.parametrize(
    "angfile_astar, expected_data",
    [
        (
            (
                (2, 5),
                (1, 1),
                np.ones(2 * 5, dtype=int),
                np.array(
                    [
                        [4.485496, 0.952426, 0.791507],
                        [1.343904, 0.276111, 0.825890],
                        [1.343904, 0.276111, 0.825890],
                        [1.343904, 0.276111, 0.825890],
                        [4.555309, 2.895152, 3.972020],
                        [1.361357, 0.276111, 0.825890],
                        [4.485496, 0.220784, 0.810182],
                        [0.959931, 2.369110, 4.058938],
                        [0.959931, 2.369110, 4.058938],
                        [4.485496, 0.220784, 0.810182],
                    ],
                ),
            ),
            np.array(
                [
                    [0.77861956, -0.12501022, 0.44104243, 0.42849224],
                    [0.46256046, -0.13302712, -0.03524667, -0.87584204],
                    [0.46256046, -0.13302712, -0.03524667, -0.87584204],
                    [0.46256046, -0.13302712, -0.03524667, -0.87584204],
                    [0.05331986, 0.95051048, 0.28534763, -0.11074093],
                    [0.45489991, -0.13271448, -0.03640618, -0.87984517],
                    [0.8752001, -0.02905178, 0.10626836, 0.47104969],
                    [0.3039118, 0.01972273, -0.92612154, 0.22259272],
                    [0.3039118, 0.01972273, -0.92612154, 0.22259272],
                    [0.8752001, -0.02905178, 0.10626836, 0.47104969],
                ]
            ),
        ),
    ],
    indirect=["angfile_astar"],
)
def test_loadang(angfile_astar, expected_data):
    loaded_data = loadang(angfile_astar)
    assert np.allclose(loaded_data.data, expected_data)


def test_loadctf():
    """ Crude test of the ctf loader """
    z = np.random.rand(100, 8)
    fname = "temp.ctf"
    np.savetxt(fname, z)

    _ = loadctf(fname)
    os.remove(fname)


class TestAngPlugin:
    @pytest.mark.parametrize(
        "angfile_tsl, map_shape, step_sizes, phase_id, n_unknown_columns, example_rot",
        [
            (
                # Read by angfile_tsl() via request.param (passed via `indirect` below)
                (
                    (5, 3),  # map_shape
                    (0.1, 0.1),  # step_sizes
                    np.zeros(5 * 3, dtype=int),  # phase_id
                    5,  # n_unknown_columns
                    np.array(
                        [[1.59942, 2.37748, 4.53419], [1.59331, 2.37417, 4.53628]]
                    ),  # rotations as rows of Euler angle triplets
                ),
                (5, 3),
                (0.1, 0.1),
                np.zeros(5 * 3, dtype=int),
                5,
                np.array(
                    [[1.59942, 2.37748, -1.74690], [1.59331, 2.37417, -1.74899]]
                ),  # rotations as rows of Euler angle triplets
            ),
            (
                (
                    (8, 4),  # map_shape
                    (1.5, 1.5),  # step_sizes
                    np.zeros(8 * 4, dtype=int),  # phase_id
                    5,  # n_unknown_columns
                    np.array(
                        [[5.81107, 2.34188, 4.47345], [6.16205, 0.79936, 1.31702]]
                    ),  # rotations as rows of Euler angle triplets
                ),
                (8, 4),
                (1.5, 1.5),
                np.zeros(8 * 4, dtype=int),
                5,
                np.array(
                    [[-0.12113, 2.34188, 1.31702], [-0.47211, 0.79936, -1.80973]]
                ),  # rotations as rows of Euler angle triplets
            ),
        ],
        indirect=["angfile_tsl"],
    )
    def test_load_ang_tsl(
        self,
        angfile_tsl,
        map_shape,
        step_sizes,
        phase_id,
        n_unknown_columns,
        example_rot,
    ):
        cm = load(angfile_tsl)

        # Fraction of non-indexed points
        non_indexed_fraction = int(np.prod(map_shape) * 0.1)
        assert non_indexed_fraction == np.sum(~cm.is_indexed)

        # Properties
        assert list(cm.prop.keys()) == [
            "iq",
            "ci",
            "unknown1",
            "fit",
            "unknown2",
            "unknown3",
            "unknown4",
            "unknown5",
        ]

        # Coordinates
        ny, nx = map_shape
        dy, dx = step_sizes
        assert np.allclose(cm.x, np.tile(np.arange(nx) * dx, ny))
        assert np.allclose(cm.y, np.sort(np.tile(np.arange(ny) * dy, nx)))

        # Map shape and size
        assert cm.shape == map_shape
        assert cm.size == np.prod(map_shape)

        # Attributes are within expected ranges or have a certain value
        assert cm.prop["ci"].max() <= 1
        assert cm["indexed"].fit.max() <= 3
        assert all(cm["not_indexed"].fit == 180)
        assert all(cm["not_indexed"].ci == -1)

        # Phase IDs (accounting for non-indexed points)
        phase_id[cm["not_indexed"].id] = -1
        assert np.allclose(cm.phase_id, phase_id)

        # Rotations
        rot_unique = np.unique(cm["indexed"].rotations.to_euler(), axis=0)
        assert np.allclose(
            np.sort(rot_unique, axis=0), np.sort(example_rot, axis=0), atol=1e-5
        )
        assert np.allclose(
            cm["not_indexed"].rotations.to_euler()[0],
            np.array([np.pi, 0, np.pi]),
            atol=1e-5,
        )

        # Phases
        assert cm.phases.size == 2  # Including non-indexed
        assert cm.phases.ids == [-1, 0]
        phase = cm.phases[0]
        assert phase.name == "Aluminum"
        assert phase.point_group.name == "432"

    @pytest.mark.parametrize(
        "angfile_astar, map_shape, step_sizes, phase_id, example_rot",
        [
            (
                # Read by angfile_astar() via request.param (passed via `indirect`
                # below)
                (
                    (9, 3),  # map_shape
                    (4.5, 4.5),  # step_sizes
                    np.ones(9 * 3, dtype=int),  # phase_id
                    np.array(
                        [
                            [1.895079, 0.739496, 1.413542],
                            [1.897871, 0.742638, 1.413717],
                        ]
                    ),
                ),
                (9, 3),
                (4.5, 4.5),
                np.ones(9 * 3, dtype=int),
                np.array(
                    [[1.895079, 0.739496, 1.413542], [1.897871, 0.742638, 1.413717]]
                ),
            ),
            (
                (
                    (11, 13),  # map_shape
                    (10, 10),  # step_sizes
                    np.ones(11 * 13, dtype=int),  # phase_id
                    np.array(
                        [
                            [1.621760, 2.368935, 4.559324],
                            [1.604481, 2.367539, 4.541870],
                        ]
                    ),
                ),
                (11, 13),
                (10, 10),
                np.ones(11 * 13, dtype=int),
                np.array(
                    [[1.621760, 2.368935, -1.723861], [1.604481, 2.367539, -1.741315]]
                ),
            ),
        ],
        indirect=["angfile_astar"],
    )
    def test_load_ang_astar(
        self, angfile_astar, map_shape, step_sizes, phase_id, example_rot,
    ):
        cm = load(angfile_astar)

        # Properties
        assert list(cm.prop.keys()) == ["ind", "rel", "relx100"]

        # Coordinates
        ny, nx = map_shape
        dy, dx = step_sizes
        assert np.allclose(cm.x, np.tile(np.arange(nx) * dx, ny))
        assert np.allclose(cm.y, np.sort(np.tile(np.arange(ny) * dy, nx)))

        # Map shape and size
        assert cm.shape == map_shape
        assert cm.size == np.prod(map_shape)

        # Attributes are within expected ranges or have a certain value
        assert cm.prop["ind"].max() <= 100
        assert cm.prop["rel"].max() <= 1
        assert cm.prop["relx100"].max() <= 100
        relx100 = (cm.prop["rel"] * 100).astype(int)
        assert np.allclose(cm.prop["relx100"], relx100)

        # Phase IDs
        assert np.allclose(cm.phase_id, phase_id)

        # Rotations
        rot_unique = np.unique(cm.rotations.to_euler(), axis=0)
        assert np.allclose(
            np.sort(rot_unique, axis=0), np.sort(example_rot, axis=0), atol=1e-6
        )

        # Phases
        assert cm.phases.size == 1
        assert cm.phases.ids == [1]
        phase = cm.phases[1]
        assert phase.name == "Nickel"
        assert phase.point_group.name == "432"

    @pytest.mark.parametrize(
        "angfile_emsoft, map_shape, step_sizes, phase_id, example_rot",
        [
            (
                # Read by angfile_emsoft() via request.param (passed via `indirect`
                # below)
                (
                    (10, 11),  # map_shape
                    (4.5, 4.5),  # step_sizes
                    np.concatenate(
                        (
                            np.ones(int(np.ceil((10 * 11) / 2))),
                            np.ones(int(np.floor((10 * 11) / 2))) * 2,
                        )
                    ),  # phase_id
                    np.array(
                        [
                            [1.895079, 0.739496, 1.413542],
                            [1.897871, 0.742638, 1.413717],
                        ]
                    ),
                ),
                (10, 11),
                (4.5, 4.5),
                np.concatenate(
                    (
                        np.ones(int(np.ceil((10 * 11) / 2))),
                        np.ones(int(np.floor((10 * 11) / 2))) * 2,
                    )
                ),
                np.array(
                    [[1.895079, 0.739496, 1.413542], [1.897871, 0.742638, 1.413717]]
                ),
            ),
            (
                (
                    (3, 6),  # map_shape
                    (10, 10),  # step_sizes
                    np.concatenate(
                        (
                            np.ones(int(np.ceil((3 * 6) / 2))),
                            np.ones(int(np.floor((3 * 6) / 2))) * 2,
                        )
                    ),  # phase_id
                    np.array(
                        [[1.62176, 2.36894, -1.72386], [1.60448, 2.36754, -1.72386]]
                    ),
                ),
                (3, 6),
                (10, 10),
                np.concatenate(
                    (
                        np.ones(int(np.ceil((3 * 6) / 2))),
                        np.ones(int(np.floor((3 * 6) / 2))) * 2,
                    )
                ),
                np.array([[1.62176, 2.36894, -1.72386], [1.60448, 2.36754, -1.72386]]),
            ),
        ],
        indirect=["angfile_emsoft"],
    )
    def test_load_ang_emsoft(
        self, angfile_emsoft, map_shape, step_sizes, phase_id, example_rot,
    ):
        cm = load(angfile_emsoft)

        # Properties
        assert list(cm.prop.keys()) == ["iq", "dp"]

        # Coordinates
        ny, nx = map_shape
        dy, dx = step_sizes
        assert np.allclose(cm.x, np.tile(np.arange(nx) * dx, ny))
        assert np.allclose(cm.y, np.sort(np.tile(np.arange(ny) * dy, nx)))

        # Map shape and size
        assert cm.shape == map_shape
        assert cm.size == np.prod(map_shape)

        # Attributes are within expected ranges or have a certain value
        assert cm.prop["iq"].max() <= 100
        assert cm.prop["dp"].max() <= 1

        # Phase IDs
        assert np.allclose(cm.phase_id, phase_id)

        # Rotations
        rot_unique = np.unique(cm.rotations.to_euler(), axis=0)
        assert np.allclose(
            np.sort(rot_unique, axis=0), np.sort(example_rot, axis=0), atol=1e-5
        )

        # Phases (change if file header is changed!)
        phases_in_data = cm["indexed"].phases_in_data
        assert phases_in_data.size == 2
        assert phases_in_data.ids == [1, 2]
        assert phases_in_data.names == ["austenite", "ferrite"]
        assert [i.name for i in phases_in_data.point_groups] == ["432"] * 2

    def test_get_header(self, temp_ang_file):
        temp_ang_file.write(ANGFILE_ASTAR_HEADER)
        temp_ang_file.close()
        assert _get_header(open(temp_ang_file.name)) == [
            "# File created from ACOM RES results",
            "# ni-dislocations.res",
            "#     ".rstrip(),
            "#     ".rstrip(),
            "# MaterialName      Nickel",
            "# Formula",
            "# Symmetry          43",
            "# LatticeConstants  3.520  3.520  3.520  90.000  90.000  90.000",
            "# NumberFamilies    4",
            "# hklFamilies       1  1  1 1 0.000000",
            "# hklFamilies       2  0  0 1 0.000000",
            "# hklFamilies       2  2  0 1 0.000000",
            "# hklFamilies       3  1  1 1 0.000000",
            "#",
            "# GRID: SqrGrid#",
        ]

    @pytest.mark.parametrize(
        "expected_vendor, expected_columns, vendor_header",
        [
            (
                "tsl",
                [
                    "iq",
                    "ci",
                    "phase_id",
                    "unknown1",
                    "fit",
                    "unknown2",
                    "unknown3",
                    "unknown4",
                    "unknown5",
                ],
                ANGFILE_TSL_HEADER,
            ),
            ("astar", ["ind", "rel", "phase_id", "relx100"], ANGFILE_ASTAR_HEADER),
            ("emsoft", ["iq", "dp", "phase_id"], ANGFILE_EMSOFT_HEADER),
        ],
    )
    def test_get_vendor_columns(
        self, expected_vendor, expected_columns, vendor_header, temp_ang_file
    ):
        expected_columns = ["euler1", "euler2", "euler3", "x", "y"] + expected_columns
        n_cols_file = len(expected_columns)

        temp_ang_file.write(vendor_header)
        temp_ang_file.close()
        header = _get_header(open(temp_ang_file.name))
        vendor, column_names = _get_vendor_columns(header, n_cols_file)

        assert vendor == expected_vendor
        assert column_names == expected_columns

    @pytest.mark.parametrize("n_cols_file", [15, 20])
    def test_get_vendor_columns_unknown(self, temp_ang_file, n_cols_file):
        temp_ang_file.write("Look at me!\nI'm Mr. .ang file!\n")
        temp_ang_file.close()
        header = _get_header(open(temp_ang_file.name))
        with pytest.warns(UserWarning, match=f"Number of columns, {n_cols_file}, "):
            vendor, column_names = _get_vendor_columns(header, n_cols_file)
            assert vendor == "unknown"
            expected_columns = [
                "euler1",
                "euler2",
                "euler3",
                "x",
                "y",
                "unknown1",
                "unknown2",
                "phase_id",
            ] + ["unknown" + str(i + 3) for i in range(n_cols_file - 8)]
            assert column_names == expected_columns

    @pytest.mark.parametrize(
        "header_phase_part, expected_names, expected_point_groups, "
        "expected_lattice_constants",
        [
            (
                [
                    [
                        "# MaterialName      Nickel",
                        "# Formula",
                        "# Symmetry          43",
                        "# LatticeConstants  3.520  3.520  3.520  90.000  90.000  "
                        "90.000",
                    ],
                    [
                        "# MaterialName      Aluminium",
                        "# Formula  Al",
                        "# Symmetry          m3m",
                        "# LatticeConstants  3.520  3.520  3.520  90.000  90.000  "
                        "90.000",
                    ],
                ],
                ["Nickel", "Aluminium"],
                ["43", "m3m"],
                [[3.52, 3.52, 3.52, 90, 90, 90], [3.52, 3.52, 3.52, 90, 90, 90]],
            ),
        ],
    )
    def test_get_phases_from_header(
        self,
        header_phase_part,
        expected_names,
        expected_point_groups,
        expected_lattice_constants,
    ):
        # Create header from parts
        header = [
            "# File created from ACOM RES results",
            "# ni-dislocations.res",
            "#     ",
            "#     ",
        ]
        hkl_families = [
            "# NumberFamilies    4",
            "# hklFamilies       1  1  1 1 0.000000",
            "# hklFamilies       2  0  0 1 0.000000",
            "# hklFamilies       2  2  0 1 0.000000",
            "# hklFamilies       3  1  1 1 0.000000",
        ]
        for phase in header_phase_part:
            header += phase + hkl_families
        header += [
            "#",
            "# GRID: SqrGrid#",
        ]
        names, point_groups, lattice_constants = _get_phases_from_header(header)

        assert names == expected_names
        assert point_groups == expected_point_groups
        assert np.allclose(lattice_constants, expected_lattice_constants)


class TestEMsoftPlugin:
    @pytest.mark.parametrize(
        (
            "temp_emsoft_h5ebsd_file, map_shape, step_sizes, example_rot, "
            "n_top_matches, refined"
        ),
        [
            (
                (
                    (7, 3),  # map_shape
                    (1.5, 1.5),  # step_sizes
                    np.array(
                        [
                            [6.148271, 0.792205, 1.324879],
                            [6.155951, 0.793078, 1.325229],
                        ]
                    ),  # rotations as rows of Euler angle triplets
                    50,  # n_top_matches
                    True,  # refined
                ),
                (7, 3),
                (1.5, 1.5),
                np.array(
                    [[6.148271, 0.792205, 1.324879], [6.155951, 0.793078, 1.325229],]
                ),
                50,
                True,
            ),
            (
                (
                    (5, 17),
                    (0.5, 0.5),
                    np.array(
                        [
                            [6.148271, 0.792205, 1.324879],
                            [6.155951, 0.793078, 1.325229],
                        ]
                    ),
                    20,
                    False,
                ),
                (5, 17),
                (0.5, 0.5),
                np.array(
                    [[6.148271, 0.792205, 1.324879], [6.155951, 0.793078, 1.325229],]
                ),
                20,
                False,
            ),
        ],
        indirect=["temp_emsoft_h5ebsd_file"],
    )
    def test_load_emsoft(
        self,
        temp_emsoft_h5ebsd_file,
        map_shape,
        step_sizes,
        example_rot,
        n_top_matches,
        refined,
    ):
        cm = load(temp_emsoft_h5ebsd_file.filename, refined=refined)

        assert cm.shape == map_shape
        assert (cm.dy, cm.dx) == step_sizes
        if refined:
            n_top_matches = 1
        assert cm.rotations_per_point == n_top_matches

        # Properties
        expected_props = [
            "AvDotProductMap",
            "CI",
            "CIMap",
            "IQ",
            "IQMap",
            "ISM",
            "ISMap",
            "KAM",
            "OSM",
            "TopDotProductList",
            "TopMatchIndices",
        ]
        if refined:
            expected_props += ["RefinedDotProducts"]
        actual_props = list(cm.prop.keys())
        actual_props.sort()
        expected_props.sort()
        assert actual_props == expected_props

        assert cm.phases["austenite"].structure == Structure(
            title="austenite",
            lattice=Lattice(a=3.595, b=3.595, c=3.595, alpha=90, beta=90, gamma=90),
        )


class TestOrixHDF5Plugin:
    def test_file_writer(self, crystal_map, temp_file_path):
        save(filename=temp_file_path, object2write=crystal_map)

        with File(temp_file_path, mode="r") as f:
            assert f["manufacturer"][()][0].decode() == "orix"
            assert f["version"][()][0].decode() == orix_version

    @pytest.mark.parametrize(
        "crystal_map_input",
        [
            ((4, 4, 3), (1, 1.5, 1.5), 1, [0, 1]),
            ((2, 4, 3), (1, 1.5, 1.5), 2, [0, 1, 2]),
        ],
        indirect=["crystal_map_input"],
    )
    def test_write_read_masked(self, crystal_map_input, temp_file_path):
        cm = CrystalMap(**crystal_map_input)
        save(filename=temp_file_path, object2write=cm[cm.x > 2])
        cm2 = load(temp_file_path)

        assert cm2.size != cm.size
        with pytest.raises(ValueError, match="operands could not be broadcast"):
            _ = np.allclose(cm2.x, cm.x)

        cm2.is_in_data = cm.is_in_data
        assert cm2.size == cm.size
        assert np.allclose(cm2.x, cm.x)

    def test_file_writer_raises(self, temp_file_path, crystal_map):
        with pytest.raises(OSError, match="Cannot write to the already open file "):
            with File(temp_file_path, mode="w") as _:
                save(temp_file_path, crystal_map, overwrite=True)

    def test_dict2hdf5group(self, temp_file_path):
        with File(temp_file_path, mode="w") as f:
            group = f.create_group(name="a_group")
            with pytest.warns(UserWarning, match="The orix HDF5 writer could not"):
                dict2hdf5group(
                    dictionary={"a": [np.array(24.5)], "c": set()}, group=group
                )

    def test_crystalmap2dict(self, temp_file_path, crystal_map_input):
        cm = CrystalMap(**crystal_map_input)
        cm_dict = crystalmap2dict(cm)

        this_dict = {"hello": "there"}
        cm_dict2 = crystalmap2dict(cm, dictionary=this_dict)

        cm_dict2.pop("hello")
        assert_dictionaries_are_equal(cm_dict, cm_dict2)

        assert np.allclose(cm_dict["data"]["x"], crystal_map_input["x"])
        assert cm_dict["header"]["z_step"] == cm.dz

    def test_phaselist2dict(self, phase_list):
        pl_dict = phaselist2dict(phase_list)
        this_dict = {"hello": "there"}
        this_dict = phaselist2dict(phase_list, dictionary=this_dict)
        this_dict.pop("hello")

        assert_dictionaries_are_equal(pl_dict, this_dict)

    def test_phase2dict(self, phase_list):
        phase_dict = phase2dict(phase_list[0])
        this_dict = {"hello": "there"}
        this_dict = phase2dict(phase_list[0], dictionary=this_dict)
        this_dict.pop("hello")

        assert_dictionaries_are_equal(phase_dict, this_dict)

    def test_phase2dict_spacegroup(self):
        """Space group is written to dict as an int or "None"."""
        sg100 = 100
        phase = Phase(space_group=sg100)
        phase_dict1 = phase2dict(phase)
        assert phase_dict1["space_group"] == sg100

        sg200 = GetSpaceGroup(200)
        phase.space_group = sg200
        phase_dict2 = phase2dict(phase)
        assert phase_dict2["space_group"] == sg200.number

        phase.space_group = None
        phase_dict3 = phase2dict(phase)
        assert phase_dict3["space_group"] == "None"

    def test_structure2dict(self, phase_list):
        structure = phase_list[0].structure
        structure_dict = structure2dict(structure)
        this_dict = {"hello": "there"}
        this_dict = structure2dict(structure, this_dict)
        this_dict.pop("hello")

        lattice1 = structure_dict["lattice"]
        lattice2 = this_dict["lattice"]
        assert np.allclose(lattice1["abcABG"], lattice2["abcABG"])
        assert np.allclose(lattice1["baserot"], lattice2["baserot"])
        assert_dictionaries_are_equal(structure_dict["atoms"], this_dict["atoms"])

    def test_hdf5group2dict_update_dict(self, temp_file_path, crystal_map):
        save(temp_file_path, crystal_map)
        with File(temp_file_path, mode="r") as f:
            this_dict = {"hello": "there"}
            this_dict = hdf5group2dict(f["crystal_map"], dictionary=this_dict)

            assert this_dict["hello"] == "there"
            assert this_dict["data"] == f["crystal_map/data"]
            assert this_dict["header"] == f["crystal_map/header"]

    def test_file_reader(self, crystal_map, temp_file_path):
        save(filename=temp_file_path, object2write=crystal_map)
        cm2 = load(filename=temp_file_path)
        assert_dictionaries_are_equal(crystal_map.__dict__, cm2.__dict__)

    def test_dict2crystalmap(self, crystal_map):
        cm2 = dict2crystalmap(crystalmap2dict(crystal_map))
        assert_dictionaries_are_equal(crystal_map.__dict__, cm2.__dict__)

    def test_dict2phaselist(self, phase_list):
        phase_list2 = dict2phaselist(phaselist2dict(phase_list))

        assert phase_list.size == phase_list2.size
        assert phase_list.ids == phase_list2.ids
        assert phase_list.names == phase_list2.names
        assert phase_list.colors == phase_list2.colors
        assert [
            s1.name == s2.name
            for s1, s2 in zip(phase_list.point_groups, phase_list2.point_groups)
        ]

    def test_dict2phase(self, phase_list):
        phase1 = phase_list[0]
        phase2 = dict2phase(phase2dict(phase1))

        assert phase1.name == phase2.name
        assert phase1.color == phase2.color
        assert phase1.space_group.number == phase2.space_group.number
        assert phase1.point_group.name == phase2.point_group.name
        assert phase1.structure.lattice.abcABG() == phase2.structure.lattice.abcABG()

    def test_dict2phase_spacegroup(self):
        """Space group number int or None is properly parsed from a dict.
        """
        phase1 = Phase(space_group=200)
        phase_dict = phase2dict(phase1)
        phase2 = dict2phase(phase_dict)
        assert phase1.space_group.number == phase2.space_group.number

        phase_dict.pop("space_group")
        phase3 = dict2phase(phase_dict)
        assert phase3.space_group is None

    def test_dict2structure(self, phase_list):
        structure1 = phase_list[0].structure
        structure2 = dict2structure(structure2dict(structure1))

        lattice1 = structure1.lattice
        lattice2 = structure2.lattice
        assert lattice1.abcABG() == lattice2.abcABG()
        assert np.allclose(lattice1.baserot, lattice2.baserot)

        assert str(structure1.element) == str(structure2.element)
        assert np.allclose(structure1.xyz, structure2.xyz)

    def test_dict2lattice(self, phase_list):
        lattice = phase_list[0].structure.lattice
        lattice2 = dict2lattice(lattice2dict(lattice))

        assert lattice.abcABG() == lattice2.abcABG()
        assert np.allclose(lattice.baserot, lattice2.baserot)

    def test_dict2atom(self, phase_list):
        atom = phase_list[0].structure[0]
        atom2 = dict2atom(atom2dict(atom))

        assert str(atom.element) == str(atom2.element)
        assert np.allclose(atom.xyz, atom2.xyz)

    def test_read_point_group_from_v0_3_x(self, temp_file_path, crystal_map):
        crystal_map.phases[0].point_group = "1"
        save(filename=temp_file_path, object2write=crystal_map)

        # First, ensure point group data set name is named "symmetry", as in v0.3.0
        with File(temp_file_path, mode="r+") as f:
            for phase in f["crystal_map/header/phases"].values():
                phase["symmetry"] = phase["point_group"]
                del phase["point_group"]

        # Then, make sure it can still be read
        cm2 = load(filename=temp_file_path)
        # And that the symmetry operations are the same, for good measure
        print(crystal_map)
        print(cm2)
        assert np.allclose(
            crystal_map.phases[0].point_group.data, cm2.phases[0].point_group.data
        )
