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

from diffpy.structure import Structure
from h5py import File
import pytest
import numpy as np

from orix.crystal_map import Phase, PhaseList
from orix.io import (
    load,
    save,
    loadctf,
    _plugin_from_footprints,
    _overwrite_or_not,
)
from orix.io.plugins import ang, emsoft_h5ebsd, orix_hdf5
from orix.quaternion.rotation import Rotation

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


def test_loadctf():
    """ Crude test of the ctf loader """
    z = np.random.rand(100, 8)
    fname = "temp.ctf"
    np.savetxt(fname, z)

    _ = loadctf(fname)
    os.remove(fname)
