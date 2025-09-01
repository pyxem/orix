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

from collections import OrderedDict
from contextlib import contextmanager
from io import StringIO
from numbers import Number
import os
import sys

from diffpy.structure import Structure
from h5py import File
import numpy as np
import pytest

from orix.constants import VisibleDeprecationWarning
from orix.crystal_map import Phase, PhaseList
from orix.io._io import (
    _overwrite_or_not,
    _plugin_from_manufacturer,
    load,
    loadctf,
    save,
)
from orix.io.plugins import bruker_h5ebsd, emsoft_h5ebsd, orix_hdf5
from orix.quaternion import Rotation


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

    def test_load_unsupported_format(self, tmp_path):
        fname = tmp_path / "unsupported_file.ktf"
        np.savetxt(fname, X=np.random.rand(100, 8))
        with pytest.raises(IOError, match="Could not read "):
            _ = load(fname)

    @pytest.mark.parametrize(
        "manufacturer, expected_plugin",
        [
            ("EMEBSDDictionaryIndexing.f90", emsoft_h5ebsd),
            ("Bruker Nano", bruker_h5ebsd),
            ("orix", orix_hdf5),
            ("Oxford", None),
        ],
    )
    def test_plugin_from_manufacturer(self, manufacturer, expected_plugin, tmp_path):
        h5ebsd_plugin_list = [bruker_h5ebsd, emsoft_h5ebsd, orix_hdf5]
        fname = tmp_path / "test.h5"
        with File(fname, mode="w") as f:
            f.create_dataset(name="Manufacturer", data=manufacturer)
            assert (
                _plugin_from_manufacturer(fname, plugins=h5ebsd_plugin_list)
                is expected_plugin
            )

    def test_overwrite_or_not(self, crystal_map, tmp_path):
        fname = tmp_path / "test.h5"
        save(fname, crystal_map)
        with pytest.warns(UserWarning, match="Not overwriting, since your terminal "):
            _overwrite_or_not(fname)

    @pytest.mark.parametrize(
        "answer, expected", [("y", True), ("n", False), ("m", None)]
    )
    def test_overwrite_or_not_input(self, crystal_map, answer, expected, tmp_path):
        fname = tmp_path / "test.h5"
        save(fname, crystal_map)
        if answer == "m":
            with replace_stdin(StringIO(answer)):
                with pytest.raises(EOFError):
                    _overwrite_or_not(fname)
        else:
            with replace_stdin(StringIO(answer)):
                assert _overwrite_or_not(fname) is expected

    @pytest.mark.parametrize("ext", ["angs", "hdf4", "h6"])
    def test_save_unsupported_raises(self, ext, crystal_map, tmp_path):
        fname = tmp_path / f"test.{ext}"
        with pytest.raises(IOError, match=f"'{ext}' does not correspond to any "):
            save(fname, crystal_map)

    def test_save_overwrite_raises(self, crystal_map, tmp_path):
        with pytest.raises(ValueError, match="`overwrite` parameter can only be "):
            save(tmp_path / "test.h5", crystal_map, overwrite=1)

    @pytest.mark.parametrize(
        "overwrite, expected_phase_name", [(True, "hepp"), (False, "")]
    )
    def test_save_overwrite(
        self, crystal_map, overwrite, expected_phase_name, tmp_path
    ):
        fname = tmp_path / "test.h5"

        assert crystal_map.phases[0].name == ""
        save(fname, crystal_map)
        assert os.path.isfile(fname) is True

        crystal_map.phases[0].name = "hepp"
        save(fname, crystal_map, overwrite=overwrite)

        crystal_map2 = load(fname)
        assert crystal_map2.phases[0].name == expected_phase_name


# TODO: Remove after 0.13.0
def test_loadctf():
    z = np.random.rand(100, 8)
    fname = "temp.ctf"
    np.savetxt(fname, z)

    with pytest.warns(VisibleDeprecationWarning):
        _ = loadctf(fname)
    os.remove(fname)
