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

from h5py import File

from orix.io import save
from orix.io.plugins._h5ebsd import hdf5group2dict


class TestH5ebsd:
    def test_hdf5group2dict_update_dict(self, temp_file_path, crystal_map):
        """Can read datasets from an HDF5 file into an existing
        dictionary.
        """
        save(temp_file_path, crystal_map)
        with File(temp_file_path, mode="r") as f:
            this_dict = {"hello": "there"}
            this_dict = hdf5group2dict(f["crystal_map"], dictionary=this_dict)
            assert this_dict["hello"] == "there"
            assert this_dict["data"] == f["crystal_map/data"]
            assert this_dict["header"] == f["crystal_map/header"]
