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


class CrystalMapProperties(dict):
    """A class to store properties with in a CrystalMap instance.

    This class is a thin wrapper around :class:`dict`. It overrides setting
    and getting property arrays in the `dict` to handle a data mask
    correctly, i.e. whether data points are considered to be in the data.

    Attributes
    ----------
    id : numpy.ndarray
        1D integer array with the id of each point in the data.
    is_in_data : numpy.ndarray
        1D boolean array with True for points in the data, of the same size
        as the data.

    """

    def __init__(self, dictionary, id, is_in_data=None):
        """Create a `CrystalMapProperties` object.

        Parameters
        ----------
        dictionary : dict
            Dictionary of properties with `key` equal to the property name
            and `value` as the numpy array.
        id : numpy.ndarray
            1D integer array with the id of each point in the entire data,
            i.e. not just points in the data.
        is_in_data : numpy.ndarray, optional
            1D boolean array with True for points in the data. If ``None``
            is passed (default), all points are considered to be in the
            data.

        """
        super().__init__(**dictionary)
        self.id = id
        if is_in_data is None:
            self.is_in_data = np.ones(id.size, dtype=bool)
        else:
            self.is_in_data = is_in_data

    def __setitem__(self, key, value):
        """Add a 1D array to or update an existing array in the
        dictionary. If `key` is the name of an existing array, only the
        points in the data (where `self.is_in_data` is True) are set.

        """
        # Get array values if `key` already present, or zeros
        array = self.setdefault(key, np.zeros(self.is_in_data.size))

        # Determine array data type from input
        if hasattr(value, "__iter__"):
            value_type = type(value[0])
        else:
            value_type = type(value)
        array = array.astype(value_type)

        array[self.is_in_data] = value
        super().__setitem__(key, array)

    def __getitem__(self, item):
        """Return a dictionary entry, ensuring that only points in the data
        are returned.

        """
        array = super().__getitem__(item)
        return array[self.is_in_data]
