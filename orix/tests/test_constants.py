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

from orix import constants

from .conftest import skipif_numpy_quaternion_missing, skipif_numpy_quaternion_present


class TestConstants:
    @skipif_numpy_quaternion_present
    def test_numpy_quaternion_not_installed(self):
        assert not constants.installed["numpy-quaternion"]

    @skipif_numpy_quaternion_missing
    def test_numpy_quaternion_installed(self):
        assert constants.installed["numpy-quaternion"]
