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

import os

import pytest

from orix import data
from orix.crystal_map import CrystalMap
from orix.quaternion import Orientation, symmetry


class TestData:
    def test_load_sdss_ferrite_austenite(self):
        """The file can be read."""
        xmap = data.sdss_ferrite_austenite(allow_download=True)
        assert isinstance(xmap, CrystalMap)
        assert xmap.phases.names == ["austenite", "ferrite"]
        assert xmap.shape == (100, 117)

    def test_load_sdss_austenite_kwargs(self):
        """Keyword arguments are passed to EMsoft HDF5 reader."""
        xmap_di = data.sdss_austenite(allow_download=True)
        xmap_ref = data.sdss_austenite(refined=True)

        assert xmap_di.rotations_per_point == 50
        assert xmap_ref.rotations_per_point == 1

    def test_load_raises(self):
        """Raises desired error message."""
        name = "sdss/sdss_ferrite_austenite.ang"
        file = data._fetcher.path / name

        # Remove file (dangerous!)
        removed = False
        if file.exists():  # pragma: no cover
            os.remove(file)
            removed = True

        with pytest.raises(ValueError, match=f"Dataset {name} must be"):
            _ = data.sdss_ferrite_austenite(allow_download=False)

        # Re-download file if removed
        if removed:  # pragma: no cover
            _ = data.sdss_ferrite_austenite(allow_download=True)

    def test_load_ti_orientations(self):
        """The file can be read."""
        g = data.ti_orientations(allow_download=True)
        assert isinstance(g, Orientation)
        assert g.symmetry == symmetry.D6
