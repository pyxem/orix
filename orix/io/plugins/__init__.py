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

"""Input/output plugins.

.. currentmodule:: orix.io.plugins

.. rubric:: Modules

.. autosummary::
    :toctree: ../generated/
    :template: custom-module-template.rst

    ang
    bruker_h5ebsd
    ctf
    emsoft_h5ebsd
    orix_hdf5
"""

from types import ModuleType

from orix.io.plugins import ang, bruker_h5ebsd, ctf, emsoft_h5ebsd, orix_hdf5

plugin_list: list[ModuleType] = [
    ang,
    bruker_h5ebsd,
    ctf,
    emsoft_h5ebsd,
    orix_hdf5,
]
