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

"""
Crystallographic map of rotations, crystal phases and key properties associated with
every spatial coordinate in a 1D, 2D or 3D space.

All properties are stored as 1D arrays, and reshaped when necessary.

This module uses logging at the DEBUG level to keep track of manipulation and plotting
of CrystalMap objects where the underlying behaviour might not be straight forward to
analyse. Use the following in a script to see this logging:

.. code-block:: python

    >>> import logging
    >>> logging.basicConfig(level=logging.DEBUG)

And ensure matplotlib (and any other packages) doesn't clutter the debug log:

.. code-block:: python

    >>> logging.getLogger("matplotlib").setLevel(logging.WARNING)

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    crystal_map
    crystal_map_properties
    phase_list
"""

from .crystal_map import CrystalMap
from .phase_list import Phase, PhaseList
from .crystal_map_properties import CrystalMapProperties
