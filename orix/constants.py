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

"""Constants and such useful across modules."""

from importlib.metadata import version

# NB! Update project config file if this list is updated!
optional_deps: list[str] = ["numpy-quaternion"]
installed: dict[str, bool] = {}
for pkg in optional_deps:
    try:
        _ = version(pkg)
        installed[pkg] = True
    except ImportError:  # pragma: no cover
        installed[pkg] = False

# Typical tolerances for comparisons in need of a precision. We
# generally use the highest precision possible (allowed by testing on
# different OS and Python versions).
eps9 = 1e-9
eps12 = 1e-12

del optional_deps
