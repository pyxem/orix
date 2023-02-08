# Copyright 2018-2023 The orix developers
#
# This file is part of orix.
#
# kikuchipy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# kikuchipy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with orix. If not, see <http://www.gnu.org/licenses/>.

import re

from outdated import check_outdated
from packaging.version import Version

with open("../../orix/__init__.py") as fid:
    for line in fid:
        if line.startswith("__version__"):
            branch_version_str = line.strip().split(" = ")[-1][1:-1]
            break

# Within a try/except because we don't want to throw the error if a new
# tagged release draft is to be made, we just want to know if the branch
# version is different (hopefully always newer if different) from the
# PyPI version
try:
    make_release, pypi_version_str = check_outdated("orix", branch_version_str)
except ValueError as err:
    pattern = re.compile(r"([\d.]+)")  # Any "word" with decimal digits and dots
    matches = []
    for line in err.args[0].split(" "):
        if pattern.match(line) is not None:
            matches.append(line)
    pypi_version_str = matches[-1]
    make_release = True

# Don't make a release if the version is a development version
branch_version = Version(branch_version_str)
pypi_version = Version(pypi_version_str)
if branch_version.is_devrelease:
    make_release = False

if make_release:
    # Determine which type of release this is (major, minor, patch?)
    if branch_version.major > pypi_version.major:
        release_type = "major"
    elif branch_version.minor > pypi_version.minor:
        release_type = "minor"
    else:
        release_type = "patch"

    # Write the relevant part of the changelog to a new file to be used
    # by the publish workflow
    with open("../../CHANGELOG.rst", mode="r") as f:
        content = f.readlines()
        changelog_start = 0
        changelog_end = 0
        for i, line in enumerate(content):
            if branch_version.base_version in line:
                changelog_start = i + 3
            elif pypi_version.base_version in line:
                changelog_end = i - 1
                break
        if changelog_start == 0:
            changelog_end = 0
    with open("release_part_in_changelog.rst", mode="w") as f:
        f.write(
            f"orix {branch_version_str} is a {release_type} release of orix, an open-source Python library for handling orientations, rotations and crystal symmetry.\n\n"
            f"See below, the `changelog <https://orix.readthedocs.io/en/stable/changelog.html>`_ or the `GitHub changelog <https://github.com/pyxem/orix/compare/v{pypi_version_str}...v{branch_version_str}>`_ for all updates from the previous release.\n\n"
        )
        for line in content[changelog_start:changelog_end]:
            f.write(line)

# These three prints are collected by a bash script using `eval` and
# passed to the publish workflow environment variables
print(make_release)
print(pypi_version_str)
print(branch_version_str)
