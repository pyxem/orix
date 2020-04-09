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

import pytest


@pytest.fixture(
    params=[
        """4.485496 0.952426 0.791507     0.000     0.000   22.2  0.060  1       6
1.343904 0.276111 0.825890    19.000     0.000   16.3  0.020  1       2""",
    ]
)
def angfile(tmpdir, request):
    f = tmpdir.mkdir("angfiles").join("angfile.ang")
    f.write(
        """# File created from ACOM RES results
# ni-dislocations.res
#
#
# MaterialName      Nickel
# Formula
# Symmetry          43
# LatticeConstants  3.520  3.520  3.520  90.000  90.000  90.000
# NumberFamilies    4
# hklFamilies       1  1  1 1 0.000000
# hklFamilies       2  0  0 1 0.000000
# hklFamilies       2  2  0 1 0.000000
# hklFamilies       3  1  1 1 0.000000
#
# GRID: SqrGrid#"""
    )
    f.write(request.param)
    return str(f)
