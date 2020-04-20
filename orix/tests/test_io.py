import pytest
import numpy as np
import os

from orix import io


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


def test_load_ang(angfile):
    """ This testing is improved in v0.3.0"""
    loaded_data = io.loadang(angfile)


def test_load_ctf():
    """ Crude test of the ctf loader """
    z = np.random.rand(100, 8)
    np.savetxt("temp.ctf", z)
    z_loaded = io.loadctf("temp.ctf")
    os.remove("temp.ctf")
