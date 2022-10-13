# -*- coding: utf-8 -*-
# Copyright 2018-2022 the orix developers
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

import dask.array as da
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as SciPyRotation

from orix.base import DimensionError
from orix.quaternion import Orientation, Quaternion
from orix.vector import Vector3d

# fmt: off
values = [
    (0.707,  0.0,  0.0, 0.707),
    (0.5  , -0.5, -0.5,   0.5),
    (0.0  ,  0.0,  0.0,   1.0),
    (1.0  ,  1.0,  1.0,   1.0),
    (
        (0.5, -0.5, -0.5, 0.5),
        (0.0,  0.0,  0.0, 1.0),
    ),
    Quaternion(
        [
            [(0.0, 0.0, 0.0, 1.0), (0.707, 0.0, 0.0, 0.707)],
            [(1.0, 1.0, 1.0, 1.0), (0.707, 0.0, 0.0, 0.707)],
        ]
    ),
    np.array((4, 3, 2, 1)),
]
singles = [
    (0.881, 0.665, 0.123, 0.517),
    (0.111, 0.222, 0.333, 0.444),
    (
        (1, 0,  0.5,  0),
        (3, 1, -1  , -2)
    ),
    [
        [[0.343  ,  0.343,  0    , -0.333], [-7, -8, -9, -10]],
        [[0.00001, -0.0001, 0.001, -0.01 ], [ 0,  0,  0,   0]],
    ],
]
# fmt: on


@pytest.fixture(params=values)
def quaternion(request):
    return Quaternion(request.param)


@pytest.fixture
def identity():
    return Quaternion((1, 0, 0, 0))


@pytest.fixture(params=singles)
def something(request):
    return Quaternion(request.param)


@pytest.mark.parametrize("input_length", [1, 2, 3, 5, 6, 8])
def test_init(input_length):
    with pytest.raises(DimensionError):
        Quaternion(tuple(range(input_length)))


class TestQuaternion:
    def test_neg(self, quaternion):
        assert np.allclose((-quaternion).data, -(quaternion.data))

    def test_norm(self, quaternion):
        q = quaternion
        assert np.allclose(q.norm, (q.data**2).sum(axis=-1) ** 0.5)

    def test_unit(self, quaternion):
        assert np.allclose(quaternion.unit.norm, 1)

    def test_conj(self, quaternion):
        q = quaternion
        assert np.allclose(q.data[..., 0], q.conj.data[..., 0])
        assert np.allclose(q.data[..., 1:], -q.conj.data[..., 1:])

    def test_mul(self, quaternion, something):
        q = quaternion
        s = something
        sa, sb, sc, sd = s.a, s.b, s.c, s.d
        qa, qb, qc, qd = q.a, q.b, q.c, q.d

        q1 = q * s
        assert isinstance(q1, Quaternion)
        assert np.allclose(q1.a, sa * qa - sb * qb - sc * qc - sd * qd)
        assert np.allclose(q1.b, qa * sb + qb * sa + qc * sd - qd * sc)
        assert np.allclose(q1.c, qa * sc - qb * sd + qc * sa + qd * sb)
        assert np.allclose(q1.d, qa * sd + qb * sc - qc * sb + qd * sa)

    def test_mul_identity(self, quaternion, identity):
        assert np.allclose((quaternion * identity).data, quaternion.data)

    def test_no_multiplicative_inverse(self, quaternion, something):
        q1 = quaternion * something
        q2 = something * quaternion
        assert np.all(q1 != q2)

    def test_inverse(self, quaternion):
        q = quaternion
        assert np.allclose((q * ~q).data, (~q * q).data)
        assert np.allclose((q * ~q).a, 1)
        assert np.allclose((q * ~q).data[..., 1:], 0)

    def test_dot(self, quaternion, something):
        q = quaternion
        assert np.allclose(q.dot(q), np.sum(q.data**2, axis=-1))
        assert np.allclose(q.dot(something), something.dot(q))

    def test_dot_outer(self, quaternion, something):
        q = quaternion
        s = something

        d = q.dot_outer(s)
        assert d.shape == q.shape + s.shape
        for i in np.ndindex(q.shape):
            for j in np.ndindex(s.shape):
                assert np.allclose(d[i + j], q[i].dot(s[j]))

    @pytest.mark.parametrize(
        "quaternion, vector, expected",
        [
            # fmt: off
            ((0.5           , 0.5, 0.5,            0.5), (1, 0, 0), ( 0,  1, 0)),
            ((np.sqrt(2) / 2, 0  , 0  , np.sqrt(2) / 2), (0, 1, 0), (-1,  0, 0)),
            ((0             , 1  , 0  ,              0), (0, 1, 0), ( 0, -1, 0)),
            (
                (0    , np.sqrt(3) / 3,  np.sqrt(3) / 3, -np.sqrt(3) / 3),
                (1    ,                  1             ,  0             ),
                (1 / 3, 1 / 3         , -4 / 3                          ),
            )
            # fmt: on
        ],
    )
    def test_multiply_vector(self, quaternion, vector, expected):
        q = Quaternion(quaternion)
        v = Vector3d(vector)
        v_new = q * v
        assert np.allclose(v_new.data, expected)

    def test_abcd_properties(self):
        quat = Quaternion([2, 2, 2, 2])
        quat.a = 1
        quat.b = 1
        quat.c = 1
        quat.d = 1
        assert np.allclose(quat.data, 1)

    def test_mean(self, quaternion):
        qm = quaternion.mean()
        assert isinstance(qm, Quaternion)
        assert qm.size == 1

    def test_antipodal(self, quaternion):
        q = quaternion
        qa = q.antipodal
        assert qa.size == 2 * q.size

    def test_edgecase_outer(self, quaternion):
        with pytest.raises(NotImplementedError, match="This operation is currently "):
            _ = quaternion.outer([3, 2])

    def test_failing_mul(self, quaternion):
        """Not implemented."""
        with pytest.raises(TypeError):
            _ = quaternion * "cant-multiply-by-this"

    @pytest.mark.parametrize("shape", [(2, 3), (4, 5, 6), (1, 5), (11,)])
    def test_outer(self, shape):
        rng = np.random.default_rng()
        new_shape = shape + (4,)
        abcd = rng.normal(size=np.prod(new_shape)).reshape(shape + (4,))
        q = Quaternion(abcd)

        qo_numpy = q.outer(q)
        assert isinstance(qo_numpy, Quaternion)
        assert qo_numpy.shape == 2 * shape

        # Returns dask array, not Quaternion
        qo_dask = q._outer_dask(q)
        assert isinstance(qo_dask, da.Array)
        qo_numpy2 = Quaternion(qo_dask.compute())
        assert qo_numpy2.shape == 2 * shape
        assert np.allclose(qo_numpy.data, qo_numpy2.data)

        # public function .outer() with Dask
        qo_dask2 = q.outer(q, lazy=True)
        assert isinstance(qo_dask2, Quaternion)
        assert qo_dask2.shape == 2 * shape
        assert np.allclose(qo_numpy.data, qo_dask2.data)

    def test_outer_lazy_chunk_size(self):
        shape = (5, 15, 4)
        rng = np.random.default_rng()
        abcd = rng.normal(size=np.prod(shape)).reshape(shape)
        q = Quaternion(abcd)

        cs1 = 10
        cs2 = 13
        outer1 = q._outer_dask(q, chunk_size=cs1)
        outer2 = q._outer_dask(q, chunk_size=cs2)

        assert outer1.chunks == ((5,), (cs1, 5), (5,), (cs1, 5), (4,))
        assert outer2.chunks == ((5,), (cs2, 2), (5,), (cs2, 2), (4,))

        assert np.allclose(outer1.compute(), outer2.compute())

    @pytest.mark.parametrize("shape", [(2, 3), (4, 5, 6), (1, 5), (11,)])
    def test_outer_vector_lazy(self, shape):
        rng = np.random.default_rng()
        new_shape = shape + (4,)
        abcd = rng.normal(size=np.prod(new_shape)).reshape(shape + (4,))
        q = Quaternion(abcd).unit

        v = Vector3d(np.random.rand(7, 4, 3)).unit

        qvo_numpy = q.outer(v)
        assert isinstance(qvo_numpy, Vector3d)
        assert qvo_numpy.shape == q.shape + v.shape

        # Returns dask array, not Vector3d
        qvo_dask = q._outer_dask(v)
        assert isinstance(qvo_dask, da.Array)
        qvo_numpy2 = Vector3d(qvo_dask.compute())
        assert qvo_numpy2.shape == qvo_numpy.shape
        assert np.allclose(qvo_numpy.data, qvo_numpy2.data)

        # public function .outer() with Dask
        qvo_dask2 = q.outer(v, lazy=True)
        assert isinstance(qvo_dask2, Vector3d)
        assert qvo_dask2.shape == qvo_numpy.shape
        assert np.allclose(qvo_numpy.data, qvo_dask2.data)

    def test_outer_lazy_progressbar_stdout(self, capsys):
        rng = np.random.default_rng()
        shape = (5, 3)
        new_shape = shape + (4,)
        abcd = rng.normal(size=np.prod(new_shape)).reshape(shape + (4,))
        q = Quaternion(abcd).unit
        # other is Quaternion
        _ = q.outer(q, lazy=True, progressbar=True)
        out, _ = capsys.readouterr()
        assert "Completed" in out
        _ = q.outer(q, lazy=True, progressbar=False)
        out, _ = capsys.readouterr()
        assert not out
        # test other is Vector3d
        v = Vector3d(np.random.rand(2, 3, 3)).unit
        _ = q.outer(v, lazy=True, progressbar=True)
        out, _ = capsys.readouterr()
        assert "Completed" in out
        _ = q.outer(v, lazy=True, progressbar=False)
        out, _ = capsys.readouterr()
        assert not out

    def test_outer_dask_wrong_type_raises(self):
        shape = (5,)
        rng = np.random.default_rng()
        new_shape = shape + (4,)
        abcd = rng.normal(size=np.prod(new_shape)).reshape(shape + (4,))
        q = Quaternion(abcd)
        # not Quaternion or Vector3d
        other = np.random.rand(7, 3)
        with pytest.raises(TypeError, match="Other must be Quaternion or Vector3d"):
            q._outer_dask(other)

    def test_from_align_vectors(self):
        a = Vector3d([[2, -1, 0], [0, 0, 1]])
        b = Vector3d([[3, 1, 0], [-1, 3, 0]])
        ori = Quaternion.from_align_vectors(a, b)
        assert type(ori) == Quaternion
        assert np.allclose(
            ori.data, np.array([[0.65328148, 0.70532785, -0.05012611, -0.27059805]])
        )
        assert np.allclose(a.unit.data, (ori * b.unit).data)
        _, e = Quaternion.from_align_vectors(a, b, return_rmsd=True)
        assert type(e) == np.float64
        _, m = Quaternion.from_align_vectors(a, b, return_sensitivity=True)
        assert type(m) == np.ndarray
        assert m.shape == (3, 3)
        out = Quaternion.from_align_vectors(
            a, b, return_rmsd=True, return_sensitivity=True
        )
        assert len(out) == 3

    def test_from_scipy_rotation(self):
        euler = np.array([15, 32, 41]) * np.pi / 180
        reference_rot = Orientation.from_euler(euler)
        scipy_rot = SciPyRotation.from_euler("zxz", euler)  # bunge convention
        quat = Quaternion.from_scipy_rotation(scipy_rot)
        assert np.allclose(
            reference_rot.data, quat.data
        )  # This fails, but I feel it should be correct :/
