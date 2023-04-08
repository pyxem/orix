# -*- coding: utf-8 -*-
# Copyright 2018-2023 the orix developers
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
import warnings

import dask.array as da
from diffpy.structure.spacegroups import sg225
import numpy as np
import pytest

from orix.base import DimensionError
from orix.quaternion import Quaternion
from orix.vector import AxAngle, Vector3d


@pytest.fixture(
    params=[
        # fmt: off
        (0.707,  0.0,  0.0, 0.707),
        (0.5  , -0.5, -0.5,   0.5),
        (0.0  ,  0.0,  0.0,   1.0),
        (1.0  ,  1.0,  1.0,   1.0),
        ((0.5, -0.5, -0.5, 0.5), (0.0,  0.0,  0.0, 1.0)),
        Quaternion(
            [
                [(0.0, 0.0, 0.0, 1.0), (0.707, 0.0, 0.0, 0.707)],
                [(1.0, 1.0, 1.0, 1.0), (0.707, 0.0, 0.0, 0.707)],
            ]
        ),
        np.array((4, 3, 2, 1)),
        # fmt: on
    ]
)
def quaternion(request):
    return Quaternion(request.param)


@pytest.fixture
def identity():
    return Quaternion((1, 0, 0, 0))


@pytest.fixture(
    params=[
        # fmt: off
    (0.881, 0.665, 0.123, 0.517),
    (0.111, 0.222, 0.333, 0.444),
    ((1, 0,  0.5,  0), (3, 1, -1  , -2)),
    [
        [[0.343  ,  0.343,  0    , -0.333], [-7, -8, -9, -10]],
        [[0.00001, -0.0001, 0.001, -0.01 ], [ 0,  0,  0,   0]],
    ],
        # fmt: on
    ]
)
def something(request):
    return Quaternion(request.param)


class TestQuaternion:
    @pytest.mark.parametrize("input_length", [1, 2, 3, 5, 6, 8])
    def test_init(self, input_length):
        with pytest.raises(DimensionError):
            Quaternion(tuple(range(input_length)))

    def test_neg(self, quaternion):
        assert np.allclose((-quaternion).data, (-quaternion.data))

    def test_norm(self, quaternion):
        assert np.allclose(quaternion.norm, (quaternion.data**2).sum(axis=-1) ** 0.5)

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
        v1 = Vector3d([[2, -1, 0], [0, 0, 1]])
        v2 = Vector3d([[3, 1, 0], [-1, 3, 0]])

        q1 = Quaternion.from_align_vectors(v1, v2)
        assert isinstance(q1, Quaternion)
        assert np.allclose(
            q1.data, np.array([[0.65328148, 0.70532785, -0.05012611, -0.27059805]])
        )
        assert np.allclose(v1.unit.data, (q1 * v2.unit).data)

        out = Quaternion.from_align_vectors(v1, v2, return_rmsd=True)
        assert isinstance(out, tuple)
        error = out[1]
        assert error == 0

        _, sens_mat = Quaternion.from_align_vectors(v1, v2, return_sensitivity=True)
        assert np.allclose(sens_mat, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.5]]))

        out = Quaternion.from_align_vectors(
            v1, v2, return_rmsd=True, return_sensitivity=True
        )
        assert len(out) == 3

        q2 = Quaternion.from_align_vectors(
            [[2, -1, 0], [0, 0, 1]], [[3, 1, 0], [-1, 3, 0]]
        )
        assert np.allclose(q2.data, q1.data)


class TestToFromEuler:
    """These tests address .to_euler() and .from_euler()."""

    def test_to_from_euler(self, eu):
        """Checks that going euler2quat2euler gives no change."""
        eu2 = Quaternion.from_euler(eu).to_euler()
        assert np.allclose(eu, eu2)

        eu3 = Quaternion.from_euler(np.rad2deg(eu), degrees=True).to_euler()
        assert np.allclose(eu, eu3)

        eu4 = Quaternion.from_euler(eu).to_euler(degrees=True)
        assert np.allclose(np.rad2deg(eu), eu4)

    def test_mtex(self, eu):
        _ = Quaternion.from_euler(eu, direction="mtex")

    def test_direction_values(self, eu):
        q_mtex = Quaternion.from_euler(eu, direction="mtex")
        q_c2l = Quaternion.from_euler(eu, direction="crystal2lab")
        q_l2c = Quaternion.from_euler(eu)
        q_default = Quaternion.from_euler(eu)
        assert np.allclose(q_default.data, q_l2c.data)
        assert np.allclose(q_mtex.data, q_c2l.data)
        assert np.allclose((q_l2c * q_c2l).data, [1, 0, 0, 0])

    def test_direction_kwarg(self, eu):
        _ = Quaternion.from_euler(eu)

    def test_direction_kwarg_dumb(self, eu):
        with pytest.raises(ValueError, match="The chosen direction is not one of "):
            _ = Quaternion.from_euler(eu, direction="dumb_direction")

    def test_edge_cases_to_euler(self):
        x = np.sqrt(1 / 2)
        q = Quaternion(np.asarray([x, 0, 0, x]))
        _ = q.to_euler()
        q = Quaternion(np.asarray([0, x, 0, 0]))
        _ = q.to_euler()

    def test_passing_degrees_warns(self):
        with pytest.warns(UserWarning, match="Angles are quite high, did you forget "):
            q = Quaternion.from_euler([90, 0, 0])
            assert np.allclose(q.data, [0.5253, 0, 0, -0.8509], atol=1e-4)

    # TODO: Remove in 0.13
    def test_from_euler_warns(self, eu):
        """Quaternion.from_euler() warns only when "convention" argument
        is passed.
        """
        # No warning is raised
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _ = Quaternion.from_euler(eu)

        msg = (
            r"Argument `convention` is deprecated and will be removed in version 0.13. "
            r"To avoid this warning, please do not use `convention`. "
            r"Use `direction` instead. See the documentation of `from_euler\(\)` for "
            "more details."
        )
        with pytest.warns(np.VisibleDeprecationWarning, match=msg):
            _ = Quaternion.from_euler(eu, convention="whatever")

    # TODO: Remove in 0.13
    def test_from_euler_convention_mtex(self, eu):
        """Passing convention="mtex" to Quaternion.from_euler() works but
        warns.
        """
        q1 = Quaternion.from_euler(eu, direction="crystal2lab")
        with pytest.warns(np.VisibleDeprecationWarning, match=r"Argument `convention`"):
            q2 = Quaternion.from_euler(eu, convention="mtex")
        assert np.allclose(q1.data, q2.data)

    # TODO: Remove in 0.13
    def test_to_euler_convention_warns(self, eu):
        """Quaternion.to_euler() warns only when "convention" argument is
        passed.
        """
        q1 = Quaternion.from_euler(eu)

        # No warning is raised
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            q2 = q1.to_euler()

        msg = (
            r"Argument `convention` is deprecated and will be removed in version 0.13. "
            r"To avoid this warning, please do not use `convention`. "
            r"See the documentation of `to_euler\(\)` for more details."
        )
        with pytest.warns(np.VisibleDeprecationWarning, match=msg):
            q3 = q1.to_euler(convention="whatever")
        assert np.allclose(q2, q3)


class TestFromToMatrix:
    def test_to_matrix(self):
        q = Quaternion([[1, 0, 0, 0], [3, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]])
        om = np.array(
            [np.eye(3), np.eye(3), np.diag([1, -1, -1]), np.diag([1, -1, -1])]
        )
        # Shapes are handled correctly
        assert np.allclose(q.reshape(2, 2).unit.to_matrix(), om.reshape(2, 2, 3, 3))

        q2 = Quaternion(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 0.91, 0.92, 0.93],
                [1, 2, 3, 4],
            ]
        ).unit
        om_from_q2 = q2.to_matrix()
        # Inverse equal to transpose
        assert all(np.allclose(np.linalg.inv(i), i.T) for i in om_from_q2)
        # Cross product of any two rows gives the third
        assert all(np.allclose(np.cross(i[:, 0], i[:, 1]), i[:, 2]) for i in om_from_q2)
        # Sum of squares of any column or row equals unity
        assert np.allclose(np.sum(np.square(om_from_q2), axis=1), 1)  # Rows
        assert np.allclose(np.sum(np.square(om_from_q2), axis=2), 1)  # Columns

    def test_from_matrix(self):
        q = Quaternion([[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]])
        ident = np.identity(3)
        rot_180x = np.diag([1, -1, -1])
        om = np.array([ident, 2 * ident, rot_180x, 2 * rot_180x])
        assert np.allclose(Quaternion.from_matrix(om).data, q.data)
        assert np.allclose(
            Quaternion.from_matrix(om.reshape((2, 2, 3, 3))).data, q.reshape(2, 2).data
        )

    def test_from_to_matrix(self):
        ident = np.identity(3)
        rot_180x = np.diag([1, -1, -1])
        assert np.allclose(
            Quaternion.from_matrix(
                [ident, 2 * ident, rot_180x, 2 * rot_180x]
            ).to_matrix(),
            [ident, ident, rot_180x, rot_180x],
        )

    def test_from_euler_to_matrix_from_matrix(self, eu):
        q = Quaternion.from_euler(eu.reshape((5, 2, 3)))
        assert np.allclose(Quaternion.from_matrix(q.to_matrix()).data, q.data)

    def test_from_matrix_to_euler_from_euler_to_matrix(self, eu):
        rot_180x = np.diag([1, -1, -1])
        rot_180y = np.diag([-1, 1, -1])
        rot_180z = np.diag([-1, -1, 1])

        om1 = np.array([rot_180x, rot_180y, rot_180z])
        quat1 = Quaternion.from_matrix(om1)
        eu = quat1.to_euler()
        quat2 = Quaternion.from_euler(eu)
        om2 = quat2.to_matrix()

        assert np.allclose(om2, om1)

    def test_get_rotation_matrix_from_diffpy(self):
        """Checking that getting rotation matrices from diffpy.structure
        works without issue.
        """
        q = Quaternion.from_matrix([i.R for i in sg225.symop_list])
        assert not np.isnan(q.data).any()


class TestFromAxesAngles:
    """These tests address Quaternion.from_axes_angles()."""

    def test_from_axes_angles(self, rotations):
        axangle = AxAngle.from_rotation(rotations)
        quat2 = Quaternion.from_neo_euler(axangle)
        quat3 = Quaternion.from_axes_angles(axangle.axis.data, axangle.angle)
        assert np.allclose(quat2.data, quat3.data)

        quat4 = Quaternion.from_axes_angles(
            axangle.axis.data, np.rad2deg(axangle.angle), degrees=True
        )
        assert np.allclose(quat4.data, quat3.data)
