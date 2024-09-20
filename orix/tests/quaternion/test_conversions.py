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

import numpy as np
import pytest

from orix.quaternion import Orientation
from orix.quaternion._conversions import (
    ax2qu,
    ax2qu_2d,
    ax2qu_single,
    ax2ro,
    ax2ro_2d,
    ax2ro_single,
    cu2ho,
    cu2ho_2d,
    cu2ho_single,
    cu2ro,
    cu2ro_2d,
    cu2ro_single,
    eu2qu,
    eu2qu_2d,
    eu2qu_single,
    get_pyramid,
    get_pyramid_2d,
    get_pyramid_single,
    ho2ax,
    ho2ax_2d,
    ho2ax_single,
    ho2ro,
    ho2ro_2d,
    ho2ro_single,
    om2qu,
    om2qu_3d,
    om2qu_single,
    qu2ax,
    qu2ax_2d,
    qu2ax_single,
    qu2eu,
    qu2eu_2d,
    qu2eu_single,
    qu2ho,
    qu2ho_2d,
    qu2ho_single,
    qu2om,
    qu2om_2d,
    qu2om_single,
    ro2ax,
    ro2ax_2d,
    ro2ax_single,
)
from orix.quaternion.symmetry import Oh


class TestRotationConversions:
    """Tests of conversions between rotation representations.

    Functions are accelerated with Numba, so both the Numba and Python
    versions of the functions are tested.
    """

    def test_eu2qu2eu(self, euler_angles, quaternions_conversions):
        qu_64 = quaternions_conversions
        eu_64 = euler_angles
        qu_32_nd = qu_64.astype(np.float32).reshape(2, 5, 4)
        eu_32_nd = eu_64.astype(np.float32).reshape(2, 5, 3)

        # Single
        for qu, eu in zip(qu_64, eu_64):
            assert np.allclose(qu2eu_single.py_func(qu), eu, atol=1e-4)
            assert np.allclose(eu2qu_single.py_func(eu), qu, atol=1e-4)

        # 2D
        assert np.allclose(qu2eu_2d.py_func(qu_64), eu_64, atol=1e-4)
        assert np.allclose(eu2qu_2d.py_func(eu_64), qu_64, atol=1e-4)

        # ND and jit
        qu_64_nd = qu_64.reshape(2, 5, 4)
        eu_64_nd = eu_64.reshape(2, 5, 3)
        assert np.allclose(eu2qu(eu_64_nd), qu_64_nd, atol=1e-4)
        assert np.allclose(qu2eu(qu_64_nd), eu_64_nd, atol=1e-4)

        # ND float32
        assert np.allclose(qu2eu(qu_32_nd), eu_32_nd, atol=1e-4)
        assert np.allclose(eu2qu(eu_32_nd), qu_32_nd, atol=1e-4)

        # Symmetry-preserving
        ori = Orientation.from_euler(eu_64[-1], Oh)
        assert ori.symmetry == Oh
        assert np.allclose(ori.data[0], qu_64[-1], atol=1e-4)

    def test_get_pyramid(self, cubochoric_coordinates):
        """Cubochoric coordinates situated in expected pyramid."""
        cu_64 = cubochoric_coordinates
        pyramid = np.array([3, 1, 3, 1, 1, 2, 3, 4, 5, 6])

        # Single
        for xyz, p in zip(cu_64, pyramid):
            assert get_pyramid_single.py_func(xyz) == p

        # 2D
        assert np.allclose(get_pyramid_2d.py_func(cu_64), pyramid)

        # ND
        cu_64_nd = cu_64.reshape(5, 2, 3)
        assert np.allclose(get_pyramid(cu_64_nd), pyramid)

    def test_cu2ho(self, cubochoric_coordinates, homochoric_vectors):
        cu_64 = cubochoric_coordinates
        ho_64 = homochoric_vectors

        # Single
        for cu, ho in zip(cu_64, ho_64):
            assert np.allclose(cu2ho_single.py_func(cu), ho, atol=1e-4)

        # 2D
        assert np.allclose(cu2ho_2d.py_func(cu_64), ho_64, atol=1e-4)

        # ND
        cu_64_nd = cu_64.reshape(5, 2, 3)
        ho_64_nd = ho_64.reshape(5, 2, 3)
        assert np.allclose(cu2ho(cu_64_nd), ho_64_nd, atol=1e-4)

        # ND float32
        cu_32_nd = cu_64_nd.astype(np.float32)
        ho_32_nd = ho_64_nd.astype(np.float32)
        assert np.allclose(cu2ho(cu_32_nd), ho_32_nd, atol=1e-4)

    def test_ho2ax(self, homochoric_vectors, axis_angle_pairs):
        ho_64 = homochoric_vectors
        ax_64 = axis_angle_pairs

        # Single
        for ho, ax in zip(ho_64, ax_64):
            assert np.allclose(ho2ax_single.py_func(ho), ax, atol=1e-4)

        # 2D
        assert np.allclose(ho2ax_2d.py_func(ho_64), ax_64, atol=1e-4)

        # ND
        ho_64_nd = ho_64.reshape(2, 5, 3)
        ax_64_nd = ax_64.reshape(2, 5, 4)
        assert np.allclose(ho2ax(ho_64_nd), ax_64_nd, atol=1e-4)

        # ND float32
        ho_32_nd = ho_64_nd.astype(np.float32)
        ax_32_nd = ax_64_nd.astype(np.float32)
        assert np.allclose(ho2ax(ho_32_nd), ax_32_nd, atol=1e-4)

    def test_ax2ro2ax(
        self, axis_angle_pairs, rodrigues_vectors, quaternions_conversions
    ):
        ax_64 = axis_angle_pairs
        ro_64 = rodrigues_vectors

        # Single
        for ax, ro in zip(ax_64, ro_64):
            assert np.allclose(ax2ro_single.py_func(ax), ro, atol=1e-4)
            assert np.allclose(ro2ax_single.py_func(ro), ax, atol=1e-4)

        # 2D
        assert np.isinf(ax2ro_single.py_func(np.array([0, 0, 0, np.pi]))[3])
        assert np.allclose(ax2ro_2d.py_func(ax_64), ro_64, atol=1e-4)
        assert np.allclose(ro2ax_2d.py_func(ro_64), ax_64, atol=1e-4)

        # ND
        ax_64_nd = ax_64.reshape(2, 5, 4)
        ro_64_nd = ro_64.reshape(2, 5, 4)
        assert np.allclose(ax2ro(ax_64_nd), ro_64_nd, atol=1e-4)
        assert np.allclose(ro2ax(ro_64_nd), ax_64_nd, atol=1e-4)

        # ND float32
        ax_32_nd = ax_64_nd.astype(np.float32)
        ro_32_nd = ro_64_nd.astype(np.float32)
        assert np.allclose(ax2ro(ax_32_nd), ro_32_nd, atol=1e-4)
        assert np.allclose(ro2ax(ro_32_nd), ax_32_nd, atol=1e-4)

    def test_ax2qu2ax(self, axis_angle_pairs, quaternions_conversions):
        ax_64 = axis_angle_pairs
        qu_64 = quaternions_conversions

        # Single
        for ax, qu in zip(ax_64, qu_64):
            assert np.allclose(ax2qu_single.py_func(ax), qu, atol=1e-4)
            assert np.allclose(qu2ax_single.py_func(qu), ax, atol=2e-4)

        # Test conversion of lower hemisphere quaternion
        qu_lower = np.array([-1, 1, 1, 0], dtype=np.float64)
        ax_lower = ax2qu_single(qu_lower)
        assert np.allclose(ax_lower, np.array([1, 0, 0, 0]), atol=1e-4)

        # 2D
        assert np.allclose(ax2qu_2d.py_func(ax_64), qu_64, atol=1e-4)
        assert np.allclose(qu2ax_2d.py_func(qu_64), ax_64, atol=2e-4)

        # ND
        ax_64_nd = ax_64.reshape(2, 5, 4)
        qu_64_nd = qu_64.reshape(2, 5, 4)
        assert np.allclose(
            ax2qu(ax_64_nd[..., :3], ax_64_nd[..., 3]), qu_64_nd, atol=1e-4
        )
        ax_64_nd_out = np.dstack(qu2ax(qu_64_nd))
        assert np.allclose(ax_64_nd_out, ax_64_nd, atol=2e-4)

        # ND float32
        ax_32_nd = ax_64_nd.astype(np.float32)
        qu_32_nd = qu_64_nd.astype(np.float32)
        axes_32_nd = ax_32_nd[..., :3]
        angles_32_nd = ax_32_nd[..., 3]
        assert np.allclose(ax2qu(axes_32_nd, angles_32_nd), qu_32_nd, atol=1e-4)
        ax_32_nd_out = np.dstack(qu2ax(qu_32_nd))
        assert np.allclose(ax_32_nd_out, ax_32_nd, atol=2e-4)

        # Make sure bad data causes the expected errors
        with pytest.raises(ValueError, match="Final dimension of axes array must be 3"):
            ax2qu(axes_32_nd.T, angles_32_nd)
        with pytest.raises(
            ValueError,
            match=r"The dimensions of axes \(2, 3\) and angles \(2, 2\) are ",
        ):
            ax2qu(axes_32_nd[:, :3], angles_32_nd[:, :2])
        with pytest.raises(
            ValueError, match="Final dimension of quaternion array must be 4"
        ):
            qu2ax(qu_64[:, :3])

    def test_qu2ax_negative_magnitude(self):
        q = np.array([-0.1, 0.2, 0.3, 0.4])
        ax = qu2ax_single.py_func(q)
        assert np.allclose(ax, [-0.3714, -0.5571, -0.7428, 3.3419], atol=1e-4)

    def test_ho2ro(self, homochoric_vectors, rodrigues_vectors):
        ho_64 = homochoric_vectors
        ro_64 = rodrigues_vectors

        # Single
        for ho, ro in zip(ho_64, ro_64):
            assert np.allclose(ho2ro_single.py_func(ho), ro, atol=1e-4)

        # 2D
        assert np.allclose(ho2ro_2d.py_func(ho_64), ro_64, atol=1e-4)

        # ND
        ho_64_nd = ho_64.reshape(2, 5, 3)
        ro_64_nd = ro_64.reshape(2, 5, 4)
        assert np.allclose(ho2ro(ho_64_nd), ro_64_nd, atol=1e-4)

        # ND float32
        ho_32_nd = ho_64_nd.astype(np.float32)
        rod_32_nd = ro_64_nd.astype(np.float32)
        assert np.allclose(ho2ro(ho_32_nd), rod_32_nd, atol=1e-4)

    def test_cu2ro(self, cubochoric_coordinates, rodrigues_vectors):
        cu_64 = cubochoric_coordinates
        ro_64 = rodrigues_vectors

        # Single
        for cu, ro in zip(cu_64, ro_64):
            assert np.allclose(cu2ro_single.py_func(cu), ro, atol=1e-4)

        # 2D
        assert np.allclose(cu2ro_2d.py_func(cu_64), ro_64, atol=1e-4)

        # ND
        cu_64_nd = cu_64.reshape(2, 5, 3)
        ro_64_nd = ro_64.reshape(2, 5, 4)
        assert np.allclose(cu2ro(cu_64_nd), ro_64_nd, atol=1e-4)

        # ND float32
        cu_32_nd = cu_64_nd.astype(np.float32)
        ro_32_nd = ro_64_nd.astype(np.float32)
        assert np.allclose(cu2ro(cu_32_nd), ro_32_nd, atol=1e-4)

    def test_om2qu2om(self, orientation_matrices, quaternions_conversions):
        # Checks both om2qu and qu2om simultaneously
        om_64 = orientation_matrices
        qu_64 = quaternions_conversions

        # Single
        for om, qu in zip(om_64, qu_64):
            assert np.allclose(om2qu_single.py_func(om), qu, atol=1e-4)
            assert np.allclose(qu2om_single.py_func(qu), om, atol=1e-4)

        # Test edge cases where q[0] is at or near zero
        rot_pi_111 = np.array([[-1, 2, 2], [2, -1, 2], [2, 2, -1]]) / 3
        qu_from_om = om2qu_single.py_func(rot_pi_111)
        qu_actual = np.array([0, np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3)])
        om_from_qu = qu2om_single.py_func(qu_actual)
        assert np.allclose(qu_from_om, qu_actual)
        assert np.allclose(om_from_qu, rot_pi_111)

        # 2D
        assert np.allclose(om2qu_3d.py_func(om_64), qu_64, atol=1e-4)
        assert np.allclose(om_64, qu2om_2d.py_func(qu_64), atol=1e-4)

        # ND
        om_64_nd = om_64.reshape(2, 5, 3, 3)
        qu_64_nd = qu_64.reshape(2, 5, 4)
        assert np.allclose(om2qu(om_64_nd), qu_64_nd, atol=1e-4)
        assert np.allclose(om_64_nd, qu2om(qu_64_nd), atol=1e-4)

        # ND float32
        om_32_nd = om_64_nd.astype(np.float32)
        qu_32_nd = qu_64_nd.astype(np.float32)
        assert np.allclose(om2qu(om_32_nd), qu_32_nd, atol=1e-4)
        assert np.allclose(om_32_nd, qu2om(qu_32_nd), atol=1e-4)

    def test_om2qu(self, orientation_matrices, quaternions_conversions):
        om_64 = orientation_matrices
        qu_64 = quaternions_conversions

        # Single
        for om, qu in zip(om_64, qu_64):
            assert np.allclose(om2qu_single.py_func(om), qu, atol=1e-4)

        # Test edge cases where q[0] is at or near zero
        rot_pi_111 = np.array([[-1, 2, 2], [2, -1, 2], [2, 2, -1]]) / 3
        qu_from_om = om2qu_single.py_func(rot_pi_111)
        qu_actual = np.array([0, np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3)])
        assert np.allclose(qu_from_om, qu_actual)

        # 2D
        assert np.allclose(om2qu_3d.py_func(om_64), qu_64, atol=1e-4)

        # ND
        om_64_nd = om_64.reshape(2, 5, 3, 3)
        qu_64_nd = qu_64.reshape(2, 5, 4)
        assert np.allclose(om2qu(om_64_nd), qu_64_nd, atol=1e-4)

        # ND float32
        om_32_nd = om_64_nd.astype(np.float32)
        qu_32_nd = qu_64_nd.astype(np.float32)
        assert np.allclose(om2qu(om_32_nd), qu_32_nd, atol=1e-4)

    def test_qu2ho(self, quaternions_conversions, homochoric_vectors):
        qu_64 = quaternions_conversions
        ho_64 = homochoric_vectors

        # Single
        for qu, ho in zip(qu_64, ho_64):
            assert np.allclose(qu2ho_single.py_func(qu), ho, atol=1e-4)

        # 2D
        assert np.allclose(qu2ho_2d.py_func(qu_64), ho_64, atol=1e-4)

        # ND
        qu_64_nd = qu_64.reshape(2, 5, 4)
        ho_64_nd = ho_64.reshape(2, 5, 3)
        assert np.allclose(qu2ho(qu_64_nd), ho_64_nd, atol=1e-4)

        # ND float32
        qu_32_nd = qu_64_nd.astype(np.float32)
        ho_32_nd = ho_64_nd.astype(np.float32)
        assert np.allclose(qu2ho(qu_32_nd), ho_32_nd, atol=1e-4)

        with pytest.raises(
            ValueError, match="Final dimension of quaternion array must be 4"
        ):
            qu2ho(qu_32_nd[..., :3])

    def test_qu2ho2ax2qu(self, quaternions_conversions):
        qu1 = quaternions_conversions
        ho = qu2ho(qu1)
        ax = ho2ax(ho)
        qu2 = ax2qu(ax[..., :3], ax[..., 3])
        assert np.allclose(qu1, qu2, atol=1e-3)  # 1e-4 errors
