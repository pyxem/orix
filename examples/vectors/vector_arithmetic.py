#
# Copyright 2018-2026 the orix developers
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

r"""
=================
Vector arithmetic
=================

This example shows how to perform the following arithmetic operations with vectors
:math:`\mathbf{v} = (x, y, z)`:

- addition :math:`\mathbf{v} + \mathbf{w}`
- subtraction :math:`\mathbf{v} - \mathbf{w}`
- scalar multiplication by a real number :math:`\lambda\mathbf{v}`
- dot product :math:`\mathbf{v} \cdot \mathbf{w}`
- cross product :math:`\mathbf{v} \times \mathbf{w}`
- tensor product :math:`\mathbf{v} \otimes \mathbf{w}`
"""

# %%
import numpy as np

import orix.vector as ove

# %%
# First, inspect some basic properties of vectors.
v = ove.Vector3d([1.0, 2, 3])

# %%
# Access the underlying data and individual components.
print(v.data)

print(v.x)

# %%
# Get the vector length and the unit vector.
print(v.norm)

print(v.unit)
print(v.unit.norm)

# %%
# Negate a vector.
print(-v)

# %%
# Addition and subtraction
# ========================
#
# Add two vectors together.
print(v + ove.Vector3d([1, 0, 1]))

# %%
# Add one vector to two vectors.
print(v + ove.Vector3d([[1, 0, 1], [2, 0, 2]]))

# %%
# Add two scalars to a vector.
print(v + [1, 2])

# %%
# Subtraction is done similarly.
print(v - ove.Vector3d([1, 0, 1]))
print(v - ove.Vector3d([[1, 0, 1], [2, 0, 2]]))
print(v - [1, 2])

# %%
# Scalar multiplication and division
# ==================================
#
# Multiply by one scalar.
print(v * 3)

# %%
# Broadcast to multiple vectors by multiplying by multiple scalars.
print(v * [2, 3])

# %%
# Division is done similarly.
print(v / 3)
print(v / [2, 3])

# %%
# Dot product
# ===========
#
# Dot product 1-1.
u = ove.Vector3d([3, 2, 1])

print(v.dot(u))

# %%
# Dot product 1-many.
print(v.dot(ove.Vector3d([[3, 2, 1], [4, 3, 2]])))

# %%
# Remember to use unit vectors to get lengths of 1.
print(v.dot(v))
print(v.unit.dot(v.unit))

# %%
# Cross product
# =============
#
# Cross product 1-1.
print(v.cross(u))

# %%
# Cross product 1-many.
print(v.cross(ove.Vector3d([[4, 5, 6], [7, 8, 9]])))

# %%
# Tensor product
# ==============
#
# The tensor product is the product of two vectors, denoted by
# :math:`\mathbf{v} \otimes \mathbf{w}`.
v1, v2 = ove.Vector3d([[1.0, 2, 3], [4.0, 5, 6]])

tensor_v12 = v1.tensor(v2)
print(tensor_v12)

# %%
# The tensor product is distributive.
a, b, c = ove.Vector3d.random(3)

tensor1 = (a + b).tensor(c)
tensor2 = a.tensor(c) + b.tensor(c)
print(np.allclose(tensor1, tensor2))

tensor3 = a.tensor(b + c)
tensor4 = a.tensor(c) + a.tensor(b)
print(np.allclose(tensor3, tensor4))

# %%
# The definition of the tensor product (given some places).
abc = np.dot(a.tensor(b)[0], c.data[0])
bca = (b.dot(c) * a).data

print(abc)
print(np.allclose(abc, bca))

# %%
# Tensor outer product.
print(ove.Vector3d.random((2, 4, 5)).tensor_outer(ove.Vector3d.random((6, 7, 8))).shape)
