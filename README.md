# texpy

texpy is a package under development for the handling of quaternion objects, built primarily on top of [numpy](http://www.numpy.org/) and [matplotlib](https://matplotlib.org/) and heavily inspired by the [MATLAB](https://www.mathworks.com/products/matlab.html) package [MTEX](http://mtex-toolbox.github.io/).

While developed and intended with crystallographic texture analysis in mind, the handling of vectors and quaternions is kept as general as possible to allow applications in other areas, given interest.

## Installation

As texpy is still under development there has been no official release, but installation via GitHub is easy.

1. Download and install [Anaconda](https://www.anaconda.com/download/) according to system instructions.
2. Open a shell to a suitable local directory. Create and activate a clean environment for texpy:
  ```
  > conda create --name texpy-env python=3
  > activate texpy-env  # Windows
  > source activate texpy-env  # Linux/MacOS  
  ```
4. Install numpy, and optionally matplotlib:
  ```
  > conda install numpy matplotlib
  ```
3. Install texpy from GitHub using pip:
  ```
  > pip install git+https://github.com/bm424/texpy.git
  ```
  This will always install the latest version. Given sufficient interest, specific versions may be released for backwards compatibility and citation purposes.

## Getting started

The use of texpy should feel familiar to the use of numpy, but rather than cells of numbers, the cells contain single 3d objects, such as vectors or quaternions. They can all be created using tuples, lists, numpy arrays, or other numpy-compatible iterables, and will raise an error if constructed with the incorrect number of dimensions. Here only a few basic examples are shown - a complete feature list will be available in future versions of the documentation.

### Vectors

Vectors are 3d objects representing positions or directions with "magnitude". They can be added and subtracted with integers, floats, or other vectors (provided the data are of compatible shapes) and have several further unique operations.

```python
>>> import numpy as np
>>> from texpy.vector.vector3d import Vector3d
>>> v = Vector3d((1, 1, -1))
>>> w_array = np.array([[[1, 0, 0], [0, 0, -1]], [[1, 1, 0], [-1, 0, -1]]])
>>> w = Vector3d(w_array)
>>> v + w
# Vector3d (2, 2)
# [[[ 2  1 -1]
#   [ 1  1 -2]]
#
#  [[ 2  2 -1]
#   [ 0  1 -2]]]
>>> v.dot(w)
# array([[1, 1],
#        [2, 0]])
>>> v.cross(w)
# Vector3d (2, 2)
# [[[ 0 -1 -1]
#   [-1  1  0]]
#
#  [[ 1 -1  0]
#   [-1  2  1]]]
>>> v.unit
# Vector3d (1,)
# [[ 0.5774  0.5774 -0.5774]]
>>> w[0]
# Vector3d (2,)
# [[ 1  0  0]
#   [ 0  0 -1]]
>>> w[:, 0]
# Vector3d (2,)
# [[1 0 0]
#  [1 1 0]]
```

### Quaternions

Quaternions are four-dimensional data structures. Unit quaternions are often used for representing rotations in 3d. Quaternion multiplication is defined and can be applied to either other quaternions or vectors.

```python
from texpy.quaternion.rotation import Rotation
>>> p = Rotation([0.5, 0.5, 0.5, 0.5])
>>> q = Rotation([0, 1, 0, 0])
>>> p.axis
# Vector3d (1,)
# [[0.5774 0.5774 0.5774]]
>>> p.angle
# array([2.0943951])
>>> p * q
# Rotation (1,)
# [[-0.5  0.5  0.5 -0.5]]
>>> p * ~p # (unit rotation)
# Rotation (1,)
# [[1. 0. 0. 0.]]
>>> p.to_euler() # (Euler angles in the Bunge convention)
# array([[1.57079633, 1.57079633, 0.        ]])

```
