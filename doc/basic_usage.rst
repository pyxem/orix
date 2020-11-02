Basic Usage
-----------

The use of orix should feel familiar to the use of numpy, but rather
than cells of numbers, the cells contain single 3d objects, such as
vectors or quaternions. Here we demonstrate a few simple functionalities as well as
providing some clarity on how a few technical terms are used within the package.

Vectors
~~~~~~~

Vectors are 3d objects representing positions or directions with
magnitudes. They can be added and subtracted with
other vectors (provided the data are of compatible shapes) and have
several further unique operations.

.. code:: python

    >>> import numpy as np
    >>> from orix.vector import Vector3d
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


Rotations
~~~~~~~~~

Within orix a Rotation describes the transformation of a vector (this formality of language will make more sense upon reading the Orientation section).
Rotations are represented with unit quaternions, as four-dimensional data structures and support
many standard operations; examples below:

.. code:: python

    >>> from orix.quaternion.rotation import Rotation
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

Orientations
~~~~~~~~~~~~

An Orientation is a rotation with a direction specified, or in an equivalent phrasing "an orientation is a rotation with respect to some reference frame".
Often this direction will be chosen so that to be a high symmettry element lies along one of the coordinate axes, and one should take extreme care that
the conventions and settings of all components of software are in agreement.

    The specification of a direction allows us to introduce the concept of symmetry equivalent rotations which we illustrate with an example below.

.. code:: python

    >>> from orix.quaternion.lorum.ipsum
