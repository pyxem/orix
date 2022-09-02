=============
API reference
=============

**Release**: |version|

**Date**: |today|

This reference manual describes the public functions, modules, and objects in orix. Many
of the descriptions include brief examples. For learning how to use orix, see the
:doc:`/examples/index` or :doc:`/tutorials/index`.

.. caution::

    orix is in continuous development, meaning that some breaking changes and changes to
    this reference are likely with each release.

orix is organized in modules (also called subpackages). It is recommended to import
functionality from the below list of functions and modules like this:

.. autolink-skip::
.. code-block:: python

    >>> from orix.quaternion import Orientation, symmetry
    >>> import numpy as np
    >>> ori = Orientation.from_axes_angles([1, 1, 1], np.pi / 2, symmetry.Oh)
    >>> ori
    Orientation (1,) m-3m
    [[0.7071 0.4082 0.4082 0.4082]]

.. currentmodule:: orix

.. rubric:: Modules

.. autosummary::
    :toctree: generated
    :template: custom-module-template.rst

    base
    data
    crystal_map
    io
    plot
    projections
    quaternion
    sampling
    vector
