Importing
=========

To keep the time for importing modules, such as :mod:`quaternion`, we use
`lazy_loader <https://scientific-python.org/specs/spec-0001/#lazy_loader>`__ (`GitHub
README <https://github.com/scientific-python/lazy-loader>`__).
In practice, this means that when a user imports a module, class, or function, only the
functionality necessary is imported and cached, while "untouched" functionality is not
imported.

For example, :mod:`numba`` functions not used in the functionality imported by the user
are not compiled, and libraries not used are not imported.
For example, quaternion module and classes in it (:class:`Orientation`,
:class:`Rotation`, etc.) based on it use functions leveraging Numba to quickly convert
to and from different conventions, such as :meth:`Orientation.from_euler` or
:meth:`Orientation.to_rodrigues`.
These are fast, but require compilation before first use, which can take time.
Since it is unlikely that a script would require all these functions, it makes sense to
only compile those needed, as they are needed.
Lazy loading takes care of this so users do not, thus speeding up imports.

This time saving is also passed on to downsteam packages that import orix.
In most cases, this results in faster import times without any additional work from
downstream developers.

New imports go in the ``__init__.pyi`` "stub files", *not* in the ``__init__.py`` files.
By using stub files, static type checkers (e.g. in a more advanced editor) are able to
read these files and infer information about the modules and functions (specifically,
their parameter and return "types"), without actually loading the modules themselves.

No imports should go in the ``__init__.py`` files except the lazy loading
functionality::

     import lazy_loader

     __getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

     del lazy_loader

The returns from lazy loader are:
- ``__getattr__``: function to access names defined by the module
- ``__dir__``: list of names a module defines
- ``__all__``: list of module, class, or function names that should be imported when
  ``from package import *`` is encountered

We delete the lazy loader import at the end to prevent importing it from our modules.
