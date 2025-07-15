Importing
=========

To keep the time for importing functions and classes short, such as :class:`Quaternion`,
we use `lazy_loader <https://scientific-python.org/specs/spec-0001/#lazy_loader>`__.
In practice, this means that when a user imports a class, function, or module, only the
functionality necessary is imported and cached, while "untouched" functionality is not
imported.
For example, Numba functions not used in the functionality imported by the user are not
compiled, and libraries not used are not imported.

Another notable benefit to lazy loading is reduced import times for downstream packages.
In most cases, they only use parts of our functionality, and will thus not have to
import other parts they are not using.

New imports go in the ``__init__.pyi`` "stub files", *not* in the ``__init__.py`` files.
Basically, nothing should go in the ``__init__.py`` files except the lazy loading
functionality.

By using stub files, console look-up of available functions in a module or methods for a
class is maintained.