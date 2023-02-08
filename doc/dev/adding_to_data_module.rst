.. _adding-data-to-data-module:

Adding data to the data module
==============================

Example datasets used in the documentation and tests are included in the
:mod:`orix.data` module via the `pooch <https://www.fatiando.org/pooch/latest/>`__
Python library.
These are listed in a file registry (``orix.data._registry.py``) with their file
verification string (hash, SHA256, obtain with e.g. ``sha256sum <file>``) and location,
the latter potentially not within the package but from the `orix-data
<https://github.com/pyxem/orix-data>`__ repository or elsewhere, since some files are
considered too large to include in the package.

If a required dataset isn't in the package, but is in the registry, it can be downloaded
from the repository when the user passes ``allow_download=True`` to e.g.
``sdss_austenite()``.
The dataset is then downloaded to a local cache, in the location returned from
``pooch.os_cache("orix")``.
The location can be overwritten with a global ``ORIX_DATA_DIR`` variable locally, e.g.
by setting export ``ORIX_DATA_DIR=~/orix_data`` in ``~/.bashrc``.
Pooch handles downloading, caching, version control, file verification (against hash)
etc.
If we have updated the file hash, pooch will re-download it.
If the file is available in the cache, it can be loaded as the other files in the data
module.

With every new version of orix, a new directory of data sets with the version name is
added to the cache directory.
Any old directories are not deleted automatically, and should then be deleted manually
if desired.
