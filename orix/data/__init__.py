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

"""Test data.

Some datasets must be downloaded from the web. Datasets are placed in a
local cache, in the location returned from `pooch.os_cache("orix")` by
default. The location can be overwritten with a global `ORIX_DATA_DIR`
environment variable.

With every new version of orix, a new directory of data sets with the
version name is added to the cache directory. Any old directories are
not deleted automatically, and should then be deleted manually if
desired.
"""

import numpy as np
import pooch

from orix import __version__, io
from orix.data._registry import registry_hashes, registry_urls
from orix.quaternion import Orientation, symmetry


__all__ = [
    "sdss_austenite",
    "sdss_ferrite_austenite",
    "ti_orientations",
]


_fetcher = pooch.create(
    path=pooch.os_cache("orix"),
    base_url="",
    version=__version__.replace(".dev", "+"),
    version_dev="master",
    env="ORIX_DATA_DIR",
    registry=registry_hashes,
    urls=registry_urls,
)


def _fetch(filename, allow_download):
    file_path_in_cache = _fetcher.path / filename
    if (
        not file_path_in_cache.exists()
        or not pooch.file_hash(file_path_in_cache) == registry_hashes[filename]
    ):
        if allow_download:
            download = pooch.HTTPDownloader(progressbar=True)
            file_path_in_cache = _fetcher.fetch(filename, downloader=download)
        else:
            raise ValueError(
                f"Dataset {filename} must be (re)downloaded from the orix-data "
                "repository on GitHub (https://github.com/pyxem/orix-data) to your "
                "local cache with the pooch Python package. Pass `allow_download=True`"
                " to allow this download."
            )
    return file_path_in_cache


def sdss_austenite(allow_download=False, **kwargs):
    """Single phase Austenite crystallographic map of shape (100, 117)
    produced from dictionary indexing of electron backscatter
    diffraction (EBSD) patterns of a super duplex stainless steel (SDSS)
    sample containing both Austenite and Ferrite grains.

    The EBSD data was acquired by Jarle Hjelen from the Norwegian
    University of Science and Technology and carries the CC BY 4.0
    license.

    Parameters
    ----------
    allow_download : bool, optional
        Whether to allow downloading the dataset from the internet to
        the local cache with the pooch Python package. Default is False.
    kwargs
        Keyword arguments passed to
        :func:`~orix.io.plugins.emsoft_h5ebsd.file_reader`.

    Returns
    -------
    CrystalMap

    Examples
    --------
    Read only refined orientations from the EMsoft HDF5 file by passing
    keyword arguments on to the reader

    >>> from orix import data
    >>> xmap = data.sdss_austenite(allow_download=True, refined=True)  # doctest: +SKIP
    >>> xmap
    Phase    Orientations       Name  Space group  Point group  Proper point group     Color
        0  11700 (100.0%)  austenite         None         m-3m                 432  tab:blue
    Properties: AvDotProductMap, CI, IQ, ISM, KAM, OSM, RefinedDotProducts, TopDotProductList, TopMatchIndices
    Scan unit: um
    """
    fname = _fetch("sdss/sdss_austenite.h5", allow_download)
    return io.load(fname, **kwargs)


def sdss_ferrite_austenite(allow_download=False, **kwargs):
    """Dual phase Austenite and Ferrite crystallographic map of shape
    (100, 117) produced from dictionary indexing of electron backscatter
    diffraction patterns of a super duplex stainless steel (SDSS)
    sample.

    The EBSD data was acquired by Jarle Hjelen from the Norwegian
    University of Science and Technology and carries the CC BY 4.0
    license.

    Parameters
    ----------
    allow_download : bool, optional
        Whether to allow downloading the dataset from the internet to
        the local cache with the pooch Python package. Default is False.
    kwargs
        Keyword arguments passed to
        :func:`~orix.io.plugins.ang.file_reader`.

    Returns
    -------
    CrystalMap

    Examples
    --------
    Read data and plot phase map

    >>> from orix import data
    >>> xmap = data.sdss_ferrite_austenite(allow_download=True)  # doctest: +SKIP
    >>> xmap
    Phase   Orientations       Name  Space group  Point group  Proper point group       Color
        1   5657 (48.4%)  austenite         None          432                 432    tab:blue
        2   6043 (51.6%)    ferrite         None          432                 432  tab:orange
    Properties: iq, dp
    Scan unit: um
    >>> xmap.plot()
    """
    fname = _fetch("sdss/sdss_ferrite_austenite.ang", allow_download)
    xmap = io.load(fname, **kwargs)
    xmap.phases["austenite/austenite"].name = "austenite"
    xmap.phases["ferrite/ferrite"].name = "ferrite"
    return xmap


def ti_orientations(allow_download=False):
    """Orientations in the MTEX orientation convention (crystal2lab)
    from an orientation map produced from Hough indexing of electron
    backscatter diffraction patterns of a commercially pure hexagonal
    close packed titanium sample (*6/mmm*) :cite:`johnstone2020density`.

    The data set is part of the supplementary material to
    :cite:`krakow2017onthree` and carries the CC BY 4.0 license.

    Parameters
    ----------
    allow_download : bool, optional
        Whether to allow downloading the dataset from the internet to
        the local cache with the pooch Python package. Default is False.

    Returns
    -------
    Orientations

    Examples
    --------
    >>> from orix import data
    >>> g = data.ti_orientations()  # doctest: +SKIP
    >>> g
    Orientation (193167,) 622
    [[ 0.3027  0.0869 -0.5083  0.8015]
     [ 0.3088  0.0868 -0.5016  0.8034]
     [ 0.3057  0.0818 -0.4995  0.8065]
     ...
     [ 0.4925 -0.1633 -0.668   0.5334]
     [ 0.4946 -0.1592 -0.6696  0.5307]
     [ 0.4946 -0.1592 -0.6696  0.5307]]
    """
    fname = _fetch("ti_orientations/ti_orientations.ctf", allow_download)
    euler = np.loadtxt(fname, skiprows=1, usecols=(0, 1, 2))
    return Orientation.from_euler(np.deg2rad(euler), symmetry.D6)
