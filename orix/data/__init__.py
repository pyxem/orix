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
local cache, in the location returned from ``pooch.os_cache("orix")`` by
default. The location can be overwritten with a global ``ORIX_DATA_DIR``
environment variable.

With every new version of orix, a new directory of data sets with the
version name is added to the cache directory. Any old directories are
not deleted automatically, and should then be deleted manually if
desired.
"""

from pathlib import Path

import numpy as np
import pooch
import warnings

from orix import __version__, io
from orix.crystal_map import CrystalMap
from orix.data._registry import registry_hashes, registry_urls
from orix.quaternion import Orientation, symmetry
import gdown

__all__ = [
    "sdss_austenite",
    "sdss_ferrite_austenite",
    "ti_orientations",
    "AF96",
]


_fetcher = pooch.create(
    path=pooch.os_cache("orix"),
    base_url="",
    version=__version__.replace(".dev", "+"),
    version_dev="develop",
    env="ORIX_DATA_DIR",
    registry=registry_hashes,
    urls=registry_urls,
)


def _fetch(filename, allow_download):
    file_path_in_cache = Path(_fetcher.path) / filename
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


def sdss_austenite(allow_download: bool = False, **kwargs) -> CrystalMap:
    """Return a single phase Austenite crystallographic map produced
    with dictionary indexing of electron backscatter diffraction
    patterns of a super duplex stainless steel (SDSS) sample containing
    both Austenite and Ferrite grains.

    Parameters
    ----------
    allow_download
        Whether to allow downloading the dataset from the internet to
        the local cache with the pooch Python package. Default is
        ``False``.
    **kwargs
        Keyword arguments passed to
        :func:`~orix.io.plugins.emsoft_h5ebsd.file_reader`.

    Returns
    -------
    xmap
        Crystal map of shape (100, 117).

    Notes
    -----
    The EBSD data was acquired by Jarle Hjelen from the Norwegian
    University of Science and Technology and carries the CC BY 4.0
    license.

    Examples
    --------
    Read only refined orientations from the EMsoft HDF5 file by passing
    keyword arguments on to the reader and plot the dot product map

    >>> from orix import data, plot
    >>> xmap = data.sdss_austenite(allow_download=True, refined=True)
    >>> xmap
    Phase    Orientations       Name  Space group  Point group  Proper point group     Color
        0  11700 (100.0%)  austenite         None         m-3m                 432  tab:blue
    Properties: AvDotProductMap, CI, IQ, ISM, KAM, OSM, RefinedDotProducts, TopDotProductList, TopMatchIndices
    Scan unit: um
    >>> ipf_key = plot.IPFColorKeyTSL(xmap.phases[0].point_group)
    >>> xmap.plot(ipf_key.orientation2color(xmap.orientations), overlay="RefinedDotProducts")
    """
    fname = _fetch("sdss/sdss_austenite.h5", allow_download)
    return io.load(fname, **kwargs)


def sdss_ferrite_austenite(allow_download: bool = False, **kwargs) -> CrystalMap:
    """Return a dual phase Austenite and Ferrite crystallographic map
    produced with dictionary indexing of electron backscatter
    diffraction patterns of a super duplex stainless steel (SDSS)
    sample.

    Parameters
    ----------
    allow_download
        Whether to allow downloading the dataset from the internet to
        the local cache with the pooch Python package. Default is
        ``False``.
    **kwargs
        Keyword arguments passed to
        :func:`~orix.io.plugins.ang.file_reader`.

    Returns
    -------
    xmap
        Crystal map of shape (100, 117).

    Notes
    -----
    The EBSD data was acquired by Jarle Hjelen from the Norwegian
    University of Science and Technology and carries the CC BY 4.0
    license.

    Examples
    --------
    Read data and plot phase map

    >>> from orix import data
    >>> xmap = data.sdss_ferrite_austenite(allow_download=True)
    >>> xmap
    Phase   Orientations       Name  Space group  Point group  Proper point group       Color
        1   5657 (48.4%)  austenite         None          432                 432    tab:blue
        2   6043 (51.6%)    ferrite         None          432                 432  tab:orange
    Properties: iq, dp
    Scan unit: um
    >>> xmap.plot(overlay="dp")
    """
    fname = _fetch("sdss/sdss_ferrite_austenite.ang", allow_download)
    xmap = io.load(fname, **kwargs)
    xmap.phases["austenite/austenite"].name = "austenite"
    xmap.phases["ferrite/ferrite"].name = "ferrite"
    return xmap


def af96_martensitic_steels(
    allow_download=False, dataset="large", subset="all", **kwargs
):
    """
    Returns EBSD data taken from a sample of low-alloy, high-performance, AF96
    Martensitic steel. Details on the processing, properties, and composition
    of this steel can be found in the following two publications:
        https://doi.org/10.1016/j.dib.2019.104471
        https://doi.org/10.1016/j.matchar.2019.109835
    and the original ang files can be found through Globus:
        https://doi.org/10.18126/iv89-3293

    There are three datasets related to this data:
        "ang" - modified versions of the original ang files with corrected
                header information.
        "large" - a set of five CrystalMaps stored using Orix's hdf5 format.
                Each file is a unique 1048-by-2116 map from the same sample.
        "small" - a set of ninety CrystalMaps stored using Orix's hdf5 format.
                Each file is a non-unique 512-by-512 map from the same sample.

    Parameters
    ----------
    allow_download : bool
        Whether to allow downloading the dataset from the internet to
        the local cache with the pooch Python package. Default is
        ``False``.
    dataset : "large" or "small" or "ang"
        sets whether to use data from the large, small, or ang datasets listed
        above. In all cases, data is returned as a CrystalMap object. The
        default is "large".
    subset : int or list of ints or 'all'
        Determines whether all the scans in a dataset or only a subset are
        returned. If 'all', returns a list of all scans in the set. If an
        integer, returns only the ith scan in the dataset. Can also pass a list
        or array of multiple integers, and it will return only that subset of
        the full set.
    **kwargs
        Keyword arguments passed to
        :func:`~orix.io.plugins.ang.file_reader`.

    Returns
    -------
    xmap : orix.CrystalMap
        either a single or a list of multiple Crystal maps. shape is either
        (1048, 2116) or (512, 512)

    Notes
    -----
    The EBSD data was acquired by Vikas Sinha at the Air Force Research Lab in
    Dayton, Ohio, USA, and carries the CC BY 4.0 license.

    Examples
    --------
    Read in multiple files

    >>> from orix import data
    >>> xmaps = af96_martensitic_steels(dataset='small', subset=np.arange(2))
    >>> xmaps
    [
     Phase     Orientations         Name  Space group  Point group  Proper point group       Color
        -1         6 (0.0%)  not_indexed         None         None                None           w
         0      2046 (0.8%)    austenite        Fm-3m         m-3m                 432    tab:blue
         1   260092 (99.2%)      ferrite        Im-3m         m-3m                 432  tab:orange
     Properties: ci, fit_parameter, iq
     Scan unit: px
     ,
     Phase     Orientations         Name  Space group  Point group  Proper point group       Color
        -1         4 (0.0%)  not_indexed         None         None                None           w
         0      1792 (0.7%)    austenite        Fm-3m         m-3m                 432    tab:blue
         1   260348 (99.3%)      ferrite        Im-3m         m-3m                 432  tab:orange
     Properties: ci, fit_parameter, iq
     Scan unit: px
     ]

    Read in all 5 large files
    >>> xmap =af96_martensitic_steels(dataset='large')

    Read in only the fourth file of the large AF96 dataset
    >>> xmap =af96_martensitic_steels(dataset='large',subset=4)

    """
    # Parse "dataset" to find the subset of AF96 files desired by the user
    AF96_keys = [x for x in registry_hashes.keys() if x[:4] == "AF96"]
    if dataset in "Largelarge":
        sub_keys = [x for x in AF96_keys if "Large" in x and ".h5" in x]
    elif dataset in "Smallsmall":
        sub_keys = [x for x in AF96_keys if "Small" in x and ".h5" in x]
    elif dataset in ".ang.Ang":
        sub_keys = [x for x in AF96_keys if ".ang" in x]
    else:
        raise ValueError(
            "AF96 contains 3 subsets of data, 'Large'(1048 by 2116 pixels),"
            "'small'(512 by 512 pixels), or '.ang',(original .ang text files)."
            "given choice of {dataset} is invalid"
        )
    # convert "subset" into a 1d array of integers
    if str(subset) == "all":
        indices = np.arange(len(sub_keys))
    else:
        indices = np.asanyarray([[subset]]).astype(int).flatten()
    if indices.max() > len(sub_keys):
        warnings.warn(
            """This AF96 dataset contains only {} scans, but subset
    has a max value of {}. Floor dividing indices to keep
    within appropropriate range""".format(
                len(sub_keys), indices.max()
            )
        )
        indices = indices % len(sub_keys)
    xmaps = []
    # for each item in dataset[subset], either load from cache or download
    for i in indices:
        path = _fetcher.path / sub_keys[i]
        if not path.exists() and not allow_download:
            filename = sub_keys[i][5:]
            raise ValueError(
                f"Dataset {filename} must be (re)downloaded from the orix-data "
                "repository on GitHub (https://github.com/pyxem/orix-data) to your "
                "local cache with the pooch Python package. Pass `allow_download=True`"
                " to allow this download."
            )
        md5 = registry_hashes[sub_keys[i]]
        drive_id = registry_urls[sub_keys[i]].split("/d/")[1]
        gdown.cached_download(id=drive_id, path=path, md5=md5)
        xmaps.append(io.load(path, **kwargs))
    if len(xmaps) == 1:
        xmaps = xmaps[0]
    return xmaps


def ti_orientations(allow_download: bool = False) -> Orientation:
    """Return orientations in the MTEX orientation convention
    (crystal2lab) from an orientation map produced from Hough indexing
    of electron backscatter diffraction patterns of a commercially pure
    hexagonal close packed titanium sample (*6/mmm*).

    Parameters
    ----------
    allow_download
        Whether to allow downloading the dataset from the internet to
        the local cache with the pooch Python package. Default is
        ``False``.

    Returns
    -------
    ori
        Ti orientations of shape (193167,).

    Notes
    -----
    The data set is part of the supplementary material to
    :cite:`krakow2017onthree` and carries the CC BY 4.0 license.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from orix import data, plot
    >>> from orix.quaternion.symmetry import D6
    >>> ori = data.ti_orientations(allow_download=True)
    >>> ori
    Orientation (193167,) 622
    [[ 0.3027  0.0869 -0.5083  0.8015]
     [ 0.3088  0.0868 -0.5016  0.8034]
     [ 0.3057  0.0818 -0.4995  0.8065]
     ...
     [ 0.4925 -0.1633 -0.668   0.5334]
     [ 0.4946 -0.1592 -0.6696  0.5307]
     [ 0.4946 -0.1592 -0.6696  0.5307]]
    >>> ipf_key = plot.IPFColorKeyTSL(D6)
    >>> fig, ax = plt.subplots()
    >>> _ = ax.imshow(ipf_key.orientation2color(ori).reshape((381, 507, 3)))
    """
    fname = _fetch("ti_orientations/ti_orientations.ctf", allow_download)
    euler = np.loadtxt(fname, skiprows=1, usecols=(0, 1, 2))
    return Orientation.from_euler(np.deg2rad(euler), symmetry.D6)
