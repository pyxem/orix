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

Some datasets must be downloaded from the web.
"""

import os
from pathlib import Path

# import numpy as np
import pooch as ppooch

from orix import __version__, io
from orix.data._registry import registry, registry_urls

# from orix.quaternion import Orientation, symmetry


__all__ = [
    "sdss_austenite",
    "sdss_ferrite_austenite",
    #    "ti_orientations",
]


fetcher = ppooch.create(
    path=ppooch.os_cache("orix"),
    base_url="",
    version=__version__.replace(".dev", "+"),
    env="ORIX_DATA_DIR",
    registry=registry,
    urls=registry_urls,
)
cache_data_path = fetcher.path.joinpath("data")
package_data_path = Path(os.path.abspath(os.path.dirname(__file__)))


def _has_hash(path, expected_hash):
    """Check if the provided path has the expected hash."""
    if not os.path.exists(path):
        return False
    else:
        return ppooch.file_hash(path) == expected_hash  # pragma: no cover


def _cautious_downloader(url, output_file, pooch):
    if pooch.allow_download:
        delattr(pooch, "allow_download")
        # HTTPDownloader() requires tqdm
        download = ppooch.HTTPDownloader(progressbar=True)
        download(url, output_file, pooch)
    else:
        raise ValueError(
            "The dataset must be (re)downloaded from the orix-data repository on GitHub"
            " (https://github.com/pyxem/orix-data) to your local cache with the pooch "
            "Python package. Pass `allow_download=True` to allow this download."
        )


def _fetch(filename, allow_download):
    resolved_path = os.path.join(package_data_path, "..", filename)
    expected_hash = registry[filename]
    if _has_hash(resolved_path, expected_hash):  # File already in data module
        return resolved_path  # pragma: no cover
    else:  # Pooch must download the data to the local cache
        fetcher.allow_download = allow_download  # Extremely ugly
        resolved_path = fetcher.fetch(filename, downloader=_cautious_downloader)
    return resolved_path


def sdss_austenite(allow_download=False, **kwargs):
    """Single phase Austenite crystallographic map of shape (100, 117)
    produced from dictionary indexing of electron backscatter
    diffraction patterns of a super duplex stainless steel (SDSS) sample
    containing both Austenite and Ferrite grains.

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
    """
    fname = _fetch("data/sdss/sdss_austenite_dp.h5", allow_download)
    return io.load(fname, **kwargs)


def sdss_ferrite_austenite(allow_download=False, **kwargs):
    """Dual phase Austenite and Ferrite crystallographic map of shape
    (100, 117) produced from dictionary indexing of electron backscatter
    diffraction patterns of a super duplex stainless steel (SDSS)
    sample.

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
    """
    fname = _fetch("data/sdss/sdss_ferrite_austenite.ang", allow_download)
    xmap = io.load(fname, **kwargs)
    xmap.phases["austenite/austenite"].name = "austenite"
    xmap.phases["ferrite/ferrite"].name = "ferrite"
    return xmap


# def ti_orientations(allow_download=False):
#    """Orientations in the MTEX orientation convention (crystal2lab)
#    from an orientation map produced from Hough indexing of electron
#    backscatter diffraction patterns of a commercially pure hexagonal
#    close packed titanium sample (*6/mmm*) :cite:`johnstone2020density`.
#
#    Parameters
#    ----------
#    allow_download : bool, optional
#        Whether to allow downloading the dataset from the internet to
#        the local cache with the pooch Python package. Default is False.
#
#    Returns
#    -------
#    Orientations
#    """
#    fname = _fetch("data/ti_orientations/ti_orientations.ctf", allow_download)
#    euler = np.loadtxt(fname, skiprows=1, usecols=(0, 1, 2))
#    return Orientation.from_euler(np.deg2rad(euler), symmetry.D6)
