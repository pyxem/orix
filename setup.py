from itertools import chain

from setuptools import find_packages, setup

from orix import __author__, __author_email__, __description__, __name__, __version__

# Projects with optional features for building the documentation and running
# tests. From setuptools:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies
# fmt: off
extra_feature_requirements = {
    "doc": [
        "ipykernel",  # Used by nbsphinx to execute notebooks
        "memory_profiler",
        "nbconvert                      >= 7.16.4",
        "nbsphinx                       >= 0.7",
        "numpydoc",
        "pydata-sphinx-theme",
        "scikit-image",
        "scikit-learn",
        "sphinx                         >= 3.0.2",
        "sphinx-codeautolink[ipython]",
        "sphinx-copybutton              >= 0.2.5",
        "sphinx-design",
        "sphinx-gallery",
        "sphinxcontrib-bibtex           >= 1.0",
    ],
    "tests": [
        "coverage                       >= 5.0",
        "numpydoc",
        "pytest                         >= 5.4",
        "pytest-cov                     >= 2.8.1",
        "pytest-rerunfailures",
        "pytest-xdist",
    ],
}
extra_feature_requirements["dev"] = [
    "black[jupyter]",
    "isort                              >= 5.10",
    "manifix",
    "outdated",
    "pre-commit                         >= 1.16",
] + list(chain(*list(extra_feature_requirements.values())))
# fmt: on

# Remove the "raw" ReStructuredText directive from the README so we can
# use it as the long_description on PyPI
readme = open("README.rst").read()
readme_split = readme.split("\n")
for i, line in enumerate(readme_split):
    if line == ".. EXCLUDE":
        break
long_description = "\n".join(readme_split[i + 2 :])

setup(
    name=__name__,
    version=str(__version__),
    license="GPLv3",
    url="https://orix.readthedocs.io",
    author=__author__,
    author_email=__author_email__,
    description=__description__,
    long_description=long_description,
    long_description_content_type="text/x-rst",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        (
            "License :: OSI Approved :: GNU General Public License v3 or later "
            "(GPLv3+)"
        ),
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    packages=find_packages(exclude=["orix/tests"]),
    extras_require=extra_feature_requirements,
    # fmt: off
    install_requires=[
        "dask[array]",
        "diffpy.structure       >= 3.0.2",
        "h5py",
        "matplotlib             >= 3.5",
        "matplotlib-scalebar",
        "numba",
        "numpy",
        "numpy-quaternion",
        "pooch                  >= 0.13",
        "scipy",
        "tqdm",
    ],
    # fmt: on
    package_data={"": ["LICENSE", "README.rst", "readthedocs.yaml"], "orix": ["*.py"]},
)
