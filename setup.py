from setuptools import setup, find_packages
from orix import __name__, __version__, __author__, __author_email__, __description__


setup(
    name=__name__,
    version=str(__version__),
    license="GPLv3",
    author=__author__,
    author_email=__author_email__,
    description=__description__,
    long_description=open("README.rst").read(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    packages=find_packages(exclude=["orix/tests"]),
    # fmt: off
    install_requires=[
        "diffpy.structure >= 3",
        "h5py",
        "matplotlib",
        "numpy",
        "scipy",
        "tqdm",
    ],
    # fmt: on
    package_data={"": ["LICENSE", "readme.rst"], "orix": ["*.py"],},
)
