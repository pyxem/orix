from setuptools import setup, find_packages
from texpy import __version__, __author__, __author_email__, __description__


setup(
    name="texpy",
    version=str(__version__),
    license="MIT",
    author=__author__,
    author_email=__author_email__,
    description=__description__,
    packages=find_packages(exclude=['texpy/tests']),
    install_requires=[
        "numpy",
        "matplotlib",
    ]
)