"""Constants and such useful across modules."""

from importlib.metadata import version

# NB! Update project config file if this list is updated!
optional_deps = ["numpy-quaternion"]
installed = {}
for pkg in optional_deps:
    try:
        _ = version(pkg)
        installed[pkg] = True
    except ImportError:
        installed[pkg] = False

del optional_deps
