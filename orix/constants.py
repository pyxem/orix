"""Constants and such useful across modules."""

from importlib.metadata import version

# NB! Update project config file if this list is updated!
optional_deps: list[str] = ["numpy-quaternion"]
installed: dict[str, bool] = {}
for pkg in optional_deps:
    try:
        _ = version(pkg)
        installed[pkg] = True
    except ImportError:
        installed[pkg] = False

# Tolerances
eps9 = 1e-9
eps12 = 1e-12

del optional_deps
