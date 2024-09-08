"""Constants and such useful across modules."""

from importlib.metadata import version
from pathlib import Path
import tomllib

# Dependencies
with open(Path(__file__).parent.parent / "pyproject.toml", "rb") as f:
    d = tomllib.load(f)
    deps = d["project"]["dependencies"]
    optional_deps = d["project"]["optional-dependencies"]["all"]
    deps += optional_deps

installed = {}
for pkg in deps:
    try:
        _ = version(pkg)
        installed[pkg] = True
    except ImportError:
        installed[pkg] = False

del version, Path, tomllib, d, deps, optional_deps
