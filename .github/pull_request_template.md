<!--
Copyright 2018-2025 the orix developers

This file is part of orix.

orix is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

orix is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with orix. If not, see <http://www.gnu.org/licenses/>.
-->
#### Description of the change


#### Progress of the PR
- [ ] [Docstrings for all functions](https://numpydoc.readthedocs.io/en/latest/example.html)
- [ ] Unit tests with pytest for all lines
- [ ] Clean code style by [running black via pre-commit](https://orix.readthedocs.io/en/latest/dev/code_style.html)

#### Minimal example of the bug fix or new feature
```python
>>> from orix import vector
>>> v = vector.Vector3d([1, 1, 1])
>>> # Your new feature...
```

#### For reviewers
<!-- Don't remove the checklist below. -->
- [ ] The PR title is short, concise, and will make sense 1 year later.
- [ ] New functions are imported in corresponding `__init__.py`.
- [ ] New features, API changes, and deprecations are mentioned in the unreleased
      section in `CHANGELOG.rst`.
- [ ] Contributor(s) are listed correctly in `__credits__` in `orix/__init__.py` and in
      `.zenodo.json`.