#### Description of the change


#### Progress of the PR
- [ ] [Docstrings for all functions](https://github.com/numpy/numpy/blob/master/doc/example.py)
- [ ] Unit tests with pytest for all lines
- [ ] Clean code style by [running black via pre-commit](https://orix.readthedocs.io/en/latest/contributing.html#code-style)

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