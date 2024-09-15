How to make a new release of ``orix``
=====================================

After version 0.9.0, orix's branching model changed to one similar to the Gitflow
Workflow (`original blog post
<https://nvie.com/posts/a-successful-git-branching-model/>`__).

orix versioning adheres to `Semantic Versioning
<https://semver.org/spec/v2.0.0.html>`__.
See the `Python Enhancement Proposal (PEP) 440 <https://peps.python.org/pep-0440/>`__
for supported version identifiers.

Preparation
-----------
- Locally, create a minor release branch from the ``develop`` branch when making a minor
  release, or create a patch release branch from the ``main`` branch when making a patch
  release. Ideally, a patch release is published immediately after a bug fix is merged
  in ``main``. Therefore, it might be best to do the release updates directly on the bug
  fix branch, so that no separate patch release branch has to be made.

- Run tutorial notebooks examples in the documentation locally and confirm that they
  produce expected results.

- Review the contributors ``__credits__`` in ``orix/__init__.py`` to ensure everyone is
  included and sorted correctly. Use the same ordering in the ``.zenodo.json`` file.
  Take care to format the Zenodo metadata file correctly (e.g. no trailing commas).

- Increment ``__version__`` in ``orix/__init__.py``, e.g. from "0.9.0" to "0.9.1" for a
  patch release. If downstream packages should test their use of the next version of
  orix in CI before it is released, or we want to ensure that the below release steps
  work as expected, a release candidate with version e.g. "0.9.1rc1" can be made. Update
  ``CHANGELOG.rst`` accordingly.

- Make a PR of the release branch to ``main``. Discuss the release and changelog with
  others. Merge.

Tag and release
---------------
- If ``__version__`` in ``orix/__init__.py`` on ``main`` has changed in a new commit, a
  tagged annotated release *draft* is automatically created. If ``__version__`` is now
  "0.9.1", the release name is "orix 0.9.1" and the tag name is "v0.9.1". The tag target
  will be ``main``. The release body contains a static description and links to the
  changelog in the documentation and GitHub. The release draft can be published as is,
  or changes to the release body can be made before publishing. Publish.

- Monitor the publish workflow to ensure the release is successfully published to PyPI.

Post-release action
-------------------
- Monitor the `documentation build <https://readthedocs.org/projects/orix/builds>`__ to
  ensure that the new stable documentation is successfully built from the release.

- Ensure that `Zenodo <https://doi.org/10.5281/zenodo.3459662>`__ displays the new
  release.

- Ensure that Binder can run the tutorial notebooks by clicking the Binder badges in the
  top banner of one of the tutorials via the `documentation
  <https://orix.readthedocs.io/en/stable>`__.

- Bring changes in ``main`` into ``develop`` by first branching from ``develop``, merge
  ``main`` into the new branch and fix potential conflicts. After these conflicts are
  fixed, update or revert ``__version__`` and make any updates to this guide if
  necessary. Make a PR to ``develop`` and merge.

- A PR to the conda-forge feedstock will be created by the conda-forge bot. Follow the
  relevant instructions from the conda-forge documentation on updating packages, as well
  as the instructions in the PR. Merge after checks pass. Monitor the Azure pipeline CI
  to ensure the release is successfully published to conda-forge.

- Tidy up GitHub issues and close the corresponding milestone.
