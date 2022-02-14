How to make a new release of ``orix``
=====================================

Create a PR to the `master` branch and go through the following steps.

Preparation
-----------
- Bump ``__version__`` in `orix/__init__.py`, for example "0.8.2".
- Update the changelog `CHANGELOG.rst`.
- Let the PR collect comments for a day to ensure that other maintainers are 
  comfortable with releasing. Merge.

Release (and tag)
-----------------
- Create a tagged, annotated (meaning with a release text) with the name 
  v0.8.2" and title "orix 0.8.2". The tag target will be the ``master`` branch.
  Draw inspiration from previous release texts. Publish the release.
- Monitor the publish GitHub Action to ensure the release is successfully 
  published to PyPI.

Post-release action
-------------------
- Monitor the documentation build at
  https://readthedocs.org/projects/orix/builds to make sure the new stable
  documentation is successfully built from the release.
- Make a post-release PR to ``master`` with ``__version__`` updated (or 
  reverted), e.g. to "0.9.dev0", and any updates to this guide if necessary.
- Tidy up and close corresponding milestone.
- A PR to the conda-forge feedstock will be created by the conda-forge bot.
