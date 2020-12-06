# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added 
- Overloaded division for Vector3d (left hand side) by numbers and suitably shaped array-like objects.

### Changed
- Continuous integration migrated from Travis CI to GitHub Actions.

## 2020-11-03 - version 0.5.1
### Added
- This project now keeps a Changelog
- Testing for Py3.8 on OSX

### Fixed
- CrystalMap properties allow arrays with number of dimensions greater than 2
- ANG file reader now recognises phase IDs defined in the header
- EMsoft file reader reads unrefined Euler angles correctly
