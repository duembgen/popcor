# Change Log

All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased] -- YYYY-MM-DD

(00-changelog)=
### Added

(01-changelog)=
### Changed

(02-changelog)=
### Fixed

## [0.0.2] - 2025-06-04


(6-changelog)=
### Added

- RangeOnlyLifter base class for 2D/3D range-only localization (base_lifters/range_only_lifter.py)
- RangeOnlyNsqLifter: specialized lifter for range-only localization without squared distances (examples/ro_nsq_lifter.py)
- RangeOnlySqLifter: specialized lifter for squared-distance-based range-only localization
- GitHub Pages deployment step to documentation.yml using peaceiris/actions-gh-pages for automatic documentation publishing

(5-changelog)=
### Removed

- docs/build/ files

(4-changelog)=
### Fixed

- Corrected return type annotations in get_grad and get_hess methods in state_lifter.py from float to np.ndarray

## [0.0.1] - 2025-05-25

This is the initial release of the toolbox. It is based on <https://github.com/utiasASRL/constraint_learning> and now included there as a submodule. 
