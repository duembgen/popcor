# Change Log

All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased] -- 2025-08-22

(0.0.0a)=
### Added
- RotationLifter: support to plot 2d frames, improved documentation for rotation conventions
- RotationLifter: implemented rank-d ("bm") lifting
- RangeOnlyLifter: new landmark sampling methods to fill available space
- Documentation instructions in CONTRIBUTING.md
- Unit tests and notebook for RotationLifter
- StateLifter: started support for rank-d instead of rank-1 formulations

(0.0.0c)=
### Changed
- RangeOnlyLifter: default sampling for landmarks in RO is now the one filling space
- RangeOnlyLifter: theta is sampled at least MIN_DIST away from landmarks

(0.0.0a)=
### Fixed
- Link fixes in documentation

## [0.0.2] - 2025-06-04

(0.0.2a)=
### Added

- RangeOnlyLifter base class for 2D/3D range-only localization (base_lifters/range_only_lifter.py)
- RangeOnlyNsqLifter: specialized lifter for range-only localization without squared distances (examples/ro_nsq_lifter.py)
- RangeOnlySqLifter: specialized lifter for squared-distance-based range-only localization
- GitHub Pages deployment step to documentation.yml using peaceiris/actions-gh-pages for automatic documentation publishing

(0.0.2r)=
### Removed

- docs/build/ files

(0.0.2f)=
### Fixed

- Corrected return type annotations in get_grad and get_hess methods in state_lifter.py from float to np.ndarray
- Documentation of MonoLifter and WahbaLifter

## [0.0.1] - 2025-05-25

This is the initial release of the toolbox. It is based on <https://github.com/utiasASRL/constraint_learning> and now included there as a submodule. 
