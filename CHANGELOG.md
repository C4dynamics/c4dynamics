# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]


## [2.4.0] - 2026-09-03
### Added
- navigation sensors: new `c4dynamics.sensors.navigation` module with `gps`,
  `imu`, and `magnetometer` models, each carrying its device error model
  (noise, bias, scale factor) plus `measure()` and `demo()` methods.
  `gps` gains an `isideal` flag to zero out noise/bias.
- quadcopter model and controller: new `c4dynamics.models` module with a
  predefined `quad` (Iris) plant, and new `c4dynamics.controllers` module
  with a cascade-PID quadcopter controller and its config.
- EKF use case: `ekf_estimation` example program - extended Kalman filter
  state estimation for a quadcopter tracking a figure-8 trajectory,
  including GPS-dropout experiments and an architecture diagram.
- mountain_car.py: predefined environment for the RL mountain car problem.

### Changed
- kalman/ekf: opt-in `P_jitter` covariance stabilization (symmetrize and
  floor `P` after each predict/update).
- kalman/ekf `update()`: optional chi-squared NIS `gate` to reject
  outlier measurements, plus `innov`/`hx` arguments for non-subtractive
  or nonlinear residuals. `update()` now returns `None` when the update
  is gated out or `S` is numerically singular, instead of raising.
- expanded navigation sensors documentation (concepts, API reference, and
  runnable examples under `examples/sensors`).


## [2.3.7] - 2026-07-22
### Added
- an envirorments file for abstracted plant models in learning controller.
  includes a predefined helicopter class for the learning controller example. 


## [2.3.6] - 2026-03-03
### Changed
- add all use-case notebooks support for google colab.


## [2.3.5] - 2026-02-25
### Changed
- update package setup to support conda installation.


## [2.3.5] - 2026-02-04
### Changed
ruff&black code style updated to fully comply with PEP-8.


## [2.3.4] - 2026-02-02
### Changed
- separate dependencies to 3 sets: basic, vision, dev.
- fix a bug of a single numpy item assignment into the state vector.


## [2.3.3] - 2026-01-17
### Changed
- fix bug in addvars(): adjust state vector legnth with new vars.
- change P() -> dist() and V() -> vel_mag() to enable these names for user variables.


## [2.3.2] - 2025-12-02
### Changed
- supersedes 2.3.1 which was tagged without the 'v' prefix.


## [2.3.1] - 2025-12-02
### Changed
- Fix the compatible Python versions to include 3.12 (<3.13).


## [2.3.0] - 2025-12-01
### Changed
- This update replaces the behavior from the previous release, where a dtype mismatch triggered a warning.
- The State object now uses a fixed internal floating-point type (np.float64) regardless of input types during initialization or assignment.
- No warning is issued for assignments with a different type .

### Breaking
- Users can no longer preserve custom numeric dtypes inside the State object.
- All state values are stored as float64 internally.
- Code that relied on modifying the dtype of the internal state vector will no longer work.


## [2.2.0] - 2025-11-15
### Breaking
- Fix: Core `State` object assignment behavior corrected — assigning a vector
	with a mismatched `dtype` to a `State` field now updates the underlying
	backing array reliably instead of silently failing to modify internal memory.
	- Impact: Code that relied on implicit (and previously unreliable) assignments
		may now see the array updated as expected or receive a clear `TypeWarning`
		when incompatible types are provided.
	- Migration: Ensure assigned arrays match the state's dtype or use the
		provided setter helper. Example migration patterns:
		```py

		# Problematic (old behavior may have silently failed)
		s = c4d.state(x = 0, y = 0, z = 0)
		state.X = np.array([1, 2, 3], dtype = np.float64)

		# Recommended (explicit dtype match)
		s = c4d.state(x = 0., y = 0., z = 0.) 	# initialize explicitly as float
		state.X = np.asarray([1, 2, 3], dtype=state.X.dtype)

		```
	- Rationale: Prevents silent state corruption and makes assignment
		semantics deterministic across NumPy versions and platforms.


## [2.0.0] - 2024-11-11
### Added
- Complete state space objects mechanism.
- Seeker and radar measurements modules.
- Kalman filter and Extended Kalman filter implementations.
- YOLOv3 object detection API.
- Functionality to fetch datasets for running examples.
- Comprehensive documentation updates.


## [1.2.0] - 2024-03-05
### Added
- `animate` function to visualize rigid body Euler angles with a model.
- Utilities: `cprint`, `gen_gif`, `plottools`, and `tictoc`.


## [1.1.0] - 2024-02-17
### Added
- New class: `fdataframe` for frame's datapoints, inheriting from `datapoint`.
- Documentation for the respective modifications.

### Changed
- Converted `norms` from a function to a property.
- Updated class architecture to include `__slots__` to limit new variable declarations.

### Fixed
- Bug in `cprint()` function.


## [1.0.0] - 2023-07-19
### Added
- Initial release of C4dynamics framework.
- Core functionalities for dynamic systems algorithm development.
- Modules for state space representations, sensors, detectors, and filters.
- Example programs and basic documentation.


