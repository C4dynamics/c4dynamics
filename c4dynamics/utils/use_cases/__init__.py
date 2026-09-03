#
# use_cases
##
# Supporting modules for the documentation use-case examples
# (https://c4dynamics.github.io/c4dynamics/ -> Use Cases).
#
# These modules hold the example-specific scaffolding - main loops, plot
# generators, reference configs - that keeps the notebooks focused on the
# problem rather than the boilerplate. They ship with the package so the
# notebooks can `from c4dynamics.utils.use_cases import <module>` instead of
# downloading the file at runtime, but they are NOT part of the public API
# and may change with the example they belong to.
#
#   dof6_modules      - pn_guidance / dof6sim: control, engine, aerodynamics
#   quad_ekf          - ekf_estimation / quad_ekf: EKF loop, sweeps, plots
#   ekf_config        - ekf_estimation / quad_ekf: reference EKF noise block
#   iris_quadcopter   - ekf_estimation: 3DR Iris parameter datasheet
#
# Submodules are imported explicitly (not here) - several pull scipy and
# other heavy dependencies that the rest of c4dynamics.utils does not need.
