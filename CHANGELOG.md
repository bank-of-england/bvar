# Changelog

## [0.3.3](https://github.com/bank-of-england/bvar/compare/v0.3.2...v0.3.3) (2026-09-04)


### Documentation

* document GitHub Pages release setup in CONTRIBUTING ([dbc8ba3](https://github.com/bank-of-england/bvar/commit/dbc8ba3b699a864f3a1d231bc5a98f0062bc2fcf))

## [0.3.2](https://github.com/bank-of-england/bvar/compare/v0.3.1...v0.3.2) (2026-09-03)


### Features

* coarsen the cross-validation grid via cv_options["grid"] ([efe4d98](https://github.com/bank-of-england/bvar/commit/efe4d98f3abdcc61af22415e83c2a067fca3f39a))
* expose bvar.version from package metadata ([efe4d98](https://github.com/bank-of-england/bvar/commit/efe4d98f3abdcc61af22415e83c2a067fca3f39a))
* return the matplotlib Figure from plot/diagnostic helpers instead of plt.show() ([efe4d98](https://github.com/bank-of-england/bvar/commit/efe4d98f3abdcc61af22415e83c2a067fca3f39a))
* show a progress bar during cross-validation optimisation ([efe4d98](https://github.com/bank-of-england/bvar/commit/efe4d98f3abdcc61af22415e83c2a067fca3f39a))


### Bug Fixes

* raise RuntimeError when forecast() is called before sample() ([efe4d98](https://github.com/bank-of-england/bvar/commit/efe4d98f3abdcc61af22415e83c2a067fca3f39a))


### Dependencies

* add opera-eco[test] to the dev extra for local ecosystem runs ([efe4d98](https://github.com/bank-of-england/bvar/commit/efe4d98f3abdcc61af22415e83c2a067fca3f39a))


### Documentation

* add Documentation project URL and clarify prior/simulate_var/cumulative_change docstrings ([efe4d98](https://github.com/bank-of-england/bvar/commit/efe4d98f3abdcc61af22415e83c2a067fca3f39a))

## 0.3.1

First public release.

### Highlights

- Bayesian vector autoregressions with natural-conjugate and independent-NIW models.
- Minnesota, sum-of-coefficients, and single-unit-root prior support.
- Unconditional and conditional forecasting, including soft and skewed constraints.
- Generalised impulse response functions and forecast counterfactual analysis.
- Cross-validation and nowcasting-prior support.
