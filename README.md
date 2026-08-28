# Bayesian Vector Autoregressions for forecasting

User manual: see the [documentation](docs/index.md).

## Features
* Natural-conjugate setting (Normal-Inverse-Wishart) with Minnesota shrinkage.
* Sum-of-coefficients and single-unit-root dummy observation priors (implemented with dummy observations).
* Hyperparameter optimisation following GLP (2015) but without the MH step.
* Covid dummies.
* Conditional and unconditional forecasting.
* Soft constraints in conditional forecasting, with uncertainty around the constraints.
* Generalised impulse response functions.
* Conditional forecast counterfactual analysis.
* Hyperparameter optimisation via cross-validation; see [this doc](docs/methods/cross_validation.md).
    * Out-of-sample predictive likelihood.
    * Features: optimise hyperparameter to maximise the predictive likelihood at a given horizon and for a subset of targeted series.
* Support for skewed constraints (aka upside and downside risks); see [this doc](docs/methods/skewed_constraints.md).
* Explicitly accounts for nowcasting uncertainty when exploiting nowcasts; see [this doc](docs/methods/nowcasts_priors.md).
* Unit tests and simulation experiments, including:
    * Simulation comparing the estimated parameters with the true data generating process.
    * Simulation evaluating forecast unbiasedness.
    * Simulation checking the moments of the constrained forecast distribution.

## Project Structure

    ├── src/                        # Source code
    ├── docs/                       # Documentation and notebooks
    ├── tests/                      # Automated tests
    ├── ...
    
## Installation
```bash
pip install bvar
```

## Selected documentation
* [General framework](docs/methods/bvar_framework.md)

## Selected notebooks
* [Illustration with simulated data](docs/notebooks/simulated_data.md)

## Contributing
* See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to the code.
* Have a question or an idea? Please open an issue.

## Main references
* General model specification, priors and marginal likelihood:
    * [GLP (2015)](https://doi.org/10.1162/REST_a_00483)
* Implementation (matrix formulation and sampling algorithm):
    * [Chan (2020)](https://link.springer.com/10.1007/978-3-030-31150-6_4)
* Covid dummies
    * [Cascaldi-Garcia (2022)](https://www.federalreserve.gov/econres/ifdp/files/ifdp1352.pdf)
* Conditional forecasting
    * [Waggoner and Zha (1999)](https://www.jstor.org/stable/2646713) - Hard constraints
    * [Antolín-Díaz et al. (2021)](https://doi.org/10.1016/j.jmoneco.2020.06.001) - Soft constraints
* Generalised IRFs
    * [Pesaran and Shin (1998)](https://doi.org/10.1016/S0165-1765(97)00214-0).

## Data Classification
Bank of England Data Classification: OFFICIAL BLUE