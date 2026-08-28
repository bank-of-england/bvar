# BVAR toolkit for macroeconomic forecasting

A versatile package for Bayesian Vector Autoregressions (BVARs). It supports macroeconomic forecasting with Bayesian shrinkage, marginal likelihood, cross-validation, and conditional forecasting.

---

## Features

- **Natural conjugate** (Normal-Inverse-Wishart) setting with Minnesota shrinkage.
- **Sum-of-coefficients** and **single-unit-root** priors via dummy observations.
- **Hyperparameter optimisation** — marginal likelihood (GLP, 2015) or cross-validation.
- **COVID-19 dummies** following Cascaldi-Garcia (2022).
- **Conditional & unconditional forecasting** with hard, soft, and skewed constraints.
- **Generalised Impulse Response Functions** (Pesaran & Shin, 1998).
- **Forecast revision analysis** for counterfactual comparisons.
- **Nowcasting uncertainty** — treats nowcasts as soft constraints.

## Quick Start

```python
import bvar as bv

# Simulate some data
data, _, _, _ = bv.simulate_var(T=200, n=3, n_lags=2, levels=True, seed=42)

# Set up the sampling model and BVAR
model = bv.NaturalConjugate(minnesota=True, soc=True, sur=True)
bvar = bv.BVAR(n_lags=2, model=model, stationary=False)

# Optimise, estimate, and forecast
bvar.optimise_hyperparameters(data)
bvar.sample(data, N_draws=5000)
bvar.forecast(H=8)
bvar.plot_forecast()
```

## Installation

```bash
pip install bvar
```

## Project Layout

```
src/bvar/           # Source code
docs/               # Documentation, notebooks & this site
tests/              # Unit and integration tests
```

## References

| Topic | Paper |
|---|---|
| Model & priors | [Giannone, Lenza & Primiceri (2015)](https://doi.org/10.1162/REST_a_00483) |
| Implementation | [Chan (2020)](https://link.springer.com/10.1007/978-3-030-31150-6_4) |
| COVID dummies | [Cascaldi-Garcia (2022)](https://www.federalreserve.gov/econres/ifdp/files/ifdp1352.pdf) |
| Hard constraints | [Waggoner & Zha (1999)](https://www.jstor.org/stable/2646713) |
| Soft constraints | [Antolín-Díaz et al. (2021)](https://doi.org/10.1016/j.jmoneco.2020.06.001) |
| GIRFs | [Pesaran & Shin (1998)](https://doi.org/10.1016/S0165-1765(97)00214-0) |
