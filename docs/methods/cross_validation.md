# Cross-Validation for Hyperparameter Selection in BVAR Models

## Introduction

In classical settings with regularisation (e.g. Ridge, LASSO), cross-validation is a popular method for selecting penalty parameters. Cross-validation selects parameters that minimise **out-of-sample** errors at a chosen forecast horizon.

[Penalties in regularised regressions closely relate to prior variances in Bayesian models](https://statisticaloddsandends.wordpress.com/2018/12/29/bayesian-interpretation-of-ridge-regression/). This relationship supports cross-validation for selecting Bayesian VAR shrinkage parameters. Unlike [marginal likelihood maximisation](bvar_framework.md#hyperparameter-optimisation), which targets in-sample one-step-ahead prediction, cross-validation directly optimises out-of-sample performance at the horizon of interest.

## Expanding Window with Predictive Likelihood

The implemented approach uses an **expanding window** strategy, common in real-time forecasting exercises:

1. Estimate the model on a training set (data up to time $t$)
2. Evaluate on a test observation at $t + p + H$ (where $p$ is the number of lags and $H$ is the forecast horizon)
3. Expand the training set by one observation and repeat

This process continues until the end of the sample. The optimisation criterion is the **log predictive likelihood** evaluated at each test observation:

$$\log p(y_{t+p+H} \mid y_{1:t+p}, \lambda) = \log \mathcal{N}(y_{t+p+H}; \hat{\mu}_{H}, \hat{\Sigma}_{H})$$

where $\hat{\mu}_{H}$ and $\hat{\Sigma}_{H}$ are the posterior mean and variance of the $H$-step-ahead forecast, computed from the model estimated on $y_{1:t+p}$.

The objective is to maximise the sum of log predictive likelihoods across all rolling windows:

$$\Lambda(\lambda) = \sum_{t=t_0}^{T-p-H} \log p(y_{t+p+H} \mid y_{1:t+p}, \lambda)$$

## Hyperparameter Selection

Cross-validation can optimise:

- **Minnesota prior parameters**: $c_1$ (overall tightness), $c_3$ (lag decay)
- **SOC prior parameter**: $\mu$ (sum-of-coefficients tightness) — if `priors.soc=True`
- **SUR prior parameter**: $\theta$ (single-unit-root tightness) — if `priors.sur=True`

A grid search covers the parameter space. The default grid sizes are:
- $c_1 \in [0.001, 1.0]$ (20 points)
- $c_3 \in [1.0, 5.0]$ (5 points)
- $\mu, \theta \in [0.01, 100]$ (11 points, log-spaced)

## Extension to Targeted Series

By default, the method averages predictive likelihood across all BVAR variables. Specify `target_series` in `optimise_hyperparameters()` to target a **specific series**, such as GDP. The resulting model optimises forecast accuracy for those variables at the chosen horizon.

## Usage Example

```python
import bvar as bv

# Load or simulate data
data, _, _, _ = bv.simulate_var(T=200, n=3, n_lags=2, levels=True, seed=42)

# Use NaturalConjugate with SOC and SUR (both will be tuned by the grid search)
model = bv.NaturalConjugate(minnesota=True, soc=True, sur=True)

# Use cross-validation instead of marginal likelihood
bvar = bv.BVAR(
    n_lags=2, model=model, stationary=False, optimisation_method="cross_validation"
)

# Optimise hyperparameters via expanding-window predictive likelihood
# targeting GDP (column 0) at an 8-quarter horizon
bvar.optimise_hyperparameters(
    data,
    target_series=[data.columns[0]],
    cv_options={"H": 8, "oos_test_window_size": 20},
)

# Estimate and forecast with the optimised hyperparameters
bvar.sample(data, N_draws=5000)
bvar.forecast(H=8)
bvar.plot_forecast()
```
