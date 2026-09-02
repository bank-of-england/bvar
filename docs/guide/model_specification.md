# Model Specification

## The VAR Model

A Vector Autoregression (VAR) of order $p$ describes $n$ time series jointly as a function of their own lags:

$$y_t = c + A_1 y_{t-1} + \cdots + A_p y_{t-p} + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \Sigma)$$

where $y_t$ is an $n$-dimensional vector of observables at time $t$, $c$ is an intercept, each $A_i$ is an $(n \times n)$ coefficient matrix, and $\Sigma$ is the covariance of the one-step-ahead forecast errors.

## Bayesian Estimation

With $n$ variables and $p$ lags, a VAR has $n(np + 1)$ unrestricted coefficients. Estimating these by OLS is feasible but produces poor out-of-sample forecasts because a typical macroeconomic sample identifies the parameters poorly. The Bayesian approach addresses this problem by incorporating prior beliefs that shrink the coefficients towards a parsimonious benchmark, reducing estimation uncertainty and improving forecast performance.

The package uses a **Normal-Inverse-Wishart (NIW)** prior, the natural conjugate prior for the VAR likelihood. This conjugacy gives a closed-form posterior and removes the need for Markov Chain Monte Carlo.

## Prior Structure

### Minnesota Prior

The Minnesota prior, introduced by Litterman (1986) and refined by Giannone, Lenza & Primiceri (2015), encodes three beliefs:

- **Own lags matter most.** Each variable is expected to be close to a random walk: the prior mean on the first own lag is 1 (or 0 for stationary series), and all other lag coefficients are centred at zero.
- **More recent lags matter more.** Prior variance shrinks as lag order increases, reflecting the intuition that distant lags carry less predictive power.
- **Own lags are more informative than cross-variable lags.** Cross-variable coefficients are shrunk more aggressively than own-lag coefficients.

The degree of shrinkage is controlled by a scalar hyperparameter $\lambda$: a small $\lambda$ pulls all coefficients strongly towards the prior; a large $\lambda$ lets the data exert more influence.

### Sum-of-Coefficients Prior

The sum-of-coefficients prior (Doan, Litterman & Sims, 1984) encodes the belief that slowly moving variables behave like individual unit-root processes. Operationally, it is implemented as an additional set of dummy observations appended to the data matrix. The prior is controlled by a hyperparameter $\mu$.

### Single-Unit-Root Prior

The single-unit-root prior (Sims & Zha, 1998) encourages a common stochastic trend across all variables — the system as a whole is close to a unit root. Like the sum-of-coefficients prior, it is implemented via dummy observations controlled by a hyperparameter $\theta$.

### COVID-19 Dummies

The COVID-19 pandemic introduced extreme outliers in 2020 Q1–Q2 that can distort both prior and likelihood. Following Cascaldi-Garcia (2022), the package supports additive date-specific dummies that absorb these observations and protect the estimated covariance structure.

## Hyperparameter Optimisation

The user can set the shrinkage hyperparameters $(\lambda, \mu, \theta)$ or select them by maximising the **marginal likelihood**, the probability of the data integrated over all parameter values. Giannone, Lenza & Primiceri (2015) show that this optimisation produces forecasts that match or outperform benchmark models across many macroeconomic datasets.

As an alternative to the marginal likelihood, hyperparameters can also be selected by **cross-validation** — searching for the values that maximise out-of-sample predictive accuracy at a target horizon and for a target set of variables. See the [Cross-Validation](../methods/cross_validation.md) page for details.

## Data Transformations

The package works with the data as supplied. The user is responsible for applying transformations (e.g. log-levels, differences) before passing data to `BVAR`. The `data_transformation` argument used in sampling and forecasting tells the model how to recover original-scale forecasts from the transformed series.

```python
import numpy as np
import bvar as bv

# Apply log to real variables, keep rates in levels
data_transformation = {"gdp": "logs", "cpi": "logs", "rrate": "levels"}

data = data.apply(
    lambda col: (
        np.log(col) if data_transformation.get(col.name, "levels") == "logs" else col
    )
)
```

Then specify the sampling model and build the BVAR:

```python
model = bv.NaturalConjugate(
    minnesota=True,  # Minnesota shrinkage
    soc=True,  # sum-of-coefficients
    sur=True,  # single-unit-root
    covid=True,  # COVID-19 dummies
)

bvar = bv.BVAR(n_lags=4, model=model, stationary=False)
```

## Data Validation and Requirements

Before fitting, `BVAR` validates the input `data` and enforces the following rules:

- **Regular time index.** The index must be a `pd.DatetimeIndex` or `pd.PeriodIndex` with a regular frequency. The package accepts any frequency that `pandas` can infer, including monthly and weekly data. A `DatetimeIndex` without an inferable frequency raises `ValueError`.
- **No missing values.** `data` must not contain NaNs; a `ValueError` identifies the first offending row and column.
- **At least two variables.** `data` must have at least two columns, since the model is a multivariate VAR.
- **Numeric columns only.** All columns must be numeric dtypes.

### Transformation Vocabulary

The `data_transformation` argument passed to `sample()` records the state of each input series after the user has transformed it. Supported input states are `"levels"`, `"logs"` (or `"log_levels"`), `"diff"`, and `"log_diff"`. It does not itself transform the data, and `qoq` or `yoy` are not valid input-state labels. The label is stored for use when `forecast()` reconstructs output from the model-space series.

For `"diff"` and `"log_diff"`, the fitted data contain differences rather than an absolute level, so `forecast()` cannot infer the level needed to reconstruct forecast rows. Pass `base_value` explicitly whenever the forecast includes rows: it may be a scalar applied to all variables or an array-like value with one base per variable. The package retains the observed history in its transformed form and reconstructs only the forecast tail. Omitting `base_value` raises `ValueError`.

In `forecast(transformations=...)`, `None` is the no-op output transformation, while `qoq` and `yoy` request quarter-on-quarter or year-on-year rates. These output values are separate from the input-state labels above and belong in `transformations`, not `data_transformation`. With `qoq` or `yoy` output for differenced data, historical rows remain differenced and the forecast tail contains rates computed against reconstructed actual levels, including the explicit last observed base.

### Retained vs Total Draws

`N_draws` passed to `sample()` and `forecast()` is the number of draws **retained** after any burn-in, not the total number of iterations run. For MCMC-based sampling (`IndependentNIW`, and the Waggoner & Zha/Andersson et al. conditional forecast samplers), an additional `N_burn` draws are generated and discarded before the retained draws are collected; `N_burn` defaults to `N_draws // 2` when not specified.

### Hard vs Soft Constraints

In conditional forecasting, `constraint_variance` values of exactly zero for a constrained cell create a **hard** constraint: the corresponding forecast equals `constraint_mean` with no uncertainty. A strictly positive `constraint_variance` creates a **soft** constraint: the constrained variable centres on `constraint_mean` and retains the specified variance. `NaN` cells in `constraint_mean` remain unrestricted.

### Stationary Flag and Dummy-Observation Priors

`stationary` is a single system-wide boolean, not a per-variable setting. `stationary=False` treats every series as I(1) (random-walk prior mean of 1 on the first own lag); `stationary=True` treats every series as I(0) (prior mean 0) and disables `soc`/`sur` for that fit even if requested on the model, because sum-of-coefficients / single-unit-root dummies are only meaningful for levels data.

For mixed I(1)/I(0) data, difference the I(1) series yourself, pass `stationary=True`, and record the transform in `data_transformation`; or keep everything in levels with `stationary=False` and accept the random-walk prior on the stationary series.

## References

| Paper | Contribution |
|---|---|
| [Giannone, Lenza & Primiceri (2015)](https://doi.org/10.1162/REST_a_00483) | Minnesota prior, marginal likelihood optimisation |
| [Chan (2020)](https://link.springer.com/10.1007/978-3-030-31150-6_4) | Matrix formulation and sampling algorithm |
| [Cascaldi-Garcia (2022)](https://www.federalreserve.gov/econres/ifdp/files/ifdp1352.pdf) | COVID-19 dummy observations |