# Estimation

## First forecast in ten lines

```python
import bvar as bv

data, _, _, _ = bv.simulate_var(T=200, n=3, n_lags=2, levels=True, seed=42)

model = bv.NaturalConjugate(minnesota=True, soc=True, sur=True)
bvar = bv.BVAR(n_lags=2, model=model, stationary=False, random_state=42)

bvar.optimise_hyperparameters(data)
bvar.sample(data, N_draws=5000)
bvar.forecast(H=8)
bvar.plot_forecast()
```

This is the canonical minimal workflow; the sections below expand on each step.

## Workflow

Estimation proceeds in two steps:

1. **Optimise hyperparameters** — find the shrinkage parameters that maximise the marginal likelihood (or a cross-validation criterion).
2. **Sample from the posterior** — draw from the model-specific posterior to obtain a Monte Carlo representation of the joint distribution of all model parameters.

## Hyperparameter Optimisation

The method `optimise_hyperparameters` searches for the values of the Minnesota hyperparameters $(\lambda, \mu, \theta)$ that maximise the log marginal likelihood. The search is carried out by gradient-based numerical optimisation over a bounded parameter space.

The argument `nb_restart` controls how many random restarts are attempted. With `nb_restart=0`, a single optimisation is run from a default initial point. With `nb_restart > 0`, additional runs are started from random points; the best solution found across all runs is kept.

Call `sample` directly with specified hyperparameters to skip optimisation.

```python
# Run optimisation with 5 random restarts for robustness
bvar.optimise_hyperparameters(data=outturns, nb_restart=5)
```

## Input Requirements

Estimation accepts a `pandas.DataFrame` with at least two numeric variables and
a unique, increasing, regular `PeriodIndex` or `DatetimeIndex`. Datetime
indexes are converted to periods at their inferred frequency, including the
calendar anchor for anchored frequencies such as `Q-MAR`. All observations
must be finite: `NaN`, positive infinity, and negative infinity are rejected
before optimisation or sampling begins. The dataset must also contain at
least one observation after the requested lag window.

## Posterior Sampling

The method `sample` draws $N$ samples from the joint posterior of the VAR coefficients $(A, \Sigma)$. For `NaturalConjugate`, the draws are independent because the posterior is available in closed form. For `IndependentNIW`, the draws come from a Gibbs chain and may be correlated.

Because the Normal-Inverse-Wishart prior is conjugate to the VAR likelihood, the posteriors are available in closed form:

$$\Sigma \mid Y \sim \mathcal{IW}(S_\text{post},\; \nu_\text{post})$$

$$A \mid \Sigma, Y \sim \mathcal{MN}(A_\text{post},\; K_A^{-1},\; \Sigma)$$

Each draw uses two steps: draw $\Sigma$ from its marginal inverse-Wishart distribution, then draw $A$ conditional on $\Sigma$ from a matrix-Normal distribution. `NaturalConjugate` samples the exact posterior without MCMC or burn-in. `IndependentNIW` uses a Gibbs sampler because its coefficients and covariance lack a joint closed-form posterior: the sampler draws $A$ and $\Sigma$ in turn from their conditional posteriors, then discards the initial `N_burn` draws.

For models with dummy-observation priors (sum-of-coefficients or single-unit-root), the package appends dummy rows to the data matrix before the posterior update. The same sampling step then applies the prior.

`beta_point` and `sigma_point` store the posterior point estimates. For
`IndependentNIW`, they are the posterior means of the retained Gibbs draws
after `N_burn` has been discarded, and `posterior_state_point` wraps those
same arrays. `NaturalConjugate` retains its analytical point estimates and
does not use burn-in.

```python
# Draw 10,000 posterior samples
# data_transformation records how the input data were transformed before
# fitting (e.g. logs, differences); it is stored on the model and later
# used to convert forecasts/GIRFs back to the variables' original scale.
bvar.sample(
    data=outturns,
    N_draws=10_000,
    data_transformation=data_transformation,
)
```

## Dummy Observations

When `soc=True` or `sur=True` is specified in the model (e.g. `NaturalConjugate`), the package constructs synthetic observation rows consistent with the chosen prior beliefs and appends them to the data matrix. This is mathematically equivalent to the corresponding prior distributions under the NIW framework — the posterior update is unchanged.

COVID-19 dummies, when enabled, add indicator variables for the pandemic quarters. These dummies absorb the extreme observations so that the estimated covariance matrix is not distorted by the crisis.

## Stationary vs Non-Stationary

The `stationary` flag in `BVAR` affects the prior mean on own-lag coefficients. When `stationary=False`, the first own-lag coefficient is centred at 1 (random-walk prior). When `stationary=True`, it is centred at 0. For most macroeconomic variables in levels, `stationary=False` is appropriate.

## Posterior Diagnostics

After estimation, `compute_fitted_values` recovers the in-sample fitted values averaged over posterior draws. `plot_fitted_values` plots the data and fitted values side by side, providing a quick visual check on model fit.

```python
bvar.compute_fitted_values()
bvar.plot_fitted_values()
```

## References

| Paper | Contribution |
|---|---|
| [Giannone, Lenza & Primiceri (2015)](https://doi.org/10.1162/REST_a_00483) | Marginal likelihood and hyperparameter optimisation |
| [Chan (2020)](https://link.springer.com/10.1007/978-3-030-31150-6_4) | Efficient posterior sampling algorithm |
