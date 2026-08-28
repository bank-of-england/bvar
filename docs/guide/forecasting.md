# Forecasting

## Overview

The package supports two types of forecasts:

- **Unconditional forecasts** — the model propagates uncertainty forward without restrictions on any variable.
- **Conditional forecasts** — one or more variables are constrained to follow a specified path over the forecast horizon, and the remaining variables adjust accordingly.

The package computes forecasts from the posterior predictive distribution. It propagates each posterior draw of $(A, \Sigma)$ over $H$ periods and combines the resulting trajectories.

Every current model draws Gaussian reduced-form innovations. A model can replace
this behaviour through the `SamplingModel` hooks `sample_innovations`,
`sample_conditional_forecast`, and `predictive_logpdf`, which control innovation
drawing, constrained-forecast sampling, and out-of-sample density evaluation.
See [Adding New Models](adding_models.md#optional-predictive-hooks) for the full contract.

## Unconditional Forecasts

An unconditional $H$-step forecast iterates the VAR forward from the last observed data point:

$$\hat{y}_{T+h} = \hat{c} + \hat{A}_1 \hat{y}_{T+h-1} + \cdots + \hat{A}_p \hat{y}_{T+h-p} + \varepsilon_{T+h}$$

for $h = 1, \ldots, H$, where parameter uncertainty is integrated over by averaging across posterior draws and shock uncertainty is introduced by drawing $\varepsilon_{T+h} \sim \mathcal{N}(0, \Sigma^{(s)})$ for each draw $s$ — the default behaviour supplied by `sample_innovations` for every current model.

```python
# Produce an unconditional 8-quarter forecast
transformations = {var: "qoq" for var in bvar.df_data.columns}
transformations["unemp"] = None  # keep unemployment in levels

bvar.forecast(H=8, transformations=transformations)
bvar.plot_forecast(var_names=["gdp", "cpi", "unemp"])
```

## Conditional Forecasts

Conditional forecasts impose restrictions on a subset of variables over the forecast horizon. Pass these restrictions through `constraint_mean`, a matrix of shape $(H \times n)$ in which `NaN` entries leave variables unrestricted and numeric entries set their values.

The package implements two conditioning approaches through
`SamplingModel.sample_conditional_forecast`. Its default implementation runs
the Gaussian constraint algorithms. A model with
`supports_gaussian_predictive=False` must override this hook for its predictive
distribution.

### Hard Constraints (Waggoner & Zha, 1999)

Hard constraints force the constrained variables to hit the specified path exactly. Conceptually, the algorithm projects the unconditional forecast distribution onto the subspace defined by the restrictions, yielding a conditional distribution for the unrestricted variables.

```python
import numpy as np

H = 8
n_vars = len(bvar.df_data.columns)

# Build a constraint matrix: NaN = unconstrained
constraint_mean = np.full((H, n_vars), np.nan)

# Pin Bank Rate (column index 0) to a flat path
bankrate_col = bvar.df_data.columns.get_loc("rrate")
constraint_mean[:, bankrate_col] = 5.0

bvar.forecast(
    H=H,
    constraint_mean=constraint_mean,
    transformations=transformations,
)
bvar.plot_forecast(alpha=0.1, from_date="2024Q1")
```

### Soft Constraints (Antolín-Díaz et al., 2021)

Soft constraints allow the constrained variables to deviate from the target path subject to a specified degree of uncertainty (`constraint_variance`). This is appropriate when the conditioning information comes from a forecast or nowcast that is itself uncertain. The model treats the constraint as an observation with measurement error, and the unrestricted variables adjust proportionally to their historical co-movement with the constrained ones.

```python
# Add uncertainty around the Bank Rate path: ±0.25pp one standard deviation
constraint_variance = np.full((H, n_vars), np.nan)
constraint_variance[:, bankrate_col] = 0.25**2  # variance, not standard deviation

bvar.forecast(
    H=H,
    constraint_mean=constraint_mean,
    constraint_variance=constraint_variance,
    transformations=transformations,
)
```

### Skewed Constraints

The package also supports skewed constraints — asymmetric restrictions that allow the user to impose upside or downside risk around a central path. See the [Skewed Constraints](../methods/skewed_constraints.md) page for a detailed treatment.

## Nowcasting Uncertainty

When the latest quarter remains unpublished, condition on a nowcast from a
nowcasting model. Treat the nowcast as a soft constraint to propagate its
uncertainty into the forecast. See [Nowcasting Uncertainty](../methods/nowcasts_priors.md).

## Transformations

The `transformations` argument tells the package how to convert model-space forecasts (which operate on the transformed series, e.g. log-levels) back to presentation-ready units. Supported transformations include:

- `qoq` — quarter-on-quarter growth rate
- `yoy` — year-on-year growth rate
- `None` — no-op output transformation

`levels` is a `data_transformation` state, not an output transformation. Use `transformations={"variable": None}` to keep a variable in levels when forecasting.

When `data_transformation` is `"diff"` or `"log_diff"`, pass `base_value` to `forecast()` to identify the absolute level at the end of the observed sample. The package needs this value to reconstruct forecast rows because the stored history contains differences. `base_value` may be a scalar or one value per variable. The package preserves the transformed history and reconstructs only the forecast tail; omitting the base raises `ValueError`.

For `qoq` or `yoy` output from `"diff"` or `"log_diff"` data, the returned array keeps the same history-plus-tail shape: historical rows remain in their stored differenced form, while forecast-tail rates are calculated from reconstructed actual levels, including the supplied last observed base.

## Visualisation

`plot_forecast` plots fan charts for the posterior predictive distribution across variables, with the history overlaid. The `alpha` argument controls the coverage of the shaded bands (e.g. `alpha=0.1` for 10–90% bands), and `from_date` allows the plot to be windowed to a recent period.

## Related Model Capabilities

Two other framework features rely on the same predictive hooks as forecasting:

- **Generalised impulse responses.** [GIRFs](girfs.md) are currently Gaussian-only: `GIRF.compute_girf` requires both `supports_girf=True` and `supports_gaussian_predictive=True` on the model, and raises `NotImplementedError` otherwise.
- **Predictive-likelihood hyperparameter optimisation.** [Cross-validation](../methods/cross_validation.md) scores each rolling-window out-of-sample point via `SamplingModel.predictive_logpdf`, which defaults to a Gaussian log density and must be overridden by non-Gaussian models.

## Reproducibility

Passing a fixed `random_state` (or `rng`) to `forecast`/`recursive_forecast` gives results that are reproducible within a single installed release of the package. This is **not** guaranteed across releases: a change to a model's posterior sampler or innovation implementation — such as the consolidation of posterior updates around `sample_posterior_state` in 0.3.0 — can change the sequence of draws consumed from a given seed even though the underlying distributions are unchanged. Pin the package version if bit-for-bit reproducibility across environments or over time is required.

## References

| Paper | Contribution |
|---|---|
| [Waggoner & Zha (1999)](https://www.jstor.org/stable/2646713) | Hard conditional forecasting |
| [Antolín-Díaz et al. (2021)](https://doi.org/10.1016/j.jmoneco.2020.06.001) | Soft conditional forecasting |
