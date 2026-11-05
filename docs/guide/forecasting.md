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

All snippets on this page continue from:

```python
import bvar as bv, numpy as np

data, _, _, _ = bv.simulate_var(T=200, n=4, n_lags=2, levels=True, seed=42)
data.columns = ["gdp", "cpi", "unemp", "rrate"]
bvar = bv.BVAR(n_lags=2, model=bv.NaturalConjugate(), stationary=False, random_state=42)
bvar.optimise_hyperparameters(data)
bvar.sample(data, N_draws=5000)
```

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

The package supports hard and distributional constraints through
`SamplingModel.sample_conditional_forecast`. Its default implementation uses
the shock-adjustment algorithm of Andersson, Palmqvist & Waggoner (2010,
Algorithm 1), selected by `method="andersson_et_al"`. A model with
`supports_gaussian_predictive=False` must override this hook for its predictive
distribution.

### Parameter Uncertainty: Gibbs Sampling

Conditional forecasting resamples the VAR parameters using the simulated
conditional path. This follows the MCMC sampler in Miranda-Agrippino & Ricco (2019),
Section 3, Algorithm 2 ([author manuscript, pp. 8-9](https://silviamirandaagrippino.com/s/BVARs_Applications.pdf#page=8)),
which they attribute to Waggoner & Zha. The method was introduced by Waggoner &
Zha (1999). The future values are latent variables: they are not observed and
are sampled jointly with the parameters. The restrictions fix some of them (or
linear combinations of them), so those enter the parameter update like observed
data. The remaining values are redrawn each iteration given the current
parameters, so the chain averages over them. Each iteration:

1. Draws future shocks from their distribution conditional on the restrictions (Eq. 10), given the current parameters, and builds the forecast path (Algorithm 2, step 1).
2. Appends that path to the historical sample and draws new VAR parameters from their full conditional posteriors, using the chosen model and priors (Algorithm 2, step 2).
3. Keeps the path and discards the parameters (Algorithm 2, step 3). The new parameters are used in the next iteration; paths before burn-in are dropped.

With $Y$ denoting historical data and $\theta=(A,\Sigma)$, the two Gibbs blocks are

$$y_F^{(s)}\sim p(y_F\mid Y,\theta^{(s-1)},Cy_F=f).$$

$$\theta^{(s)}\sim p(\theta\mid Y,y_F^{(s)}).$$

**Step 1 and Eq. (10).** Miranda-Agrippino & Ricco write the restrictions as
$Gu=g$ (Eq. 9), with $G=RC_0$ and $g=r-RE(y_F\mid Y,\theta)$, and draw the
reduced-form shocks from

$$u\mid(Gu=g)\sim\mathcal{N}\!\left(\Sigma_HG'(G\Sigma_HG')^{-1}g,\;
\Sigma_H-\Sigma_HG'(G\Sigma_HG')^{-1}G\Sigma_H\right).\quad\text{(Eq. 10)}$$

The package works with structural shocks $\varepsilon\sim\mathcal{N}(0,I)$,
so $\Sigma_H=I$, $G=D=CM^T$ and $g=f-Cb$. The default shock-adjustment draw
$\varepsilon=\xi+D^+(g-D\xi)$, with $\xi\sim\mathcal{N}(0,I)$, has mean
$D^+g$ and covariance $I-D^+D$. When $D$ has full row rank,
$D^+=D'(DD')^{-1}$, so this matches Eq. (10) exactly for hard restrictions.

The rationale is to sample the joint distribution of parameters and future
observations conditional on the scenario. Miranda-Agrippino & Ricco (p. 9)
describe step 2 as ensuring "consistency of the posterior distribution with
the restriction in Eq. (9)": if the restricted path occurs, it carries
information about the parameters as well as the future shocks. The simulated
path is redrawn each iteration; it is not permanently added as observed data.

This treats the conditioning path as information about the future, which suits
conditioning on nowcasts, market expectations or published data. For a purely
hypothetical policy path, letting the path update the parameters is a modelling
choice rather than a requirement.

Algorithm 2 covers hard restrictions. For soft or skewed constraints, the
package uses the same loop but draws the constrained values from the supplied
distribution in step 1. That extension is ours; it is not part of Algorithm 2.

Other differences from Algorithm 2:

- `NaturalConjugate` draws $(A,\Sigma)$ jointly from the augmented-data
  posterior; `IndependentNIW` draws $A\mid\Sigma$ and then $\Sigma\mid A$.
  Algorithm 2 draws $\Sigma$ first. Neither choice changes the target distribution.
- The chain starts from the fitted point estimate. It runs `N_draws`
  iterations, capped at the number of stored posterior draws, and drops the
  first `N_burn`.
- With sum-of-coefficients or single-unit-root priors, the dummy observations
  for variables not in levels use the sample mean of $Y$. That sample includes
  the simulated path, so the prior changes from one iteration to the next.
  Algorithm 2 keeps the prior fixed.

This differs from drawing parameters once from the historical-data posterior
and adjusting only future shocks, which leaves parameter uncertainty unchanged
by the scenario. Two implementations can therefore agree on unconditional
forecasts yet differ on conditional forecasts if only one performs this Gibbs
update. Establishing agreement with a particular MATLAB implementation requires
checking its parameter-update step, not just its forecast recursion.

With `point_only=True`, the package instead computes one conditional path at
the fitted point parameters without the Gibbs update. The default
`method="andersson_et_al"` selects the constrained-path draw inside the loop,
not whether parameters are resampled.

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

Soft constraints allow the constrained variables to deviate from the target path subject to a specified degree of uncertainty (`constraint_variance`). This is appropriate when the conditioning information comes from a forecast or nowcast that is itself uncertain. In the default algorithm, each iteration draws a target $z\sim\mathcal{N}(f,\Sigma_f)$ and imposes $Cy=z$ exactly. Thus the supplied distribution specifies the marginal distribution of the constrained path; it is not a noisy-observation likelihood whose target is combined with the VAR forecast through a measurement-error update (Andersson, Palmqvist & Waggoner, 2010; Antolín-Díaz et al., 2021, Section 2.2).

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

Passing a fixed `random_state` (or `rng`) to `forecast`/`recursive_forecast` gives results that are reproducible within a single installed release of the package. This is **not** guaranteed across releases: a change to a model's posterior sampler or innovation implementation — such as the consolidation of posterior updates around `sample_posterior_state` before the first public release — can change the sequence of draws consumed from a given seed even though the underlying distributions are unchanged. Pin the package version if bit-for-bit reproducibility across environments or over time is required.

## References

| Paper | Contribution |
|---|---|
| [Miranda-Agrippino & Ricco (2019), *Bayesian Vector Autoregressions: Applications*](https://doi.org/10.1093/acrefore/9780190625979.013.478), *Oxford Research Encyclopedia of Economics and Finance* | Conditional-path data augmentation and parameter resampling (Section 3, Algorithm 2; author manuscript, p. 9) |
| [Waggoner & Zha (1999), *Conditional forecasts in dynamic multivariate models*](https://www.jstor.org/stable/2646713), *Review of Economics and Statistics*, 81(4), 639-651 | Hard conditional forecasting and parameter uncertainty |
| [Andersson, Palmqvist & Waggoner (2010), *Density-conditional forecasts in dynamic multivariate models*](https://www.econstor.eu/handle/10419/83026), Sveriges Riksbank Working Paper 243 | Default shock-adjustment algorithm (Algorithm 1) and distributional constraints |
| [Antolín-Díaz, Petrella & Rubio-Ramírez (2021), *Structural scenario analysis with SVARs*](https://doi.org/10.1016/j.jmoneco.2020.06.001), *Journal of Monetary Economics*, 117, 798-815 | Distributional constraints (Section 2.2) and conditional-forecast Gibbs sampling |
