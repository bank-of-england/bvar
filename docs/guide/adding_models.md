# Adding New Models

Subclass `SamplingModel` and implement `sample()` and `sample_posterior_state()`.
Pass an instance to `BVAR` to use it for estimation and forecasting.

## Which Methods Do I Need to Write?

| Method | Must implement? | Purpose | Return value |
|---|---|---|---|
| `sample` | Yes | Fit the model to observed data | `SamplingResult`: all parameter draws and point estimates |
| `sample_posterior_state` | Yes | Draw the next posterior parameter state given `Y`, `Z`, and the current state | `PosteriorState`: one new parameter draw |
| `__init__` | Only for model-specific settings | Initialise settings and call `super().__init__(...)` | `None` |
| `sample_innovations` | Only if the Gaussian default is unsuitable | Generate shocks for recursive and unconditional forecasts | Array of shape `(H, n)` |
| `sample_conditional_forecast` | Only if the Gaussian default is unsuitable | Generate a forecast path under constraints | Array of shape `(H*n,)` |
| `predictive_logpdf` | Only if the Gaussian default is unsuitable | Score an out-of-sample observation during cross-validation | A scalar log-density |
| Hyperparameter methods listed below | Only when changing the hyperparameter interface | Set priors and expose parameters to optimisation | Depends on the method |

The two required methods are abstract: Python will not let you instantiate the
subclass until both are implemented. If your model only supports estimation and
unconditional forecasts, still define `sample_posterior_state`, but raise a clear
`NotImplementedError` explaining that conditional posterior updates are unsupported.

For a Gaussian model with the existing hyperparameters, the two required methods
are sufficient. Helpers such as `_posterior` and `_draw_posterior` in the example
below are private implementation choices, not methods that `BVAR` calls.

## Required Contract

### `sample`: Fit the model

Keep the argument names and defaults shown in the example. `BVAR.sample()`
validates and prepares the input data, then calls your method with these inputs:

| Argument | Meaning |
|---|---|
| `data` | NumPy array of shape `(T_total, n)`, including the initial observations needed to construct lags; not a pandas DataFrame |
| `n_lags` | Number of VAR lags |
| `covid_indices` | Integer indices of outlier observations relative to `data`; an empty array means no outlier dummies |
| `vars_in_levels` | Array of shape `(n,)` containing numeric 0/1 indicators for prior construction; `BVAR` currently sets every entry to 0 when `stationary=True`, and to 1 otherwise, rather than inferring individual transformations |
| `N_draws` | Total number of draws your method must return, including any burn-in that `BVAR` will discard |
| `point_only` | If `True`, compute point estimates without posterior simulation; see below |
| `progressbar` | Whether to display sampling progress |
| `soc`, `sur` | Per-fit dummy-prior overrides; `None` uses `self.soc` or `self.sur` |
| `rng` | NumPy random generator to use for every random draw; create `np.random.default_rng()` only if it is `None` |

Your method owns the estimation pipeline:

1. Resolve the effective `soc` and `sur` flags without changing the model's settings.
2. Construct the coefficient and covariance priors using the current hyperparameters.
3. Build the response matrix `Y` and design matrix `Z` with `construct_Y_Z`.
4. If supported and enabled, stack SOC and SUR prior observations with `stack_dummies`, which prepends these rows.
5. Compute point estimates and draw posterior parameters, or run an MCMC chain.
6. Cache the prior information needed by `sample_posterior_state`, then return a `SamplingResult`.

`construct_Y_Z` removes the initial `n_lags` rows from the response, giving
`Y.shape == (T_total - n_lags, n)`. `Z` has the same row count and
`k = 1 + n * n_lags + len(covid_indices)` columns, in this order: intercept,
all variables at lag 1, all variables at lag 2, and so on, then outlier dummies.
SOC and SUR prior observations add rows, not columns. Keep this column order
when constructing and returning coefficients: the forecasting code expects it.

For example, the built-in models cache the prior mean as `self.beta_0`, prior
precision as `self.V_A_inv`, and covariance-prior parameters in `self.pars`.
Your implementation can choose other attributes. Cache priors, not a posterior
that is valid only for the original sample: the update method must use its new
`Y` and `Z`.

Return a `SamplingResult` with these fields. Here `n` is the number of variables,
`k` is the number of regressors per equation, and `nk = n * k`:

| Field | Shape or value | How `BVAR` uses it |
|---|---|---|
| `beta_draws` | `(N_draws, nk)` | Stores retained rows as `bvar.beta`; forecasts use one coefficient draw per path |
| `sigma_draws` | `(N_draws, n**2)` | Stores retained rows as `bvar.sigma`; predictive hooks use the corresponding covariance draw |
| `beta_point` | `(nk,)` | Uses it as `bvar.beta_point` for direct samplers |
| `sigma_point` | `(n**2,)` | Uses it as `bvar.sigma_point` for direct samplers |
| `extras_point` | Auxiliary point state, or `None` | Includes it in `bvar.posterior_state_point`, which seeds conditional forecasts |
| `extras_draws` | One auxiliary payload per draw, or `None` | Stores retained entries as `bvar.extras` and passes them to predictive hooks |

Flatten coefficients equation by equation and covariance matrices row by row.
For a coefficient matrix `A` of shape `(k, n)` in `Y = Z @ A + errors`, return
`A.T.flatten()` as `beta`. Recover it with `beta.reshape(n, k).T`.
For a covariance matrix `Sigma` of shape `(n, n)`, return `Sigma.flatten()`;
recover it with `sigma.reshape(n, n)`. The unconditional forecasting code uses
a Cholesky factorisation, so each covariance must be positive definite even
when you override `sample_innovations`.
`SamplingResult` rejects mismatched draw counts, including `extras_draws`.

Do not discard estimation burn-in inside your method. With `requires_burnin=True`,
`BVAR` requests the retained draw count plus burn-in, discards the initial rows
and auxiliary entries, and computes `beta_point` and `sigma_point` from the
retained draws. It still uses your `extras_point`; choose an auxiliary point
state consistent with how your model uses these coefficient and covariance means.

If you advertise `supports_point_only=True`, handle `point_only=True` without
random parameter draws. The example fills each draw row with the point estimate
so the result keeps the same shapes. Cross-validation refits the model in this
mode. If you cannot support it, set the flag to `False` and reject the request
with a clear error, as `IndependentNIW` does.

### `sample_posterior_state`: Update one posterior state

Use this signature:

```python
def sample_posterior_state(self, Y, Z, current_state, rng=None) -> PosteriorState:
    ...
```

This is a single update, not another full estimation run. It receives:

| Argument | Meaning |
|---|---|
| `Y` | Supplied response matrix of shape `(T, n)`, with prior dummy rows already stacked if applicable |
| `Z` | Supplied design matrix of shape `(T, k)` for the same rows; use this matrix as supplied |
| `current_state` | Previous `PosteriorState`, containing flattened `beta`, flattened `sigma`, and optional `extras` |
| `rng` | Generator for this parameter update |

Here `T` is the number of supplied rows. These matrices need not include a
forecast path; augmentation is how the conditional forecast loop uses this
method, not a requirement of the method itself.

Use the priors cached during `sample()` and recompute the posterior quantities
that depend on `Y` and `Z`. Do not stack SOC or SUR rows again or re-estimate
the priors from the simulated forecast path.

Return a fresh `PosteriorState(beta, sigma, extras)`, with `beta` of shape `(nk,)`
and `sigma` of shape `(n**2,)`, using the same flattening convention as `sample()`.
Direct samplers such as `NaturalConjugate` ignore `current_state`. An MCMC model
uses it for a transition targeting the posterior defined by the supplied `Y`
and `Z`. For example, `IndependentNIW` performs one Gibbs sweep: it starts from
`current_state.sigma` and samples new coefficients before updating the covariance.

The conditional forecast loop uses your method as follows:

1. Copy `bvar.posterior_state_point` to initialise the parameter state.
2. Call `sample_conditional_forecast` to draw a constrained path given that state.
3. Append the path to the observed history and construct new `Y` and `Z`, including enabled prior dummy rows.
4. Call `sample_posterior_state(Y, Z, state, rng=...)` and use the returned state in the next iteration.

The point-only conditional forecast skips the posterior-update step.
Unconditional forecasts use the draws already returned by `sample()` and do not
call `sample_posterior_state`.

### Auxiliary state and copying

Store auxiliary state in `extras` and carry it forward on each update.
See `_ExtrasCarryingModel` in `tests/test_posterior_state.py` for an example.
For instance, a Student-t model could store sampled degrees of freedom there,
return its point value in `extras_point`, and one value per draw in `extras_draws`.

The conditional chain starts from a copy of `BVAR.posterior_state_point`.
Predictive hooks also receive copies. `PosteriorState.copy()` copies both arrays
and deep-copies `extras`; it raises `TypeError` for payloads it cannot copy.
Keep auxiliary payloads copyable and treat hook inputs as read-only.

### Class attributes

| Attribute | Inherited default | Effect when `True` |
|---|---|---|
| `requires_burnin` | `False` | `BVAR` discards the first `N_burn` draws and computes coefficient and covariance point estimates from retained draws |
| `supports_ml` | `False` | Allows the existing conjugate marginal-likelihood hyperparameter optimiser; see the limitation below |
| `supports_point_only` | `True` | Allows point-only fits, required for cross-validation |
| `supports_gaussian_predictive` | `True` | Enables the default Gaussian predictive hooks |
| `supports_girf` | `True` | Allows GIRFs, provided Gaussian prediction is also supported |

The last two flags default to `True`. Set both to `False` for non-Gaussian models:
the current GIRF implementation requires a Gaussian predictive distribution.

## Optional Predictive Hooks

Override these hooks to change the predictive distribution:

| Method | Called by | Return value | Default (Gaussian) behaviour |
|---|---|---|---|
| `sample_innovations` | Recursive and unconditional forecasts, once per parameter draw | `(H, n)` innovations, one row per future period | Draws innovations from `N(0, Sigma)`, where `Sigma = state.sigma.reshape(n, n)`, or zeros for `point_only=True` |
| `sample_conditional_forecast` | Conditional-forecast Gibbs loop, once per iteration | `(H*n,)` forecast values, flattened period by period | Calls `draw_constrained_forecasts`; for `point_only=True`, returns a deterministic path imposing the constraint locations `f` |
| `predictive_logpdf` | Cross-validation scoring, for each scored observation | Scalar log-density, not a density or negative log-density | Evaluates the multivariate normal log-density |

Each hook receives the full `PosteriorState`, including `extras`.
Gaussian models can keep the defaults, which ignore `extras`.
With `supports_gaussian_predictive=False`, the defaults raise
`NotImplementedError`; override every hook your workflow uses.

### Hook signatures and inputs

```python
def sample_innovations(self, state, H, rng=None, point_only=False) -> np.ndarray:
    ...

def sample_conditional_forecast(
    self, state, C, f, Sigma_f, shape_f, last_p_obs, p, n, h, H,
    point_only, constraint_sampler=None, method="andersson_et_al", rng=None,
) -> np.ndarray:
    ...

def predictive_logpdf(self, state, observation, mean, covariance) -> float:
    ...
```

`sample_innovations` returns shocks only. Recursive forecasting adds each row
to the one-step VAR mean; companion-form forecasting propagates the same shocks
through the forecast matrices. `H` is the forecast horizon. Respect `point_only`
so point forecasts do not acquire random shocks.

`sample_conditional_forecast` returns the complete future path, not shocks and
not the observed history. The caller reshapes it with `.reshape(H, n)`. Its
remaining inputs describe the constraints and the start of the forecast:

| Argument | Meaning |
|---|---|
| `C` | Constraint-selection matrix of shape `(nb_constraints, H*n)` |
| `f` | Constraint locations of shape `(nb_constraints,)` |
| `Sigma_f` | Constraint scale matrix of shape `(nb_constraints, nb_constraints)` |
| `shape_f` | Constraint shape parameters of shape `(nb_constraints,)` |
| `last_p_obs` | Observations immediately before the forecast, of shape `(p, n)` |
| `p`, `n`, `h`, `H` | Lag count, variable count, outlier-dummy count, and forecast horizon |
| `point_only` | Request a conditional point forecast rather than a random path |
| `constraint_sampler`, `method` | Optional constraint-distribution sampler and requested conditional-forecast algorithm |
| `rng` | Generator for the forecast draw |

In the default implementation, `point_only=True` uses `f` but ignores `Sigma_f`,
`shape_f`, `constraint_sampler`, and `method`. This path is not generally the
mean of a skewed or custom constraint distribution. A custom hook must define
its own point-forecast behavior.

`predictive_logpdf` receives `observation` and `mean` with shape `(n_targets,)`,
and `covariance` with shape `(n_targets, n_targets)`. These can cover only the
selected target variables rather than all model variables. Evaluate the log
predictive density at the supplied forecast horizon using these inputs and any
required auxiliary state; do not replace `covariance` with `state.sigma`, which
is the innovation covariance rather than the supplied forecast covariance.

Overriding this hook changes the scoring density, not how `BVAR` computes the
supplied moments: the forecasting code still propagates a linear VAR mean and
covariance based on `state.sigma`. If those moments are unsuitable for your
model, a custom log-density alone is not enough to make cross-validation valid.

## Minimal Example

This direct sampler uses a proper conjugate prior with weak coefficient
shrinkage, not an improper flat prior. It omits Minnesota shrinkage and rejects
SOC and SUR dummy observations.
Run this definition and the usage block below in the same Python session.

```python
import numpy as np
from typing import Optional
from scipy.stats import invwishart

from bvar.models import PosteriorState, SamplingModel, SamplingResult
from bvar.utils import construct_Y_Z, get_dimensions


class FlatPrior(SamplingModel):
    """BVAR with weak coefficient shrinkage and a proper covariance prior."""

    requires_burnin: bool = False
    supports_ml: bool = False
    supports_point_only: bool = True

    def __init__(
        self,
        minnesota: bool = False,
        soc: bool = False,
        sur: bool = False,
        covid: bool = False,
        covid_dates: Optional[list] = None,
    ) -> None:
        if soc or sur:
            raise ValueError("FlatPrior does not support soc or sur")
        if minnesota:
            raise ValueError("FlatPrior does not support Minnesota shrinkage")
        super().__init__(
            minnesota=minnesota,
            soc=False,
            sur=False,
            covid=covid,
            covid_dates=covid_dates,
        )

    def sample(
        self,
        data,
        n_lags,
        covid_indices,
        vars_in_levels,
        N_draws,
        point_only=False,
        progressbar=True,
        *,
        soc=None,
        sur=None,
        rng=None,
    ) -> SamplingResult:
        rng = rng if rng is not None else np.random.default_rng()
        if soc or sur:
            raise ValueError("FlatPrior does not support soc or sur")
        _, n_vars, n_regressors, n_coefficients, _ = get_dimensions(
            data, n_lags, covid_indices
        )
        if self.pars.nu_0 is None:
            self.pars.nu_0 = n_vars + 2
        if self.pars.S_0 is None:
            self.pars.S_0 = np.eye(n_vars) * 1e-4
        self.V_A_inv = np.eye(n_regressors) * 1e-6
        Y, Z = construct_Y_Z(data, n_lags, covid_indices)
        posterior = self._posterior(Y, Z)
        mean, _, scale, degrees = posterior
        beta_point = mean.T.flatten()
        sigma_point = (scale / (degrees - n_vars - 1)).flatten()

        beta_draws = np.empty((N_draws, n_coefficients))
        sigma_draws = np.empty((N_draws, n_vars**2))
        if point_only:
            beta_draws[:] = beta_point
            sigma_draws[:] = sigma_point
        else:
            for draw_index in range(N_draws):
                state = self._draw_posterior(posterior, rng)
                beta_draws[draw_index] = state.beta
                sigma_draws[draw_index] = state.sigma

        return SamplingResult(
            beta_draws=beta_draws,
            sigma_draws=sigma_draws,
            beta_point=beta_point,
            sigma_point=sigma_point,
        )

    def sample_posterior_state(self, Y, Z, current_state, rng=None) -> PosteriorState:
        """Draw independently of current_state using the updated data."""
        rng = rng if rng is not None else np.random.default_rng()
        return self._draw_posterior(self._posterior(Y, Z), rng)

    def _posterior(self, Y, Z):
        precision = self.V_A_inv + Z.T @ Z
        chol = np.linalg.cholesky(precision)
        mean = np.linalg.solve(chol.T, np.linalg.solve(chol, Z.T @ Y))
        residuals = Y - Z @ mean
        scale = self.pars.S_0 + residuals.T @ residuals + mean.T @ self.V_A_inv @ mean
        return mean, chol, scale, self.pars.nu_0 + Y.shape[0]

    def _draw_posterior(self, posterior, rng):
        mean, chol, scale, degrees = posterior
        sigma = np.atleast_2d(invwishart.rvs(degrees, scale, random_state=rng))
        noise = rng.normal(size=mean.shape)
        beta = mean + np.linalg.solve(chol.T, noise) @ np.linalg.cholesky(sigma).T
        return PosteriorState(beta=beta.T.flatten(), sigma=sigma.flatten())
```

### Wire it up

Pass `FlatPrior()` directly to `BVAR`; no registration is required.
To expose it as `bv.FlatPrior`, place the class in
`src/bvar/models/flat_prior/model.py`, export it from `models/__init__.py`,
and re-export it from `bvar/__init__.py`. These exports are optional.

Then use it like any other model:

```python
import bvar as bv
import pandas as pd

data = pd.DataFrame(
    np.random.default_rng(42).normal(size=(120, 2)),
    index=pd.period_range("2000Q1", periods=120, freq="Q"),
    columns=["output", "inflation"],
)
model = FlatPrior(soc=False, sur=False)
bvar = bv.BVAR(n_lags=2, model=model, stationary=False, optimisation_method="none")
bvar.sample(data, N_draws=2000, random_state=42, progressbar=False)
bvar.forecast(H=8, random_state=43, progressbar=False)
```

## Customising the Hyperparameter Interface

For additional hyperparameters, override `set_priors`, `_compute_nb_hyper_pars`,
`fill_in_from_vector`, `to_vector`, and, if needed, `hyperparameter_grid`.
`IndependentNIW` shows how to add the cross-variable shrinkage parameter `c2`.

| Method | Purpose and use | Return value |
|---|---|---|
| `set_priors(...)` | Validate and store hyperparameters used by `sample`; the base constructor calls it to initialise defaults, and users can call it to change priors | `None`; updates `self.pars` and any model-specific attributes |
| `_compute_nb_hyper_pars()` | Record the number of parameters exposed to optimisation; the base constructor calls it after `set_priors` | `None`; sets `self.nb_hyper_pars` |
| `fill_in_from_vector(pars)` | Apply a candidate parameter vector during optimisation; validate values before storing them | `None`; updates the same parameters that `sample` reads |
| `to_vector()` | Extract the current searchable parameters for inspection or to construct an explicit `initial_values` argument | One-dimensional NumPy array of length `self.nb_hyper_pars` |
| `hyperparameter_grid()` | Supply the candidate values for cross-validation's grid search | List of one-dimensional arrays, one array per searchable parameter |

The inherited vector order is `[c1, c3, mu?, theta?]`, with `mu` included when
`self.soc=True` and `theta` when `self.sur=True`. All three vector/grid methods
must use the same order, and the grid length must match `self.nb_hyper_pars`.
For example, `IndependentNIW` inserts `c2` after `c3`, giving
`[c1, c3, c2, mu?, theta?]`. Calling `fill_in_from_vector(model.to_vector())`
should leave the model's hyperparameters unchanged.

`BVAR` does not call `to_vector()` automatically. To initialise marginal-likelihood
optimisation from the current parameters, explicitly pass
`initial_values=bvar.model.to_vector()` to `bvar.optimise_hyperparameters(...)`.

If your overridden `set_priors` needs an attribute such as `self.c2`, initialise
that attribute before calling `super().__init__`, because the base constructor
calls the override. Call `super().set_priors(...)` when retaining the shared
hyperparameters. These methods configure priors; they do not build the
data-dependent prior matrices or sample parameters.

Cross-validation calls `hyperparameter_grid`, applies each candidate with
`fill_in_from_vector`, refits with `sample(..., point_only=True)`, and scores
forecasts with `predictive_logpdf`. It applies the best vector at the end.
The optional `cv_options["grid"]` coarsening currently assumes the built-in
parameter names and order; omit it when using a custom vector layout.

### Marginal-likelihood limitation

`supports_ml=True` is not a general extension point for a new marginal-likelihood
formula. `BVAR` currently calls `tune_priors` in the conjugate model's
`marginal_likelihood` module, rather than a method on your model. There is no
required `log_marginal_likelihood` method to implement on `SamplingModel`.
Leave `supports_ml=False` unless your model is compatible with that existing
conjugate calculation. Supporting a different formula requires changes to the
optimisation code beyond adding a model subclass.

For a new model, start with `BVAR(..., optimisation_method="none")` and set
hyperparameters manually. Enable cross-validation only after testing the
point-only fit, parameter-vector interface, and predictive scoring.

## Checklist

- [ ] Implement both abstract methods with compatible signatures.
- [ ] Match the regressor order, flattening conventions, and return shapes.
- [ ] Cache priors during `sample` and use the supplied `Y` and `Z` during posterior updates.
- [ ] Set capability flags to match the methods you support.
- [ ] Preserve auxiliary state and override predictive hooks where needed.
- [ ] Test draws, point estimates, conditional forecasts, and seeded repeatability.
- [ ] Check that unsupported options raise clear errors.

## Reproducibility

Use the supplied `rng` for every random draw. Seeds reproduce results within an
installed release, but sampler changes can alter them between releases.
Pin the package version when you need bit-for-bit reproducibility.
