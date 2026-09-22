# Adding New Models

Subclass `SamplingModel` and implement `sample()` and `sample_posterior_state()`.
Pass an instance to `BVAR` to use it for estimation and forecasting.

## Required Contract

Keep the argument names and defaults shown in the example. `BVAR` passes these
inputs to `sample()`:

| Arguments | Purpose |
|---|---|
| `data`, `n_lags` | Observation array and lag order |
| `covid_indices`, `vars_in_levels` | Outlier-dummy locations and level-variable flags |
| `N_draws`, `point_only`, `progressbar` | Draw count, point-estimate mode, progress display |
| `soc`, `sur` | Per-fit dummy-prior overrides; `None` uses the model setting |
| `rng` | NumPy random generator |

The posterior-update method receives `Y`, `Z`, `current_state`, and `rng`.

`sample()` owns prior construction and estimation. Cache the state needed to
update the posterior when conditional forecasting supplies new `Y` and `Z`.
Return a `SamplingResult` with these fields, where `n` is the number of variables,
`k` the regressors per equation, and `nk = n * k`:

| Field | Shape or value |
|---|---|
| `beta_draws` | `(N_draws, nk)` |
| `sigma_draws` | `(N_draws, n**2)` |
| `beta_point` | `(nk,)` |
| `sigma_point` | `(n**2,)` |
| `extras_point` | Auxiliary point state, or `None` |
| `extras_draws` | One auxiliary payload per draw, or `None` |

Flatten coefficients equation by equation and covariance matrices row by row.
`SamplingResult` rejects mismatched draw counts, including `extras_draws`.

### Posterior state

`sample_posterior_state()` returns a fresh `PosteriorState(beta, sigma, extras)`.
Direct samplers such as `NaturalConjugate` ignore `current_state`; MCMC samplers
use it for one Gibbs sweep. `IndependentNIW` starts from `current_state.sigma`
and samples new coefficients before updating the covariance.

Store auxiliary state in `extras` and carry it forward on each update.
See `_ExtrasCarryingModel` in `tests/test_posterior_state.py` for an example.

The conditional chain starts from a copy of `BVAR.posterior_state_point`.
Predictive hooks also receive copies. `PosteriorState.copy()` copies both arrays
and deep-copies `extras`; it raises `TypeError` for payloads it cannot copy.
Keep auxiliary payloads copyable and treat hook inputs as read-only.

### Class attributes

| Attribute | Effect when `True` |
|---|---|
| `requires_burnin` | `BVAR` discards the first `N_burn` draws |
| `supports_ml` | Allows marginal-likelihood hyperparameter optimisation |
| `supports_point_only` | Allows point-only fits, required for cross-validation |
| `supports_gaussian_predictive` | Enables the default Gaussian predictive hooks |
| `supports_girf` | Allows GIRFs, provided Gaussian prediction is also supported |

The last two flags default to `True`. Set both to `False` for non-Gaussian models:
the current GIRF implementation requires a Gaussian predictive distribution.

## Optional Predictive Hooks

Override these hooks to change the predictive distribution:

| Method | Called by | Default (Gaussian) behaviour |
|---|---|---|
| `sample_innovations` | Recursive and unconditional forecasts | Draws innovations from `N(0, state.sigma)` |
| `sample_conditional_forecast` | Conditional-forecast Gibbs loop | Calls `draw_constrained_forecasts` |
| `predictive_logpdf` | Cross-validation scoring | Evaluates the multivariate normal log-density |

Each hook receives the full `PosteriorState`, including `extras`.
Gaussian models can keep the defaults, which ignore `extras`.
With `supports_gaussian_predictive=False`, the defaults raise
`NotImplementedError`; override every hook your workflow uses.

## Minimal Example

This direct sampler uses a diffuse, proper conjugate prior, not an improper flat
prior. It omits Minnesota shrinkage and rejects SOC and SUR dummy observations.
Run this definition and the usage block below in the same Python session.

```python
import numpy as np
from typing import Optional
from scipy.stats import invwishart

from bvar.models import PosteriorState, SamplingModel, SamplingResult
from bvar.utils import construct_Y_Z, get_dimensions


class FlatPrior(SamplingModel):
    """BVAR with a near-flat but proper Bayesian prior."""

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

## Checklist

- [ ] Preserve auxiliary state and override predictive hooks where needed.
- [ ] Test draws, point estimates, conditional forecasts, and seeded repeatability.
- [ ] Check that unsupported options raise clear errors.

## Reproducibility

Use the supplied `rng` for every random draw. Seeds reproduce results within an
installed release, but sampler changes can alter them between releases.
Pin the package version when you need bit-for-bit reproducibility.
