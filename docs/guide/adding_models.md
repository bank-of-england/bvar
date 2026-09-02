# Adding New Models

All estimation models in the package are subclasses of `SamplingModel`, an abstract base class defined in `src/bvar/models/base.py`. Adding a new model means implementing two abstract methods and, optionally, overriding the hyperparameter interface and the predictive hooks. Once a model satisfies the abstract contract below, `BVAR` and conditional forecasting pick it up automatically; cross-validation additionally requires `supports_point_only=True`, and the current GIRF implementation further requires both `supports_girf=True` and `supports_gaussian_predictive=True`.

## Required Contract

Any new model must implement:

```python
def sample(
    self,
    data: np.ndarray,
    n_lags: int,
    covid_indices: np.ndarray,
    vars_in_levels: np.ndarray,
    N_draws: int,
    point_only: bool = False,
    progressbar: bool = True,
    soc: Optional[bool] = None,
    sur: Optional[bool] = None,
    rng: Optional[np.random.Generator] = None,
) -> SamplingResult:
    """Run the full estimation pipeline; cache whatever internal state
    ``sample_posterior_state`` will need."""


def sample_posterior_state(
    self,
    Y: np.ndarray,
    Z: np.ndarray,
    current_state: PosteriorState,
    rng: Optional[np.random.Generator] = None,
) -> PosteriorState:
    """Return one posterior draw (an analytic draw for direct samplers,
    one Gibbs sweep for MCMC samplers) as a ``PosteriorState``, using the
    fit state and posterior machinery cached during ``sample()`` — used by
    the conditional-forecast sampler. This is the sole posterior-update
    extension point."""
```

``current_state.beta``/``current_state.sigma`` carry the flattened coefficients and
covariance from the previous draw in the chain. Direct samplers (e.g.
`NaturalConjugate`) draw independently of the current state and ignore
`current_state`; MCMC samplers use
`current_state.sigma` (and, if relevant, `current_state.beta`) as the
starting point for the next Gibbs sweep.

**Current implementation status:** `NaturalConjugate` ignores
`current_state` entirely by design, as its posterior draws are
independent of any chain state. `IndependentNIW.sample_posterior_state` seeds
each Gibbs sweep's covariance from `current_state.sigma` (reshaped
to `(n, n)`) via a shared `_gibbs_step` kernel, also used by the batch
`_sample_gibbs` chain during `sample()`. `current_state.beta` is accepted for
interface generality but not used by this Gibbs kernel, since β is
freshly sampled conditional on Σ within each sweep.

### The `PosteriorState` carrier

Conditional forecasting (`Forecasting._conditional_forecast`) runs its Gibbs
chain through a `PosteriorState` dataclass
(`beta`, `sigma`, `extras`) rather than raw `(beta, sigma)` tuples, so
models that carry auxiliary state between draws, such as a sampled degrees of
freedom parameter for a Student-*t* model, can store it there.
`SamplingModel.sample_posterior_state(Y, Z, current_state, rng)`
is the abstract method the conditional-forecast loop calls each
iteration; every concrete model must implement it and return a fresh
`PosteriorState`. Models that only need `beta`/`sigma` (both current
models) return `extras=None`. A model that owns `extras` should update and
forward its own payload — see `_ExtrasCarryingModel` in
`tests/test_posterior_state.py` for a minimal example.

`BVAR.posterior_state_point` provides a `PosteriorState` view of `beta_point`/
`sigma_point` (plus `extras_point`, `None` for both current models) used to
seed the conditional-forecast chain. `_conditional_forecast` copies it
before iterating via `PosteriorState.copy()`, whose default policy is to
isolate the seed *completely* from the fitted point-estimate state: `beta`/`sigma`
are array-copied and `extras` is **deep**-copied, so nested mutable
containers inside `extras` (e.g. a `dict` holding a `list`) cannot leak
mutations back into `BVAR.posterior_state_point`. If a model's `extras`
payload genuinely cannot be deep-copied (e.g. it holds a live resource
such as a lock or file handle), `copy()` raises `TypeError` rather than
silently aliasing it; such a model should override
`sample_posterior_state` to manage that payload's isolation itself.

Every call into a predictive hook (`sample_innovations`,
`sample_conditional_forecast`, `predictive_logpdf`) is likewise given a
fresh `state.copy()`, never a state object aliasing `beta_point`/
`sigma_point`/`posterior_state_point` or a live `beta_draws`/`sigma_draws`
row. In particular, `GridSearch.marginal_likelihood_H` passes
`bvar.posterior_state_point.copy()` into `predictive_logpdf` for each
rolling-window out-of-sample point, so a mutating override can never
corrupt the fitted point-estimate state used by the next grid point. A model
override should therefore treat the `state` it receives as read-only
logical input: even a hook that mutates `state.beta`/`state.sigma`/
`state.extras` in place cannot leak that mutation back into the fitted
posterior arrays or into the state threaded through the next Gibbs
iteration.


### Class attributes

Five boolean class attributes control framework behaviour:

```python
requires_burnin: bool  # True → BVAR discards the first N_burn draws
supports_ml: bool  # True → BVAR.optimise_hyperparameters() allows "ml"
supports_point_only: bool  # True → the model can be sampled with point_only=True
# (required for optimisation_method="cross_validation")
supports_gaussian_predictive: bool  # True → the model's predictive distribution is the
# Gaussian used by the default sample_innovations,
# sample_conditional_forecast and predictive_logpdf
supports_girf: bool  # True → GIRF.compute_girf may be used with this model
# (also requires supports_gaussian_predictive=True; see below)
```

`supports_gaussian_predictive` and `supports_girf` both default to `True`, matching every model currently in the package (all of which are Gaussian). A non-Gaussian model — e.g. one with a Student-*t* predictive distribution — must set `supports_gaussian_predictive = False` and override whichever of the predictive hooks below its forecasting workflow needs; see [Optional Predictive Hooks](#optional-predictive-hooks).

The **only framework requirement** is that after `sample()` completes, `sample_posterior_state(Y, Z, current_state, rng)` must be able to return a valid `PosteriorState` draw. What that means in practice is entirely up to the model:

- `NaturalConjugate` stores `self.beta_0` (prior mean vector) and `self.V_A_inv` (per-equation precision matrix).
- `IndependentNIW` stores the same names but with different shapes (`(nk, nk)` full-system precision).
- A model based on a Student-*t* prior or a hierarchical prior might cache something completely different.

The interface only dictates the **inputs and outputs**; the internal state is an implementation detail.



`sample()` must return a `SamplingResult` dataclass:

```python
@dataclass
class SamplingResult:
    beta_draws: np.ndarray  # shape (N_draws, nk)
    sigma_draws: np.ndarray  # shape (N_draws, n²)
    beta_point: np.ndarray  # shape (nk,)
    sigma_point: np.ndarray  # shape (n²,)
    extras_point: Optional[Any] = None  # model-owned auxiliary point state, if any
    extras_draws: Optional[list] = None  # model-owned auxiliary per-draw state, if any
```

`extras_point`/`extras_draws` are `None` for every current model; they only need populating by models that carry auxiliary state (via `PosteriorState.extras`) beyond `beta`/`sigma`. If provided, `extras_draws` must be draw-aligned: `SamplingResult` validates at construction that it has exactly one entry per `beta_draws`/`sigma_draws` row, raising `ValueError` immediately rather than letting a misaligned payload surface as an `IndexError` later during forecasting.

## Optional Predictive Hooks

Beyond the required contract, `SamplingModel` defines three concrete hooks that shape the model's *predictive* distribution — how forecast innovations are drawn, how constrained forecasts are sampled, and how out-of-sample densities are evaluated:

| Method | Called by | Default (Gaussian) behaviour |
|---|---|---|
| `sample_innovations(state, H, rng, point_only)` | `Forecasting.recursive_forecast` / `Forecasting._unconditional_forecast`, once per retained draw | Draws `~ N(0, Sigma)` reduced-form innovations from `state.sigma`, ignoring `state.extras` |
| `sample_conditional_forecast(state, C, f, Sigma_f, shape_f, last_p_obs, p, n, h, H, point_only, ...)` | `Forecasting._conditional_forecast`, once per Gibbs iteration | Routes to `bvar.forecast.conditional.draw_constrained_forecasts` using `state.beta`/`state.sigma`, ignoring `state.extras` |
| `predictive_logpdf(state, observation, mean, covariance)` | `GridSearch.marginal_likelihood_H`, once per rolling-window out-of-sample point | Evaluates `scipy.stats.multivariate_normal.logpdf(observation, mean, covariance)`, ignoring `state.extras` |

All three receive the complete `PosteriorState` for the draw, including
`beta`, `sigma`, and any model-owned `extras`. A model can use that state to
shape a different predictive distribution, such as a Student-*t* distribution
with a sampled degrees of freedom parameter. Each default implementation
raises `NotImplementedError` when `supports_gaussian_predictive` is `False`;
the model must override every hook that it uses.

Models that only need `beta`/`sigma` (every current model) can leave all three hooks at their defaults.

### GIRFs (`supports_girf`)

`supports_girf` gates whether `GIRF.compute_girf` may be used with a model at all, but the current GIRF implementation is hard-coded to the Gaussian reduced-form predictive distribution. `compute_girf` therefore also checks `supports_gaussian_predictive` at runtime and raises `NotImplementedError` unless **both** `supports_girf=True` and `supports_gaussian_predictive=True`. Setting `supports_girf=True` on a model with `supports_gaussian_predictive=False` does not enable GIRFs for it — a non-Gaussian model should leave `supports_girf=False` until a GIRF implementation compatible with its own predictive distribution exists.

## Minimal Example

The scaffold below shows the smallest possible model — a flat-prior direct sampler:

```python
# src/bvar/models/flat_prior/model.py

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
        minnesota: bool = True,
        soc: bool = False,
        sur: bool = False,
        covid: bool = False,
        covid_dates: Optional[list] = None,
    ) -> None:
        if soc or sur:
            raise ValueError("FlatPrior does not support soc or sur")
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
        _, n, k, nk, _ = get_dimensions(data, n_lags, covid_indices)

        #  ── S_0, nu_0 ──────────────────────────────────────────────────
        if self.pars.nu_0 is None:
            self.pars.nu_0 = n + 2  # barely proper IW
        if self.pars.S_0 is None:
            self.pars.S_0 = np.eye(n) * 1e-4  # nearly flat

        nu_0 = self.pars.nu_0
        S_0 = self.pars.S_0

        #  ── cache the state sample_posterior_state will need ──────────
        self.beta_0 = np.zeros(nk)
        self.V_A_inv = np.eye(k) * 1e-6  # (k, k) per-equation precision

        #  ── data matrices ────────────────────────────────────────────
        Y, Z = construct_Y_Z(data, n_lags, covid_indices)

        #  ── posterior update ─────────────────────────────────────────
        A_0 = self.beta_0.reshape(n, k).T
        K_A = self.V_A_inv + Z.T @ Z
        C_K_A = np.linalg.cholesky(K_A)
        A_post = np.linalg.solve(
            C_K_A.T,
            np.linalg.solve(C_K_A, self.V_A_inv @ A_0 + Z.T @ Y),
        )
        S_post = S_0 + A_0.T @ self.V_A_inv @ A_0 + Y.T @ Y - A_post.T @ K_A @ A_post
        T = Y.shape[0]

        beta_point = A_post.T.flatten()
        sigma_point = (S_post / (nu_0 + T - n - 1)).flatten()

        #  ── draw samples ─────────────────────────────────────────────
        beta_draws = np.empty((N_draws, nk))
        sigma_draws = np.empty((N_draws, n**2))

        if point_only:
            beta_draws[:] = beta_point
            sigma_draws[:] = sigma_point
        else:
            for i in range(N_draws):
                Sigma = invwishart.rvs(df=nu_0 + T, scale=S_post, random_state=rng)
                U = rng.normal(size=(k, n))
                C_Sig = np.linalg.cholesky(Sigma)
                beta = A_post + np.linalg.solve(C_K_A.T, U) @ C_Sig.T
                beta_draws[i] = beta.T.flatten()
                sigma_draws[i] = Sigma.flatten()

        return SamplingResult(
            beta_draws=beta_draws,
            sigma_draws=sigma_draws,
            beta_point=beta_point,
            sigma_point=sigma_point,
        )

    def sample_posterior_state(self, Y, Z, current_state, rng=None) -> PosteriorState:
        """One posterior draw — called by the conditional-forecast Gibbs loop.

        This is a direct sampler, so ``current_state`` (the previous draw in
        the chain) is ignored.
        """
        rng = rng if rng is not None else np.random.default_rng()
        n, k = Y.shape[1], Z.shape[1]
        nu_0, S_0 = self.pars.nu_0, self.pars.S_0

        A_0 = self.beta_0.reshape(n, k).T
        K_A = self.V_A_inv + Z.T @ Z
        C_K_A = np.linalg.cholesky(K_A)
        A_post = np.linalg.solve(
            C_K_A.T,
            np.linalg.solve(C_K_A, self.V_A_inv @ A_0 + Z.T @ Y),
        )
        S_post = S_0 + A_0.T @ self.V_A_inv @ A_0 + Y.T @ Y - A_post.T @ K_A @ A_post
        T = Y.shape[0]

        Sigma = invwishart.rvs(df=nu_0 + T, scale=S_post, random_state=rng)
        U = rng.normal(size=(k, n))
        C_Sig = np.linalg.cholesky(Sigma)
        beta = A_post + np.linalg.solve(C_K_A.T, U) @ C_Sig.T

        return PosteriorState(beta=beta.T.flatten(), sigma=Sigma.flatten())
```

### Wire it up

This step is optional. A `SamplingModel` subclass instance can be passed
straight to `BVAR(...)` without editing any package `__init__.py` files;
wiring it up only makes the class importable as `bv.FlatPrior`.

Expose the new class from the models package and optionally from the top-level namespace:

```python
# src/bvar/models/__init__.py
from .flat_prior.model import FlatPrior

# src/bvar/__init__.py
from .models import FlatPrior
```

Then use it like any other model:

```python
import bvar as bv

model = FlatPrior(soc=False, sur=False)
bvar = bv.BVAR(n_lags=2, model=model, stationary=False, optimisation_method="none")
bvar.sample(data, N_draws=2000)
bvar.forecast(H=8)
```

## Customising the Hyperparameter Interface

If the new model has additional hyperparameters (beyond `c1`, `c3`, `mu`, `theta`), override `set_priors`, `_compute_nb_hyper_pars`, `fill_in_from_vector`, `to_vector`, and optionally `hyperparameter_grid`.

See `IndependentNIW` for a worked example that adds a `c2` cross-variable shrinkage parameter.

## Checklist

- [ ] Subclass `SamplingModel` and set `requires_burnin`, `supports_ml` and `supports_point_only`
- [ ] Implement `sample()` — cache whatever state `sample_posterior_state` needs, return `SamplingResult`
- [ ] Implement `sample_posterior_state()` — single draw returned as a `PosteriorState`, using the cached fit state and posterior machinery (and `current_state.sigma`/`current_state.beta` if the sampler is state-dependent)
- [ ] If the model carries auxiliary state beyond `beta`/`sigma`, update and forward its own `PosteriorState.extras` in `sample_posterior_state`
- [ ] If the predictive distribution is non-Gaussian, set `supports_gaussian_predictive = False` and override `sample_innovations`, `sample_conditional_forecast` and/or `predictive_logpdf` as needed; leave `supports_girf = False` unless a compatible GIRF implementation exists
- [ ] If `supports_ml=False`, ensure `optimisation_method="ml"` raises a clear `ValueError`
- [ ] Export the class from `models/__init__.py`
- [ ] Add tests in `tests/`

## Reproducibility

Seeded results (`random_state=...`/`rng=...`) are reproducible within a single installed release, but are **not** guaranteed to be reproducible across releases: a change to a model's sampler or innovation implementation (such as the `sample_posterior_state` consolidation in 0.3.0) can alter the sequence of draws consumed from a given generator even when the underlying distributions are unchanged. Pin the package version if you need bit-for-bit reproducibility across environments or over time.
