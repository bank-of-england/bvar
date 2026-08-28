"""Tests for the conditional-forecast and predictive-density hooks (Phase 3, slice 2).

Covers:
* ``SamplingModel.sample_innovations`` raising ``NotImplementedError`` when
  ``supports_gaussian_predictive`` is ``False``.
* ``SamplingModel.sample_conditional_forecast``, the concrete hook that
    receives the full state and matches
  ``draw_constrained_forecasts`` by default and dispatching through a
  model's own override (carrying ``extras``) when called from
  ``Forecasting._conditional_forecast``.
* ``SamplingModel.predictive_logpdf`` matching
  ``scipy.stats.multivariate_normal.logpdf`` by default, and raising
  ``NotImplementedError`` when ``supports_gaussian_predictive`` is ``False``.
* ``GIRF.compute_girf`` raising ``NotImplementedError`` when
  ``self.model.supports_girf`` is ``False``.
"""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from bvar.BVAR import BVAR
from bvar.forecast.conditional import draw_constrained_forecasts, get_constraint
from bvar.models import NaturalConjugate
from bvar.models.base import PosteriorState, SamplingModel, SamplingResult
from bvar.utils import simulate_var


class _FakeSamplingModel(SamplingModel):
    """Minimal concrete model double for guard tests."""

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def sample_posterior_state(self, Y, Z, current_state, rng=None):
        return current_state


class _NonGaussianModel(_FakeSamplingModel):
    supports_gaussian_predictive = False


# ----------------------------------------------------------------------
# sample_innovations guard
# ----------------------------------------------------------------------


def test_sample_innovations_raises_when_gaussian_predictive_unsupported():
    model = _NonGaussianModel()
    state = PosteriorState(beta=np.zeros(1), sigma=np.eye(2).flatten())

    with pytest.raises(NotImplementedError):
        model.sample_innovations(state, H=3)


# ----------------------------------------------------------------------
# sample_conditional_forecast: default routes to draw_constrained_forecasts
# ----------------------------------------------------------------------


def test_sample_conditional_forecast_default_matches_draw_constrained_forecasts():
    n, p, H = 2, 1, 2
    k = n * p + 1
    rng_state = np.random.default_rng(0)
    beta = rng_state.standard_normal((k, n))
    A = rng_state.standard_normal((n, n))
    sigma = A @ A.T + np.eye(n)
    last_p_obs = rng_state.standard_normal((p, n))

    constraint_loc = np.full((H, n), np.nan)
    constraint_loc[0, 0] = 1.0
    C, f, Sigma_f, shape_f = get_constraint(constraint_loc, None, None)

    state = PosteriorState(beta=beta.T.flatten(), sigma=sigma.flatten())
    model = NaturalConjugate(minnesota=True, soc=True, sur=True, covid=False)

    hook_forecast = model.sample_conditional_forecast(
        state,
        C=C,
        f=f,
        Sigma_f=Sigma_f,
        shape_f=shape_f,
        last_p_obs=last_p_obs,
        p=p,
        n=n,
        h=0,
        H=H,
        point_only=False,
        constraint_sampler=None,
        method="andersson_et_al",
        rng=np.random.default_rng(42),
    )
    direct_forecast = draw_constrained_forecasts(
        sigma=sigma,
        beta=beta,
        C=C,
        f=f,
        Sigma_f=Sigma_f,
        shape_f=shape_f,
        last_p_obs=last_p_obs,
        p=p,
        n=n,
        h=0,
        H=H,
        point_only=False,
        constraint_sampler=None,
        method="andersson_et_al",
        rng=np.random.default_rng(42),
    )

    np.testing.assert_allclose(hook_forecast, direct_forecast)


def test_sample_conditional_forecast_raises_when_gaussian_predictive_unsupported():
    model = _NonGaussianModel()
    state = PosteriorState(beta=np.zeros(3), sigma=np.eye(1).flatten())

    with pytest.raises(NotImplementedError):
        model.sample_conditional_forecast(
            state,
            C=np.zeros((0, 1)),
            f=np.zeros(0),
            Sigma_f=np.zeros((0, 0)),
            shape_f=np.zeros(0),
            last_p_obs=np.zeros((1, 1)),
            p=1,
            n=1,
            h=0,
            H=1,
            point_only=True,
        )


# ----------------------------------------------------------------------
# Fake model whose sample_conditional_forecast dispatches on extras
# ----------------------------------------------------------------------


class _ExtrasConstraintModel(SamplingModel):
    """Fake model whose constrained forecast is a constant driven by ``extras``.

    ``beta``/``sigma`` never change; ``sample_posterior_state`` increments a
    scalar ``extras`` counter each Gibbs iteration, and
    ``sample_conditional_forecast`` ignores the constraint matrices entirely
    and returns a constant array equal to that counter -- so any variation
    across draws is attributable purely to ``extras`` being threaded through
    ``Forecasting._conditional_forecast``.
    """

    def sample(
        self,
        data,
        n_lags,
        covid_indices,
        vars_in_levels,
        N_draws,
        point_only=False,
        progressbar=True,
        soc=None,
        sur=None,
        rng=None,
    ) -> SamplingResult:
        n = data.shape[1]
        k = n * n_lags + 1 + len(covid_indices)
        beta_flat = np.zeros(n * k)
        sigma_flat = np.eye(n).flatten()
        n_draws = 1 if point_only else N_draws
        beta_draws = np.tile(beta_flat, (n_draws, 1))
        sigma_draws = np.tile(sigma_flat, (n_draws, 1))
        return SamplingResult(
            beta_draws=beta_draws,
            sigma_draws=sigma_draws,
            beta_point=beta_flat,
            sigma_point=sigma_flat,
            extras_point=0.0,
        )

    def sample_posterior_state(self, Y, Z, current_state, rng=None):
        extras = (current_state.extras or 0.0) + 1.0
        return PosteriorState(
            beta=current_state.beta.copy(),
            sigma=current_state.sigma.copy(),
            extras=extras,
        )

    def sample_conditional_forecast(
        self,
        state,
        C,
        f,
        Sigma_f,
        shape_f,
        last_p_obs,
        p,
        n,
        h,
        H,
        point_only,
        constraint_sampler=None,
        method="andersson_et_al",
        rng=None,
    ):
        offset = 0.0 if state.extras is None else float(state.extras)
        return np.full(H * n, offset)


def test_conditional_forecast_dispatches_through_model_hook_with_extras():
    H, N_draws = 3, 4
    data, _, _, _ = simulate_var(30, 2, 1, seed=99)
    model = _ExtrasConstraintModel(minnesota=True, soc=False, sur=False, covid=False)
    fitted = BVAR(1, model, stationary=True, optimisation_method="none")
    fitted.sample(data=data, N_draws=N_draws, progressbar=False)

    constraint_mean = np.full((H, fitted.n), np.nan)
    fitted.forecast(
        H=H,
        constraint_mean=constraint_mean,
        N_draws=N_draws,
        N_burn=0,
        point_only=False,
        progressbar=False,
    )

    forecast_conditional = fitted.forecast_conditional
    for draw in range(N_draws):
        np.testing.assert_allclose(
            forecast_conditional[draw, -H:, :], np.full((H, fitted.n), float(draw))
        )


# ----------------------------------------------------------------------
# sample_conditional_forecast receives an isolated PosteriorState copy
# ----------------------------------------------------------------------


class _MutatingConditionalForecastModel(SamplingModel):
    """Fake model whose ``sample_conditional_forecast`` mutates the state it
    receives in place.

    Used to verify that ``Forecasting._conditional_forecast`` always passes
    an isolated ``state.copy()`` into the hook, so such a mutation cannot
    alter the ``state`` object subsequently threaded into
    ``sample_posterior_state``.
    """

    def sample(
        self,
        data,
        n_lags,
        covid_indices,
        vars_in_levels,
        N_draws,
        point_only=False,
        progressbar=True,
        soc=None,
        sur=None,
        rng=None,
    ) -> SamplingResult:
        n = data.shape[1]
        k = n * n_lags + 1 + len(covid_indices)
        beta_flat = np.arange(1, n * k + 1, dtype=float)
        sigma_flat = np.eye(n).flatten()
        n_draws = 1 if point_only else N_draws
        beta_draws = np.tile(beta_flat, (n_draws, 1))
        sigma_draws = np.tile(sigma_flat, (n_draws, 1))
        return SamplingResult(
            beta_draws=beta_draws,
            sigma_draws=sigma_draws,
            beta_point=beta_flat,
            sigma_point=sigma_flat,
        )

    def sample_posterior_state(self, Y, Z, current_state, rng=None):
        self.received_betas.append(current_state.beta.copy())
        self.received_sigmas.append(current_state.sigma.copy())
        return PosteriorState(
            beta=current_state.beta.copy(),
            sigma=current_state.sigma.copy(),
            extras=current_state.extras,
        )

    def sample_conditional_forecast(
        self,
        state,
        C,
        f,
        Sigma_f,
        shape_f,
        last_p_obs,
        p,
        n,
        h,
        H,
        point_only,
        constraint_sampler=None,
        method="andersson_et_al",
        rng=None,
    ):
        # Mutate the received state in place; this must not affect the
        # `state` object used for the subsequent posterior update.
        state.beta += 999.0
        state.sigma += 999.0
        return np.zeros(H * n)


def test_conditional_forecast_hook_mutation_does_not_leak_into_posterior_update():
    H, N_draws = 3, 4
    data, _, _, _ = simulate_var(30, 2, 1, seed=99)
    model = _MutatingConditionalForecastModel(
        minnesota=True, soc=False, sur=False, covid=False
    )
    fitted = BVAR(1, model, stationary=True, optimisation_method="none")
    # BVAR deep-copies the model in __init__, so attach the tracking lists
    # to fitted.model (the instance actually exercised by forecast()).
    fitted.model.received_betas = []
    fitted.model.received_sigmas = []
    fitted.sample(data=data, N_draws=N_draws, progressbar=False)

    beta_point_before = fitted.beta_point.copy()
    sigma_point_before = fitted.sigma_point.copy()

    constraint_mean = np.full((H, fitted.n), np.nan)
    fitted.forecast(
        H=H,
        constraint_mean=constraint_mean,
        N_draws=N_draws,
        N_burn=0,
        point_only=False,
        progressbar=False,
    )

    # sample_posterior_state must see the un-mutated seed state on the
    # first Gibbs iteration, and the fitted point-estimate state must remain intact.
    np.testing.assert_allclose(fitted.model.received_betas[0], beta_point_before)
    np.testing.assert_allclose(fitted.model.received_sigmas[0], sigma_point_before)
    np.testing.assert_allclose(fitted.beta_point, beta_point_before)
    np.testing.assert_allclose(fitted.sigma_point, sigma_point_before)


# ----------------------------------------------------------------------
# predictive_logpdf
# ----------------------------------------------------------------------


def test_predictive_logpdf_default_matches_scipy_multivariate_normal():
    model = NaturalConjugate(minnesota=True, soc=True, sur=True, covid=False)
    state = PosteriorState(beta=np.zeros(1), sigma=np.eye(2).flatten())
    observation = np.array([0.5, -0.2])
    mean = np.array([0.0, 0.0])
    covariance = np.array([[1.0, 0.2], [0.2, 1.5]])

    result = model.predictive_logpdf(state, observation, mean, covariance)
    expected = multivariate_normal.logpdf(observation, mean, covariance)

    assert result == pytest.approx(expected)


def test_predictive_logpdf_raises_when_gaussian_predictive_unsupported():
    model = _NonGaussianModel()
    state = PosteriorState(beta=np.zeros(1), sigma=np.eye(1).flatten())

    with pytest.raises(NotImplementedError):
        model.predictive_logpdf(state, np.array([0.0]), np.array([0.0]), np.eye(1))


# ----------------------------------------------------------------------
# GIRF guard
# ----------------------------------------------------------------------


def test_compute_girf_raises_when_model_does_not_support_girf():
    data, _, _, _ = simulate_var(60, 2, 1, seed=0)
    priors = NaturalConjugate(minnesota=True, soc=False, sur=False, covid=False)
    fitted = BVAR(1, priors, stationary=True, optimisation_method="none")
    fitted.sample(data=data, N_draws=10, progressbar=False)
    fitted.model.supports_girf = False

    with pytest.raises(NotImplementedError):
        fitted.compute_girf(H=2, N_draws=10, progressbar=False)


def test_compute_girf_raises_when_gaussian_predictive_unsupported_with_default_supports_girf():
    """A model declaring ``supports_gaussian_predictive=False`` must not fall
    through to the hard-coded Gaussian GIRF implementation merely because it
    inherits ``supports_girf=True`` from the base class."""
    data, _, _, _ = simulate_var(60, 2, 1, seed=0)
    priors = NaturalConjugate(minnesota=True, soc=False, sur=False, covid=False)
    fitted = BVAR(1, priors, stationary=True, optimisation_method="none")
    fitted.sample(data=data, N_draws=10, progressbar=False)
    fitted.model.supports_gaussian_predictive = False
    assert fitted.model.supports_girf is True

    with pytest.raises(NotImplementedError):
        fitted.compute_girf(H=2, N_draws=10, progressbar=False)
