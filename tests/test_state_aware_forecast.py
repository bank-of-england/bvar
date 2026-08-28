"""Tests for forecast innovations that receive the full posterior state (Phase 3, slice 1).

Covers:
* ``SamplingResult.extras_draws`` preserving per-draw ``extras`` alongside
  ``beta_draws``/``sigma_draws``, defaulting to ``None`` for current models.
* ``SamplingModel.sample_innovations``, the concrete innovation hook that
    receives the full state and provides the default Gaussian behaviour.
* ``Forecasting.recursive_forecast``/``Forecasting._unconditional_forecast``
  dispatching through that hook using each draw's complete
  ``PosteriorState`` (or ``posterior_state_point`` when ``point_only``).
"""

import numpy as np
import pytest

from bvar.BVAR import BVAR
from bvar.models import NaturalConjugate
from bvar.models.base import PosteriorState, SamplingModel, SamplingResult
from bvar.utils import simulate_var

# ----------------------------------------------------------------------
# SamplingResult.extras_draws
# ----------------------------------------------------------------------


def test_sampling_result_extras_draws_defaults_to_none():
    result = SamplingResult(
        beta_draws=np.zeros((2, 3)),
        sigma_draws=np.zeros((2, 4)),
        beta_point=np.zeros(3),
        sigma_point=np.zeros(4),
    )

    assert result.extras_draws is None


def test_sampling_result_extras_draws_preserved_alongside_arrays():
    extras_draws = [{"df": 1.0}, {"df": 2.0}]
    result = SamplingResult(
        beta_draws=np.zeros((2, 3)),
        sigma_draws=np.zeros((2, 4)),
        beta_point=np.zeros(3),
        sigma_point=np.zeros(4),
        extras_draws=extras_draws,
    )

    assert result.extras_draws is extras_draws


def test_sampling_result_extras_draws_misaligned_length_raises_value_error():
    with pytest.raises(ValueError):
        SamplingResult(
            beta_draws=np.zeros((2, 3)),
            sigma_draws=np.zeros((2, 4)),
            beta_point=np.zeros(3),
            sigma_point=np.zeros(4),
            extras_draws=[{"df": 1.0}, {"df": 2.0}, {"df": 3.0}],
        )


def test_sampling_result_mismatched_beta_sigma_draw_rows_raises_value_error():
    with pytest.raises(ValueError):
        SamplingResult(
            beta_draws=np.zeros((2, 3)),
            sigma_draws=np.zeros((3, 4)),
            beta_point=np.zeros(3),
            sigma_point=np.zeros(4),
        )


def test_bvar_extras_is_none_for_current_models(bvar):
    assert bvar.extras is None


# ----------------------------------------------------------------------
# SamplingModel.sample_innovations (default Gaussian hook)
# ----------------------------------------------------------------------


def test_sample_innovations_default_matches_gaussian_multivariate_normal():
    model = NaturalConjugate(minnesota=True, soc=True, sur=True, covid=False)
    sigma = np.array([[1.0, 0.2, 0.0], [0.2, 1.5, 0.1], [0.0, 0.1, 2.0]])
    n = sigma.shape[0]
    state = PosteriorState(beta=np.zeros(1), sigma=sigma.flatten())
    H = 5

    innovations = model.sample_innovations(state, H, rng=np.random.default_rng(42))
    expected = np.random.default_rng(42).multivariate_normal(np.zeros(n), sigma, size=H)

    assert innovations.shape == (H, n)
    np.testing.assert_allclose(innovations, expected)


def test_sample_innovations_point_only_returns_zeros():
    model = NaturalConjugate(minnesota=True, soc=True, sur=True, covid=False)
    sigma = np.eye(2)
    state = PosteriorState(beta=np.zeros(1), sigma=sigma.flatten())

    innovations = model.sample_innovations(state, H=4, point_only=True)

    assert innovations.shape == (4, 2)
    np.testing.assert_allclose(innovations, np.zeros((4, 2)))


def test_sample_innovations_ignores_extras_by_default():
    model = NaturalConjugate(minnesota=True, soc=True, sur=True, covid=False)
    sigma = np.eye(2)
    state_no_extras = PosteriorState(beta=np.zeros(1), sigma=sigma.flatten())
    state_with_extras = PosteriorState(
        beta=np.zeros(1), sigma=sigma.flatten(), extras={"df": 3.0}
    )

    a = model.sample_innovations(state_no_extras, H=3, rng=np.random.default_rng(7))
    b = model.sample_innovations(state_with_extras, H=3, rng=np.random.default_rng(7))

    np.testing.assert_allclose(a, b)


# ----------------------------------------------------------------------
# Fake model whose extras change the innovation output
# ----------------------------------------------------------------------


class _ExtrasOffsetModel(SamplingModel):
    """Minimal model double whose ``extras`` shift the innovation output.

    ``beta`` is always zero and ``sigma`` is always the identity, so any
    non-zero forecast is attributable entirely to the innovations returned
    by ``sample_innovations``, which returns a constant array equal to the
    draw's ``extras`` value (or ``0.0`` when ``extras`` is ``None``).
    """

    requires_burnin = False
    supports_ml = False

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
        extras_draws = [float(i) for i in range(n_draws)]
        return SamplingResult(
            beta_draws=beta_draws,
            sigma_draws=sigma_draws,
            beta_point=beta_flat,
            sigma_point=sigma_flat,
            extras_point=0.0,
            extras_draws=extras_draws,
        )

    def sample_posterior_state(self, Y, Z, current_state, rng=None):
        return current_state

    def sample_innovations(self, state, H, rng=None, point_only=False):
        n = int(round(np.sqrt(state.sigma.shape[0])))
        offset = 0.0 if state.extras is None else float(state.extras)
        return np.full((H, n), offset)


@pytest.fixture
def extras_bvar():
    data, _, _, _ = simulate_var(30, 2, 1, seed=99)
    model = _ExtrasOffsetModel(minnesota=True, soc=False, sur=False, covid=False)
    fitted = BVAR(1, model, stationary=True, optimisation_method="none")
    fitted.sample(data=data, N_draws=4, progressbar=False)
    return fitted


def test_bvar_preserves_extras_draws_from_model(extras_bvar):
    assert extras_bvar.extras == [0.0, 1.0, 2.0, 3.0]


def test_recursive_forecast_routes_extras_through_hook(extras_bvar):
    H = 3
    T = extras_bvar.T
    extras_bvar.recursive_forecast(H=H, N_draws=4, point_only=False, progressbar=False)
    forecast = extras_bvar.forecast_unconditional

    for draw in range(4):
        np.testing.assert_allclose(
            forecast[draw, T:, :], np.full((H, extras_bvar.n), float(draw))
        )


def test_unconditional_forecast_routes_extras_through_hook(extras_bvar):
    H = 3
    T = extras_bvar.T
    extras_bvar.forecast(H=H, N_draws=4, point_only=False, progressbar=False)
    forecast = extras_bvar.forecast_unconditional

    for draw in range(4):
        np.testing.assert_allclose(
            forecast[draw, T:, :],
            np.full((H, extras_bvar.n), float(draw)),
            atol=1e-8,
        )


# ----------------------------------------------------------------------
# sample_innovations receives an isolated PosteriorState copy
# ----------------------------------------------------------------------


class _MutatingInnovationsModel(SamplingModel):
    """Fake model whose ``sample_innovations`` mutates the state it receives.

    Used to verify that ``Forecasting`` always passes an isolated
    ``PosteriorState`` copy into ``sample_innovations``, so a hook that
    mutates ``state.beta``/``state.sigma`` in place can never leak into the
    fitted posterior arrays (``beta``/``sigma``/``beta_point``/``sigma_point``/
    ``posterior_state_point``).
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
        return current_state

    def sample_innovations(self, state, H, rng=None, point_only=False):
        n = int(round(np.sqrt(state.sigma.shape[0])))
        state.beta += 1000.0
        state.sigma += 1000.0
        return np.zeros((H, n))


@pytest.fixture
def mutating_bvar():
    data, _, _, _ = simulate_var(30, 2, 1, seed=99)
    model = _MutatingInnovationsModel(minnesota=True, soc=False, sur=False, covid=False)
    fitted = BVAR(1, model, stationary=True, optimisation_method="none")
    fitted.sample(data=data, N_draws=4, progressbar=False)
    return fitted


@pytest.mark.parametrize("point_only", [False, True])
def test_forecast_innovations_hook_mutation_does_not_leak_into_fitted_state(
    mutating_bvar, point_only
):
    beta_before = mutating_bvar.beta.copy()
    sigma_before = mutating_bvar.sigma.copy()
    beta_point_before = mutating_bvar.beta_point.copy()
    sigma_point_before = mutating_bvar.sigma_point.copy()

    mutating_bvar.forecast(H=3, N_draws=4, point_only=point_only, progressbar=False)

    np.testing.assert_allclose(mutating_bvar.beta, beta_before)
    np.testing.assert_allclose(mutating_bvar.sigma, sigma_before)
    np.testing.assert_allclose(mutating_bvar.beta_point, beta_point_before)
    np.testing.assert_allclose(mutating_bvar.sigma_point, sigma_point_before)
    np.testing.assert_allclose(
        mutating_bvar.posterior_state_point.beta, beta_point_before
    )
    np.testing.assert_allclose(
        mutating_bvar.posterior_state_point.sigma, sigma_point_before
    )


@pytest.mark.parametrize("point_only", [False, True])
def test_recursive_forecast_innovations_hook_mutation_does_not_leak_into_fitted_state(
    mutating_bvar, point_only
):
    beta_before = mutating_bvar.beta.copy()
    sigma_before = mutating_bvar.sigma.copy()
    beta_point_before = mutating_bvar.beta_point.copy()
    sigma_point_before = mutating_bvar.sigma_point.copy()

    mutating_bvar.recursive_forecast(
        H=3, N_draws=4, point_only=point_only, progressbar=False
    )

    np.testing.assert_allclose(mutating_bvar.beta, beta_before)
    np.testing.assert_allclose(mutating_bvar.sigma, sigma_before)
    np.testing.assert_allclose(mutating_bvar.beta_point, beta_point_before)
    np.testing.assert_allclose(mutating_bvar.sigma_point, sigma_point_before)
