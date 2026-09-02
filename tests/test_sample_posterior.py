"""Tests for the ``PosteriorState``-based ``sample_posterior_state`` contract.

``sample_posterior_state`` is the sole posterior-update extension point:
the legacy tuple-based ``sample_posterior``/``draw_posterior`` hooks have
been removed entirely.
"""

import copy

import numpy as np
import pandas as pd
import pytest

import bvar as bv
from bvar.BVAR import BVAR
from bvar.models import IndependentNIW, NaturalConjugate
from bvar.models.base import PosteriorState, SamplingModel, SamplingResult
from bvar.utils import construct_Y_Z, simulate_var


def _fitted_model_and_matrices(model_cls=NaturalConjugate):
    T, n, n_lags = 100, 2, 1
    data, _, _, _ = simulate_var(T, n, n_lags, seed=1234)
    data = data.to_numpy()
    covid_indices = np.array([], dtype=int)
    vars_in_levels = np.zeros(n, dtype=bool)

    model = model_cls(minnesota=True, soc=True, sur=True, covid=False)
    model.sample(
        data,
        n_lags,
        covid_indices,
        vars_in_levels,
        N_draws=1,
        point_only=model_cls is NaturalConjugate,
        progressbar=False,
    )

    Y, Z = construct_Y_Z(data, n_lags, covid_indices)
    return model, Y, Z


class _BurninExtrasModel(SamplingModel):
    """Small model double for retained draw and extras alignment."""

    requires_burnin = True
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
    ):
        n = data.shape[1]
        nk = n * (n * n_lags + 1)
        beta_draws = np.arange(N_draws, dtype=float).reshape(-1, 1)
        beta_draws = np.broadcast_to(beta_draws, (N_draws, nk)).copy()
        sigma_draws = np.arange(N_draws, dtype=float).reshape(-1, 1)
        sigma_draws = np.broadcast_to(sigma_draws, (N_draws, n**2)).copy()
        return SamplingResult(
            beta_draws=beta_draws,
            sigma_draws=sigma_draws,
            beta_point=np.full(nk, -1.0),
            sigma_point=np.full(n**2, -1.0),
            extras_point={"summary": True},
            extras_draws=[{"draw": i} for i in range(N_draws)],
        )

    def sample_posterior_state(self, Y, Z, current_state, rng=None):
        return current_state


class _FailingSampleModel(NaturalConjugate):
    """Mutate model-owned state and fail, to test atomic BVAR refits."""

    def __init__(self, *args, **kwargs):
        self.fail = False
        super().__init__(*args, **kwargs)

    def sample(self, *args, **kwargs):
        if self.fail:
            if kwargs.get("rng") is not None:
                kwargs["rng"].normal()
            self.pars.c1 = 999.0
            self.pars.S_0 = np.full((2, 2), 999.0)
            self.beta_0 = np.full(6, 999.0)
            self.V_A_inv = np.full((3, 3), 999.0)
            raise RuntimeError("synthetic sampling failure")
        return super().sample(*args, **kwargs)


def test_parameter_draw_is_removed():
    with pytest.raises(ImportError):
        from bvar.models.base import ParameterDraw  # noqa: F401


def test_sample_posterior_state_returns_state_with_expected_shapes():
    model, Y, Z = _fitted_model_and_matrices()
    current_state = PosteriorState(
        beta=np.zeros_like(model.beta_0), sigma=model.pars.S_0.flatten()
    )

    next_state = model.sample_posterior_state(
        Y, Z, current_state, rng=np.random.default_rng(0)
    )

    assert isinstance(next_state, PosteriorState)
    assert isinstance(next_state.beta, np.ndarray)
    assert isinstance(next_state.sigma, np.ndarray)
    assert next_state.beta.shape == model.beta_0.shape
    assert next_state.sigma.shape == model.pars.S_0.flatten().shape


def test_sample_posterior_state_ignores_current_state_for_natural_conjugate():
    model, Y, Z = _fitted_model_and_matrices()

    seed_state = PosteriorState(
        beta=np.zeros_like(model.beta_0), sigma=model.pars.S_0.flatten()
    )
    stale_state = PosteriorState(
        beta=np.full_like(model.beta_0, 999.0),
        sigma=np.full(model.pars.S_0.size, 999.0),
    )

    state_a = model.sample_posterior_state(
        Y, Z, seed_state, rng=np.random.default_rng(1)
    )
    state_b = model.sample_posterior_state(
        Y, Z, stale_state, rng=np.random.default_rng(1)
    )

    np.testing.assert_allclose(state_a.beta, state_b.beta)
    np.testing.assert_allclose(state_a.sigma, state_b.sigma)


def test_sample_posterior_hook_is_removed():
    model, _, _ = _fitted_model_and_matrices()

    assert not hasattr(model, "sample_posterior")


def test_draw_posterior_hook_is_removed():
    model, _, _ = _fitted_model_and_matrices()

    assert not hasattr(model, "draw_posterior")


def test_independent_niw_sample_posterior_hook_is_removed():
    model, _, _ = _fitted_model_and_matrices(model_cls=IndependentNIW)

    assert not hasattr(model, "sample_posterior")


def test_independent_niw_draw_posterior_hook_is_removed():
    model, _, _ = _fitted_model_and_matrices(model_cls=IndependentNIW)

    assert not hasattr(model, "draw_posterior")


def test_independent_niw_is_instantiable():
    model = IndependentNIW(minnesota=True, soc=True, sur=True, covid=False)

    assert isinstance(model, IndependentNIW)


def test_independent_niw_sample_posterior_state_returns_state_with_expected_shapes():
    model, Y, Z = _fitted_model_and_matrices(model_cls=IndependentNIW)
    current_state = PosteriorState(
        beta=np.zeros_like(model.beta_0), sigma=model.pars.S_0.flatten()
    )

    next_state = model.sample_posterior_state(
        Y, Z, current_state, rng=np.random.default_rng(0)
    )

    assert isinstance(next_state, PosteriorState)
    assert isinstance(next_state.beta, np.ndarray)
    assert isinstance(next_state.sigma, np.ndarray)
    assert next_state.beta.shape == model.beta_0.shape
    assert next_state.sigma.shape == model.pars.S_0.flatten().shape


def test_independent_niw_sample_posterior_state_uses_current_sigma():
    """Different current_state.sigma values must change the resulting draw."""
    model, Y, Z = _fitted_model_and_matrices(model_cls=IndependentNIW)
    current_beta = np.zeros_like(model.beta_0)

    state_a = PosteriorState(beta=current_beta, sigma=model.pars.S_0.flatten())
    state_b = PosteriorState(beta=current_beta, sigma=(2.0 * model.pars.S_0).flatten())

    next_a = model.sample_posterior_state(Y, Z, state_a, rng=np.random.default_rng(42))
    next_b = model.sample_posterior_state(Y, Z, state_b, rng=np.random.default_rng(42))

    assert not np.allclose(next_a.beta, next_b.beta)
    assert not np.allclose(next_a.sigma, next_b.sigma)


def test_independent_niw_sample_posterior_state_matches_first_gibbs_step():
    """A current_state.sigma equal to the chain's initial covariance must
    give the same one-step result as the first iteration of
    ``_sample_gibbs`` under the same seed."""
    model, Y, Z = _fitted_model_and_matrices(model_cls=IndependentNIW)
    current_beta = np.zeros_like(model.beta_0)
    n = Y.shape[1]
    initial_sigma = model.pars.S_0 / (model.pars.nu_0 + n + 1)
    current_state = PosteriorState(beta=current_beta, sigma=initial_sigma.flatten())

    next_state = model.sample_posterior_state(
        Y, Z, current_state, rng=np.random.default_rng(7)
    )

    beta_draws, sigma_draws, _, _ = IndependentNIW._sample_gibbs(
        Y,
        Z,
        model.beta_0,
        model.pars.S_0,
        model.V_A_inv,
        model.pars.nu_0,
        N_draws=1,
        progressbar=False,
        rng=np.random.default_rng(7),
    )

    np.testing.assert_allclose(next_state.beta, beta_draws[0])
    np.testing.assert_allclose(next_state.sigma, sigma_draws[0])


def test_independent_niw_sample_gibbs_shapes_and_validity():
    model, Y, Z = _fitted_model_and_matrices(model_cls=IndependentNIW)
    n = Y.shape[1]
    N_draws = 5

    beta_draws, sigma_draws, beta_point, sigma_point = IndependentNIW._sample_gibbs(
        Y,
        Z,
        model.beta_0,
        model.pars.S_0,
        model.V_A_inv,
        model.pars.nu_0,
        N_draws=N_draws,
        progressbar=False,
        rng=np.random.default_rng(3),
    )

    nk = model.beta_0.shape[0]
    assert beta_draws.shape == (N_draws, nk)
    assert sigma_draws.shape == (N_draws, n**2)
    assert beta_point.shape == (nk,)
    assert sigma_point.shape == (n**2,)
    assert np.all(np.isfinite(beta_draws))
    assert np.all(np.isfinite(sigma_draws))

    for row in sigma_draws:
        Sigma = row.reshape(n, n)
        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.all(eigvals > 0)


def test_independent_niw_sample_posterior_state_ignores_current_beta():
    """current_state.beta must not affect the draw: this model's sweep
    order (\u03b2 | \u03a3 then \u03a3 | \u03b2) always draws a fresh \u03b2 conditional on \u03a3."""
    model, Y, Z = _fitted_model_and_matrices(model_cls=IndependentNIW)
    current_sigma = model.pars.S_0.flatten()

    state_a = PosteriorState(beta=np.zeros_like(model.beta_0), sigma=current_sigma)
    state_b = PosteriorState(
        beta=np.full_like(model.beta_0, 999.0), sigma=current_sigma
    )

    next_a = model.sample_posterior_state(Y, Z, state_a, rng=np.random.default_rng(11))
    next_b = model.sample_posterior_state(Y, Z, state_b, rng=np.random.default_rng(11))

    np.testing.assert_allclose(next_a.beta, next_b.beta)
    np.testing.assert_allclose(next_a.sigma, next_b.sigma)


def test_independent_niw_sample_posterior_state_covariance_is_positive_definite():
    model, Y, Z = _fitted_model_and_matrices(model_cls=IndependentNIW)
    current_state = PosteriorState(
        beta=np.zeros_like(model.beta_0), sigma=model.pars.S_0.flatten()
    )
    n = Y.shape[1]

    next_state = model.sample_posterior_state(
        Y, Z, current_state, rng=np.random.default_rng(5)
    )

    eigvals = np.linalg.eigvalsh(next_state.sigma.reshape(n, n))
    assert np.all(eigvals > 0)


def test_natural_conjugate_sample_direct_is_callable():
    """The renamed ``_sample_direct`` helper preserves the previous
    ``_sample`` behaviour and remains directly callable."""
    model, Y, Z = _fitted_model_and_matrices(model_cls=NaturalConjugate)

    beta_draws, sigma_draws, beta_point, sigma_point = NaturalConjugate._sample_direct(
        Y,
        Z,
        model.beta_0,
        model.pars.S_0,
        model.V_A_inv,
        model.pars.nu_0,
        N_draws=1,
        point_only=False,
        progressbar=False,
        rng=np.random.default_rng(0),
    )

    assert beta_draws.shape == (1, model.beta_0.shape[0])
    assert sigma_draws.shape == (1, model.pars.S_0.size)
    assert beta_point.shape == (model.beta_0.shape[0],)
    assert sigma_point.shape == (model.pars.S_0.size,)


def test_natural_conjugate_sample_direct_point_only_populates_deterministic_draws():
    """Point-only direct draws are finite, shaped, and equal to the point estimate."""
    model, Y, Z = _fitted_model_and_matrices(model_cls=NaturalConjugate)
    N_draws = 3

    result_a = NaturalConjugate._sample_direct(
        Y,
        Z,
        model.beta_0,
        model.pars.S_0,
        model.V_A_inv,
        model.pars.nu_0,
        N_draws=N_draws,
        point_only=True,
        progressbar=False,
        rng=np.random.default_rng(0),
    )
    result_b = NaturalConjugate._sample_direct(
        Y,
        Z,
        model.beta_0,
        model.pars.S_0,
        model.V_A_inv,
        model.pars.nu_0,
        N_draws=N_draws,
        point_only=True,
        progressbar=False,
        rng=np.random.default_rng(1),
    )

    beta_draws_a, sigma_draws_a, beta_point, sigma_point = result_a
    beta_draws_b, sigma_draws_b, _, _ = result_b
    assert beta_draws_a.shape == (N_draws, model.beta_0.shape[0])
    assert sigma_draws_a.shape == (N_draws, model.pars.S_0.size)
    assert np.all(np.isfinite(beta_draws_a))
    assert np.all(np.isfinite(sigma_draws_a))
    np.testing.assert_allclose(
        beta_draws_a, np.broadcast_to(beta_point, beta_draws_a.shape)
    )
    np.testing.assert_allclose(
        sigma_draws_a, np.broadcast_to(sigma_point, sigma_draws_a.shape)
    )
    np.testing.assert_array_equal(beta_draws_a, beta_draws_b)
    np.testing.assert_array_equal(sigma_draws_a, sigma_draws_b)


def test_bvar_point_only_supports_fitted_values_and_point_forecasts():
    """Point-only fits remain usable by fitted-value and forecast consumers."""
    data, _, _, _ = simulate_var(40, 2, 1, seed=4321)
    model = NaturalConjugate(minnesota=True, soc=False, sur=False, covid=False)
    bvar = BVAR(1, model, stationary=True, optimisation_method="none")

    bvar.sample(data, N_draws=3, point_only=True, progressbar=False, random_state=7)
    bvar.compute_fitted_values()

    expected_fitted = bvar.X @ bvar.beta_point.reshape(-1, 1)
    assert bvar.fitted_values.shape == (3, bvar.T, bvar.n)
    np.testing.assert_allclose(
        bvar.fitted_values,
        np.broadcast_to(
            expected_fitted.reshape(1, bvar.T, bvar.n), bvar.fitted_values.shape
        ),
    )

    bvar.forecast(H=2, point_only=True, progressbar=False)
    matrix_forecast = bvar.forecast_unconditional.copy()
    bvar.recursive_forecast(H=2, point_only=True, progressbar=False)

    assert matrix_forecast.shape == (1, bvar.T + 2, bvar.n)
    assert np.all(np.isfinite(matrix_forecast))
    np.testing.assert_allclose(bvar.forecast_unconditional, matrix_forecast)


def test_independent_niw_rejects_point_only_sampling():
    """IndependentNIW continues to reject the unsupported point-only path."""
    data, _, _, _ = simulate_var(20, 2, 1, seed=4321)
    model = IndependentNIW(minnesota=True, soc=False, sur=False, covid=False)
    bvar = BVAR(1, model, stationary=True, optimisation_method="none")

    with pytest.raises(ValueError, match="point_only"):
        bvar.sample(data, N_draws=2, point_only=True, progressbar=False)


def test_conditional_forecast_forwards_extras_through_state_adapter(bvar, monkeypatch):
    """Conditional forecasting must seed the chain from the fitted point-estimate
    state, call ``sample_posterior_state`` each iteration, forward whatever
    ``extras`` it returns to the next iteration in order, and leave the
    fitted point-estimate state untouched."""
    calls = []

    nk = bvar.beta_point.shape[0]
    n = bvar.n

    beta_point_before = bvar.beta_point.copy()
    sigma_point_before = bvar.sigma_point.copy()

    returned_beta = np.linspace(0.01, 0.02, nk)
    returned_sigma = (1.5 * np.eye(n)).flatten()

    def spy_sample_posterior_state(Y, Z, current_state, rng=None):
        n_calls = (current_state.extras or {}).get("n_calls", 0) + 1
        calls.append((current_state.beta.copy(), current_state.sigma.copy(), n_calls))
        return PosteriorState(
            beta=returned_beta, sigma=returned_sigma, extras={"n_calls": n_calls}
        )

    monkeypatch.setattr(
        bvar.model, "sample_posterior_state", spy_sample_posterior_state
    )

    H = 2
    constraint_mean = np.full((H, n), np.nan)
    constraint_mean[:, 0] = 0.0

    bvar.forecast(
        H=H,
        constraint_mean=constraint_mean,
        N_draws=2,
        point_only=False,
        progressbar=False,
        random_state=1234,
    )

    assert len(calls) == 2
    first_beta, first_sigma, first_n_calls = calls[0]
    second_beta, second_sigma, second_n_calls = calls[1]

    np.testing.assert_allclose(first_beta, bvar.beta_point)
    np.testing.assert_allclose(first_sigma, bvar.sigma_point)
    assert first_n_calls == 1

    np.testing.assert_allclose(second_beta, returned_beta)
    np.testing.assert_allclose(second_sigma, returned_sigma)
    assert second_n_calls == 2

    # the fitted point-estimate state must survive the conditional forecast loop
    np.testing.assert_allclose(bvar.beta_point, beta_point_before)
    np.testing.assert_allclose(bvar.sigma_point, sigma_point_before)
    assert bvar.posterior_state_point.beta is bvar.beta_point
    assert bvar.posterior_state_point.sigma is bvar.sigma_point


def test_conditional_forecast_cannot_mutate_fitted_extras_through_nested_state(
    bvar, monkeypatch
):
    """A model whose ``extras`` carries nested mutable state must not be able
    to mutate the fitted point-estimate payload via the copy that seeds the Gibbs
    chain, even when the sampler mutates the ``extras`` it receives."""
    nk = bvar.beta_point.shape[0]
    n = bvar.n

    history = []
    fitted_extras = {"history": history}
    monkeypatch.setattr(
        bvar,
        "posterior_state_point",
        PosteriorState(
            beta=bvar.beta_point, sigma=bvar.sigma_point, extras=fitted_extras
        ),
    )

    returned_beta = np.linspace(0.01, 0.02, nk)
    returned_sigma = (1.5 * np.eye(n)).flatten()

    def mutating_sample_posterior_state(Y, Z, current_state, rng=None):
        # mutate the nested payload threaded through from the (copied) seed
        current_state.extras["history"].append("mutated")
        return PosteriorState(
            beta=returned_beta, sigma=returned_sigma, extras=current_state.extras
        )

    monkeypatch.setattr(
        bvar.model, "sample_posterior_state", mutating_sample_posterior_state
    )

    H = 2
    constraint_mean = np.full((H, n), np.nan)
    constraint_mean[:, 0] = 0.0

    bvar.forecast(
        H=H,
        constraint_mean=constraint_mean,
        N_draws=2,
        point_only=False,
        progressbar=False,
        random_state=1234,
    )

    assert history == []
    assert bvar.posterior_state_point.extras["history"] == []


def _fitted_independent_niw_bvar(N_draws=100, N_burn=None, seed=321):
    T, n, n_lags = 100, 2, 1
    data, _, _, _ = simulate_var(T, n, n_lags, covid=False, levels=False, seed=seed)

    priors = IndependentNIW(minnesota=True, soc=True, sur=True, covid=False)
    bvar = BVAR(n_lags, priors, True, optimisation_method="none")
    bvar.sample(
        data,
        N_draws=N_draws,
        N_burn=N_burn,
        progressbar=False,
        random_state=seed,
    )
    return bvar


def test_independent_niw_point_estimates_use_retained_draws():
    """Independent-NIW summaries use draws after explicit burn-in."""
    bvar = _fitted_independent_niw_bvar(N_draws=8, N_burn=4)

    np.testing.assert_allclose(bvar.beta_point, bvar.beta.mean(axis=0))
    np.testing.assert_allclose(bvar.sigma_point, bvar.sigma.mean(axis=0))
    assert bvar.posterior_state_point.beta is bvar.beta_point
    assert bvar.posterior_state_point.sigma is bvar.sigma_point


def test_independent_niw_burn_in_changes_retained_draw_summary():
    """Changing burn-in changes summaries to match the corresponding chain slice."""
    all_draws = _fitted_independent_niw_bvar(N_draws=7, N_burn=0)
    retained = _fitted_independent_niw_bvar(N_draws=4, N_burn=3)

    np.testing.assert_allclose(retained.beta, all_draws.beta[3:])
    np.testing.assert_allclose(retained.sigma, all_draws.sigma[3:])
    np.testing.assert_allclose(retained.beta_point, retained.beta.mean(axis=0))
    np.testing.assert_allclose(retained.sigma_point, retained.sigma.mean(axis=0))


def test_independent_niw_sampling_is_reproducible_with_fixed_seed():
    """A fixed seed reproduces Independent-NIW draws and posterior moments."""
    first = _fitted_independent_niw_bvar(N_draws=8, N_burn=2, seed=321)
    second = _fitted_independent_niw_bvar(N_draws=8, N_burn=2, seed=321)

    np.testing.assert_allclose(first.beta, second.beta)
    np.testing.assert_allclose(first.sigma, second.sigma)
    np.testing.assert_allclose(first.beta_point, second.beta_point)
    np.testing.assert_allclose(first.sigma_point, second.sigma_point)


def test_natural_conjugate_keeps_analytical_point_estimates():
    """Direct samplers continue to expose their analytical point estimates."""
    data, _, _, _ = simulate_var(100, 2, 1, seed=321)
    model_a = NaturalConjugate(minnesota=True, soc=True, sur=True, covid=False)
    model_b = NaturalConjugate(minnesota=True, soc=True, sur=True, covid=False)
    bvar_a = BVAR(1, model_a, True, optimisation_method="none")
    bvar_b = BVAR(1, model_b, True, optimisation_method="none")

    bvar_a.sample(data, N_draws=4, N_burn=0, progressbar=False, random_state=321)
    bvar_b.sample(data, N_draws=4, N_burn=3, progressbar=False, random_state=321)

    np.testing.assert_allclose(bvar_a.beta_point, bvar_b.beta_point)
    np.testing.assert_allclose(bvar_a.sigma_point, bvar_b.sigma_point)


def test_bvar_aligns_extras_with_retained_burnin_draws():
    """Burn-in slicing keeps optional per-draw extras aligned."""
    data, _, _, _ = simulate_var(20, 2, 1, seed=321)
    model = _BurninExtrasModel(minnesota=False, soc=False, sur=False, covid=False)
    bvar = BVAR(1, model, True, optimisation_method="none")

    bvar.sample(data, N_draws=3, N_burn=2, progressbar=False)

    assert bvar.extras == [{"draw": 2}, {"draw": 3}, {"draw": 4}]
    assert bvar.posterior_state_point.extras == {"summary": True}
    np.testing.assert_allclose(bvar.beta_point, bvar.beta.mean(axis=0))
    np.testing.assert_allclose(bvar.sigma_point, bvar.sigma.mean(axis=0))


@pytest.mark.parametrize("invalid_draws", [0, -1, 1.5, True])
def test_bvar_sample_rejects_invalid_N_draws(invalid_draws):
    """Retained posterior draws must be positive, non-boolean integers."""
    data, _, _, _ = simulate_var(20, 2, 1, seed=321)
    model = NaturalConjugate(minnesota=True, soc=False, sur=False, covid=False)
    bvar = BVAR(1, model, True, optimisation_method="none")

    with pytest.raises(ValueError, match="N_draws"):
        bvar.sample(data, N_draws=invalid_draws, progressbar=False)


@pytest.mark.parametrize("invalid_burn", [-1, 1.5, True])
def test_bvar_sample_rejects_invalid_N_burn(invalid_burn):
    """Burn-in must be a non-negative, non-boolean integer."""
    data, _, _, _ = simulate_var(20, 2, 1, seed=321)
    model = IndependentNIW(minnesota=True, soc=False, sur=False, covid=False)
    bvar = BVAR(1, model, True, optimisation_method="none")

    with pytest.raises(ValueError, match="N_burn"):
        bvar.sample(data, N_draws=4, N_burn=invalid_burn, progressbar=False)


def test_bvar_sample_rejects_burn_in_equal_to_requested_draws():
    """Burn-in must leave at least one effective retained draw."""
    data, _, _, _ = simulate_var(20, 2, 1, seed=321)
    model = IndependentNIW(minnesota=True, soc=False, sur=False, covid=False)
    bvar = BVAR(1, model, True, optimisation_method="none")

    with pytest.raises(ValueError, match="N_burn"):
        bvar.sample(data, N_draws=4, N_burn=4, progressbar=False)


def test_failed_sample_refit_preserves_model_and_fitted_state():
    """A sampler exception cannot partially replace a previous fit."""
    data, _, _, _ = simulate_var(40, 2, 1, seed=321)
    model = _FailingSampleModel(minnesota=True, soc=False, sur=False, covid=False)
    bvar = BVAR(1, model, True, optimisation_method="none")
    bvar.sample(data, N_draws=4, progressbar=False, random_state=321)

    beta_before = bvar.beta.copy()
    sigma_before = bvar.sigma.copy()
    beta_point_before = bvar.beta_point.copy()
    sigma_point_before = bvar.sigma_point.copy()
    posterior_state_before = bvar.posterior_state_point
    posterior_beta_before = posterior_state_before.beta.copy()
    posterior_sigma_before = posterior_state_before.sigma.copy()
    df_data_before = bvar.df_data.copy()
    model_before = bvar.model
    pars_before = model_before.pars.__dict__.copy()
    beta_0_before = bvar.model.beta_0.copy()
    V_A_inv_before = bvar.model.V_A_inv.copy()
    rng_before = copy.deepcopy(bvar.rng.bit_generator.state)
    bvar.model.fail = True

    with pytest.raises(RuntimeError, match="synthetic"):
        bvar.sample(data.iloc[1:].copy(), N_draws=4, progressbar=False)

    np.testing.assert_array_equal(bvar.beta, beta_before)
    np.testing.assert_array_equal(bvar.sigma, sigma_before)
    np.testing.assert_array_equal(bvar.beta_point, beta_point_before)
    np.testing.assert_array_equal(bvar.sigma_point, sigma_point_before)
    assert bvar.posterior_state_point is posterior_state_before
    np.testing.assert_array_equal(
        bvar.posterior_state_point.beta, posterior_beta_before
    )
    np.testing.assert_array_equal(
        bvar.posterior_state_point.sigma, posterior_sigma_before
    )
    pd.testing.assert_frame_equal(bvar.df_data, df_data_before)
    assert bvar.model is model_before
    assert bvar.model.pars.__dict__.keys() == pars_before.keys()
    for name, value in pars_before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(bvar.model.pars, name), value)
        else:
            assert getattr(bvar.model.pars, name) == value
    np.testing.assert_array_equal(bvar.model.beta_0, beta_0_before)
    np.testing.assert_array_equal(bvar.model.V_A_inv, V_A_inv_before)
    np.testing.assert_equal(bvar.rng.bit_generator.state, rng_before)


@pytest.mark.parametrize("model_cls", [NaturalConjugate, IndependentNIW])
def test_bvar_sample_rejects_invalid_N_draws_on_fitted_object(model_cls):
    """Invalid draw counts fail without changing an existing fit."""
    data, _, _, _ = simulate_var(20, 2, 1, seed=322)
    model = model_cls(minnesota=True, soc=False, sur=False, covid=False)
    bvar = BVAR(1, model, True, optimisation_method="none")
    bvar.sample(data, N_draws=2, progressbar=False, random_state=322)

    beta_before = bvar.beta.copy()
    sigma_before = bvar.sigma.copy()
    beta_point_before = bvar.beta_point.copy()
    sigma_point_before = bvar.sigma_point.copy()
    for invalid_draws in [0, -1, 1.5, True]:
        with pytest.raises(ValueError, match="N_draws"):
            bvar.sample(data, N_draws=invalid_draws, progressbar=False)

        np.testing.assert_array_equal(bvar.beta, beta_before)
        np.testing.assert_array_equal(bvar.sigma, sigma_before)
        np.testing.assert_array_equal(bvar.beta_point, beta_point_before)
        np.testing.assert_array_equal(bvar.sigma_point, sigma_point_before)


def test_independent_niw_conditional_forecast_end_to_end():
    """An Independent-NIW conditional forecast must exercise the real Gibbs
    forecast loop and return finite, non-degenerate retained draws."""
    H = 3
    bvar = _fitted_independent_niw_bvar()

    bvar.forecast(H=H, point_only=True)
    uncond = bvar.forecast_unconditional[0, bvar.T :, :]

    constraint_mean = np.full((H, bvar.n), np.nan)
    constraint_mean[:, 0] = uncond[:, 0]

    bvar.forecast(
        H=H,
        constraint_mean=constraint_mean,
        N_draws=20,
        point_only=False,
        progressbar=False,
        random_state=321,
    )

    forecast_conditional = bvar.forecast_conditional
    assert np.all(np.isfinite(forecast_conditional))

    # The constrained variable should track the target closely.
    cond_mean = forecast_conditional[:, bvar.T :, 0].mean(axis=0)
    assert np.abs(cond_mean - uncond[:, 0]).max() < 0.5

    # Retained draws for the unconstrained variable must not be degenerate:
    # a zero-variance chain would indicate sample_posterior_state is not
    # sampling.
    cond_var = np.var(forecast_conditional[:, bvar.T :, 1], axis=0)
    assert np.all(cond_var > 0)


def test_independent_niw_conditional_forecast_carries_state_forward(monkeypatch):
    """The real conditional-forecast loop must feed each Gibbs draw's
    returned covariance forward as the next call's ``current_state.sigma``,
    verified by wrapping (not replacing) the real
    ``sample_posterior_state``."""
    H = 2
    bvar = _fitted_independent_niw_bvar()

    constraint_mean = np.full((H, bvar.n), np.nan)
    constraint_mean[:, 0] = 0.0

    calls = []
    returns = []
    original_sample_posterior_state = bvar.model.sample_posterior_state

    def spy(Y, Z, current_state, rng=None):
        calls.append((current_state.beta.copy(), current_state.sigma.copy()))
        next_state = original_sample_posterior_state(Y, Z, current_state, rng=rng)
        returns.append((next_state.beta.copy(), next_state.sigma.copy()))
        return next_state

    monkeypatch.setattr(bvar.model, "sample_posterior_state", spy)

    N_draws = 6
    bvar.forecast(
        H=H,
        constraint_mean=constraint_mean,
        N_draws=N_draws,
        point_only=False,
        progressbar=False,
        random_state=321,
    )

    assert len(calls) == N_draws

    # The first call must seed from the model's own point estimate.
    np.testing.assert_allclose(calls[0][1], bvar.sigma_point)

    # Each later call must carry forward the previous call's returned sigma.
    for i in range(1, N_draws):
        np.testing.assert_allclose(calls[i][1], returns[i - 1][1])


def test_independent_niw_hyperparameter_grid_and_vector():
    m = bv.IndependentNIW(minnesota=True, soc=False, sur=False)
    grid = m.hyperparameter_grid()
    assert len(grid) >= 1
    v = np.array([np.asarray(g).ravel()[0] for g in grid], dtype=float)
    m.fill_in_from_vector(v)
