"""Tests for the extensible posterior-state carrier.

Covers ``PosteriorState`` construction/copy semantics, the
``sample_posterior_state`` contract (the sole posterior-update extension
point), extras preservation via a model that carries auxiliary state, and
that the existing NaturalConjugate / IndependentNIW samplers satisfy it.
"""

import threading

import numpy as np
import pytest

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


class _FakeSamplingModel(SamplingModel):
    """Minimal concrete model double for state-contract tests.

    ``sample`` is never exercised here; ``sample_posterior_state`` returns a
    deterministic shift so the tests can verify the ``PosteriorState`` contract
    without a real estimation pipeline.
    """

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def sample_posterior_state(self, Y, Z, current_state, rng=None):
        return PosteriorState(
            beta=current_state.beta + 1.0, sigma=current_state.sigma + 1.0
        )


class _ExtrasCarryingModel(_FakeSamplingModel):
    """Overrides the shift to also thread an ``extras`` payload forward."""

    def sample_posterior_state(self, Y, Z, current_state, rng=None):
        next_state = super().sample_posterior_state(Y, Z, current_state, rng=rng)
        n_calls = (current_state.extras or {}).get("n_calls", 0) + 1
        return PosteriorState(
            beta=next_state.beta, sigma=next_state.sigma, extras={"n_calls": n_calls}
        )


def test_posterior_state_construction_and_access():
    beta = np.array([1.0, 2.0])
    sigma = np.array([3.0, 4.0])
    extras = {"df": 5.0}

    state = PosteriorState(beta=beta, sigma=sigma, extras=extras)

    assert state.beta is beta
    assert state.sigma is sigma
    assert state.extras is extras


def test_posterior_state_extras_defaults_to_none():
    state = PosteriorState(beta=np.zeros(2), sigma=np.zeros(2))

    assert state.extras is None


def test_posterior_state_copy_returns_independent_beta_and_sigma():
    beta = np.array([1.0, 2.0])
    sigma = np.array([3.0, 4.0])
    state = PosteriorState(beta=beta, sigma=sigma)

    copied = state.copy()
    copied.beta[0] = 999.0
    copied.sigma[0] = 999.0

    assert copied.beta is not beta
    assert copied.sigma is not sigma
    np.testing.assert_allclose(beta, np.array([1.0, 2.0]))
    np.testing.assert_allclose(sigma, np.array([3.0, 4.0]))


def test_posterior_state_copy_deep_copies_nested_extras():
    history = []
    extras = {"history": history}
    state = PosteriorState(beta=np.zeros(2), sigma=np.zeros(2), extras=extras)

    copied = state.copy()
    copied.extras["history"].append(1)

    # The top-level container is independent.
    assert copied.extras is not extras
    # The nested mutable state is independent too.
    assert history == []


def test_posterior_state_copy_raises_clear_error_for_uncopyable_extras():
    state = PosteriorState(beta=np.zeros(2), sigma=np.zeros(2), extras=threading.Lock())

    with pytest.raises(TypeError, match="deep-copied"):
        state.copy()


def test_posterior_state_copy_preserves_none_extras():
    state = PosteriorState(beta=np.zeros(2), sigma=np.zeros(2))

    assert state.copy().extras is None


def test_fake_sampling_model_sample_posterior_state_returns_shifted_state():
    model = _FakeSamplingModel()
    current_state = PosteriorState(beta=np.zeros(2), sigma=np.ones(2))

    next_state = model.sample_posterior_state(None, None, current_state)

    assert isinstance(next_state, PosteriorState)
    np.testing.assert_allclose(next_state.beta, np.ones(2))
    np.testing.assert_allclose(next_state.sigma, 2 * np.ones(2))
    assert next_state.extras is None


def test_overridden_sample_posterior_state_preserves_extras():
    model = _ExtrasCarryingModel()
    state_0 = PosteriorState(beta=np.zeros(2), sigma=np.ones(2), extras=None)

    state_1 = model.sample_posterior_state(None, None, state_0)
    state_2 = model.sample_posterior_state(None, None, state_1)

    assert state_1.extras == {"n_calls": 1}
    assert state_2.extras == {"n_calls": 2}
    np.testing.assert_allclose(state_2.beta, 2 * np.ones(2))


def test_natural_conjugate_sample_posterior_state_is_deterministic_given_rng():
    model, Y, Z = _fitted_model_and_matrices(model_cls=NaturalConjugate)
    current_state = PosteriorState(
        beta=np.zeros_like(model.beta_0), sigma=model.pars.S_0.flatten()
    )

    state_a = model.sample_posterior_state(
        Y, Z, current_state, rng=np.random.default_rng(0)
    )
    state_b = model.sample_posterior_state(
        Y, Z, current_state, rng=np.random.default_rng(0)
    )

    np.testing.assert_allclose(state_a.beta, state_b.beta)
    np.testing.assert_allclose(state_a.sigma, state_b.sigma)
    assert state_a.extras is None


def test_independent_niw_sample_posterior_state_is_deterministic_given_rng():
    model, Y, Z = _fitted_model_and_matrices(model_cls=IndependentNIW)
    current_state = PosteriorState(
        beta=np.zeros_like(model.beta_0), sigma=model.pars.S_0.flatten()
    )

    state_a = model.sample_posterior_state(
        Y, Z, current_state, rng=np.random.default_rng(3)
    )
    state_b = model.sample_posterior_state(
        Y, Z, current_state, rng=np.random.default_rng(3)
    )

    np.testing.assert_allclose(state_a.beta, state_b.beta)
    np.testing.assert_allclose(state_a.sigma, state_b.sigma)
    assert state_a.extras is None


def test_sampling_result_state_point_reflects_beta_sigma_point():
    T, n, n_lags = 100, 2, 1
    data, _, _, _ = simulate_var(T, n, n_lags, seed=1234)
    data = data.to_numpy()
    covid_indices = np.array([], dtype=int)
    vars_in_levels = np.zeros(n, dtype=bool)

    model = NaturalConjugate(minnesota=True, soc=True, sur=True, covid=False)
    result = model.sample(
        data,
        n_lags,
        covid_indices,
        vars_in_levels,
        N_draws=1,
        point_only=True,
        progressbar=False,
    )

    state_point = result.state_point

    assert isinstance(state_point, PosteriorState)
    np.testing.assert_allclose(state_point.beta, result.beta_point)
    np.testing.assert_allclose(state_point.sigma, result.sigma_point)
    assert state_point.extras is None


def test_sampling_result_extras_point_defaults_to_none():
    result = SamplingResult(
        beta_draws=np.zeros((1, 2)),
        sigma_draws=np.zeros((1, 2)),
        beta_point=np.zeros(2),
        sigma_point=np.zeros(2),
    )

    assert result.extras_point is None
    assert result.state_point.extras is None


def test_bvar_stores_posterior_state_point_after_sample(bvar):
    assert isinstance(bvar.posterior_state_point, PosteriorState)
    np.testing.assert_allclose(bvar.posterior_state_point.beta, bvar.beta_point)
    np.testing.assert_allclose(bvar.posterior_state_point.sigma, bvar.sigma_point)
    assert bvar.posterior_state_point.extras is None
    # posterior_state_point is a lightweight summary: it reuses beta_point/
    # sigma_point rather than duplicating them.
    assert bvar.posterior_state_point.beta is bvar.beta_point
    assert bvar.posterior_state_point.sigma is bvar.sigma_point
