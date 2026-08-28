"""Tests for ``GridSearch.marginal_likelihood_H`` (code-review fixes).

Covers:
* The rolling out-of-sample predictive log density keeps the full target
    covariance submatrix and its correlations.
  predictive log density for correlated multi-target series.
* Calling ``marginal_likelihood_H`` with the default ``target_indices=None``
  and ``rolling_oos_starting_id=None`` no longer raises ``TypeError``.
* One shared ``numpy.random.Generator`` serves every rolling-window
    ``sample``/``forecast`` call in
  ``marginal_likelihood_H``, and across grid points and rolling windows in
  ``grid_search``.
"""

import copy

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal

from bvar.BVAR import BVAR
from bvar.models import NaturalConjugate
from bvar.utils import simulate_var


def _make_validation_bvar_and_data(T=20):
    """Build an unfitted BVAR for argument-validation tests."""
    data, _, _, _ = simulate_var(T, 2, 1, seed=123)
    model = NaturalConjugate(minnesota=True, soc=False, sur=False, covid=False)
    return BVAR(1, model, stationary=True, optimisation_method="none"), data


class _SampleCountingBVAR(BVAR):
    sample_calls = 0

    def sample(self, *args, **kwargs):
        type(self).sample_calls += 1
        return super().sample(*args, **kwargs)


class _SampleDataRecordingBVAR(BVAR):
    sampled_data = []

    def sample(self, data, *args, **kwargs):
        type(self).sampled_data.append(data.copy())
        return super().sample(data, *args, **kwargs)


class _RollingEvaluationCountingBVAR(BVAR):
    marginal_likelihood_calls = 0

    def marginal_likelihood_H(self, *args, **kwargs):
        type(self).marginal_likelihood_calls += 1
        return super().marginal_likelihood_H(*args, **kwargs)


def test_marginal_likelihood_H_preserves_full_target_covariance():
    """The predictive density must use the full off-diagonal covariance
    submatrix for the target variables, not just their diagonal variances."""
    n, n_lags, H = 2, 1, 1
    ar_mat = np.diag([0.4, 0.3])
    Sigma = np.array([[1.0, 0.7], [0.7, 1.5]])
    df_data, _, _, _ = simulate_var(40, n, n_lags, ar_mat=ar_mat, Sigma=Sigma, seed=0)
    data = df_data.to_numpy()

    priors = NaturalConjugate(minnesota=True, soc=False, sur=False, covid=False)
    bvar = BVAR(n_lags, priors, stationary=True, optimisation_method="none")
    bvar.sample(df_data, N_draws=10, progressbar=False)

    target_indices = [0, 1]
    p = bvar.n_lags
    # Exactly one rolling-window iteration.
    rolling_oos_starting_id = data.shape[0] - p - H

    ml = bvar.marginal_likelihood_H(data, H, target_indices, rolling_oos_starting_id)

    # Independently replicate the single rolling-window step to obtain the
    # predictive mean/covariance directly.
    t = rolling_oos_starting_id
    replica = copy.deepcopy(bvar)
    replica.data = data[: t + p, :]
    replica.sample(bvar.df_data.iloc[: t + p], progressbar=False, point_only=True)
    replica.forecast(H=H, progressbar=False, point_only=True)

    mean = replica.mean_H[0, target_indices]
    full_covariance = replica.variance_H[0][np.ix_(target_indices, target_indices)]
    target = data[t + p + H - 1, target_indices]

    expected = multivariate_normal.logpdf(target, mean, full_covariance)
    assert ml == pytest.approx(expected)

    # The previous (buggy) implementation collapsed the submatrix to its
    # diagonal, dropping the cross-target covariance -- for correlated
    # targets this must give a different (wrong) log density.
    diagonal_only_covariance = np.diag(full_covariance)
    wrong = multivariate_normal.logpdf(target, mean, diagonal_only_covariance)
    assert ml != pytest.approx(wrong)


def test_marginal_likelihood_H_refits_from_supplied_training_history():
    """Rolling refits must use the caller's data, including altered history."""
    _SampleDataRecordingBVAR.sampled_data.clear()

    df_data, _, _, _ = simulate_var(30, 2, 1, seed=16)
    supplied_data = df_data.to_numpy().copy()
    supplied_data[0, 0] += 10.0
    held_out_target = supplied_data[-1].copy()

    model = NaturalConjugate(minnesota=True, soc=False, sur=False, covid=False)
    bvar = _SampleDataRecordingBVAR(
        1, model, stationary=True, optimisation_method="none"
    )
    bvar.sample(df_data, N_draws=10, progressbar=False)
    _SampleDataRecordingBVAR.sampled_data.clear()

    bvar.marginal_likelihood_H(
        supplied_data,
        H=1,
        target_indices=[0, 1],
        rolling_oos_starting_id=supplied_data.shape[0] - bvar.n_lags - 1,
    )

    assert np.array_equal(held_out_target, supplied_data[-1])
    assert len(_SampleDataRecordingBVAR.sampled_data) == 1
    sampled_data = _SampleDataRecordingBVAR.sampled_data[0]
    np.testing.assert_array_equal(sampled_data.to_numpy(), supplied_data[:-1])
    pd.testing.assert_index_equal(sampled_data.index, df_data.index[:-1])
    pd.testing.assert_index_equal(sampled_data.columns, df_data.columns)


def test_marginal_likelihood_H_defaults_do_not_raise_type_error():
    """``target_indices=None`` and ``rolling_oos_starting_id=None`` should
    fall back to sensible defaults (all variables, a single out-of-sample
    point at the end of the sample) instead of raising ``TypeError``."""
    n, n_lags, H = 2, 1, 1
    df_data, _, _, _ = simulate_var(30, n, n_lags, seed=1)
    data = df_data.to_numpy()

    priors = NaturalConjugate(minnesota=True, soc=False, sur=False, covid=False)
    bvar = BVAR(n_lags, priors, stationary=True, optimisation_method="none")
    bvar.sample(df_data, N_draws=10, progressbar=False)

    ml = bvar.marginal_likelihood_H(data, H)

    assert isinstance(ml, float)
    assert np.isfinite(ml)


@pytest.mark.parametrize("invalid_horizon", [True, 0, -1, 1.5])
def test_marginal_likelihood_H_rejects_invalid_horizon(invalid_horizon):
    """Direct marginal likelihood evaluation validates its horizon."""
    bvar, data = _make_validation_bvar_and_data()

    with pytest.raises(ValueError, match="H must be a positive integer"):
        bvar.marginal_likelihood_H(data.to_numpy(), invalid_horizon)


@pytest.mark.parametrize(
    "invalid_target_indices",
    [[], [True], [2], [-1], [0.5]],
)
def test_marginal_likelihood_H_rejects_invalid_target_indices(invalid_target_indices):
    """Direct marginal likelihood evaluation validates target indices."""
    bvar, data = _make_validation_bvar_and_data()

    with pytest.raises(ValueError, match="target_indices"):
        bvar.marginal_likelihood_H(data.to_numpy(), 1, invalid_target_indices)


@pytest.mark.parametrize("invalid_start", [True, 1.5, 0, -1, 20])
def test_marginal_likelihood_H_rejects_invalid_rolling_window_start(invalid_start):
    """Invalid rolling starts cannot silently produce an empty likelihood."""
    bvar, data = _make_validation_bvar_and_data()

    with pytest.raises(ValueError, match="rolling_oos_starting_id"):
        bvar.marginal_likelihood_H(
            data.to_numpy(), 1, rolling_oos_starting_id=invalid_start
        )


def test_marginal_likelihood_H_rejects_first_window_without_ar1_initialisation():
    """The first rolling training sample must support AR(1) initialisation."""
    df_data, _, _, _ = simulate_var(4, 2, 1, seed=13)
    model = NaturalConjugate(minnesota=True, soc=False, sur=False, covid=False)
    bvar = _SampleCountingBVAR(1, model, stationary=True, optimisation_method="none")
    bvar.sample(df_data, N_draws=4, progressbar=False)
    _SampleCountingBVAR.sample_calls = 0

    with pytest.raises(ValueError, match=r"AR\(1\)|residual variance"):
        bvar.marginal_likelihood_H(
            df_data.to_numpy(), 1, [0, 1], rolling_oos_starting_id=1
        )
    assert _SampleCountingBVAR.sample_calls == 0


def test_marginal_likelihood_H_counts_usable_covid_regressors_in_first_window():
    """Usable COVID columns count towards the first AR(1) sample requirement."""
    df_data, _, _, _ = simulate_var(7, 2, 2, seed=14)
    model = NaturalConjugate(
        minnesota=True,
        soc=False,
        sur=False,
        covid=True,
        covid_dates=["1990Q2", "1990Q2"],
    )
    bvar = BVAR(2, model, stationary=True, optimisation_method="none")
    bvar.sample(df_data, N_draws=4, progressbar=False)

    with pytest.raises(ValueError, match=r"AR\(1\)|residual variance"):
        bvar.marginal_likelihood_H(
            df_data.to_numpy(), 1, [0, 1], rolling_oos_starting_id=2
        )


def test_grid_search_rejects_first_window_without_ar1_initialisation():
    """Grid search validates AR(1) initialisation before evaluating a grid."""
    df_data, _, _, _ = simulate_var(4, 2, 1, seed=15)
    model = NaturalConjugate(minnesota=True, soc=False, sur=False, covid=False)
    bvar = _RollingEvaluationCountingBVAR(
        1, model, stationary=True, optimisation_method="none"
    )
    bvar.sample(df_data, N_draws=4, progressbar=False)
    _RollingEvaluationCountingBVAR.marginal_likelihood_calls = 0

    with pytest.raises(ValueError, match=r"AR\(1\)|residual variance"):
        bvar.grid_search(
            df_data.to_numpy(), cv_options={"H": 1, "oos_test_window_size": 1}
        )
    assert _RollingEvaluationCountingBVAR.marginal_likelihood_calls == 0


class _MutatingPredictiveLogpdfModel(NaturalConjugate):
    """Model whose ``predictive_logpdf`` mutates its ``state`` argument in
    place and records array identities on class-level lists (rather than
    instance attributes) so evidence of aliasing survives
    ``GridSearch.marginal_likelihood_H``'s internal ``copy.deepcopy`` of the
    whole ``BVAR``/model.
    """

    sampled_beta_point_ids: list = []
    sampled_sigma_point_ids: list = []
    received_state_beta_ids: list = []
    received_state_sigma_ids: list = []

    def sample(self, *args, **kwargs):
        result = super().sample(*args, **kwargs)
        type(self).sampled_beta_point_ids.append(id(result.beta_point))
        type(self).sampled_sigma_point_ids.append(id(result.sigma_point))
        return result

    def predictive_logpdf(self, state, observation, mean, covariance):
        type(self).received_state_beta_ids.append(id(state.beta))
        type(self).received_state_sigma_ids.append(id(state.sigma))
        state.beta += 1000.0
        state.sigma += 1000.0
        return super().predictive_logpdf(state, observation, mean, covariance)


def test_marginal_likelihood_H_isolates_state_from_mutating_predictive_logpdf():
    """``predictive_logpdf`` must receive an isolated copy of
    ``bvar.posterior_state_point``, never the point-estimate arrays themselves --
    otherwise a mutating override would corrupt the fitted
    ``beta_point``/``sigma_point`` relied on by later rolling-window
    iterations."""
    _MutatingPredictiveLogpdfModel.sampled_beta_point_ids.clear()
    _MutatingPredictiveLogpdfModel.sampled_sigma_point_ids.clear()
    _MutatingPredictiveLogpdfModel.received_state_beta_ids.clear()
    _MutatingPredictiveLogpdfModel.received_state_sigma_ids.clear()

    n, n_lags, H = 2, 1, 1
    df_data, _, _, _ = simulate_var(35, n, n_lags, seed=2)
    data = df_data.to_numpy()

    model = _MutatingPredictiveLogpdfModel(
        minnesota=True, soc=False, sur=False, covid=False
    )
    bvar = BVAR(n_lags, model, stationary=True, optimisation_method="none")
    bvar.sample(df_data, N_draws=10, progressbar=False)

    p = bvar.n_lags
    # Several rolling-window iterations, to exercise repeated mutation. Only
    # refits inside marginal_likelihood_H are of interest, so clear the
    # initial bvar.sample() call recorded above.
    _MutatingPredictiveLogpdfModel.sampled_beta_point_ids.clear()
    _MutatingPredictiveLogpdfModel.sampled_sigma_point_ids.clear()
    rolling_oos_starting_id = data.shape[0] - p - H - 2

    bvar.marginal_likelihood_H(data, H, [0, 1], rolling_oos_starting_id)

    ids = _MutatingPredictiveLogpdfModel
    assert len(ids.received_state_beta_ids) == len(ids.sampled_beta_point_ids)
    assert len(ids.received_state_beta_ids) > 1
    for received_id, fitted_id in zip(
        ids.received_state_beta_ids, ids.sampled_beta_point_ids
    ):
        assert received_id != fitted_id
    for received_id, fitted_id in zip(
        ids.received_state_sigma_ids, ids.sampled_sigma_point_ids
    ):
        assert received_id != fitted_id


class _RngRecordingModel(NaturalConjugate):
    """Records the identity of the ``rng`` passed to ``sample()`` on every
    call (on a class-level list, since ``GridSearch`` deep-copies the whole
    ``BVAR``/model for each rolling-window refit), so tests can verify a
    one shared generator serves every call instead of restarting."""

    sample_rng_ids: list = []

    def sample(self, *args, **kwargs):
        type(self).sample_rng_ids.append(id(kwargs.get("rng")))
        return super().sample(*args, **kwargs)


class _SmallGridRngRecordingModel(_RngRecordingModel):
    """As ``_RngRecordingModel``, with a two-point hyperparameter grid so
    ``grid_search`` regression tests stay fast."""

    def hyperparameter_grid(self):
        return [np.array([0.2]), np.array([2.0, 3.0])]


class _FailingOptimisationModel(NaturalConjugate):
    """Mutate a candidate prior and fail during grid-search evaluation."""

    def hyperparameter_grid(self):
        return [np.array([0.2]), np.array([2.0])]

    def fill_in_from_vector(self, pars):
        super().fill_in_from_vector(pars)
        self.pars.c1 = 999.0
        raise RuntimeError("synthetic optimisation failure")


class _FailingGridSearchModel(NaturalConjugate):
    """Provide a small grid for the direct rollback regression."""

    def hyperparameter_grid(self):
        return [np.array([0.2]), np.array([2.0])]


class _FailingGridSearchBVAR(BVAR):
    """Fail after a real staged refit has replaced fitted state."""

    def marginal_likelihood_H(
        self,
        data,
        H,
        target_indices=None,
        rolling_oos_starting_id=None,
        random_state=None,
    ):
        self.sample(self.df_data, N_draws=2, progressbar=False)
        self.model.pars.c1 = 999.0
        self.rng.normal()
        raise RuntimeError("synthetic grid-search failure")


def test_marginal_likelihood_H_reuses_shared_generator_across_rolling_windows():
    """A single shared generator (``bvar.rng``) must be reused for every
    rolling-window ``sample()`` call inside ``marginal_likelihood_H``, not a
    fresh copy restarted at each window."""
    _RngRecordingModel.sample_rng_ids.clear()

    n, n_lags, H = 2, 1, 1
    df_data, _, _, _ = simulate_var(40, n, n_lags, seed=3)
    data = df_data.to_numpy()

    model = _RngRecordingModel(minnesota=True, soc=False, sur=False, covid=False)
    bvar = BVAR(n_lags, model, stationary=True, optimisation_method="none")
    bvar.sample(df_data, N_draws=10, progressbar=False)

    p = bvar.n_lags
    _RngRecordingModel.sample_rng_ids.clear()
    rolling_oos_starting_id = data.shape[0] - p - H - 2

    bvar.marginal_likelihood_H(data, H, [0, 1], rolling_oos_starting_id)

    ids = _RngRecordingModel.sample_rng_ids
    assert len(ids) == 3
    assert len(set(ids)) == 1
    assert ids[0] == id(bvar.rng)


@pytest.mark.parametrize(
    "random_state_factory",
    [lambda: 42, lambda: np.random.default_rng(42)],
    ids=["int_seed", "generator_instance"],
)
def test_marginal_likelihood_H_reuses_explicit_random_state(random_state_factory):
    """An explicit ``random_state`` (int seed or ``numpy.random.Generator``)
    must be honoured and reused across all rolling windows, consistent with
    the ``random_state`` convention used elsewhere in the BVAR API."""
    _RngRecordingModel.sample_rng_ids.clear()

    n, n_lags, H = 2, 1, 1
    df_data, _, _, _ = simulate_var(40, n, n_lags, seed=4)
    data = df_data.to_numpy()

    model = _RngRecordingModel(minnesota=True, soc=False, sur=False, covid=False)
    bvar = BVAR(n_lags, model, stationary=True, optimisation_method="none")
    bvar.sample(df_data, N_draws=10, progressbar=False)

    p = bvar.n_lags
    _RngRecordingModel.sample_rng_ids.clear()
    rolling_oos_starting_id = data.shape[0] - p - H - 2
    random_state = random_state_factory()

    bvar.marginal_likelihood_H(
        data, H, [0, 1], rolling_oos_starting_id, random_state=random_state
    )

    ids = _RngRecordingModel.sample_rng_ids
    assert len(ids) == 3
    assert len(set(ids)) == 1
    if isinstance(random_state, np.random.Generator):
        assert ids[0] == id(random_state)


def test_grid_search_reuses_shared_generator_by_default():
    """``grid_search`` with the default ``random_state=None`` must reuse
    ``bvar.rng`` itself across every grid point and rolling window, not
    restart from a fresh copy each time."""
    _SmallGridRngRecordingModel.sample_rng_ids.clear()

    n, n_lags, H = 2, 1, 1
    df_data, _, _, _ = simulate_var(40, n, n_lags, seed=5)
    data = df_data.to_numpy()

    model = _SmallGridRngRecordingModel(
        minnesota=True, soc=False, sur=False, covid=False
    )
    bvar = BVAR(n_lags, model, stationary=True, optimisation_method="none")
    bvar.sample(df_data, N_draws=10, progressbar=False)

    _SmallGridRngRecordingModel.sample_rng_ids.clear()
    cv_options = {"H": H, "oos_test_window_size": 2}

    bvar.grid_search(data, cv_options=cv_options)

    ids = _SmallGridRngRecordingModel.sample_rng_ids
    # 2 grid points x 3 rolling windows each.
    assert len(ids) == 6
    assert set(ids) == {id(bvar.rng)}
    assert bvar.target_indices == [0, 1]


def test_grid_search_valid_seeded_run_is_deterministic_with_default_targets():
    """A valid seeded grid search gives the same selected parameters each run."""
    df_data, _, _, _ = simulate_var(30, 2, 1, seed=12)
    cv_options = {"H": 1, "oos_test_window_size": 1}

    results = []
    for _ in range(2):
        model = _SmallGridRngRecordingModel(
            minnesota=True, soc=False, sur=False, covid=False
        )
        bvar = BVAR(1, model, stationary=True, optimisation_method="none")
        bvar.sample(df_data, N_draws=4, progressbar=False, random_state=12)
        bvar.grid_search(
            df_data.to_numpy(),
            cv_options=cv_options,
            random_state=99,
            progressbar=False,
        )
        results.append((bvar.model.to_vector().copy(), bvar.target_indices.copy()))

    np.testing.assert_array_equal(results[0][0], results[1][0])
    assert results[0][1] == results[1][1] == [0, 1]


@pytest.mark.parametrize(
    "random_state_factory",
    [lambda: 123, lambda: np.random.default_rng(123)],
    ids=["int_seed", "generator_instance"],
)
def test_grid_search_reuses_explicit_random_state_across_grid_points(
    random_state_factory,
):
    """An explicit ``random_state`` (int seed or ``numpy.random.Generator``)
    passed to ``grid_search`` must be honoured and reused (never restarted)
    across every grid point and rolling window."""
    _SmallGridRngRecordingModel.sample_rng_ids.clear()

    n, n_lags, H = 2, 1, 1
    df_data, _, _, _ = simulate_var(40, n, n_lags, seed=6)
    data = df_data.to_numpy()

    model = _SmallGridRngRecordingModel(
        minnesota=True, soc=False, sur=False, covid=False
    )
    bvar = BVAR(n_lags, model, stationary=True, optimisation_method="none")
    bvar.sample(df_data, N_draws=10, progressbar=False)

    _SmallGridRngRecordingModel.sample_rng_ids.clear()
    cv_options = {"H": H, "oos_test_window_size": 2}
    random_state = random_state_factory()

    bvar.grid_search(data, cv_options=cv_options, random_state=random_state)

    ids = _SmallGridRngRecordingModel.sample_rng_ids
    assert len(ids) == 6
    assert len(set(ids)) == 1
    if isinstance(random_state, np.random.Generator):
        assert ids[0] == id(random_state)


def test_optimise_hyperparameters_forwards_random_state_to_grid_search():
    """``BVAR.optimise_hyperparameters`` must forward its ``random_state``
    argument into ``grid_search`` in the ``cross_validation`` branch, rather
    than silently falling back to ``self.rng``."""
    _SmallGridRngRecordingModel.sample_rng_ids.clear()

    n, n_lags, H = 2, 1, 1
    df_data, _, _, _ = simulate_var(40, n, n_lags, seed=7)

    model = _SmallGridRngRecordingModel(
        minnesota=True, soc=False, sur=False, covid=False
    )
    bvar = BVAR(n_lags, model, stationary=True, optimisation_method="cross_validation")

    explicit_rng = np.random.default_rng(999)
    cv_options = {"H": H, "oos_test_window_size": 2}

    bvar.optimise_hyperparameters(
        df_data, cv_options=cv_options, random_state=explicit_rng
    )

    ids = _SmallGridRngRecordingModel.sample_rng_ids
    # 2 grid points x 3 rolling windows each.
    assert len(ids) == 6
    assert set(ids) == {id(explicit_rng)}


def test_optimise_hyperparameters_cross_validation_reproducible_with_seed():
    """Two equivalent ``cross_validation`` optimisation runs (via
    ``BVAR.optimise_hyperparameters``) with the same ``random_state`` must
    produce identical optimised prior parameters.

    ``point_only`` cross-validation fits are themselves deterministic, so the
    optimised-parameter equality below would hold even if ``random_state``
    were silently ignored. The generator-identity assertions pin down actual
    forwarding: the recorded ``sample()`` generator must be the one resolved
    from the explicit ``random_state``, never the instance's own ``rng``.
    """
    n, n_lags, H = 2, 1, 1
    df_data, _, _, _ = simulate_var(40, n, n_lags, seed=8)
    cv_options = {"H": H, "oos_test_window_size": 2}

    _SmallGridRngRecordingModel.sample_rng_ids.clear()
    model1 = _SmallGridRngRecordingModel(
        minnesota=True, soc=False, sur=False, covid=False
    )
    bvar1 = BVAR(
        n_lags, model1, stationary=True, optimisation_method="cross_validation"
    )
    bvar1.optimise_hyperparameters(df_data, cv_options=cv_options, random_state=42)
    ids1 = list(_SmallGridRngRecordingModel.sample_rng_ids)

    _SmallGridRngRecordingModel.sample_rng_ids.clear()
    model2 = _SmallGridRngRecordingModel(
        minnesota=True, soc=False, sur=False, covid=False
    )
    bvar2 = BVAR(
        n_lags, model2, stationary=True, optimisation_method="cross_validation"
    )
    bvar2.optimise_hyperparameters(df_data, cv_options=cv_options, random_state=42)
    ids2 = list(_SmallGridRngRecordingModel.sample_rng_ids)

    np.testing.assert_allclose(bvar1.model.pars.c1, bvar2.model.pars.c1)
    np.testing.assert_allclose(bvar1.model.pars.c3, bvar2.model.pars.c3)

    # 2 grid points x 3 rolling windows each, one shared generator per run.
    assert len(ids1) == len(ids2) == 6
    assert len(set(ids1)) == 1
    assert len(set(ids2)) == 1
    # An ignored random_state would fall back to self.rng and match the
    # instance generator's recorded id.
    assert ids1[0] != id(bvar1.rng)
    assert ids2[0] != id(bvar2.rng)


def test_optimise_hyperparameters_cross_validation_does_not_mutate_global_rng():
    """``cross_validation`` optimisation must draw only from the resolved
    private generator, never from the legacy global ``numpy.random`` state."""
    n, n_lags, H = 2, 1, 1
    df_data, _, _, _ = simulate_var(40, n, n_lags, seed=9)
    cv_options = {"H": H, "oos_test_window_size": 2}

    model = NaturalConjugate(minnesota=True, soc=False, sur=False, covid=False)
    bvar = BVAR(n_lags, model, stationary=True, optimisation_method="cross_validation")

    state_before = np.random.get_state()
    bvar.optimise_hyperparameters(df_data, cv_options=cv_options, random_state=42)
    state_after = np.random.get_state()

    np.testing.assert_equal(state_before, state_after)


def test_failed_optimisation_preserves_fitted_metadata_and_prior_state():
    """A failed candidate must not commit model or fitted-object mutations."""
    df_data, _, _, _ = simulate_var(30, 2, 1, seed=10)
    model = _FailingOptimisationModel(minnesota=True, soc=False, sur=False, covid=False)
    bvar = BVAR(1, model, stationary=True, optimisation_method="cross_validation")
    bvar.sample(df_data, N_draws=4, progressbar=False, random_state=10)

    beta_before = bvar.beta.copy()
    sigma_before = bvar.sigma.copy()
    df_data_before = bvar.df_data.copy()
    model_before = bvar.model
    pars_before = bvar.model.pars.__dict__.copy()

    with pytest.raises(RuntimeError, match="synthetic"):
        bvar.optimise_hyperparameters(
            df_data,
            cv_options={"H": 1, "oos_test_window_size": 2},
        )

    np.testing.assert_array_equal(bvar.beta, beta_before)
    np.testing.assert_array_equal(bvar.sigma, sigma_before)
    pd.testing.assert_frame_equal(bvar.df_data, df_data_before)
    assert bvar.model is model_before
    for name, value in pars_before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(bvar.model.pars, name), value)
        else:
            assert getattr(bvar.model.pars, name) == value


def test_direct_grid_search_failure_preserves_fitted_state_and_rng():
    """A failed public grid search must not partially mutate its BVAR."""
    df_data, _, _, _ = simulate_var(30, 2, 1, seed=11)
    model = _FailingGridSearchModel(minnesota=True, soc=False, sur=False, covid=False)
    bvar = _FailingGridSearchBVAR(1, model, stationary=True, optimisation_method="none")
    bvar.sample(df_data, N_draws=4, progressbar=False, random_state=11)
    bvar.target_indices = [1]

    beta_before = bvar.beta.copy()
    sigma_before = bvar.sigma.copy()
    beta_point_before = bvar.beta_point.copy()
    sigma_point_before = bvar.sigma_point.copy()
    posterior_state_before = bvar.posterior_state_point
    posterior_beta_before = posterior_state_before.beta.copy()
    posterior_sigma_before = posterior_state_before.sigma.copy()
    df_data_before = bvar.df_data.copy()
    data_before = bvar.data.copy()
    T_before = bvar.T
    N_draws_before = bvar.N_draws
    point_only_before = bvar.point_only
    model_before = bvar.model
    pars_before = copy.deepcopy(bvar.model.pars.__dict__)
    beta_0_before = bvar.model.beta_0.copy()
    V_A_inv_before = bvar.model.V_A_inv.copy()
    target_indices_before = bvar.target_indices.copy()
    rng_before = copy.deepcopy(bvar.rng.bit_generator.state)

    with pytest.raises(RuntimeError, match="synthetic grid-search"):
        bvar.grid_search(
            df_data.to_numpy(),
            cv_options={"H": 1, "oos_test_window_size": 2},
        )

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
    np.testing.assert_array_equal(bvar.data, data_before)
    assert bvar.T == T_before
    assert bvar.N_draws == N_draws_before
    assert bvar.point_only is point_only_before
    assert bvar.model is model_before
    for name, value in pars_before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(bvar.model.pars, name), value)
        else:
            assert getattr(bvar.model.pars, name) == value
    np.testing.assert_array_equal(bvar.model.beta_0, beta_0_before)
    np.testing.assert_array_equal(bvar.model.V_A_inv, V_A_inv_before)
    assert bvar.target_indices == target_indices_before
    np.testing.assert_equal(bvar.rng.bit_generator.state, rng_before)


@pytest.mark.parametrize(
    "cv_options",
    [{"oos_test_window_size": 2}, {"H": 1}],
    ids=["missing_h", "missing_window"],
)
def test_grid_search_rejects_missing_cv_options(cv_options):
    """Grid search requires both horizon and out-of-sample window options."""
    bvar, data = _make_validation_bvar_and_data(T=20)

    with pytest.raises(ValueError, match="cv_options"):
        bvar.grid_search(data.to_numpy(), cv_options=cv_options)


@pytest.mark.parametrize(
    "option, value",
    [
        ("H", 0),
        ("H", -1),
        ("H", 1.5),
        ("H", True),
        ("oos_test_window_size", 0),
        ("oos_test_window_size", -1),
        ("oos_test_window_size", 1.5),
        ("oos_test_window_size", True),
    ],
)
def test_grid_search_rejects_invalid_horizon_and_window(option, value):
    """Horizon and rolling-window sizes must be positive integers."""
    bvar, data = _make_validation_bvar_and_data(T=20)
    cv_options = {"H": 1, "oos_test_window_size": 2}
    cv_options[option] = value

    with pytest.raises(ValueError, match="positive integer"):
        bvar.grid_search(data.to_numpy(), cv_options=cv_options)


def test_grid_search_rejects_impossible_rolling_window():
    """Grid search rejects data too short for even its first training window."""
    bvar, data = _make_validation_bvar_and_data(T=4)

    with pytest.raises(ValueError, match="rolling window"):
        bvar.grid_search(
            data.to_numpy(), cv_options={"H": 1, "oos_test_window_size": 2}
        )


@pytest.mark.parametrize("target_indices", [[], [-1], [2], [0.5]])
def test_grid_search_rejects_empty_or_invalid_targets(target_indices):
    """Target indices must identify at least one in-range variable."""
    bvar, data = _make_validation_bvar_and_data(T=20)

    with pytest.raises(ValueError, match="target_indices"):
        bvar.grid_search(
            data.to_numpy(),
            cv_options={"H": 1, "oos_test_window_size": 2},
            target_indices=target_indices,
        )


def test_grid_search_rejects_unsupported_method():
    """Unsupported cross-validation methods fail before grid evaluation."""
    bvar, data = _make_validation_bvar_and_data(T=20)

    with pytest.raises(ValueError, match="predictive_ml"):
        bvar.grid_search(
            data.to_numpy(),
            cv_method="k_fold",
            cv_options={"H": 1, "oos_test_window_size": 2},
        )
