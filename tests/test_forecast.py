from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from bvar.BVAR import BVAR
from bvar.forecast.mixin import Forecasting
from bvar.models import NaturalConjugate

# `bvar` is the shared, session-fitted model from conftest.py. The fixture
# fits it once per session. Each test passes `random_state=seed` to its first
# stochastic call, which keeps results deterministic and independent of order.
seed = 1234
H_PAR = 8


def _transformation_helper(freq="Q"):
    helper = Forecasting()
    helper.data_transformation = {"series": "levels"}
    helper.df_data = pd.DataFrame(columns=["series"])
    helper.data = np.empty((1, 1))
    helper.n_lags = 0
    helper.freq = freq
    return helper


def test_forecast_transformation_labels_are_exact_and_documented():
    helper = _transformation_helper()
    forecasts = np.array([[[np.log(2.0)], [np.log(3.0)]]])

    for label in ["log_levels", "log levels", "logs"]:
        helper.data_transformation = {"series": label}
        transformed = helper._apply_forecast_transformations(
            forecasts, {"series": None}, None
        )
        np.testing.assert_allclose(transformed[:, :, 0], [[2.0, 3.0]])

    helper.data_transformation = {"series": "log_diff_extra"}
    with pytest.raises(ValueError, match="Unknown data_state"):
        helper._apply_forecast_transformations(forecasts, {"series": None}, None)


def test_forecast_transformation_rejects_boolean_key_before_integer_indexing():
    helper = _transformation_helper()
    forecasts = np.array([[[1.0]]])

    with pytest.raises(TypeError, match="Variable key must be str or int"):
        helper._apply_forecast_transformations(forecasts, {True: None}, None)


def test_forecast_transformation_prefers_name_over_index_metadata():
    helper = _transformation_helper()
    helper.data_transformation = {"series": "levels", 0: "logs"}
    forecasts = np.array([[[2.0]]])

    transformed = helper._apply_forecast_transformations(forecasts, {0: None}, None)

    np.testing.assert_array_equal(transformed, forecasts)


def test_forecast_diff_reconstruction_preserves_history_and_uses_forecast_tail():
    helper = _transformation_helper()
    helper.data_transformation = {"series": "diff"}
    levels = np.array([[100.0], [110.0], [130.0]])
    helper.data = np.diff(levels, axis=0)
    forecasts = np.concatenate([helper.data[None, :, :], [[[1.0], [2.0]]]], axis=1)
    helper.n_lags = 0

    transformed = helper._apply_forecast_transformations(
        forecasts, {"series": None}, base_value=100.0
    )

    np.testing.assert_allclose(transformed[:, :, 0], [[10.0, 20.0, 101.0, 103.0]])


def test_forecast_log_diff_reconstruction_preserves_history_and_uses_base():
    helper = _transformation_helper()
    helper.data_transformation = {"series": "log diff"}
    levels = np.exp(np.array([[0.0], [2.0], [6.0]]))
    helper.data = np.diff(np.log(levels), axis=0)
    forecasts = np.concatenate(
        [helper.data[None, :, :], [[[np.log(1.1)], [np.log(2.0)]]]], axis=1
    )
    helper.n_lags = 0

    transformed = helper._apply_forecast_transformations(
        forecasts, {"series": None}, base_value=4.0
    )

    np.testing.assert_allclose(transformed[:, :, 0], [[2.0, 4.0, 4.4, 8.8]], rtol=1e-12)


def test_forecast_diff_qoq_uses_reconstructed_levels_for_tail_only():
    helper = _transformation_helper()
    helper.data_transformation = {"series": "diff"}
    levels = np.array([[100.0], [110.0], [130.0], [140.0]])
    helper.data = np.diff(levels, axis=0)
    forecasts = np.concatenate([helper.data[None, :, :], [[[14.0], [7.0]]]], axis=1)

    transformed = helper._apply_forecast_transformations(
        forecasts, {"series": "qoq"}, base_value=140.0
    )

    assert transformed.shape == forecasts.shape
    np.testing.assert_array_equal(transformed[:, :3, 0], forecasts[:, :3, 0])
    np.testing.assert_allclose(
        transformed[:, 3:, 0], [[14.0 / 140.0, 7.0 / 154.0]], rtol=1e-12
    )


def test_forecast_log_diff_yoy_uses_reconstructed_observed_levels():
    helper = _transformation_helper(freq="Q")
    helper.data_transformation = {"series": "log_diff"}
    log_growth = np.log(1.1)
    helper.data = np.full((5, 1), log_growth)
    forecasts = np.concatenate(
        [helper.data[None, :, :], [[[log_growth], [log_growth]]]], axis=1
    )

    transformed = helper._apply_forecast_transformations(
        forecasts, {"series": "yoy"}, base_value=146.41
    )

    assert transformed.shape == forecasts.shape
    np.testing.assert_array_equal(transformed[:, :5, 0], forecasts[:, :5, 0])
    np.testing.assert_allclose(transformed[:, 5:, 0], [[0.4641, 0.4641]], rtol=1e-12)


def test_forecast_diff_yoy_uses_monthly_reconstructed_levels():
    helper = _transformation_helper(freq="M")
    helper.data_transformation = {"series": "diff"}
    helper.data = np.arange(1.0, 13.0).reshape(-1, 1)
    forecasts = np.concatenate([helper.data[None, :, :], [[[5.0], [7.0]]]], axis=1)

    transformed = helper._apply_forecast_transformations(
        forecasts, {"series": "yoy"}, base_value=178.0
    )

    assert transformed.shape == forecasts.shape
    np.testing.assert_array_equal(transformed[:, :12, 0], forecasts[:, :12, 0])
    np.testing.assert_allclose(
        transformed[:, 12:, 0],
        [[(183.0 - 101.0) / 101.0, (190.0 - 103.0) / 103.0]],
        rtol=1e-12,
    )


def test_public_forecast_reconstructs_differenced_tail_from_base_value():
    rng = np.random.default_rng(seed)
    levels = np.column_stack(
        [
            100.0 + np.cumsum(rng.normal(0.0, 0.2, 40)),
            50.0 + np.cumsum(rng.normal(0.0, 0.1, 40)),
        ]
    )
    level_data = pd.DataFrame(
        levels,
        index=pd.period_range("2000Q1", periods=40, freq="Q"),
        columns=["diff_series", "level_series"],
    )
    data = level_data.diff().iloc[1:]
    model = NaturalConjugate(minnesota=True, soc=True, sur=True, covid=False)
    fitted = BVAR(1, model, True, optimisation_method="none")
    fitted.sample(
        data,
        N_draws=8,
        point_only=True,
        progressbar=False,
        data_transformation={"diff_series": "diff", "level_series": "levels"},
        random_state=seed,
    )

    fitted.forecast(H=3, point_only=True, progressbar=False, transformations=None)
    raw_forecast = fitted.forecast_unconditional.copy()
    fitted.forecast(
        H=3,
        point_only=True,
        progressbar=False,
        transformations={"diff_series": None},
        base_value=level_data["diff_series"].iloc[-1],
    )

    history_length = len(data) - fitted.n_lags
    assert fitted.forecast_unconditional.shape == (1, history_length + 3, 2)
    np.testing.assert_array_equal(
        fitted.forecast_unconditional[:, :history_length, :],
        raw_forecast[:, :history_length, :],
    )
    np.testing.assert_allclose(
        fitted.forecast_unconditional[:, history_length:, 0],
        np.cumsum(raw_forecast[:, history_length:, 0], axis=1)
        + level_data["diff_series"].iloc[-1],
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    ("data_state", "base_value"),
    [
        ("diff", np.nan),
        ("diff", np.inf),
        ("log_diff", np.nan),
        ("log_diff", np.inf),
    ],
)
def test_forecast_rejects_non_finite_reconstruction_base(data_state, base_value):
    helper = _transformation_helper()
    helper.data_transformation = {"series": data_state}
    helper.data = np.array([[1.0]])
    forecasts = np.array([[[1.0], [1.0]]])

    with pytest.raises(ValueError, match="finite"):
        helper._apply_forecast_transformations(
            forecasts, {"series": None}, base_value=base_value
        )


def test_forecast_diff_and_log_diff_reconstruction_support_per_variable_bases():
    helper = Forecasting()
    helper.data_transformation = {"diff_series": "diff", "log_series": "log_diff"}
    helper.df_data = pd.DataFrame(columns=["diff_series", "log_series"])
    levels = np.array(
        [[100.0, 100.0], [110.0, 110.0], [130.0, 121.0]],
    )
    helper.data = np.column_stack(
        [np.diff(levels[:, 0]), np.diff(np.log(levels[:, 1]))]
    )
    helper.n_lags = 0
    helper.freq = "Q"
    forecasts = np.concatenate(
        [
            helper.data[None, :, :],
            [[[1.0, np.log(1.1)], [2.0, np.log(2.0)]]],
        ],
        axis=1,
    )

    transformed = helper._apply_forecast_transformations(
        forecasts,
        {"diff_series": None, "log_series": None},
        base_value=[100.0, 121.0],
    )

    np.testing.assert_allclose(
        transformed[0, :, :],
        [
            [10.0, np.log(1.1)],
            [20.0, np.log(1.1)],
            [101.0, 133.1],
            [103.0, 266.2],
        ],
        rtol=1e-12,
    )


@pytest.mark.parametrize("data_state", ["diff", "log_diff"])
def test_forecast_diff_reconstruction_requires_base_value(data_state):
    helper = _transformation_helper()
    helper.data_transformation = {"series": data_state}
    levels = np.array([[100.0], [110.0], [130.0]])
    helper.data = np.diff(levels, axis=0)
    helper.n_lags = 0
    forecasts = np.concatenate([helper.data[None, :, :], [[[1.0], [2.0]]]], axis=1)

    with pytest.raises(ValueError, match="base_value is required"):
        helper._apply_forecast_transformations(
            forecasts,
            {"series": None},
            base_value=None,
        )


def test_forecast_yoy_uses_monthly_frequency():
    helper = _transformation_helper(freq="M")
    helper.data_transformation = {"series": "levels"}
    forecasts = np.arange(1.0, 14.0).reshape(1, 13, 1)

    transformed = helper._apply_forecast_transformations(
        forecasts, {"series": "yoy"}, None
    )

    assert np.isnan(transformed[:, :12, :]).all()
    np.testing.assert_allclose(transformed[:, 12, 0], (13.0 - 1.0) / 1.0)


def test_forecast_yoy_retains_quarterly_frequency():
    helper = _transformation_helper(freq="Q")
    helper.data_transformation = {"series": "levels"}
    forecasts = np.arange(1.0, 6.0).reshape(1, 5, 1)

    transformed = helper._apply_forecast_transformations(
        forecasts, {"series": "yoy"}, None
    )

    assert np.isnan(transformed[:, :4, :]).all()
    np.testing.assert_allclose(transformed[:, 4, 0], (5.0 - 1.0) / 1.0)


@pytest.mark.parametrize("data_state", ["levels", "diff", "log diff"])
def test_forecast_transformations_accept_zero_length_horizon(data_state):
    helper = _transformation_helper(freq="M")
    helper.data_transformation = {"series": data_state}
    forecasts = np.empty((2, 0, 1))

    transformed = helper._apply_forecast_transformations(
        forecasts, {"series": "yoy"}, None
    )

    assert transformed.shape == forecasts.shape


@pytest.mark.parametrize("method_name", ["recursive_forecast", "forecast"])
@pytest.mark.parametrize("invalid_horizon", [True, 0, -1, 1.5])
def test_forecast_rejects_invalid_horizon_before_state_mutation(
    bvar, method_name, invalid_horizon
):
    """Forecast entry points reject non-positive or non-integral horizons early."""
    rng_before = deepcopy(bvar.rng.bit_generator.state)

    with pytest.raises(ValueError, match="H must be a positive integer"):
        getattr(bvar, method_name)(
            H=invalid_horizon,
            point_only=True,
            random_state=9876,
        )

    np.testing.assert_equal(bvar.rng.bit_generator.state, rng_before)


def test_forecast_recursive_mean(bvar):
    H = H_PAR
    T = bvar.T

    # mean only
    bvar.recursive_forecast(H=H, point_only=True)
    mean_only = bvar.forecast_unconditional[0]

    # distribution
    bvar.recursive_forecast(
        H=H, point_only=False, N_draws=bvar.N_draws, random_state=seed
    )
    with_distribution = bvar.forecast_unconditional

    difference = np.sum(
        np.mean(
            with_distribution[
                :,
                T:,
            ],
            axis=0,
        )
        - mean_only[T:,]
    )

    assert np.abs(difference) < 0.5, (
        "mean-only and with-distribution recursive forecasts do not match."
    )


def test_forecast_matrix_mean(bvar):
    H = H_PAR
    T = bvar.T

    # mean only
    bvar.forecast(H=H, point_only=True)
    mean_only = bvar.forecast_unconditional[0]

    # distribution
    bvar.forecast(H=H, point_only=False, random_state=seed)
    with_distribution = bvar.forecast_unconditional

    difference = (
        np.mean(
            with_distribution[
                :,
                T:,
            ],
            axis=0,
        )
        - mean_only[T:,]
    )

    sum_difference = np.sum(difference)

    assert np.abs(sum_difference) < 0.5, (
        "mean-only and with-distribution matrix forecasts do not match."
    )


def test_forecast_fixed_seed_reproduces_unconditional_draws(bvar):
    """The same forecast seed must reproduce the complete simulated paths."""
    bvar.forecast(H=2, N_draws=16, point_only=False, random_state=seed)
    first_draws = bvar.forecast_unconditional.copy()

    bvar.forecast(H=2, N_draws=16, point_only=False, random_state=seed)

    np.testing.assert_array_equal(bvar.forecast_unconditional, first_draws)


def test_forecast_matrix_recursive_mean(bvar):
    """Check that both forecasting algorithms agree without constraints."""
    # forecast
    H = H_PAR
    T = bvar.T
    bvar.recursive_forecast(H=H, point_only=True)
    recursive_forecast = bvar.forecast_unconditional[0]

    # matrix forecast
    bvar.forecast(H=H, point_only=True)

    matrix_forecast = bvar.forecast_unconditional[0]
    difference = np.mean(matrix_forecast[T:,] - recursive_forecast[T:,])

    assert np.abs(difference) < 1e-6, (
        "Matrix and recursive forecasts do not have same mean."
    )


def test_forecast_matrix_recursive_with_variance(bvar):
    """Check that both forecasting algorithms agree with simulated variance."""
    # forecast
    H = H_PAR
    T = bvar.T
    bvar.recursive_forecast(H=H, point_only=False, random_state=seed)
    recursive_forecast = bvar.forecast_unconditional[:, T:, :]

    # matrix forecast
    bvar.forecast(H=H, point_only=False, random_state=seed)
    matrix_forecast = bvar.forecast_unconditional[:, T:, :]

    diff = np.var(matrix_forecast, axis=0) - np.var(recursive_forecast, axis=0)

    assert np.abs(np.mean(diff)) < 0.2, (
        "Matrix and recursive forecasts do not have same varianceS."
    )


def test_conditional_forecasting_basic(bvar):
    """Check that conditional forecasting with conditioning set to unconditional forecasts g gives the same results as the unconditional forecasts."""
    # forecast
    H = H_PAR
    T = bvar.T
    n = bvar.n

    # unconditional forecast
    bvar.forecast(
        H=H,
        point_only=True,
    )
    forecast_unconditional = bvar.forecast_unconditional[0]

    # conditional forecast
    # set constraints
    T = bvar.T
    constraint_mean = np.full((H, n), np.nan)
    constraint_mean[:, 1] = forecast_unconditional[T:, 1]

    # forecast with constraints
    bvar.forecast(
        H=H,
        constraint_mean=constraint_mean,
        point_only=True,
    )

    forecast_conditional = bvar.forecast_conditional[0]
    difference = np.mean(forecast_conditional[T:, 0:] - forecast_unconditional[T:, 0:])

    assert np.abs(difference) < 1e-6, (
        "Conditional forecasts with unconditional forecast as constraint does not match unconditional forecast"
    )


def test_conditional_forecast_gibbs(bvar):
    """Check that Gibbs conditioning follows the recursive forecast without constraints."""
    # forecast
    H = H_PAR
    T = bvar.T
    n = bvar.n

    # unconditional forecast
    bvar.forecast(
        H=H,
        point_only=True,
    )
    forecast_unconditional = bvar.forecast_unconditional[0]

    constraint_mean = np.full((H, n), np.nan)
    constraint_mean[:, 0] = forecast_unconditional[T:, 0]

    # forecast with constraints
    bvar.forecast(
        H=H,
        constraint_mean=constraint_mean,
        point_only=False,
        random_state=seed,
    )

    forecast_conditional = bvar.forecast_conditional.mean(axis=0)
    difference = np.mean(forecast_conditional[T:, 1:] - forecast_unconditional[T:, 1:])

    assert np.abs(difference) < 0.1, "Gibbs sampler for conditional forecasting works"


def test_conditional_forecast_gibbs_with_variance(bvar):
    """Check that Gibbs conditioning preserves the forecast with variance constraints."""
    # forecast
    H = H_PAR
    T = bvar.T
    n = bvar.n

    # unconditional forecast
    bvar.forecast(
        H=H,
        point_only=True,
    )
    forecast_unconditional = bvar.forecast_unconditional[0]

    constraint_mean = np.full((H, n), np.nan)
    constraint_mean[:, 0] = forecast_unconditional[T:, 0]

    constraint_variance = np.full((H, n), np.nan)
    constraint_variance[:, 0] = 1.0

    # forecast with constraints
    bvar.forecast(
        H=H,
        constraint_mean=constraint_mean,
        constraint_variance=constraint_variance,
        point_only=False,
        random_state=seed,
    )

    forecast_conditional = bvar.forecast_conditional.mean(axis=0)
    difference = np.mean(forecast_conditional[T:, 1:] - forecast_unconditional[T:, 1:])

    assert np.abs(difference) < 0.1, "Gibbs sampler for conditional forecasting works"


def test_conditional_forecasts_match_constraints(bvar):
    """Check that conditional draws follow the requested constraints."""
    # forecast
    H = H_PAR
    T = bvar.T
    n = bvar.n

    constraint_mean = np.full((H, n), np.nan)
    constraint_mean[:, 0] = 0.0

    constraint_variance = np.full((H, n), np.nan)
    constraint_variance[:, 0] = 2.0

    # forecast with constraints
    bvar.forecast(
        H=H,
        constraint_mean=constraint_mean,
        constraint_variance=constraint_variance,
        point_only=False,
        random_state=seed,
    )

    forecast_conditional = bvar.forecast_conditional[:, T:, 0]
    mean_error = np.mean(forecast_conditional) - 0.0

    assert np.abs(mean_error) < 0.05, (
        "Constrained forecasts don't match constraint mean"
    )

    variance_error = np.var(forecast_conditional) - 2.0
    assert np.abs(variance_error) < 0.05, (
        "Constrained forecasts don't match constraint variance"
    )


def test_conditional_skew_zero_skewness_andersson(bvar):
    """Check that zero skewness gives the expected conditional forecast."""
    # forecast
    H = H_PAR
    T = bvar.T
    n = bvar.n

    loc_constraint = 0.0
    constraint_mean = np.full((H, n), np.nan)
    constraint_mean[:, 1] = loc_constraint

    requested_variance = 2.0
    constraint_variance = np.full((H, n), np.nan)
    constraint_variance[:, 1] = requested_variance

    shape_constraint = 0.0
    constraint_shape = np.full((H, n), np.nan)
    constraint_shape[:, 1] = shape_constraint

    # forecast with Skew Normal
    bvar.forecast(
        H=H,
        constraint_mean=constraint_mean,
        constraint_variance=constraint_variance,
        constraint_shape=constraint_shape,
        point_only=False,
        random_state=seed,
    )
    forecast_skew = bvar.forecast_conditional

    # forecast with Normal
    bvar.forecast(
        H=H,
        constraint_mean=constraint_mean,
        constraint_variance=constraint_variance,
        point_only=False,
        random_state=seed,
    )
    forecast_normal = bvar.forecast_conditional

    # Get the forecast data
    forecast_normal = forecast_normal[:, T:, 0]
    forecast_skew = forecast_skew[:, T:, 0]

    # Check mean and variance
    mean_normal = np.mean(forecast_normal)
    mean_skew = np.mean(forecast_skew)

    var_normal = np.var(forecast_normal)
    var_skew = np.var(forecast_skew)
    assert np.abs(mean_normal - mean_skew) < 0.05, (
        "Means are not equal for zero skewness"
    )
    assert np.abs(var_normal - var_skew) < 0.05, (
        "Variances are not equal for zero skewness"
    )

    # Check that the forecasts for the constraints match the constraint
    forecast_constraint = bvar.forecast_conditional[:, T:, 1]
    mean_constraint = np.mean(forecast_constraint)
    observed_variance = np.var(forecast_constraint)

    assert np.abs(mean_constraint - loc_constraint) < 0.05, (
        "Means are not equal to constraint for zero skewness"
    )

    assert np.abs(observed_variance - requested_variance) < 0.05, (
        "Variances are not equal to constraint for zero skewness"
    )


def test_conditional_skew_zero_skewness_renzetti(bvar):
    """Check that zero skewness gives the expected conditional forecast."""
    # forecast
    H = H_PAR
    T = bvar.T
    n = bvar.n

    loc_constraint = 0.0
    constraint_mean = np.full((H, n), np.nan)
    constraint_mean[:, 1] = loc_constraint

    requested_variance = 2.0
    constraint_variance = np.full((H, n), np.nan)
    constraint_variance[:, 1] = requested_variance

    shape_constraint = 0.0
    constraint_shape = np.full((H, n), np.nan)
    constraint_shape[:, 1] = shape_constraint

    # forecast with Skew Normal
    bvar.forecast(
        H=H,
        constraint_mean=constraint_mean,
        constraint_variance=constraint_variance,
        constraint_shape=constraint_shape,
        point_only=False,
        method="labonne_renzetti",
        random_state=seed,
    )
    forecast_skew = bvar.forecast_conditional

    # forecast with Normal
    bvar.forecast(
        H=H,
        constraint_mean=constraint_mean,
        constraint_variance=constraint_variance,
        method="antolin_diaz_et_al",
        point_only=False,
        random_state=seed,
    )
    forecast_normal = bvar.forecast_conditional

    # Get the forecast data
    forecast_normal = forecast_normal[:, T:, 0]
    forecast_skew = forecast_skew[:, T:, 0]

    # Check mean and variance
    mean_normal = np.mean(forecast_normal)
    mean_skew = np.mean(forecast_skew)

    var_normal = np.var(forecast_normal)
    var_skew = np.var(forecast_skew)

    assert np.abs(mean_normal - mean_skew) < 0.05, (
        "Means are not equal for zero skewness"
    )
    assert np.abs(var_normal - var_skew) < 0.05, (
        "Variances are not equal for zero skewness"
    )

    # Check that the forecasts for the constraints match the constraint
    forecast_constraint = bvar.forecast_conditional[:, T:, 1]
    mean_constraint = np.mean(forecast_constraint)
    observed_variance = np.var(forecast_constraint)

    assert np.abs(mean_constraint - loc_constraint) < 0.05, (
        "Means are not equal to constraint for zero skewness"
    )

    assert np.abs(observed_variance - requested_variance) < 0.05, (
        "Variances are not equal to constraint for zero skewness"
    )


def test_conditional_skewness(bvar):
    H = 1
    T = bvar.T
    n = bvar.n

    constraint_mean = np.full((H, n), np.nan)
    constraint_mean[:, 1] = 0.0

    constraint_variance = np.full((H, n), np.nan)
    constraint_variance[:, 1] = 1.0

    constraint_shape = np.full((H, n), np.nan)
    constraint_shape[:, 1] = 1000

    # forecast with Skew Normal
    bvar.forecast(
        H=H,
        constraint_mean=constraint_mean,
        constraint_variance=constraint_variance,
        constraint_shape=constraint_shape,
        point_only=False,
        random_state=seed,
    )
    forecast_skew = bvar.forecast_conditional

    # Get the forecast data
    from bvar.skew_normal import draw_skew_normal

    forecast_constrained_var = forecast_skew[:, T, 1]

    constraint = draw_skew_normal(
        cov=np.array([[1.0]]),
        alpha=np.array([1000]),
        size=10000,
        rng=np.random.default_rng(seed),
    ).reshape(-1)

    # check mean
    mean_forecast = np.mean(forecast_constrained_var)
    mean_constraint = np.mean(constraint)
    assert np.abs(mean_forecast - mean_constraint) < 0.05, (
        "Mean of constrained forecasts not equal to constraint"
    )

    # check variance
    variance_forecast = np.var(forecast_constrained_var)
    variance_constraint = np.var(constraint)
    assert np.abs(variance_forecast - variance_constraint) < 0.05, (
        "Mean of constrained forecasts not equal to constraint"
    )

    # check skewness
    from scipy.stats import skew

    skew_forecast = skew(forecast_constrained_var)
    skew_constraint = skew(constraint)

    assert np.abs(skew_forecast - skew_constraint) < 0.1, (
        "Skewness of constrained forecasts not equal to constraint"
    )


def test_conditional_skewness_renzetti(bvar):
    H = 1
    T = bvar.T
    n = bvar.n

    constraint_mean = np.full((H, n), np.nan)
    constraint_mean[:, 1] = 0.0

    constraint_variance = np.full((H, n), np.nan)
    constraint_variance[:, 1] = 1.0

    constraint_shape = np.full((H, n), np.nan)
    constraint_shape[:, 1] = 1000

    # forecast with Skew Normal
    bvar.forecast(
        H=H,
        constraint_mean=constraint_mean,
        constraint_variance=constraint_variance,
        constraint_shape=constraint_shape,
        method="labonne_renzetti",
        point_only=False,
        N_draws=10000,
        random_state=seed,
    )

    forecast_skew = bvar.forecast_conditional

    plt.figure(figsize=(8, 6))
    plt.hist(forecast_skew[:, T, :], bins=100, alpha=0.7, edgecolor="black")
    plt.grid(True)
    # plt.show()
    # plt.close()

    # compute the skewness of the constrained variable
    from scipy.stats import skew

    forecast_constrained_var = forecast_skew[:, T, 0]
    skew_forecast = skew(forecast_constrained_var)
    print(f"Skewness of constrained variable: {skew_forecast}")


def test_mean_forecast_unconditional(bvar):
    """Check that when selecting the return only the conditional mean, we actually get the conditional mean"""
    H = H_PAR
    T = bvar.T

    # unconditional forecast
    bvar.forecast(
        H=H,
        N_draws=10000,
        random_state=seed,
    )
    forecast_unconditional = np.mean(bvar.forecast_unconditional, axis=0)

    # only returns the mean (from the uncon)
    # set constraints
    bvar.forecast(
        H=H,
        point_only=True,
    )

    difference = np.mean(
        bvar.forecast_unconditional[
            0,
            T:,
        ]
        - forecast_unconditional[T:,]
    )

    assert np.abs(difference) < 0.1, (
        "Mean-forecast-only accurate when working with unconditional forecasts"
    )


def test_mean_forecast_unconditional_recursive_H8(bvar):
    """Check that matrix and recursive forecasts give same conditional-mean-only forecasts"""
    H = H_PAR
    T = bvar.T

    # unconditional forecast
    bvar.recursive_forecast(H=H, point_only=True)
    forecast_recursive = bvar.forecast_unconditional

    # only returns the mean (from the uncon)
    # set constraints
    bvar.forecast(
        H=H,
        point_only=True,
    )

    difference = np.mean(
        bvar.forecast_unconditional[
            0,
            T:,
        ]
        - forecast_recursive[
            0,
            T:,
        ]
    )

    assert np.abs(difference) < 0.01, (
        "Mean-forecast-only accurate when working with unconditional forecasts"
    )


# ---------------------------------------------------------------------------
# Retained conditional draw counts (review finding #26)
#
# `N_draws` is the total number of Gibbs iterations run; `N_burn` draws are
# discarded as burn-in, so `forecast_conditional.shape[0] == N_draws - N_burn`.
# These regression tests pin down that contract so it can't silently change.
# ---------------------------------------------------------------------------


def test_conditional_forecast_retains_requested_draws_default_burn(bvar):
    """With N_burn left at its default (N_draws // 2), N_draws - N_burn draws should be retained."""
    H = 1
    n = bvar.n
    N_draws = 20
    N_burn = N_draws // 2

    constraint_mean = np.full((H, n), np.nan)
    constraint_mean[:, 0] = 0.0

    bvar.forecast(
        H=H,
        constraint_mean=constraint_mean,
        point_only=False,
        N_draws=N_draws,
        random_state=seed,
    )

    assert bvar.forecast_conditional.shape[0] == N_draws - N_burn


def test_conditional_forecast_retains_requested_draws_explicit_burn(bvar):
    """An explicit N_burn should leave exactly N_draws - N_burn draws retained."""
    H = 1
    n = bvar.n
    N_draws = 20
    N_burn = 5

    constraint_mean = np.full((H, n), np.nan)
    constraint_mean[:, 0] = 0.0

    bvar.forecast(
        H=H,
        constraint_mean=constraint_mean,
        point_only=False,
        N_draws=N_draws,
        N_burn=N_burn,
        random_state=seed,
    )

    assert bvar.forecast_conditional.shape[0] == N_draws - N_burn


def test_conditional_forecast_point_only_retains_single_draw(bvar):
    """point_only conditional forecasts should always retain exactly one draw."""
    H = 1
    n = bvar.n

    constraint_mean = np.full((H, n), np.nan)
    constraint_mean[:, 0] = 0.0

    bvar.forecast(
        H=H,
        constraint_mean=constraint_mean,
        point_only=True,
    )

    assert bvar.forecast_conditional.shape[0] == 1


def test_conditional_forecast_draws_capped_at_available_posterior_draws(bvar):
    """Requesting more draws than are in the posterior sample should be capped
    at bvar.N_draws (the available posterior sample size), not silently
    return the larger requested count."""
    H = 1
    n = bvar.n
    requested_draws = bvar.N_draws + 1000

    constraint_mean = np.full((H, n), np.nan)
    constraint_mean[:, 0] = 0.0

    bvar.forecast(
        H=H,
        constraint_mean=constraint_mean,
        point_only=False,
        N_draws=requested_draws,
        N_burn=0,
        random_state=seed,
    )

    assert bvar.forecast_conditional.shape[0] == bvar.N_draws


def test_conditional_forecast_capped_draws_with_default_burn_retains_draws(bvar):
    """Requesting more draws than stored, with N_burn left at its default,
    must not leave zero retained draws: the default burn-in should be derived
    from the capped (effective) draw count, not the originally requested one."""
    H = 1
    n = bvar.n
    requested_draws = bvar.N_draws + 1000

    constraint_mean = np.full((H, n), np.nan)
    constraint_mean[:, 0] = 0.0

    bvar.forecast(
        H=H,
        constraint_mean=constraint_mean,
        point_only=False,
        N_draws=requested_draws,
        random_state=seed,
    )

    expected_retained = bvar.N_draws - (bvar.N_draws // 2)
    assert bvar.forecast_conditional.shape[0] == expected_retained
    assert bvar.forecast_conditional.shape[0] > 0
    assert np.all(np.isfinite(bvar.forecast_conditional))


# ---------------------------------------------------------------------------
# Conditional forecast lifecycle hardening: N_draws/N_burn validation and
# point_only update suppression.
# ---------------------------------------------------------------------------


def test_conditional_forecast_non_positive_N_draws_raises(bvar):
    H = 1
    n = bvar.n

    constraint_mean = np.full((H, n), np.nan)
    constraint_mean[:, 0] = 0.0

    with pytest.raises(ValueError, match="N_draws"):
        bvar.forecast(
            H=H,
            constraint_mean=constraint_mean,
            point_only=False,
            N_draws=0,
        )


def test_conditional_forecast_non_integer_N_draws_raises(bvar):
    """A non-integer N_draws (e.g. a float) must be rejected with a clear
    ValueError rather than reaching min()/array allocation and raising an
    incidental TypeError."""
    H = 1
    n = bvar.n

    constraint_mean = np.full((H, n), np.nan)
    constraint_mean[:, 0] = 0.0

    with pytest.raises(ValueError, match="N_draws"):
        bvar.forecast(
            H=H,
            constraint_mean=constraint_mean,
            point_only=False,
            N_draws=2.5,
        )


@pytest.mark.parametrize("invalid_draws", [0, -1, 2.5, True])
def test_unconditional_forecast_rejects_invalid_N_draws(bvar, invalid_draws):
    """Unconditional matrix forecasts require positive integer draw counts."""
    with pytest.raises(ValueError, match="N_draws"):
        bvar.forecast(H=1, point_only=False, N_draws=invalid_draws)


@pytest.mark.parametrize("invalid_draws", [0, -1, 2.5, True])
def test_recursive_forecast_rejects_invalid_N_draws(bvar, invalid_draws):
    """Recursive forecasts require positive integer draw counts."""
    with pytest.raises(ValueError, match="N_draws"):
        bvar.recursive_forecast(H=1, point_only=False, N_draws=invalid_draws)


def test_conditional_forecast_negative_N_burn_raises(bvar):
    H = 1
    n = bvar.n

    constraint_mean = np.full((H, n), np.nan)
    constraint_mean[:, 0] = 0.0

    with pytest.raises(ValueError, match="N_burn"):
        bvar.forecast(
            H=H,
            constraint_mean=constraint_mean,
            point_only=False,
            N_draws=20,
            N_burn=-1,
        )


def test_conditional_forecast_N_burn_too_large_raises(bvar):
    """N_burn must be strictly less than the effective (capped) N_draws."""
    H = 1
    n = bvar.n
    N_draws = 20

    constraint_mean = np.full((H, n), np.nan)
    constraint_mean[:, 0] = 0.0

    with pytest.raises(ValueError, match="N_burn"):
        bvar.forecast(
            H=H,
            constraint_mean=constraint_mean,
            point_only=False,
            N_draws=N_draws,
            N_burn=N_draws,
        )


def test_conditional_forecast_N_burn_non_integer_raises(bvar):
    H = 1
    n = bvar.n

    constraint_mean = np.full((H, n), np.nan)
    constraint_mean[:, 0] = 0.0

    with pytest.raises(ValueError, match="N_burn"):
        bvar.forecast(
            H=H,
            constraint_mean=constraint_mean,
            point_only=False,
            N_draws=20,
            N_burn=2.5,
        )


def test_conditional_forecast_point_only_does_not_call_sample_posterior_state(
    bvar, monkeypatch
):
    """Point-only conditional forecasts retain one draw and skip
    sample_posterior_state because the path has no later iteration."""
    H = 1
    n = bvar.n
    calls = []

    def spy_sample_posterior_state(Y, Z, current_state, rng=None):
        calls.append(1)
        raise AssertionError("sample_posterior_state must not be called")

    monkeypatch.setattr(
        bvar.model, "sample_posterior_state", spy_sample_posterior_state
    )

    constraint_mean = np.full((H, n), np.nan)
    constraint_mean[:, 0] = 0.0

    bvar.forecast(
        H=H,
        constraint_mean=constraint_mean,
        point_only=True,
    )

    assert calls == []
    assert bvar.forecast_conditional.shape[0] == 1
