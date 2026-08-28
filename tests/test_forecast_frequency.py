"""Regression tests for non-quarterly frequency support.

The bulk of the suite fits quarterly ``PeriodIndex`` data. These tests
exercise the frequency-dependent code paths that were previously untested:

- monthly forecast dates (``PeriodIndex``),
- frequency inference from a ``DatetimeIndex``,
- formatted forecast output at a non-quarterly frequency,
- plotting non-quarterly forecasts (non-interactive backend),
- rejection of invalid / non-inferrable indexes.
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend so plotting never blocks

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from bvar.BVAR import BVAR
from bvar.forecast.mixin import _periods_per_year
from bvar.models import IndependentNIW, NaturalConjugate
from bvar.utils import simulate_var

SEED = 1234
T = 150  # Number of time periods
N_VARS = 2
N_LAGS = 1
H = 6

_AR_MAT = np.array([[0.1, 0.25], [0.5, 0.75]])
_CONSTANT = np.array([-1, 1])
_SIGMA = np.array([[1, 0.5], [0.5, 2]])


def _simulate_data(index):
    """Simulate a two-variable VAR(1) and attach ``index`` as its time axis."""
    data, _, _, _ = simulate_var(
        T,
        N_VARS,
        N_LAGS,
        ar_mat=_AR_MAT,
        constant=_CONSTANT,
        Sigma=_SIGMA,
        seed=SEED,
        covid=False,
        levels=False,
    )
    data = data.iloc[: len(index)].copy()
    data.index = index
    data.index.name = "date"
    return data


def _fit(index):
    """Fit a small ``NaturalConjugate`` BVAR to data carrying ``index``."""
    data = _simulate_data(index)
    model = NaturalConjugate(minnesota=True, soc=True, sur=True, covid=False)
    fitted = BVAR(N_LAGS, model, False, optimisation_method="ml")
    fitted.optimise_hyperparameters(data=data, nb_restart=1, random_state=SEED)
    fitted.sample(data=data, N_draws=1000, random_state=SEED)
    return fitted


@pytest.fixture(scope="module")
def bvar_monthly_period():
    """BVAR fitted to monthly data indexed by a ``PeriodIndex``."""
    index = pd.period_range(start="2000-01", periods=T, freq="M")
    return _fit(index)


@pytest.fixture(scope="module")
def bvar_monthly_datetime():
    """BVAR fitted to monthly data indexed by a month-end ``DatetimeIndex``."""
    index = pd.date_range(start="2000-01-31", periods=T, freq="ME")
    return _fit(index)


def test_monthly_period_index_freq(bvar_monthly_period):
    """A monthly ``PeriodIndex`` yields a monthly stored frequency."""
    assert bvar_monthly_period.freq == "M"


@pytest.mark.parametrize(
    ("freq", "expected"),
    [
        ("2M", 6),
        ("2ME", 6),
        ("2MS", 6),
        ("2Q-DEC", 2),
        ("2QE-DEC", 2),
        ("2QS-JAN", 2),
        ("M", 12),
        ("ME", 12),
        ("MS", 12),
        ("Q-DEC", 4),
        ("QE-DEC", 4),
        ("QS-JAN", 4),
    ],
)
def test_forecast_periods_per_year_supports_multipliers_and_anchors(freq, expected):
    """Forecast YoY periods match the GIRF frequency interpretation."""
    assert _periods_per_year(freq) == expected


def test_monthly_recursive_forecast_dates(bvar_monthly_period):
    """Recursive forecast dates are monthly and continue from the last observation."""
    last_obs = bvar_monthly_period.df_data.index[-1]

    bvar_monthly_period.recursive_forecast(H=H, point_only=True)
    dates = bvar_monthly_period.dates_forecast

    assert isinstance(dates, pd.PeriodIndex)
    assert dates.freqstr == "M"
    assert len(dates) == H
    assert dates[0] == last_obs + 1
    assert dates[-1] == last_obs + H


def test_datetime_index_infers_monthly_freq(bvar_monthly_datetime):
    """The model converts a month-end ``DatetimeIndex`` to monthly periods."""
    assert bvar_monthly_datetime.freq == "M"
    assert isinstance(bvar_monthly_datetime.df_data.index, pd.PeriodIndex)

    bvar_monthly_datetime.recursive_forecast(H=H, point_only=True)
    dates = bvar_monthly_datetime.dates_forecast

    assert isinstance(dates, pd.PeriodIndex)
    assert dates.freqstr == "M"
    assert len(dates) == H


def test_anchored_quarterly_period_index_is_accepted():
    """A quarterly PeriodIndex keeps its calendar-year anchor."""
    index = pd.period_range(start="2000Q2", periods=T, freq="Q-MAR")
    data = _simulate_data(index)
    model = NaturalConjugate(minnesota=True, soc=True, sur=True, covid=False)
    fitted = BVAR(N_LAGS, model, False, optimisation_method="none")

    fitted.optimise_hyperparameters(data=data)

    assert fitted.freq == "Q-MAR"
    assert fitted.df_data.index.equals(index)


def test_formatted_monthly_forecast_dates(bvar_monthly_period):
    """Formatted output carries monthly dates covering observations and horizon."""
    last_obs = bvar_monthly_period.df_data.index[-1]

    bvar_monthly_period.forecast(
        H=H, point_only=False, N_draws=1000, format=True, random_state=SEED
    )
    df = bvar_monthly_period.df_forecasts_unconditional

    assert isinstance(df, pd.DataFrame)
    assert {"date", "quantile", "variable", "value"}.issubset(df.columns)

    dates = pd.PeriodIndex(df["date"].unique())
    assert dates.freqstr == "M"
    # Horizon extends exactly H monthly periods beyond the last observation.
    assert dates.max() == last_obs + H


def test_plot_monthly_forecast_unconditional(bvar_monthly_period):
    """Plotting an unconditional monthly forecast does not raise."""
    bvar_monthly_period.forecast(H=H, point_only=False, N_draws=1000, random_state=SEED)
    try:
        bvar_monthly_period.plot_forecast()
    finally:
        plt.close("all")


@pytest.mark.parametrize("from_date", [pd.Period("2000-06", freq="M"), "2000-06"])
def test_plot_monthly_forecast_accepts_period_and_string_dates(
    bvar_monthly_period, from_date
):
    """Plotting accepts dates expressed at the fitted monthly frequency."""
    bvar_monthly_period.forecast(H=H, point_only=False, N_draws=1000, random_state=SEED)
    try:
        bvar_monthly_period.plot_forecast(from_date=from_date)
    finally:
        plt.close("all")


def test_plot_forecast_uses_ordered_percentile_bounds(bvar_monthly_period, monkeypatch):
    """Forecast intervals use the lower and upper tails in the right order."""
    bvar_monthly_period.forecast(H=H, point_only=False, N_draws=1000, random_state=SEED)
    percentile = np.percentile
    percentiles = []

    def record_percentile(values, percentile_level, axis=None):
        percentiles.append(percentile_level)
        return percentile(values, percentile_level, axis=axis)

    monkeypatch.setattr(np, "percentile", record_percentile)
    try:
        bvar_monthly_period.plot_forecast()
    finally:
        plt.close("all")

    assert percentiles == [50, 5, 95, 50, 5, 95]


def test_plot_monthly_forecast_conditional(bvar_monthly_period):
    """Plotting a conditional monthly forecast does not raise."""
    constraint_mean = np.full((H, N_VARS), np.nan)
    constraint_mean[:, 0] = 0.0

    bvar_monthly_period.forecast(
        H=H,
        constraint_mean=constraint_mean,
        point_only=False,
        N_draws=1000,
        random_state=SEED,
    )
    try:
        bvar_monthly_period.plot_forecast()
    finally:
        plt.close("all")


def test_plot_fitted_values_immediately_after_sample(bvar_monthly_period):
    """Fitted values are lazily computed after sampling."""
    try:
        bvar_monthly_period.plot_fitted_values()
        assert bvar_monthly_period.fitted_values.shape == (
            bvar_monthly_period.N_draws,
            bvar_monthly_period.T,
            bvar_monthly_period.n,
        )
    finally:
        plt.close("all")


def test_integer_index_rejected():
    """Data preparation rejects an integer index."""
    data = _simulate_data(pd.RangeIndex(T))
    model = NaturalConjugate(minnesota=True, soc=True, sur=True, covid=False)
    fitted = BVAR(N_LAGS, model, False, optimisation_method="ml")

    with pytest.raises(TypeError):
        fitted.optimise_hyperparameters(data=data, nb_restart=1, random_state=SEED)


def test_irregular_datetime_index_rejected():
    """Data preparation rejects a ``DatetimeIndex`` without an inferable frequency."""
    irregular = pd.DatetimeIndex(
        pd.date_range(start="2000-01-31", periods=T, freq="ME").to_list()[:-1]
        + [pd.Timestamp("2100-01-31")]
    )
    data = _simulate_data(irregular)
    model = NaturalConjugate(minnesota=True, soc=True, sur=True, covid=False)
    fitted = BVAR(N_LAGS, model, False, optimisation_method="ml")

    with pytest.raises(ValueError):
        fitted.optimise_hyperparameters(data=data, nb_restart=1, random_state=SEED)


@pytest.mark.parametrize(
    "index_factory",
    [
        lambda index: index[:10].append(index[11:]),
        lambda index: index.insert(1, index[0]).delete(-1),
        lambda index: index[::-1],
    ],
    ids=["missing", "duplicate", "descending"],
)
def test_non_regular_period_index_rejected(index_factory):
    """Missing, duplicate, and descending periods are not treated as adjacent."""
    index = pd.period_range(start="2000-01", periods=T, freq="M")
    data = _simulate_data(index_factory(index))
    model = NaturalConjugate(minnesota=True, soc=True, sur=True, covid=False)
    fitted = BVAR(N_LAGS, model, False, optimisation_method="none")

    with pytest.raises(ValueError, match="unique|increasing|regular"):
        fitted.optimise_hyperparameters(data=data)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_non_finite_data_rejected(invalid_value):
    """The model rejects NaN and both infinity signs before estimation."""
    data = _simulate_data(pd.period_range(start="2000-01", periods=T, freq="M"))
    data.iloc[4, 1] = invalid_value
    model = NaturalConjugate(minnesota=True, soc=True, sur=True, covid=False)
    fitted = BVAR(N_LAGS, model, False, optimisation_method="none")

    with pytest.raises(ValueError, match="finite"):
        fitted.optimise_hyperparameters(data=data)


def test_sample_without_post_lag_observation_rejected():
    """A sample with no usable dependent-variable row fails early."""
    data = _simulate_data(pd.period_range(start="2000-01", periods=N_LAGS, freq="M"))
    model = NaturalConjugate(minnesota=True, soc=True, sur=True, covid=False)
    fitted = BVAR(N_LAGS, model, False, optimisation_method="none")

    with pytest.raises(ValueError, match="post-lag|observations"):
        fitted.sample(data=data, N_draws=10, progressbar=False)


@pytest.mark.parametrize("model_cls", [NaturalConjugate, IndependentNIW])
def test_ml_optimisation_rejects_insufficient_ar1_scale_sample(model_cls):
    """ML optimisation reports a model-specific AR(1) scale error."""
    data = _simulate_data(pd.period_range(start="2000-01", periods=2, freq="M"))
    model = model_cls(minnesota=True, soc=True, sur=True, covid=False)
    fitted = BVAR(N_LAGS, model, False, optimisation_method="ml")
    model_before = fitted.model
    pars_before = deepcopy(model_before.pars.__dict__)
    rng_before = deepcopy(fitted.rng.bit_generator.state)

    with pytest.raises(ValueError, match=r"AR\(1\)|residual variance"):
        fitted.optimise_hyperparameters(data=data, random_state=SEED)

    assert fitted.model is model_before
    for name, value in pars_before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(fitted.model.pars, name), value)
        else:
            assert getattr(fitted.model.pars, name) == value
    np.testing.assert_equal(fitted.rng.bit_generator.state, rng_before)


def test_ml_optimisation_rejects_non_positive_ar1_residual_variance():
    """ML optimisation rejects a degenerate AR(1) covariance scale."""
    index = pd.period_range(start="2000-01", periods=4, freq="M")
    data = pd.DataFrame({"first": 1.0, "second": 2.0}, index=index)
    model = NaturalConjugate(minnesota=True, soc=True, sur=True, covid=False)
    fitted = BVAR(N_LAGS, model, False, optimisation_method="ml")

    with pytest.raises(ValueError, match="residual variance"):
        fitted.optimise_hyperparameters(data=data, random_state=SEED)


@pytest.mark.parametrize("model_cls", [NaturalConjugate, IndependentNIW])
def test_none_sampling_rejects_insufficient_ar1_initialisation(model_cls):
    """Direct sampling reports an AR(1) initialisation error before linalg."""
    data = _simulate_data(pd.period_range(start="2000-01", periods=2, freq="M"))
    model = model_cls(minnesota=True, soc=True, sur=True, covid=False)
    fitted = BVAR(N_LAGS, model, False, optimisation_method="none")

    with pytest.raises(ValueError, match=r"AR\(1\)|residual variance"):
        fitted.sample(data=data, N_draws=10, progressbar=False)


@pytest.mark.parametrize("model_cls", [NaturalConjugate, IndependentNIW])
def test_none_sampling_rejects_non_positive_ar1_residual_variance(model_cls):
    """Direct sampling rejects a degenerate AR(1) covariance scale."""
    index = pd.period_range(start="2000-01", periods=4, freq="M")
    data = pd.DataFrame({"first": 1.0, "second": 2.0}, index=index)
    model = model_cls(minnesota=True, soc=True, sur=True, covid=False)
    fitted = BVAR(N_LAGS, model, False, optimisation_method="none")

    with pytest.raises(ValueError, match="residual variance"):
        fitted.sample(data=data, N_draws=10, progressbar=False)


def test_cross_validation_rejects_ar1_invalid_rolling_fit():
    """Every rolling CV fit must validate its own AR(1) initialisation."""
    data = _simulate_data(pd.period_range(start="2000-01", periods=8, freq="M"))
    model = NaturalConjugate(minnesota=True, soc=False, sur=False, covid=False)
    fitted = BVAR(N_LAGS, model, False, optimisation_method="cross_validation")

    with pytest.raises(ValueError, match=r"AR\(1\)|residual variance"):
        fitted.optimise_hyperparameters(
            data=data,
            cv_options={"H": 1, "oos_test_window_size": 5},
        )


def test_failed_refit_with_gap_preserves_fitted_state():
    """A unique, increasing but gapped refit must not replace fitted state."""
    index = pd.period_range(start="2000-01", periods=T, freq="M")
    data = _simulate_data(index)
    model = NaturalConjugate(
        minnesota=True,
        soc=True,
        sur=True,
        covid=True,
        covid_dates=["2000-03", "2000-06"],
    )
    fitted = BVAR(N_LAGS, model, False, optimisation_method="none")
    fitted.sample(
        data,
        N_draws=8,
        data_transformation={"first": "log_diff"},
        progressbar=False,
        random_state=SEED,
    )

    state_before = {
        "beta": fitted.beta.copy(),
        "sigma": fitted.sigma.copy(),
        "beta_point": fitted.beta_point.copy(),
        "sigma_point": fitted.sigma_point.copy(),
        "df_data": fitted.df_data.copy(),
        "data": fitted.data.copy(),
        "freq": fitted.freq,
        "covid_indices": fitted.covid_indices.copy(),
        "vars_in_levels": fitted.vars_in_levels.copy(),
        "soc_": fitted.soc_,
        "sur_": fitted.sur_,
        "data_transformation": deepcopy(fitted.data_transformation),
        "model": deepcopy(fitted.model),
    }

    gapped = data.drop(index=index[10])
    with pytest.raises(ValueError, match="regular"):
        fitted.sample(gapped, N_draws=8, progressbar=False)

    np.testing.assert_array_equal(fitted.beta, state_before["beta"])
    np.testing.assert_array_equal(fitted.sigma, state_before["sigma"])
    np.testing.assert_array_equal(fitted.beta_point, state_before["beta_point"])
    np.testing.assert_array_equal(fitted.sigma_point, state_before["sigma_point"])
    pd.testing.assert_frame_equal(fitted.df_data, state_before["df_data"])
    np.testing.assert_array_equal(fitted.data, state_before["data"])
    assert fitted.freq == state_before["freq"]
    np.testing.assert_array_equal(fitted.covid_indices, state_before["covid_indices"])
    np.testing.assert_array_equal(fitted.vars_in_levels, state_before["vars_in_levels"])
    assert fitted.soc_ == state_before["soc_"]
    assert fitted.sur_ == state_before["sur_"]
    assert fitted.data_transformation == state_before["data_transformation"]
    np.testing.assert_equal(
        fitted.model.pars.__dict__, state_before["model"].pars.__dict__
    )
