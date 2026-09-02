"""Tests for compare_forecasts and plot_delta_forecast."""

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from bvar import compare_forecasts, plot_delta_forecast


@pytest.fixture(scope="module")
def two_forecasts(bvar):
    bvar.forecast(H=4, N_draws=500, random_state=1234, format=True)
    df_a = bvar.df_forecasts_unconditional.copy()

    cm = np.full((4, bvar.n), np.nan)
    cm[:, 0] = bvar.data[-1, 0]
    bvar.forecast(H=4, constraint_mean=cm, N_draws=500, random_state=1234, format=True)
    df_b = bvar.df_forecasts_conditional.copy()
    return df_a, df_b


def test_difference_is_b_minus_a(two_forecasts):
    df_a, df_b = two_forecasts
    out = compare_forecasts(df_a, df_b, H=4)
    assert set(out["type"].unique()) == {"forecast_a", "forecast_b", "difference"}
    key = ["date", "quantile", "variable"]
    a = out[out.type == "forecast_a"].set_index(key)["value"]
    b = out[out.type == "forecast_b"].set_index(key)["value"]
    d = out[out.type == "difference"].set_index(key)["value"]
    pd.testing.assert_series_equal(d, (b - a), check_names=False)


def test_difference_with_self_is_zero(two_forecasts):
    df_a, _ = two_forecasts
    out = compare_forecasts(df_a, df_a, H=4)
    assert np.allclose(out.loc[out.type == "difference", "value"], 0.0)


def test_labels_override(two_forecasts):
    df_a, df_b = two_forecasts
    out = compare_forecasts(df_a, df_b, H=4, labels=["base", "scenario"])
    assert {"base", "scenario", "difference"} == set(out["type"].unique())


def test_n_outturns_retains_history(two_forecasts):
    df_a, df_b = two_forecasts
    n0 = compare_forecasts(df_a, df_b, H=4, n_outturns=0)["date"].nunique()
    n3 = compare_forecasts(df_a, df_b, H=4, n_outturns=3)["date"].nunique()
    assert n3 == n0 + 3


@pytest.mark.parametrize("show", ["difference", "forecasts"])
def test_plot_delta_forecast_returns_figure(two_forecasts, show):
    df_a, df_b = two_forecasts
    out = compare_forecasts(df_a, df_b, H=4)
    fig = plot_delta_forecast(
        out, var_names=list(out["variable"].unique())[:2], show=show
    )
    assert fig is not None


def test_plot_delta_forecast_rejects_bad_show(two_forecasts):
    df_a, df_b = two_forecasts
    out = compare_forecasts(df_a, df_b, H=4)
    with pytest.raises(ValueError):
        plot_delta_forecast(out, show="nonsense")
