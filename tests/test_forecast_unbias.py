import numpy as np
import pandas as pd

from bvar.BVAR import BVAR
from bvar.models import NaturalConjugate
from bvar.utils import simulate_var


def model_and_test_variables():
    # Generate synthetic data
    T = 2000  # Time periods
    n = 2  # Number of variables
    n_lags = 5  # Number of lags
    H = 12

    N_test = T // 2
    N_train = T - N_test - H

    horizon = np.array(np.arange(1, H + 1)).reshape(-1, 1)

    covid = False
    levels = False
    seed = 1234
    data, true_b, true_sigma, _ = simulate_var(
        T, n, n_lags, covid=covid, levels=levels, seed=seed
    )

    # set model
    model = NaturalConjugate(
        minnesota=True,
        soc=False,
        sur=False,
        covid=covid,
    )

    # Estimate BVAR
    bvar = BVAR(n_lags, model, levels, optimisation_method="ml")
    bvar.optimise_hyperparameters(data, nb_restart=5, random_state=seed)
    bvar.sample(data, N_draws=10000, random_state=seed)

    return bvar, N_test, N_train, H, n, horizon


def test_recursive_forecast_is_unbiased():
    bvar, N_test, N_train, H, n, horizon = model_and_test_variables()

    data = bvar.data.copy()  # BVAR.__init__ creates a numpy version
    error = []
    for t in range(N_test):
        data_t = data[: N_train + t]
        target_t = data[N_train + t : N_train + t + H]

        bvar.data = data_t  # no issue with changing the data because no covid
        bvar.recursive_forecast(H=H, point_only=True, progressbar=False)

        forecasts = bvar.forecast_unconditional[0][-H:, :]
        error_t = forecasts - target_t
        error.append(np.hstack([error_t, horizon]))

    column_names = [f"var_{i}" for i in range(n)] + ["horizon"]
    df = pd.DataFrame(np.vstack(error), columns=column_names)

    df_long = pd.melt(
        df,
        id_vars=["horizon"],  # columns to keep
        var_name="series",  # name for variable column
        value_name="value",  # name for value column
    )

    df_stats = (
        df_long.groupby(["horizon", "series"])["value"]
        .agg(mean="mean", sd="std")
        .reset_index()
    )

    df_stats["t_stat"] = df_stats["mean"] / (df_stats["sd"] / np.sqrt(N_test))

    assert (np.abs(df_stats["t_stat"]) < 1.96).all(), "Recursive forecast is biased"


def test_matrix_forecast_is_unbiased():
    bvar, N_test, N_train, H, n, horizon = model_and_test_variables()

    data = bvar.data.copy()  # BVAR.__init__ creates a numpy version
    error = []
    for t in range(N_test):
        data_t = data[: N_train + t]
        target_t = data[N_train + t : N_train + t + H]

        bvar.data = data_t  # no issue with changing the data because no covid
        bvar.forecast(H=H, point_only=True, progressbar=False)

        forecasts = bvar.forecast_unconditional[0][-H:, :]
        error_t = forecasts - target_t
        error.append(np.hstack([error_t, horizon]))

    column_names = [f"var_{i}" for i in range(n)] + ["horizon"]
    df = pd.DataFrame(np.vstack(error), columns=column_names)

    df_long = pd.melt(
        df,
        id_vars=["horizon"],  # columns to keep
        var_name="series",  # name for variable column
        value_name="value",  # name for value column
    )

    df_stats = (
        df_long.groupby(["horizon", "series"])["value"]
        .agg(mean="mean", sd="std")
        .reset_index()
    )

    df_stats["t_stat"] = df_stats["mean"] / (df_stats["sd"] / np.sqrt(N_test))

    assert (np.abs(df_stats["t_stat"]) < 1.96).all(), "Recursive forecast is biased"
