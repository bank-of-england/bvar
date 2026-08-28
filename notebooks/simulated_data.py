import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Load the library
    """)
    return


@app.cell
def _():
    import bvar as bv
    import numpy as np
    import matplotlib.pyplot as plt

    return bv, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Simulate a VAR
    """)
    return


@app.cell
def _(bv, np):
    n = 2
    T = 200
    p = 1
    N_draws = 10000
    covid = False
    levels = False
    ar_mat = np.array([[0.7, 0.5], [0.2, 0.5]])
    constant = np.array([0.0, 0.0])
    Sigma = np.eye(n)
    Sigma[0, 1] = 0.95
    Sigma[1, 0] = 0.95
    data, true_b, true_sigma, _ = bv.simulate_var(
        T,
        n,
        p,
        covid,
        levels,
        ar_mat=ar_mat,
        Sigma=Sigma,
        constant=constant,
        seed=123,
    )
    return N_draws, T, covid, data, levels, n, p, true_b


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot the data
    """)
    return


@app.cell
def _(data, plt):
    data.plot()
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bayesian estimation

    #### (1) Create a Bayesian VAR instance by combining data, model and number of lags.
    """)
    return


@app.cell
def _(bv, covid, levels, p):
    # set sampling model
    model = bv.NaturalConjugate(
        minnesota=True,
        soc=False,
        sur=False,
        covid=covid,
    )

    # Create a BVAR instance by combining data with model and features like number of lags. The data should a pandas dataframe.
    bvar = bv.BVAR(
        p,
        model,
        not levels,
        optimisation_method="ml",
        random_state=123,
    )
    return (bvar,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### (2) Optimise hyperparameters
    """)
    return


@app.cell
def _(bvar, data):
    bvar.optimise_hyperparameters(data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### (3) Sampling
    """)
    return


@app.cell
def _(N_draws, bvar, data):
    bvar.sample(data, N_draws=N_draws)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Inspecting results
    #### Posterior for the mean
    """)
    return


@app.cell
def _(bv, bvar, true_b):
    bv.mcmc_posterior(bvar.beta, true_b.flatten())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot fitted values
    """)
    return


@app.cell
def _(bvar):
    bvar.compute_fitted_values()
    bvar.plot_fitted_values()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Forecasting
    #### Unconditional forecasting
    """)
    return


@app.cell
def _(N_draws, T, bvar, p):
    H = 13
    t = T - p  # First forecast period

    # forecast with constraints
    bvar.forecast(H=H, N_draws=N_draws)
    bvar.plot_forecast(
        alpha=0.05,
    )

    forecasts_unconditional = bvar.forecast_unconditional[:, t, :]
    return H, forecasts_unconditional, t


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Conditional forecasting - Hard constraints
    """)
    return


@app.cell
def _(H, N_draws, bvar, n, np, t):
    # mean constraints
    _mean_constrained = np.full((H, n), np.nan)
    _mean_constrained[:, 1] = 0.0
    bvar.forecast(H=H, constraint_mean=_mean_constrained, N_draws=N_draws)
    # forecast with constraints
    bvar.plot_forecast(alpha=0.05)
    forecasts_hard = bvar.forecast_conditional[:, t, 0]
    return (forecasts_hard,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Conditional forecasting - Constraints with Normal distribution (soft constraints)
    """)
    return


@app.cell
def _(H, N_draws, bvar, n, np, t):
    # mean constraints
    _mean_constrained = np.full((H, n), np.nan)
    _mean_constrained[:, 1] = 0.0
    _var_constrained = np.full((H, n), np.nan)
    # var constraints
    _var_constrained[:, 1] = 1.0
    bvar.forecast(
        H=H,
        constraint_mean=_mean_constrained,
        constraint_variance=_var_constrained,
        N_draws=N_draws,
    )
    bvar.plot_forecast(alpha=0.05)
    # forecast with constraints
    forecasts_normal = bvar.forecast_conditional[:, t, 0]
    constraint_normal = bvar.forecast_conditional[:, t, 1]
    return constraint_normal, forecasts_normal


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Conditional forecasting - Constraints with Skew Normal distribution
    """)
    return


@app.cell
def _(H, N_draws, bvar, n, np, t):
    # mean constraint
    _mean_constrained = np.full((H, n), np.nan)
    _mean_constrained[:, 1] = 0.4
    _var_constrained = np.full((H, n), np.nan)
    # var constraints
    _var_constrained[:, 1] = 3
    shape_constrained = np.full((H, n), np.nan)
    shape_constrained[:, 1] = -10
    bvar.forecast(
        H=H,
        constraint_mean=_mean_constrained,
        constraint_variance=_var_constrained,
        constraint_shape=shape_constrained,
        N_draws=N_draws,
    )
    bvar.plot_forecast(alpha=0.05)
    forecasts_skew = bvar.forecast_conditional[:, t, 0]
    # forecast with constraints
    # Get conditional forecasts
    constraint_skew = bvar.forecast_conditional[:, t, 1]
    return constraint_skew, forecasts_skew


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot forecasts and constraints
    """)
    return


@app.cell
def _(
    bv,
    constraint_normal,
    constraint_skew,
    forecasts_hard,
    forecasts_normal,
    forecasts_skew,
    forecasts_unconditional,
    np,
    plt,
):
    colors = plt.cm.tab10(np.linspace(0, 1, 7))
    colors = np.vstack(
        [colors[0:3,], np.array([[0, 0, 0, 1]])]
    )  # Add black (RGBA) as the last color

    # forecasts
    forecast_data = np.column_stack(
        [
            forecasts_unconditional[: forecasts_hard.shape[0], 0],
            forecasts_hard,
            forecasts_normal,
            forecasts_skew,
        ]
    )
    labels_forecasts = [
        "Unconditional",
        "Conditional on Hard Constraint",
        "Conditional on Normal Constraint",
        "Conditional on Skewed Constraint",
    ]

    # constraints
    constraint_data = np.column_stack(
        [
            forecasts_unconditional[: forecasts_hard.shape[0], 1],
            constraint_normal,
            constraint_skew,
        ]
    )
    labels_constraints = [
        "Unconditional forecast",
        "Normal Constraint",
        "Skewed Constraint",
    ]

    # Create stacked figure with two rows
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)

    # Forecast densities (top)
    bv.plot_histogram(
        forecast_data,
        labels_forecasts,
        title="Conditional Forecast Densities",
        bins=100,
        alpha=0.4,
        colors=colors[[2, 3, 0, 1]],
        ax=axs[0],
    )

    # Constraint densities (bottom)
    bv.plot_histogram(
        constraint_data,
        labels_constraints,
        title="Constraints",
        bins=100,
        alpha=0.4,
        colors=colors[[2, 0, 1]],
        ax=axs[1],
    )
    dash_line = axs[1].axvline(
        0, color=colors[-1], linestyle="--", linewidth=5, label="Hard Constraint"
    )

    # Add legend for the dashed line
    handles, labels = axs[1].get_legend_handles_labels()
    handles.append(dash_line)
    axs[1].legend(handles, labels)

    plt.rcParams["font.size"] = 18
    plt.show()
    return


if __name__ == "__main__":
    app.run()
