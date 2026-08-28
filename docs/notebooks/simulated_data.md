---
title: Simulated Data
marimo-version: 0.24.0
---

```python {.marimo}
import marimo as mo
```

#### Load the library

```python {.marimo}
import bvar as bv
import numpy as np
import matplotlib.pyplot as plt
```

#### Simulate a VAR

```python {.marimo}
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
```

#### Plot the data

```python {.marimo}
data.plot()
plt.legend()
plt.show()
```

### Bayesian estimation

#### (1) Create a Bayesian VAR instance by combining data, model and number of lags.

```python {.marimo}
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
```

#### (2) Optimise hyperparameters

```python {.marimo}
bvar.optimise_hyperparameters(data)
```

#### (3) Sampling

```python {.marimo}
bvar.sample(data, N_draws=N_draws)
```

### Inspecting results
#### Posterior for the mean

```python {.marimo}
bv.mcmc_posterior(bvar.beta, true_b.flatten())
```

#### Plot fitted values

```python {.marimo}
bvar.compute_fitted_values()
bvar.plot_fitted_values()
```

### Forecasting
#### Unconditional forecasting

```python {.marimo}
H = 13
t = T - p  # First forecast period

# forecast with constraints
bvar.forecast(H=H, N_draws=N_draws)
bvar.plot_forecast(
    alpha=0.05,
)

forecasts_unconditional = bvar.forecast_unconditional[:, t, :]
```

#### Conditional forecasting - Hard constraints

```python {.marimo}
# mean constraints
_mean_constrained = np.full((H, n), np.nan)
_mean_constrained[:, 1] = 0.0
bvar.forecast(H=H, constraint_mean=_mean_constrained, N_draws=N_draws)
# forecast with constraints
bvar.plot_forecast(alpha=0.05)
forecasts_hard = bvar.forecast_conditional[:, t, 0]
```

#### Conditional forecasting - Constraints with Normal distribution (soft constraints)

```python {.marimo}
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
```

#### Conditional forecasting - Constraints with Skew Normal distribution

```python {.marimo}
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
```

#### Plot forecasts and constraints

```python {.marimo}
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
```