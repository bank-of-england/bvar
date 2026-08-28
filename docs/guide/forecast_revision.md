# Forecast Revision Analysis

## Overview

Forecast revision analysis — sometimes called delta forecast or counterfactual analysis — decomposes the difference between two forecasts into the contribution of each conditioning assumption. It answers the question: *by how much, and in which direction, does the forecast change when we move from one set of constraints to another?*

## Motivation

In a conditional forecasting workflow, the analyst typically produces a baseline forecast conditioned on a set of external assumptions (e.g. a prescribed path for Bank Rate, trade policy assumptions, or nowcast values). A natural follow-up question is what happens to the forecast for variables of interest when one or more of those assumptions changes.

Rather than interpreting the two forecasts individually, forecast revision analysis directly quantifies the revision — the difference in the predictive distribution across the two scenarios — making it easier to communicate the impact of changing assumptions.

## Methodology

Given two forecasts, a baseline and an alternative, define the revision to variable $i$ at horizon $h$ as:

$$\Delta \hat{y}_{i,h} = \hat{y}_{i,h}^{\text{alternative}} - \hat{y}_{i,h}^{\text{baseline}}$$

Both forecasts are posterior predictive distributions, so their difference also forms a distribution. Its spread measures uncertainty around the directional impact of changed assumptions.

## Typical Use Cases

- **Conditional vs unconditional:** quantify how much of the forecast is driven by the conditioning assumptions versus the model's own dynamics.
- **Scenario analysis:** compare a central projection against an upside or downside scenario with different external assumptions.
- **Sensitivity analysis:** test how sensitive the forecast for GDP or inflation is to a change in a single conditioning variable.

## Workflow

1. Produce a **baseline forecast** — typically an unconditional or lightly conditioned forecast — and save the result.
2. Produce an **alternative forecast** with a modified set of constraints.
3. Pass both formatted forecast DataFrames to `compare_forecasts` to compute the revision matrix.
4. Use `plot_delta_forecast` to visualise the revision for a subset of variables, showing both the baseline and alternative forecasts and the gap between them.

The `show` argument in `plot_delta_forecast` controls what is displayed: `"forecasts"` overlays the two forecast paths, while `"difference"` plots only the revision.

```python
import bvar as bv

# Step 1: produce and save the baseline (unconditional) forecast
bvar.forecast(H=8, transformations=transformations, format=True)
df_baseline = model.df_forecasts_unconditional.copy()

# Step 2: produce the alternative forecast with a constraint
bvar.forecast(
    H=8,
    constraint_mean=constraint_mean,
    transformations=transformations,
    format=True,
)

# Step 3: compute the revision
df_revision = bv.compare_forecasts(
    df_baseline,
    bvar.df_forecasts_conditional,
    H=8,
)

# Step 4: plot
bv.plot_delta_forecast(
    df_revision,
    var_names=["gdp", "cpi", "unemp"],
    title="Forecast revision: conditional vs baseline",
    show="forecasts",
)
```

## Interpretation

- A positive revision for GDP means the alternative scenario implies a higher growth forecast than the baseline at that horizon.
- The width of the revision distribution reflects both model uncertainty and the degree of co-movement between the constrained and target variables.
- Near-zero revisions show that the variable responds little to the changed constraint. Weak correlation with the constrained variable or a small constraint change relative to forecast uncertainty can produce this result.
