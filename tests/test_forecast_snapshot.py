"""Aggregate regression check for the formatted unconditional forecast table.

An earlier version pinned an exact CSV snapshot of every forecast value.
That comparison is not portable: the ``bvar`` fixture runs an L-BFGS-B
hyperparameter optimisation whose iterate path depends on the BLAS/LAPACK
backend, so macOS (Accelerate) and the Linux CI runner (OpenBLAS) converge
to slightly different hyperparameters and every downstream forecast value
differs past the sixth decimal.

Instead we assert the table's structure, the quantile-ordering invariant,
and a few aggregate statistics within tolerance. Aggregates over the
203-date column wash out per-path floating-point noise, so they still
catch a genuine regression without being tied to one platform.
"""

import numpy as np

# Group statistics of ``value`` by (variable, quantile), recorded from a
# local run. Compared with a loose tolerance, not for exact equality.
EXPECTED_GROUP_MEAN = {
    (0, 0.16): 0.121596,
    (0, 0.50): 0.143210,
    (0, 0.84): 0.165133,
    (1, 0.16): 4.323342,
    (1, 0.50): 4.365600,
    (1, 0.84): 4.406923,
}
EXPECTED_GROUP_STD = {
    (0, 0.16): 1.472524,
    (0, 0.50): 1.469170,
    (0, 0.84): 1.482022,
    (1, 0.16): 3.966425,
    (1, 0.50): 3.967836,
    (1, 0.84): 3.991959,
}


def test_forecast_summary_aggregates(bvar):
    bvar.forecast(H=4, N_draws=500, random_state=1234, format=True)
    df = bvar.df_forecasts_unconditional

    # Structure.
    assert list(df.columns) == ["date", "quantile", "variable", "value"]
    assert len(df) == 1218
    dates = df["date"].astype(str)
    assert dates.nunique() == 203
    assert dates.min() == "1990Q2"
    assert dates.max() == "2040Q4"
    np.testing.assert_allclose(sorted(df["quantile"].unique()), [0.16, 0.5, 0.84])
    assert sorted(int(v) for v in df["variable"].unique()) == [0, 1]
    assert np.isfinite(df["value"]).all()

    # Quantile ordering holds row-by-row (platform-independent).
    wide = df.pivot_table(
        index=["date", "variable"], columns="quantile", values="value"
    )
    assert (wide[0.16] <= wide[0.5] + 1e-9).all()
    assert (wide[0.5] <= wide[0.84] + 1e-9).all()

    # Aggregate magnitudes within tolerance.
    grouped = df.groupby(["variable", "quantile"])["value"]
    keys = sorted(EXPECTED_GROUP_MEAN)
    np.testing.assert_allclose(
        [grouped.mean().loc[k] for k in keys],
        [EXPECTED_GROUP_MEAN[k] for k in keys],
        rtol=0.05,
        atol=0.05,
    )
    np.testing.assert_allclose(
        [grouped.std().loc[k] for k in keys],
        [EXPECTED_GROUP_STD[k] for k in keys],
        rtol=0.05,
        atol=0.05,
    )
