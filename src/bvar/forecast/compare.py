"""
Forecast comparison utility.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def compare_forecasts(
    df_forecast_a: pd.DataFrame,
    df_forecast_b: pd.DataFrame,
    H: int,
    labels: Optional[list[str]] = None,
    n_outturns: int = 0,
) -> pd.DataFrame:
    """
    Compare two conditional forecasts and compute revisions.

    Takes two formatted forecast DataFrames (output from ``forecast(..., format=True)``)
    and computes their differences.

    Parameters
    ----------
    df_forecast_a : pd.DataFrame
        First forecast DataFrame in long format with columns:
        date, quantile, variable, value.
    df_forecast_b : pd.DataFrame
        Second forecast DataFrame in long format (same structure).
    H : int
        Forecast horizon used for filtering.
    labels : Optional[list[str]]
        Labels for the two forecasts. Default is ["forecast_a", "forecast_b"].
    n_outturns : int
        Number of preceding outturns to retain.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame in long format containing:
        - First forecast values (labelled with labels[0])
        - Second forecast values (labelled with labels[1])
        - Differences (labelled "difference")
        With columns: date, quantile, variable, value, type.
    """
    if labels is None:
        labels = ["forecast_a", "forecast_b"]

    # keep the forecast horizon plus, optionally, some preceding outturns (which
    # are already present in the formatted forecast as the in-sample history)
    keep = H + n_outturns
    df_forecast_a = df_forecast_a[
        df_forecast_a["date"] > df_forecast_a["date"].max() - keep
    ]
    df_forecast_b = df_forecast_b[
        df_forecast_b["date"] > df_forecast_b["date"].max() - keep
    ]

    key_cols = ["date", "quantile", "variable"]
    df_m = df_forecast_a.merge(df_forecast_b, on=key_cols, suffixes=("_a", "_b"))

    df_a = df_m[key_cols].copy()
    df_a["value"] = df_m["value_a"].values
    df_a["type"] = labels[0]

    df_b = df_m[key_cols].copy()
    df_b["value"] = df_m["value_b"].values
    df_b["type"] = labels[1]

    df_diff = df_m[key_cols].copy()
    df_diff["value"] = (df_m["value_b"] - df_m["value_a"]).values
    df_diff["type"] = "difference"

    return pd.concat([df_a, df_b, df_diff], ignore_index=True)
