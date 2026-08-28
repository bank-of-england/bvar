"""
Miscellaneous Plotting Utilities
================================

Standalone plotting functions for forecast comparisons, density estimation,
and histogram visualisation.
"""

from typing import Optional

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def plot_delta_forecast(
    df: pd.DataFrame,
    var_names: Optional[str | list[str]] = None,
    title: str = "Forecast revision",
    figsize_per_plot: tuple = (5, 3.5),
    show: str = "difference",
    n_rows: int = 1,
    metric_labels: Optional[dict] = None,
    extra_data: Optional[pd.DataFrame] = None,
) -> None:
    """
    Plot forecast comparisons from the DataFrame produced by ``compare_forecasts``.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame with columns [date, quantile, variable, value, type]
        returned by :func:`compare_forecasts`.
    var_names : Optional[str | list[str]]
        Variable(s) to plot. Defaults to all variables.
    title : str
        Figure suptitle.
    figsize_per_plot : tuple
        ``(width, height)`` of each subplot panel.
    show : str
        ``"difference"`` (default) plots the revision;
        ``"forecasts"`` overlays both forecast series.
    n_rows : int
        Number of rows in the subplot grid. Default is 1.
    metric_labels : Optional[dict]
        Mapping ``variable -> str`` appended to each subplot title (e.g. the
        reporting transformation such as ``"yoy"``/``"qoq"``/``"levels"``).
    extra_data : Optional[pd.DataFrame]
        Long-format DataFrame with columns ``[date, variable, value]`` (and
        optionally ``series`` and ``quantile``) drawn as extra dashed line(s)
        per subplot when ``show="forecasts"`` (e.g. the mpr forecast). If a
        ``series`` column is present, one line is drawn per series and labelled
        with its name; otherwise a single line labelled ``"extra"`` is drawn.
        The method aligns dates to the plotted timeline and ignores dates outside
        the axis.
        ignored.

    Raises
    ------
    ValueError
        If ``show`` is not ``"difference"`` or ``"forecasts"``.
    """
    if show not in ("difference", "forecasts"):
        raise ValueError(f"show must be 'difference' or 'forecasts', got '{show}'")

    if var_names is None:
        plot_vars = list(df["variable"].unique())
    else:
        plot_vars = [var_names] if isinstance(var_names, str) else list(var_names)

    quantiles = sorted(df["quantile"].unique())
    q_lo, q_mid, q_hi = quantiles[0], quantiles[len(quantiles) // 2], quantiles[-1]

    forecast_types = [t for t in df["type"].unique() if t != "difference"]

    n = len(plot_vars)
    n_rows = max(1, min(n_rows, n))
    n_cols = int(np.ceil(n / n_rows))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows),
        squeeze=False,
    )
    axes = axes.flatten()

    for ax, v in zip(axes, plot_vars):

        def q(type_, level):
            mask = (
                (df["variable"] == v)
                & (df["type"] == type_)
                & (df["quantile"] == level)
            )
            return df[mask].sort_values("date")["value"].values

        dates = (
            df[
                (df["variable"] == v)
                & (df["type"] == "difference")
                & (df["quantile"] == q_mid)
            ]
            .sort_values("date")["date"]
            .values
        )
        x = range(len(dates))

        if show == "difference":
            ax.fill_between(
                x,
                q("difference", q_lo),
                q("difference", q_hi),
                alpha=0.2,
            )
            ax.plot(x, q("difference", q_mid))
        else:
            for t in forecast_types:
                ax.fill_between(x, q(t, q_lo), q(t, q_hi), alpha=0.2, label=t)
                ax.plot(x, q(t, q_mid))
            if extra_data is not None:
                e = extra_data[extra_data["variable"] == v]
                if "quantile" in e.columns:
                    e = e[e["quantile"] == q_mid]
                pos = {d: i for i, d in enumerate(dates)}
                if "series" in e.columns:
                    groups = e.groupby("series", sort=False)
                else:
                    groups = [("extra", e)]
                for series_name, g in groups:
                    pts = [
                        (pos[d], val)
                        for d, val in zip(g["date"], g["value"])
                        if d in pos
                    ]
                    if pts:
                        xs, ys = zip(*sorted(pts))
                        ax.plot(xs, ys, label=str(series_name))
            ax.legend()

        subplot_title = v
        if metric_labels is not None and metric_labels.get(v):
            subplot_title = f"{v} ({metric_labels[v]})"
        ax.set_title(subplot_title)
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(d) for d in dates])

    # hide any unused axes in the grid
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title)
    plt.show()


def plot_density(
    data: np.ndarray,
    labels: Optional[list[str]] = None,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
    bw: Optional[float | str] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Plot KDEs for multiple series from a 2D data array.

    Parameters
    ----------
    data : np.ndarray
        Data for which to compute and plot KDEs, with shape
        ``(n_samples, n_series)``.
    labels : Optional[list[str]]
        Labels for each series. If None, uses "Series 0", "Series 1", etc.
    title : Optional[str]
        Title for the plot.
    figsize : tuple
        Figure size (width, height).
    bw : Optional[float | str]
        Bandwidth for KDE. If None, uses default ('scott').
    ax : Optional[matplotlib.axes.Axes]
        Axes object to plot on. If None, creates a new figure and axes.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.

    Raises
    ------
    ValueError
        If ``labels`` does not match the number of data series.
    """
    data = np.asarray(data)
    n_samples, n_series = data.shape

    if labels is None:
        labels = [f"Series {i}" for i in range(n_series)]
    elif len(labels) != n_series:
        raise ValueError("Length of labels must match number of series")

    # Use provided ax or create new figure/axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Find global min/max for x_grid
    x_min = np.min(data)
    x_max = np.max(data)
    x_grid = np.linspace(x_min, x_max, 500)

    for i in range(n_series):
        bw_method = bw if bw is not None else "scott"
        kde = gaussian_kde(data[:, i], bw_method=bw_method)
        density = kde(x_grid)
        ax.fill_between(
            x_grid,
            density,
            alpha=0.1,
        )
        ax.plot(
            x_grid,
            density,
            label=labels[i],
            alpha=0.7,
        )

    ax.legend()
    if title:
        ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")

    return fig, ax


def plot_histogram(
    data: np.ndarray,
    labels: Optional[list[str]] = None,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
    bins: int = 30,
    alpha: float = 0.5,
    colors: Optional[list] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Plot histograms for multiple series from a 2D data array.

    Parameters
    ----------
    data : np.ndarray
        Data for which to plot histograms, with shape
        ``(n_samples, n_series)``.
    labels : Optional[list[str]]
        Labels for each series. If None, uses "Series 0", "Series 1", etc.
    title : Optional[str]
        Title for the plot.
    figsize : tuple
        Figure size (width, height).
    bins : int
        Number of bins for the histograms. Default is 30.
    alpha : float
        Opacity of the histogram bars.  Default is 0.5.
    colors : Optional[list]
        Colours for each series.  If None, uses the default colour cycle.
    ax : Optional[matplotlib.axes.Axes]
        Axes object to plot on. If None, creates a new figure and axes.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.

    Raises
    ------
    ValueError
        If ``labels`` does not match the number of data series.
    """
    data = np.asarray(data)
    n_samples, n_series = data.shape

    if labels is None:
        labels = [f"Series {i}" for i in range(n_series)]
    elif len(labels) != n_series:
        raise ValueError("Length of labels must match number of series")

    # Use provided ax or create new figure/axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for i in range(n_series):
        ax.hist(
            data[:, i],
            bins=bins,
            alpha=alpha,
            label=labels[i],
            density=True,  # Normalize to density for comparability
        )

    ax.legend()
    if title:
        ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")

    return fig, ax
