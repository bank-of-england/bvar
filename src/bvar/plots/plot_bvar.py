"""
BVAR Plotting Module
====================

Provides plotting methods for fitted values and forecast visualisation
as a mixin class for the BVAR model.
"""

from typing import Optional

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PlotBVAR:
    """Plotting methods for BVAR class"""

    def plot_fitted_values(
        self,
        confidence_level: float = 95,
        max_cols: int = 3,
        figsize_per_plot: tuple = (5, 3),
        var_names: Optional[str | list[str]] = None,
    ) -> matplotlib.figure.Figure:
        """
        Plot fitted values with credible intervals against actual data.

        Parameters
        ----------
        confidence_level : float
            Confidence level for credible intervals (in percent).
        max_cols : int
            Maximum number of subplot columns per row.
        figsize_per_plot : tuple
            Size of each subplot (width, height).
        var_names : Optional[str | list[str]]
            Column name(s) to plot. If ``None``, the method plots every series.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the fitted-value subplots.

        Raises
        ------
        ValueError
            If ``var_names`` contains invalid column names.
        """
        if self.fitted_values is None:
            self.compute_fitted_values()

        # Set percentiles based on confidence level
        lower_pct = (100 - confidence_level) / 2
        upper_pct = 100 - lower_pct

        # Calculate statistics across MCMC draws
        mean_fitted = np.mean(self.fitted_values, axis=0)
        lower_bound = np.percentile(self.fitted_values, lower_pct, axis=0)
        upper_bound = np.percentile(self.fitted_values, upper_pct, axis=0)

        # All variable names
        all_names = list(self.df_data.columns)

        # Validate var_names argument
        if var_names is None:
            selected = list(range(self.n))
            labels = all_names
        else:
            # Convert single str to list
            if isinstance(var_names, str):
                var_names = [var_names]
            elif not isinstance(var_names, list) or not all(
                isinstance(x, str) for x in var_names
            ):
                raise ValueError(
                    "var_names must be a string or list of strings (column names)."
                )

            missing = [name for name in var_names if name not in all_names]
            if missing:
                raise ValueError(
                    f"The following var_names are not in df_data.columns: {missing}"
                )
            selected = [all_names.index(name) for name in var_names]
            labels = var_names

        n_series = len(selected)
        n_cols = min(max_cols, n_series)
        n_rows = int(np.ceil(n_series / n_cols))
        figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)

        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        # add dates (same logic as plot_forecast)
        dates = self.df_data.index[self.n_lags :].to_timestamp()

        for plot_idx, i in enumerate(selected):
            ax = axes[plot_idx]

            # Plot credible interval
            ax.fill_between(
                dates,
                lower_bound[:, i],
                upper_bound[:, i],
                alpha=0.3,
                label=f"{confidence_level}% Credible Interval",
            )

            # Plot mean fitted values
            ax.plot(
                dates,
                mean_fitted[:, i],
                label="Mean Fitted",
            )

            # Plot actual data (adjust for lag structure)
            ax.plot(
                dates,
                self.data[self.n_lags :, i],
                label="Actual Data",
                alpha=0.75,
            )

            ax.set_title(labels[plot_idx])
            ax.legend()

        # Hide unused axes
        for j in range(n_series, len(axes)):
            fig.delaxes(axes[j])

        return fig

    def plot_forecast(
        self,
        alpha: float = 0.05,
        max_cols: int = 3,
        figsize_per_plot: tuple = (5, 3),
        var_names: Optional[str | list[str]] = None,
        from_date: Optional[pd.Period | str] = None,
    ) -> matplotlib.figure.Figure:
        """
        Plot forecast means and credible intervals for each series.

        Parameters
        ----------
        alpha : float
            Significance level for credible intervals (e.g., 0.05 for 95% interval).
            Default is 0.05.
        max_cols : int
            Maximum number of subplot columns per row.
        figsize_per_plot : tuple
            Size of each subplot (width, height).
        var_names : Optional[str | list[str]]
            Column name(s) to plot. If ``None``, the method plots every series.
        from_date : Optional[pd.Period | str]
            Starting date for the forecast plot. If None, uses default date range.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the forecast subplots.

        Raises
        ------
        RuntimeError
            If forecast results are not available.
        ValueError
            If ``var_names`` contains invalid column names.
        """

        lower_pct = alpha * 100
        upper_pct = (1 - alpha) * 100

        # All variable names
        all_names = list(self.df_data.columns)

        # Validate var_names argument
        if var_names is None:
            selected = list(range(self.n))
            labels = all_names
        else:
            # Convert single str to list
            if isinstance(var_names, str):
                var_names = [var_names]
            elif not isinstance(var_names, list) or not all(
                isinstance(x, str) for x in var_names
            ):
                raise ValueError(
                    "var_names must be a string or list of strings (column names)."
                )

            missing = [name for name in var_names if name not in all_names]
            if missing:
                raise ValueError(
                    f"The following var_names are not in df_data.columns: {missing}"
                )
            selected = [all_names.index(name) for name in var_names]
            labels = var_names

        n_series = len(selected)
        n_cols = min(max_cols, n_series)
        n_rows = int(np.ceil(n_series / n_cols))
        figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        forecast_draws = None
        forecast_unconditional_med = None

        if self.forecast_unconditional is not None:
            forecast_draws = self.forecast_unconditional

        if self.forecast_conditional is not None:
            forecast_draws = self.forecast_conditional
            forecast_unconditional_med = self.forecast_unconditional[0]
            # when doing conditional forecasts we just show the first unconditional draw.

        if forecast_draws is None:
            raise RuntimeError(
                "No forecast available to plot. Call forecast() or "
                "recursive_forecast() first."
            )

        # add dates
        dates_obs = self.df_data.index[self.n_lags :]
        dates_forecasts = pd.period_range(
            start=dates_obs[-1] + 1, periods=self.H, freq=self.freq
        )
        dates_period = dates_obs.append(dates_forecasts)
        if from_date is not None:
            from_period = pd.Period(from_date, freq=self.freq)
            dates_period = dates_period[dates_period >= from_period]
        dates = dates_period.to_timestamp()

        for plot_idx, i in enumerate(selected):
            forecast_series = forecast_draws[:, -len(dates) :, i]

            median = np.percentile(forecast_series, 50, axis=0)
            lower = np.percentile(forecast_series, lower_pct, axis=0)
            upper = np.percentile(forecast_series, upper_pct, axis=0)

            ax = axes[plot_idx]
            ax.plot(dates, median, label="Median Forecast")
            ax.fill_between(
                dates,
                lower,
                upper,
                alpha=0.25,
                label=f"{int((1 - alpha) * 100)}% Interval",
            )
            # add unconditional median if conditional forecast
            if forecast_unconditional_med is not None:
                ax.plot(
                    dates,
                    forecast_unconditional_med[-len(dates) :, i],
                    label="Unconditional Median",
                )

            ax.set_title(labels[plot_idx])
            ax.legend()

        # Hide unused axes.
        for j in range(n_series, len(axes)):
            fig.delaxes(axes[j])

        return fig
