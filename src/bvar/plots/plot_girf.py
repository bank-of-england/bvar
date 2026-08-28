"""
GIRF Plotting Module
====================

Provides plotting methods for Generalised Impulse Response Functions
as a mixin class for the BVAR model.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


class PlotGIRF:
    """Plotting methods for Generalised Impulse Response Functions."""

    def plot_girf(
        self,
        shock_var: Optional[str | int | list] = None,
        response_var: Optional[str | int | list] = None,
        quantiles: tuple[float, float, float] = (0.16, 0.50, 0.84),
        figsize_per_plot: tuple[float, float] = (4.0, 3.0),
        max_cols: int = 4,
        title: Optional[str] = None,
        zero_line: bool = True,
    ) -> plt.Figure:
        """
        Plot posterior distributions of Generalised Impulse Response Functions.

        Displays the posterior median and a credible band for each
        (shock variable, response variable) pair.

        Parameters
        ----------
        shock_var : Optional[str | int | list]
            Variable(s) to shock.  Can be a column name, a 0-based index,
            or a list of names/indices.  If None, all variables are shown.
        response_var : Optional[str | int | list]
            Variable(s) whose responses to plot.  Same format as
            ``shock_var``.  If None, all variables are shown.
        quantiles : tuple[float, float, float]
            (lower, median, upper) quantile levels for the credible band.
            Default is (0.16, 0.50, 0.84).
        figsize_per_plot : tuple[float, float]
            (width, height) of each individual subplot.  Default is (4, 3).
        max_cols : int
            Maximum number of subplot columns per row.  Default is 4.
        title : Optional[str]
            Overall figure title. If ``None``, the method uses a default title.
        zero_line : bool
            If True, draw a horizontal zero line on each subplot.
            Default is True.

        Returns
        -------
        fig : plt.Figure
            The figure object.

        Raises
        ------
        RuntimeError
            If ``compute_girf()`` has not been called yet.
        """
        if not hasattr(self, "irf_draws") or self.irf_draws is None:
            raise RuntimeError(
                "GIRFs have not been computed yet. Call compute_girf() first."
            )

        draws = self.irf_draws
        var_names = self.irf_var_names
        H = self.irf_H
        horizons = np.arange(H + 1)

        # Resolve variable lists
        from ..girf import _resolve_var_indices

        shock_indices = _resolve_var_indices(shock_var, var_names, "shock_var")
        response_indices = _resolve_var_indices(response_var, var_names, "response_var")

        q_low, q_med, q_high = quantiles

        # Create subplots
        pairs = [(s, r) for s in shock_indices for r in response_indices]
        n_plots = len(pairs)
        n_cols = min(max_cols, n_plots)
        n_rows = int(np.ceil(n_plots / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows),
            squeeze=False,
        )
        axes_flat = axes.flatten()

        for idx, (s, r) in enumerate(pairs):
            ax = axes_flat[idx]

            med = np.quantile(draws[:, :, s, r], q_med, axis=0)
            lo = np.quantile(draws[:, :, s, r], q_low, axis=0)
            hi = np.quantile(draws[:, :, s, r], q_high, axis=0)

            ax.fill_between(horizons, lo, hi, alpha=0.25)
            ax.plot(horizons, med)

            if zero_line:
                ax.axhline(0)

            # Get response type for this variable
            response_types = self.irf_response_type
            resp_type = response_types.get(var_names[r], "raw")

            # Format response type for y-axis label
            type_labels = {
                "raw": "Response",
                "raw_cumulated": "Cumulative Response",
                "level_change": "Level Deviation",
                "pct_change": "Deviation (pp.)",
                "change_yoy": "YoY Level Deviation",
                "pct_change_yoy": "YoY Deviation (pp.)",
            }
            type_display = type_labels.get(resp_type)

            # Build shock size label for subtitle
            shock_var_name = var_names[s]
            shock_size_dict = getattr(self, "irf_shock_size", {})

            if shock_size_dict and shock_var_name in shock_size_dict:
                # User provided shock size in natural units
                shock_size_natural = shock_size_dict[shock_var_name]
                shock_label = f"{shock_size_natural:.2g} {shock_var_name}"
            else:
                # Default: 1 std-dev
                shock_label = f"1 std {shock_var_name}"

            # Subtitle with shock and response info
            subtitle = f"{shock_label} → {var_names[r]}"

            ax.set_title(subtitle)
            ax.set_xlabel("Horizon")
            ax.set_ylabel(type_display)

        # Hide unused subplots
        for idx in range(n_plots, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        label = "Generalised Impulse Response Functions"
        fig.suptitle(label)

        return fig
