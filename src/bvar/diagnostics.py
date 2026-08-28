"""
Diagnostics Module for BVAR
===========================

This module contains functions for MCMC diagnostics, convergence assessment,
and visualisation of estimation results.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def mcmc_posterior(
    draws: np.ndarray,
    true_pars: Optional[np.ndarray] = None,
    max_cols: int = 3,
    figsize_per_plot: tuple = (5, 3),
) -> None:
    """
    Plot posterior distributions (histograms) for MCMC draws of parameters.

    Parameters
    ----------
    draws : np.ndarray
        Array of MCMC draws for each parameter, with shape
        ``(n_draws, n_params)``.
    true_pars : Optional[np.ndarray]
        Array of true parameter values for reference (default: None).
    max_cols : int
        Maximum number of columns in the subplot grid.
    figsize_per_plot : tuple
        Size of each subplot (width, height).

    Returns
    -------
    None
        Displays the posterior distribution plots.
    """
    # Get dimensions
    n_draws, n_params = draws.shape

    n_cols = min(max_cols, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i in range(n_params):
        axes[i].hist(
            draws[:, i],
            bins=50,
            alpha=0.7,
            density=True,
        )
        axes[i].axvline(
            np.mean(draws[:, i]),
            label="Posterior Mean",
        )
        if true_pars is not None:
            axes[i].axvline(
                true_pars[i],
                label=f"True: {true_pars[i]:.3f}",
            )
        axes[i].set_title(f"Parameter {i + 1}")
        axes[i].legend()

    # Hide unused axes
    for j in range(n_params, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Posterior Distributions")
    plt.show()
