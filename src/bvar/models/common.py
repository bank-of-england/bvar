"""
Shared utilities for BVAR estimation models.

Functions used during prior initialisation (OLS-based residual variances)
and Minnesota prior mean construction.  These are internal to the
``models`` package and not part of the public API.
"""

from typing import List, Tuple

import numpy as np

from ..utils import _normalise_covid_indices

# ======================================================================
# Minnesota prior mean (shared by both formulations)
# ======================================================================


def _build_prior_mean(n: int, k: int, nk: int, levels: List[bool]) -> np.ndarray:
    """Build the Minnesota prior mean vector β₀.

    Zero everywhere except the first own-lag coefficient for variables in
    levels, which is set to 1 (random-walk belief).
    """
    mean_0 = np.zeros(nk)
    for i in range(n):
        if levels[i]:
            mean_0[i * k + 1 + i] = 1.0
    return mean_0


# ======================================================================
# OLS-based initialisation of the prior scale matrix S₀
# ======================================================================


def mse_ols_ar1(data: np.ndarray, covid_indices: np.ndarray) -> np.ndarray:
    """Compute MSE from variable-by-variable AR(1) regressions.

    Used to initialise the Minnesota prior scale matrix S₀ by estimating
    residual variances from simple AR(1) models.

    Parameters
    ----------
    data : np.ndarray
        Time series data with shape ``(T, n)``.
    covid_indices : np.ndarray
        Indices for COVID dummy observations.

    Returns
    -------
    ols_mse : np.ndarray
        Mean squared OLS residuals for each variable, with shape ``(n,)``.
    """
    Y, X = _AR1_Y_X(data, covid_indices)
    _, residuals = _OLS(Y, X)
    return np.mean(residuals**2, axis=0)


def ar1_mse(data: np.ndarray, covid_indices: np.ndarray) -> np.ndarray:
    """Validate and return the residual MSEs for AR(1) initialisation."""
    ar1_covid_indices = _normalise_covid_indices(
        covid_indices, data.shape[0], lag_cutoff=1
    )
    n_ar1_observations = data.shape[0] - 1
    n_ar1_regressors = 2 + len(ar1_covid_indices)
    if n_ar1_observations <= n_ar1_regressors:
        raise ValueError(
            "Data sample is too short to initialise the AR(1) residual variance."
        )

    mse = mse_ols_ar1(data, covid_indices)
    if not np.isfinite(mse).all() or np.any(mse <= np.finfo(float).eps):
        raise ValueError(
            "AR(1) initialisation produced non-positive or non-finite "
            "residual variance."
        )
    return mse


def _AR1_Y_X(
    data: np.ndarray, covid_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct dependent and regressor matrices for AR(1) regressions.

    Builds variable-by-variable AR(1) matrices used to obtain initial OLS
    residual variances for the Minnesota prior.
    """
    covid_indices = _normalise_covid_indices(covid_indices, data.shape[0], 1)
    h = len(covid_indices)
    T, n = data.shape
    T = T - 1

    Y = data[1:, :]
    X = data[:-1, :]
    X_all = X.T.reshape(n, T, 1)
    constant = np.ones((n, T, 1))
    X_all = np.concatenate([constant, X_all], axis=2)

    if h > 0:
        covid_dummies = np.zeros((data.shape[0], h))
        for i, idx in enumerate(covid_indices):
            covid_dummies[idx, i] = 1.0

        covid_dummies = covid_dummies[1:, :]
        covid_dummies = np.broadcast_to(covid_dummies, (n, T, h))

        X_all = np.concatenate([X_all, covid_dummies], axis=2)

    return Y, X_all


def _OLS(Y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Variable-by-variable OLS estimation.

    Solves independent OLS regressions, falling back to ``np.linalg.lstsq``
    when the normal equations are singular.
    """
    T, n = Y.shape
    residuals = np.zeros((T, n))
    beta_long = np.zeros((X.shape[2], n))

    for i in range(n):
        y_i = Y[:, i].copy()
        x_i = X[i, :, :].copy()

        try:
            beta = np.linalg.solve(x_i.T @ x_i, x_i.T @ y_i)
        except np.linalg.LinAlgError:
            beta, _, _, _ = np.linalg.lstsq(x_i, y_i, rcond=None)

        residuals[:, i] = y_i - x_i @ beta
        beta_long[:, i] = beta

    return beta_long.T.flatten(), residuals
