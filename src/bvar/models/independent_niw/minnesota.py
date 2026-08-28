"""
Minnesota Prior — Independent NIW Formulation
===============================================

Returns a full-system ``(nk, nk)`` precision matrix that encodes the
``σ_i / σ_j`` cross-variable scaling directly, because the prior on β
is independent of Σ.

References
----------
Litterman, R. B. (1986). Forecasting with Bayesian vector autoregressions.
"""

from typing import Any, List, Tuple

import numpy as np

from ...utils import get_dimensions
from ..common import _build_prior_mean


def prior_minnesota_independent(
    data: np.ndarray,
    p: int,
    covid_indices: np.ndarray,
    levels: List[bool],
    prior_pars: Any,
    c2: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct Minnesota prior for the **independent NIW** model.

    Returns
    -------
    mean_0 : np.ndarray
        Prior mean vector with shape ``(nk,)``.
    variance_inv_0 : np.ndarray
        Full-system prior precision.  Cross-variable scaling ``σ_i / σ_j``
        the method encodes it directly because the prior on β is independent of Σ, with
        shape ``(nk, nk)``.

    Parameters
    ----------
    data : np.ndarray
        Input data array with shape ``(T, n)``.
    p : int
        Number of lags.
    covid_indices : np.ndarray
        Indices for COVID dummy observations.
    levels : List[bool]
        Whether each variable is in levels, with length ``n``.
    prior_pars : Any
        Prior hyperparameters.
    c2 : float
        Cross-variable shrinkage factor (Litterman 1986 uses 0.5).
    """
    _, n, k, nk, h = get_dimensions(data, p, covid_indices)

    lambda_constant = prior_pars.lambda_constant
    c1 = prior_pars.c1
    c3 = prior_pars.c3
    lambda_covid = prior_pars.lambda_covid

    ols_mse = np.diag(prior_pars.S_0) / (prior_pars.nu_0 - n - 1)
    sigma_ols = np.sqrt(ols_mse)

    mean_0 = _build_prior_mean(n, k, nk, levels)

    lag_decay = 1 / np.arange(1, p + 1) ** c3

    # Cross-variable scaled shrinkage: c1² · (σ_i/σ_j)² · (own + c2·cross)
    sig_ratio = sigma_ols[:, np.newaxis] / sigma_ols[np.newaxis, :]  # (n, n)
    own_mask = np.eye(n)
    cross_mask = 1.0 - own_mask
    shrinkage_mat = (own_mask + cross_mask * c2) * sig_ratio**2 * c1**2  # (n, n)

    # Per-equation rows: [constant, lag1_vars, lag2_vars, ..., covid_dummies]
    constant_col = np.full((n, 1), lambda_constant)

    var_lags = shrinkage_mat  # lag 1
    for ell in range(1, p):
        var_lags = np.concatenate([var_lags, shrinkage_mat * lag_decay[ell]], axis=1)

    if h > 0:
        var_covid = np.full((n, h), lambda_covid)
        var = np.concatenate([constant_col, var_lags, var_covid], axis=1)
    else:
        var = np.concatenate([constant_col, var_lags], axis=1)  # (n, k)

    variance_inv_0 = np.diag(1.0 / var.flatten())  # (nk, nk)

    return mean_0, variance_inv_0
