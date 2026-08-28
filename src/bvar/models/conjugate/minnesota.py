"""
Minnesota Prior — Conjugate Formulation
========================================

Returns a per-equation ``(k, k)`` precision matrix.  Cross-variable scaling
Σ absorbs it in the Kronecker product ``Σ ⊗ V_A⁻¹``.

References
----------
Litterman, R. B. (1986). Forecasting with Bayesian vector autoregressions.
Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection for
    vector autoregressions.
"""

from typing import Any, List, Tuple

import numpy as np

from ...utils import get_dimensions
from ..common import _build_prior_mean


def prior_minnesota(
    data: np.ndarray,
    p: int,
    covid_indices: np.ndarray,
    levels: List[bool],
    prior_pars: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct Minnesota prior for the **natural conjugate** model.

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

    Returns
    -------
    mean_0 : np.ndarray
        Prior mean vector with shape ``(nk,)``.
    variance_inv_0 : np.ndarray
        Per-equation prior precision matrix V_A⁻¹ with shape ``(k, k)``.
    """
    _, n, k, nk, h = get_dimensions(data, p, covid_indices)

    lambda_constant = np.array(prior_pars.lambda_constant).reshape(1, -1)
    c1 = prior_pars.c1
    c3 = prior_pars.c3
    lambda_covid = prior_pars.lambda_covid

    ols_mse = np.diag(prior_pars.S_0) / (prior_pars.nu_0 - n - 1)

    mean_0 = _build_prior_mean(n, k, nk, levels)

    lag_decay = 1 / np.arange(1, p + 1) ** c3

    # Shrinkage: c1² / ols_mse  (same for own and cross lags)
    shrinkage_mat = (c1**2 / ols_mse).reshape(1, -1)

    var_lags = shrinkage_mat
    for i in range(1, p):
        var_lags = np.concatenate([var_lags, shrinkage_mat * lag_decay[i]], axis=1)

    if h > 0:
        var_covid = (np.zeros(h) + lambda_covid).reshape(1, -1)
        var = np.concatenate([lambda_constant, var_lags, var_covid], axis=1)
    else:
        var = np.concatenate([lambda_constant.reshape(1, -1), var_lags], axis=1)

    variance_inv_0 = np.diag(1.0 / var.flatten())  # (k, k)

    return mean_0, variance_inv_0
