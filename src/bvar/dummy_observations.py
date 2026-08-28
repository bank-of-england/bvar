from typing import Any, List, Optional, Tuple

import numpy as np

from .utils import _normalise_covid_indices

"""Dummy observations for BVARs following GLP (2015)."""


def sum_of_coefficients_prior(
    y: np.ndarray,
    n_lags: int,
    mu: float,
    levels: List[bool],
    covid_indices: np.ndarray = np.array([]),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct sum-of-coefficients dummy observations for BVAR prior.

    Parameters
    ----------
    y : np.ndarray
        Data matrix with shape ``(T, n)``.
    n_lags : int
        Number of lags.
    mu : float
        Tightness parameter for the sum-of-coefficients prior.
    levels : List[bool]
        Indicates which variables are in levels (True) or differences (False),
        with length ``n``.
    covid_indices : np.ndarray
        Indices for COVID dummies (default: empty array).

    Returns
    -------
    yd : np.ndarray
        Dummy dependent variable matrix with shape ``(n, n)``.
    xd : np.ndarray
        Dummy regressor matrix with shape ``(n, 1 + n_lags*n +
        len(covid_indices))``.
    """
    y0 = np.zeros(y.shape[1])

    for i in range(y.shape[1]):
        if levels[i]:
            y0[i] = np.mean(y[0:n_lags, i])
        else:
            y0[i] = np.mean(y[:, i])

    n = y.shape[1]

    yd = np.diag(y0) / mu

    constant = np.zeros((n, 1))
    xd = np.hstack([constant, np.tile(yd, n_lags), np.zeros((n, len(covid_indices)))])
    return yd, xd


def single_unit_root_prior(
    y: np.ndarray,
    n_lags: int,
    theta: float,
    levels: List[bool],
    covid_indices: np.ndarray = np.array([]),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct single-unit-root dummy observations for BVAR prior.

    Parameters
    ----------
    y : np.ndarray
        Data matrix with shape ``(T, n)``.
    n_lags : int
        Number of lags.
    theta : float
        Tightness parameter for the single-unit-root prior.
    levels : List[bool]
        Indicates which variables are in levels (True) or differences (False),
        with length ``n``.
    covid_indices : np.ndarray
        Indices for COVID dummies (default: empty array).

    Returns
    -------
    yd : np.ndarray
        Dummy dependent variable vector with shape ``(n,)``.
    xd : np.ndarray
        Dummy regressor vector with shape ``(1 + n_lags*n +
        len(covid_indices),)``.
    """
    y0 = np.zeros(y.shape[1])

    for i in range(y.shape[1]):
        if levels[i]:
            y0[i] = np.mean(y[0:n_lags, i])
        else:
            y0[i] = np.mean(y[:, i])

    yd = y0 / theta
    xd = np.concatenate(
        [[1 / theta], np.tile(yd, n_lags), np.zeros(len(covid_indices))]
    )
    return yd, xd


def stack_dummies(
    Y: np.ndarray,
    Z: np.ndarray,
    n_lags: int,
    levels: List[bool],
    priors: Any,
    covid_indices: np.ndarray = np.array([]),
    soc: Optional[bool] = None,
    sur: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Stack dummy observations for sum-of-coefficients and single-unit-root priors.

    Parameters
    ----------
    Y : np.ndarray
        Dependent variable matrix with shape ``(T, n)`` before stacking.
    Z : np.ndarray
        Regressor matrix with shape ``(T, k)`` before stacking.
    n_lags : int
        Number of lags.
    levels : List[bool]
        Indicates which variables are in levels (True) or differences (False).
    priors : Any
        Prior specification object with attributes 'soc' and 'sur'.
        And pars.mu and pars.theta for the tightness parameters.
    covid_indices : np.ndarray
        Indices for COVID dummies (default: empty array).
    soc : Optional[bool]
        Effective sum-of-coefficients flag; defaults to ``priors.soc`` if
        not given. Callers can override this per-fit (e.g. to disable it
        when no variable is in levels) without mutating ``priors.soc``.
    sur : Optional[bool]
        Effective single-unit-root flag; defaults to ``priors.sur`` if not
        given, analogous to ``soc``.

    Returns
    -------
    Y : np.ndarray
        Augmented dependent variable matrix with shape
        ``(T + nb_dummy_obs, n)``.
    Z : np.ndarray
        Augmented regressor matrix with shape ``(T + nb_dummy_obs, k)``.
    nb_dummy_obs : int
        Number of dummy observations added.
    """
    soc = priors.soc if soc is None else soc
    sur = priors.sur if sur is None else sur
    covid_indices = _normalise_covid_indices(
        covid_indices, Y.shape[0] + n_lags, lag_cutoff=n_lags
    )

    Y_original = Y.copy()
    Y_stacked = Y.copy()
    Z_stacked = Z.copy()

    nb_dummy_obs = 0
    if soc:
        Y_soc, Z_soc = sum_of_coefficients_prior(
            Y_original, n_lags, priors.pars.mu, levels, covid_indices=covid_indices
        )
        Y_stacked = np.vstack((Y_soc, Y))
        Z_stacked = np.vstack((Z_soc, Z))
        nb_dummy_obs = Y_soc.shape[0]

    if sur:
        Y_sur, Z_sur = single_unit_root_prior(
            Y_original, n_lags, priors.pars.theta, levels, covid_indices=covid_indices
        )
        Y_stacked = np.vstack((Y_sur, Y_stacked))
        Z_stacked = np.vstack((Z_sur, Z_stacked))
        nb_dummy_obs += 1

    return Y_stacked, Z_stacked, nb_dummy_obs
