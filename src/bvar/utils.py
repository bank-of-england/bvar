"""
Utility Functions for BVAR
==========================

This module contains utility functions for parameter manipulation,
matrix operations, and other helper functions used across the BVAR package.
"""

from numbers import Integral
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


def _validate_positive_integer(value: object, name: str) -> int:
    """Validate and normalise a positive integer argument."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def simulate_var(
    T: int,
    n: int,
    n_lags: int,
    covid: bool = False,
    levels: bool = False,
    ar_mat: Optional[np.ndarray] = None,
    constant: Optional[np.ndarray] = None,
    Sigma: Optional[np.ndarray] = None,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate synthetic VAR(p) data, optionally with COVID dummies and integration.

    Parameters
    ----------
    T : int
        Number of time periods.
    n : int
        Number of variables.
    n_lags : int
        Number of lags.
    covid : bool
        Whether to include COVID dummies (default: False).
    levels : bool
        If True, returns integrated (non-stationary) data (default: False).
    ar_mat : Optional[np.ndarray]
        First AR coefficient matrix with shape ``(n, n)`` (default: None,
        random generation).
    constant : Optional[np.ndarray]
        Constant vector with shape ``(n,)`` (default: None, random generation).
    Sigma : Optional[np.ndarray]
        Covariance matrix for error terms with shape ``(n, n)`` (default: None).
    seed : Optional[Union[int, np.random.Generator]]
        Seed or ``numpy.random.Generator`` for reproducibility (default: None).
        The function uses a local generator and leaves the global NumPy
        random state unchanged.

    Returns
    -------
    y : pd.DataFrame
        Simulated time series data with shape ``(T, n)`` and a quarterly
        ``pd.PeriodIndex`` starting 1990Q1 (COVID dummies, when enabled, assume
        the 2020Q1-2021Q4 window). Re-index the frame for a different frequency.
    b : np.ndarray
        True parameter vector with shape
        ``(n * (1 + n*n_lags + h),)`` where ``h`` is 8 when COVID dummies are
        enabled and 0 otherwise.
    sigma : np.ndarray
        Covariance matrix used for simulation with shape ``(n, n)``.
    eps : np.ndarray
        Simulated error terms with shape ``(T, n)``.
    """

    rng = np.random.default_rng(seed)
    y = np.zeros((T, n))
    dates = pd.period_range(start="1990Q1", periods=T, freq="Q")

    # Variance =========================================================
    if Sigma is None:
        Sigma = np.eye(n)
        if n > 1:
            Sigma[0, 1] = 0.5
            Sigma[1, 0] = 0.5

    eps = rng.multivariate_normal(np.zeros(n), Sigma, T)

    # AR coefficients =================================================
    ar = np.zeros((n_lags, n, n))

    if ar_mat is not None:
        ar[0] = ar_mat
    else:
        ar[0] = np.diag(np.clip(np.abs(rng.normal(0, 1, n)), 0.0, 0.5))

    ar_stacked = np.concatenate(ar, axis=1)

    # Constant ========================================================
    if constant is not None:
        ar_stacked = np.concatenate([constant.reshape(-1, 1), ar_stacked], axis=1)
    else:
        ar_stacked = np.concatenate(
            [rng.normal(1, 1, n).reshape(-1, 1), ar_stacked], axis=1
        )

    # COVID dummies =================================================
    if covid:
        covid_start = pd.Period("2020Q1", freq="Q")
        covid_end = pd.Period("2021Q4", freq="Q")
        h = 8
        ar_stacked = np.concatenate([ar_stacked, np.zeros((n, h)) - 10.0], axis=1)

    # Matrix form =====================================================
    b = ar_stacked.flatten()

    y[:n_lags, :] = eps[:n_lags, :]

    covid_dummy = 0

    for t in range(n_lags, T):
        x = np.array([1.0])

        for lag in range(n_lags):
            x = np.hstack([x, y[t - lag - 1, :]])

        k = len(x)

        if covid:
            x = np.hstack([x, np.zeros(h)])

            if dates[t] >= covid_start and dates[t] <= covid_end:
                x[covid_dummy + k] = 1.0
                covid_dummy += 1

        X = np.kron(np.eye(n), x)

        y[t, :] = (X @ b.reshape(-1, 1)).T + eps[t, :]

    if levels:
        y = np.cumsum(y, axis=0)

    # Add some dates
    y = pd.DataFrame(y, index=dates)
    y.index.name = "date"

    return y, b, Sigma, eps


def simulate_var_simple(
    T: int,
    n: int,
    n_lags: int,
    ar_mat: Optional[np.ndarray] = None,
    constant: Optional[np.ndarray] = None,
    Sigma: Optional[np.ndarray] = None,
    eps: Optional[np.ndarray] = None,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    """
    Generate synthetic VAR data for testing and simulation.

    Simpler variant of :func:`simulate_var` that returns only the data array
    (no metadata).  Supports a single lag and does not include COVID dummies
    or integration.

    Parameters
    ----------
    T : int
        Number of time periods (sample size).
    n : int
        Number of variables (dimension of the system).
    n_lags : int
        Number of lags in the VAR model.
    ar_mat : Optional[np.ndarray]
        First AR coefficient matrix with shape ``(n, n)``. If ``None``, the
        function generates a random diagonal matrix.
    constant : Optional[np.ndarray]
        Constant vector with shape ``(n,)``. If None, drawn from N(1, 1).
    Sigma : Optional[np.ndarray]
        Covariance matrix for error terms with shape ``(n, n)``. If None, uses
        a default matrix.
    eps : Optional[np.ndarray]
        Pre-generated error terms with shape ``(T, n)``. If None, drawn from
        N(0, Sigma).
    seed : Optional[Union[int, np.random.Generator]]
        Seed or generator for reproducibility. The function uses a local
        generator and leaves the global NumPy random state unchanged.

    Returns
    -------
    y : np.ndarray
        Generated time series data with shape ``(T, n)``.

    Raises
    ------
    ValueError
        If the supplied error array has an incompatible shape.

    Notes
    -----
    This function generates data from a VAR(p) model:
    y_t = c + A1*y_{t-1} + A2*y_{t-2} + ... + Ap*y_{t-p} + ε_t
    where ε_t ~ N(0, Σ) and Σ is the error covariance matrix.
    """

    rng = np.random.default_rng(seed)

    y = np.zeros((T, n))

    # Variance =========================================================
    if Sigma is None:
        Sigma = np.eye(n)
        if n > 1:
            Sigma[0, 1] = 0.5
            Sigma[1, 0] = 0.5

    if eps is None:
        eps = rng.multivariate_normal(np.zeros(n), Sigma, T)
    elif eps.shape != (T, n):
        raise ValueError(f"eps must have shape ({T}, {n}), got {eps.shape}.")

    # AR coefficients =================================================
    if ar_mat is not None:
        ar = ar_mat
    else:
        ar = np.diag(np.clip(np.abs(rng.normal(0, 1, n)), 0.0, 0.5))

    # Constant ========================================================
    constant = rng.normal(1, 1, n) if constant is None else constant

    y[0:n_lags, :] = eps[0:n_lags,]

    for t in range(n_lags, T):
        y[t, :] = constant + ar @ y[t - 1, :] + eps[t, :]

    return y


def reconstruct_var_parameters(
    B: np.ndarray, n: int, n_lags: int, h: int = 0
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Reconstruct VAR parameters from a vectorised coefficient vector.

    Parameters
    ----------
    B : np.ndarray
        Vectorized parameter vector from the model Y = Z * B + E. After
        reshaping, the coefficient matrix has shape ``(n*n_lags + 1 + h, n)``.
    n : int
        Number of variables.
    n_lags : int
        Number of lags.
    h : int
        Number of dummy variables (e.g., covid dummies) (default: 0).

    Returns
    -------
    constant : np.ndarray
        Constant vector with shape ``(n,)``.
    ar : np.ndarray
        Lag coefficient matrices with shape ``(n_lags, n, n)``.
    dummies : Optional[np.ndarray]
        Dummy coefficients with shape ``(h, n)``, or None if h == 0.
    """
    if B.ndim == 1:
        B = B.reshape(-1, n)

    constant = B[0, :].copy()
    ar = np.zeros((n_lags, n, n))
    for lag in range(n_lags):
        start_row = 1 + n * lag
        end_row = start_row + n
        ar[lag] = B[start_row:end_row, :]
    if h > 0:
        dummies = B[1 + n * n_lags : 1 + n * n_lags + h, :]
    else:
        dummies = None

    return constant, ar, dummies


def _normalise_covid_indices(
    covid_indices: np.ndarray, n_observations: int, lag_cutoff: int
) -> np.ndarray:
    """Canonicalise COVID row indexes for a specific usable sample."""
    indices = np.asarray(covid_indices)
    if indices.size == 0:
        return np.empty(0, dtype=int)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("COVID indices must be a one-dimensional integer array.")

    usable = (indices >= lag_cutoff) & (indices < n_observations)
    return np.unique(indices[usable])


def construct_X(
    y: np.ndarray, n_lags: int, covid_indices: np.ndarray = np.array([])
) -> np.ndarray:
    """
    Construct the design matrix X for VAR estimation.

    This function creates the regressor matrix for the vectorised VAR model
    by stacking lagged values and using Kronecker products.

    Parameters
    ----------
    y : np.ndarray
        Time series data with shape ``(T, n)``.
    n_lags : int
        Number of lags.
    covid_indices : np.ndarray
        Indices for COVID dummies (default: empty list).

    Returns
    -------
    X : np.ndarray
        Design matrix with shape ``((T-n_lags)*n, (n_lags*n + 1 + h)*n)``.

    Notes
    -----
    The VAR model in matrix form is: Y = X*B + E
    where Y is vectorised, X contains lagged values, and B contains coefficients.
    Each row block of X corresponds to one time period and contains:
    [1, y_{t-1,1}, y_{t-1,2}, ..., y_{t-p,1}, y_{t-p,2}, ...]
    replicated using Kronecker products for the system structure.
    """
    T, n = y.shape
    covid_indices = _normalise_covid_indices(covid_indices, T, n_lags)
    h = len(covid_indices)

    X = np.zeros((n * T, (n_lags * n + 1 + h) * n))
    x = np.zeros(1 + n * n_lags + h)
    x[0] = 1.0

    nb_dummy = 0

    for t in range(n_lags, T):
        r = n * t
        xt = x.copy()

        xt[1 : 1 + n * n_lags] = y[t - n_lags : t, :][::-1].flatten()

        if t in covid_indices:
            xt[1 + n * n_lags + nb_dummy] = 1.0
            nb_dummy += 1

        X[r : r + n, :] = np.kron(np.eye(n), xt)

    return X[n_lags * n :,]


def construct_Y_Z(
    data: np.ndarray, n_lags: int, covid_indices: np.ndarray = np.array([])
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct the dependent variable Y and design matrix Z for VAR estimation.

    Parameters
    ----------
    data : np.ndarray
        Time series data with shape ``(T_total, n)``.
    n_lags : int
        Number of lags.
    covid_indices : np.ndarray
        Indices for COVID dummies (default: empty list).

    Returns
    -------
    Y : np.ndarray
        Dependent variable matrix with shape ``(T, n)``, where
        ``T = T_total - n_lags``.
    Z : np.ndarray
        Design matrix with shape ``(T, k)``, where ``k = n*n_lags + 1 + h``.
    """
    covid_indices = _normalise_covid_indices(covid_indices, data.shape[0], n_lags)
    T, n, k, _, _ = get_dimensions(data, n_lags, covid_indices)
    Y = data[n_lags:, :]

    nb_dummy = 0

    Z = np.zeros((T + n_lags, k))
    Z[:, 0] = 1.0

    for t in range(n_lags, T + n_lags):
        Z[t, 1 : 1 + n * n_lags] = data[t - n_lags : t, :][::-1].flatten()

        if t in covid_indices:
            Z[t, 1 + n * n_lags + nb_dummy] = 1.0
            nb_dummy += 1

    return Y, Z[n_lags:, :]


def get_dimensions(
    data: np.ndarray, n_lags: int, covid_indices: np.ndarray = np.array([])
) -> Tuple[int, int, int, int, int]:
    """
    Get dimensions for VAR estimation.

    Parameters
    ----------
    data : np.ndarray
        Time series data.
    n_lags : int
        Number of lags.
    covid_indices : np.ndarray
        Indices for COVID dummies (default: empty list).

    Returns
    -------
    T : int
        Effective sample size after lags.
    n : int
        Number of variables.
    k : int
        Number of regressors per equation.
    nk : int
        Total number of parameters in the system.
    h : int
        Number of COVID dummies.
    """
    covid_indices = _normalise_covid_indices(covid_indices, data.shape[0], n_lags)
    h = len(covid_indices)
    T, n = data.shape
    T = T - n_lags
    k = n * n_lags + 1 + h
    nk = n * k

    return T, n, k, nk, h


def check_covid(
    data: Union[pd.DataFrame, np.ndarray], covid_dates: Optional[list]
) -> np.ndarray:
    """
    Identify indices corresponding to COVID period in the data.

    Parameters
    ----------
    data : Union[pd.DataFrame, np.ndarray]
        Input data with shape ``(T, n)``. For a DataFrame with a PeriodIndex or
        DatetimeIndex, the function determines the COVID period from the index.
    covid_dates : Optional[list]
        Two-element list ``[start, end]`` defining the COVID period. Each
        element may be anything pandas can interpret as a date (string,
        ``pd.Timestamp`` or ``pd.Period``); the bounds are converted to the
        data's own frequency before matching, so any frequency is supported.

    Returns
    -------
    covid_indices : np.ndarray
        Integer indices of rows falling within the COVID period.
        Empty array if data is not a DataFrame, if dates are not found, or if
        ``covid_dates`` is None (e.g. when the model has ``covid=False``).
    """
    if covid_dates is None:
        return np.array([])

    if isinstance(data, pd.DataFrame):
        index = data.index

        if isinstance(index, pd.PeriodIndex):
            # Match the COVID bounds to the data frequency.
            covid_start = pd.Period(covid_dates[0], freq=index.freq)
            covid_end = pd.Period(covid_dates[1], freq=index.freq)
        else:
            covid_start = pd.Timestamp(covid_dates[0])
            covid_end = pd.Timestamp(covid_dates[1])

        covid_mask = (index >= covid_start) & (index <= covid_end)
        covid_indices = np.where(covid_mask)[0]
    else:
        covid_indices = np.array([])

    return covid_indices


def cumulative_change(data: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """
    Compute cumulative changes (cumulated growth) for forecast paths.

    Parameters
    ----------
    data : np.ndarray
        Forecast data (log-levels or log-differences) with shape ``(T, n)``.
    levels : np.ndarray
        Indicator array with shape ``(n,)``: 0 for first differences, 1 for
        levels.

    Returns
    -------
    sum_diff : np.ndarray
        Cumulative changes from the second observation onward, with shape
        ``(T-1, n)``.

    Raises
    ------
    ValueError
        If the variables are a mix of levels and differences.

    Notes
    -----
    Returns the cumulative sum of first differences of a levels or
    log-difference series (the first row is dropped as it is not a forecast).
    """

    if np.sum(levels) == 0:
        series_format = "log_diff"
    elif np.sum(levels) == len(levels):
        series_format = "log_levels"
    else:
        raise ValueError("All series must be either in levels or in first differences")

    if series_format == "log_levels":
        data_diff = np.diff(data, axis=0)
        sum_diff = np.cumsum(data_diff, axis=0)
    elif series_format == "log_diff":
        sum_diff = np.cumsum(
            data[1:, :], axis=0
        )  # skip the first row which is not a forecast

    return sum_diff
