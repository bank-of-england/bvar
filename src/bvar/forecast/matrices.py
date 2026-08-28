"""
Companion-form matrix construction for VAR forecasting.

Contains the numba-compiled building blocks used by both the unconditional
and conditional forecast paths.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from ..utils import reconstruct_var_parameters

# ------------------------------------------------------------------
# Public wrapper
# ------------------------------------------------------------------


def construct_forecast_matrices(
    y: np.ndarray,
    A_0: np.ndarray,
    beta: np.ndarray,
    p: int,
    n: int,
    h: int,
    H: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct matrices for the companion form forecast representation.

    The forecast follows: y_{T+1:T+H} = b + M' @ ε
    where b is the deterministic component and M maps shocks to forecasts.

    Parameters
    ----------
    y : np.ndarray
        Last p observations before forecasting, with shape ``(p, n)``.
    A_0 : np.ndarray
        Cholesky factor of Σ (structural impact matrix), with shape ``(n, n)``.
    beta : np.ndarray
        VAR coefficient matrix [c, B_1, ..., B_p, D]', with shape ``(k, n)``.
    p : int
        Number of lags.
    n : int
        Number of variables.
    h : int
        Number of COVID dummy variables.
    H : int
        Forecast horizon.

    Returns
    -------
    b : np.ndarray
        Deterministic forecast component from constant and lagged values, with
        shape ``(H, n)``.
    big_M : np.ndarray
        Block lower triangular matrix mapping shocks to forecasts, with shape
        ``(H*n, H*n)``.
    M : np.ndarray
        Impulse response matrices M_0, M_1, ..., M_{H-1}, with shape
        ``(H, n, n)``.
    N : np.ndarray
        Lag effect matrices for each lag and horizon, with shape
        ``(p, H, n, n)``.
    K : np.ndarray
        Constant propagation matrices, with shape ``(H, n, n)``.

    Notes
    -----
    See Antolín-Díaz et al. (2021) for details.
    """
    c, B, _ = reconstruct_var_parameters(beta, n, p, h=h)
    b, big_M, M, N, K = _construct_b_B_M_N_K(y, A_0, c, B, p, n, H)
    return b, big_M, M, N, K


# ------------------------------------------------------------------
# Numba-compiled internals
# ------------------------------------------------------------------


@njit
def _construct_b_B_M_N_K(
    y: np.ndarray,
    A_0: np.ndarray,
    c: np.ndarray,
    B: np.ndarray,
    p: int,
    n: int,
    H: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numba-accelerated construction of forecast matrices.

    Builds the matrices M, K, N, and b needed for VAR forecasting
    in companion form: y = b + M' @ ε.
    """
    # The recursion below references lag coefficients up to index p+H-2,
    # but the VAR only has p lag matrices. Pad with zeros: the model has no
    # coefficients beyond lag p, so those "extra" lag matrices are zero.
    B_padded = np.zeros((p + H, n, n))
    B_padded[:p] = B

    M = np.zeros((H, n, n))
    M[0] = A_0

    K = np.zeros((H, n, n))
    K[0] = np.eye(n)

    for i in range(1, H):
        K[i] = np.eye(n)
        for j in range(1, i + 1):
            B_slice = B_padded[j - 1].copy()
            M_slice = M[i - j].copy()
            K_slice = K[i - j].copy()
            M[i] += M_slice @ B_slice
            K[i] += K_slice @ B_slice

    # N matrices
    N = np.zeros((p, H, n, n))
    for lag in range(p):
        N[lag, 0] = B_padded[lag]
        for i in range(1, H):
            for j in range(1, i + 1):
                B_slice = B_padded[j - 1].copy()
                N_slice = N[lag, i - j].copy()
                N[lag, i] += N_slice @ B_slice
            N[lag, i] += B_padded[i + lag]

    # b vectors
    b = np.zeros((H, n))
    for h_idx in range(H):
        for lag in range(p):
            N_slice = N[lag, h_idx].copy()
            y_slice = y[-1 - lag].copy()
            b[h_idx, :] += y_slice @ N_slice
        K_slice = K[h_idx].copy()
        b[h_idx, :] += c @ K_slice

    # big_M
    big_M = np.zeros((H * n, H * n))
    for i in range(H):
        for j in range(H - i):
            big_M[i * n : (i + 1) * n, i * n + j * n : i * n + (j + 1) * n] = M[j]

    return b, big_M, M, N, K


@njit
def h_step_forecast_loop(
    y_h: np.ndarray,
    draw: int,
    last_obs_index: int,
    H: int,
    n: int,
    p: int,
    beta: np.ndarray,
    epsilon: np.ndarray,
) -> np.ndarray:
    """Numba-accelerated h-step ahead recursive forecasting.

    Iteratively computes forecasts using the VAR recursion:
        y_t = c + B_1 y_{t-1} + ... + B_p y_{t-p} + ε_t
    """
    x_t = np.zeros(beta.shape[0])
    x_t[0] = 1.0

    for t in range(H):
        start = last_obs_index + t - p
        end = last_obs_index + t
        x_t[1 : 1 + n * p] = y_h[draw, start:end, :][::-1].flatten()
        y_h[draw, last_obs_index + t, :] = x_t @ beta + epsilon[t, :]

    return y_h
