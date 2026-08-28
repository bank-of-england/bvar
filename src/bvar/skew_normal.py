"""
Skew-Normal Distribution Utilities
==================================

This module provides functions for working with multivariate skew-normal
distributions, including sampling and parameter transformations under
affine mappings.

The skew-normal distribution is parameterised following Azzalini & Capitanio (1999):
    X ~ SN_p(ξ, Ω, α)
where ξ is the location, Ω is the scale matrix, and α is the shape parameter.

Functions
---------
draw_skew_normal : function
    Draw samples from a multivariate skew-normal distribution.
draw_sun : function
    Draw samples from a Unified Skew-Normal distribution.

References
----------
Azzalini, A., & Capitanio, A. (1999). Statistical applications of the multivariate
    skew normal distribution. Journal of the Royal Statistical Society: Series B,
    61(3), 579-602.
"""

from typing import Optional

import numpy as np
from scipy.stats import multivariate_normal as mvn


def draw_skew_normal(
    cov: np.ndarray,
    alpha: np.ndarray,
    size: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Draw samples from a zero-mean multivariate skew-normal distribution.

    Uses the stochastic representation: X = |Z_0| δ + (I - δδ')^{1/2} Z_1
    where Z_0 ~ N(0,1) and Z_1 ~ N(0, Ω).

    Parameters
    ----------
    cov : np.ndarray
        Covariance (scale) matrix Ω with shape ``(d, d)``.
    alpha : np.ndarray
        Shape parameter controlling skewness, with shape ``(d,)`` or
        ``(d, 1)``.
    size : int
        Number of samples to draw. Default is 1.
    rng : Optional[np.random.Generator]
        Generator for the draws. If None, the global NumPy random state is
        used (via ``scipy.stats``), which is not reproducible unless the
        caller has separately seeded it. Default is None.

    Returns
    -------
    x1_scaled : np.ndarray
        Samples from SN(0, cov, alpha), with shape ``(size, d)``.

    References
    ----------
    https://gregorygundersen.com/blog/2020/12/29/multivariate-skew-normal/
    """

    if alpha.ndim == 1:
        alpha = alpha.reshape(-1, 1)

    dim = cov.shape[0]

    # Compute δ from α: δ = (1 + α'Ωα)^{-1/2} Ω α
    delta = (1 / np.sqrt(1 + alpha.T @ cov @ alpha)) * cov @ alpha

    # Construct augmented covariance for (Z_0, Z_1)
    cov_star = np.block([[np.ones(1), delta.T], [delta, cov]])

    # Draw from augmented normal and apply sign flip
    x = (
        mvn(np.zeros(dim + 1), cov_star)
        .rvs(size, random_state=rng)
        .reshape(size, dim + 1)
    )
    x0, x1 = x[:, 0], x[:, 1:]
    inds = x0 <= 0
    x1[inds] = -1 * x1[inds]

    return x1


def draw_skew_normal_univariate(
    n: int, pos: float, scale: float, shape: float
) -> np.ndarray:
    """
    Draw univariate skew-normal samples using Henze's representation.

    Parameters
    ----------
    n : int
        Number of samples to draw.
    pos : float
        Location parameter (mean shift).
    scale : float
        Scale parameter (standard deviation).
    shape : float
        Shape parameter controlling skewness.

    Returns
    -------
    samples : np.ndarray
        Samples from SN(pos, scale, shape), with shape ``(n,)``.
    """
    t0 = np.random.normal(0, 1, n)
    t1 = np.random.normal(0, 1, n)
    delta = shape / np.sqrt(1 + shape**2)

    # Z ~ SN(0,1,alpha)
    Z = delta * np.abs(t0) + np.sqrt(1 - delta**2) * t1

    # Y ~ SN(pos, scale, alpha)
    return pos + scale * Z


def draw_sun(
    xi: np.ndarray,
    Omega: np.ndarray,
    Delta: np.ndarray,
    gamma: np.ndarray,
    Gamma: np.ndarray,
    size: int = 1,
    rng: Optional[np.random.Generator] = None,
    max_attempts: Optional[int] = None,
) -> np.ndarray:
    """
    Draw samples from a Unified Skew-Normal distribution via selection.

    Uses the stochastic representation based on truncation of a joint normal
    distribution. The SUN distribution is defined as the conditional distribution
    of V given U > -γ, where (U, V) ~ N([0; ξ], [Γ, Δ'; Δ, Ω]).

    Parameters
    ----------
    xi : np.ndarray
        Location parameter for the observable component, with shape ``(d,)``.
    Omega : np.ndarray
        Scale matrix for the observable component, with shape ``(d, d)``.
    Delta : np.ndarray
        Coupling matrix between latent and observable components, with shape
        ``(d, k)``.
    gamma : np.ndarray
        Truncation threshold for the latent component, with shape ``(k,)``.
    Gamma : np.ndarray
        Scale matrix for the latent component, with shape ``(k, k)``.
    size : int
        Number of samples to draw. Default is 1.
    rng : Optional[np.random.Generator]
        Generator for the draws. Defaults to a fresh
        ``numpy.random.default_rng()`` if not given.
    max_attempts : Optional[int]
        Maximum number of joint-normal proposals. By default, this is
        ``max(10_000, 100 * size)``. Must be a positive integer.

    Returns
    -------
    Y : np.ndarray
        Samples from SUN_{d,k}(ξ, Ω, Δ, γ, Γ), with shape ``(size, d)``.

    Notes
    -----
    This function uses rejection sampling, so the actual number of normal
    draws may be much larger than `size` depending on the truncation probability.
    The function raises ``RuntimeError`` when the attempt budget expires before
    it accepts all samples.

    References
    ----------
    Arellano-Valle, R. B., & Azzalini, A. (2006). On the unification of families
        of skew-normal distributions. Scandinavian Journal of Statistics, 33(3),
        561-574.

    Raises
    ------
    RuntimeError
        If rejection sampling exhausts the attempt budget.
    ValueError
        If ``max_attempts`` is not a positive integer.
    """
    # Ensure correct shapes
    d = len(xi)
    k = len(gamma)

    if max_attempts is None:
        max_attempts = max(10_000, 100 * size)
    elif isinstance(max_attempts, bool) or not isinstance(
        max_attempts, (int, np.integer)
    ):
        raise ValueError("max_attempts must be a positive integer")

    if max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")

    if rng is None:
        rng = np.random.default_rng()

    # Construct joint covariance matrix for (U, V)
    # S = [Γ,  Δ']
    #     [Δ,  Ω ]
    S = np.block([[Gamma, Delta.T], [Delta, Omega]])

    # Cholesky decomposition for efficient sampling
    L = np.linalg.cholesky(S)

    # Storage for accepted samples
    Y = np.zeros((size, d))

    # Rejection sampling
    i = 0
    attempts = 0
    while i < size and attempts < max_attempts:
        attempts += 1
        # Draw from joint normal
        z = L @ rng.standard_normal(k + d)
        U = z[:k]  # Latent component
        V = z[k:]  # Observable component

        # Accept samples that satisfy the truncation condition.
        if np.all(U > -gamma):
            Y[i, :] = xi + V
            i += 1

    if i < size:
        raise RuntimeError(
            "SUN rejection sampling exhausted max_attempts="
            f"{max_attempts} after accepting {i} of {size} samples"
        )

    return Y


def sun_conditional_forecast(
    f: np.ndarray,
    sigma_f: np.ndarray,
    shape_f: np.ndarray,
    C: np.ndarray,
    b: np.ndarray,
    BigM: np.ndarray,
    n_draws: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Compute conditional forecasts using a Unified Skew-Normal distribution.

    Transforms the conditional forecasting problem into a SUN posterior
    and draws from it using rejection sampling.

    Parameters
    ----------
    f : np.ndarray
        Location of the constraints (target values), with shape ``(k,)``.
    sigma_f : np.ndarray
        Scale (variance) of the constraints, with shape ``(k,)``.
    shape_f : np.ndarray
        Shape (skewness) parameters of the constraints, with shape ``(k,)``.
    C : np.ndarray
        Selection matrix mapping forecast variables to constraints, with shape
        ``(k, nH)``.
    b : np.ndarray
        Unconditional mean forecast (deterministic component), with shape
        ``(nH,)``.
    BigM : np.ndarray
        IRF propagation matrix (M transposed), with shape ``(nH, nH)``.
    n_draws : int
        Number of draws to generate.
    rng : Optional[np.random.Generator]
        Random number generator used for sampling.

    Returns
    -------
    Y_full_constraint : np.ndarray
        Draws from the SUN conditional forecast distribution, with shape
        ``(n_draws, nH)``.
    """

    # k is the dimension of the constraints
    # n is the dimension of the full set of variables
    k, n = C.shape

    Omega = np.diag(sigma_f)
    omega_vec = np.sqrt(sigma_f)
    delta = shape_f / np.sqrt(1 + shape_f**2)
    Delta = np.diag((omega_vec * delta).flatten())

    # parameters for the independent case
    gamma = np.zeros((k, 1))

    Gamma = np.eye(k)

    m_r = C @ b
    Sigma_0 = BigM @ BigM.T
    Sigma_r = C @ Sigma_0 @ C.T

    inv_Sigma_0 = np.linalg.inv(Sigma_0)
    inv_Omega = np.linalg.inv(Omega)
    inv_Sigma_r = np.linalg.inv(Sigma_r)

    # Calculate precision matrix Q_new
    Q_new = inv_Sigma_0 + C.T @ (inv_Omega - inv_Sigma_r) @ C
    Sigma_new = np.linalg.inv(Q_new)

    # Calculate new mean b_new -> mu_new
    term_constraints = inv_Omega @ f - inv_Sigma_r @ m_r
    b_new = inv_Sigma_0 @ b + C.T @ term_constraints
    mu_new = Sigma_new @ b_new

    # Map to SUN parameters
    xi_new = mu_new
    Omega_new = Sigma_new

    A_new = C.T @ (inv_Omega @ Delta)
    Delta_new = Omega_new @ A_new

    # Update gamma (truncation threshold)
    gamma_new = gamma + Delta.T @ inv_Omega @ (C @ mu_new - f)
    Gamma_new = Gamma  # Unchanged

    Y_full_constraint = draw_sun(
        xi_new, Omega_new, Delta_new, gamma_new, Gamma_new, size=n_draws, rng=rng
    )

    return Y_full_constraint
