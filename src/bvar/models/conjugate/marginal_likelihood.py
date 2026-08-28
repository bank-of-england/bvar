"""
Marginal Likelihood and Hyperparameter Optimisation
====================================================

Implements the marginal likelihood computation and hyperparameter
optimisation of Giannone, Lenza, & Primiceri (2015).  Only valid for
the natural-conjugate (Normal-Inverse-Wishart) model.

References
----------
Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection for
    vector autoregressions.
"""

import logging
from typing import List, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from ...dummy_observations import stack_dummies
from ...utils import construct_Y_Z
from ..base import SamplingModel
from .minnesota import prior_minnesota

logger = logging.getLogger(__name__)


# ======================================================================
# Helpers
# ======================================================================


def softplus(x: np.ndarray) -> np.ndarray:
    """Softplus transformation: f(x) = log(1 + exp(x)).

    Maps unbounded values to the positive real line.  Used during
    hyperparameter optimisation to enforce positivity constraints
    more smoothly than the exponential function.
    """
    return np.logaddexp(0, x)


def log_gammapdf(x: float, k: float, theta: float) -> float:
    """Log PDF of the Gamma distribution.

    log p(x; k, θ) = (k-1) log(x) - x/θ - k log(θ) - log Γ(k)
    """
    return (k - 1) * np.log(x) - x / theta - k * np.log(theta) - gammaln(k)


# ======================================================================
# Marginal likelihood
# ======================================================================


def log_marginal_likelihood(
    Y: np.ndarray,
    Z: np.ndarray,
    beta_0: np.ndarray,
    V_A_inv: np.ndarray,
    S_0: np.ndarray,
    nu_0: float,
    n: int,
) -> float:
    """Compute the log marginal likelihood of a conjugate BVAR.

    The marginal likelihood integrates out the VAR parameters (A, Σ):
        p(Y | λ) = ∫∫ p(Y | A, Σ) p(A, Σ | λ) dA dΣ

    This has a closed-form expression for Normal-Inverse-Wishart priors
    (GLP 2015, Appendix A).

    Parameters
    ----------
    Y : np.ndarray
        Dependent-variable matrix with shape ``(T, n)``.
    Z : np.ndarray
        Regressor matrix with shape ``(T, k)``.
    beta_0 : np.ndarray
        Prior mean vector with shape ``(nk,)``.
    V_A_inv : np.ndarray
        Per-equation prior precision matrix with shape ``(k, k)``.
    S_0 : np.ndarray
        Inverse-Wishart scale matrix with shape ``(n, n)``.
    nu_0 : float
    n : int

    Returns
    -------
    float
        Log marginal likelihood (returns -1e5 on failure).
    """
    k = len(beta_0) // n
    A_0 = beta_0.reshape(n, k).T
    b = A_0
    x = Z
    y = Y
    T = Y.shape[0]
    omega_inv = V_A_inv
    d = nu_0
    psi = np.diag(S_0)

    try:
        B_hat = np.linalg.solve(x.T @ x + omega_inv, x.T @ y + omega_inv @ b)
    except np.linalg.LinAlgError:
        B_hat = b

    residuals = y - x @ B_hat

    sqrt_omega = np.diag(1.0 / np.sqrt(np.diag(omega_inv)))
    aaa = sqrt_omega @ (x.T @ x) @ sqrt_omega

    inv_sqrt_psi = np.diag(1.0 / np.sqrt(psi))
    middle_term = residuals.T @ residuals + (B_hat - b).T @ omega_inv @ (B_hat - b)
    bbb = inv_sqrt_psi @ middle_term @ inv_sqrt_psi

    eigen_A = np.real(np.linalg.eigvals(aaa))
    eigen_A[eigen_A < 1e-12] = 0
    eigen_A = eigen_A + 1

    eigen_B = np.real(np.linalg.eigvals(bbb))
    eigen_B[eigen_B < 1e-12] = 0
    eigen_B = eigen_B + 1

    logML = (
        -n * T * np.log(np.pi) / 2
        + np.sum(gammaln((T + d - np.arange(n)) / 2) - gammaln((d - np.arange(n)) / 2))
        - T * np.sum(np.log(psi)) / 2
        - n * np.sum(np.log(eigen_A)) / 2
        - (T + d) * np.sum(np.log(eigen_B)) / 2
    )

    if not np.isfinite(logML) or np.isnan(logML):
        logML = -1e5

    return logML


# ======================================================================
# Objective & optimiser
# ======================================================================


def objective_function(
    pars: np.ndarray,
    data: np.ndarray,
    n_lags: int,
    covid_indices: List[int],
    levels: np.ndarray,
    nb_dummy_obs: int,
    model: SamplingModel,
    Y: np.ndarray,
    Z: np.ndarray,
    add_priors: bool,
    soc: bool,
    sur: bool,
) -> float:
    """Negative log posterior for hyperparameter optimisation.

    Computes  −[ log p(Y | λ) + log p(λ) ]  where *p(Y | λ)* is the
    marginal likelihood and *p(λ)* the Gamma hyperprior.

    Parameters
    ----------
    pars : np.ndarray
        Unconstrained hyperparameter vector with shape ``(n_hyperparameters,)``.
    data : np.ndarray
        Input data array with shape ``(T_total, n)``.
    n_lags : int
        Number of VAR lags.
    covid_indices : List[int]
        Indices for COVID dummy observations.
    levels : np.ndarray
        Indicators for variables in levels with shape ``(n,)``.
    nb_dummy_obs : int
        Number of dummy observations.
    model : SamplingModel
        Sampling model containing the prior settings.
    Y : np.ndarray
        Dependent-variable matrix with shape ``(T, n)``.
    Z : np.ndarray
        Regressor matrix with shape ``(T, k)``.
    add_priors : bool
        Whether to include hyperprior terms.
    soc : bool
        Effective SOC/SUR flags for this fit, used to decide whether the
        corresponding dummy rows are actually stacked before evaluating the
        marginal likelihood. Independent of ``model.soc``/``model.sur``,
        which fix the hyperparameter vector layout (whether mu/theta are
        parameters at all).
    sur : bool
        Effective SUR flag for this fit.

    Returns
    -------
    float
        Negative log posterior.
    """
    _, n = Y.shape

    model.fill_in_from_vector(softplus(pars))

    if model.minnesota:
        beta_0, V_A_inv = prior_minnesota(
            data, n_lags, covid_indices, levels, model.pars
        )
    else:
        n_vars, k = Y.shape[1], Z.shape[1]
        beta_0 = np.zeros(n_vars * k)
        V_A_inv = np.eye(k) * 1e-10

    S_0 = model.pars.S_0
    nu_0 = model.pars.nu_0

    if soc or sur:
        Y_for_ML, Z_for_ML, nb_dummy_obs = stack_dummies(
            Y, Z, n_lags, levels, model, covid_indices, soc=soc, sur=sur
        )
    else:
        Y_for_ML = Y.copy()
        Z_for_ML = Z.copy()

    logp = log_marginal_likelihood(Y_for_ML, Z_for_ML, beta_0, V_A_inv, S_0, nu_0, n)

    if nb_dummy_obs > 0:
        Y_dummies = Y_for_ML[0:nb_dummy_obs, :]
        Z_dummies = Z_for_ML[0:nb_dummy_obs, :]
        logp -= log_marginal_likelihood(
            Y_dummies, Z_dummies, beta_0, V_A_inv, S_0, nu_0, n
        )

    if add_priors:
        logp += log_gammapdf(model.pars.c1, model.pars.c1_k, model.pars.c1_theta)
        logp += log_gammapdf(model.pars.c3, model.pars.c3_k, model.pars.c3_theta)
        if model.soc:
            logp += log_gammapdf(model.pars.mu, model.pars.mu_k, model.pars.mu_theta)
        if model.sur:
            logp += log_gammapdf(
                model.pars.theta, model.pars.theta_k, model.pars.theta_theta
            )

    return -logp


def tune_priors(
    data: np.ndarray,
    n_lags: int,
    covid_indices: List[int],
    levels: np.ndarray,
    model: SamplingModel,
    nb_restart: int = 0,
    initial_values: Optional[np.ndarray] = None,
    add_priors: bool = True,
    soc: Optional[bool] = None,
    sur: Optional[bool] = None,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """Optimise hyperparameters by maximising the marginal likelihood.

    Uses BFGS with optional multi-start.  Modifies *model* in place.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    n_lags : int
        Number of VAR lags.
    covid_indices : List[int]
        Indices for COVID dummy observations.
    levels : np.ndarray
        Indicators for variables in levels.
    model : SamplingModel
        Sampling model containing the prior settings.
    nb_restart : int
        Number of additional optimisation restarts.
    initial_values : Optional[np.ndarray]
        Optional initial hyperparameter values.
    add_priors : bool
        Whether to include hyperprior terms.
    soc : Optional[bool]
        Effective SOC/SUR flags for this fit, passed through to
        :func:`objective_function`. Default to ``model.soc``/``model.sur``
        if not given.
    sur : Optional[bool]
        Effective SUR flag for this fit. Defaults to ``model.sur`` if not given.
    rng : Optional[np.random.Generator]
        Seed or generator used for the random perturbations applied to the
        initial guess and to each multi-start restart. Normalised via
        ``np.random.default_rng(rng)``, so a plain seed, an existing
        ``Generator``, or ``None`` (nondeterministic) are all accepted. The
        global NumPy random state is never touched.

    Raises
    ------
    ValueError
        If ``initial_values`` contains a non-positive value.
    """
    soc = model.soc if soc is None else soc
    sur = model.sur if sur is None else sur
    rng = np.random.default_rng(rng)

    Y, Z = construct_Y_Z(data, n_lags, covid_indices)
    nb_dummy_obs = 0

    if initial_values is not None:
        if np.any(initial_values <= 0):
            raise ValueError("initial_values must be strictly positive")
        initial_guess = np.log(np.expm1(initial_values))  # stable inverse softplus
    else:
        initial_guess = np.zeros(model.nb_hyper_pars)
        initial_guess += rng.normal(0, 0.1, len(initial_guess))

    logger.info("Optimising hyperparameters...")
    best_result = None
    best_fun = np.inf

    for restart in range(nb_restart + 1):
        if restart == 0:
            x0 = initial_guess.copy()
        else:
            noise = rng.normal(0, 0.1, len(best_result.x))
            x0 = best_result.x + noise

        result = minimize(
            objective_function,
            x0=x0,
            args=(
                data,
                n_lags,
                covid_indices,
                levels,
                nb_dummy_obs,
                model,
                Y,
                Z,
                add_priors,
                soc,
                sur,
            ),
            method="BFGS",
            options={"maxiter": 1000},
        )

        if result.fun < best_fun:
            best_result = result
            best_fun = result.fun
            logger.info(
                "Restart %d: New best objective = %.4f", restart + 1, result.fun
            )

    result = best_result

    model.fill_in_from_vector(softplus(result.x))

    logger.info("Optimised hyperparameters:")
    logger.info("  Shrinkage (c1): %.3f", model.pars.c1)
    logger.info("  Lag decay (c3): %.3f", model.pars.c3)
    if soc:
        logger.info("  SOC (mu): %.3f", model.pars.mu)
    if sur:
        logger.info("  SUR (theta): %.3f", model.pars.theta)
