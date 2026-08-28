"""
Conditional forecast algorithms and constraint processing.

Implements the constrained-draw routines of Waggoner & Zha (1999),
Antolín-Díaz et al. (2021), and the Andersson, Palmqvist & Waggoner (2010)
shock-adjustment approach.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.stats import skewnorm

from ..skew_normal import sun_conditional_forecast
from .matrices import construct_forecast_matrices

# ------------------------------------------------------------------
# Constraint matrix construction
# ------------------------------------------------------------------


def _coerce_constraint_array(value: np.ndarray, name: str) -> np.ndarray:
    """Convert a constraint input to a numeric array with a clear error."""
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array.") from exc


def validate_constraint_inputs(
    constraint_loc: np.ndarray,
    constraint_scale: Optional[np.ndarray],
    constraint_shape: Optional[np.ndarray],
    expected_shape: Optional[tuple[int, int]] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Validate forecast constraint arrays before forecast construction."""
    constraint_loc = _coerce_constraint_array(constraint_loc, "constraint_mean")
    if constraint_loc.ndim != 2:
        raise ValueError(
            f"constraint_mean must have shape (H, n), got {constraint_loc.shape}."
        )
    if expected_shape is not None and constraint_loc.shape != expected_shape:
        raise ValueError(
            "constraint_mean must have shape "
            f"{expected_shape}, got {constraint_loc.shape}."
        )

    constrained = ~np.isnan(constraint_loc)
    if not np.isfinite(constraint_loc[constrained]).all():
        raise ValueError(
            "Selected constraint_mean values must be finite; use NaN "
            "for unconstrained locations."
        )

    validated_optional = []
    for name, values in (
        ("constraint_variance", constraint_scale),
        ("constraint_shape", constraint_shape),
    ):
        if values is None:
            validated_optional.append(None)
            continue
        values = _coerce_constraint_array(values, name)
        if values.shape != constraint_loc.shape:
            raise ValueError(
                f"{name} must have shape {constraint_loc.shape}, got {values.shape}."
            )
        selected_values = values[constrained]
        if not np.isfinite(selected_values).all():
            raise ValueError(f"Selected {name} values must be finite.")
        if name == "constraint_variance" and np.any(selected_values < 0):
            raise ValueError(
                "Selected constraint_variance values must be non-negative."
            )
        validated_optional.append(values)

    return constraint_loc, validated_optional[0], validated_optional[1]


def get_constraint(
    constraint_loc: np.ndarray,
    constraint_scale: Optional[np.ndarray],
    constraint_shape: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process and format forecast constraints into matrix form.

    Converts constraint arrays into the matrix representation needed for
    the conditional forecasting algorithm.

    Parameters
    ----------
    constraint_loc : np.ndarray
        Location (mean) constraints with shape ``(H, n)`` and NaNs for
        unconstrained values.
    constraint_scale : Optional[np.ndarray]
        Scale (variance) constraints with shape ``(H, n)`` and NaNs for
        unconstrained values, or ``None``.
    constraint_shape : Optional[np.ndarray]
        Shape (skewness) constraints with shape ``(H, n)`` and NaNs for
        unconstrained values, or ``None``.

    Returns
    -------
    C : np.ndarray
        Selection matrix with shape ``(nb_constraints, H*n)`` indicating which
        entries impose constraints.
    constraint_loc : np.ndarray
        Vector of location constraints with shape ``(nb_constraints,)``
        (flattened, NaNs removed).
    constraint_scale : np.ndarray
        Diagonal matrix with shape ``(nb_constraints, nb_constraints)`` of
        scale constraints (zeros if None provided).
    constraint_shape : np.ndarray
        Vector of shape constraints with shape ``(nb_constraints,)`` (zeros if
        None provided).
    """
    constraint_loc, constraint_scale, constraint_shape = validate_constraint_inputs(
        constraint_loc,
        constraint_scale,
        constraint_shape,
    )
    H, n = constraint_loc.shape

    constraint_loc = constraint_loc.flatten()
    constraint_indices = np.where(~np.isnan(constraint_loc))[0]
    constraint_loc = constraint_loc[constraint_indices]
    nb_constraints = len(constraint_indices)

    C = np.zeros((len(constraint_indices), H * n))
    for i in range(len(constraint_indices)):
        C[i, constraint_indices[i]] = 1

    if constraint_scale is not None:
        constraint_scale = constraint_scale.flatten()
        constraint_scale = constraint_scale[constraint_indices]
        constraint_scale = np.diag(constraint_scale)
    else:
        constraint_scale = np.zeros((nb_constraints, nb_constraints))

    if constraint_shape is not None:
        constraint_shape = constraint_shape.flatten()
        constraint_shape = constraint_shape[constraint_indices]
    else:
        constraint_shape = np.zeros(nb_constraints)

    return C, constraint_loc, constraint_scale, constraint_shape


# ------------------------------------------------------------------
# Main constrained-draw routine
# ------------------------------------------------------------------


def draw_constrained_forecasts(
    sigma: np.ndarray,
    beta: np.ndarray,
    C: np.ndarray,
    f: np.ndarray,
    Sigma_f: np.ndarray,
    shape_f: np.ndarray,
    last_p_obs: np.ndarray,
    p: int,
    n: int,
    h: int,
    H: int,
    point_only: bool,
    constraint_sampler: Optional[Callable],
    method: str,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Draw forecasts subject to distributional constraints on selected variables.

    Parameters
    ----------
    sigma : np.ndarray
        Error covariance matrix Σ with shape ``(n, n)``.
    beta : np.ndarray
        VAR coefficient matrix with shape ``(k, n)``.
    C : np.ndarray
        Selection matrix for constraints with shape ``(nb_constraints, H*n)``.
    f : np.ndarray
        Location (mean) of constraint distribution with shape
        ``(nb_constraints,)``.
    Sigma_f : np.ndarray
        Scale (covariance) of constraint distribution with shape
        ``(nb_constraints, nb_constraints)``.
    shape_f : np.ndarray
        Shape (skewness) parameters of constraint distribution with shape
        ``(nb_constraints,)``.
    last_p_obs : np.ndarray
        Last p observations before forecast period with shape ``(p, n)``.
    p : int
        Number of lags.
    n : int
        Number of variables.
    h : int
        Number of COVID dummies.
    H : int
        Lags, variables, COVID dummies, horizon.
    point_only : bool
        If True, return only the conditional mean (no sampling).
    constraint_sampler : Optional[Callable]
        Custom function to sample from constraint distribution.
    method : str
        Algorithm: ``"andersson_et_al"``, ``"antolin_diaz_et_al"``,
        or ``"labonne_renzetti"``.
    rng : Optional[np.random.Generator]
        Generator for the stochastic draws. Defaults to a fresh
        ``numpy.random.default_rng()`` if not given.

    Returns
    -------
    forecast_i : np.ndarray
        Forecast values (flattened) with shape ``(H*n,)``.

    Raises
    ------
    NotImplementedError
        If ``method`` is not implemented.
    ValueError
        If a constraint or forecast input is invalid.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Contemporaneous matrix (lower Cholesky factor)
    A_0 = np.linalg.cholesky(sigma).T

    # Construct forecast matrices: y = b + M'ε
    b, bigM, _, _, _ = construct_forecast_matrices(
        last_p_obs,
        A_0,
        beta,
        p,
        n,
        h,
        H,
    )

    # D = C @ M' maps structural shocks to constrained forecasts
    D = C @ bigM.T
    D_star = np.linalg.pinv(D, rcond=1e-8)

    if point_only:
        mean_eps = D_star @ (f - C @ b.flatten())
        forecast_i = b.flatten() + mean_eps @ bigM
    else:
        if method == "andersson_et_al":
            forecast_i = _conditional_forecast_andersson_et_al(
                n,
                H,
                constraint_sampler,
                C,
                b,
                f,
                Sigma_f,
                shape_f,
                D,
                D_star,
                bigM,
                rng,
            )
        elif method == "antolin_diaz_et_al":
            if np.sum(Sigma_f) == 0:
                raise ValueError(
                    "'antolin_diaz_et_al' method do not work well when"
                    "the constraint variance is zero. Use 'andersson_et_al' instead."
                )
            mean_eps = D_star @ (f - C @ b.flatten())
            var_eps = (
                D_star @ Sigma_f @ D_star.T
                + np.eye(D_star.shape[0])
                - D_star @ D @ D.T @ D_star.T
            )
            epsilon_i = rng.multivariate_normal(mean_eps, var_eps, size=1)
            forecast_i = b.flatten() + epsilon_i.flatten() @ bigM
        elif method == "labonne_renzetti":
            forecast_i = sun_conditional_forecast(
                f=f,
                sigma_f=np.diag(Sigma_f),
                shape_f=shape_f,
                C=C,
                b=b.flatten(),
                BigM=bigM.T,
                n_draws=1,
                rng=rng,
            )
        else:
            raise NotImplementedError(
                "method should be andersson_et_al, antolin_diaz_et_al or labonne_renzetti"
            )

    return forecast_i


# ------------------------------------------------------------------
# Andersson et al. shock-adjustment algorithm
# ------------------------------------------------------------------


def _conditional_forecast_andersson_et_al(
    n: int,
    H: int,
    constraint_sampler: Optional[Callable],
    C: np.ndarray,
    b: np.ndarray,
    f: np.ndarray,
    Sigma_f: np.ndarray,
    shape_f: np.ndarray,
    D: np.ndarray,
    D_star: np.ndarray,
    bigM: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Shock-adjustment algorithm (Algorithm 1) from Andersson et al. (2010)."""
    if rng is None:
        rng = np.random.default_rng()

    # 1. Draw unconstrained shocks
    xi = rng.standard_normal(n * H)

    # 2. Draw target values for constraints
    if constraint_sampler is not None:
        constraint_draw = constraint_sampler() - C @ b.flatten()
    else:
        constraint_target_mean = f - C @ b.flatten()
        constraint_target_std = np.sqrt(np.diag(Sigma_f))

        if np.sum(shape_f) == 0:
            constraint_draw = (
                constraint_target_mean
                + constraint_target_std
                * rng.standard_normal(len(constraint_target_mean))
            )
        else:
            constraint_draw = skewnorm(
                a=shape_f,
                loc=constraint_target_mean,
                scale=constraint_target_std,
            ).rvs(random_state=rng)

    # 3. Adjust shocks to satisfy constraints: ε = ξ + D*(z - Dξ)
    epsilon_i = xi + D_star @ (constraint_draw - D @ xi)

    # 4. Compute forecast
    forecast_i = b.flatten() + bigM.T @ epsilon_i.flatten()

    return forecast_i
