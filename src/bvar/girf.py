"""
Generalised Impulse Response Functions (GIRFs) for BVAR
========================================================

Unlike orthogonalised IRFs (which depend on a Cholesky decomposition and are
therefore sensitive to variable ordering), GIRFs are order-invariant.  They
measure the expected response of all variables at horizon h to a one-standard-
deviation shock in variable i, integrating out the contemporaneous correlation
structure of the shocks via the reduced-form error covariance matrix Σ.

For a VAR(p) model:
    y_t = c + B_1 y_{t-1} + ... + B_p y_{t-p} + ε_t,   ε_t ~ N(0, Σ)

the MA(∞) representation (row-vector convention) is:
    y_t = μ + Σ_{h=0}^{∞} ε_{t-h} Ψ_h

where the reduced-form MA matrices Ψ_h (row-vector convention) satisfy:
    Ψ_0 = I_n
    Ψ_h = Σ_{j=1}^{min(h,p)} Ψ_{h-j} B_j,   h ≥ 1

Note: this uses the **row-vector** convention that matches the beta
parameterisation (Y = Z @ beta).  Ψ_h here equals the **transpose** of
the Ψ_h defined in Pesaran & Shin (1998), which uses column vectors.

The GIRF for a one-std-dev shock to variable i at horizon h is:
    GIRF_i(h) = Ψ_h.T @ Σ @ e_i / sqrt(Σ_{ii})
             = Σ[i, :] @ Ψ_h  / sqrt(Σ_{ii})   (equivalent, Σ symmetric)

where e_i is the i-th unit vector and Σ_{ii} = Σ[i, i].

The method propagates full BVAR uncertainty by computing GIRFs for every
posterior draw and reporting posterior quantiles.

Classes
-------
GIRF : class
    Mixin class providing GIRF methods for the BVAR class.

Functions
---------
_compute_ma_matrices : function
    Numba-accelerated computation of reduced-form MA matrices Ψ_h.
_compute_girf_for_draw : function
    Compute GIRFs for a single posterior draw.

References
----------
Pesaran, M. H., & Shin, Y. (1998). Generalised impulse response analysis in
    linear multivariate models. Economics Letters, 58(1), 17-29.
Koop, G., Pesaran, M. H., & Potter, S. M. (1996). Impulse response analysis in
    nonlinear multivariate models. Journal of Econometrics, 74(1), 119-147.
"""

from __future__ import annotations

import re
from numbers import Integral
from typing import Optional

import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm

from .utils import _validate_positive_integer

_ABSOLUTE_BASELINE_RESPONSES = {
    "level_change",
    "pct_change",
    "change_yoy",
    "pct_change_yoy",
}
_VALID_TRANSFORMATION_STATES = {"levels", "logs", "log levels", "diff", "log diff"}


def _validate_non_negative_integer(value: object, name: str) -> int:
    """Validate and normalise a non-negative integer argument."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _normalise_base_values(
    base_value: Optional[float | list[float] | np.ndarray], n: int
) -> Optional[np.ndarray]:
    """Normalise a scalar or per-variable baseline to a one-dimensional array."""
    if base_value is None:
        return None

    try:
        values = np.asarray(base_value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "base_value must be a numeric scalar or one value per variable."
        ) from exc

    if values.ndim == 0:
        values = np.full(n, values.item(), dtype=float)
    elif values.shape != (n,):
        raise ValueError(f"base_value must be a scalar or an array of length {n}.")
    if not np.isfinite(values).all():
        raise ValueError("base_value must contain only finite values.")
    return values.copy()


def _get_metadata_value(
    metadata: dict, var_name: object, var_idx: int, default: object = None
) -> object:
    """Resolve metadata by variable name, then by non-boolean integer index."""
    if isinstance(var_name, str) and var_name in metadata:
        return metadata[var_name]

    for key, value in metadata.items():
        if (
            isinstance(key, Integral)
            and not isinstance(key, (bool, np.bool_))
            and int(key) == var_idx
        ):
            return value
    return default


def _normalise_transformation_state(value: object, var_name: object) -> str:
    """Normalise and validate a variable's data transformation state."""
    state = " ".join(
        str(value).strip().lower().replace("_", " ").replace("-", " ").split()
    )
    if state not in _VALID_TRANSFORMATION_STATES:
        expected = ", ".join(sorted(_VALID_TRANSFORMATION_STATES))
        raise ValueError(
            f"Unknown data transformation '{value}' for variable '{var_name}'. "
            f"Expected one of: {expected}."
        )
    return state


def _get_transformation(data_transformation: dict, var_name: object, var_idx: int):
    """Return a normalised variable transformation, defaulting to levels."""
    value = _get_metadata_value(data_transformation, var_name, var_idx, "levels")
    return _normalise_transformation_state(value, var_name)


def _validate_log_diff_base(base_value: float, var_name: object) -> None:
    """Require a valid absolute level for a log-differenced variable."""
    if base_value <= 0:
        raise ValueError(
            f"base_value for log_diff variable '{var_name}' must be strictly positive."
        )


def _validate_percentage_base(base_value: float, var_name: object) -> None:
    """Require a nonzero baseline for percentage responses."""
    if base_value == 0:
        raise ValueError(
            f"base_value for percentage response of variable '{var_name}' "
            "must be nonzero."
        )


class GIRF:
    """
    Generalised Impulse Response Function methods for the BVAR class.

    This mixin class provides methods for computing and plotting GIRFs from
    posterior draws of VAR coefficients and error covariance matrices.  All
    methods operate on the posterior distributions already estimated by
    ``BVAR.sample()``.

    GIRFs are order-invariant: unlike Cholesky-based orthogonalised IRFs they
    do not depend on the ordering of variables in the system.

    See module docstring for the mathematical details.
    """

    def compute_girf(
        self,
        H: int,
        N_draws: int = 5000,
        point_only: bool = False,
        data_transformation: Optional[dict] = None,
        response_type: Optional[dict] = None,
        shock_size: Optional[dict] = None,
        progressbar: bool = False,
        base_value: Optional[float | list[float] | np.ndarray] = None,
    ) -> GIRF:
        """
        Compute Generalised Impulse Response Functions from posterior draws.

        For each posterior draw of (β, Σ), the method computes the reduced-form
        MA matrices Ψ_0, …, Ψ_H and evaluates the GIRF formula of Pesaran &
        Shin (1998).

        IRF responses are kept in their raw model representation and also converted
        to a common "level_changes" representation for response transformations.

        Parameters
        ----------
        H : int
            Maximum horizon (number of periods ahead).
        N_draws : int
            Number of posterior draws to use. Default is 5000.
        point_only : bool
            If True, compute only for the posterior point estimates. Default is
            False.
        data_transformation : Optional[dict]
            Dictionary mapping variable names or integer indices to data
            transformation types: "logs", "log_diff", "levels", or "diff".
            Example: {"GDP": "log_diff", "Inflation": "levels"}. Default is None.
        response_type : Optional[dict]
            Dictionary mapping variable names to response types: "raw" (untransformed GIRF),
            "raw_cumulated", "level_change", "pct_change", "change_yoy", or "pct_change_yoy".
            YoY types are only available for data with sufficient frequency
            (e.g., monthly, quarterly). If None, all default to "raw".
            Example: {"GDP": "pct_change_yoy", "Inflation": "level_change"}.
            Default is None.
        shock_size : Optional[dict]
            Dictionary mapping variable names to shock magnitudes in the variable's
            natural units. If None or a variable is missing, defaults to 1 std-dev.
            Example: {"GDP": 0.02, "Inflation": 0.5} for 2% increase in log GDP
            and 0.5 unit increase in inflation (interpreted in data units).
            Default is None.
        progressbar : bool
            Whether to display a tqdm progress bar. Default is False.
        base_value : Optional[float | list[float] | np.ndarray]
            Absolute last observed level for variables recorded as ``"diff"`` or
            ``"log_diff"``. A scalar applies to every variable; an array-like
            value supplies one value per variable. Required for ``level_change``,
            ``pct_change``, ``change_yoy``, or ``pct_change_yoy`` responses of
            differenced variables. ``"logs"`` and ``"log_levels"`` derive their
            baseline from the exponentiated last observation instead.
        Returns
        -------
        GIRF
            The BVAR object with GIRF results stored in its IRF attributes.
            ``irf_draws`` has shape ``(N_draws, H+1, n, n)`` and
            ``irf_var_names`` has length ``n``.

        Raises
        ------
        NotImplementedError
            If the fitted model does not support GIRFs.
        RuntimeError
            If the model has not been fitted.
        ValueError
            If a supplied horizon, draw count, shock, or response is invalid.

        Notes
        -----
        **Two-stage pipeline:**

          1. **Conditional conversion to level_changes**: Raw IRFs are converted to a
              common "level_changes" representation for non-raw response types.

           - "logs" or "log": level_change = baseline * dlog(y)
           - "log_diff": level_change = baseline * Δlog(y)
           - "levels": level_change = y (already level changes)
           - "diff": level_change = Δy (already level changes)

        2. **Apply transformations**: From the appropriate representation, compute
           requested response_type for each variable independently.

           - If response_type[var]=="raw": untouched GIRF
           - Otherwise apply response_type (level_change or pct_change, etc.)

        **Shock scaling:**

        The default GIRF computes responses to a one-standard-deviation shock.
        Use shock_size to specify shocks in natural units. For example,
        shock_size={"GDP": 0.02} for a 2 percentage point shock to log GDP.
        The method converts natural units to standard-deviation equivalents
        with the error covariance matrix.
        """
        H = _validate_non_negative_integer(H, "H")
        N_draws = _validate_positive_integer(N_draws, "N_draws")

        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before computing GIRFs. Call sample() first."
            )
        if not (self.model.supports_girf and self.model.supports_gaussian_predictive):
            raise NotImplementedError(
                f"{type(self.model).__name__} does not support GIRFs "
                f"(supports_girf={self.model.supports_girf}, "
                f"supports_gaussian_predictive="
                f"{self.model.supports_gaussian_predictive}); this "
                "implementation assumes a Gaussian reduced-form predictive "
                "distribution, so both flags must be True. A non-Gaussian "
                "model needs its own GIRF implementation before it can set "
                "supports_girf=True."
            )

        n = self.n
        p = self.n_lags
        var_names = list(self.df_data.columns)
        transformations = dict(
            (self.data_transformation or {})
            if data_transformation is None
            else data_transformation
        )
        response_types = dict(response_type or {})
        shock_sizes = dict(shock_size or {})
        base_values = _normalise_base_values(base_value, n)

        for var_idx, var_name in enumerate(var_names):
            trans_type = _get_transformation(transformations, var_name, var_idx)
            requested_response = response_types.get(var_name, "raw")
            is_differenced = trans_type in {"diff", "log diff"}
            is_log_differenced = trans_type == "log diff"
            if base_values is not None and is_log_differenced:
                _validate_log_diff_base(base_values[var_idx], var_name)
            if is_differenced and requested_response in _ABSOLUTE_BASELINE_RESPONSES:
                if base_values is None:
                    raise ValueError(
                        f"base_value is required for "
                        f"{'log_diff' if trans_type == 'log diff' else trans_type} variable "
                        f"'{var_name}' when response_type is "
                        f"'{requested_response}'."
                    )
                if requested_response in {"pct_change", "pct_change_yoy"}:
                    _validate_percentage_base(base_values[var_idx], var_name)

        if point_only:
            N_actual = 1
            beta_draws = self.beta_point.reshape(1, -1)
            sigma_draws = self.sigma_point.reshape(1, -1)
        else:
            N_actual = min(self.N_draws, N_draws)
            beta_draws = self.beta[:N_actual, :]
            sigma_draws = self.sigma[:N_actual, :]

        # Storage: [draw, horizon, shock_var, response_var]
        irf_draws = np.zeros((N_actual, H + 1, n, n))

        for d in tqdm(
            range(N_actual),
            desc="Computing GIRFs",
            unit="draw",
            disable=not progressbar,
        ):
            beta_d = beta_draws[d, :].reshape(n, -1).T  # shape (k, n)
            sigma_d = sigma_draws[d, :].reshape(n, n)

            # Extract B_1, ..., B_p from beta (skip constant row 0 and covid dummies)
            B = np.zeros((p, n, n))
            for lag in range(p):
                B[lag] = beta_d[1 + lag * n : 1 + (lag + 1) * n, :]

            # Reduced-form MA matrices Ψ_0, ..., Ψ_H
            Psi = _compute_ma_matrices(B, n, p, H)

            # GIRF: response of all variables to 1-std-dev shock in each variable
            irf_draws[d] = _compute_girf_for_draw(Psi, sigma_d, n, H)

            shock_scale = _get_shock_scale(
                shock_sizes,
                var_names,
                transformations,
                sigma_d,
                self.data,
                self.df_data,
            )
            for s in range(n):
                irf_draws[d, :, s, :] *= shock_scale[s]

        # Two-stage conversion pipeline
        # Stage 1: Build the level-change representation without replacing raw IRFs.
        irf_level_changes = _convert_to_level_changes(
            irf_draws,
            transformations,
            self.df_data,
            self.data,
            n,
            base_value=base_values,
            require_differenced_base=any(
                _get_transformation(transformations, var_name, var_idx) == "log diff"
                and response_types.get(var_name, "raw") in _ABSOLUTE_BASELINE_RESPONSES
                for var_idx, var_name in enumerate(var_names)
            ),
        )

        # Stage 2: Apply response_type transformations
        irf_draws = _apply_response_transforms(
            irf_draws,
            irf_level_changes,
            transformations,
            response_types,
            self.df_data,
            self.data,
            n,
            self.freq,
            base_value=base_values,
        )

        # Summarise with posterior quantiles
        quantile_levels = [0.16, 0.50, 0.84]
        irf_summary = {q: np.quantile(irf_draws, q, axis=0) for q in quantile_levels}

        # Store results
        self.irf_draws = irf_draws
        self.irf_summary = irf_summary
        self.irf_H = H
        self.irf_var_names = var_names
        self.irf_response_type = response_types.copy()
        self.irf_shock_size = shock_sizes.copy()

        # Compute and store tidy dataframe
        self.irf_df = _girf_to_dataframe(irf_draws, self.irf_var_names, H, n)

        return self


# ---------------------------------------------------------------------------
# Module-level helper functions: two-stage conversion pipeline
# ---------------------------------------------------------------------------


def _get_shock_scale(
    shock_size: Optional[dict],
    var_names: list[str],
    data_transformation: dict,
    sigma: np.ndarray,
    data: np.ndarray,
    df_data: pd.DataFrame,
) -> np.ndarray:
    """
    Get shock scaling factors for each variable, converting natural units to std-dev equivalents.

    Parameters
    ----------
    shock_size : Optional[dict]
        Dictionary mapping variable names to shock magnitudes in natural units.
        If None or a variable is missing, defaults to 1.0 (one std-dev).
    var_names : list[str]
        Variable names.
    data_transformation : dict
        Variable -> transformation type mapping.
    sigma : np.ndarray
        Error covariance matrix (typically the posterior point estimate), with
        shape ``(n, n)``.
    data : np.ndarray
        Raw data with shape ``(T, n)``.
    df_data : pd.DataFrame
        DataFrame with variable names.

    Returns
    -------
    scale : np.ndarray
        Shock scaling factors with shape ``(n,)``: natural_unit / std_dev for
        each variable. Default is 1.0 (one std-dev).
    """
    n = len(var_names)
    scale = np.ones(n)

    for idx, var_name in enumerate(var_names):
        _get_transformation(data_transformation, var_name, idx)

    if shock_size:
        # Compute standard deviations for each variable
        stds = np.sqrt(np.diag(sigma))

        for idx, var_name in enumerate(var_names):
            if var_name in shock_size:
                shock_natural = shock_size[var_name]
                # Convert natural units to std-dev scale factor
                # scale[idx] = shock_natural / stds[idx] means:
                # if shock_natural = stds[idx], scale = 1.0 (one std-dev)
                # if shock_natural = 2*stds[idx], scale = 2.0 (two std-devs)
                scale[idx] = shock_natural / stds[idx]

    return scale


def _get_baseline_values(
    data: np.ndarray,
    df_data: pd.DataFrame,
    data_transformation: dict,
    base_value: Optional[np.ndarray] = None,
    require_differenced_base: bool = False,
) -> np.ndarray:
    """
    Get baseline levels for normalisation.

    For log-transformed variables, exponentiate the last observation to recover
    the level. For level/diff variables, use the last observation directly.

    Parameters
    ----------
    data : np.ndarray
        Raw data from model with shape ``(T, n)``.
    df_data : pd.DataFrame
        DataFrame with variable names.
    data_transformation : dict
        Mapping of variable names or integer indices to transformation types.
    base_value : Optional[np.ndarray]
        Absolute levels supplied for differenced variables, as a scalar or an
        array with shape ``(n,)``.
    require_differenced_base : bool
        Whether a missing ``base_value`` should raise for differenced variables.
        If ``False``, raw-only responses return ``nan`` for missing
        log-difference baselines, while diff baselines use the last value.

    Returns
    -------
    baseline : np.ndarray
        Baseline level for each variable, with shape ``(n,)``.

    Raises
    ------
    ValueError
        If a required differenced baseline is missing or invalid.
    """
    n = data.shape[1]
    baseline = np.zeros(n)
    var_names = list(df_data.columns)
    base_values = _normalise_base_values(base_value, n)

    for var_idx, var_name in enumerate(var_names):
        trans_type = _get_transformation(data_transformation, var_name, var_idx)

        if trans_type == "log diff":
            if base_values is None:
                if require_differenced_base:
                    raise ValueError(
                        f"base_value is required for log_diff variable '{var_name}'."
                    )
                baseline[var_idx] = np.nan
            else:
                _validate_log_diff_base(base_values[var_idx], var_name)
                baseline[var_idx] = base_values[var_idx]
        elif trans_type == "diff":
            if base_values is None:
                if require_differenced_base:
                    raise ValueError(
                        f"base_value is required for diff variable '{var_name}'."
                    )
                baseline[var_idx] = data[-1, var_idx]
            else:
                baseline[var_idx] = base_values[var_idx]
        elif trans_type in {"logs", "log levels"}:
            # For log-transformed, recover level from exp of last observation
            baseline[var_idx] = np.exp(data[-1, var_idx])
        else:
            # For level-based, use last observation directly
            baseline[var_idx] = data[-1, var_idx]

    return baseline


def _convert_to_level_changes(
    irf_draws: np.ndarray,
    data_transformation: dict,
    df_data: pd.DataFrame,
    data: np.ndarray,
    n: int,
    base_value: Optional[np.ndarray] = None,
    require_differenced_base: bool = True,
) -> np.ndarray:
    """
    Stage 1: Convert raw IRFs to common "level_changes" representation.

    For all raw IRF responses, convert them to period-on-period changes in levels.
    The code uses this common baseline for every response type.

    Conversions:
    - "logs" (undifferenced): IRF is cumulative log-level; differentiate along h
      to get period-on-period log changes, then scale: level_change[h] = baseline * (irf[h] - irf[h-1])
    - "log_diff" (differenced): IRF is already period-on-period growth rate;
      level_change[h] = baseline * irf[h]
    - "levels" (undifferenced): level_change = y (already period-on-period level changes)
    - "diff" (differenced): level_change = Δy (already period-on-period level changes)

    Parameters
    ----------
    irf_draws : np.ndarray
        Raw IRF draws from _compute_girf_for_draw, with shape
        ``(N_draws, H+1, n, n)``.
    data_transformation : dict
        Variable names or integer indices -> transformation type mapping.
    df_data : pd.DataFrame
        DataFrame with variable names.
    data : np.ndarray
        Raw data with shape ``(T, n)``.
    n : int
        Number of variables.
    base_value : Optional[np.ndarray]
        Absolute levels supplied for differenced variables, as a scalar or an
        array with shape ``(n,)``.
    require_differenced_base : bool
        Whether missing ``base_value`` should raise for differenced variables.
        Raw-only GIRFs can leave this False because no absolute conversion is
        required.

    Returns
    -------
    irf_level_changes : np.ndarray
        IRFs converted to level_changes representation, with shape
        ``(N_draws, H+1, n, n)``.
    """
    irf_level_changes = irf_draws.copy()
    baseline = _get_baseline_values(
        data,
        df_data,
        data_transformation,
        base_value=base_value,
        require_differenced_base=require_differenced_base,
    )
    var_names = list(df_data.columns)

    for resp_idx, var_name in enumerate(var_names):
        trans_type = _get_transformation(data_transformation, var_name, resp_idx)

        # For log-transformed variables, convert to level changes
        if trans_type == "log diff":
            # log_diff (differenced): IRF at h is the period-on-period growth rate.
            # Multiply by baseline to get period-on-period level changes.
            if base_value is not None:
                irf_level_changes[:, :, :, resp_idx] = (
                    irf_draws[:, :, :, resp_idx] * baseline[resp_idx]
                )
        elif trans_type in {"logs", "log levels"}:
            # logs (undifferenced): IRF at h is the cumulative log-level response.
            # Differentiate along the horizon axis to get period-on-period log changes,
            # then scale by baseline to get period-on-period level changes.
            raw = irf_draws[:, :, :, resp_idx]  # shape (N_draws, H+1, n)
            zero_pad = np.zeros_like(raw[:, 0:1, :])
            irf_level_changes[:, :, :, resp_idx] = (
                np.diff(np.concatenate([zero_pad, raw], axis=1), axis=1)
                * baseline[resp_idx]
            )
        # For level-based variables: already period-on-period level changes, keep as-is

    return irf_level_changes


def _get_periods_per_year(freq_str: str) -> int:
    """
    Map pandas frequency string to periods per year.

    Parameters
    ----------
    freq_str : str
        Pandas frequency string, possibly anchored (e.g., "M", "Q", "Q-DEC",
        "W-SUN", "2M").

    Returns
    -------
    periods : int
        Number of periods in a year. Returns 1 if frequency is annual or unknown.

    Raises
    ------
    ValueError
        If the frequency multiplier is non-positive or too large.
    """
    freq_str = str(freq_str).upper() if freq_str else "unknown"

    freq_map = {
        "D": 365,  # daily
        "B": 252,  # business days
        "W": 52,  # weekly
        "M": 12,  # monthly
        "ME": 12,  # month-end
        "MS": 12,  # month-start
        "Q": 4,  # quarterly
        "QE": 4,  # quarter-end
        "QS": 4,  # quarter-start
        "Y": 1,  # yearly
        "A": 1,  # annual
        "H": 8760,  # hourly
    }

    match = re.fullmatch(r"(?P<multiplier>\d+)?(?P<base>[A-Z]+)(?:-.+)?", freq_str)
    if match is None:
        return 1

    multiplier = int(match.group("multiplier") or 1)
    base_freq = match.group("base")
    periods = freq_map.get(base_freq)
    if periods is None:
        return 1
    if multiplier <= 0:
        raise ValueError("Frequency multiplier must be positive.")
    if multiplier > periods:
        raise ValueError(
            f"Frequency multiplier {multiplier} is too large for '{base_freq}'."
        )

    # Represent a non-divisor with the largest complete number of periods.
    return periods // multiplier


def _apply_response_transforms(
    irf_raw: np.ndarray,
    irf_level_changes: np.ndarray,
    data_transformation: dict,
    response_type: dict,
    df_data: pd.DataFrame,
    data: np.ndarray,
    n: int,
    freq_str: str,
    base_value: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Stage 2: Apply response_type transformations.

    Selects the appropriate baseline IRF representation (raw or level_changes)
    based on response_type, applies cumulative transform if requested, then
    applies response_type-specific transformations (including YoY if applicable).

    Parameters
    ----------
    irf_raw : np.ndarray
        Raw IRFs after any shock scaling, with shape
        ``(N_draws, H+1, n, n)``.
    irf_level_changes : np.ndarray
        IRFs converted to level_changes representation (from stage 1), with
        shape ``(N_draws, H+1, n, n)``.
    data_transformation : dict
        Variable -> transformation type mapping.
    response_type : dict
        Variable -> response type ("raw", "raw_cumulated", "level_change", "pct_change", "change_yoy", "pct_change_yoy").
    df_data : pd.DataFrame
        DataFrame with variable names.
    data : np.ndarray
        Raw data with shape ``(T, n)``.
    n : int
        Number of variables.
    freq_str : str
        Pandas frequency string (e.g., "M", "Q").
    base_value : Optional[np.ndarray]
        Absolute levels supplied for differenced variables, as a scalar or an
        array with shape ``(n,)``.

    Returns
    -------
    irf_transformed : np.ndarray
        IRFs with transformations applied, with shape
        ``(N_draws, H+1, n, n)``.

    Raises
    ------
    ValueError
        If a response type or baseline value is invalid.
    """
    irf_transformed = irf_raw.copy()
    var_names = list(df_data.columns)
    base_values = _normalise_base_values(base_value, n)
    periods_per_year = _get_periods_per_year(freq_str)
    needs_baseline = any(
        response_type.get(var_name, "raw") in _ABSOLUTE_BASELINE_RESPONSES
        for var_name in var_names
    )
    baseline = (
        _get_baseline_values(
            data,
            df_data,
            data_transformation,
            base_value=base_values,
            require_differenced_base=False,
        )
        if needs_baseline
        else None
    )

    for resp_idx, var_name in enumerate(var_names):
        # Get response type (default: raw)
        resp_type = response_type.get(var_name, "raw")
        trans_type = _get_transformation(data_transformation, var_name, resp_idx)
        if (
            trans_type in {"diff", "log diff"}
            and resp_type in _ABSOLUTE_BASELINE_RESPONSES
        ):
            if base_values is None:
                raise ValueError(
                    f"base_value is required for {trans_type} variable "
                    f"'{var_name}' when response_type is '{resp_type}'."
                )
            if trans_type == "log diff":
                _validate_log_diff_base(base_values[resp_idx], var_name)
            if resp_type in {"pct_change", "pct_change_yoy"}:
                _validate_percentage_base(base_values[resp_idx], var_name)

        # Apply response type transformation
        if resp_type == "raw":
            pass

        # Apply cumulative transform if requested
        elif resp_type == "raw_cumulated":
            irf_transformed[:, :, :, resp_idx] = np.cumsum(
                irf_raw[:, :, :, resp_idx], axis=1
            )

        elif resp_type == "level_change":
            irf_transformed[:, :, :, resp_idx] = irf_level_changes[:, :, :, resp_idx]

        elif resp_type == "pct_change":
            # Convert to percentage change relative to baseline
            irf_transformed[:, :, :, resp_idx] = irf_level_changes[:, :, :, resp_idx]
            base_val = baseline[resp_idx]
            _validate_percentage_base(base_val, var_name)
            irf_transformed[:, :, :, resp_idx] = (
                irf_transformed[:, :, :, resp_idx] / base_val
            ) * 100.0

        elif resp_type == "change_yoy":
            # Year-over-year change: cumulative response over periods_per_year
            # For first periods_per_year horizons, use cumsum from start (no prior year available)
            irf_transformed[:, :, :, resp_idx] = irf_level_changes[:, :, :, resp_idx]
            H = irf_transformed.shape[1] - 1
            yoy_responses = np.zeros_like(irf_transformed[:, :, :, resp_idx])
            for h in range(H + 1):
                if h < periods_per_year:
                    # No prior-year data yet: cumsum from 0 to h
                    yoy_responses[:, h, :] = np.sum(
                        irf_transformed[:, 0 : h + 1, :, resp_idx], axis=1
                    )
                else:
                    # Full year available: sum from h-periods_per_year+1 to h
                    min_h = h - periods_per_year + 1
                    yoy_responses[:, h, :] = np.sum(
                        irf_transformed[:, min_h : h + 1, :, resp_idx], axis=1
                    )
            irf_transformed[:, :, :, resp_idx] = yoy_responses

        elif resp_type == "pct_change_yoy":
            # Year-over-year percentage change: cumulative % change over periods_per_year
            # For first periods_per_year horizons, use cumsum from start (no prior year available)
            irf_transformed[:, :, :, resp_idx] = irf_level_changes[:, :, :, resp_idx]
            H = irf_transformed.shape[1] - 1
            base_val = baseline[resp_idx]
            yoy_responses = np.zeros_like(irf_transformed[:, :, :, resp_idx])
            for h in range(H + 1):
                if h < periods_per_year:
                    # No prior-year data yet: cumsum from 0 to h
                    cumsum_yoy = np.sum(
                        irf_transformed[:, 0 : h + 1, :, resp_idx], axis=1
                    )
                else:
                    # Full year available: sum from h-periods_per_year+1 to h
                    min_h = h - periods_per_year + 1
                    cumsum_yoy = np.sum(
                        irf_transformed[:, min_h : h + 1, :, resp_idx], axis=1
                    )

                _validate_percentage_base(base_val, var_name)
                yoy_responses[:, h, :] = (cumsum_yoy / base_val) * 100.0
            irf_transformed[:, :, :, resp_idx] = yoy_responses

        else:
            raise ValueError(
                f"Unknown response_type '{resp_type}' for variable '{var_name}'. "
                f"Must be 'raw', 'raw_cumulated', 'level_change', 'pct_change', 'change_yoy', or 'pct_change_yoy'."
            )

    return irf_transformed


def _girf_to_dataframe(
    irf_draws: np.ndarray,
    var_names: list[str],
    H: int,
    n: int,
    quantiles: tuple[float, ...] = (0.16, 0.50, 0.84),
) -> pd.DataFrame:
    """
    Convert GIRF draws to tidy long-format DataFrame.

    Parameters
    ----------
    irf_draws : np.ndarray
        IRF draws already transformed to desired response types, with shape
        ``(N_draws, H+1, n, n)``.
    var_names : list[str]
        Variable names with length ``n``.
    H : int
        Maximum horizon.
    n : int
        Number of variables.
    quantiles : tuple[float, ...]
        Quantile levels to include. Default is (0.16, 0.50, 0.84).

    Returns
    -------
    df : pd.DataFrame
        Long-format DataFrame with columns:
        ``["horizon", "shock", "response", "quantile", "value"]``.
    """
    rows = []

    for q in quantiles:
        q_vals = np.quantile(irf_draws, q, axis=0)  # shape (H+1, n, n)
        for h in range(H + 1):
            for s in range(n):
                for r in range(n):
                    rows.append(
                        {
                            "horizon": h,
                            "shock": var_names[s],
                            "response": var_names[r],
                            "quantile": q,
                            "value": q_vals[h, s, r],
                        }
                    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helper functions used by compute_girf()
# ---------------------------------------------------------------------------


@njit
def _compute_ma_matrices(
    B: np.ndarray,
    n: int,
    p: int,
    H: int,
) -> np.ndarray:
    """
    Compute reduced-form MA coefficient matrices Ψ_0, …, Ψ_H.

    Uses the VAR recursion:
        Ψ_0 = I_n
        Ψ_h = Σ_{j=1}^{min(h,p)} Ψ_{h-j} B_j,   h ≥ 1

    Parameters
    ----------
    B : np.ndarray
        VAR lag coefficient matrices B_1, …, B_p, with shape ``(p, n, n)``.
    n : int
        Number of variables.
    p : int
        Number of lags.
    H : int
        Maximum horizon.

    Returns
    -------
    Psi : np.ndarray
        Reduced-form MA matrices with shape ``(H+1, n, n)``.

    Notes
    -----
    Compiled with Numba ``@njit`` for performance.
    """
    Psi = np.zeros((H + 1, n, n))

    # Ψ_0 = I_n
    for i in range(n):
        Psi[0, i, i] = 1.0

    for h in range(1, H + 1):
        for j in range(1, min(h, p) + 1):
            Psi[h] += Psi[h - j] @ B[j - 1]

    return Psi


def _compute_girf_for_draw(
    Psi: np.ndarray,
    sigma: np.ndarray,
    n: int,
    H: int,
) -> np.ndarray:
    """
    Compute GIRFs for a single posterior draw given MA matrices and Σ.

    GIRF formula (row-vector convention matching beta parameterisation):
        GIRF_i(h) = Ψ_h.T @ Σ @ e_i / sqrt(Σ[i, i])
                  = Σ[i,:] @ Ψ_h / sqrt(Σ[i, i])   (Σ symmetric)

    Parameters
    ----------
    Psi : np.ndarray
        Reduced-form MA matrices with shape ``(H+1, n, n)``.
    sigma : np.ndarray
        Error covariance matrix Σ with shape ``(n, n)``.
    n : int
        Number of variables.
    H : int
        Maximum horizon.

    Returns
    -------
    girf : np.ndarray
        Generalised impulse responses with shape ``(H+1, n, n)``. ``girf[h, i,
        j]`` is the response of variable j at horizon h to a one-std-dev shock
        in variable i.
    """
    girf = np.zeros((H + 1, n, n))

    for i in range(n):
        # σ_i: i-th column of Σ  (contemporaneous covariances with variable i)
        sigma_col_i = sigma[:, i]
        std_i = np.sqrt(sigma[i, i])

        for h in range(H + 1):
            # The beta/Psi matrices use the row-vector convention:
            #   y_t = c + y_{t-1} B_1 + ... + ε_t  (row-vector)
            # so Psi[h] = Ψ_h^{row} = (Ψ_h^{PS})ᵀ.
            # In row-vector convention the GIRF is Σ[i,:] @ Ψ_h^{row} / std_i
            # which equals Ψ_h^{row}.T @ Σ[:,i] / std_i (Σ is symmetric).
            girf[h, i, :] = Psi[h].T @ sigma_col_i / std_i

    return girf


def _resolve_var_indices(
    var: Optional[str | int | list],
    var_names: list[str],
    argname: str,
) -> list[int]:
    """
    Resolve variable names or integer indices to a list of 0-based integers.

    Parameters
    ----------
    var : Optional[str | int | list]
        Variable specification.  If None, returns all indices.
    var_names : list[str]
        Full list of variable names.
    argname : str
        Name of the argument (for error messages).

    Returns
    -------
    indices : list[int]

    Raises
    ------
    TypeError
        If ``var`` contains an unsupported specification.
    ValueError
        If a variable name or index is invalid.
    """
    if var is None:
        return list(range(len(var_names)))

    if isinstance(var, (str, int)):
        var = [var]

    indices = []
    for v in var:
        if isinstance(v, str):
            if v not in var_names:
                raise ValueError(
                    f"'{v}' not found in variable names for argument '{argname}'. "
                    f"Available variables: {var_names}."
                )
            indices.append(var_names.index(v))
        elif isinstance(v, int):
            if not (0 <= v < len(var_names)):
                raise ValueError(
                    f"Index {v} out of range for argument '{argname}'. "
                    f"Valid indices: 0 to {len(var_names) - 1}."
                )
            indices.append(v)
        else:
            raise TypeError(
                f"Elements of '{argname}' must be strings or integers, got {type(v)}."
            )

    return indices
