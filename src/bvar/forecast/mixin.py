"""
Forecasting mixin for BVAR
===========================

Provides ``Forecasting``, the mixin class that ``BVAR`` inherits to gain
unconditional / conditional forecast methods.

Sibling modules handle the main calculations:

* :mod:`._matrices`     – companion-form matrix construction (numba)
* :mod:`._conditional`  – constrained-draw algorithms
* :mod:`._compare`      – forecast comparison utility
"""

from __future__ import annotations

import re
from numbers import Integral
from typing import Callable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..dummy_observations import stack_dummies
from ..models.base import PosteriorState
from ..utils import _validate_positive_integer, construct_Y_Z
from .conditional import get_constraint, validate_constraint_inputs
from .matrices import construct_forecast_matrices, h_step_forecast_loop


def _periods_per_year(freq: str) -> int:
    """Return the number of observations in a calendar year for ``freq``."""
    freq = str(freq).upper() if freq else "unknown"
    freq_map = {
        "D": 365,
        "B": 252,
        "W": 52,
        "M": 12,
        "ME": 12,
        "MS": 12,
        "Q": 4,
        "QE": 4,
        "QS": 4,
        "Y": 1,
        "A": 1,
        "H": 8760,
    }

    match = re.fullmatch(r"(?P<multiplier>\d+)?(?P<base>[A-Z]+)(?:-.+)?", freq)
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

    return periods // multiplier


class Forecasting:
    """
    Forecasting methods for the BVAR class.

    This class provides methods for generating both unconditional and conditional
    forecasts from Bayesian Vector Autoregression (BVAR) models. It implements
    the Waggoner & Zha (1999) and Antolín-Díaz et al. (2021) algorithms for
    conditional forecasting. It supports mean, variance, and skewness
    constraints.
    """

    @staticmethod
    def _validate_n_draws(N_draws: int) -> int:
        """Validate and normalise a requested forecast draw count."""
        if (
            isinstance(N_draws, bool)
            or not isinstance(N_draws, Integral)
            or N_draws <= 0
        ):
            raise ValueError("N_draws must be a positive integer.")
        return int(N_draws)

    @staticmethod
    def _validate_horizon(H: int) -> int:
        """Validate and normalise a requested forecast horizon."""
        return _validate_positive_integer(H, "H")

    def recursive_forecast(
        self,
        H: int,
        N_draws: int = 5000,
        point_only: bool = False,
        progressbar: bool = False,
        random_state: Optional[int] = None,
    ) -> Forecasting:
        """
        Generate unconditional forecasts from the BVAR posterior draws using recursive form.

        This function is mainly useful when using the conditional mean only (otherwise use self.forecast).
        Also used to double-check the results of self.forecast which is faster but more complex.

        Parameters
        ----------
        H : int
            Forecast horizon (number of steps ahead to forecast).
        N_draws : int
            Number of posterior draws to use for forecasting. Default is 5000.
        point_only : bool
            If True, compute a single plug-in forecast using the stored
            posterior point estimates. Default is False.
        progressbar : bool
            Whether to display a progress bar. Default is False.
        random_state : Optional[int]
            Seed or generator controlling the residual draws. If given,
            overrides the generator set at construction and is reused by
            later ``forecast``/``recursive_forecast`` calls; otherwise the
            method uses the instance generator. The global NumPy random
            state remains unchanged. Default is None.

        Returns
        -------
        self : Forecasting
            The BVAR object with forecasts stored in self.forecast_unconditional
            (shape: N_draws x (T+H) x n).

        Raises
        ------
        RuntimeError
            If the model has no posterior draws (``sample()`` was not called).
        """
        H = self._validate_horizon(H)
        n = self.n

        N_draws = self._validate_n_draws(N_draws)
        if not self.is_fitted:
            raise RuntimeError(
                "No posterior draws available. Call sample() before forecast()."
            )
        if random_state is not None:
            self.rng = np.random.default_rng(random_state)

        if point_only:
            N_draws = 1
            beta_draws = np.zeros((1, self.beta_point.shape[0]))
            beta_draws[0] = self.beta_point
            sigma_draws = np.zeros((1, self.sigma_point.shape[0]))
            sigma_draws[0] = self.sigma_point
        else:
            beta_draws = self.beta
            sigma_draws = self.sigma

        # Unpack
        p = self.n_lags
        N_draws = min(self.N_draws, N_draws)

        last_obs_index = self.data.shape[0]
        T = last_obs_index - p

        data = self.data

        y_h = np.zeros((N_draws, T + H + p, n))
        y_h[:, :last_obs_index, :] = data

        for draw in tqdm(
            range(N_draws),
            desc="Unconditional Forecast (Recursive form)",
            unit="iteration",
            disable=not progressbar,
        ):
            # Prepare the coefficient and covariance draw.
            beta = beta_draws[draw, :].reshape(n, -1).T

            # Pass a private copy of the full state to the innovation hook so
            # custom hooks cannot alter fitted posterior arrays.
            if point_only:
                state = self.posterior_state_point.copy()
            else:
                extras = self.extras[draw] if self.extras is not None else None
                state = PosteriorState(
                    beta=beta_draws[draw, :], sigma=sigma_draws[draw, :], extras=extras
                ).copy()
            epsilon = self.model.sample_innovations(
                state, H, rng=self.rng, point_only=point_only
            )

            x_t = np.zeros(beta.shape[0])  # initialisation
            x_t[0] = 1.0

            # H step ahead forecast
            y_h = h_step_forecast_loop(
                y_h, draw, last_obs_index, H, n, p, beta, epsilon
            )

        self.forecast_unconditional = y_h[:, p:, :]
        self.forecast_conditional = None
        self.H = H
        self.dates_forecast = pd.period_range(
            start=self.df_data.index[last_obs_index - 1] + 1, periods=H, freq=self.freq
        )

        return self

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def forecast(
        self,
        H: int,
        constraint_mean: Optional[np.ndarray] = None,
        constraint_variance: Optional[np.ndarray] = None,
        constraint_shape: Optional[np.ndarray] = None,
        method: str = "andersson_et_al",
        N_draws: int = 5000,
        N_burn: Optional[int] = None,
        point_only: bool = False,
        format: bool = False,
        quantiles: Optional[list] = None,
        base_value: Optional[np.ndarray] = None,
        constraint_sampler: Optional[Callable] = None,
        progressbar: bool = False,
        transformations: Optional[dict] = None,
        random_state: Optional[int] = None,
    ) -> Forecasting:
        """
        Generate conditional forecasts from the BVAR posterior draws.

        Implementation following Waggoner & Zha (1999) and Antolín-Díaz et al. (2021).
        Supports mean, variance, and shape/skewness constraints on forecasts.

        Parameters
        ----------
        H : int
            Forecast horizon (number of steps ahead).
        constraint_mean : Optional[np.ndarray]
            Values to impose for the mean of the conditioned variables.
            Shape ``(H, n)``. Unconstrained values should be NaNs. Default is
            None (unconditional forecast).
        constraint_variance : Optional[np.ndarray]
            Values to impose for the variance of the conditioned variables.
            Shape ``(H, n)``. Unconstrained values should be NaNs. Default is
            None (no variance constraints).
        constraint_shape : Optional[np.ndarray]
            Values to impose for the shape/skewness of the conditioned variables.
            Shape ``(H, n)``. Unconstrained values should be NaNs. Default is
            None (no skewness constraints).
        method : str
            Method for handling constraints. Default is "andersson_et_al".
        N_draws : int
            Number of MCMC draws used to simulate the uncertainty
            of conditional forecasts. Default is 5000. Must be positive;
            capped at ``self.N_draws`` (the number of stored posterior
            draws) before the method sets the default burn-in.
        N_burn : Optional[int]
            Number of burn-in draws to discard. Default is
            ``effective_N_draws // 2`` if None, where ``effective_N_draws``
            is ``N_draws`` after capping. If given explicitly, must be an
            integer in ``[0, effective_N_draws)``.
        point_only : bool
            If True, returns only the conditional mean using the
            median draw from the Bayesian sampling (much faster). Default is False.
        format : bool
            Whether to format the output as a DataFrame with dates. Default is False.
        quantiles : Optional[list]
            List of quantiles to compute if format is True. Default is [0.16, 0.5, 0.84].
        base_value : Optional[np.ndarray]
            Absolute level at the end of the observed sample, required when a
            forecast tail contains rows reconstructed from ``"diff"`` or
            ``"log_diff"`` data. May be a scalar or one value per variable.
        constraint_sampler : Optional[Callable]
            Custom constraint sampler function. Default is None.
        progressbar : bool
            Whether to display a progress bar during forecasting. Default is False.
        transformations : Optional[dict]
            Dictionary mapping variable names or indices to transformation types
            to APPLY to the forecasts.  Supported: ``"qoq"``, ``"yoy"``.
        random_state : Optional[int]
            Seed or generator controlling the stochastic forecast draws. If
            given, overrides the generator set at construction and is reused
            in later ``forecast``/``recursive_forecast`` calls; otherwise the
            method uses the instance generator. The global NumPy random
            state remains unchanged. Default is None.

        Returns
        -------
        Forecasting
            The BVAR object with forecast results stored in its forecast,
            summary, horizon, and last-observation attributes.

        Raises
        ------
        RuntimeError
            If the model has no posterior draws (``sample()`` was not called).
        ValueError
            If forecast constraints, draw counts, or transformations are invalid.

        References
        ----------
        Waggoner, D. F., & Zha, T. (1999). Conditional forecasts in dynamic
            multivariate models.
        Antolín-Díaz, J., Petrella, I., & Rubio-Ramírez, J. F. (2021).
            Structural scenario analysis with SVARs.
        """
        H = self._validate_horizon(H)
        N_draws = self._validate_n_draws(N_draws)
        if not self.is_fitted:
            raise RuntimeError(
                "No posterior draws available. Call sample() before forecast()."
            )
        if constraint_mean is not None:
            validate_constraint_inputs(
                constraint_mean,
                constraint_variance,
                constraint_shape,
                expected_shape=(H, self.n),
            )
        if random_state is not None:
            self.rng = np.random.default_rng(random_state)

        if self.point_only and not point_only and constraint_mean is None:
            raise ValueError(
                "We need parameters draws to compute the distribution of the unconditional forecasts"
            )

        if quantiles is None:
            quantiles = [0.16, 0.5, 0.84]

        # N_burn is resolved inside _conditional_forecast, after N_draws is
        # capped to self.N_draws, so the default (N_draws // 2) is derived
        # from the effective draw count rather than the requested one.

        last_obs_index = self.data.shape[0]

        # Unconditional forecast
        if constraint_mean is None:
            y_unconditional, mean_H, variance_H = self._unconditional_forecast(
                H,
                N_draws,
                point_only,
                progressbar,
            )
        else:
            y_unconditional, _, _ = self._unconditional_forecast(
                H,
                N_draws,
                True,
                progressbar,
            )
            mean_H, variance_H = None, None

        # Conditional forecast
        if constraint_mean is not None:
            y_conditional = self._conditional_forecast(
                constraint_mean,
                constraint_variance,
                constraint_shape,
                H,
                N_draws,
                point_only,
                progressbar,
                N_burn,
                constraint_sampler,
                method,
            )
        else:
            y_conditional = None

        # Apply forecast transformations (qoq, yoy, etc.)
        y_unconditional = self._apply_forecast_transformations(
            y_unconditional,
            transformations,
            base_value,
        )
        y_conditional = self._apply_forecast_transformations(
            y_conditional,
            transformations,
            base_value,
        )

        # Store results
        self.forecast_unconditional = y_unconditional
        self.forecast_conditional = y_conditional
        self.mean_H = mean_H
        self.variance_H = variance_H
        self.H = H
        self.last_obs_index = last_obs_index

        if format:
            self.df_forecasts_unconditional = self._summarise_with_dates(
                y_unconditional,
                quantiles,
            )
            self.df_forecasts_conditional = self._summarise_with_dates(
                y_conditional,
                quantiles,
            )

        return self

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _summarise_with_dates(
        self,
        forecasts: Optional[np.ndarray],
        quantiles: list,
    ) -> Optional[pd.DataFrame]:
        """Summarise forecasts with quantiles and add dates."""
        if forecasts is None:
            return None

        forecasts_quantiles = np.quantile(forecasts, quantiles, axis=0)

        df_forecasts_all = []

        for q in range(len(quantiles)):
            df_forecasts = pd.DataFrame(
                forecasts_quantiles[q],
                columns=self.df_data.columns,
            )
            df_forecasts["quantile"] = quantiles[q]

            dates_obs = self.df_data.index[self.n_lags :]
            dates_forecasts = pd.period_range(
                start=dates_obs[-1] + 1,
                periods=self.H,
                freq=self.freq,
            )
            dates = dates_obs.append(dates_forecasts)
            df_forecasts["date"] = dates

            df_forecasts_all.append(df_forecasts)

        df_forecasts_all = pd.concat(df_forecasts_all, axis=0)

        df_forecasts_long = df_forecasts_all.melt(
            id_vars=["date", "quantile"],
            var_name="variable",
            value_name="value",
        ).reset_index(drop=True)

        return df_forecasts_long

    def _apply_forecast_transformations(
        self,
        forecasts: Optional[np.ndarray],
        forecast_transformations: Optional[dict],
        base_value: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Apply forecast transformations (qoq, yoy) based on data transformations.

        Converts variables from their original state to actual levels, then
        applies requested growth-rate transformations.

        Supported data states (via ``data_transformation``):
        ``"log diff"``, ``"diff"``, ``"logs"``, ``"levels"``.

        Supported output transformations:
        ``"qoq"``  – quarter-over-quarter growth rate,
        ``"yoy"``  – year-over-year growth rate.
        ``None``  – no-op output transformation.

        For differenced data with a growth-rate output, historical rows remain
        in their stored differenced form and only the forecast tail contains
        rates computed from reconstructed levels.
        """
        import warnings

        if forecasts is None:
            return None

        if forecast_transformations is None or len(forecast_transformations) == 0:
            return forecasts

        if self.data_transformation is None:
            warnings.warn(
                "Cannot apply forecast transformations: data_transformation was not set during sampling. "
                "Please provide data_transformation in the sample() method. Skipping transformations.",
                UserWarning,
            )
            return forecasts

        forecasts_transformed = forecasts.copy()
        var_names = list(self.df_data.columns)

        for var_key, transform_type in forecast_transformations.items():
            # Resolve variable index and name
            if isinstance(var_key, (bool, np.bool_)):
                raise TypeError(f"Variable key must be str or int, got {type(var_key)}")
            if isinstance(var_key, str):
                if var_key not in var_names:
                    raise ValueError(f"Variable '{var_key}' not found in data columns.")
                var_idx = var_names.index(var_key)
            elif isinstance(var_key, int):
                if var_key < 0 or var_key >= len(var_names):
                    raise IndexError(f"Variable index {var_key} out of bounds.")
                var_idx = var_key
            else:
                raise TypeError(f"Variable key must be str or int, got {type(var_key)}")

            variable_name = var_names[var_idx]
            if variable_name in self.data_transformation:
                data_state = self.data_transformation[variable_name]
            else:
                data_state = None
                for metadata_key, metadata_value in self.data_transformation.items():
                    if (
                        isinstance(metadata_key, Integral)
                        and not isinstance(metadata_key, (bool, np.bool_))
                        and int(metadata_key) == var_idx
                    ):
                        data_state = metadata_value
                        break
            if data_state is None:
                raise ValueError(
                    f"Variable (key={var_key}, idx={var_idx}) not found in data_transformation dict"
                )

            # Normalise documented underscore labels while keeping aliases exact.
            state = " ".join(
                str(data_state)
                .strip()
                .lower()
                .replace("_", " ")
                .replace("-", " ")
                .split()
            )
            history_length = min(
                forecasts.shape[1],
                max(self.data.shape[0] - self.n_lags, 0),
            )
            forecast_values = forecasts_transformed[:, history_length:, var_idx]

            if state in {"log diff", "diff"} and forecast_values.shape[1] > 0:
                if base_value is None:
                    raise ValueError(
                        "base_value is required to reconstruct a forecast "
                        f"tail for '{data_state}' data."
                    )

                base_array = np.asarray(base_value)
                if base_array.ndim == 0:
                    base_val = base_array.item()
                elif base_array.ndim == 1 and base_array.size == len(var_names):
                    base_val = base_array[var_idx]
                else:
                    raise ValueError(
                        "base_value must be a scalar or contain one value per variable."
                    )

                try:
                    base_is_finite = np.isfinite(base_val)
                except TypeError:
                    base_is_finite = False
                if state == "log diff" and (not base_is_finite or base_val <= 0):
                    raise ValueError(
                        "base_value must be positive and finite for 'log_diff' "
                        "reconstruction."
                    )
                if not base_is_finite:
                    raise ValueError(
                        "base_value must be finite for differenced data reconstruction."
                    )

            # Step 1: Convert forecast values to actual levels (not logs).
            if state == "log diff":
                if forecast_values.shape[1] > 0:
                    cumulative = np.cumsum(forecast_values, axis=1)
                    cumulative = cumulative + np.log(base_val)
                    forecasts_transformed[:, history_length:, var_idx] = np.exp(
                        cumulative
                    )
            elif state == "diff":
                if forecast_values.shape[1] > 0:
                    forecasts_transformed[:, history_length:, var_idx] = (
                        np.cumsum(forecast_values, axis=1) + base_val
                    )

            elif state in {"logs", "log levels"}:
                forecasts_transformed[:, :, var_idx] = np.exp(
                    forecasts_transformed[:, :, var_idx]
                )

            elif state == "levels":
                pass
            else:
                raise ValueError(
                    f"Unknown data_state '{data_state}' for variable '{var_key}'. "
                    f"Expected one of: 'log diff', 'diff', 'logs', 'log levels', 'levels'"
                )

            # Step 2: Apply transformation on actual levels
            if (
                state in {"diff", "log diff"}
                and transform_type in {"qoq", "yoy"}
                and forecasts.shape[1] - history_length > 0
            ):
                history_end = min(self.data.shape[0], self.n_lags + history_length)
                observed_data = self.data[:, var_idx]
                observed_levels = np.empty(observed_data.shape[0])
                if observed_levels.size:
                    observed_levels[-1] = base_val
                    if observed_levels.size > 1:
                        cumulative = np.cumsum(observed_data[:0:-1])[::-1]
                        if state == "log diff":
                            observed_levels[:-1] = base_val * np.exp(-cumulative)
                        else:
                            observed_levels[:-1] = base_val - cumulative

                level_history = observed_levels[:history_end]
                if level_history.size == 0:
                    level_history = np.array([base_val])
                    history_end = 1
                level_path = np.concatenate(
                    [
                        np.broadcast_to(
                            level_history,
                            (forecasts.shape[0], level_history.size),
                        ),
                        forecasts_transformed[:, history_length:, var_idx],
                    ],
                    axis=1,
                )
                periods = 1 if transform_type == "qoq" else _periods_per_year(self.freq)
                tail_start = history_end
                tail_length = forecasts.shape[1] - history_length
                tail_transformed = np.full((forecasts.shape[0], tail_length), np.nan)
                first_valid = max(periods - tail_start, 0)
                if first_valid < tail_length:
                    current = level_path[
                        :, tail_start + first_valid : tail_start + tail_length
                    ]
                    previous = level_path[
                        :,
                        tail_start + first_valid - periods : tail_start
                        + tail_length
                        - periods,
                    ]
                    tail_transformed[:, first_valid:] = (current - previous) / previous
                forecasts_transformed[:, history_length:, var_idx] = tail_transformed

            elif transform_type == "qoq" and state not in {"diff", "log diff"}:
                forecasts_transformed[:, 1:, var_idx] = (
                    forecasts_transformed[:, 1:, var_idx]
                    - forecasts_transformed[:, :-1, var_idx]
                ) / forecasts_transformed[:, :-1, var_idx]
                forecasts_transformed[:, 0, var_idx] = np.nan

            elif transform_type == "yoy" and state not in {"diff", "log diff"}:
                periods = _periods_per_year(self.freq)
                forecasts_transformed[:, periods:, var_idx] = (
                    forecasts_transformed[:, periods:, var_idx]
                    - forecasts_transformed[:, :-periods, var_idx]
                ) / forecasts_transformed[:, :-periods, var_idx]
                forecasts_transformed[:, :periods, var_idx] = np.nan
            elif (
                state in {"diff", "log diff"}
                and transform_type in {"qoq", "yoy"}
                or transform_type is None
            ):
                pass
            else:
                raise ValueError(
                    f"Unsupported transformation type: '{transform_type}'. "
                    f"Supported types: 'qoq', 'yoy'"
                )

        return forecasts_transformed

    # ------------------------------------------------------------------
    # Unconditional forecast (companion form)
    # ------------------------------------------------------------------

    def _unconditional_forecast(
        self,
        H: int,
        N_draws: int,
        point_only: bool,
        progressbar: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate unconditional forecasts using the companion form representation.
        """
        if point_only:
            N_unconditional = 1
            beta_draws = np.zeros((1, self.beta_point.shape[0]))
            beta_draws[0] = self.beta_point
            sigma_draws = np.zeros((1, self.sigma_point.shape[0]))
            sigma_draws[0] = self.sigma_point
        else:
            N_unconditional = min(self.N_draws, N_draws)
            beta_draws = self.beta
            sigma_draws = self.sigma

        data = self.data.copy()

        p = self.n_lags
        T, n = data.shape
        T = T - p

        last_p_obs = data[-p:, :]
        h = len(self.covid_indices)

        y_unconditional = np.zeros((N_unconditional, T + H, n))
        y_unconditional[:, :T, :] = data[p:, :]

        mean_H = np.zeros((N_unconditional, n))
        variance_H = np.zeros((N_unconditional, n, n))

        for i in tqdm(
            range(N_unconditional),
            desc="Unconditional Forecast",
            unit="iteration",
            disable=not progressbar,
        ):
            sigma = sigma_draws[i, :].reshape(n, n)
            beta = beta_draws[i, :].reshape(n, -1).T

            A_0 = np.linalg.cholesky(sigma).T

            b, bigM, _, _, _ = construct_forecast_matrices(
                last_p_obs,
                A_0,
                beta,
                p,
                n,
                h,
                H,
            )

            conditional_mean = b.flatten()

            # Pass a private copy of the full state to the innovation hook.
            # Convert its reduced-form innovations to the structural shocks
            # required by the companion form.
            if point_only:
                state = self.posterior_state_point.copy()
            else:
                extras = self.extras[i] if self.extras is not None else None
                state = PosteriorState(
                    beta=beta_draws[i, :], sigma=sigma_draws[i, :], extras=extras
                ).copy()
            innovations = self.model.sample_innovations(
                state, H, rng=self.rng, point_only=point_only
            )
            epsilon = np.linalg.solve(A_0.T, innovations.T).T.flatten()

            forecast_i = conditional_mean + epsilon.flatten().T @ bigM

            y_unconditional[i, T : T + H, :] = forecast_i.reshape(H, n)

            mean_H[i, :] = conditional_mean.reshape(H, n)[-1]
            variance = bigM.T @ bigM
            start = (H - 1) * n
            end = H * n
            variance_H[i, :, :] = variance[start:end, start:end]

        return y_unconditional, mean_H, variance_H

    # ------------------------------------------------------------------
    # Conditional forecast (Gibbs sampler)
    # ------------------------------------------------------------------

    def _conditional_forecast(
        self,
        constraint_mean: np.ndarray,
        constraint_variance: Optional[np.ndarray],
        constraint_shape: Optional[np.ndarray],
        H: int,
        N_draws: int,
        point_only: bool,
        progressbar: bool,
        N_burn: Optional[int] = None,
        constraint_sampler: Optional[Callable] = None,
        method: str = "andersson_et_al",
    ) -> np.ndarray:
        """
        Generate conditional forecasts using the Gibbs sampler of
        Antolín-Díaz et al. (2021).

        The method sends the constrained-forecast draw through
        ``self.model.sample_conditional_forecast(...)``, whose default
        implementation routes to
        :func:`~bvar.forecast.conditional.draw_constrained_forecasts` using
        the current iteration's ``beta``/``sigma``.

        The method threads the parameter-update step through
        :class:`~bvar.models.base.PosteriorState` rather than raw
        ``(beta, sigma)`` tuples: the chain starts from a copy of
        ``self.posterior_state_point`` and each iteration calls
        ``self.model.sample_posterior_state(...)``, carrying the full
        returned state (including any model-owned ``extras``) into the next
        iteration. The seed copy is fully isolated -- including nested
        mutable objects inside ``extras`` -- so this never mutates
        ``self.beta_point``/``self.sigma_point``/``self.posterior_state_point``
        (see :meth:`~bvar.models.base.PosteriorState.copy`). When
        With ``point_only``, the method generates only the single point-estimate forecast and
        ``sample_posterior_state`` is not called because the point-only path
        has no subsequent iteration.

        ``sample_conditional_forecast`` itself is also given a fresh
        ``state.copy()`` on every iteration, so a hook that mutates the
        state it receives in place cannot alter the ``state`` object carried
        forward into ``sample_posterior_state``.

        Raises
        ------
        ValueError
            If ``constraint_mean`` is empty, if ``N_draws`` is not a
            positive integer (when ``point_only`` is ``False``), if the
            effective (capped) draw count is not positive, or if
            ``N_burn`` is not an integer in ``[0, effective_N_draws)``
            (where ``effective_N_draws`` is ``N_draws`` after capping at
            ``self.N_draws``).

        Parameters
        ----------
        constraint_mean : np.ndarray
            Mean constraints for the forecast horizon.
        constraint_variance : Optional[np.ndarray]
            Variance constraints for the forecast horizon.
        constraint_shape : Optional[np.ndarray]
            Shape constraints for the forecast horizon.
        H : int
            Forecast horizon.
        N_draws : int
            Number of posterior draws.
        point_only : bool
            Whether to generate only the point forecast.
        progressbar : bool
            Whether to display a progress bar.
        N_burn : Optional[int]
            Number of burn-in draws.
        constraint_sampler : Optional[Callable]
            Custom constraint sampler.
        method : str
            Conditional-forecast algorithm.

        Returns
        -------
        np.ndarray
            Conditional forecast draws.
        """

        validate_constraint_inputs(
            constraint_mean,
            constraint_variance,
            constraint_shape,
            expected_shape=(H, self.n),
        )
        if point_only:
            N_draws = 1
            N_burn = 0
        else:
            if not isinstance(N_draws, (int, np.integer)) or isinstance(N_draws, bool):
                raise ValueError(f"N_draws must be an integer, got {N_draws!r}.")
            if N_draws <= 0:
                raise ValueError(f"N_draws must be positive, got {N_draws}.")
            # Cap to the available posterior sample *before* deriving the
            # default burn-in, so the default is a fraction of what is
            # actually available rather than of the (possibly much larger)
            # requested count.
            N_draws = min(self.N_draws, N_draws)
            if N_draws <= 0:
                raise ValueError(
                    "Effective (capped) N_draws must be positive, got "
                    f"{N_draws} after capping at self.N_draws={self.N_draws}."
                )
            if N_burn is None:
                N_burn = N_draws // 2
            else:
                if not isinstance(N_burn, (int, np.integer)) or isinstance(
                    N_burn, bool
                ):
                    raise ValueError(f"N_burn must be an integer, got {N_burn!r}.")
                if not (0 <= N_burn < N_draws):
                    raise ValueError(
                        "N_burn must be in [0, N_draws) using the effective "
                        f"(capped) draw count of {N_draws}, got {N_burn}."
                    )

        data = self.data.copy()

        p = self.n_lags
        T_full, n = data.shape

        last_p_obs = data[-p:, :]
        h = len(self.covid_indices)

        # Store full data (including first p lags) so that covid_indices
        # (which are relative to the original data) remain correct when
        # we call construct_Y_Z on augmented arrays.
        y_conditional_full = np.zeros((N_draws, T_full + H, n))
        y_conditional_full[:, :T_full, :] = data

        # Construct constraint matrices
        C, f, Sigma_f, shape_f = get_constraint(
            constraint_mean,
            constraint_variance,
            constraint_shape,
        )

        # Start the Gibbs chain from an isolated copy of the fitted point
        # state; nested ``extras`` values receive the same protection.
        state = self.posterior_state_point.copy()

        for i in tqdm(
            range(N_draws),
            desc="Conditional Forecast",
            unit="iteration",
            disable=not progressbar,
        ):
            # Give the constrained-forecast hook an isolated copy of the
            # complete state so it can use model-owned ``extras`` safely.
            forecast_i = self.model.sample_conditional_forecast(
                state.copy(),
                C=C,
                f=f,
                Sigma_f=Sigma_f,
                shape_f=shape_f,
                last_p_obs=last_p_obs,
                p=p,
                n=n,
                h=h,
                H=H,
                point_only=point_only,
                constraint_sampler=constraint_sampler,
                method=method,
                rng=self.rng,
            )

            y_conditional_full[i, T_full : T_full + H, :] = forecast_i.reshape(H, n)

            if point_only:
                # The point estimate needs no subsequent state update.
                continue

            # Posterior draws for beta and sigma
            # Pass full array (lags + effective sample + forecasts) so that
            # covid_indices (relative to the original data) are correct.
            Y, Z = construct_Y_Z(y_conditional_full[i, :, :], p, self.covid_indices)

            if self.soc_ or self.sur_:
                Y, Z, _ = stack_dummies(
                    Y,
                    Z,
                    p,
                    self.vars_in_levels,
                    self.model,
                    self.covid_indices,
                    soc=self.soc_,
                    sur=self.sur_,
                )

            # Carry the full state -- including any model-owned extras --
            # into the next iteration via sample_posterior_state, so models
            # that need more than beta/sigma can persist it between draws.
            state = self.model.sample_posterior_state(Y, Z, state, rng=self.rng)

        # Discard burn-in and strip the leading p lag rows to match
        # the unconditional forecast shape: (N_draws, T + H, n)
        y_conditional = y_conditional_full[N_burn:, p:, :]

        return y_conditional
