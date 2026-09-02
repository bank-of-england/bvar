"""
Bayesian Vector Autoregression (BVAR) Model Module
==================================================

This module provides the main BVAR class for estimating Bayesian Vector Autoregression
models with various prior specifications. The implementation supports Minnesota priors,
sum-of-coefficients priors, single-unit-root priors, and automatic hyperparameter
optimisation using marginal likelihood maximisation.

Classes
-------
BVAR : class
    Main class for Bayesian Vector Autoregression estimation and forecasting.

Notes
-----
The BVAR class integrates several components:
- Natural conjugate sampling for efficient posterior simulation (Chan, 2020)
- Minnesota-style priors with optional hyperparameter optimisation (GLP, 2015)
- Dummy observation implementation for sum-of-coefficients and single-unit-root priors
- Unconditional and conditional forecasting capabilities (Waggoner & Zha, 1999; Antolín-Díaz et al., 2021)
- Support for skewed constraint distributions in conditional forecasting

The model supports various data transformations and can handle:
- Variables in levels or differences
- COVID-19 dummy variables for structural break modelling
- Recursive forecasting for out-of-sample evaluation

References
----------
Chan, J. C. C. (2020). Large Bayesian vector autoregressions.
Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection for vector
    autoregressions. Review of Economics and Statistics, 97(2), 436-451.
Antolín-Díaz, J., Petrella, I., & Rubio-Ramírez, J. F. (2021). Structural scenario
    analysis with SVARs. Journal of Monetary Economics, 117, 798-815.
"""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral
from typing import Optional

import numpy as np
import pandas as pd

from .forecast import Forecasting
from .girf import GIRF
from .grid_search import GridSearch
from .models import PosteriorState, SamplingModel
from .models.common import ar1_mse
from .models.conjugate.marginal_likelihood import tune_priors
from .plots import PlotBVAR, PlotGIRF
from .utils import (
    check_covid,
    construct_X,
    get_dimensions,
)


class BVAR(Forecasting, GIRF, PlotBVAR, PlotGIRF, GridSearch):
    """
    Bayesian Vector Autoregression (BVAR) model class.

    Provides methods for estimating posterior distributions of VAR parameters,
    generating fitted values, and producing unconditional and conditional forecasts.

    The model is specified as:
        y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + ε_t
    where ε_t ~ N(0, Σ).

    Notes
    -----
    df_data : pd.DataFrame
        Original input data.
    data : np.ndarray
        Data as numpy array.
    n_lags : int
        Number of lags.
    n : int
        Number of variables.
    T : int
        Effective sample size (total observations minus lags).
    k : int
        Number of regressors per equation (1 + n*p + h).
    nk : int
        Total number of coefficients (n * k).
    model : SamplingModel
        Sampling model instance (e.g. ``NaturalConjugate`` or ``IndependentNIW``).
    vars_in_levels : np.ndarray
        Boolean array indicating which variables are in levels.
    covid_indices : list
        Indices of COVID-19 dummy observations.
    soc_ : bool or None
        Effective sum-of-coefficients flag for the last fit (``model.soc``
        combined with whether any variable is in levels). Computed per-fit
        rather than mutating ``self.model.soc``. Callers can reuse the same
        model instance with independent BVAR fits.
    sur_ : bool or None
        Effective single-unit-root flag for the last fit, analogous to
        ``soc_``.
    data_transformation : dict or None
        Dictionary describing the original transformations of the input data.
        Maps variable names/indices to input data states such as "levels",
        "logs" (or "log_levels"), "diff", and "log_diff".
        Set during sampling via the sample() method.
    beta : np.ndarray
        Posterior draws of VAR coefficients, shape (N_draws, n*k).
    sigma : np.ndarray
        Posterior draws of covariance matrix (flattened), shape (N_draws, n²).
    extras : list, optional
        Model-owned auxiliary state for each retained draw in ``beta``/
        ``sigma``, or ``None`` for models that carry no such state (the
        default for all current models).
    beta_point : np.ndarray
        Posterior point estimate of VAR coefficients.
    sigma_point : np.ndarray
        Posterior point estimate of the covariance matrix. For the conjugate
        model this is the posterior mean.
    posterior_state_point : PosteriorState
        Extensible state carrier wrapping ``*_point`` plus any model-owned
        ``extras`` (``None`` for current models).
    """

    # Declare attributes up front to reduce memory use and prevent typos.
    __slots__ = (
        "H",
        "N_draws",
        "T",
        "X",
        "beta",
        "covid_indices",
        "cv_test_obs",
        "data",
        "data_transformation",
        "df_data",
        "df_forecasts_conditional",
        "df_forecasts_unconditional",
        "extras",
        "fitted_values",
        "forecast_conditional",
        "forecast_unconditional",
        "freq",
        "irf_H",
        "irf_df",
        "irf_draws",
        "irf_response_type",
        "irf_shock_size",
        "irf_summary",
        "irf_var_names",
        "k",
        "last_obs_index",
        "mean_H",
        "model",
        "n",
        "n_lags",
        "nk",
        "optimisation_method",
        "point_only",
        "posterior_state_point",
        "rng",
        "sigma",
        "sigma_point",
        "soc_",
        "stationary",
        "sur_",
        "target_indices",
        "target_series",
        "variance_H",
        "vars_in_levels",
        "beta_point",
    )

    # Default hyperparameters
    _DEFAULT_N_DRAWS = 5000
    _VALID_OPTIMISATION_METHODS = {
        "ml",
        "cross_validation",
        "none",
    }

    def _commit_staged_state(self, staged: BVAR) -> None:
        """Commit a completed staged operation to this instance."""
        for name in self.__slots__:
            if hasattr(staged, name):
                setattr(self, name, getattr(staged, name))

    def __init__(
        self,
        n_lags: int,
        model: SamplingModel,
        stationary: bool,
        optimisation_method: str = "ml",
        random_state: Optional[int] = None,
    ) -> None:
        if n_lags < 1:
            raise ValueError("n_lags must be at least 1")
        if optimisation_method not in self._VALID_OPTIMISATION_METHODS:
            raise ValueError(
                f"optimisation_method must be one of {self._VALID_OPTIMISATION_METHODS}, "
                f"got '{optimisation_method}'"
            )

        # Give each BVAR a private model copy so callers can share or refit
        # one model object without transferring per-fit state between models.
        self.model = deepcopy(model)
        self.n_lags = n_lags
        self.optimisation_method = optimisation_method
        self.stationary = stationary

        # Private random generator for all stochastic draws (never touches the
        # global NumPy random state). Can be overridden per call by passing
        # ``random_state`` to ``optimise_hyperparameters`` or ``sample``.
        self.rng = np.random.default_rng(random_state)

        # Initialise attributes that later methods populate.
        self.df_data: Optional[pd.DataFrame] = None
        self.data: Optional[np.ndarray] = None
        self.vars_in_levels: Optional[np.ndarray] = None
        self.covid_indices: list[int] = []
        self.soc_: Optional[bool] = None
        self.sur_: Optional[bool] = None
        self.data_transformation: Optional[dict] = None
        self.extras: Optional[list] = None
        self.fitted_values: Optional[np.ndarray] = None
        self.forecast_unconditional: Optional[np.ndarray] = None
        self.forecast_conditional: Optional[np.ndarray] = None
        self.df_forecasts_unconditional: Optional[pd.DataFrame] = None
        self.df_forecasts_conditional: Optional[pd.DataFrame] = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_lags={self.n_lags}, "
            f"model={self.model.__class__.__name__}, "
            f"optimisation_method='{self.optimisation_method}')"
        )

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted (posterior samples drawn)."""
        return hasattr(self, "beta") and self.beta is not None

    @property
    def dimensions(self) -> tuple[int, int, int, int]:
        """Return (n_vars, n_regressors, n_total_coeffs, n_effective_obs)."""
        if not hasattr(self, "n"):
            raise RuntimeError("Model has not been fitted yet. Call sample() first.")
        return (self.n, self.k, self.nk, self.T)

    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate data structure and types.

        Parameters
        ----------
        data : pd.DataFrame
            Data to validate.

        Raises
        ------
        TypeError
            If data is not a DataFrame or has an unsupported index type.
        ValueError
            If data contains non-numeric columns, non-finite values, duplicate
            or unordered periods, or an irregular frequency.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input 'data' must be a pandas DataFrame.")

        if data.shape[1] < 2:
            raise ValueError("Input 'data' must have at least two columns (variables).")

        if not all(data.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            raise ValueError("All columns in 'data' must be numeric.")

        if not isinstance(data.index, (pd.PeriodIndex, pd.DatetimeIndex)):
            raise TypeError(
                "Data index must be a DatetimeIndex or PeriodIndex with a "
                "regular frequency."
            )

        if not data.index.is_unique:
            raise ValueError("Data index must contain unique time periods.")
        if not data.index.is_monotonic_increasing:
            raise ValueError("Data index must be increasing.")

        values = data.to_numpy()
        finite = np.isfinite(values)
        if not finite.all():
            row, column = np.argwhere(~finite)[0]
            raise ValueError(
                "Argument 'data' must contain only finite values."
                f" First invalid value found at index {data.index[row]}"
                f" (variable '{data.columns[column]}')."
            )

    def _process_index(self, data: pd.DataFrame) -> None:
        """Convert index to PeriodIndex if needed and extract COVID indices and frequency.

        Parameters
        ----------
        data : pd.DataFrame
            Time series data with the index to process.

        Raises
        ------
        ValueError
            If a frequency cannot be inferred or the index is not regular.
        """
        if isinstance(data.index, pd.DatetimeIndex):
            # Infer the frequency from the data instead of assuming quarterly.
            freq = data.index.freq or pd.infer_freq(data.index)
            if freq is None:
                raise ValueError(
                    "Could not infer a frequency from the DatetimeIndex. "
                    "Provide regularly spaced data or pass a PeriodIndex."
                )
            expected_index = pd.date_range(
                start=data.index[0], periods=len(data.index), freq=freq
            )
            if not data.index.equals(expected_index):
                raise ValueError(
                    "Data index must contain consecutive observations at a "
                    "regular frequency."
                )
            # pandas >= 2.2 emits period-incompatible offset aliases
            # (e.g. "QE-DEC", "ME"); pass the offset object so to_period
            # resolves the period frequency across pandas versions.
            period_freq = pd.tseries.frequencies.to_offset(freq)
            self.df_data.index = data.index.to_period(period_freq)
        else:
            expected_index = pd.period_range(
                start=data.index[0], periods=len(data.index), freq=data.index.freq
            )
            if not data.index.equals(expected_index):
                raise ValueError(
                    "Data index must contain consecutive observations at a "
                    "regular frequency."
                )
        self.covid_indices = check_covid(self.df_data, self.model.covid_dates)

        # Store the full period frequency (including any anchor, e.g. "Q-DEC",
        # "W-SUN") for later use in forecasting. Keeping the anchor ensures
        # forecast date ranges stay aligned with the data's own calendar.
        self.freq = self.df_data.index.freqstr

    def _validate_and_prepare_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Validate and prepare data for analysis.

        Parameters
        ----------
        data : pd.DataFrame
            Time series data with variables in columns.

        Returns
        -------
        np.ndarray
            Validated data as numpy array.

        Raises
        ------
        ValueError
            If data contains non-numeric columns, non-finite values, no
            post-lag observations, or if a regular frequency cannot be
            established for the index.
        """
        self._validate_data(data)
        if len(data) <= self.n_lags:
            raise ValueError(
                "Input 'data' must contain at least one post-lag observation."
            )
        self.df_data = data.copy()
        self._process_index(data)

        # Set variables in levels
        self.vars_in_levels = (
            np.zeros(data.shape[1]) if self.stationary else np.ones(data.shape[1])
        )

        # Compute fit-specific SOC/SUR flags. The dummy-observation priors
        # have no economic meaning when every variable uses differences.
        # Store the flags on BVAR rather than mutating self.model, so callers
        # can reuse one model with different data and stationarity settings.
        has_levels = bool(np.any(self.vars_in_levels))
        self.soc_ = bool(self.model.soc) and has_levels
        self.sur_ = bool(self.model.sur) and has_levels

        ar1_mse(self.df_data.to_numpy(), self.covid_indices)

        return self.df_data.to_numpy()

    def optimise_hyperparameters(
        self,
        data: pd.DataFrame,
        nb_restart: int = 0,
        initial_values: Optional[np.ndarray] = None,
        target_series: Optional[list[str]] = None,
        cv_options: Optional[dict] = None,
        add_priors: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Optimise prior hyperparameters using the specified method.

        This method updates ``self.model.pars`` in place based on the
        ``optimisation_method`` set during initialisation.

        Parameters
        ----------
        data : pd.DataFrame
            Time series data with variables in columns. If covid is True,
            the index should be a regularly-spaced pd.PeriodIndex or
            pd.DatetimeIndex (any frequency, not only quarterly).
        nb_restart : int
            Number of random restarts for the L-BFGS-B optimiser to avoid local minima.
            Default is 0 (single optimisation run).
        initial_values : Optional[np.ndarray]
            Initial hyperparameter values. If ``None``, the method starts from
            default values with small random perturbations.
        target_series : Optional[list[str]]
            Series names to target when scoring cross-validation error (used
            only when ``optimisation_method="cross_validation"``). If ``None``,
            the method averages predictive accuracy over every series.
        cv_options : Optional[dict]
            Settings for cross-validation methods (e.g., number of folds,
            window size).
        add_priors : bool
            Whether to add Gamma hyperpriors on the hyperparameters when
            computing the marginal likelihood. Default is True.
        random_state : Optional[int]
            Seed or generator controlling the random perturbations applied to
            the initial guess and to each multi-start restart during "ml"
            hyperparameter optimisation, and the stochastic draws used across
            grid points and rolling windows during "cross_validation"
            optimisation (unused by "none"). If given, overrides the
            generator set at construction for this call; otherwise the
            the method uses the instance generator. The global NumPy random
            state remains unchanged. Default is None.

        Returns
        -------
        None
            The method updates ``self.model.pars`` with the optimised
            hyperparameters.

        Raises
        ------
        Exception
            If a staged optimisation operation fails.

        Notes
        -----
        For "ml" method, optimises c1, c3, and optionally mu (if SOC prior)
        and theta (if SUR prior) by maximising the marginal likelihood.
        """
        staged = deepcopy(self)
        staged.rng = self.rng
        operation_rng = (
            staged.rng if random_state is None else np.random.default_rng(random_state)
        )
        rng_state = deepcopy(operation_rng.bit_generator.state)
        try:
            staged._optimise_hyperparameters_in_place(
                data,
                nb_restart=nb_restart,
                initial_values=initial_values,
                target_series=target_series,
                cv_options=cv_options,
                add_priors=add_priors,
                random_state=random_state,
            )
        except Exception:
            operation_rng.bit_generator.state = rng_state
            raise
        self._commit_staged_state(staged)

    def _optimise_hyperparameters_in_place(
        self,
        data: pd.DataFrame,
        nb_restart: int = 0,
        initial_values: Optional[np.ndarray] = None,
        target_series: Optional[list[str]] = None,
        cv_options: Optional[dict] = None,
        add_priors: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        """Run hyperparameter optimisation on this staging instance."""
        # Prepare data
        data_array = self._validate_and_prepare_data(data)
        # Override the instance generator only if a seed is supplied here.
        rng = self.rng if random_state is None else np.random.default_rng(random_state)

        # Store dimensions
        _, n, k, _, _ = get_dimensions(data_array, self.n_lags, self.covid_indices)
        self.target_series = self.df_data.columns

        # Prior hyperparameters for covariance matrix ====================================
        if self.model.pars.nu_0 is None:
            self.model.pars.nu_0 = n + 4

        if self.model.pars.S_0 is None:
            self.model.pars.S_0 = np.eye(n) * (self.model.pars.nu_0 - n - 1)
        # ================================================================================

        if self.optimisation_method == "ml":
            if not self.model.supports_ml:
                raise ValueError(
                    f"Marginal-likelihood optimisation is not available for "
                    f"{self.model.__class__.__name__}. Use 'cross_validation' "
                    f"or 'none' instead."
                )

            # Initialise S_0 with OLS residuals
            mse = ar1_mse(data_array, self.covid_indices)
            S_0 = np.diag(mse) * (self.model.pars.nu_0 - n - 1)
            self.model.pars.S_0 = S_0

            # Estimate prior parameters following GLP (2015)
            tune_priors(
                data_array,
                self.n_lags,
                self.covid_indices,
                self.vars_in_levels,
                self.model,
                nb_restart,
                initial_values,
                add_priors=add_priors,
                soc=self.soc_,
                sur=self.sur_,
                rng=rng,
            )

        elif self.optimisation_method in ["cross_validation"]:
            if not self.model.supports_point_only:
                raise ValueError(
                    f"optimisation_method='cross_validation' is not available for "
                    f"{self.model.__class__.__name__}: cross-validation requires "
                    f"refitting the model with point_only=True, but this model has "
                    f"no closed-form posterior point estimate. Use "
                    f"optimisation_method='none' "
                    f"and set hyperparameters manually instead."
                )

            # TODO: Why not initialising S_0 with OLS residuals as well?

            if target_series is None:
                target_series = self.df_data.columns

            target_indices = [
                self.df_data.columns.get_loc(col) for col in target_series
            ]

            self.grid_search(
                data=data_array,
                cv_options=cv_options,
                target_indices=target_indices,
                random_state=rng,
                progressbar=True,
            )

            self.target_series = target_series
        elif self.optimisation_method == "none":
            pass

    def sample(
        self,
        data: pd.DataFrame,
        N_draws: Optional[int] = None,
        N_burn: Optional[int] = None,
        point_only: bool = False,
        progressbar: bool = True,
        data_transformation: Optional[dict] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Draw samples from the posterior distribution of VAR parameters.

        ``self.model`` determines the sampler:

        - ``NaturalConjugate``: Direct sampling from the known
          Normal-Inverse-Wishart posterior (Chan, 2020). The sampler needs no
          burn-in.
        - ``IndependentNIW``: Gibbs sampler with independent
          Normal-Inverse-Wishart priors. The sampler discards burn-in draws.

        Parameters
        ----------
        data : pd.DataFrame
            Time series data with variables in columns.
        N_draws : Optional[int]
            Number of posterior draws to *retain* after burn-in. Default is 5000.
        N_burn : Optional[int]
            Number of initial burn-in draws to discard. Only relevant for MCMC
            samplers (e.g. ``"independent_niw"``). Ignored for direct samplers.
            Default is ``N_draws // 2`` for MCMC samplers, 0 for direct samplers.
        point_only : bool
            If True, only compute the posterior point estimate without drawing
            samples. Useful for fast point estimates. Default is False.
        progressbar : bool
            Whether to display a progress bar during sampling. Default is True.
        data_transformation : Optional[dict]
            Dictionary describing the original transformations of the input data.
            Maps variable names or indices to input data states. Supported
            values are "levels", "logs" (or "log_levels"), "diff", and
            "log_diff". These labels describe transformations applied before
            sampling; they do not transform the data themselves. Forecast
            output transformations such as "qoq" and "yoy" belong in the
            ``transformations`` argument of forecast(), not here.
            Example: {"GDP": "log_diff", "CPI": "levels"}.
            The method stores this information and uses it to transform output
            in ``forecast()``.
            Default is None.
        random_state : Optional[int]
            Seed or generator controlling the posterior draws (and any
            subsequent forecast simulation). If given, overrides the generator
            set at construction and reused in later ``forecast`` calls;
            otherwise the method uses the instance generator. The global
            NumPy random state remains unchanged. Default is None.

        Returns
        -------
        None
            The method stores results in these attributes:
            - self.beta : Posterior draws of coefficients, shape (N_draws, n*k)
            - self.sigma : Posterior draws of Σ (flattened), shape (N_draws, n²)
            - self.beta_point : Posterior point estimate of coefficients
            - self.sigma_point : Posterior point estimate of Σ
            - self.posterior_state_point : ``PosteriorState`` view of the point estimate

        Raises
        ------
        Exception
            If posterior sampling fails.
        ValueError
            If a draw or burn-in count is invalid.
        """
        model = self.model
        if N_draws is None:
            N_draws = self._DEFAULT_N_DRAWS
        elif (
            isinstance(N_draws, bool)
            or not isinstance(N_draws, Integral)
            or N_draws <= 0
        ):
            raise ValueError("N_draws must be a positive integer.")
        else:
            N_draws = int(N_draws)

        if N_burn is not None and (
            isinstance(N_burn, bool) or not isinstance(N_burn, Integral) or N_burn < 0
        ):
            raise ValueError("N_burn must be a non-negative integer.")
        if model.requires_burnin and N_burn is not None and N_burn >= N_draws:
            raise ValueError("N_burn must be less than N_draws.")

        staged = deepcopy(self)
        if random_state is not None:
            staged.rng = np.random.default_rng(random_state)
        else:
            staged.rng = self.rng
        rng_state = deepcopy(staged.rng.bit_generator.state)
        try:
            staged._sample_in_place(
                data,
                N_draws=N_draws,
                N_burn=N_burn,
                point_only=point_only,
                progressbar=progressbar,
                data_transformation=data_transformation,
            )
        except Exception:
            staged.rng.bit_generator.state = rng_state
            raise
        self._commit_staged_state(staged)

    def _sample_in_place(
        self,
        data: pd.DataFrame,
        N_draws: int,
        N_burn: Optional[int],
        point_only: bool,
        progressbar: bool,
        data_transformation: Optional[dict],
    ) -> None:
        """Run posterior sampling on this staging instance."""
        model = self.model

        # Burn-in: only for MCMC samplers
        if N_burn is None:
            N_burn = N_draws // 2 if model.requires_burnin else 0
        elif not model.requires_burnin:
            N_burn = 0

        N_total = N_draws + N_burn

        # Prepare data
        data_array = self._validate_and_prepare_data(data)

        n_lags = self.n_lags
        covid_indices = self.covid_indices

        # Store dimensions
        _, n, k, nk, _ = get_dimensions(data_array, n_lags, covid_indices)
        self.data = data_array
        self.n = n
        self.k = k
        self.nk = nk
        self.T = data_array.shape[0] - n_lags
        self.N_draws = N_draws
        self.data_transformation = data_transformation

        # Delegate to the sampling model
        result = model.sample(
            data=data_array,
            n_lags=n_lags,
            covid_indices=covid_indices,
            vars_in_levels=self.vars_in_levels,
            N_draws=N_total,
            point_only=point_only,
            progressbar=progressbar,
            soc=self.soc_,
            sur=self.sur_,
            rng=self.rng,
        )

        # Discard burn-in and store results
        self.beta = result.beta_draws[N_burn:]
        self.sigma = result.sigma_draws[N_burn:]
        self.extras = (
            None if result.extras_draws is None else result.extras_draws[N_burn:]
        )
        if model.requires_burnin:
            self.beta_point = self.beta.mean(axis=0)
            self.sigma_point = self.sigma.mean(axis=0)
        else:
            self.beta_point = result.beta_point
            self.sigma_point = result.sigma_point
        self.posterior_state_point = PosteriorState(
            beta=self.beta_point,
            sigma=self.sigma_point,
            extras=result.extras_point,
        )
        self.point_only = point_only

    def compute_fitted_values(self) -> None:
        """
        Calculate fitted values across all posterior draws.

        Raises
        ------
        RuntimeError
            If model has not been fitted yet.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before computing fitted values. Call sample() first."
            )

        self.X = construct_X(self.data, self.n_lags, self.covid_indices)
        self.fitted_values = np.zeros((self.N_draws, self.T, self.n))

        for i in range(self.N_draws):
            signal = self.X @ self.beta[i, :].reshape(-1, 1)
            self.fitted_values[i, :, :] = signal.reshape(self.T, self.n)
