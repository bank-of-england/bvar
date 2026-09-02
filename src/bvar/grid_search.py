"""
Cross-validation for BVARs
===================================

OOS with the predictive likelihood works well.
K-fold CV is still experimental.

References
----------

Bergmeir, C., Hyndman, R. J., & Koo, B. (2018). A note on the validity of cross-validation for evaluating autoregressive time series prediction. *Computational Statistics & Data Analysis*, 120, 70-83. [link](https://doi.org/10.1016/j.csda.2017.11.003)
"""

from __future__ import annotations

import copy
import itertools
from numbers import Integral
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .models.common import ar1_mse
from .utils import _validate_positive_integer


class GridSearch:
    """
    Cross-validation methods for BVAR class.

    This class provides hyperparameter optimisation via grid search on
    cross-validated errors. Currently only the out-of-sample predictive
    marginal likelihood method (``cv_method="predictive_ml"``) is available.

    The method requires a model with ``supports_point_only=True`` and a
    closed-form posterior point estimate. It fits each grid point with
    ``point_only=True``.
    """

    @staticmethod
    def _validate_target_indices(
        data: np.ndarray, target_indices: Optional[List[int]]
    ) -> list[int]:
        """Validate and normalise target variable indices."""
        if target_indices is None:
            return list(range(data.shape[1]))

        try:
            target_indices = list(target_indices)
        except TypeError as error:
            raise ValueError(
                "target_indices must contain at least one in-range integer index."
            ) from error

        if not target_indices or any(
            isinstance(index, bool)
            or not isinstance(index, Integral)
            or index < 0
            or index >= data.shape[1]
            for index in target_indices
        ):
            raise ValueError(
                "target_indices must contain at least one in-range integer index."
            )
        return [int(index) for index in target_indices]

    def _validate_rolling_initialisation(
        self, data: np.ndarray, rolling_oos_starting_id: int
    ) -> None:
        """Validate the first rolling training sample before evaluation."""
        training_end = rolling_oos_starting_id + self.n_lags
        ar1_mse(data[:training_end], self.covid_indices)

    def grid_search(
        self,
        data: np.ndarray,
        cv_method: str = "predictive_ml",
        cv_options: Optional[dict] = None,
        target_indices: Optional[List[int]] = None,
        random_state: Optional[int] = None,
        progressbar: bool = False,
    ) -> None:
        """
        Optmise hyperparameters with a grid search on cross-validated errors.

        Parameters
        ----------
        data : np.ndarray
            Input data array.
        cv_method : str
            Cross-validation method. Only supports "predictive_ml".
        cv_options : Optional[dict]
            Options for cross-validation. Required keys: ``H`` (forecast
            horizon) and ``oos_test_window_size``. Optional key ``grid``: a
            dict mapping hyperparameter names (``c1``, ``c3``, ``mu``,
            ``theta``) to the number of points to keep on that axis, evenly
            spaced across the model's default grid, to coarsen the search.
            Default is None.
        target_indices : Optional[List[int]]
            Indices of target variables for evaluation. If ``None``, the method
            uses every variable.
        random_state : Optional[int]
            Seed or generator controlling the stochastic draws used across
            the grid search. Resolved once into a single private
            ``numpy.random.Generator`` (falling back to ``self.rng`` if
            None) and reused -- by identity, never restarted -- for every
            grid point and rolling window evaluated by
            ``marginal_likelihood_H``. Default is None.
        progressbar : bool
            Whether to display a progress bar. Default is True.

        Returns
        -------
        None
            The method stores optimised hyperparameters in ``self.model.pars``.

        Raises
        ------
        Exception
            If a staged grid-search operation fails.
        ValueError
            If ``cv_method`` is not ``"predictive_ml"``, or if ``cv_options``
            is missing or does not contain the required ``"H"`` and
            ``"oos_test_window_size"`` keys.
        """
        if cv_method != "predictive_ml":
            raise ValueError(f"cv_method must be 'predictive_ml', got '{cv_method}'.")

        # Resolve once and reuse (by identity) across every grid point and
        # rolling window, rather than each marginal_likelihood_H call
        # restarting from a fresh copy of self.rng.
        staged = copy.deepcopy(self)
        staged.rng = self.rng
        operation_rng = (
            staged.rng if random_state is None else np.random.default_rng(random_state)
        )
        rng_state = copy.deepcopy(operation_rng.bit_generator.state)
        try:
            staged._grid_search_in_place(
                data,
                cv_method=cv_method,
                cv_options=cv_options,
                target_indices=target_indices,
                random_state=operation_rng,
                progressbar=progressbar,
            )
        except Exception:
            operation_rng.bit_generator.state = rng_state
            raise
        self._commit_staged_state(staged)

    def _grid_search_in_place(
        self,
        data: np.ndarray,
        cv_method: str = "predictive_ml",
        cv_options: Optional[dict] = None,
        target_indices: Optional[List[int]] = None,
        random_state: Optional[int] = None,
        progressbar: bool = False,
    ) -> None:
        """Run grid search on this staging instance."""
        rng = self.rng if random_state is None else np.random.default_rng(random_state)

        required_keys = {"H", "oos_test_window_size"}
        if cv_options is None or not required_keys.issubset(cv_options):
            raise ValueError(
                f"cv_options must be a dict with keys {sorted(required_keys)}, "
                f"got {cv_options!r}."
            )

        H = _validate_positive_integer(cv_options["H"], "H")
        oos_test_window_size = cv_options["oos_test_window_size"]
        if (
            isinstance(oos_test_window_size, bool)
            or not isinstance(oos_test_window_size, Integral)
            or oos_test_window_size <= 0
        ):
            raise ValueError("oos_test_window_size must be a positive integer.")

        starting_t = data.shape[0] - self.n_lags - H - oos_test_window_size
        if starting_t < self.n_lags:
            raise ValueError(
                "Data is too short for the requested rolling window and "
                "forecast horizon."
            )
        self._validate_rolling_initialisation(data, starting_t)

        optim_pars = None

        # generate grid from the model's hyperparameter space
        grid = self.model.hyperparameter_grid()

        # Optionally coarsen the grid for a faster run. ``cv_options["grid"]``
        # maps hyperparameter names to the number of points to keep, evenly
        # spaced across each default axis. Names follow ``fill_in_from_vector``
        # order: ``c1``, ``c3``, then ``mu`` (if ``soc``) and ``theta`` (if
        # ``sur``).
        grid_sizes = cv_options.get("grid")
        if grid_sizes is not None:
            names = ["c1", "c3"]
            if self.model.soc:
                names.append("mu")
            if self.model.sur:
                names.append("theta")
            unknown = set(grid_sizes) - set(names)
            if unknown:
                raise ValueError(
                    f"cv_options['grid'] has unknown keys {sorted(unknown)}; "
                    f"expected a subset of {names}."
                )
            for axis, name in enumerate(names):
                if name not in grid_sizes:
                    continue
                n_points = _validate_positive_integer(
                    grid_sizes[name], f"cv_options['grid']['{name}']"
                )
                axis_values = grid[axis]
                n_points = min(n_points, len(axis_values))
                keep = np.unique(
                    np.linspace(0, len(axis_values) - 1, n_points).round().astype(int)
                )
                grid[axis] = axis_values[keep]

        grid_size = int(np.prod([len(g) for g in grid]))

        target_indices = self._validate_target_indices(data, target_indices)

        # grid search
        min_error = float("inf")

        for pars in tqdm(
            itertools.product(*grid),
            total=grid_size,
            desc="Cross-Validation:",
            disable=not progressbar,
        ):
            self.model.fill_in_from_vector(np.array(pars))

            error = -self.marginal_likelihood_H(
                data,
                H,
                target_indices,
                starting_t,
                random_state=rng,
            )

            if error < min_error:
                min_error = error
                optim_pars = np.array(pars)

        self.model.fill_in_from_vector(optim_pars)
        self.target_indices = target_indices

    def marginal_likelihood_H(
        self,
        data: np.ndarray,
        H: int,
        target_indices: Optional[List[int]] = None,
        rolling_oos_starting_id: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> float:
        """
        Compute the marginal likelihood at forecast horizon H using rolling window.

        Parameters
        ----------
        data : np.ndarray
            Input data array.
        H : int
            Forecast horizon.
        target_indices : Optional[List[int]]
            Indices of target variables for evaluation. If ``None``, the method
            uses every variable.
        rolling_oos_starting_id : Optional[int]
            Starting index for the rolling window. If None, defaults to
            ``data.shape[0] - n_lags - H``, i.e. training on as much data as
            possible and evaluating a single out-of-sample point at the end
            of the sample -- the same convention ``grid_search()`` uses via
            ``starting_t`` when ``oos_test_window_size=1``. A literal ``0``
            would always leave zero in-sample observations for the first
            fit (``data[: 0 + n_lags]`` has exactly ``n_lags`` rows), which
            is degenerate.
        random_state : Optional[int]
            Seed or generator controlling the stochastic draws used across
            rolling windows. Resolved once into a single private
            ``numpy.random.Generator`` (falling back to ``self.rng`` if
            None) and reused -- by identity, never restarted -- for every
            rolling-window ``sample``/``forecast`` call. Default is None.

        Returns
        -------
        ml : float
            Sum of log marginal likelihoods across all rolling windows.

        Raises
        ------
        ValueError
            If the horizon, rolling-window index, or target indices are invalid.
        """
        H = _validate_positive_integer(H, "H")
        p = self.n_lags
        target_indices = self._validate_target_indices(data, target_indices)

        max_rolling_start = data.shape[0] - p - H
        if rolling_oos_starting_id is None:
            rolling_oos_starting_id = max_rolling_start
        elif isinstance(rolling_oos_starting_id, bool) or not isinstance(
            rolling_oos_starting_id, Integral
        ):
            raise ValueError(
                "rolling_oos_starting_id must be an integer in a valid range."
            )
        else:
            rolling_oos_starting_id = int(rolling_oos_starting_id)

        if not p <= rolling_oos_starting_id <= max_rolling_start:
            raise ValueError(
                "rolling_oos_starting_id must be an integer in a valid range."
            )
        self._validate_rolling_initialisation(data, rolling_oos_starting_id)

        # Resolve once and reuse (by identity) for every rolling window,
        # rather than the deepcopy below restarting the stream.
        rng = self.rng if random_state is None else np.random.default_rng(random_state)

        log_ml = []
        bvar = copy.deepcopy(self)
        bvar.rng = rng

        for t in range(rolling_oos_starting_id, data.shape[0] - p - H + 1):
            target = data[t + p + H - 1, target_indices]
            rolling_data = pd.DataFrame(
                data[: t + p, :],
                index=self.df_data.index[: t + p],
                columns=self.df_data.columns,
            )
            bvar.data = rolling_data.to_numpy()
            bvar.sample(rolling_data, progressbar=False, point_only=True)
            bvar.forecast(
                H=H,
                progressbar=False,
                point_only=True,
            )

            mean = bvar.mean_H[0, target_indices]
            # Two-axis fancy indexing to keep the full (n_targets, n_targets)
            # covariance submatrix -- paired indexing here would collapse it
            # to just the diagonal variances, dropping cross-target
            # covariance.
            variance = bvar.variance_H[0][np.ix_(target_indices, target_indices)]

            # Isolated copy: a mutating predictive_logpdf hook must never
            # corrupt the fitted beta_point/sigma_point/extras_point state.
            log_ml.append(
                bvar.model.predictive_logpdf(
                    bvar.posterior_state_point.copy(), target, mean, variance
                )
            )

        ml = float(np.sum(log_ml))
        return ml
