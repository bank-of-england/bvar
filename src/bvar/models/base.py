"""
Base class for BVAR estimation models.

Each model encapsulates the full estimation pipeline — prior specification,
data-matrix construction, dummy-observation stacking, and posterior sampling —
together with the prior hyperparameters that govern that pipeline.

``BVAR.sample()`` delegates to :meth:`SamplingModel.sample` and only
handles data validation and result storage.
"""

from __future__ import annotations

import copy as copy_module
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

# ======================================================================
# Hyperparameter helpers (formerly in priors.py)
# ======================================================================


def _validate_positive_finite(name: str, value: object) -> None:
    """Require a non-empty numeric value that is finite and strictly positive."""
    try:
        raw_value = np.asarray(value)
        values = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and strictly positive.") from exc
    if (
        raw_value.dtype.kind == "b"
        or values.size == 0
        or not np.isfinite(values).all()
        or np.any(values <= 0)
    ):
        raise ValueError(f"{name} must be finite and strictly positive.")


def _validate_prior_values(values: dict[str, object]) -> None:
    for name, value in values.items():
        _validate_positive_finite(name, value)


def gamma_coef(mode: float, sd: float) -> tuple[float, float]:
    """Compute Gamma shape *k* and scale *θ* from mode and std-dev.

    Parameters
    ----------
    mode : float
        Mode of the Gamma distribution.
    sd : float
        Standard deviation of the Gamma distribution.

    Returns
    -------
    k : float
        Shape parameter.
    theta : float
        Scale parameter.
    """
    _validate_positive_finite("mode", mode)
    _validate_positive_finite("sd", sd)
    k = (2 + mode**2 / sd**2 + np.sqrt((4 + mode**2 / sd**2) * mode**2 / sd**2)) / 2
    theta = np.sqrt(sd**2 / k)
    return k, theta


class PriorPars:
    """Container for BVAR prior hyperparameters.

    The constructor sets all fields to ``None``; :meth:`SamplingModel.set_priors` populates them.
    :meth:`SamplingModel.set_priors`.
    """

    def __init__(self):
        self.c1 = None
        self.c3 = None
        self.lambda_constant = None
        self.mu = None
        self.theta = None
        self.lambda_covid = None
        self.nu_0 = None
        self.S_0 = None
        # Gamma hyperprior parameters
        self.c1_mode = None
        self.c1_sd = None
        self.c3_mode = None
        self.c3_sd = None
        self.mu_mode = None
        self.mu_sd = None
        self.theta_mode = None
        self.theta_sd = None
        self.c1_k = None
        self.c1_theta = None
        self.c3_k = None
        self.c3_theta = None
        self.mu_k = None
        self.mu_theta = None
        self.theta_k = None
        self.theta_theta = None
        # Cross-variable shrinkage (IndependentNIW only); declared here so the
        # Fix the attribute set instead of creating attributes dynamically in subclasses.
        self.c2_mode = None
        self.c2_sd = None
        self.c2_k = None
        self.c2_theta = None


@dataclass
class PosteriorState:
    """Extensible carrier for a single posterior draw's state.

    ``beta`` and ``sigma`` are the required core VAR parameters shared by
    every model. ``extras`` is an optional, model-owned payload for any
    additional state a model needs to carry between draws (e.g. a sampled
    degrees of freedom parameter for a Student-t model). ``extras`` is left
    untyped so each model may choose its representation, such as an array,
    scalar, or mapping.

    ``Forecasting._conditional_forecast`` threads this dataclass through its
    Gibbs chain (seeded from a copy of ``BVAR.posterior_state_point``) via
    :meth:`SamplingModel.sample_posterior_state`, carrying ``extras``
    forward between iterations rather than discarding it. The seed copy is
    fully isolated from the fitted point-estimate state -- including nested mutable
    objects inside ``extras`` -- so the chain can never mutate
    ``BVAR.posterior_state_point`` (see :meth:`copy`).

    Attributes
    ----------
    beta : np.ndarray
        Flattened VAR coefficients.
    sigma : np.ndarray
        Flattened covariance matrix.
    extras : Optional[Any]
        Arbitrary model-owned auxiliary state, or ``None`` if the model has
        no additional state beyond ``beta``/``sigma``.
    """

    beta: np.ndarray
    sigma: np.ndarray
    extras: Optional[Any] = None

    def copy(self) -> PosteriorState:
        """Return an independent copy for use as the seed of a draw chain.

        The method copies ``beta``/``sigma`` with ``np.ndarray.copy`` so a
        chain iterating over the returned state cannot mutate arrays owned by
        the caller (e.g. ``BVAR.beta_point``/``sigma_point``). It deep-copies
        ``extras`` via ``copy.deepcopy`` to isolate nested mutable
        containers (e.g. a ``dict`` holding a ``list``) as well as the
        top-level object. A model that mutates its own ``extras`` payload
        inside ``sample_posterior_state`` therefore leaves the fitted
        point-estimate state unchanged.

        Raises
        ------
        TypeError
            If ``extras`` holds an object that cannot be deep-copied (e.g. a
            lock or other live resource). The method raises this error instead
            of aliasing the original object. A model with such a payload should override
            ``sample_posterior_state`` to manage that state's isolation
            itself instead of relying on this default ``copy()``.

        Returns
        -------
        PosteriorState
            An independent copy of this state.
        """
        if self.extras is None:
            extras = None
        else:
            try:
                extras = copy_module.deepcopy(self.extras)
            except Exception as exc:
                raise TypeError(
                    "PosteriorState.extras could not be deep-copied "
                    f"({type(self.extras).__name__}: {exc}). Deep-copying "
                    "extras must stay separate from nested mutable containers "
                    "in the fitted point-estimate state. If this payload cannot be "
                    "deep-copied, override sample_posterior_state on the "
                    "model to manage its isolation explicitly instead of "
                    "relying on PosteriorState.copy()."
                ) from exc
        return PosteriorState(
            beta=self.beta.copy(), sigma=self.sigma.copy(), extras=extras
        )


@dataclass
class SamplingResult:
    """Container for the output of a model's ``sample()`` method.

    Attributes
    ----------
    beta_draws : np.ndarray
        Posterior draws of VAR coefficients (vectorised), with shape
        ``(N_draws, nk)``.
    sigma_draws : np.ndarray
        Posterior draws of the covariance matrix (flattened), with shape
        ``(N_draws, n**2)``.
    beta_point : np.ndarray
        Posterior point estimate of VAR coefficients, with shape ``(nk,)``.
    sigma_point : np.ndarray
        Posterior point estimate of the covariance matrix, with shape
        ``(n**2,)``. For the conjugate model this is the posterior mean.
    extras_point : Optional[Any]
        Model-owned auxiliary state accompanying the point estimate, or
        ``None`` for models that carry no state beyond ``beta``/``sigma``
        (the default for all current models).
    extras_draws : Optional[list]
        Model-owned auxiliary state for each retained draw, one entry per
        row of ``beta_draws``/``sigma_draws``, or ``None`` for models that
        carry no per-draw state beyond ``beta``/``sigma`` (the default for
        all current models). Validated at construction (see
        :meth:`__post_init__`) to have exactly one entry per draw.
    """

    beta_draws: np.ndarray
    sigma_draws: np.ndarray
    beta_point: np.ndarray
    sigma_point: np.ndarray
    extras_point: Optional[Any] = None
    extras_draws: Optional[list] = None

    def __post_init__(self) -> None:
        """Validate draw counts are consistent across beta/sigma/extras.

        Catching a misalignment here -- rather than later, when forecasting
        indexes a row/entry by draw number -- turns a confusing
        ``IndexError``/broadcasting error deep in the forecast loop into an
        immediate, clear ``ValueError``.
        """
        n_beta_draws = self.beta_draws.shape[0]
        n_sigma_draws = self.sigma_draws.shape[0]
        if n_beta_draws != n_sigma_draws:
            raise ValueError(
                "beta_draws and sigma_draws must have the same number of "
                f"rows, got {n_beta_draws} and {n_sigma_draws}."
            )
        if self.extras_draws is None:
            return
        n_extras = len(self.extras_draws)
        if n_extras != n_beta_draws:
            raise ValueError(
                "extras_draws must have exactly one entry per beta_draws/"
                f"sigma_draws row ({n_beta_draws} draws), got {n_extras} "
                "entries."
            )

    @property
    def state_point(self) -> PosteriorState:
        """``PosteriorState`` view of the posterior point estimates.

        Built on demand rather than stored, so models that do not use
        ``extras_point`` incur no extra allocation.
        """
        return PosteriorState(
            beta=self.beta_point, sigma=self.sigma_point, extras=self.extras_point
        )


class SamplingModel(ABC):
    """Abstract base class for BVAR estimation models.

    Each subclass owns the **entire** estimation pipeline and the prior
    hyperparameters that drive it.  Common prior *flags* (minnesota, soc,
    sur, covid) and the shared hyperparameter infrastructure live here;
    subclasses add model-specific parameters.

    Parameters
    ----------
    minnesota : bool
        Whether to use the Minnesota prior.
    soc : bool
        Whether to use the sum-of-coefficients prior.
    sur : bool
        Whether to use the single-unit-root prior.
    covid : bool
        Whether to include COVID dummies.
    covid_dates : Optional[list]
        Two-element list ``[start, end]`` defining the COVID period. Each
        element may be anything pandas can interpret as a date (e.g. a
        string ``"2020-03-01"``, a ``pd.Timestamp`` or a ``pd.Period``).
        The method matches dates against the data index at the data's own
        frequency, so it supports monthly, quarterly, and other frequencies.
        Defaults to the 2020Q1-2021Q4 window if ``None``.

    Attributes
    ----------
    requires_burnin : bool
        ``True`` for MCMC samplers whose initial draws should be discarded;
        ``False`` for direct samplers.
    supports_ml : bool
        ``True`` if a closed-form marginal likelihood is available for
        hyperparameter optimisation (GLP 2015).
    supports_point_only : bool
        ``True`` if the model has a closed-form posterior point estimate and
        supports ``point_only=True`` sampling. Required for
        ``optimisation_method="cross_validation"``, which repeatedly refits
        the model with ``point_only=True`` inside the grid search.
    supports_gaussian_predictive : bool
        ``True`` when the model uses the reduced-form
        Gaussian used by :meth:`sample_innovations`,
        :meth:`sample_conditional_forecast` and :meth:`predictive_logpdf`.
        ``True`` for every current (Gaussian) model. A non-Gaussian model
        should set this to ``False`` and override each hook that needs a
        different predictive distribution; the default
        implementations raise ``NotImplementedError`` in that case.
    supports_girf : bool
        ``True`` if :meth:`~bvar.girf.GIRF.compute_girf` may be used with
        this model. ``True`` for every current (Gaussian) model. The
        current GIRF implementation is hard-coded to the Gaussian
        reduced-form predictive distribution, so ``compute_girf`` also
        requires ``supports_gaussian_predictive=True``; setting
        ``supports_girf=True`` on a model with
        ``supports_gaussian_predictive=False`` does not enable GIRFs. A
        non-Gaussian model should leave this ``False`` until a GIRF
        implementation compatible with its own predictive distribution
        exists.
    """

    requires_burnin: bool = False
    supports_ml: bool = False
    supports_point_only: bool = True
    supports_gaussian_predictive: bool = True
    supports_girf: bool = True

    def __init__(
        self,
        minnesota: bool = True,
        soc: bool = True,
        sur: bool = True,
        covid: bool = False,
        covid_dates: Optional[list] = None,
    ) -> None:
        import pandas as pd

        self.minnesota = minnesota
        self.soc = soc
        self.sur = sur
        self.covid = covid
        self.covid_dates = covid_dates
        self.pars = PriorPars()

        if self.covid and self.covid_dates is None:
            # Set the default dates here; ``check_covid`` converts them to
            # the data frequency when it computes the COVID indices.
            self.covid_dates = [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2021-12-31"),
            ]

        self.set_priors()
        self._compute_nb_hyper_pars()

    # ------------------------------------------------------------------
    # Prior hyperparameters
    # ------------------------------------------------------------------

    def set_priors(
        self,
        c1: float = 0.2,
        c3: float = 2.0,
        lambda_constant: float = 10.0,
        mu: float = 1.0,
        theta: float = 1.0,
        lambda_covid: float = 10_000.0,
        c1_mode: float = 0.2,
        c1_sd: float = 0.4,
        c3_mode: float = 2.0,
        c3_sd: float = 0.5,
        mu_mode: float = 1.0,
        mu_sd: float = 1.0,
        theta_mode: float = 1.0,
        theta_sd: float = 1.0,
    ) -> None:
        """Set prior hyperparameters.

        Subclasses may override to accept model-specific parameters
        (e.g. ``c2`` for :class:`IndependentNIW`).
        """
        _validate_prior_values(
            {
                "c1": c1,
                "c3": c3,
                "lambda_constant": lambda_constant,
                "mu": mu,
                "theta": theta,
                "lambda_covid": lambda_covid,
                "c1_mode": c1_mode,
                "c1_sd": c1_sd,
                "c3_mode": c3_mode,
                "c3_sd": c3_sd,
                "mu_mode": mu_mode,
                "mu_sd": mu_sd,
                "theta_mode": theta_mode,
                "theta_sd": theta_sd,
            }
        )
        c1_k, c1_theta = gamma_coef(c1_mode, c1_sd)
        c3_k, c3_theta = gamma_coef(c3_mode, c3_sd)
        mu_k, mu_theta = gamma_coef(mu_mode, mu_sd)
        theta_k, theta_theta = gamma_coef(theta_mode, theta_sd)

        self.pars.nu_0 = None
        self.pars.S_0 = None
        self.pars.c1 = c1
        self.pars.c3 = c3
        self.pars.lambda_constant = lambda_constant
        self.pars.mu = mu
        self.pars.theta = theta
        self.pars.lambda_covid = lambda_covid
        self.pars.c1_mode = c1_mode
        self.pars.c1_sd = c1_sd
        self.pars.c3_mode = c3_mode
        self.pars.c3_sd = c3_sd
        self.pars.mu_mode = mu_mode
        self.pars.mu_sd = mu_sd
        self.pars.theta_mode = theta_mode
        self.pars.theta_sd = theta_sd
        self.pars.c1_k, self.pars.c1_theta = c1_k, c1_theta
        self.pars.c3_k, self.pars.c3_theta = c3_k, c3_theta
        self.pars.mu_k, self.pars.mu_theta = mu_k, mu_theta
        self.pars.theta_k, self.pars.theta_theta = theta_k, theta_theta

    def _compute_nb_hyper_pars(self) -> None:
        """Compute the number of searchable hyperparameters."""
        self.nb_hyper_pars = 2  # c1, c3
        if self.soc:
            self.nb_hyper_pars += 1
        if self.sur:
            self.nb_hyper_pars += 1

    def fill_in_from_vector(self, pars: np.ndarray) -> None:
        """Update hyperparameters from an optimisation vector.

        The vector layout is ``[c1, c3, ...]`` followed by ``mu`` (if SOC)
        and ``theta`` (if SUR).  Subclasses may override to insert
        model-specific hyperparameters.
        """
        values = {"c1": pars[0], "c3": pars[1]}
        i = 2
        if self.soc:
            values["mu"] = pars[i]
            i += 1
        if self.sur:
            values["theta"] = pars[i]
        _validate_prior_values(values)
        self.pars.c1 = values["c1"]
        self.pars.c3 = values["c3"]
        i = 2
        if self.soc:
            self.pars.mu = values["mu"]
            i += 1
        if self.sur:
            self.pars.theta = values["theta"]

    def to_vector(self) -> np.ndarray:
        """Extract searchable hyperparameters as a vector."""
        pars = np.zeros(self.nb_hyper_pars)
        pars[0] = self.pars.c1
        pars[1] = self.pars.c3
        i = 2
        if self.soc:
            pars[i] = self.pars.mu
            i += 1
        if self.sur:
            pars[i] = self.pars.theta
        return pars

    def hyperparameter_grid(self) -> list[np.ndarray]:
        """Return default grid arrays for cross-validation.

        The order matches :meth:`fill_in_from_vector`.
        """
        grid: list[np.ndarray] = [
            np.linspace(0.001, 1.0, 20),  # c1
            np.linspace(1.0, 5.0, 5),  # c3
        ]
        if self.soc:
            grid.append(np.logspace(-2, 2, 11))  # mu
        if self.sur:
            grid.append(np.logspace(-2, 2, 11))  # theta
        return grid

    # ------------------------------------------------------------------
    # Estimation (abstract)
    # ------------------------------------------------------------------

    @abstractmethod
    def sample(
        self,
        data: np.ndarray,
        n_lags: int,
        covid_indices: np.ndarray,
        vars_in_levels: np.ndarray,
        N_draws: int,
        point_only: bool = False,
        progressbar: bool = True,
        soc: Optional[bool] = None,
        sur: Optional[bool] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> SamplingResult:
        """Run the full estimation pipeline and return posterior draws.

        During estimation, store the prior mean
        (``self.beta_0``) and prior precision (``self.V_A_inv``)
        so that :meth:`sample_posterior_state` can reuse them.

        Parameters
        ----------
        data : np.ndarray
            Raw data array with shape ``(T_total, n)`` (before lag trimming).
        n_lags : int
            Number of VAR lags.
        covid_indices : np.ndarray
            Observation indices for COVID dummy variables.
        vars_in_levels : np.ndarray
            Boolean array with shape ``(n,)`` indicating which variables are in
            levels.
        N_draws : int
            Total number of posterior draws to generate, including draws
            that the caller will discard as burn-in.
        point_only : bool
            If ``True``, compute only the posterior point estimate.
        progressbar : bool
            Whether to display a progress bar.
        soc : Optional[bool]
            Effective sum-of-coefficients flag for this fit. The flag records
            whether the code stacks SOC dummy observations. It defaults to
            ``self.soc``; callers should pass a fit-specific value, such as
            ``False`` when no variable uses levels, rather than mutate
            ``self.soc``.
        sur : Optional[bool]
            Effective single-unit-root flag for this fit, analogous to
            ``soc``. Defaults to ``self.sur`` if not given.
        rng : Optional[np.random.Generator]
            Generator for the posterior draws. Defaults to a fresh
            ``numpy.random.default_rng()`` if not given.

        Returns
        -------
        SamplingResult
            Dataclass containing posterior draws.
        """
        ...

    @abstractmethod
    def sample_posterior_state(
        self,
        Y: np.ndarray,
        Z: np.ndarray,
        current_state: PosteriorState,
        rng: Optional[np.random.Generator] = None,
    ) -> PosteriorState:
        """Return the next posterior state given the current one.

        This is the sole posterior-update extension point: called each
        iteration by ``Forecasting._conditional_forecast``, which threads a
        :class:`PosteriorState` through its Gibbs chain (seeded from a copy
        of ``BVAR.posterior_state_point``) to re-sample parameters after
        augmenting the data with constrained forecasts, so model-owned
        ``extras`` persist across iterations. The method uses
        ``self.beta_0``, ``self.V_A_inv``, and ``self.pars`` (S_0, nu_0)
        that :meth:`sample` stored.

        Parameters
        ----------
        Y : np.ndarray
            Dependent-variable matrix (with dummies already stacked
            if applicable).
        Z : np.ndarray
            Regressor matrix (with dummies already stacked if
            applicable).
        current_state : PosteriorState
            The previous draw's state. Direct samplers (e.g.
            :class:`~bvar.models.conjugate.NaturalConjugate`) sample
            independently and ignore this state.
            MCMC samplers (e.g.
            :class:`~bvar.models.independent_niw.IndependentNIW`) use
            ``current_state.sigma`` as the starting point for the next
            Gibbs sweep; the Independent-NIW Gibbs kernel does not need
            ``current_state.beta`` either — β is freshly sampled
            conditional on Σ within each sweep.
        rng : Optional[np.random.Generator]
            Generator for the draw. Defaults to a fresh
            ``numpy.random.default_rng()`` if not given.

        Returns
        -------
        PosteriorState
            The next draw's state. ``extras`` is ``None`` unless the model
            carries auxiliary state beyond ``beta``/``sigma``, in which
            case the override should update and forward its own payload.
        """
        ...

    def sample_innovations(
        self,
        state: PosteriorState,
        H: int,
        rng: Optional[np.random.Generator] = None,
        point_only: bool = False,
    ) -> np.ndarray:
        """Draw reduced-form forecast innovations for a complete posterior state.

        Dispatched once per retained draw by
        ``Forecasting.recursive_forecast`` and
        ``Forecasting._unconditional_forecast``, passing the draw's complete
        :class:`PosteriorState` (``beta``/``sigma``/``extras``) so models
        that carry auxiliary predictive state in ``extras`` (e.g. a sampled
        degrees of freedom parameter for a Student-*t* model) can shape the
        innovation distribution accordingly. The default implementation
        below only uses ``state.sigma`` and draws Gaussian reduced-form
        innovations ``~ N(0, Sigma)`` -- the existing behaviour for every
        current model, which ignores ``extras``. Models that need a
        different predictive distribution should override this method.

        Parameters
        ----------
        state : PosteriorState
            Complete posterior draw state for this draw (or
            ``BVAR.posterior_state_point`` when ``point_only``).
        H : int
            Forecast horizon (number of steps ahead).
        rng : Optional[np.random.Generator]
            Generator for the draw. Defaults to a fresh
            ``numpy.random.default_rng()`` if not given.
        point_only : bool
            If ``True``, return an all-zero array (point-estimate forecast, no
            random innovations). Default ``False``.

        Returns
        -------
        np.ndarray
            Reduced-form innovations, one row per forecast step.

        Raises
        ------
        NotImplementedError
            If ``supports_gaussian_predictive`` is ``False``: the default
            Gaussian implementation does not apply, and the subclass must
            override this method.
        """
        if not self.supports_gaussian_predictive:
            raise NotImplementedError(
                f"{type(self).__name__} does not support the default "
                "Gaussian predictive distribution (supports_gaussian_"
                "predictive=False); it must override sample_innovations "
                "to draw reduced-form innovations for its own predictive "
                "distribution."
            )
        n = int(round(np.sqrt(state.sigma.shape[0])))
        if point_only:
            return np.zeros((H, n))
        if rng is None:
            rng = np.random.default_rng()
        sigma = state.sigma.reshape(n, n)
        return rng.multivariate_normal(np.zeros(n), sigma, size=H)

    def sample_conditional_forecast(
        self,
        state: PosteriorState,
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
        constraint_sampler: Optional[Callable] = None,
        method: str = "andersson_et_al",
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Draw a constrained forecast for a complete posterior state.

        Dispatched once per Gibbs iteration by
        ``Forecasting._conditional_forecast``, passing the iteration's
        complete :class:`PosteriorState` (``beta``/``sigma``/``extras``) so
        models that carry auxiliary predictive state in ``extras`` can
        shape the constrained-forecast distribution accordingly. The
        default implementation below only uses ``state.beta``/
        ``state.sigma`` and routes to the existing
        :func:`~bvar.forecast.conditional.draw_constrained_forecasts`
        algorithm -- the existing behaviour for every current model, which
        ignores ``extras``. Models that need a different predictive
        distribution should override this method.

        Parameters
        ----------
        state : PosteriorState
            Complete posterior draw state for this Gibbs iteration.
        C : np.ndarray
            Constraint-selection matrix, as returned by
            :func:`~bvar.forecast.conditional.get_constraint`, with shape
            ``(nb_constraints, H*n)``.
        f : np.ndarray
            Constraint locations with shape ``(nb_constraints,)``.
        Sigma_f : np.ndarray
            Constraint scale matrix with shape
            ``(nb_constraints, nb_constraints)``.
        shape_f : np.ndarray
            Constraint shape parameters with shape ``(nb_constraints,)``.
        last_p_obs : np.ndarray
            Last *p* observations before the forecast period, with shape
            ``(p, n)``.
        p : int
            Number of lags.
        n : int
            Number of variables.
        h : int
            Number of COVID dummies.
        H : int
            Forecast horizon.
        point_only : bool
            If ``True``, return only the conditional mean (no sampling).
        constraint_sampler : Optional[Callable]
            Custom function to sample from the constraint distribution.
        method : str
            Algorithm: ``"andersson_et_al"``, ``"antolin_diaz_et_al"``, or
            ``"labonne_renzetti"``. Default ``"andersson_et_al"``.
        rng : Optional[np.random.Generator]
            Generator for the draw. Defaults to a fresh
            ``numpy.random.default_rng()`` if not given.

        Returns
        -------
        np.ndarray
            Forecast values (flattened), with shape ``(H*n,)``.

        Raises
        ------
        NotImplementedError
            If ``supports_gaussian_predictive`` is ``False``: the default
            Gaussian implementation does not apply, and the subclass must
            override this method.
        """
        if not self.supports_gaussian_predictive:
            raise NotImplementedError(
                f"{type(self).__name__} does not support the default "
                "Gaussian predictive distribution (supports_gaussian_"
                "predictive=False); it must override "
                "sample_conditional_forecast to draw constrained forecasts "
                "for its own predictive distribution."
            )
        # Import locally because the module-level import path would be circular:
        # bvar.forecast imports mixin.py, which imports this module.
        from ..forecast.conditional import draw_constrained_forecasts

        sigma = state.sigma.reshape(n, n)
        beta = state.beta.reshape(n, -1).T
        return draw_constrained_forecasts(
            sigma=sigma,
            beta=beta,
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
            rng=rng,
        )

    def predictive_logpdf(
        self,
        state: PosteriorState,
        observation: np.ndarray,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> float:
        """Evaluate the log predictive density of an out-of-sample observation.

        Dispatched by ``GridSearch.marginal_likelihood_H`` for each
            rolling-window out-of-sample point, passing the fitted model's
        ``posterior_state_point`` so models that carry auxiliary predictive
        state in ``extras`` can shape the predictive distribution
        accordingly. The default implementation below only uses *mean* and
        *covariance* and evaluates the Gaussian
        ``scipy.stats.multivariate_normal.logpdf`` -- the existing
        behaviour for every current model, which ignores ``state.extras``.
        Models that need a different predictive distribution should
        override this method.

        Parameters
        ----------
        state : PosteriorState
            Complete posterior state (typically ``BVAR.posterior_state_point``)
            accompanying *mean*/*covariance*.
        observation : np.ndarray
            The realised out-of-sample observation with shape
            ``(n_targets,)``.
        mean : np.ndarray
            Predictive mean at the forecast horizon with shape
            ``(n_targets,)``.
        covariance : np.ndarray
            Predictive covariance at the forecast horizon with shape
            ``(n_targets, n_targets)``.

        Returns
        -------
        float
            Log predictive density of *observation*.

        Raises
        ------
        NotImplementedError
            If ``supports_gaussian_predictive`` is ``False``: the default
            Gaussian implementation does not apply, and the subclass must
            override this method.
        """
        if not self.supports_gaussian_predictive:
            raise NotImplementedError(
                f"{type(self).__name__} does not support the default "
                "Gaussian predictive distribution (supports_gaussian_"
                "predictive=False); it must override predictive_logpdf to "
                "evaluate log densities under its own predictive "
                "distribution."
            )
        from scipy.stats import multivariate_normal

        return float(multivariate_normal.logpdf(observation, mean, covariance))
