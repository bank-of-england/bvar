"""
Independent Normal-Inverse-Wishart Model for BVAR
===================================================

Gibbs sampler where the prior on β is independent of Σ:

    β  ~ N(β₀, V_β)        (nk × nk prior precision)
    Σ  ~ IW(S₀, ν₀)

Because the prior is **not** conjugate to the VAR likelihood the
posterior does not have a closed form and we resort to MCMC.  The Gibbs
sampler alternates between the full-conditionals:

    β | Σ, Y  ~ N(β̃, Ṽ)
    Σ | β, Y  ~ IW(S̃, ν̃)

References
----------
Kadiyala, K. R. & Karlsson, S. (1997). Numerical methods for estimation
    and inference in Bayesian VAR-models.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
from scipy.stats import invwishart
from tqdm import tqdm

from ...dummy_observations import stack_dummies
from ...utils import construct_Y_Z, get_dimensions
from ..base import (
    PosteriorState,
    SamplingModel,
    SamplingResult,
    _validate_positive_finite,
    _validate_prior_values,
    gamma_coef,
)
from ..common import ar1_mse
from .minnesota import prior_minnesota_independent


class IndependentNIW(SamplingModel):
    """BVAR with independent Normal-Inverse-Wishart priors.

    The prior is ``β ~ N(β₀, V_β)`` independently of ``Σ ~ IW(S₀, ν₀)``,
    where ``V_β`` is a full ``(nk, nk)`` precision matrix encoding
    cross-variable scaling ``σ_i / σ_j``.

    Parameters
    ----------
    c2 : float
        Cross-variable shrinkage (Litterman 1986 uses 0.5).
    minnesota : bool
        Whether to use the Minnesota prior.
    soc : bool
        Whether to use the sum-of-coefficients prior.
    sur : bool
        Whether to use the single-unit-root prior.
    covid : bool
        Whether to include COVID dummy observations.
    covid_dates : Optional[list]
        Start and end dates for the COVID period.

    Attributes
    ----------
    requires_burnin : bool
        ``True`` — Gibbs draws require burn-in.
    supports_ml : bool
        ``False`` — no closed-form marginal likelihood.
    supports_point_only : bool
        ``False`` — the posterior has no closed-form point estimate, so
        ``optimisation_method="cross_validation"`` is not supported. Use
        ``"none"`` and set hyperparameters manually.
    """

    requires_burnin: bool = True
    supports_ml: bool = False
    supports_point_only: bool = False

    def __init__(
        self,
        c2: float = 0.5,
        minnesota: bool = True,
        soc: bool = True,
        sur: bool = True,
        covid: bool = False,
        covid_dates: Optional[list] = None,
    ) -> None:
        self.c2 = c2
        super().__init__(
            minnesota=minnesota,
            soc=soc,
            sur=sur,
            covid=covid,
            covid_dates=covid_dates,
        )

    # ------------------------------------------------------------------
    # Hyperparameter interface (adds c2)
    # ------------------------------------------------------------------

    def set_priors(self, *, c2: float | None = None, **kwargs) -> None:
        """Set prior hyperparameters including cross-variable shrinkage *c2*.

        Also computes Gamma hyperprior parameters for c2.
        """
        c2_value = self.c2 if c2 is None else c2
        _validate_positive_finite("c2", c2_value)
        super().set_priors(**kwargs)
        # Gamma hyperprior for c2
        c2_k, c2_theta = gamma_coef(c2_value, 0.5)
        self.c2 = c2_value
        self.pars.c2_mode = c2_value
        self.pars.c2_sd = 0.5
        self.pars.c2_k, self.pars.c2_theta = c2_k, c2_theta

    def _compute_nb_hyper_pars(self) -> None:
        """c2 sits between c3 and (mu, theta) in the vector."""
        super()._compute_nb_hyper_pars()
        self.nb_hyper_pars += 1  # c2

    def fill_in_from_vector(self, pars: np.ndarray) -> None:
        """Vector layout: [c1, c3, c2, mu?, theta?]."""
        values = {"c1": pars[0], "c3": pars[1], "c2": pars[2]}
        i = 3
        if self.soc:
            values["mu"] = pars[i]
            i += 1
        if self.sur:
            values["theta"] = pars[i]
        _validate_prior_values(values)
        self.pars.c1 = values["c1"]
        self.pars.c3 = values["c3"]
        self.c2 = values["c2"]
        i = 3
        if self.soc:
            self.pars.mu = values["mu"]
            i += 1
        if self.sur:
            self.pars.theta = values["theta"]

    def to_vector(self) -> np.ndarray:
        pars = np.zeros(self.nb_hyper_pars)
        pars[0] = self.pars.c1
        pars[1] = self.pars.c3
        pars[2] = self.c2
        i = 3
        if self.soc:
            pars[i] = self.pars.mu
            i += 1
        if self.sur:
            pars[i] = self.pars.theta
        return pars

    def hyperparameter_grid(self) -> list[np.ndarray]:
        grid: list[np.ndarray] = [
            np.linspace(0.001, 1.0, 20),  # c1
            np.linspace(1.0, 5.0, 5),  # c3
            np.linspace(0.01, 1.0, 10),  # c2
        ]
        if self.soc:
            grid.append(np.logspace(-2, 2, 11))  # mu
        if self.sur:
            grid.append(np.logspace(-2, 2, 11))  # theta
        return grid

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def _compute_S0_nu0(
        self, data: np.ndarray, n: int, covid_indices: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Initialise the IW scale matrix from AR(1) OLS residuals."""
        nu_0 = n + 4 if self.pars.nu_0 is None else self.pars.nu_0

        mse = ar1_mse(data, covid_indices)
        S_0 = np.diag(mse) * (nu_0 - n - 1)
        return S_0, nu_0

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
        """Run the full independent-NIW Gibbs estimation pipeline.

        Parameters
        ----------
        data : np.ndarray
            Input data array.
        n_lags : int
            Number of VAR lags.
        covid_indices : np.ndarray
            Indices for COVID dummy observations.
        vars_in_levels : np.ndarray
            Indicators for variables in levels.
        N_draws : int
            Number of posterior draws.
        point_only : bool
            Whether to request a point estimate.
        progressbar : bool
            Whether to display a progress bar.
        soc : Optional[bool]
            Effective sum-of-coefficients flag.
        sur : Optional[bool]
            Effective single-unit-root flag.
        rng : Optional[np.random.Generator]
            Random number generator.

        Returns
        -------
        SamplingResult
            Posterior draws and point estimates.

        Raises
        ------
        ValueError
            If *point_only* is ``True`` (no closed-form posterior point estimate).
        """
        if point_only:
            raise ValueError(
                "point_only is not supported for IndependentNIW because "
                "the posterior has no closed-form point estimate.  Run with "
                "point_only=False instead."
            )

        soc = self.soc if soc is None else soc
        sur = self.sur if sur is None else sur
        if rng is None:
            rng = np.random.default_rng()

        _, n, k, nk, _ = get_dimensions(data, n_lags, covid_indices)

        # Prior scale matrix
        S_0, nu_0 = self._compute_S0_nu0(data, n, covid_indices)

        # Prior mean and precision
        if self.minnesota:
            prior_pars = deepcopy(self.pars)
            prior_pars.nu_0 = nu_0
            prior_pars.S_0 = S_0
            beta_0, V_A_inv = prior_minnesota_independent(
                data,
                n_lags,
                covid_indices,
                vars_in_levels,
                prior_pars,
                self.c2,
            )
        else:
            beta_0 = np.zeros(nk)
            V_A_inv = np.eye(nk) * 1e-10

        # Store prior info for sample_posterior_state
        self.beta_0 = beta_0
        self.V_A_inv = V_A_inv

        # Data matrices
        Y, Z = construct_Y_Z(data, n_lags, covid_indices)

        # Dummy observations (SOC / SUR)
        if soc or sur:
            Y, Z, _ = stack_dummies(
                Y, Z, n_lags, vars_in_levels, self, covid_indices, soc=soc, sur=sur
            )

        # Gibbs sampler
        beta_draws, sigma_draws, beta_point, sigma_point = self._sample_gibbs(
            Y,
            Z,
            beta_0,
            S_0,
            V_A_inv,
            nu_0,
            N_draws,
            progressbar=progressbar,
            rng=rng,
        )

        self.pars.nu_0 = nu_0
        self.pars.S_0 = S_0
        self.beta_0 = beta_0
        self.V_A_inv = V_A_inv

        return SamplingResult(
            beta_draws=beta_draws,
            sigma_draws=sigma_draws,
            beta_point=beta_point,
            sigma_point=sigma_point,
        )

    def sample_posterior_state(
        self,
        Y: np.ndarray,
        Z: np.ndarray,
        current_state: PosteriorState,
        rng: Optional[np.random.Generator] = None,
    ) -> PosteriorState:
        """Return the next Gibbs-sampled posterior state.

        Performs one full Gibbs sweep (β|Σ then Σ|β) using the stored prior
        from the last call to :meth:`sample`. ``current_state.sigma`` seeds
        the sweep's covariance. The Gibbs kernel accepts
        ``current_state.beta`` for interface compatibility but ignores it;
        it samples β conditional on Σ within this sweep rather than carrying
        β forward.
        """
        if rng is None:
            rng = np.random.default_rng()

        n = Y.shape[1]
        Sigma = current_state.sigma.reshape(n, n)

        beta, Sigma = self._gibbs_step(
            Y,
            Z,
            self.beta_0,
            self.pars.S_0,
            self.V_A_inv,
            self.pars.nu_0,
            Sigma,
            rng,
        )

        return PosteriorState(beta=beta, sigma=Sigma.flatten())

    # ------------------------------------------------------------------
    # Gibbs sampler
    # ------------------------------------------------------------------

    @staticmethod
    def _gibbs_step(
        Y: np.ndarray,
        Z: np.ndarray,
        beta_0: np.ndarray,
        S_0: np.ndarray,
        V_A_inv: np.ndarray,
        nu_0: float,
        Sigma: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One Gibbs sweep (β | Σ then Σ | β) starting from *Sigma*.

        * β | Σ, Y  ~  N(β̃, Ṽ)
              Ṽ⁻¹ = V_A⁻¹ + Σ⁻¹ ⊗ Z'Z
              β̃   = Ṽ (V_A⁻¹ β₀ + vec(Z'Y Σ⁻¹))

        * Σ | β, Y  ~  IW(S₀ + E'E,  ν₀ + T)

        Parameters
        ----------
        Y : np.ndarray
            Dependent-variable matrix with shape ``(T, n)``.
        Z : np.ndarray
            Regressor matrix with shape ``(T, k)``.
        beta_0 : np.ndarray
            Prior mean vector with shape ``(nk,)``.
        S_0 : np.ndarray
            Prior inverse-Wishart scale matrix with shape ``(n, n)``.
        V_A_inv : np.ndarray
            Prior coefficient precision matrix with shape ``(nk, nk)``.
        nu_0 : float
            Prior inverse-Wishart degrees of freedom.
        Sigma : np.ndarray
            Current covariance matrix with shape ``(n, n)``.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        beta : np.ndarray
            Sampled coefficient vector with shape ``(nk,)``.
        Sigma : np.ndarray
            Sampled covariance matrix with shape ``(n, n)``, not flattened.
        """
        T, n = Y.shape
        k = len(beta_0) // n
        nk = n * k

        ZtZ = Z.T @ Z
        ZtY = Z.T @ Y

        # --- β | Σ, Y ---
        Sigma_inv = np.linalg.inv(Sigma)
        V_post_inv = V_A_inv + np.kron(Sigma_inv, ZtZ)

        prior_info = V_A_inv @ beta_0
        data_term = (ZtY @ Sigma_inv).T.flatten()

        C_post = np.linalg.cholesky(V_post_inv)
        beta_mean = np.linalg.solve(
            C_post.T, np.linalg.solve(C_post, prior_info + data_term)
        )
        u = rng.normal(size=nk)
        beta = beta_mean + np.linalg.solve(C_post.T, u)

        # --- Σ | β, Y ---
        A = beta.reshape(n, k).T
        resid = Y - Z @ A
        S_post = S_0 + resid.T @ resid
        Sigma = invwishart.rvs(df=nu_0 + T, scale=S_post, random_state=rng)

        return beta, Sigma

    @staticmethod
    def _sample_gibbs(
        Y: np.ndarray,
        Z: np.ndarray,
        beta_0: np.ndarray,
        S_0: np.ndarray,
        V_A_inv: np.ndarray,
        nu_0: float,
        N_draws: int,
        progressbar: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Gibbs sampler for the independent NIW model.

        Starts a new chain from the prior covariance and repeatedly calls
        :meth:`_gibbs_step`, storing each draw.

        Parameters
        ----------
        Y : np.ndarray
            Dependent-variable matrix with shape ``(T, n)``.
        Z : np.ndarray
            Regressor matrix with shape ``(T, k)``.
        beta_0 : np.ndarray
            Prior mean vector with shape ``(nk,)``.
        S_0 : np.ndarray
            Prior inverse-Wishart scale matrix with shape ``(n, n)``.
        V_A_inv : np.ndarray
            Prior coefficient precision matrix with shape ``(nk, nk)``.
        nu_0 : float
            Prior inverse-Wishart degrees of freedom.
        N_draws : int
            Number of posterior draws.
        progressbar : bool
            Whether to display a progress bar.
        rng : Optional[np.random.Generator]
            Random number generator.

        Returns
        -------
        beta_draws : np.ndarray
            Posterior coefficient draws with shape ``(N_draws, nk)``.
        sigma_draws : np.ndarray
            Posterior covariance draws with shape ``(N_draws, n**2)``.
        beta_point : np.ndarray
            Posterior mean from the Gibbs run with shape ``(nk,)``.
        sigma_point : np.ndarray
            Posterior mean from the Gibbs run with shape ``(n**2,)``.
        """
        T, n = Y.shape
        k = len(beta_0) // n
        nk = n * k

        if rng is None:
            rng = np.random.default_rng()

        beta_draws = np.empty((N_draws, nk))
        sigma_draws = np.empty((N_draws, n**2))

        # Initialise Σ at prior mode
        Sigma = S_0 / (nu_0 + n + 1)

        for i in tqdm(
            range(N_draws),
            desc="Gibbs Sampling",
            unit="iteration",
            disable=not progressbar,
        ):
            beta, Sigma = IndependentNIW._gibbs_step(
                Y, Z, beta_0, S_0, V_A_inv, nu_0, Sigma, rng
            )
            beta_draws[i] = beta
            sigma_draws[i] = Sigma.flatten()

        # Store posterior means as point estimates
        beta_point = beta_draws.mean(axis=0)
        sigma_point = sigma_draws.mean(axis=0)

        return beta_draws, sigma_draws, beta_point, sigma_point
