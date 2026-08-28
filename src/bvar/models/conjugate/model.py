"""
Natural Conjugate Model for BVAR
=================================

Direct sampling from the Normal-Inverse-Wishart posterior.  Because the
prior is conjugate to the VAR likelihood, the posterior distributions have
closed forms, and the sampler needs no MCMC burn-in.

References
----------
Chan, J. C. C. (2020). Large Bayesian vector autoregressions.
Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection for
    vector autoregressions.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
from scipy.stats import invwishart
from tqdm import tqdm

from ...dummy_observations import stack_dummies
from ...utils import construct_Y_Z, get_dimensions
from ..base import PosteriorState, SamplingModel, SamplingResult
from ..common import ar1_mse
from .minnesota import prior_minnesota


class NaturalConjugate(SamplingModel):
    """BVAR with Natural-Conjugate (Normal-Inverse-Wishart) priors.

    The prior is ``vec(A) | Σ ~ N(β₀, Σ ⊗ V_A⁻¹)`` and
    ``Σ ~ IW(S₀, ν₀)``.  Posterior sampling follows Chan (2020).

    Attributes
    ----------
    requires_burnin : bool
        ``False`` — draws come directly from the known posterior.
    supports_ml : bool
        ``True`` — a closed-form marginal likelihood is available (GLP 2015).
    """

    requires_burnin: bool = False
    supports_ml: bool = True

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
        """Run the full conjugate estimation pipeline."""
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
            beta_0, V_A_inv = prior_minnesota(
                data, n_lags, covid_indices, vars_in_levels, prior_pars
            )
        else:
            beta_0 = np.zeros(nk)
            V_A_inv = np.eye(k) * 1e-10

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

        # Sample from the posterior
        beta_draws, sigma_draws, beta_point, sigma_point = self._sample_direct(
            Y,
            Z,
            beta_0,
            S_0,
            V_A_inv,
            nu_0,
            N_draws,
            point_only=point_only,
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
        """Return a single random posterior draw as a ``PosteriorState``.

        Draws one sample from the known Normal-Inverse-Wishart posterior
        using the stored prior (``self.beta_0``, ``self.V_A_inv``,
        ``self.pars.S_0``, ``self.pars.nu_0``) from the last call to
        :meth:`sample`. The method accepts ``current_state`` for interface
        compatibility but ignores it because the posterior comes directly
        from the closed-form distribution rather than an MCMC update.
        """
        beta_draws, sigma_draws, _, _ = self._sample_direct(
            Y,
            Z,
            self.beta_0,
            self.pars.S_0,
            self.V_A_inv,
            self.pars.nu_0,
            N_draws=1,
            point_only=False,
            progressbar=False,
            rng=rng,
        )
        return PosteriorState(beta=beta_draws[0], sigma=sigma_draws[0])

    # ------------------------------------------------------------------
    # Posterior sampler (static so forecast code can re-use it)
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_direct(
        Y: np.ndarray,
        Z: np.ndarray,
        beta_0: np.ndarray,
        S_0: np.ndarray,
        V_A_inv: np.ndarray,
        nu_0: float,
        N_draws: int,
        point_only: bool = False,
        progressbar: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Direct sampler for the Normal-Inverse-Wishart posterior.

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
            Prior coefficient precision matrix with shape ``(k, k)``.
        nu_0 : float
            Prior inverse-Wishart degrees of freedom.
        N_draws : int
            Number of posterior draws.
        point_only : bool
            Whether to compute only the posterior point estimate.
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
            Posterior point estimate of the coefficients with shape ``(nk,)``.
        sigma_point : np.ndarray
            Posterior mean of the covariance matrix with shape ``(n**2,)``.
        """
        if rng is None:
            rng = np.random.default_rng()

        T, n = Y.shape
        k = len(beta_0) // n
        A_0 = beta_0.reshape(n, k).T  # (k, n)

        beta_draws = np.empty((N_draws, n * k))
        Sigma_draws = np.empty((N_draws, n**2))

        # Posterior precision
        K_A = V_A_inv + Z.T @ Z
        C_K_A = np.linalg.cholesky(K_A)
        A_post = np.linalg.solve(
            C_K_A.T, np.linalg.solve(C_K_A, V_A_inv @ A_0 + Z.T @ Y)
        )

        # Posterior scale
        S_post = S_0 + A_0.T @ V_A_inv @ A_0 + Y.T @ Y - A_post.T @ K_A @ A_post

        # Chan's point estimate is the posterior mean of the covariance.
        point_Sigma = S_post / (nu_0 + T - n - 1)

        if point_only:
            beta_draws[:] = A_post.T.flatten()
            Sigma_draws[:] = point_Sigma.flatten()
        else:
            for i in tqdm(
                range(N_draws),
                desc="Direct Sampling",
                unit="iteration",
                disable=not progressbar,
            ):
                Sigma = invwishart.rvs(df=nu_0 + T, scale=S_post, random_state=rng)
                U = rng.normal(size=(k, n))
                C_Sigma = np.linalg.cholesky(Sigma)
                beta = A_post + np.linalg.solve(C_K_A.T, U) @ C_Sigma.T

                beta_draws[i, :] = beta.T.flatten()
                Sigma_draws[i, :] = Sigma.flatten()

        return (
            beta_draws,
            Sigma_draws,
            A_post.T.flatten(),
            point_Sigma.flatten(),
        )
