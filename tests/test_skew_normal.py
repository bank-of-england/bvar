import numpy as np
import pytest

from bvar.skew_normal import draw_sun, sun_conditional_forecast


def test_draw_sun_raises_when_truncation_is_impossible():
    with pytest.raises(RuntimeError, match="max_attempts=3"):
        draw_sun(
            xi=np.zeros(1),
            Omega=np.eye(1),
            Delta=np.zeros((1, 1)),
            gamma=np.array([-np.inf]),
            Gamma=np.eye(1),
            max_attempts=3,
            rng=np.random.default_rng(0),
        )


@pytest.mark.parametrize("max_attempts", [0, -1, 1.5, True, "3"])
def test_draw_sun_rejects_invalid_max_attempts(max_attempts):
    with pytest.raises(ValueError, match="positive integer"):
        draw_sun(
            xi=np.zeros(1),
            Omega=np.eye(1),
            Delta=np.zeros((1, 1)),
            gamma=np.zeros(1),
            Gamma=np.eye(1),
            max_attempts=max_attempts,
        )


def test_skew_normal_sun_constrained():
    """Check that the we recover the actual constrained distributions from
    the posterior SUN distribution.

    """

    # ==========================================
    # 1. Setup Prior (Multivariate Normal)
    # ==========================================
    mu_0 = np.zeros((3, 1))

    # Correlated case
    Sigma_0 = np.array([[1, -0.1, 0.9], [-0.1, 1, 0.2], [0.9, 0.2, 1]])

    R = np.array([[1, 0, 0], [0, 1, 0]])

    # ==========================================
    # 2. Skew Normal Constraints
    # ==========================================
    pos1, scale1, shape1 = 1.1, 2.0, 10.0
    pos2, scale2, shape2 = -2.0, 2.5, 10.0

    xi = np.array([pos1, pos2])  # location
    omega_vec = np.array([scale1, scale2])  # standard deviations
    Omega = np.diag(omega_vec**2)  # Covariance matrix
    alpha = np.array([shape1, shape2])  # asymmetry

    # Calculate delta parameters
    delta = alpha / np.sqrt(1 + alpha**2)  # in (-1, 1)

    # Delta matrix (Coupling)
    # Note: element-wise multiplication for diagonal construction
    Delta = np.diag((omega_vec * delta).flatten())

    gamma = np.zeros((2, 1))
    Gamma = np.eye(2)

    n_draws = 10000

    rng = np.random.default_rng(0)

    # Draw samples from the Constraint distribution (just for validation)
    Y_constraint = draw_sun(xi, Omega, Delta, gamma, Gamma, size=n_draws, rng=rng)

    # ==========================================
    # 3. Posterior Computation (fusing Normal Prior + SUN Constraints)
    # ==========================================

    Y_full_constraint = sun_conditional_forecast(
        f=xi,
        sigma_f=omega_vec**2,
        shape_f=alpha,
        C=R,
        b=mu_0.flatten(),
        BigM=np.linalg.cholesky(Sigma_0),
        n_draws=n_draws,
        rng=rng,
    )

    # ==========================================
    # 4. Analysis
    # ==========================================

    # Reproject full posterior back to the constrained space (first 2 dimensions)
    RY_full = Y_full_constraint[:, :2]  # equivalent to (R * Y_full')'
    RY_con = Y_constraint

    # check that the mean, variance and skewness match
    mean_full = np.mean(RY_full, axis=0)
    mean_con = np.mean(RY_con, axis=0)
    np.testing.assert_allclose(mean_full, mean_con, atol=0.1)

    cov_full = np.cov(RY_full.T)
    cov_con = np.cov(RY_con.T)
    np.testing.assert_allclose(cov_full, cov_con, atol=0.1)

    from scipy.stats import skew

    skew_full = skew(RY_full, axis=0)
    skew_con = skew(RY_con, axis=0)
    np.testing.assert_allclose(skew_full, skew_con, atol=0.1)
