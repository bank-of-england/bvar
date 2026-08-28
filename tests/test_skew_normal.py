import matplotlib.pyplot as plt
import numpy as np
import pytest

from bvar.plots import plot_density
from bvar.skew_normal import draw_skew_normal, draw_sun, sun_conditional_forecast


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


@pytest.mark.skip("Not currently used.")
def test_skew_normal_multivariate_draws():
    dim = 3
    cov = np.ones((dim, dim)) * 0
    cov = cov + np.eye(dim)
    alpha = np.ones(dim) * -1000
    # alpha[1] = 0  # Skewness only in first variable
    data = draw_skew_normal(cov, alpha, size=10000)
    labels = [f"Variable {i + 1}" for i in range(dim)]

    # Plot
    plot_density(data, labels=labels)
    plt.close()


@pytest.mark.skip("Not currently used.")
def test_draw_sun_normal_case():
    """
    Test that SUN reduces to multivariate normal when gamma -> -infinity.

    When the truncation threshold gamma is very large and negative,
    the selection event U > -gamma is always satisfied, so the SUN
    distribution should match the marginal normal distribution of V.
    """
    np.random.seed(42)

    # Parameters for a simple 2D normal case
    d = 2
    k = 1
    xi = np.array([1.0, 2.0])
    Omega = np.array([[1.0, 0.3], [0.3, 1.0]])
    Delta = np.zeros((d, k))  # No coupling
    gamma = np.array([1.0])  # Very large negative threshold
    Gamma = np.array([[10.0]])

    # Draw samples
    n_samples = 5000

    samples = draw_sun(xi, Omega, Delta, gamma, Gamma, size=n_samples)

    # Check shape
    assert samples.shape == (n_samples, d)

    # Check that mean is close to xi (location parameter)
    sample_mean = samples.mean(axis=0)
    np.testing.assert_allclose(sample_mean, xi, atol=0.1)

    # Check that covariance is close to Omega
    sample_cov = np.cov(samples.T)
    np.testing.assert_allclose(sample_cov, Omega, atol=0.1)

    # Draw samples from theoretical normal distribution for comparison
    samples_normal = np.random.multivariate_normal(xi, Omega, size=n_samples)

    # Plot density comparison using KDE
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, var_name in enumerate(["Variable 1", "Variable 2"]):
        # Compute KDEs
        kde_sun = gaussian_kde(samples[:, idx])
        kde_normal = gaussian_kde(samples_normal[:, idx])

        # Create evaluation grid
        x_min = min(samples[:, idx].min(), samples_normal[:, idx].min())
        x_max = max(samples[:, idx].max(), samples_normal[:, idx].max())
        x_grid = np.linspace(x_min, x_max, 200)

        # Plot densities
        axes[idx].plot(x_grid, kde_sun(x_grid), "b-", linewidth=2, label="SUN (γ → -∞)")
        axes[idx].plot(
            x_grid,
            kde_normal(x_grid),
            "orange",
            linewidth=2,
            label="Normal",
            linestyle="--",
        )
        axes[idx].axvline(
            samples[:, idx].mean(),
            color="blue",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
        )
        axes[idx].axvline(
            samples_normal[:, idx].mean(),
            color="orange",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
        )
        axes[idx].set_xlabel("Value")
        axes[idx].set_ylabel("Density")
        axes[idx].set_title(var_name)
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)

    plt.suptitle("SUN Distribution: Normal Case (γ → -∞) vs N(ξ, Ω)", fontsize=14)
    plt.tight_layout()
    # plt.show()


@pytest.mark.skip("Not currently used.")
def test_draw_sun_skewed_case():
    """
    Test that SUN reduces to multivariate normal when gamma -> -infinity.

    When the truncation threshold gamma is very large and negative,
    the selection event U > -gamma is always satisfied, so the SUN
    distribution should match the marginal normal distribution of V.
    """
    np.random.seed(42)

    # Parameters for a simple 2D normal case
    d = 2
    k = 1
    xi = np.array([1.0, 2.0])
    Omega = np.array([[1.0, 0.0], [0.0, 1.2]])
    Delta = np.zeros((d, k))
    # add skew to the first variable
    Delta[0, :] = -1.5
    gamma = np.array([-0.0])  # Very large negative threshold
    Gamma = np.array([[1.0]])

    # Draw samples
    n_samples = 5000
    samples = draw_sun(xi, Omega, Delta, gamma, Gamma, size=n_samples)

    # Check shape
    assert samples.shape == (n_samples, d)

    # Compute skewness of samples
    from scipy.stats import skew

    estimated_skewness = skew(samples, axis=0)
    print(f"Skewness of samples: {estimated_skewness}")

    # Draw samples from theoretical normal distribution for comparison
    samples_normal = np.random.multivariate_normal(xi, Omega, size=n_samples)

    # Plot density comparison using KDE
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, var_name in enumerate(["Variable 1", "Variable 2"]):
        # Compute KDEs
        kde_sun = gaussian_kde(samples[:, idx])
        kde_normal = gaussian_kde(samples_normal[:, idx])

        # Create evaluation grid
        x_min = min(samples[:, idx].min(), samples_normal[:, idx].min())
        x_max = max(samples[:, idx].max(), samples_normal[:, idx].max())
        x_grid = np.linspace(x_min, x_max, 200)

        # Plot densities
        axes[idx].plot(x_grid, kde_sun(x_grid), "b-", linewidth=2, label="SUN (γ → -∞)")
        axes[idx].plot(
            x_grid,
            kde_normal(x_grid),
            "orange",
            linewidth=2,
            label="Normal",
            linestyle="--",
        )
        axes[idx].axvline(
            samples[:, idx].mean(),
            color="blue",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
        )
        axes[idx].axvline(
            samples_normal[:, idx].mean(),
            color="orange",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
        )
        axes[idx].set_xlabel("Value")
        axes[idx].set_ylabel("Density")
        axes[idx].set_title(var_name)
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)

    plt.suptitle("SUN Distribution: Skewed Case (γ → -∞) vs N(ξ, Ω)", fontsize=14)
    plt.tight_layout()
    # plt.show()


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
    # 4. Analysis & Plotting
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

    # quick plot comparison
    # Variable 1 Comparison: Stack columns to treat them as two series on one plot
    # data_var1 = np.column_stack((RY_full[:, 0], RY_con[:, 0]))
    # plot_density(
    #     data_var1,
    #     labels=["Full Posterior", "Constraint Only"],
    #     title="Variable 1: Marginal Distribution",
    # )
    # plt.show()

    # Variable 2 Comparison
    # data_var2 = np.column_stack((RY_full[:, 1], RY_con[:, 1]))
    # plot_density(
    #     data_var2,
    #     labels=["Full Posterior", "Constraint Only"],
    #     title="Variable 2: Marginal Distribution",
    # )
    # plt.show()

    # plot the unconstrained series
    data_unconstrained = Y_full_constraint[:, 2].reshape(-1, 1)
    # plot_density(
    #     data_unconstrained,
    #     labels=["Unconstrained Variable"],
    #     title="Unconstrained Series",
    # )
    # plt.show()

    # compute skewness of data_unconstrained
    skew_unconstrained = skew(data_unconstrained, axis=0)
    print(f"Skewness of Unconstrained Variable: {skew_unconstrained}")
