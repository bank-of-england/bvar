import numpy as np
import pytest

import bvar as bv


def test_stationary_bvar_estimation():
    """Estimate a stationary BVAR on simulated data; posterior means recover the true coefficients and covariance."""

    # Generate synthetic data
    T = 1000  # Time periods
    n = 2  # Number of variables
    n_lags = 2  # Number of lags
    levels = False
    covid = False
    seed = 123
    data, true_b, true_sigma, _ = bv.simulate_var(
        T, n, n_lags, covid=covid, levels=levels, seed=seed
    )

    # set priors
    priors = bv.NaturalConjugate(
        minnesota=True,
        soc=False,
        sur=False,
        covid=covid,
    )

    # create BVAR model
    bvar = bv.BVAR(n_lags, priors, not levels, optimisation_method="ml")

    # estimation
    bvar.optimise_hyperparameters(data, nb_restart=5, random_state=seed)
    bvar.sample(data, N_draws=2000, random_state=seed)

    beta_diff = bvar.beta.mean(axis=0) - true_b
    sigma_diff = bvar.sigma.mean(axis=0) - true_sigma.flatten()

    assert np.mean(np.abs(beta_diff)) < 0.1, (
        "Beta coefficients differ significantly from true values"
    )
    assert np.mean(np.abs(sigma_diff)) < 0.1, (
        "Sigma coefficients differ significantly from true values"
    )


def test_stationary_bvar_estimation_no_optim():
    # Generate synthetic data
    T = 1000  # Time periods
    n = 2  # Number of variables
    n_lags = 2  # Number of lags
    levels = False
    covid = False
    seed = 123
    data, true_b, true_sigma, _ = bv.simulate_var(
        T, n, n_lags, covid=covid, levels=levels, seed=seed
    )

    # set priors
    priors = bv.NaturalConjugate(
        minnesota=True,
        soc=False,
        sur=False,
        covid=covid,
    )

    # create BVAR model
    bvar = bv.BVAR(n_lags, priors, not levels, optimisation_method="none")

    # estimation
    bvar.sample(data, N_draws=2000, random_state=seed)

    beta_diff = bvar.beta.mean(axis=0) - true_b
    sigma_diff = bvar.sigma.mean(axis=0) - true_sigma.flatten()

    assert np.mean(np.abs(beta_diff)) < 0.1, (
        "Beta coefficients differ significantly from true values"
    )
    assert np.mean(np.abs(sigma_diff)) < 0.1, (
        "Sigma coefficients differ significantly from true values"
    )


def test_nonstationary_bvar_estimation():
    # Generate synthetic data
    T = 100  # Time periods
    n = 2  # Number of variables
    n_lags = 2  # Number of lags
    levels = True
    covid = False
    seed = 123
    data, true_b, true_sigma, _ = bv.simulate_var(
        T, n, n_lags, covid=covid, levels=levels, seed=seed
    )

    # set priors
    priors = bv.NaturalConjugate(
        minnesota=True,
        soc=False,
        sur=False,
        covid=covid,
    )

    # create BVAR model
    bvar = bv.BVAR(n_lags, priors, not levels, optimisation_method="ml")

    # estimation
    bvar.optimise_hyperparameters(data, nb_restart=0, random_state=seed)
    bvar.sample(data, N_draws=1000, random_state=seed)

    beta_mean = bvar.beta.mean(axis=0)
    sigma_mean = bvar.sigma.mean(axis=0).reshape(n, n)
    assert np.isfinite(beta_mean).all()
    assert np.all(np.linalg.eigvalsh(sigma_mean) > 0)
    assert np.mean(np.abs(sigma_mean.flatten() - true_sigma.flatten())) < 0.5


def test_stationary_bvar_estimation_cv():
    """Estimate a stationary BVAR with cross-validated hyperparameters; posterior means recover the true coefficients and covariance."""

    # Generate synthetic data
    T = 1000  # Time periods
    n = 2  # Number of variables
    n_lags = 2  # Number of lags
    levels = False
    covid = False
    seed = 123
    data, true_b, true_sigma, _ = bv.simulate_var(
        T, n, n_lags, covid=covid, levels=levels, seed=seed
    )

    # set priors
    priors = bv.NaturalConjugate(
        minnesota=True,
        soc=False,
        sur=False,
        covid=covid,
    )

    # create BVAR model
    bvar = bv.BVAR(n_lags, priors, not levels, optimisation_method="cross_validation")

    # estimation
    cv_options = {
        "H": 1,
        "oos_test_window_size": 100,
    }

    bvar.optimise_hyperparameters(data, cv_options=cv_options, random_state=seed)
    bvar.sample(data, N_draws=2000, random_state=seed)

    beta_diff = bvar.beta.mean(axis=0) - true_b
    sigma_diff = bvar.sigma.mean(axis=0) - true_sigma.flatten()

    assert np.mean(np.abs(beta_diff)) < 0.1, (
        "Beta coefficients differ significantly from true values"
    )
    assert np.mean(np.abs(sigma_diff)) < 0.1, (
        "Sigma coefficients differ significantly from true values"
    )


def test_stationary_bvar_estimation_covid():
    """Estimate a stationary BVAR with COVID dummies; posterior means recover the true coefficients and covariance."""

    # Generate synthetic data
    T = 200  # Time periods
    n = 2  # Number of variables
    n_lags = 1  # Number of lags

    covid = True
    levels = False
    data, true_b, true_sigma, _ = bv.simulate_var(
        T, n, n_lags, covid=covid, levels=levels, seed=123
    )

    # set priors
    priors = bv.NaturalConjugate(
        minnesota=True,
        soc=True,
        sur=True,
        covid=covid,
    )

    # create BVAR model
    bvar = bv.BVAR(n_lags, priors, not levels, optimisation_method="ml")

    # estimation
    bvar.sample(data, N_draws=10000, random_state=123)

    beta_diff = bvar.beta.mean(axis=0) - true_b
    sigma_diff = bvar.sigma.mean(axis=0) - true_sigma.flatten()

    assert np.mean(np.abs(beta_diff)) < 1, (
        "Beta coefficients differ significantly from true values"
    )
    assert np.mean(np.abs(sigma_diff)) < 0.1, (
        "Sigma coefficients differ significantly from true values"
    )


@pytest.mark.parametrize("model_cls", [bv.NaturalConjugate, bv.IndependentNIW])
def test_model_estimation_canonicalises_raw_covid_indices(model_cls):
    """Both samplers handle invalid, duplicate, and pre-lag COVID rows."""
    T = 30
    n = 2
    n_lags = 2
    data, _, _, _ = bv.simulate_var(T, n, n_lags, seed=123)
    data_array = data.to_numpy()
    covid_indices = np.array([-1, 0, 1, 4, 4, T, T + 1])
    model = model_cls(minnesota=True, soc=True, sur=True, covid=True)

    result = model.sample(
        data_array,
        n_lags,
        covid_indices,
        np.ones(n, dtype=bool),
        N_draws=3,
        progressbar=False,
        rng=np.random.default_rng(123),
    )

    effective_observations = T - n_lags
    n_covid = 1
    n_regressors = 1 + n * n_lags + n_covid
    n_parameters = n * n_regressors

    assert result.beta_draws.shape == (3, n_parameters)
    assert result.sigma_draws.shape == (3, n**2)
    assert result.beta_point.shape == (n_parameters,)
    assert result.sigma_point.shape == (n**2,)
    assert effective_observations > 0


def test_conditional_forecast_with_covid():
    """
    Test that conditional forecasting works correctly with COVID dummies.

    Verifies that:
    1. COVID dummy indices are correctly aligned in the Gibbs sampler
       (Bug 3 fix: indices are relative to original data, not stripped array).
    2. sample_posterior_state returns a proper random draw, not just a point estimate
       (Bug 2 fix: ensures parameter uncertainty propagates in the Gibbs chain).
    3. The conditional forecast matches constraints on mean.
    """
    T = 200
    n = 2
    n_lags = 2
    H = 4

    covid = True
    levels = True
    seed = 42
    data, _, _, _ = bv.simulate_var(T, n, n_lags, covid=covid, levels=levels, seed=seed)

    priors = bv.NaturalConjugate(
        minnesota=True,
        soc=True,
        sur=True,
        covid=covid,
    )

    bvar = bv.BVAR(n_lags, priors, not levels, optimisation_method="ml")
    bvar.optimise_hyperparameters(data, random_state=seed)
    bvar.sample(data, N_draws=500, random_state=seed)

    # Get unconditional forecast to use as constraint target
    bvar.forecast(H=H, point_only=True)
    uncond = bvar.forecast_unconditional[0, bvar.T :, :]

    # Set constraints on first variable
    constraint_mean = np.full((H, n), np.nan)
    constraint_mean[:, 0] = uncond[:, 0]

    # Run conditional forecast Gibbs sampler (exercises sample_posterior_state + covid alignment)
    bvar.forecast(
        H=H,
        constraint_mean=constraint_mean,
        point_only=False,
        N_draws=500,
        random_state=seed,
    )

    # The constrained variable should match the target closely
    cond_mean = bvar.forecast_conditional[:, bvar.T :, 0].mean(axis=0)
    diff_constrained = np.abs(cond_mean - uncond[:, 0]).max()
    assert diff_constrained < 0.5, (
        f"Constrained variable deviates from target: max diff = {diff_constrained:.4f}"
    )

    # The unconstrained variable should not deviate wildly from unconditional
    cond_other = bvar.forecast_conditional[:, bvar.T :, 1].mean(axis=0)
    diff_other = np.abs(cond_other - uncond[:, 1]).mean()
    assert diff_other < 2.0, (
        f"Unconstrained variable differs too much: mean diff = {diff_other:.4f}"
    )

    # Verify that parameter uncertainty is propagated (variance > 0 across draws)
    cond_var = np.var(bvar.forecast_conditional[:, bvar.T :, 1], axis=0)
    assert np.all(cond_var > 0), (
        "Conditional forecast shows zero variance — sample_posterior_state may not be sampling"
    )


def test_minnesota_false_uses_flat_prior_conjugate():
    """minnesota=False should store a near-zero V_A_inv, not the Minnesota precision."""
    T, n, n_lags = 200, 2, 1
    data, _, _, _ = bv.simulate_var(T, n, n_lags, seed=0)

    bvar_mn = bv.BVAR(
        n_lags,
        bv.NaturalConjugate(minnesota=True, soc=False, sur=False),
        True,
        optimisation_method="none",
    )
    bvar_flat = bv.BVAR(
        n_lags,
        bv.NaturalConjugate(minnesota=False, soc=False, sur=False),
        True,
        optimisation_method="none",
    )

    bvar_mn.sample(data, N_draws=10, random_state=0)
    bvar_flat.sample(data, N_draws=10, random_state=0)

    # Minnesota prior has large diagonal precision; flat prior is near zero
    assert np.all(bvar_mn.model.V_A_inv.diagonal() > 1e-6)
    assert np.all(bvar_flat.model.V_A_inv.diagonal() <= 1e-9)


def test_minnesota_false_uses_flat_prior_independent_niw():
    """minnesota=False should store a near-zero V_A_inv for IndependentNIW."""
    T, n, n_lags = 200, 2, 1
    data, _, _, _ = bv.simulate_var(T, n, n_lags, seed=0)

    bvar_mn = bv.BVAR(
        n_lags,
        bv.IndependentNIW(minnesota=True, soc=False, sur=False),
        True,
        optimisation_method="none",
    )
    bvar_flat = bv.BVAR(
        n_lags,
        bv.IndependentNIW(minnesota=False, soc=False, sur=False),
        True,
        optimisation_method="none",
    )

    bvar_mn.sample(data, N_draws=50, random_state=0)
    bvar_flat.sample(data, N_draws=50, random_state=0)

    assert np.all(bvar_mn.model.V_A_inv.diagonal() > 1e-6)
    assert np.all(bvar_flat.model.V_A_inv.diagonal() <= 1e-9)
