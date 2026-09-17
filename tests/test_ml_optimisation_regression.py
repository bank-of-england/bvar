"""End-to-end fixtures for the conjugate prior defaults.

The fixtures use the $n+2$ inverse-Wishart degrees-of-freedom default. Both
BFGS runs converged for each fixture with one BLAS thread.
"""

import numpy as np
import pandas as pd
import pytest

from bvar import BVAR
from bvar.models import NaturalConjugate
from bvar.models.conjugate import marginal_likelihood as ml
from bvar.utils import construct_Y_Z, simulate_var


BASELINES = {
    "stationary": {
        "objective": 240.0875382237443,
        "negative_logml": 240.07926037782374,
        "hyperparameters": [0.3376586073386309, 2.150366523012336],
        "beta": [
            2.7193304397910487,
            0.45219689016036324,
            0.20163823711492176,
            0.05058777623646452,
            -0.09067851073841497,
            0.9740357972784816,
            -0.05386796726608815,
            0.559841220415983,
            -0.054892287495657686,
            0.02542969112082788,
        ],
        "sigma": [
            1.1142492026322923,
            0.4234460718488248,
            0.42344607184882627,
            1.009515959258546,
        ],
        "forecast": [
            [5.158249102419066, 0.2707998701411135],
            [5.377277767271693, 0.5717960074577471],
            [5.502603285045468, 0.7282260135236011],
        ],
    },
    "levels_soc_sur": {
        "objective": 57.07631853254138,
        "negative_logml": 55.03430662817173,
        "hyperparameters": [
            0.06321475946397209,
            1.9589650314785536,
            1.0354583321850184,
            0.44010928874020794,
        ],
        "beta": [
            -0.1491712799189247,
            0.9751119996747741,
            0.05636487582618323,
            -0.009390513019313195,
            0.02877981745932111,
            0.14161445236163706,
            0.007409328417852664,
            0.9725316718290185,
            -0.004013253520983827,
            -0.0039842403044355575,
        ],
        "sigma": [
            0.07741150058096515,
            0.00927835022375026,
            0.009278350223749569,
            0.10635736010003878,
        ],
        "forecast": [
            [5.262367129937143, 4.79255922627982],
            [5.341352456173421, 4.801679386983076],
            [5.418404309641791, 4.81076164767085],
        ],
    },
}


@pytest.mark.parametrize("case", BASELINES)
def test_entire_ml_optimisation_matches_baseline(case, monkeypatch):
    """Run real BFGS including a restart; check its selected fit end to end.

    Tolerances were fixed against the stashed baseline before testing the edits.
    The penalised optimum is locally flat, so use 1e-7 for its objective but
    1e-4 for fitted hyperparameters and the unpenalised likelihood (whose
    gradient need not vanish at that optimum). Output tolerances are 1e-5 for
    coefficients/forecasts and 1e-6 for covariance, all absolute (rtol=0).
    Neither the iteration count nor the finite-difference path must be identical.
    """
    expected = BASELINES[case]
    levels = case == "levels_soc_sur"
    if levels:
        rng = np.random.default_rng(1234)
        data = pd.DataFrame(
            np.cumsum(rng.normal(0.02, 0.3, (80, 2)), axis=0),
            index=pd.period_range("1980Q1", periods=80, freq="Q"),
        )
    else:
        data, _, _, _ = simulate_var(
            80,
            2,
            2,
            covid=False,
            levels=False,
            seed=1234,
        )
    fit = BVAR(
        2,
        NaturalConjugate(minnesota=True, soc=levels, sur=levels, covid=False),
        stationary=not levels,
        optimisation_method="ml",
    )
    results = []
    minimise = ml.minimize

    def record_result(*args, **kwargs):
        # Observe the real SciPy optimiser; do not substitute its output.
        result = minimise(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(ml, "minimize", record_result)
    fit.optimise_hyperparameters(
        data,
        nb_restart=1,
        random_state=42,
        initial_values=np.array([0.2, 2.0] + ([1.0, 1.0] if levels else [])),
    )

    assert len(results) == 2
    for result in results:
        assert result.success, (
            f"BFGS failed: {result.message}; objective={result.fun}; "
            f"gradient={result.jac}"
        )
        assert np.linalg.norm(result.jac, ord=np.inf) <= 1e-5
    best = min(results, key=lambda result: result.fun)
    np.testing.assert_allclose(best.fun, expected["objective"], atol=1e-7, rtol=0)

    pars = fit.model.pars
    hyperparameters = np.array(
        [pars.c1, pars.c3] + ([pars.mu, pars.theta] if levels else []),
    )
    np.testing.assert_allclose(hyperparameters, ml.softplus(best.x), atol=0, rtol=0)
    np.testing.assert_allclose(
        hyperparameters,
        expected["hyperparameters"],
        atol=1e-4,
        rtol=0,
    )

    # Re-evaluate the committed fitted parameters, not the starting vector.
    array = data.to_numpy()
    Y, Z = construct_Y_Z(array, 2, fit.covid_indices)
    args = (
        np.log(np.expm1(hyperparameters)),
        array,
        2,
        fit.covid_indices,
        fit.vars_in_levels,
        0,
        fit.model,
        Y,
        Z,
    )
    objective = ml.objective_function(*args, True, fit.soc_, fit.sur_)
    negative_logml = ml.objective_function(*args, False, fit.soc_, fit.sur_)
    np.testing.assert_allclose(objective, expected["objective"], atol=1e-7, rtol=0)
    np.testing.assert_allclose(
        negative_logml,
        expected["negative_logml"],
        atol=1e-4,
        rtol=0,
    )

    fit.sample(data, N_draws=1, point_only=True, progressbar=False)
    fit.forecast(H=3, point_only=True)
    np.testing.assert_allclose(fit.beta_point, expected["beta"], atol=1e-5, rtol=0)
    np.testing.assert_allclose(fit.sigma_point, expected["sigma"], atol=1e-6, rtol=0)
    np.testing.assert_allclose(
        fit.forecast_unconditional[0, -3:],
        expected["forecast"],
        atol=1e-5,
        rtol=0,
    )
