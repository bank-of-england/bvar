from pathlib import Path

import numpy as np
import pytest

from bvar.utils import construct_Y_Z

ADDING_MODELS = Path(__file__).parents[1] / "docs" / "guide" / "adding_models.md"


def _flat_prior_scaffold():
    content = ADDING_MODELS.read_text(encoding="utf-8")
    example = content[content.index("## Minimal Example") :]
    return example.split("```python\n", 1)[1].split("\n```", 1)[0]


def test_flat_prior_scaffold_executes_and_guards():
    scaffold = _flat_prior_scaffold()
    ns: dict = {}
    exec(
        compile(scaffold, "<adding_models.md:FlatPrior>", "exec"),
        ns,
    )
    FlatPrior = ns["FlatPrior"]

    FlatPrior(soc=False, sur=False)
    with pytest.raises(ValueError, match="does not support"):
        FlatPrior(soc=True)
    with pytest.raises(ValueError, match="does not support"):
        FlatPrior(sur=True)
    with pytest.raises(ValueError, match="does not support"):
        FlatPrior(minnesota=True)


def test_flat_prior_usage_runs_as_printed():
    namespace = {}
    exec(compile(_flat_prior_scaffold(), str(ADDING_MODELS), "exec"), namespace)
    content = ADDING_MODELS.read_text(encoding="utf-8")
    usage = content[content.index("Then use it like any other model:") :]
    code = usage.split("```python\n", 1)[1].split("\n```", 1)[0]
    exec(compile(code, str(ADDING_MODELS), "exec"), namespace)

    fitted = namespace["bvar"]
    assert fitted.beta.shape[0] == 2000
    assert np.isfinite(fitted.forecast_unconditional).all()

    constraint_mean = np.full((3, fitted.n), np.nan)
    constraint_mean[0, 0] = 0.5
    fitted.forecast(
        H=3,
        constraint_mean=constraint_mean,
        N_draws=4,
        N_burn=0,
        random_state=44,
        progressbar=False,
    )
    assert np.isfinite(fitted.forecast_conditional).all()
    np.testing.assert_allclose(fitted.forecast_conditional[:, -3, 0], 0.5)


@pytest.mark.parametrize("n_vars", [1, 2])
@pytest.mark.parametrize("point_only", [False, True])
def test_flat_prior_draws_and_posterior_updates(n_vars, point_only):
    namespace = {}
    exec(compile(_flat_prior_scaffold(), str(ADDING_MODELS), "exec"), namespace)
    model = namespace["FlatPrior"]()
    data = np.random.default_rng(42).normal(size=(40, n_vars))
    covid_indices = np.array([], dtype=int)
    arguments = dict(
        data=data,
        n_lags=2,
        covid_indices=covid_indices,
        vars_in_levels=np.zeros(n_vars, dtype=bool),
        N_draws=4,
        point_only=point_only,
        progressbar=False,
    )
    result = model.sample(**arguments, rng=np.random.default_rng(43))
    repeated = model.sample(**arguments, rng=np.random.default_rng(43))
    assert result.beta_draws.shape == (4, n_vars * (2 * n_vars + 1))
    assert result.sigma_draws.shape == (4, n_vars**2)
    np.testing.assert_array_equal(result.beta_draws, repeated.beta_draws)
    np.testing.assert_array_equal(result.sigma_draws, repeated.sigma_draws)
    assert np.isfinite(result.beta_draws).all()
    assert np.isfinite(result.sigma_draws).all()
    assert (np.linalg.eigvalsh(result.sigma_draws.reshape(4, n_vars, n_vars)) > 0).all()
    if point_only:
        np.testing.assert_array_equal(
            result.beta_draws,
            np.broadcast_to(result.beta_point, result.beta_draws.shape),
        )
        np.testing.assert_array_equal(
            result.sigma_draws,
            np.broadcast_to(result.sigma_point, result.sigma_draws.shape),
        )

    response, design = construct_Y_Z(data, 2, covid_indices)
    state = model.sample_posterior_state(
        response, design, result.state_point, rng=np.random.default_rng(43)
    )
    assert state.beta.shape == result.beta_point.shape
    assert state.sigma.shape == result.sigma_point.shape
    assert np.isfinite(state.beta).all()
    assert np.isfinite(state.sigma).all()
    if not point_only:
        np.testing.assert_array_equal(state.beta, result.beta_draws[0])
        np.testing.assert_array_equal(state.sigma, result.sigma_draws[0])
    updated = model.sample_posterior_state(
        response + 1.0, design, state, rng=np.random.default_rng(43)
    )
    assert not np.allclose(updated.beta, state.beta)
    with pytest.raises(ValueError, match="does not support"):
        model.sample(**arguments, soc=True)
    with pytest.raises(ValueError, match="does not support"):
        model.sample(**arguments, sur=True)
