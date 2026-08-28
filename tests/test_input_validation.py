from copy import deepcopy

import numpy as np
import pytest

from bvar.forecast.conditional import get_constraint
from bvar.models import IndependentNIW, NaturalConjugate


def test_public_forecast_rejects_wrong_constraint_shape_before_rng_mutation(bvar):
    rng_before = deepcopy(bvar.rng.bit_generator.state)

    with pytest.raises(ValueError, match="constraint_mean.*shape"):
        bvar.forecast(
            H=2,
            constraint_mean=np.zeros((1, bvar.n)),
            point_only=True,
            random_state=9876,
        )

    np.testing.assert_equal(bvar.rng.bit_generator.state, rng_before)


@pytest.mark.parametrize(
    "constraint_mean",
    [np.zeros(2), np.zeros((2, 2, 1))],
)
def test_get_constraint_rejects_non_matrix_mean(constraint_mean):
    with pytest.raises(ValueError, match="constraint_mean.*shape"):
        get_constraint(constraint_mean, None, None)


def test_get_constraint_rejects_mismatched_optional_shapes():
    mean = np.zeros((2, 2))

    with pytest.raises(ValueError, match="constraint_variance.*shape"):
        get_constraint(mean, np.zeros((2, 1)), None)
    with pytest.raises(ValueError, match="constraint_shape.*shape"):
        get_constraint(mean, None, np.zeros((1, 2)))


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("constraint_mean", np.array([[np.inf, np.nan]])),
        ("constraint_shape", np.array([[np.nan, 0.0]])),
        ("constraint_variance", np.array([[-1.0, np.nan]])),
        ("constraint_variance", np.array([[np.inf, np.nan]])),
        ("constraint_variance", np.array([[np.nan, np.nan]])),
    ],
)
def test_get_constraint_rejects_invalid_selected_values(name, values):
    mean = np.array([[1.0, np.nan]])
    kwargs = {"constraint_scale": None, "constraint_shape": None}
    if name == "constraint_mean":
        mean = values
    else:
        kwargs["constraint_scale" if name == "constraint_variance" else name] = values

    with pytest.raises(ValueError, match="finite|non-negative"):
        get_constraint(mean, **kwargs)


def test_get_constraint_allows_zero_variance_for_selected_constraint():
    C, f, variance, shape = get_constraint(
        np.array([[1.0, np.nan]]), np.array([[0.0, 123.0]]), None
    )

    np.testing.assert_array_equal(C, [[1.0, 0.0]])
    np.testing.assert_array_equal(f, [1.0])
    np.testing.assert_array_equal(variance, [[0.0]])
    np.testing.assert_array_equal(shape, [0.0])


@pytest.mark.parametrize("model_class", [NaturalConjugate, IndependentNIW])
@pytest.mark.parametrize(
    "parameter",
    [
        "c1",
        "c3",
        "mu",
        "theta",
        "c1_mode",
        "c1_sd",
        "c3_mode",
        "c3_sd",
        "mu_mode",
        "mu_sd",
        "theta_mode",
        "theta_sd",
    ],
)
@pytest.mark.parametrize("invalid", [0.0, -1.0, np.inf, np.nan])
def test_model_set_priors_rejects_invalid_shared_values_without_mutation(
    model_class, parameter, invalid
):
    model = model_class(minnesota=True, soc=True, sur=True, covid=False)
    before = model.to_vector().copy()

    with pytest.raises(ValueError, match="finite and strictly positive"):
        model.set_priors(**{parameter: invalid})

    np.testing.assert_array_equal(model.to_vector(), before)


@pytest.mark.parametrize("parameter", ["lambda_constant", "lambda_covid"])
@pytest.mark.parametrize("invalid", [0.0, -1.0, np.inf, np.nan])
def test_natural_conjugate_set_priors_rejects_invalid_scale_without_mutation(
    parameter, invalid
):
    model = NaturalConjugate(minnesota=True, soc=True, sur=True, covid=False)
    before = model.to_vector().copy()

    with pytest.raises(ValueError, match="finite and strictly positive"):
        model.set_priors(**{parameter: invalid})

    np.testing.assert_array_equal(model.to_vector(), before)


@pytest.mark.parametrize("invalid", [0.0, -1.0, np.inf, np.nan])
def test_independent_niw_set_priors_rejects_invalid_c2_without_mutation(invalid):
    model = IndependentNIW(minnesota=True, soc=True, sur=True, covid=False)
    before = model.to_vector().copy()
    c2_before = model.c2

    with pytest.raises(ValueError, match="finite and strictly positive"):
        model.set_priors(c2=invalid)

    np.testing.assert_array_equal(model.to_vector(), before)
    assert model.c2 == c2_before
