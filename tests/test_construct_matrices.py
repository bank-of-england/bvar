import numpy as np
import pytest

from bvar.dummy_observations import stack_dummies
from bvar.models import NaturalConjugate
from bvar.models.common import _AR1_Y_X
from bvar.utils import (
    construct_X,
    construct_Y_Z,
    get_dimensions,
    simulate_var,
    simulate_var_simple,
)


def test_construct_X_basic():
    y = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    n_lags = 2
    covid_indices = []

    X = construct_X(y, n_lags, covid_indices)
    X_true = np.array(
        [
            [1.0, 3.0, 4.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 4.0, 1.0, 2.0],
            [1.0, 5.0, 6.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 6.0, 3.0, 4.0],
        ]
    )

    assert (X == X_true).all(), "construct_X did not produce expected output."


def test_construct_X_covid():
    y = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    n_lags = 2
    covid_indices = [2, 3]

    X = construct_X(y, n_lags, covid_indices)
    X_true = np.array(
        [
            [1.0, 3.0, 4.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 4.0, 1.0, 2.0, 1.0, 0.0],
            [1.0, 5.0, 6.0, 3.0, 4.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 6.0, 3.0, 4.0, 0.0, 1.0],
        ]
    )

    assert (X == X_true).all(), (
        "construct_X with covid did not produce expected output."
    )


def test_construct_Y_Z_basic():
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    covid_indices = []
    n_lags = 2

    Y_true = np.array([[5.0, 6.0], [7.0, 8.0]])
    Z_true = np.array([[1.0, 3.0, 4.0, 1.0, 2.0], [1.0, 5.0, 6.0, 3.0, 4.0]])

    Y, Z = construct_Y_Z(data, n_lags=n_lags, covid_indices=covid_indices)

    assert (Y == Y_true).all(), "construct_Y_Z did not produce expected Y."
    assert (Z == Z_true).all(), "construct_Y_Z did not produce expected Z."


def test_construct_Y_Z_covid():
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    covid_indices = [2, 3]
    n_lags = 2

    Y_true = np.array([[5.0, 6.0], [7.0, 8.0]])
    Z_true = np.array(
        [[1.0, 3.0, 4.0, 1.0, 2.0, 1.0, 0.0], [1.0, 5.0, 6.0, 3.0, 4.0, 0.0, 1.0]]
    )

    Y, Z = construct_Y_Z(data, n_lags=n_lags, covid_indices=covid_indices)

    assert (Y == Y_true).all(), "construct_Y_Z with covid did not produce expected Y."
    assert (Z == Z_true).all(), "construct_Y_Z with covid did not produce expected Z."


def test_covid_indices_are_aligned_with_main_var_rows():
    """Pre-lag and out-of-range COVID indices do not create empty columns."""
    data = np.arange(10.0).reshape(5, 2)
    n_lags = 2
    covid_indices = [-1, 0, 1, 4, 5, 4]

    X = construct_X(data, n_lags, covid_indices)
    Y, Z = construct_Y_Z(data, n_lags, covid_indices)
    dimensions = get_dimensions(data, n_lags, covid_indices)

    assert dimensions == (3, 2, 6, 12, 1)
    assert X.shape == (6, 12)
    assert Y.shape == (3, 2)
    assert Z.shape == (3, 6)
    assert np.array_equal(Z[:, -1], [0.0, 0.0, 1.0])
    assert np.array_equal(
        X[:, [5, 11]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    )


def test_covid_indices_are_sorted_and_deduplicated():
    """Duplicate and unordered COVID indices have a stable canonical order."""
    data = np.arange(10.0).reshape(5, 2)
    covid_indices = [3, 2, 3]

    _, Z = construct_Y_Z(data, n_lags=1, covid_indices=covid_indices)

    assert get_dimensions(data, 1, covid_indices)[-1] == 2
    assert np.array_equal(Z[:, -2:], [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])


def test_stack_dummies_uses_canonical_covid_columns():
    """SOC and SUR dummy rows match the main VAR COVID columns."""
    data = np.arange(10.0).reshape(5, 2)
    n_lags = 2
    covid_indices = [-1, 0, 1, 4, 5, 4]
    levels = np.ones(data.shape[1], dtype=bool)
    model = NaturalConjugate(minnesota=False, soc=True, sur=True, covid=True)
    Y, Z = construct_Y_Z(data, n_lags, covid_indices)

    Y_stacked, Z_stacked, nb_dummy_obs = stack_dummies(
        Y,
        Z,
        n_lags,
        levels,
        model,
        covid_indices,
        soc=True,
        sur=True,
    )

    assert nb_dummy_obs == data.shape[1] + 1
    assert Y_stacked.shape == (Y.shape[0] + nb_dummy_obs, data.shape[1])
    assert Z_stacked.shape == (Z.shape[0] + nb_dummy_obs, Z.shape[1])


def test_covid_indices_must_be_integers():
    """Matrix builders reject fractional COVID row indexes."""
    data = np.arange(10.0).reshape(5, 2)

    with pytest.raises(ValueError, match="integer"):
        get_dimensions(data, n_lags=1, covid_indices=[2.5])


def test_ar1_covid_indices_use_the_one_lag_cutoff():
    """AR(1) initialisation keeps COVID rows that precede the main VAR cutoff."""
    data = np.arange(10.0).reshape(5, 2)

    Y, X = _AR1_Y_X(data, covid_indices=[1, 2])

    assert Y.shape == (4, 2)
    assert X.shape == (2, 4, 4)
    assert np.array_equal(
        X[0, :, -2:], [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
    )


def test_var_formulation():
    # check that the vectorised formulation gives the same as the standard formulation

    T = 100  # Number of time periods
    n = 2  # Number of variables
    n_lags = 1  # Number of lags
    covid = False
    levels = False

    ar_mat = np.array([[0.1, 0.11], [0.21, 0.22]])
    constant = np.array([-1, 1])
    Sigma = np.array([[1, 0.5], [0.5, 2]])
    seed = 1234

    data, true_b, true_sigma, _ = simulate_var(
        T,
        n,
        n_lags,
        ar_mat=ar_mat,
        constant=constant,
        Sigma=Sigma,
        seed=seed,
        covid=covid,
        levels=levels,
    )

    data_check = simulate_var_simple(
        T, n, n_lags, ar_mat=ar_mat, constant=constant, Sigma=Sigma, seed=seed
    )

    difference = np.abs((data.values - data_check).sum())

    assert difference < 1e-10, "VAR formulations do not match."


def test_simulate_var_supports_univariate_default_covariance():
    """The default covariance is valid for a one-variable simulation."""
    data, _, Sigma, eps = simulate_var(T=20, n=1, n_lags=1, seed=1234)

    assert data.shape == (20, 1)
    assert Sigma.shape == (1, 1)
    assert eps.shape == (20, 1)


def test_simulate_var_simple_supports_univariate_default_covariance():
    """The simple simulator accepts its default covariance for one variable."""
    data = simulate_var_simple(T=20, n=1, n_lags=1, seed=1234)

    assert data.shape == (20, 1)
