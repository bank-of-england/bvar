"""
Tests for the Generalised Impulse Response Function (GIRF) module.
"""

from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import bvar as bv
from bvar.girf import (
    _apply_response_transforms,
    _compute_girf_for_draw,
    _compute_ma_matrices,
    _convert_to_level_changes,
    _get_periods_per_year,
    _normalise_base_values,
)

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

n, p, T, H = 3, 2, 300, 10
N_DRAWS = 200
ar_mat = np.diag([0.4, 0.3, 0.2])
Sigma = np.array([[1.0, 0.4, 0.1], [0.4, 2.0, 0.3], [0.1, 0.3, 1.5]])
data, _, true_sigma, _ = bv.simulate_var(T, n, p, ar_mat=ar_mat, Sigma=Sigma, seed=0)

priors = bv.NaturalConjugate(minnesota=True, soc=False, sur=False, covid=False)


@pytest.fixture(scope="module")
def model():
    m = bv.BVAR(
        p,
        bv.NaturalConjugate(minnesota=True, soc=False, sur=False, covid=False),
        stationary=True,
        optimisation_method="none",
    )
    m.sample(data, N_draws=N_DRAWS, progressbar=False, random_state=0)
    m.compute_girf(H=H, N_draws=N_DRAWS, progressbar=False)
    return m


# ---------------------------------------------------------------------------
# Unit tests: _compute_ma_matrices
# ---------------------------------------------------------------------------


def test_ma_h0_is_identity():
    """Ψ_0 must always be the identity matrix."""
    B = np.random.randn(p, n, n) * 0.1
    Psi = _compute_ma_matrices(B, n, p, H)
    np.testing.assert_array_almost_equal(Psi[0], np.eye(n))


def test_ma_recursion_p1():
    """For p=1: Ψ_h = B_1^h (scalar check with diagonal B)."""
    b_val = 0.5
    B = np.array([[[b_val, 0.0], [0.0, b_val]]])  # shape (1, 2, 2)
    Psi = _compute_ma_matrices(B, 2, 1, 4)
    for h in range(5):
        np.testing.assert_array_almost_equal(Psi[h], np.eye(2) * b_val**h)


# ---------------------------------------------------------------------------
# Unit tests: _compute_girf_for_draw
# ---------------------------------------------------------------------------


def test_girf_h0_analytical():
    """
    At horizon 0, Ψ_0 = I, so GIRF_i(0) = Σ[:,i] / sqrt(Σ[i,i]).
    """
    B = np.zeros((p, n, n))
    Psi = _compute_ma_matrices(B, n, p, H)
    girf = _compute_girf_for_draw(Psi, Sigma, n, H)

    for i in range(n):
        expected = Sigma[:, i] / np.sqrt(Sigma[i, i])
        np.testing.assert_array_almost_equal(girf[0, i, :], expected)


def test_girf_shape():
    B = np.zeros((p, n, n))
    Psi = _compute_ma_matrices(B, n, p, H)
    girf = _compute_girf_for_draw(Psi, Sigma, n, H)
    assert girf.shape == (H + 1, n, n)


# ---------------------------------------------------------------------------
# Integration tests: BVAR.compute_girf
# ---------------------------------------------------------------------------


def test_irf_draws_shape(model):
    assert model.irf_draws.shape == (N_DRAWS, H + 1, n, n)


def test_irf_summary_keys(model):
    assert set(model.irf_summary.keys()) == {0.16, 0.50, 0.84}


def test_natural_unit_shock_size_is_fixed_across_posterior_draws(model):
    test_model = deepcopy(model)
    shock_var = test_model.df_data.columns[0]
    shock_natural = 0.25

    test_model.compute_girf(
        H=0,
        N_draws=N_DRAWS,
        shock_size={shock_var: shock_natural},
        progressbar=False,
    )

    np.testing.assert_allclose(
        test_model.irf_draws[:, 0, 0, 0], shock_natural, atol=1e-12
    )


def test_raw_response_keeps_logged_girf_in_model_units(model):
    raw = np.array([[[[1.0]], [[2.0]]]])
    data = np.array([[np.log(100.0)]])
    df_data = model.df_data.iloc[:1, :1].copy()
    df_data.iloc[0, 0] = data[0, 0]
    data_transformation = {df_data.columns[0]: "logs"}

    level_changes = _convert_to_level_changes(
        raw, data_transformation, df_data, data, n=1
    )
    transformed = _apply_response_transforms(
        raw,
        level_changes,
        data_transformation,
        {df_data.columns[0]: "raw"},
        df_data,
        data,
        n=1,
        freq_str="A",
    )

    np.testing.assert_array_equal(transformed, raw)


def test_log_diff_level_change_uses_supplied_base_value(model):
    raw = np.array([[[[0.1]]]])
    data = np.array([[0.01]])
    df_data = model.df_data.iloc[:1, :1].copy()
    data_transformation = {df_data.columns[0]: "log_diff"}

    level_changes = _convert_to_level_changes(
        raw,
        data_transformation,
        df_data,
        data,
        n=1,
        base_value=100.0,
    )

    np.testing.assert_allclose(level_changes, raw * 100.0)


def test_log_diff_level_change_requires_base_value(model):
    raw = np.array([[[[0.1]]]])
    data = np.array([[0.01]])
    df_data = model.df_data.iloc[:1, :1].copy()
    data_transformation = {df_data.columns[0]: "log_diff"}

    with pytest.raises(ValueError, match="base_value.*log_diff"):
        _convert_to_level_changes(raw, data_transformation, df_data, data, n=1)


@pytest.mark.parametrize(
    ("response_type", "expected"),
    [
        ("pct_change", [10.0, 20.0]),
        ("change_yoy", [10.0, 30.0]),
        ("pct_change_yoy", [10.0, 30.0]),
    ],
)
def test_diff_absolute_responses_use_supplied_level_base(
    model, response_type, expected
):
    raw = np.array([[[[10.0]], [[20.0]]]])
    data = np.array([[2.0]])
    df_data = model.df_data.iloc[:1, :1].copy()
    var_name = df_data.columns[0]

    transformed = _apply_response_transforms(
        raw,
        raw.copy(),
        {var_name: "diff"},
        {var_name: response_type},
        df_data,
        data,
        n=1,
        freq_str="Q",
        base_value=100.0,
    )

    np.testing.assert_allclose(transformed[0, :, 0, 0], expected)


@pytest.mark.parametrize(
    "response_type", ["level_change", "pct_change", "change_yoy", "pct_change_yoy"]
)
def test_compute_girf_diff_absolute_response_requires_base_value(model, response_type):
    var_name = model.df_data.columns[0]

    with pytest.raises(ValueError, match="base_value.*diff"):
        deepcopy(model).compute_girf(
            H=0,
            N_draws=1,
            data_transformation={var_name: "diff"},
            response_type={var_name: response_type},
            progressbar=False,
        )


def test_logs_level_change_uses_exponentiated_observation(model):
    raw = np.array([[[[0.1]]]])
    data = np.array([[np.log(100.0)]])
    df_data = model.df_data.iloc[:1, :1].copy()
    data_transformation = {df_data.columns[0]: "log_levels"}

    level_changes = _convert_to_level_changes(
        raw,
        data_transformation,
        df_data,
        data,
        n=1,
        base_value=1000.0,
    )

    np.testing.assert_allclose(level_changes, raw * 100.0)


def test_compute_girf_uses_stored_transformations_and_copies_metadata(model):
    var_name = model.df_data.columns[0]
    stored_transformations = {var_name: "logs"}
    response_type = {var_name: "level_change"}
    shock_size = {var_name: 0.5}

    test_model = deepcopy(model)
    test_model.data_transformation = stored_transformations
    test_model.compute_girf(
        H=0,
        N_draws=1,
        response_type=response_type,
        shock_size=shock_size,
        progressbar=False,
    )

    assert test_model.irf_response_type == response_type
    assert test_model.irf_shock_size == shock_size
    response_type[var_name] = "raw"
    shock_size[var_name] = 1.0
    stored_transformations[var_name] = "levels"
    assert test_model.irf_response_type[var_name] == "level_change"
    assert test_model.irf_shock_size[var_name] == 0.5

    explicit_model = deepcopy(model)
    explicit_model.data_transformation = {var_name: "logs"}
    explicit_model.compute_girf(
        H=0,
        N_draws=1,
        data_transformation={var_name: "levels"},
        response_type={var_name: "level_change"},
        shock_size={var_name: 0.5},
        progressbar=False,
    )
    np.testing.assert_allclose(
        test_model.irf_draws[:, 0, 0, 0],
        explicit_model.irf_draws[:, 0, 0, 0] * np.exp(model.data[-1, 0]),
    )


def test_compute_girf_log_diff_level_change_uses_base_value(model):
    var_name = model.df_data.columns[0]
    raw_model = deepcopy(model)
    raw_model.compute_girf(
        H=0,
        N_draws=1,
        data_transformation={var_name: "log_diff"},
        response_type={var_name: "raw"},
        progressbar=False,
    )

    level_model = deepcopy(model)
    level_model.compute_girf(
        H=0,
        N_draws=1,
        data_transformation={var_name: "log_diff"},
        response_type={var_name: "level_change"},
        base_value=100.0,
        progressbar=False,
    )

    np.testing.assert_allclose(
        level_model.irf_draws[:, 0, 0, 0],
        raw_model.irf_draws[:, 0, 0, 0] * 100.0,
    )


def test_compute_girf_log_diff_level_change_requires_base_value(model):
    var_name = model.df_data.columns[0]
    test_model = deepcopy(model)
    previous_horizon = test_model.irf_H

    with pytest.raises(ValueError, match="base_value.*log_diff"):
        test_model.compute_girf(
            H=0,
            N_draws=1,
            data_transformation={var_name: "log_diff"},
            response_type={var_name: "level_change"},
            progressbar=False,
        )

    assert test_model.irf_H == previous_horizon


@pytest.mark.parametrize("base_value", [np.nan, np.inf, -np.inf])
def test_normalise_base_values_rejects_non_finite_values(base_value):
    with pytest.raises(ValueError, match="finite"):
        _normalise_base_values(base_value, n=2)

    with pytest.raises(ValueError, match="finite"):
        _normalise_base_values([1.0, base_value], n=2)


@pytest.mark.parametrize("base_value", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_compute_girf_rejects_invalid_log_diff_base_before_state_mutation(
    model,
    base_value,
):
    var_name = model.df_data.columns[0]
    test_model = deepcopy(model)
    previous_irf_draws = test_model.irf_draws.copy()
    previous_horizon = test_model.irf_H

    with pytest.raises(ValueError, match="base_value.*positive|finite"):
        test_model.compute_girf(
            H=0,
            N_draws=1,
            data_transformation={var_name: "log_diff"},
            response_type={var_name: "level_change"},
            base_value=base_value,
            progressbar=False,
        )

    assert test_model.irf_H == previous_horizon
    np.testing.assert_array_equal(test_model.irf_draws, previous_irf_draws)


def test_compute_girf_rejects_nonpositive_log_diff_per_variable_base(model):
    test_model = deepcopy(model)
    test_model.df_data = test_model.df_data.copy()
    test_model.df_data.columns = ["first", "second", "third"]
    previous_irf_draws = test_model.irf_draws.copy()

    with pytest.raises(ValueError, match="base_value.*positive"):
        test_model.compute_girf(
            H=0,
            N_draws=1,
            data_transformation={1: "log_diff"},
            response_type={"second": "raw"},
            base_value=[100.0, 0.0, 100.0],
            progressbar=False,
        )

    np.testing.assert_array_equal(test_model.irf_draws, previous_irf_draws)


@pytest.mark.parametrize("transformation", ["log_diff_extra", "diff-log"])
def test_compute_girf_rejects_malformed_transformation_before_state_mutation(
    model,
    transformation,
):
    var_name = model.df_data.columns[0]
    test_model = deepcopy(model)
    previous_irf_draws = test_model.irf_draws.copy()
    previous_horizon = test_model.irf_H

    with pytest.raises(ValueError, match="Unknown data transformation"):
        test_model.compute_girf(
            H=0,
            N_draws=1,
            data_transformation={var_name: transformation},
            response_type={var_name: "raw"},
            progressbar=False,
        )

    assert test_model.irf_H == previous_horizon
    np.testing.assert_array_equal(test_model.irf_draws, previous_irf_draws)


@pytest.mark.parametrize("use_explicit_metadata", [False, True])
def test_compute_girf_resolves_integer_keyed_transformations(
    model,
    use_explicit_metadata,
):
    test_model = deepcopy(model)
    test_model.df_data = test_model.df_data.copy()
    test_model.df_data.columns = ["first", "second", "third"]
    var_name = test_model.df_data.columns[0]
    if use_explicit_metadata:
        test_model.data_transformation = None
        metadata = {0: "logs"}
    else:
        test_model.data_transformation = {0: "logs"}
        metadata = None

    test_model.compute_girf(
        H=0,
        N_draws=1,
        data_transformation=metadata,
        response_type={var_name: "level_change"},
        progressbar=False,
    )

    levels_model = deepcopy(model)
    levels_model.df_data = levels_model.df_data.copy()
    levels_model.df_data.columns = ["first", "second", "third"]
    levels_model.compute_girf(
        H=0,
        N_draws=1,
        data_transformation={var_name: "levels"},
        response_type={var_name: "level_change"},
        progressbar=False,
    )

    np.testing.assert_allclose(
        test_model.irf_draws[:, 0, 0, 0],
        levels_model.irf_draws[:, 0, 0, 0] * np.exp(model.data[-1, 0]),
    )


def test_compute_girf_does_not_treat_boolean_metadata_key_as_integer_index(model):
    bool_key_model = deepcopy(model)
    bool_key_model.df_data = bool_key_model.df_data.copy()
    bool_key_model.df_data.columns = ["first", "second", "third"]
    var_name = bool_key_model.df_data.columns[1]
    bool_key_model.data_transformation = {True: "logs"}
    bool_key_model.compute_girf(
        H=0,
        N_draws=1,
        response_type={var_name: "level_change"},
        progressbar=False,
    )

    levels_model = deepcopy(model)
    levels_model.df_data = levels_model.df_data.copy()
    levels_model.df_data.columns = ["first", "second", "third"]
    levels_model.compute_girf(
        H=0,
        N_draws=1,
        data_transformation={var_name: "levels"},
        response_type={var_name: "level_change"},
        progressbar=False,
    )

    np.testing.assert_allclose(
        bool_key_model.irf_draws[:, 0, 1, 1], levels_model.irf_draws[:, 0, 1, 1]
    )


@pytest.mark.parametrize(
    "freq, expected",
    [
        ("2M", 6),
        ("2ME", 6),
        ("2MS", 6),
        ("2Q-DEC", 2),
        ("2QE-DEC", 2),
        ("2QS-JAN", 2),
        ("M", 12),
        ("ME", 12),
        ("MS", 12),
        ("Q-DEC", 4),
        ("QE-DEC", 4),
        ("QS-JAN", 4),
    ],
)
def test_periods_per_year_supports_multipliers_and_anchors(freq, expected):
    assert _get_periods_per_year(freq) == expected


def test_periods_per_year_uses_integer_floor_for_non_divisors():
    assert _get_periods_per_year("5M") == 2


def test_periods_per_year_rejects_non_positive_multiplier():
    with pytest.raises(ValueError, match="multiplier must be positive"):
        _get_periods_per_year("0M")


@pytest.mark.parametrize("H", [True, 1.5, -1])
def test_compute_girf_rejects_invalid_horizon(model, H):
    with pytest.raises(ValueError, match="H must be a non-negative integer"):
        model.compute_girf(H=H, N_draws=1, progressbar=False)


@pytest.mark.parametrize("N_draws", [True, 1.5, 0, -1])
def test_compute_girf_rejects_invalid_draw_count(model, N_draws):
    with pytest.raises(ValueError, match="N_draws must be a positive integer"):
        model.compute_girf(H=0, N_draws=N_draws, progressbar=False)


def test_compute_girf_accepts_zero_horizon_and_caps_draws(model):
    test_model = deepcopy(model)
    test_model.compute_girf(H=0, N_draws=N_DRAWS + 100, progressbar=False)
    assert test_model.irf_draws.shape == (N_DRAWS, 1, n, n)


def test_order_invariance(model):
    """
    GIRFs are order-invariant: fitting the same model with variables in reverse
    order should produce the same responses (up to reordering of rows/columns).
    """
    data_rev = data.iloc[:, ::-1].copy()
    model_rev = bv.BVAR(p, priors, stationary=True, optimisation_method="none")
    model_rev.sample(data_rev, N_draws=N_DRAWS, progressbar=False, random_state=0)
    model_rev.compute_girf(H=H, N_draws=N_DRAWS, progressbar=False)

    med_orig = model.irf_summary[0.50]  # (H+1, n, n): [h, shock, response]
    med_rev = model_rev.irf_summary[0.50]

    # Reorder both axes: shock and response are both reversed
    idx = list(range(n - 1, -1, -1))
    med_rev_reordered = med_rev[:, idx, :][:, :, idx]

    # Should agree within sampling noise — check correlation > 0.99
    corr = np.corrcoef(med_orig.flatten(), med_rev_reordered.flatten())[0, 1]
    assert corr > 0.99, f"Order invariance failed: correlation = {corr:.4f}"


def test_not_fitted_raises():
    blank = bv.BVAR(p, priors, stationary=True, optimisation_method="none")
    with pytest.raises(RuntimeError):
        blank.compute_girf(H=4)


# ---------------------------------------------------------------------------
# Integration tests: girf_to_dataframe
# ---------------------------------------------------------------------------


def test_dataframe_shape(model):
    df = model.irf_df
    # (H+1) horizons * n shocks * n responses * 3 quantiles
    assert len(df) == (H + 1) * n * n * 3
    assert set(df.columns) == {"horizon", "shock", "response", "quantile", "value"}


# ---------------------------------------------------------------------------
# Integration tests: plot_girf
# ---------------------------------------------------------------------------


def test_plot_returns_figure(model):
    fig = model.plot_girf()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_subset(model):
    fig = model.plot_girf(shock_var=0, response_var=[0, 1])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Consistency test: GIRFs ↔ conditional forecasts
# ---------------------------------------------------------------------------


def test_girf_conditional_forecast_consistency(model):
    """
        The GIRF is the difference between two point forecasts:
            - uncond: the unconditional point forecast from the current history
            - cond:   the point forecast with ALL n variables at h=1 pinned to
                uncond[h=1] + GIRF_i(0)

    Constraining all n variables at the same step fully identifies the
    n contemporaneous shocks, so no future shocks fire and the propagation
    is identical to the GIRF.  Holds exactly (up to floating-point) for
    any estimated model when point_only=True.

    Checked for every shock i = 0, …, n-1 using the module-level model.
    """
    H_c = 8

    # Point-only GIRF from the already-estimated trivariate model
    # Use response_type="raw" to get untransformed GIRFs
    model.compute_girf(
        H=H_c,
        N_draws=1,
        point_only=True,
        response_type={"x": "raw", "y": "raw", "z": "raw"},
        progressbar=False,
    )
    girf = model.irf_summary[0.50]  # (H_c+1, n, n): girf[h, shock, response]

    # Unconditional point forecast
    model.forecast(H=H_c, point_only=True, N_draws=1, progressbar=False)
    uncond = model.forecast_unconditional[0, -H_c:, :]  # (H_c, n)

    for shock_i in range(n):
        # Constrain ALL n variables at h=1 to uncond[h=1] + GIRF_i(h=0).
        # This identifies the contemporaneous shock uniquely; future shocks
        # stay zero (minimum-norm solution), so the remaining path = GIRF.
        constraint_mean = np.full((H_c, n), np.nan)
        constraint_mean[0, :] = uncond[0, :] + girf[0, shock_i, :]

        model.forecast(
            H=H_c,
            constraint_mean=constraint_mean,
            point_only=True,
            N_draws=1,
            progressbar=False,
        )
        cond = model.forecast_conditional[0, -H_c:, :]  # (H_c, n)

        np.testing.assert_allclose(
            cond - uncond,
            girf[:H_c, shock_i, :],
            atol=1e-6,
            err_msg=f"GIRF/forecast mismatch for shock_i={shock_i}",
        )
