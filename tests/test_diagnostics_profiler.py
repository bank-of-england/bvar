"""Focused coverage for diagnostics plotting and profiling utilities."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bvar.diagnostics import mcmc_posterior
from bvar.profiler import profile_code


def test_mcmc_posterior_handles_one_parameter_without_interactive_display():
    """A one-parameter posterior renders and returns its figure."""
    draws = np.arange(20.0).reshape(10, 2)[:, :1]

    try:
        fig = mcmc_posterior(draws, true_pars=np.array([4.0]))
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        true_lines = [
            line for line in fig.axes[0].lines if line.get_label().startswith("True:")
        ]
        assert len(true_lines) == 1
        np.testing.assert_allclose(true_lines[0].get_xdata(), [4.0, 4.0])
    finally:
        plt.close("all")


def test_mcmc_posterior_removes_unused_subplot_axes():
    """Several parameters leave no unused axes in the subplot grid."""
    draws = np.arange(50.0).reshape(10, 5)
    true_pars = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    try:
        fig = mcmc_posterior(draws, true_pars=true_pars, max_cols=3)
        axes = fig.axes
        assert len(axes) == len(true_pars)
        plotted_true_values = [
            line.get_xdata()[0]
            for axis in axes
            for line in axis.lines
            if line.get_label().startswith("True:")
        ]
        np.testing.assert_allclose(plotted_true_values, true_pars)
    finally:
        plt.close("all")


def test_profile_code_reports_on_normal_exit(capsys):
    """The profiler prints statistics when the context exits normally."""
    with profile_code(top_n=1):
        sum(range(10))

    assert "function calls" in capsys.readouterr().out


def test_profile_code_reports_before_reraising_exception(capsys):
    """The profiler reports the block and disables itself when it raises."""
    with pytest.raises(RuntimeError, match="expected"):
        with profile_code(top_n=1) as profiler:
            raise RuntimeError("expected")

    assert "function calls" in capsys.readouterr().out
    stats_after_exit = profiler.getstats()
    sum(range(10))
    assert profiler.getstats() == stats_after_exit
