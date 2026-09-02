import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from bvar.plots import plot_density, plot_histogram


@pytest.fixture
def two_series():
    rng = np.random.default_rng(0)
    return np.c_[rng.normal(0, 1, 500), rng.normal(2, 0.5, 500)]


@pytest.mark.parametrize("fn", [plot_histogram, plot_density])
def test_returns_fig_ax(fn, two_series):
    fig, ax = fn(two_series, labels=["a", "b"], title="t")
    assert fig is not None and ax is not None
    assert len(ax.get_legend().get_texts()) == 2


@pytest.mark.parametrize("fn", [plot_histogram, plot_density])
def test_label_length_mismatch_raises(fn, two_series):
    with pytest.raises(ValueError):
        fn(two_series, labels=["only-one"])
