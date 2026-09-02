from importlib.metadata import PackageNotFoundError, version

from .BVAR import BVAR
from .diagnostics import mcmc_posterior
from .forecast import compare_forecasts
from .girf import GIRF
from .models import IndependentNIW, NaturalConjugate
from .plots import plot_delta_forecast, plot_density, plot_histogram
from .utils import cumulative_change, simulate_var

__version__ = version("bvar")

__all__ = [
    "BVAR",
    "GIRF",
    "IndependentNIW",
    "NaturalConjugate",
    "compare_forecasts",
    "cumulative_change",
    "mcmc_posterior",
    "plot_delta_forecast",
    "plot_density",
    "plot_histogram",
    "simulate_var",
]
