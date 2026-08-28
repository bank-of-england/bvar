"""
Forecasting package for BVAR.

Re-exports the public API:

* :class:`Forecasting`       – mixin class inherited by ``BVAR``
* :func:`compare_forecasts`  – compare two forecast DataFrames
"""

from .compare import compare_forecasts
from .mixin import Forecasting

__all__ = [
    "Forecasting",
    "compare_forecasts",
]
