"""Shared fixtures for the bvar test suite.

Fitting a ``NaturalConjugate`` BVAR requires costly hyperparameter
optimisation with multiple restarts and a 5,000-draw posterior sample.
The suite previously ran this work as module-level code in
``test_forecast.py`` (and, unused, in ``test_skew_normal.py``), which meant
pytest therefore paid the cost at *collection* time whenever it imported the
module, even when it selected no test from that module.

The fixture defers this cost until a selected test requests it. Its
``scope="session"`` setting runs the fit at most once per test session.

The tests share the fitted instance. Posterior draws (``beta``/``sigma``) are
never mutated after fitting, and ``forecast()`` / ``recursive_forecast()``
overwrite their own ``forecast_*`` outputs on every call, so nothing leaks
between tests there. The only shared mutable state is ``rng``; each test
controls it by passing ``random_state=SEED`` to its first stochastic call.
This keeps results deterministic and independent of execution order without
copying the large posterior draw arrays.
"""

import numpy as np
import pytest

from bvar.BVAR import BVAR
from bvar.models import NaturalConjugate
from bvar.utils import simulate_var

T = 200  # Number of time periods
N_VARS = 2  # Number of variables
N_LAGS = 1  # Number of lags
SEED = 1234

_AR_MAT = np.array([[0.1, 0.25], [0.5, 0.75]])
_CONSTANT = np.array([-1, 1])
_SIGMA = np.array([[1, 0.5], [0.5, 2]])


@pytest.fixture(scope="session")
def bvar():
    """The shared, session-fitted BVAR instance.

    Tests pass ``random_state=SEED`` to the first stochastic
    ``forecast()``/``recursive_forecast()`` call to keep results deterministic
    and independent of test order (see the module docstring).
    """
    data, _, _, _ = simulate_var(
        T,
        N_VARS,
        N_LAGS,
        ar_mat=_AR_MAT,
        constant=_CONSTANT,
        Sigma=_SIGMA,
        seed=SEED,
        covid=False,
        levels=False,
    )

    model = NaturalConjugate(minnesota=True, soc=True, sur=True, covid=False)

    fitted = BVAR(N_LAGS, model, False, optimisation_method="ml")
    fitted.optimise_hyperparameters(data=data, nb_restart=1, random_state=SEED)
    fitted.sample(data=data, N_draws=5000, random_state=SEED)

    return fitted
