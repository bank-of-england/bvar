import numpy as np

from bvar.forecast.matrices import construct_forecast_matrices


def test_var1_impulse_blocks_follow_companion_recursion():
    """For a VAR(1), the impulse matrices M_h must equal chol(Σ) propagated
    h times through the lag-1 coefficient block."""
    n, p, h, H = 2, 1, 0, 4
    A1_block = np.array([[0.5, 0.1], [0.0, 0.3]])  # becomes the lag-1 block
    Sigma = np.array([[1.0, 0.2], [0.2, 0.5]])
    A_0 = np.linalg.cholesky(Sigma)

    # beta layout: [constant, B_1]' with shape (1 + n*p, n)
    beta = np.vstack([np.zeros((1, n)), A1_block])
    y = np.zeros((p, n))

    _, _, M, _, _ = construct_forecast_matrices(y, A_0, beta, p, n, h, H)

    expected = A_0.copy()
    for horizon in range(H):
        np.testing.assert_allclose(M[horizon], expected, atol=1e-10)
        expected = expected @ A1_block
