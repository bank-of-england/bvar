# IndependentNIW

The `IndependentNIW` model implements Bayesian VAR estimation with **independent Normal-Inverse-Wishart** priors. Unlike the natural-conjugate specification, the prior on the VAR coefficients is independent of $\Sigma$, which allows a richer $(nk \times nk)$ full-system prior covariance matrix whose inverse encodes cross-variable shrinkage scaling by $\sigma_i / \sigma_j$.

Because the prior is not conjugate, the posterior has no closed form and the model uses a **Gibbs sampler** that alternates between drawing $\beta \mid \Sigma, Y$ and $\Sigma \mid \beta, Y$. Burn-in draws are automatically discarded.

The public `beta_point` and `sigma_point` estimates come from the
retained Gibbs draws, after burn-in. They are posterior means rather than
posterior modes, and `posterior_state_point` references those same point
arrays for forecasting with the full posterior state.

!!! note
    `IndependentNIW` does not support marginal-likelihood optimisation, and its Gibbs
    sampler has no closed-form posterior point estimate, so `optimisation_method="cross_validation"`
    is not supported either (it requires refitting with `point_only=True`). Use
    `optimisation_method="none"` and set hyperparameters manually when constructing `BVAR`.

Constructor arguments (`minnesota`, `soc`, `sur`, `covid`, `covid_dates`) are documented on [`SamplingModel`](index.md#bvar.models.SamplingModel); `soc` and `sur` default to `True`.

## Prior Structure

$$\beta \,\sim\, \mathcal{N}(\beta_0, V_\beta)$$

$$\Sigma \,\sim\, \mathcal{IW}(S_0, \nu_0)$$

where $V_\beta$ is a full $(nk \times nk)$ covariance matrix. Its inverse is the prior precision used in the conditional posterior. The cross-variable off-diagonal blocks are scaled by a parameter `c2` (Litterman, 1986), giving tighter shrinkage on cross-equation coefficients relative to own-equation coefficients. The Gibbs sampler iterates:

$$\beta \mid \Sigma, Y \;\sim\; \mathcal{N}\!\left(\tilde\beta,\; \tilde{V}\right), \qquad \tilde{V}^{-1} = V_\beta^{-1} + \Sigma^{-1} \otimes Z'Z$$

$$\Sigma \mid \beta, Y \;\sim\; \mathcal{IW}\!\left(S_0 + E'E,\; \nu_0 + T\right)$$

## Hyperparameters

| Parameter | Description | Default |
|---|---|---|
| `c2` | Cross-variable shrinkage | 0.5 |
| `c1` | Overall tightness | 0.2 |
| `c3` | Lag-decay exponent | 2.0 |
| `mu` | SOC tightness (if `soc=True`) | 1.0 |
| `theta` | SUR tightness (if `sur=True`) | 1.0 |

## Example

```python
import bvar as bv

model = bv.IndependentNIW(
    c2=0.5,  # cross-variable shrinkage
    minnesota=True,
    soc=False,
    sur=False,
    covid=False,
)

# ML and cross-validation optimisation are not available; use "none"
bvar = bv.BVAR(n_lags=4, model=model, stationary=False, optimisation_method="none")
bvar.sample(data, N_draws=5000)
```

## API

::: bvar.IndependentNIW
    options:
      show_root_heading: true
      members_order: source
      merge_init_into_class: true
