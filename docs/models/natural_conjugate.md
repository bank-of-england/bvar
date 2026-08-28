# NaturalConjugate

The `NaturalConjugate` model implements Bayesian VAR estimation with a **Natural-Conjugate Normal-Inverse-Wishart** prior. Conjugacy gives a closed-form posterior, so the model needs neither MCMC nor burn-in. This makes it a fast and versatile option for most applications.

## Prior Structure

The prior is specified jointly over the coefficients and the covariance matrix:

$$\text{vec}(A) \mid \Sigma \,\sim\, \mathcal{N}\!\left(\beta_0,\; \Sigma \otimes V_A^{-1}\right)$$

$$\Sigma \,\sim\, \mathcal{IW}(S_0, \nu_0)$$

where $V_A^{-1}$ is the $(k \times k)$ **per-equation** prior precision matrix (Minnesota form) and $\beta_0$ encodes the random-walk prior means. After observing data $Y$, the posteriors update to known closed-form distributions, from which draws are taken directly.

## Hyperparameters

| Parameter | Description | Default |
|---|---|---|
| `c1` | Overall shrinkage / tightness | 0.2 |
| `c3` | Lag-decay exponent | 2.0 |
| `mu` | SOC tightness (if `soc=True`) | 1.0 |
| `theta` | SUR tightness (if `sur=True`) | 1.0 |
| `lambda_covid` | COVID-dummy prior variance | 10 000 |

Hyperparameters can be set manually via `set_priors()` or optimised automatically with `BVAR.optimise_hyperparameters()` (uses marginal-likelihood maximisation).

## Example

```python
import bvar as bv

model = bv.NaturalConjugate(
    minnesota=True,
    soc=True,
    sur=True,
    covid=True,
)

bvar = bv.BVAR(n_lags=4, model=model, stationary=False, optimisation_method="ml")
bvar.optimise_hyperparameters(data)
bvar.sample(data, N_draws=5000)
```

## API

::: bvar.NaturalConjugate
    options:
      show_root_heading: true
      members_order: source
      merge_init_into_class: true
