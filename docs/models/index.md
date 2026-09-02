# Models

The sampling model determines both the prior specification and the estimation algorithm. Every model is a subclass of `SamplingModel` and owns the complete estimation pipeline — prior construction, data-matrix assembly, dummy-observation stacking, and posterior sampling.

## Selecting a Model

| Model | Prior structure | Estimation | Marginal likelihood |
|---|---|---|---|
| [`NaturalConjugate`](natural_conjugate.md) | $\text{vec}(A) \mid \Sigma \sim \mathcal{N}(\beta_0,\, \Sigma \otimes V_A^{-1})$, $\Sigma \sim \mathcal{IW}(S_0, \nu_0)$ | Direct (no MCMC) | ✔ |
| [`IndependentNIW`](independent_niw.md) | $\beta \sim \mathcal{N}(\beta_0, V_\beta)$ independently of $\Sigma \sim \mathcal{IW}(S_0, \nu_0)$ | Gibbs sampler | ✘ |

The **Marginal likelihood** column indicates whether `optimise_hyperparameters` with `optimisation_method="ml"` is supported. `IndependentNIW` (✘) has no closed-form marginal likelihood, and its Gibbs sampler has no closed-form point estimate, so neither `"ml"` nor `"cross_validation"` is available for it — passing either raises `ValueError`. Use `optimisation_method="none"` and set hyperparameters manually.

Use `NaturalConjugate` for most applications: it runs faster without burn-in, supports marginal-likelihood hyperparameter optimisation, and covers the standard Minnesota / SOC / SUR prior setup. Use `IndependentNIW` when you need a **full** $(nk \times nk)$ prior precision matrix that scales cross-variable shrinkage independently of $\Sigma$.

## Usage

```python
import bvar as bv

# Natural conjugate (fast, supports ML optimisation)
model = bv.NaturalConjugate(
    minnesota=True,
    soc=True,
    sur=True,
    covid=False,
)

bvar = bv.BVAR(n_lags=4, model=model, stationary=False)
bvar.optimise_hyperparameters(data)
bvar.sample(data, N_draws=5000)
```

```python
# Independent NIW (full-system prior precision, Gibbs sampler)
model = bv.IndependentNIW(
    c2=0.5,  # cross-variable shrinkage
    minnesota=True,
    soc=False,
    sur=False,
)

bvar = bv.BVAR(n_lags=4, model=model, stationary=False, optimisation_method="none")
bvar.sample(data, N_draws=5000)
```

## Base Class

::: bvar.models.SamplingModel
    options:
      show_root_heading: true
      members_order: source
      merge_init_into_class: true

---

::: bvar.models.SamplingResult
    options:
      show_root_heading: true
      merge_init_into_class: true
