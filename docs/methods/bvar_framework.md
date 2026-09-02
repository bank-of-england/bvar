# Bayesian Vector Autoregression (BVAR) Framework

## Introduction

This library implements Bayesian Vector Autoregressions (BVARs) with natural conjugate (Normal-Inverse-Wishart) priors. The Bayesian approach regularises the large number of VAR parameters through economically motivated shrinkage, making the model well suited to macroeconomic forecasting.

The exposition below (likelihood, priors, posteriors and forecasting) describes the **Gaussian reduced-form model built into every current implementation** (`NaturalConjugate`, `IndependentNIW`). The framework's abstract `SamplingModel` interface does not itself require Gaussianity; see [Model-Owned Predictive State and Capability Boundaries](#model-owned-predictive-state-and-capability-boundaries) for how a differently-distributed model would fit in.

## VAR Model

A VAR of order $p$ models $n$ time series as the Gaussian reduced-form:

$$y_t = c + A_1 y_{t-1} + \cdots + A_p y_{t-p} + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \Sigma)$$

where $y_t$ is $(n \times 1)$, $c$ is an $(n \times 1)$ intercept vector, each $A_i$ is an $(n \times n)$ coefficient matrix, and $\Sigma$ is the $(n \times n)$ error covariance matrix.

**Matrix form.** For $T$ observations the system is:

$$Y = Z A + E$$

where $Y$ is $(T \times n)$, $Z$ is $(T \times k)$ with rows $[1,\; y_{t-1}',\; \ldots,\; y_{t-p}',\; d_t']$, $A$ is $(k \times n)$, and $k = 1 + np + h$ ($h$ = number of exogenous dummies). The errors satisfy $E \sim \mathcal{MN}(0, I_T, \Sigma)$.

## Bayesian Framework

### Likelihood

Under the Gaussian innovations assumed above, the likelihood is:

$$p(Y \mid A, \Sigma) \propto |\Sigma|^{-T/2} \exp\!\Bigl(-\tfrac{1}{2}\text{tr}\bigl[(Y - ZA)' \Sigma^{-1} (Y - ZA)\bigr]\Bigr)$$

### Prior Distributions

The natural conjugate prior is Normal-Inverse-Wishart:

$$\Sigma \sim \mathcal{IW}(S_0,\; \nu_0)$$

$$A \mid \Sigma \sim \mathcal{MN}(A_0,\; V_A,\; \Sigma)$$

where $A_0$ is the $(k \times n)$ prior mean, $V_A$ is the $(k \times k)$ prior covariance (controlling shrinkage), $S_0$ is the $(n \times n)$ prior scale matrix, and $\nu_0$ is the prior degrees of freedom.

### Posterior Distributions

Conjugacy yields closed-form posteriors:

$$\Sigma \mid Y \sim \mathcal{IW}(S_\text{post},\; \nu_\text{post})$$

$$A \mid \Sigma, Y \sim \mathcal{MN}(A_\text{post},\; K_A^{-1},\; \Sigma)$$

with:

$$K_A = V_A^{-1} + Z'Z$$

$$A_\text{post} = K_A^{-1}\bigl(V_A^{-1} A_0 + Z'Y\bigr)$$

$$\nu_\text{post} = \nu_0 + T$$

$$S_\text{post} = S_0 + A_0' V_A^{-1} A_0 + Y'Y - A_\text{post}' K_A\, A_\text{post}$$

Because the posteriors are known analytically, the `sample()` method draws directly (no MCMC required) following Chan (2020): first draw $\Sigma$ from its marginal, then draw $A$ conditional on $\Sigma$.

## Prior Specifications

### Minnesota Prior

The Minnesota prior encodes three beliefs:

1. **Random walk.** The prior mean for the first own lag is 1 for variables in levels and 0 for variables in differences. All other coefficients have prior mean 0.

2. **Own lags matter more than cross lags.** Cross-variable coefficients are shrunk more aggressively than own-lag coefficients.

3. **Recent lags matter more.** Shrinkage increases with lag order.

These beliefs are encoded in the diagonal prior precision matrix $V_A^{-1}$. For the coefficient on lag $\ell$ of variable $j$ in the equation for variable $i$:

$$[V_A^{-1}]_{ij,\ell} = \frac{c_1^2}{\ell^{\,c_3} \cdot s_i}$$

where $c_1$ is the overall tightness, $c_3$ controls lag decay, and $s_i$ is the residual variance from a univariate AR(1) for variable $i$ (scaling puts shrinkage on a common scale). Intercepts and COVID dummies receive fixed, loose prior variances.

The prior scale matrix $S_0$ is set to $\text{diag}(s_1, \ldots, s_n) \times (\nu_0 - n - 1)$ with $\nu_0 = n + 4$, so that the prior mean of $\Sigma$ equals the AR(1) residual variances.

### Sum-of-Coefficients Prior

The sum-of-coefficients (SOC) prior (Doan, Litterman and Sims, 1984) discourages the model from differencing away persistent levels by encouraging the sum of lag coefficients on each variable to equal one. It is implemented by appending $n$ dummy observations to the data:

$$Y_d = \frac{1}{\mu}\,\text{diag}(\bar{y}), \qquad Z_d = \bigl[\,0 \;\;\; \underbrace{\text{diag}(\bar{y}) \;\cdots\; \text{diag}(\bar{y})}_{p\text{ times}} \;\;\; 0\,\bigr] / \mu$$

where $\bar{y}$ is a vector of initial sample means and $\mu$ controls tightness. As $\mu \to 0$ the prior becomes dogmatic.

### Single-Unit-Root Prior

The single-unit-root (SUR) prior (Sims, 1993) imposes a common stochastic trend across all variables. It adds one dummy observation:

$$Y_d = \bar{y}' / \theta, \qquad Z_d = \bigl[\,1/\theta \;\;\; \underbrace{\bar{y}' \;\cdots\; \bar{y}'}_{p\text{ times}} \;\;\; 0\,\bigr] / \theta$$

where $\theta$ controls tightness.

Both SOC and SUR priors are stacked with the actual data before estimation, preserving the conjugate structure.

### COVID-19 Dummies

Dummy variables for the COVID-19 period (default: 2020Q1–2021Q4) are included as additional columns in $Z$. Their prior variance is fixed at a large value ($\lambda_\text{covid} = 10{,}000$) so the data determines their coefficients, following Cascaldi-Garcia (2022).

## Hyperparameter Optimisation

The hyperparameters $\lambda = (c_1, c_3, \mu, \theta)$ can be:

- **Fixed** at user-supplied values, or
- **Optimised** by maximising the log marginal likelihood (Giannone, Lenza and Primiceri, 2015):

$$\log p(Y \mid \lambda) = \int p(Y \mid A, \Sigma)\, p(A, \Sigma \mid \lambda)\, dA\, d\Sigma$$

which has a closed form under the conjugate prior. Optional Gamma hyperpriors on $\lambda$ can regularise the optimisation.

The optimisation uses L-BFGS with a softplus transformation to enforce positivity, and supports a multi-start strategy to reduce sensitivity to local optima.

Alternative optimisation criteria are also available:
- **Predictive marginal likelihood** — out-of-sample predictive density
- **Cross-validation** — expanding rolling-window predictive likelihood; see the [cross-validation documentation](cross_validation.md)

## Forecasting

### Unconditional Forecasts

The `forecast()` method (without constraints) generates $H$-step-ahead forecasts. For each posterior draw $(A^{(i)}, \Sigma^{(i)})$:

$$\hat{y}_{T+h}^{(i)} = c^{(i)} + A_1^{(i)} \hat{y}_{T+h-1}^{(i)} + \cdots + A_p^{(i)} \hat{y}_{T+h-p}^{(i)} + \varepsilon_{T+h}^{(i)}, \qquad \varepsilon_{T+h}^{(i)} \sim \mathcal{N}(0, \Sigma^{(i)})$$

The collection of paths characterises the predictive distribution, capturing both parameter uncertainty and future shock uncertainty.

### Conditional Forecasts

The `forecast()` method also supports conditional forecasting — imposing constraints on the path of selected variables — following Waggoner and Zha (1999) and Antolín-Díaz, Petrella and Rubio-Ramírez (2021).

**Structural representation.** Using the Cholesky factor $A_0 = \text{chol}(\Sigma)'$, the $H$-step forecast is written as:

$$y_{T+1:T+H} = b + \mathcal{M}^{\top} \varepsilon$$

where $b$ is the deterministic component (intercepts and lagged values), $\varepsilon$ is the $(nH \times 1)$ vector of structural shocks, and $\mathcal{M}$ is the block lower-triangular impulse-response matrix with blocks $M_h$ built recursively (matching the [skewed constraints](skewed_constraints.md) page):

$$M_0 = A_0, \qquad M_h = \sum_{j=1}^{\min(h,p)} M_{h-j}\, B_j \quad (h \ge 1)$$

**Constraints.** A selection matrix $C$ and target vector $f$ encode the conditions $C\, y_{T+1:T+H} = f$. The structural shocks consistent with the constraints are:

$$\varepsilon = \xi + D^*(z - D\xi)$$

where $\xi \sim \mathcal{N}(0, I_{nH})$ is an unconstrained shock draw, $D = C\mathcal{M}'$, $D^{*} = D'(DD')^{-1}$ is the Moore-Penrose pseudo-inverse of $D$, and $z$ is drawn from the constraint distribution.

**Hard constraints** set $z = f - Cb$ exactly. **Soft constraints** draw $z$ from a normal (or skew-normal) distribution centred on the target, allowing constraints subject to uncertainty. See the [skewed constraints documentation](skewed_constraints.md) for details.

The conditional forecasting algorithm iterates between drawing constrained forecasts and updating the posterior state: each iteration augments the data with the newly drawn forecast, constructs the corresponding $(Y, Z)$ matrices, and invokes `sample_posterior_state` to draw the next $(A, \Sigma)$ — it does not re-run the full `BVAR.sample` pipeline or recompute the priors.

## Model-Owned Predictive State and Capability Boundaries

The forecasting mechanics above are the **default** behaviour supplied by `SamplingModel`, not a framework-wide constraint. Three concrete hooks — `sample_innovations`, `sample_conditional_forecast` and `predictive_logpdf` — determine, respectively, how forecast innovations are drawn, how constrained forecasts are sampled, and how out-of-sample predictive densities are evaluated. Each hook is passed the full `PosteriorState` for a draw (`beta`, `sigma`, plus a model-owned `extras` payload), and each default implementation is Gaussian, matching every model currently in the package.

A model with a non-Gaussian predictive distribution sets `supports_gaussian_predictive = False` and overrides each hook that its forecasting workflow uses. The defaults raise `NotImplementedError` instead of applying the Gaussian formulas. The model can carry auxiliary state, such as a sampled degrees of freedom parameter, in `PosteriorState.extras`; `sample_posterior_state` passes that state through the conditional-forecast Gibbs chain and deep-copies it when the chain starts from the fitted point estimate.

This also bounds two related capabilities:

- **Generalised impulse responses.** The current `GIRF` implementation is hard-coded to the Gaussian reduced-form predictive distribution, so `compute_girf` requires both `supports_girf=True` and `supports_gaussian_predictive=True`.
- **Predictive-likelihood optimisation.** Cross-validation and predictive marginal-likelihood scoring (see [Hyperparameter Optimisation](#hyperparameter-optimisation)) route through `predictive_logpdf`, so a non-Gaussian model must override it to be used with those optimisation criteria.

See [Adding New Models](../guide/adding_models.md#optional-predictive-hooks) for the full hook contract.

Seeded results (`random_state`/`rng`) are reproducible within a single installed release but not guaranteed across releases: see [Reproducibility](../guide/forecasting.md#reproducibility) for details, including the breaking change to the `sample_posterior_state` sampler contract introduced in 0.3.0.

## Usage Example

```python
import numpy as np
import bvar as bv

# --- 1. Simulate or load data ---
data, true_b, true_sigma, _ = bv.simulate_var(
    T=200,
    n=3,
    n_lags=2,
    levels=True,
    seed=42,
)

# --- 2. Specify the sampling model ---
model = bv.NaturalConjugate(
    minnesota=True,  # Minnesota shrinkage
    soc=True,  # Sum-of-coefficients prior
    sur=True,  # Single-unit-root prior
    covid=False,
)

# --- 3. Create the BVAR and optimise hyperparameters ---
bvar = bv.BVAR(
    n_lags=2, model=model, stationary=False, optimisation_method="ml", random_state=42
)
bvar.optimise_hyperparameters(data)

# --- 4. Sample from the posterior ---
bvar.sample(data, N_draws=5000)

# --- 5. Unconditional forecast ---
bvar.forecast(H=8)
bvar.plot_forecast()

# --- 6. Conditional forecast (fix variable 0 for the first 4 quarters) ---
H, n = 8, bvar.n
constraint_mean = np.full((H, n), np.nan)
constraint_mean[:4, 0] = bvar.data[-1, 0]  # hold at last observed value

constraint_variance = np.full((H, n), np.nan)
constraint_variance[:4, 0] = 0.25**2  # 0.25 standard deviation

bvar.forecast(
    H=H,
    constraint_mean=constraint_mean,
    constraint_variance=constraint_variance,
    N_draws=5000,
)
bvar.plot_forecast(alpha=0.05)
```

## References

- Antolín-Díaz, J., Petrella, I. and Rubio-Ramírez, J. F. (2021). [Structural scenario analysis with SVARs](https://doi.org/10.1016/j.jmoneco.2020.06.001). *Journal of Monetary Economics*, 117, 798–815.
- Cascaldi-Garcia, D. (2022). [Pandemic priors](https://www.federalreserve.gov/econres/ifdp/files/ifdp1352.pdf). *International Finance Discussion Papers*, 1352.
- Chan, J. C. C. (2020). [Large Bayesian vector autoregressions](https://link.springer.com/10.1007/978-3-030-31150-6_4). In P. Fuleky (Ed.), *Macroeconomic Forecasting in the Era of Big Data* (pp. 95–125). Springer.
- Doan, T., Litterman, R. and Sims, C. (1984). Forecasting and conditional projection using realistic prior distributions. *Econometric Reviews*, 3(1), 1–100.
- Giannone, D., Lenza, M. and Primiceri, G. E. (2015). [Prior selection for vector autoregressions](https://doi.org/10.1162/REST_a_00483). *Review of Economics and Statistics*, 97(2), 436–451.
- Sims, C. A. (1993). A nine-variable probabilistic macroeconomic forecasting model. In J. H. Stock and M. W. Watson (Eds.), *Business Cycles, Indicators, and Forecasting* (pp. 179–212). University of Chicago Press.
- Waggoner, D. F. and Zha, T. (1999). [Conditional forecasts in dynamic multivariate models](https://www.jstor.org/stable/2646713). *Review of Economics and Statistics*, 81(4), 639–651.
