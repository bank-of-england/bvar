# Skewed Constraints for Conditional Forecasting

Paul Labonne, Andrea Renzetti

## Introduction

This document describes conditional forecasting with skewed (non-normal) constraint distributions in the BVAR model. The approach lets users impose distributional constraints on selected variables while accounting for parameter and shock uncertainty.

## Setup

The conditional forecast is constructed as:

$$y = b + M^T \varepsilon$$

where:
- $y$ is the $(Hn \times 1)$ forecast vector over $H$ steps
- $b$ is the $(Hn \times 1)$ deterministic component (from constant and lagged values)
- $M$ is the $(Hn \times Hn)$ block lower-triangular impulse response matrix
- $\varepsilon$ is the $(Hn \times 1)$ vector of structural shocks

We impose distributional constraints:

$$Cy \sim SN_{\text{DP}}(f, \Sigma_f, \alpha)$$

where:
- $C$ is the $(m \times Hn)$ selection matrix (m = number of constraints)
- $SN_{\text{DP}}(\cdot)$ is the Skew Normal distribution (Azzalini & Capitanio, 1999, Section 4.1)
- $f$ is the location vector (constraint mean)
- $\Sigma_f$ is the scale matrix (constraint covariance)
- $\alpha$ is the shape vector (constraint skewness)

---

## Algorithm: Shock Adjustment with Constraint Sampling

The implemented method, which `forecast()` uses by default, avoids numerical instability and reduces computation time.

### Key Idea

We decompose the structural shocks into an unconstrained component and a constraint-satisfying adjustment:

$$\varepsilon = \xi + \psi$$

where $\xi \sim N(0, I_{Hn})$ represents unconstrained shocks and $\psi$ is the adjustment ensuring $Cy = z$ exactly.

### Four-Step Algorithm

**Step 1: Draw unconstrained shocks**

$$\xi \sim N(0, I_{Hn})$$

**Step 2: Draw constraint targets from the constraint distribution**

$$z \sim SN_{\text{DP}}(f - Cb, \Sigma_f, \alpha)$$

where $z$ represents the target value for $Cy - Cb = D\varepsilon$ (with $D = CM^T$).

**Step 3: Adjust shocks to satisfy constraints**

$$\varepsilon = \xi + D^{*}(z - D\xi)$$

where $D^{*} = (D^T D)^{-1}D^T$ is the Moore–Penrose inverse of $D$.

**Step 4: Compute the forecast**

$$y = b + M^T\varepsilon$$

### Mathematical Intuition

Starting from the constraint requirement $Cy = z$:

$$C(b + M^T\varepsilon) = z$$

$$CM^T\varepsilon = z - Cb$$

$$D\varepsilon = z - Cb$$

We decompose $\varepsilon = \xi + \psi$ and solve for $\psi$:

$$D\xi + D\psi = z - Cb$$

$$\psi = D^{*}\bigl[(z - Cb) - D\xi\bigr]$$

Thus:

$$\varepsilon = \xi + D^{*}(z - Cb - D\xi) = \xi + D^{*}(z - Cb) - D^{*}D\xi$$

Key property: $(I - D^{*}D)$ projects onto the null space of $D$, so unconstrained directions in $\xi$ only affect the forecast through shock directions orthogonal to the constraints.

---

## Implementation Paths

### Path 1: Conditional Mean Only (`point_only=True`)

Computes the conditional forecast mean without sampling:

$$\mathbb{E}[\varepsilon] = D^{*}(f - Cb)$$

$$y_{\text{mean}} = b + M^T \mathbb{E}[\varepsilon]$$

Use this for fast point estimates when uncertainty quantification is not needed.

### Path 2: Default Constraint Sampling

If no custom sampler is provided, constraint targets are drawn from a Skew Normal distribution:

$$z_i \sim SN(f_i - (Cb)_i, \sqrt{(\Sigma_f)_{ii}}, \alpha_i) \quad \text{for each constraint } i$$

When all shape parameters satisfy $\alpha_i = 0$, the distribution becomes Normal. The method supports independent distributions for each constraint.

### Path 3: Custom Constraint Sampler

Users can provide a `constraint_sampler` callable that returns draws from an arbitrary constraint distribution:

```python
def my_constraint_sampler():
    # User-defined sampling logic
    return constraint_draw  # Shape: (n_constraints,)


forecast = bvar.forecast(
    H=8,
    constraint_mean=constraint_mean,
    constraint_variance=constraint_variance,
    constraint_sampler=my_constraint_sampler,
)
```

This enables flexible specifications such as truncated distributions or mixtures, with the constraint values automatically transformed internally to $z - Cb$ for the shock constraint.

---

## Constraint Specification

Constraints are provided to the `forecast()` method as three $(H \times n)$ arrays:

```python
constraint_mean = np.full((H, n), np.nan)  # NaNs for unconstrained
constraint_mean[0:2, 0] = 2.5  # Fix variable 0 to 2.5 for steps 0–1

constraint_variance = np.full((H, n), np.nan)
constraint_variance[:, 0] = 0.5  # Allow ±0.71 standard deviation for variable 0

constraint_shape = np.full((H, n), np.nan)
constraint_shape[:, 0] = 2.0  # Right-skewed constraint for variable 0
```

The `get_constraint()` function internally converts these arrays into:
- Selection matrix $C$ (shape: $m \times Hn$)
- Location vector $f$ (shape: $m$)
- Scale matrix $\Sigma_f$ (shape: $m \times m$, diagonal)
- Shape vector $\alpha$ (shape: $m$)

where $m$ is the number of non-NaN constraint entries.

---

## Gibbs Sampler for Parameter Uncertainty

To account for uncertainty in both the VAR parameters and the constraint distribution, the conditional forecasting procedure embeds the shock adjustment algorithm within a Gibbs sampler:

**Initialization:** Use posterior point estimates $\beta_{\text{point}}$ and
$\Sigma_{\text{point}}$.

**For each MCMC iteration $i = 1, \ldots, N_{\text{draws}}$:**

1. Draw constrained forecast $y^{(i)}$ using the shock adjustment algorithm above
2. Augment the dataset with the conditional forecast: $y^{(i)}_{\text{data}} = [y_{\text{obs}}, y^{(i)}]$
3. Re-estimate the BVAR on the augmented data using its posterior sampler with the full state to obtain posterior draws $(\beta^{(i)}, \Sigma^{(i)})$
4. Use $(\beta^{(i)}, \Sigma^{(i)})$ for the next iteration

**Post-processing:** Discard the first $N_{\text{burn}}$ draws (default: $N_{\text{draws}} / 2$).

This ensures internal consistency: the conditional forecast paths follow the BVAR dynamics while propagating parameter and forecast uncertainty.

---

## Alternative Algorithms (Reference)

### Method 1: Affine Transformation of Skew Normal (Antolín-Díaz et al., 2021)

If $X \sim SN_{\text{DP}}(\mu, \Omega, \alpha)$, then for any matrix $A$ and vector $c$:

$$Y = AX + c \sim SN_{\text{DP}}(A\mu + c, A\Omega A^T, \alpha^*)$$

where the shape parameter transforms according to equation (10) of Azzalini & Capitanio (1999).

**Drawback:** Computing the posterior shock covariance $\Omega_\varepsilon = D^{*}\Sigma_f(D^{*})^T + I - D^{*}DD^T(D^{*})^T$ can produce non-positive-definite matrices when variance constraints are very tight, causing numerical instability.

### Method 2: Null Space Decomposition (Andersson, Palmqvist & Waggoner, 2010)

Decomposes the shock vector into constrained and unconstrained components using the null space of $D$:

$$y = M^T[D^* z + \hat{D} z_2]$$

where $\hat{D}$ is an orthonormal basis for $\text{null}(D)$.

**Drawback:** Requires explicit null space computation at each iteration, which is slower than the shock adjustment method.

---

## Usage Example

```python
import numpy as np
import bvar as bv

# Estimate model
bvar = bv.BVAR(n_lags=2, model=bv.NaturalConjugate(minnesota=True), stationary=False)
bvar.sample(data, N_draws=5000)

# Set up constraints: fix GDP growth at 2% for next 2 quarters
H = 8
constraint_mean = np.full((H, bvar.n), np.nan)
constraint_mean[0:2, 0] = 2.0  # GDP index

constraint_variance = np.full((H, bvar.n), np.nan)
constraint_variance[0:2, 0] = 0.25  # Tight around the mean

constraint_shape = np.full((H, bvar.n), np.nan)
constraint_shape[0:2, 0] = 0.0  # Normal distribution (no skew)

# Generate conditional forecast
bvar.forecast(
    H=H,
    constraint_mean=constraint_mean,
    constraint_variance=constraint_variance,
    constraint_shape=constraint_shape,
    method="andersson_et_al",
    N_draws=5000,
    progressbar=True,
)

# Extract results
y_cond = bvar.forecast_conditional  # Shape: (5000-2500, T+H, n) after burn-in
```

---

## References

- Andersson, M. K., Palmqvist, S., & Waggoner, D. F. (2010). [Density-conditional forecasts in dynamic multivariate models](https://www.econstor.eu/handle/10419/83026). *Sveriges Riksbank Working Paper Series*, 243.
- Antolín-Díaz, J., Petrella, I., & Rubio-Ramírez, J. F. (2021). [Structural scenario analysis with SVARs](https://doi.org/10.1016/j.jmoneco.2020.06.001). *Journal of Monetary Economics*, 117, 798–815.
- Azzalini, A., & Capitanio, A. (1999). [Statistical applications of the multivariate skew normal distribution](https://doi.org/10.1111/1467-9868.00213). *Journal of the Royal Statistical Society: Series B*, 61(3), 579–602.
- Waggoner, D. F., & Zha, T. (1999). [Conditional forecasts in dynamic multivariate models](https://www.jstor.org/stable/2646713). *Review of Economics and Statistics*, 81(4), 639–651.