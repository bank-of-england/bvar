# Explicitly accounting for nowcasting uncertainty in BVARs

## Problem
Nowcasts carry substantial uncertainty, but the current workflow treats them as
hard data and adds them to the dataset for BVAR estimation and forecasting.
The model therefore gives nowcasts the same weight as actual observations.
Representing that uncertainty should improve forecast performance.

## Solution
We represent a nowcast as a random variable with a prior distribution rather than as data.

The prior distribution of the nowcast is then given by:

$$f_t \sim N(\hat{f}_t, \hat{\sigma}^2)$$

where $\hat{f}_t$ is the point estimate of the nowcast and $\hat{\sigma}^2$ its estimated variance.

## Implementation
In practice, this prior matches a conditional forecasting exercise in which the
constraint $\hat{f}_t$ has uncertainty $\sigma^2$, the variance of the
historical nowcast error. Antolín-Díaz, Petrella & Rubio-Ramírez (2021, Section 2.2)
describe random (soft) constraints; Andersson, Palmqvist & Waggoner (2010) present
the underlying density-conditioning idea.

## Summary
We treat a nowcast as a constraint in the form of a random variable. Its mean
equals the point nowcast adjusted for bias and inefficiency, and its variance
equals the nowcasting error variance.

## Usage Example

```python
import numpy as np
import bvar as bv

# Suppose we have a fitted BVAR with n variables
bvar = bv.BVAR(n_lags=2, model=bv.NaturalConjugate(), stationary=False, random_state=42)
bvar.optimise_hyperparameters(data)
bvar.sample(data, N_draws=5000)

H = 8
n = bvar.n

# The nowcast for variable 0 at t+1 is 2.1 with historical RMSE of 0.3
nowcast_mean = 2.1
nowcast_variance = 0.3**2  # squared RMSE

# Encode as a soft constraint on the first forecast step
constraint_mean = np.full((H, n), np.nan)
constraint_mean[0, 0] = nowcast_mean

constraint_variance = np.full((H, n), np.nan)
constraint_variance[0, 0] = nowcast_variance

# The forecast now treats the nowcast as uncertain rather than fixed
bvar.forecast(
    H=H,
    constraint_mean=constraint_mean,
    constraint_variance=constraint_variance,
    N_draws=5000,
)
bvar.plot_forecast(alpha=0.05)
```

## References

- Andersson, M. K., Palmqvist, S., & Waggoner, D. F. (2010). [Density-conditional forecasts in dynamic multivariate models](https://www.econstor.eu/handle/10419/83026). *Sveriges Riksbank Working Paper Series*, 243.
- Antolín-Díaz, J., Petrella, I., & Rubio-Ramírez, J. F. (2021). [Structural scenario analysis with SVARs](https://doi.org/10.1016/j.jmoneco.2020.06.001). *Journal of Monetary Economics*, 117, 798-815.