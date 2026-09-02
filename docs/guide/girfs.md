# Generalised Impulse Response Functions

## What are IRFs?

Impulse Response Functions (IRFs) trace how a shock to one variable propagates through the system over time. They are the principal tool for studying the dynamic causal structure of a VAR.

## Standard IRFs and the Identification Problem

In a standard VAR, the residuals $\varepsilon_t$ are contemporaneously correlated. To interpret a shock as a structural event (e.g. a monetary policy tightening), one must first identify which combination of residuals corresponds to each structural shock. The most common method — Cholesky decomposition — is computationally straightforward but requires the researcher to impose a triangular causal ordering on the variables, which is often controversial.

## Generalised IRFs

Generalised Impulse Response Functions (GIRFs), proposed by Pesaran & Shin (1998), avoid the identification problem entirely. Rather than decomposing the residual covariance into structural shocks, GIRFs ask a simpler question: given a one-standard-deviation shock to variable $j$, how does the expected path of variable $i$ change relative to its baseline?

Formally, the GIRF for variable $i$ at horizon $h$ following a shock to variable $j$ is:

$$\text{GIRF}_{ij}(h) = \mathbb{E}[y_{i,t+h} \mid \varepsilon_{jt} = \sigma_{jj}^{1/2}] - \mathbb{E}[y_{i,t+h}]$$

where $\sigma_{jj}$ is the $j$-th diagonal element of $\Sigma$. The conditioning is on a shock of one standard deviation in variable $j$, with all other shocks set to their conditional expectation given that shock.

Because each GIRF conditions on the estimated covariance structure rather than an arbitrary orthogonalisation, the results are invariant to the ordering of variables in the VAR. This makes GIRFs particularly useful for exploratory analysis when the structural model is not fully specified.

## Interpretation

- A positive GIRF for variable $i$ following a shock to $j$ means that a positive innovation in $j$ is historically associated with a rise in $i$ after $h$ periods, once all contemporaneous and lagged co-movements are accounted for.
- GIRFs are asymmetric by variable: the response of $i$ to a shock in $j$ generally differs from the response of $j$ to a shock in $i$.
- The magnitude of the shock is normalised to one standard deviation, so responses across variables are directly comparable in percentage or level terms depending on the chosen `response_type`.

## Response Types

The `response_type` argument in `compute_girf` controls how the impulse response is expressed:

- `pct_change_yoy` — year-on-year percentage change
- `level_change` — absolute change in levels (suitable for rates such as unemployment or Bank Rate)

For `diff` and `log_diff` input data, the differenced observations do not contain
an absolute level. Pass `base_value` to `compute_girf()` when requesting
`level_change`, `pct_change`, `change_yoy`, or `pct_change_yoy`; it may be a
scalar or one value per variable. Omitting it raises `ValueError`. Input recorded as `logs` or
`log_levels` derives its baseline by exponentiating the last stored observation.
When `data_transformation` is omitted, GIRFs use the mapping supplied to
`sample()`. Explicitly passing it takes precedence. Frequency multipliers use
the complete-period floor for non-divisors, so `5M` represents 2 periods per year.

```python
import bvar as bv

data, _, _, _ = bv.simulate_var(T=200, n=4, n_lags=2, levels=True, seed=42)
data.columns = ["gdp", "cpi", "unemp", "rrate"]
model_vars = list(data.columns)
data_transformation = {
    "gdp": "logs",
    "cpi": "logs",
    "unemp": "levels",
    "rrate": "levels",
}

bvar = bv.BVAR(n_lags=2, model=bv.NaturalConjugate(), stationary=False, random_state=0)
bvar.optimise_hyperparameters(data)
bvar.sample(data, N_draws=2000)

# GIRFs: most variables in yoy pct change, rates in level change
response_types = {var: "pct_change_yoy" for var in model_vars}
response_types["rrate"] = "level_change"
response_types["unemp"] = "level_change"

bvar.compute_girf(
    H=12,
    N_draws=2000,
    data_transformation=data_transformation,
    response_type=response_types,
)

bvar.plot_girf(shock_var="rrate", response_var=["gdp", "cpi", "unemp"])
```

## Uncertainty Bands

Because the GIRF is computed over posterior draws, the package produces full posterior distributions for each response. The `plot_girf` method displays the median response and associated uncertainty bands, giving a sense of the precision with which the dynamic co-movements are estimated.

## References

| Paper | Contribution |
|---|---|
| [Pesaran & Shin (1998)](https://doi.org/10.1016/S0165-1765(97)00214-0) | Generalised IRF methodology |
