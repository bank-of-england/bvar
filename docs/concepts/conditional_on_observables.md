## Conditioning Approaches: Observables vs. Shocks

Conditional forecasting and scenario analysis distinguish between **conditioning on observables** and **conditioning on shocks**.

### Conditioning on Observables

This is the approach implemented when using reduced-form VARs where the contemporaneous relationship between variables is not estimated.

- **Definition:**  
  Conditioning on observables means imposing constraints directly on the future paths of observed variables (e.g., GDP, inflation). For example, you might require that inflation follows a specific trajectory over the forecast horizon.
- **Implementation:**  
  This is typically done by specifying a set of linear restrictions on the forecasted values of the endogenous variables.
- **Interpretation:**  
  The model finds the distribution of shocks and parameter draws that are consistent with the imposed path for the observables, without specifying which shocks are responsible for achieving the scenario.
- **Use case:**  
  Useful for scenario analysis where the focus is on the outcome (e.g., "What if inflation is 2% next year?"), regardless of the underlying structural drivers.

### Conditioning on Shocks

Reduced-form VARs cannot impose this restriction because the analyst must estimate the contemporaneous relationship between variables and their causes.

- **Definition:**  
  Conditioning on shocks means specifying the path of one or more structural shocks (e.g., a monetary policy shock, a supply shock) over the forecast horizon.
- **Implementation:**  
  The model imposes constraints on the sequence of structural shocks, and then computes the implied path for the observables. This requires identification of the structural shocks (e.g., via SVARs).
- **Interpretation:**  
  The resulting forecast shows the evolution of observables that would occur if the specified shocks materialize, holding all other shocks at their typical (zero) values.
- **Use case:**  
  Useful for policy analysis or counterfactuals (e.g., "What would happen to output and inflation after a sequence of negative supply shocks?").

### Key Difference

- **Condition-on-observables** answers:  
  *"What happens if the interest rate is at the ZLB for the next year?"*
- **Condition-on-shocks** answers:  
  *"What happens if a monetary policy shock brings the interest rate to the ZLB for the next year?"*

**Reference:**  
Antolín-Díaz, J., Petrella, I., & Rubio-Ramírez, J. F. (2021). [Structural scenario analysis with SVARs](https://doi.org/10.1016/j.jmoneco.2020.06.001).