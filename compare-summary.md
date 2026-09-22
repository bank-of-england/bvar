# Condensed Conjugate BVAR Comparison

GLP means Giannone, Lenza and Primiceri, *Prior Selection for Vector Autoregressions* (2015).
As requested, the supplied `GLPreplicationWeb` is treated as that paper's replication package.
The paper column uses the author's October 2013 manuscript, sections 3-4; final typeset wording was not checked.
Compare default level priors with Minnesota, SOC and SUR enabled; $n$ is the variable count and $p$ the lag count.
The added column covers Danilo Cascaldi-Garcia's **GLP extension**, not his separate baseline or asymmetric-prior implementations.
The repository cites his *Pandemic Priors* (2022); the inspected extension is dated November 2023.

| Area | Python (`NaturalConjugate`) | Bank MATLAB | GLP paper | GLP replication package | Cascaldi-Garcia: GLP + Pandemic Priors |
| --- | --- | --- | --- | --- | --- |
| Model family | Conjugate normal-inverse-Wishart | Same | Same | Same | Same, with pandemic regressors |
| Base IW degrees of freedom | $n+2$ | $n+2$ | $n+2$ | $n+2$ | $n+2$ |
| Base prior covariance mean | Fixed at AR(1) residual MSE | Fixed at AR(4) residual MSE | Diagonal scales inferred | Scales optimised by default (`MNpsi=1`) | Scales optimised (`MNpsi=1`) |
| Role of AR(1) scales | Sets prior scales; optional COVID dummies | Uses AR(4), no COVID adjustment | Fixed AR(1) scales only in simplified comparison | Initial values and bounds; fixed if `MNpsi=0` | Same as GLP code; no COVID dummies in AR regression |
| Intercept variance multiplier | 10 | 10,000 | Not verified | 10,000,000 (`Vc=10e6`) | 10,000,000 |
| SOC/SUR reference means | First $p$ post-lag rows | Whole prior-construction sample | First $p$ original observations | First $p$ original observations | First $p$ original observations |
| Tightness hyperprior mode / SD | 0.2 / 0.4 | 0.5 / 1.0 | 0.2 / 0.4 | 0.2 / 0.4 | 0.2 / 0.4 |
| SOC/SUR hyperprior mode / SD | 1.0 / 1.0 | 1.0 / 4.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |
| Lag decay | Estimated; Gamma prior | Estimated; Gamma prior | Fixed exponent 2 | Fixed at 2; optional estimation (`MNalpha=1`) | Same as GLP code |
| Covariance-scale hyperprior | None: fixed scales | None in default path | Inverse-Gamma, shape and scale $0.02^2$ | Same when `MNpsi=1` | Same as GLP code |
| Hyperparameter uncertainty | Plug-in optimum | Plug-in optimum | Integrated in predictive densities | Metropolis-Hastings available; `mcmc=0` by default | Example enables MCMC, including `phi`; function default is off |
| Optimisation | SciPy BFGS; softplus | `fminunc`; exponential | Numerical routine not verified | Sims's `csminwel`; bounded logistic transform | Same routine and transform; tighter stopping tolerance |
| Optimisation domain | Positive, no finite upper bounds | Positive, no finite upper bounds | Numerical bounds not verified | Tightness $(0.0001,5)$; SOC/SUR $(0.0001,50)$ | Same, plus `phi` $(0.0001,5)$ |
| COVID regressors | Optional outlier dummies | Outlier dummies | Not part of this paper | None in this package | One per selected period; example uses March-August 2020 |
| COVID variance multiplier | Fixed default 10,000 | Fixed 10,000 | Not applicable | Not applicable | $1/\phi$; one shared, estimated precision |
| COVID hyperprior | None | None | Not applicable | Not applicable | Gamma on `phi`: mode 0.2, SD 1 |
| Scenario conditioning | Gibbs updates using simulated futures | Historical-posterior mixture; no scenario reweighting | Not part of the forecasting exercise compared | Unconditional forecasts in `bvarGLP` | Unconditional forecasts; future dummy values zero |
| Companion-root rejection | None | Reject modulus above 1.5 | Not verified | None in the parameter-draw routine | None in the parameter-draw routine |

**Additional Python capabilities.**

- **Independent NIW model.** `IndependentNIW` uses a full $(nk \times nk)$ coefficient precision matrix, `c2` cross-variable shrinkage, and Gibbs sampling with burn-in. It does not support marginal-likelihood or cross-validation optimisation.
- **Levels and differences.** The public API records whether each series is in levels, logs, differences, or log-differences. The `stationary` setting controls Minnesota prior means, and forecast and GIRF outputs can be reconstructed in the original scale.
- **Cross-validation.** Expanding-window predictive likelihood tunes prior hyperparameters for a chosen horizon and target series.
- **JAX optimisation.** An optional JAX backend supplies automatic gradients to SciPy BFGS; the NumPy/SciPy implementation remains available as a fallback.
- **Generalised IRFs.** The package computes order-invariant Pesaran-Shin responses with posterior uncertainty and output in the original data scale.
- **Skewed constraints.** Conditional forecasts support skew-normal constraint distributions for asymmetric upside and downside risks.
- **Nowcasting uncertainty.** Soft constraints can carry uncertainty from an external nowcast into the forecast distribution.

**Base prior.** IW quantities above precede SOC/SUR. Under the stated level priors, these add $n+1$ degrees of freedom and multiply the base covariance mean by $1/(n+2)$ in all implementations.
**Scale details.** Replication AR(1) regressions use post-lag data and residual degrees of freedom; Python and Bank routines use MSE.
Replication scale bounds are AR(1) estimates divided/multiplied by 100; optional lag-decay bounds are $(0.1,5)$. GLP calibrates the covariance-scale hyperprior for annualised log data ($4\times\log$); it is not invariant to rescaling the data.
**Key COVID difference:** fixed variance 10,000 corresponds to $\phi=10^{-4}$, the extension's lower optimisation bound, not its estimated value.

**Five-observation detail.** In the Bank posterior-prior path, `get_GLP_prior_dummy(Y(1:end-5,:),...)` drops five observations before constructing the prior. The truncated sample determines the AR(4) variance estimate and the SOC/SUR reference mean. The Bank still uses the full estimation sample in the ML and posterior likelihoods. The Bank can initialise forecasts after the estimation end; Python and GLP's driver use fitted history.

**Forecasts.** Python's default Gaussian conditioning and Bank's exported `YfcZV` agree for matched parameters, history and constraints, up to numerical tolerances; `YfcCF` instead retains unconditional covariance. Their parameter mixtures differ, and Python's conditional Gibbs loop needs burn-in.
Python uses the covariance posterior mean; Bank `Sig` is neither that mean nor the IW mode.
GLP code returns the IW covariance mode. Bank `yuc_mode` adds random shocks; Python's point-only forecast and GLP's `postmax.forecast` are deterministic.
The paper instead evaluates point forecasts as predictive medians, integrating hyperparameter uncertainty.

Latest verification: 118 focused Python tests passed, plus synthetic checks of the combined-prior mean and conditioning covariance. MATLAB columns are source comparisons; no MATLAB implementation was executed.
## References
- Python: [bvar repository](https://github.com/bank-of-england/bvar); [detailed Python/Bank comparison](compare.md).
- Bank MATLAB: [local repository](../../ma-BVARproject/); [active driver](../../ma-BVARproject/Step_1_Run_the_BVAR.m).
- GLP (2015), *Prior Selection for Vector Autoregressions*: [author's October 2013 manuscript](https://faculty.wcas.northwestern.edu/gep575/Draft_GLP_V24.pdf).
- GLP replication: [repository/package](https://bank-of-england-internal.ghe.com/bankwide/ma-bvar-lenza-primiceri/tree/main/GLPreplicationWeb); [defaults](../../ma-bvar-lenza-primiceri/GLPreplicationWeb/setpriors.m), [driver](../../ma-bvar-lenza-primiceri/GLPreplicationWeb/bvarGLP.m), [likelihood](../../ma-bvar-lenza-primiceri/GLPreplicationWeb/logMLVAR_formin.m), [posterior draws](../../ma-bvar-lenza-primiceri/GLPreplicationWeb/logMLVAR_formcmc.m), [AR variance](../../ma-bvar-lenza-primiceri/GLPreplicationWeb/subroutines/ols1.m).
- Cascaldi-Garcia (2022), *Pandemic Priors*: [paper page](https://www.federalreserve.gov/econres/ifdp/pandemic-priors.htm); [PDF](https://www.federalreserve.gov/econres/ifdp/files/ifdp1352.pdf).
- Cascaldi-Garcia replication: [repository](https://github.com/dcascaldi/pandemic_priors); [GLP extension at b5b56e3](https://github.com/dcascaldi/pandemic_priors/tree/b5b56e3e3468f7f227f043fe08054a37619222aa/Other_implementations/GLP-Pandemic%20Priors), including `ExamplePandemicPriors`, `setpriors_pp`, `bvarGLP_pp`, `logMLVAR_formin_pp` and `logMLVAR_formcmc_pp`.