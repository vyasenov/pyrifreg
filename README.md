# pyrifreg

ADD EXAMPLE OF QREG ALONG ENTIRE DISTRIBUTION
ADD BOOTSRAP SUPPORT
ADD NOTE ON INFERENCE AND IMPACT OF ALREADY ESTIMATED QUANTITIES

A Python package for Recentered Influence Function (RIF) regression analysis. This package provides tools for analyzing distributional effects in econometrics and data science applications.

## Installation

You can install the package using pip:

```bash
pip install pyrifreg
```

## Features

- Implementation of Recentered Influence Function (RIF) regression
- Support for various distributional statistics (mean, quantiles, variance, etc.)
- Easy-to-use API for regression analysis
- Integration with pandas and scikit-learn

## Background

### Influence Functions

An **influence function (IF)** characterizes the first‐order effect on a statistical functional $T(F)$ (e.g. a quantile or a variance) of an infinitesimal contamination at point $y$.  Formally, if $F$ is the true distribution and $F_\varepsilon = (1-\varepsilon)F + \varepsilon \delta_y$, then

$$
\mathrm{IF}(y; T, F)
\;=\;
\lim_{\varepsilon\to 0}\frac{T(F_\varepsilon) - T(F)}{\varepsilon}.
$$

Intuitively, $\mathrm{IF}(y)$ tells you how much an observation at $y$ “pulls” the estimator away from its nominal value.  Influence functions underpin robust statistics and are central to asymptotic variance calculations.

### Recentered Influence Functions (RIFs)

While $\mathrm{IF}(y)$ has mean zero under $F$, many applications (e.g. regression) require a variable whose expectation equals the functional of interest.  A **recentered influence function (RIF)** adds back the original functional:

$$
\mathrm{RIF}(y; T, F) \;=\; T(F) \;+\; \mathrm{IF}(y; T, F).
$$

Because $\mathbb{E}_F[\mathrm{RIF}(Y)] = T(F)$, one can regress the RIF on covariates to study how covariates shift the target functional.  This idea was popularized by Firpo, Fortin & Lemieux (2009) for decompositions of wage distributions and quantile regressions.

### RIF Regression

A **RIF regression** proceeds in two steps:

1. **Compute** $T(F)$ and the influence function $\mathrm{IF}(y_i; T, F)$ for each observation $i$.
2. **Form** the recentered outcome

   $$
     r_i = \mathrm{RIF}(y_i; T, F) = T(F) + \mathrm{IF}(y_i; T, F),
   $$

   and **fit** a linear (or nonlinear) model

   $$
     r_i = x_i^\top \beta \;+\;\varepsilon_i.
   $$

   Under mild regularity conditions, the estimated $\beta$ captures the marginal association between covariates $X$ and the target functional $T$, analogously to how OLS coefficients reveal shifts in the conditional mean.

**Key features**:

* **Flexibility**: Targets arbitrary distributional statistics—means, variances, quantiles, Gini coefficients, etc.
* **Interpretability**: Linear–model coefficients have a clear meaning in terms of shifting the population functional.
* **Inference**: Standard errors can be obtained via the “sandwich” variance formula, since the RIF regression is semiparametrically efficient for many functionals.


## Quick Start

```python
import numpy as np
import pandas as pd
from pyrifreg import RIFRegression

# Create sample data
X = np.random.randn(1000, 2)
y = np.random.randn(1000)

# Initialize and fit RIF regression
rif_reg = RIFRegression(statistic='mean')
rif_reg.fit(X, y)

# Get regression results
results = rif_reg.summary()

# For Gini coefficient
gini_rif = RIFRegression(statistic='gini')

# For IQR
iqr_rif = RIFRegression(statistic='iqr')

# For entropy
entropy_rif = RIFRegression(statistic='entropy')

# For quantiles (as before)
median_rif = RIFRegression(statistic='quantile', q=0.5)
```

## References

* Firpo, S., Fortin, N. M., & Lemieux, T. (2009). *Unconditional Quantile Regressions*. Econometrica, 77(3), 953–973.
* Hampel, F. R. (1974). *The Influence Curve and Its Role in Robust Estimation*. Journal of the American Statistical Association, 69(346), 383–393.
* Rios-Avila, F. (2020). *Recentered influence functions (RIFs) in Stata: RIF regression and RIF decomposition*. The Stata Journal, 20(1), 51-94.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
