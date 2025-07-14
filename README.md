# pyrifreg

![](https://img.shields.io/badge/license-MIT-green)

A Python package for Recentered Influence Function (RIF) regression analysis. Provides tools for analyzing distributional effects in econometrics and data science applications. 

## Installation

You can install the package using pip:

```bash
pip install pyrifreg
```

## Features

- Implementation of Recentered Influence Function (RIF) regression
- Support for various distributional statistics (mean, quantiles, variance, gini, etc.)
- Easy-to-use API for regression analysis
- Integration with pandas and scikit-learn


## Quick Start

```python
import numpy as np
import pandas as pd
from pyrifreg import RIFRegression

# Create sample data
X = np.random.randn(1000, 2)
y = np.random.randn(1000)

# Initialize and fit RIF regression
median_rif = RIFRegression(statistic='quantile', q=0.5)
median_rif.fit(X, y)

# Get regression results
results = median_rif.summary()
print(results)
```

You can find more examples in [example.py](https://github.com/vyasenov/pyrifreg/blob/main/example.py).

## Examples

You can find detailed usage examples in the  `examples/` directory.

## Background

### From Conditional to Unconditional Effects

Many regression models focus on *conditional* statistics like:

$$
\mathbb{E}[Y \mid X = x]
$$

or conditional quantiles

$$
Q_\tau(Y \mid X = x).
$$

But policy questions often require understanding how a variable like education or income influences the *entire* distribution of an outcome, not just its mean or conditional parts. For example:

* How would expanding access to college change the 90th percentile of the wage distribution?
* What is the effect of a tax policy on income inequality or the Gini index?

Instead of looking at changes within subgroups (conditional on $X$), RIF regression helps us estimate how changes in covariates shift the *overall*, or *unconditional*, distribution of $Y$.

Let $F_Y$ be the original distribution of $Y$, and suppose an intervention shifts it to $G_Y$. For a statistic $\nu$ (like the mean, a quantile, or variance), we want to estimate:

$$
\Delta\nu = \nu(G_Y) - \nu(F_Y),
$$

i.e., how that statistic changes when the distribution shifts. RIF regression provides a way to estimate how different variables contribute to such shifts.

### Influence Functions (IF)

The influence function measures how sensitive a statistic is to a small change in the data. More precisely, it tells us how much an individual observation $y$ influences a statistic like the mean or a quantile.

Formally, imagine a slightly perturbed distribution:

$$
F_\varepsilon = (1 - \varepsilon) F + \varepsilon\, \delta_y,
$$

where $\delta_y$ is a point mass at $y$. Then the influence function is:

$$
\mathrm{IF}(y; T, F) = \lim_{\varepsilon \to 0} \frac{T(F_\varepsilon) - T(F)}{\varepsilon}.
$$

This gives us a first-order approximation of how $y$ affects the statistic $T$.

### Recentered Influence Functions (RIF)

Because the average of the influence function is always zero, we can’t use it directly in a regression. To fix this, we “recenter” it by adding the original statistic back:

$$
\mathrm{RIF}(y; T, F) = T(F) + \mathrm{IF}(y; T, F).
$$

Now, the expected value of the RIF is equal to the statistic itself:

$$
\mathbb{E}[\mathrm{RIF}(Y)] = T(F).
$$

This makes it a useful outcome variable for regression, allowing us to relate changes in the statistic $T$ to changes in covariates.

### RIF Regression

RIF regression works in two main steps:

1. Estimate the target statistic $T(F)$ (e.g. median or Gini) and compute the influence value for each observation.
2. Construct the RIF pseudo-outcome for each data point and regress it on $X$ using linear regression:

   $$
   r_i = x_i^\top \beta + \varepsilon_i.
   $$

The regression coefficients $\beta_j$ can then be interpreted as the marginal effect of each $X_j$ on the statistic of interest.

### Unconditional Quantile Regression (UQR)

UQR is a special case of RIF regression, where the statistic of interest is an unconditional quantile $Q_\tau(Y)$. For each observation $y_i$, we compute:

$$
r_i = Q_\tau(Y) + \frac{\tau - \mathbf{1}\{y_i \le Q_\tau\}}{f_Y(Q_\tau)},
$$

where $f_Y(Q_\tau)$ is the density at the $\tau$-th quantile. Regressing $r_i$ on $X$ tells us how each covariate shifts the $\tau$-th quantile of the overall outcome distribution.

This is in contrast to conditional quantile regression (Koenker & Bassett, 1978), which examines changes in $Q_\tau(Y \mid X)$—a different and often less intuitive object for understanding broad policy effects.

### Confidence Intervals

Since RIFs are estimated in a first step before regression, the usual OLS standard errors are biased. To correct this, inference proceeds in two stages:

1. Estimate the statistic $T$, the influence function, and any needed density estimates.
2. Run the regression and compute corrected standard errors using bootstrap.

The package includes support for bootstrap inference out of the box.

## References

* Firpo, S., Fortin, N. M., & Lemieux, T. (2009). *Unconditional Quantile Regressions*. Econometrica, 77(3), 953–973.
* Koenker, R., & Bassett Jr, G. (1978). Regression quantiles. Econometrica: journal of the Econometric Society, 33-50.
* Rios-Avila, F. (2020). *Recentered influence functions (RIFs) in Stata: RIF regression and RIF decomposition*. The Stata Journal, 20(1), 51-94.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

To cite this package in publications, please use the following BibTeX entry:

```bibtex
@misc{yasenov2025pyrifreg,
  author       = {Vasco Yasenov},
  title        = {pyrifreg: Python Tools for Recentered Influence Function (RIF) Regression},
  year         = {2025},
  howpublished = {\url{https://github.com/vyasenov/pyrifreg}},
  note         = {Version 0.1.0}
}
```