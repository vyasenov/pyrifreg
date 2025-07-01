# pyrifreg

A Python package for Recentered Influence Function (RIF) regression analysis. Provides tools for analyzing distributional effects in econometrics and data science applications. Bridges the gap between Python developers and econometricians to enable deeper unconditional distributional analysis.

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

## Background

#### Motivation: From Conditional to Unconditional Effects

Most of you are familiar with conditional moments—e.g.

$$
\mathbb{E}[Y \mid X = x]
$$

or conditional quantiles

$$
Q_\tau(Y \mid X = x).
$$

But policy questions often concern how a change in some covariate $X$ shifts the *entire* (marginal or unconditional) distribution of an outcome $Y$.  For instance:

* Inequality analysis:  How would increasing education change the 90th vs.\ 10th percentile of the wage distribution?
* Welfare evaluation:  What is the impact of a cash transfer on the variance or Gini of consumption?

Formally, let $F_Y$ be the baseline distribution of $Y$, and imagine a small intervention on $X$ that perturbs $F_Y$ to $G_Y$.  For a scalar functional $\nu(\cdot)$ (e.g. mean, variance, quantile), define the *unconditional effect*:

$$
\Delta\nu =\nu(G_Y)-\nu(F_Y).
$$

Our goal is to estimate how “marginal shifts” in $X$ translate into $\Delta\nu$.

---

#### Influence Functions (IF)

An influence function captures the *first‐order* sensitivity of a distributional functional $T(F)$ to an infinitesimal contamination at the point $y$.  Concretely, define

$$
F_\varepsilon = (1-\varepsilon)F + \varepsilon\,\delta_y,
$$

where $\delta_y$ is a point‐mass at $y$.  Then

$$
\mathrm{IF}(y,T, F)
=
\lim_{\varepsilon\to 0}
\frac{T(F_\varepsilon) - T(F)}{\varepsilon}.
$$

IFs tell us "how much does a single observation at $y$ “pull” the estimator of $T$ away from its nominal value".

---

#### Recentered Influence Functions (RIF)

Since $\mathbb{E}[\mathrm{IF}(Y;T,F)] = 0$, we cannot regress $\mathrm{IF}(Y)$ directly to target $T(F)$.  The recentered influence function adds back the functional itself:

$$
\mathrm{RIF}(y,T, F)
=
T(F)+\mathrm{IF}(y,T, F).
$$

Its key property is:

$$
\mathbb{E}[\mathrm{RIF}(Y)] = T(F).
$$

Thus $\mathrm{RIF}(Y)$ is an *unbiased* “pseudo‐outcome” for $T(F)$, which we can now relate to covariates.

---

#### RIF Regression

A RIF regression proceeds in two steps:

1. Compute  the plug‐in estimate $T(\widehat F)$ and the influence function $\mathrm{IF}(y_i;T,\widehat F)$ for each $i$.
2. Form the RIF outcome $r_i = T(\widehat F) + \mathrm{IF}(y_i,T,\widehat F),$ and estimate the linear model

   $$
     r_i = x_i^\top\beta +\varepsilon_i.
   $$

Under regularity conditions (smoothness of $T$, overlap in $X$, etc.), each component $\beta_j$ approximates the *marginal effect* of $X_j$ on the unconditional functional $T(F_Y)$.

---

#### Unconditional Quantile Regression (UQR)

Unconditional quantile regression is simply RIF regression with $T(F)=Q_\tau(Y)$.  Then:

$$
r_i = Q_\tau(Y) +\frac{\tau - \mathbf{1}\{y_i \le Q_\tau\}}{f_Y(Q_\tau)},
$$

and regressing $r_i$ on $X$ yields estimates of how a marginal change in each $X_j$ shifts the $\tau$-th *marginal* quantile of $Y$. 

Traditional conditional quantile regression (Koenker & Bassett, 1978) estimates how covariates $X$ shift the conditional quantile $Q_\tau(Y\mid X)$, which effectively amounts to examining the unconditional distribution of the residual $\varepsilon$. By contrast, unconditional quantile regression (UQR) assesses how marginal changes in $X$ directly alter the overall distribution of $Y$. Personally, I find the conditional approach far less interpretable and meaningful.


---

#### Inference

Inference in RIF regression proceeds via a two‐stage procedure. First,  estimating the target functional $T(\widehat F)$, any necessary density (e.g.\ $f_Y(Q_\tau)$), and the influence values $\mathrm{IF}(y_i)$. Next, regressing the recentered outcomes on covariates. Because the RIFs are themselves estimated, naïve OLS standard errors are inconsistent and must be adjusted. The package supports bootstrap estimation of standard errors.

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

## References

* Firpo, S., Fortin, N. M., & Lemieux, T. (2009). *Unconditional Quantile Regressions*. Econometrica, 77(3), 953–973.
* Hampel, F. R. (1974). *The Influence Curve and Its Role in Robust Estimation*. Journal of the American Statistical Association, 69(346), 383–393.
* Koenker, R., & Bassett Jr, G. (1978). Regression quantiles. Econometrica: journal of the Econometric Society, 33-50.
* Rios-Avila, F. (2020). *Recentered influence functions (RIFs) in Stata: RIF regression and RIF decomposition*. The Stata Journal, 20(1), 51-94.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.