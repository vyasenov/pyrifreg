"""
Basic usage example of pyRifReg package.
"""

import numpy as np
import pandas as pd
from pyRifReg import RIFRegression

# Generate sample data
np.random.seed(42)
n_samples = 1000

# Generate features
X = np.random.randn(n_samples, 2)
# Generate target with some non-linear relationship
y = 2 * X[:, 0] + np.sin(X[:, 1]) + np.random.randn(n_samples) * 0.5

# Create and fit mean RIF regression
mean_rif = RIFRegression(statistic='mean')
mean_rif.fit(X, y)
print("\nMean RIF Regression Results:")
print(mean_rif.summary())

# Create and fit quantile RIF regression (median)
median_rif = RIFRegression(statistic='quantile', q=0.5)
median_rif.fit(X, y)
print("\nMedian RIF Regression Results:")
print(median_rif.summary())

# Create and fit variance RIF regression
var_rif = RIFRegression(statistic='variance')
var_rif.fit(X, y)
print("\nVariance RIF Regression Results:")
print(var_rif.summary())

# Make predictions
X_new = np.random.randn(5, 2)
predictions = mean_rif.predict(X_new)
print("\nPredictions for new data:")
print(predictions) 