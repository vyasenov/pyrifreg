"""
Basic usage example of pyrifreg package.
"""

import numpy as np
import pandas as pd
from pyrifreg import RIFRegression

# Generate sample data
np.random.seed(42)
n_samples = 1000

# Generate features
X = np.random.randn(n_samples, 2)
# Generate target with some non-linear relationship and heteroskedasticity
y = 2 * X[:, 0] + np.sin(X[:, 1]) + np.random.randn(n_samples) * (0.5 + 0.3 * np.abs(X[:, 0]))

print("=" * 80)
print("BASIC RIF REGRESSION EXAMPLES")
print("=" * 80)

# Create and fit mean RIF regression with default HC1 robust standard errors
mean_rif = RIFRegression(statistic='mean')
mean_rif.fit(X, y)
print("\nMean RIF Regression Results (HC1 robust standard errors):")
print(mean_rif.summary())

# Create and fit quantile RIF regression (median) with bootstrap
median_rif = RIFRegression(statistic='quantile', q=0.5, cov_type='bootstrap', bootstrap_reps=500)
median_rif.fit(X, y)
print("\nMedian RIF Regression Results (Bootstrap standard errors):")
print(median_rif.summary())

# Get bootstrap information
bootstrap_info = median_rif.get_bootstrap_info()
if bootstrap_info:
    print(f"\nBootstrap Information:")
    print(f"  Total replications: {bootstrap_info['total_reps']}")
    print(f"  Successful replications: {bootstrap_info['successful_reps']}")
    print(f"  Success rate: {bootstrap_info['successful_reps']/bootstrap_info['total_reps']:.1%}")

# Create and fit variance RIF regression with HC3 robust standard errors
var_rif = RIFRegression(statistic='variance', cov_type='HC3')
var_rif.fit(X, y)
print("\nVariance RIF Regression Results (HC3 robust standard errors):")
print(var_rif.summary())

# Create and fit Gini coefficient RIF regression with bootstrap (more replications)
gini_rif = RIFRegression(statistic='gini', cov_type='bootstrap', bootstrap_reps=1000)
gini_rif.fit(X, y)
print("\nGini Coefficient RIF Regression Results (Bootstrap standard errors):")
print(gini_rif.summary())

# Get bootstrap information for Gini
gini_bootstrap_info = gini_rif.get_bootstrap_info()
if gini_bootstrap_info:
    print(f"\nGini Bootstrap Information:")
    print(f"  Total replications: {gini_bootstrap_info['total_reps']}")
    print(f"  Successful replications: {gini_bootstrap_info['successful_reps']}")
    print(f"  Success rate: {gini_bootstrap_info['successful_reps']/gini_bootstrap_info['total_reps']:.1%}")

# Create and fit IQR RIF regression with homoskedastic standard errors
iqr_rif = RIFRegression(statistic='iqr', cov_type='nonrobust')
iqr_rif.fit(X, y)
print("\nIQR RIF Regression Results (Homoskedastic standard errors):")
print(iqr_rif.summary())

# Create and fit entropy RIF regression with HC2 robust standard errors
entropy_rif = RIFRegression(statistic='entropy', cov_type='HC2')
entropy_rif.fit(X, y)
print("\nEntropy RIF Regression Results (HC2 robust standard errors):")
print(entropy_rif.summary())

print("\n" + "=" * 80)
print("COMPARISON OF DIFFERENT COVARIANCE TYPES")
print("=" * 80)

# Compare different covariance types for the same statistic
print("\nComparing different covariance types for Mean RIF regression:")

# Homoskedastic
mean_homo = RIFRegression(statistic='mean', cov_type='nonrobust')
mean_homo.fit(X, y)
print("\n1. Homoskedastic standard errors:")
print(mean_homo.summary())

# HC1 (default)
mean_hc1 = RIFRegression(statistic='mean', cov_type='HC1')
mean_hc1.fit(X, y)
print("\n2. HC1 robust standard errors (default):")
print(mean_hc1.summary())

# Bootstrap
mean_boot = RIFRegression(statistic='mean', cov_type='bootstrap', bootstrap_reps=300)
mean_boot.fit(X, y)
print("\n3. Bootstrap standard errors:")
print(mean_boot.summary())

# Get bootstrap info
boot_info = mean_boot.get_bootstrap_info()
if boot_info:
    print(f"\nBootstrap Details:")
    print(f"  Successful replications: {boot_info['successful_reps']}/{boot_info['total_reps']}")
    print(f"  Success rate: {boot_info['successful_reps']/boot_info['total_reps']:.1%}")

print("\n" + "=" * 80)
print("BOOTSTRAP WITH DIFFERENT REPLICATION COUNTS")
print("=" * 80)

# Demonstrate bootstrap with different replication counts
replication_counts = [100, 500, 1000]

for reps in replication_counts:
    print(f"\nBootstrap with {reps} replications:")
    try:
        boot_model = RIFRegression(statistic='quantile', q=0.75, 
                                  cov_type='bootstrap', bootstrap_reps=reps)
        boot_model.fit(X, y)
        
        # Get bootstrap info
        boot_info = boot_model.get_bootstrap_info()
        if boot_info:
            success_rate = boot_info['successful_reps'] / boot_info['total_reps']
            print(f"  Success rate: {success_rate:.1%}")
            print(f"  Successful replications: {boot_info['successful_reps']}")
        
        # Show coefficient estimates
        params = boot_model.results.params
        print(f"  Intercept: {params[0]:.4f}")
        print(f"  X1 coefficient: {params[1]:.4f}")
        print(f"  X2 coefficient: {params[2]:.4f}")
        
    except Exception as e:
        print(f"  Error: {str(e)}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("This example demonstrates:")
print("1. Different RIF statistics (mean, quantile, variance, gini, iqr, entropy)")
print("2. Various covariance types (nonrobust, HC1, HC2, HC3, bootstrap)")
print("3. Bootstrap with different replication counts")
print("4. How to access bootstrap information")
print("5. Comparison of standard error estimates across methods")