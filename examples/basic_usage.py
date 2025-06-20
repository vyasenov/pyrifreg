"""
Basic usage example of pyrifreg package.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyrifreg import RIFRegression

# Generate sample data
np.random.seed(1988)
n = 1000

# Generate features
X = np.random.randn(n, 2)
# Generate target with some non-linear relationship and heteroskedasticity
y = 10 + 2 * X[:, 0] + np.sin(X[:, 1]) + np.random.randn(n) * (0.5 + 0.3 * np.abs(X[:, 0]))

#############
############# MEAN RIF REGRESSION
#############

print("=" * 80)
print("BASIC RIF REGRESSION EXAMPLES")
print("=" * 80)

# Create and fit mean RIF regression with default HC1 robust standard errors
mean_rif = RIFRegression(statistic='mean')
mean_rif.fit(X, y)
print("\nMean RIF Regression Results (HC1 robust standard errors):")
print(mean_rif.summary())

#############
############# MEDIAN RIF REGRESSION
#############

# Create and fit quantile RIF regression (median) with bootstrap
median_rif = RIFRegression(statistic='quantile', q=0.5, cov_type='bootstrap', bootstrap_reps=100)
median_rif.fit(X, y)
print("\nMedian RIF Regression Results (Bootstrap standard errors):")
print(median_rif.summary())

#############
############# VARIANCE RIF REGRESSION
#############

# Create and fit variance RIF regression with HC3 robust standard errors
var_rif = RIFRegression(statistic='variance', cov_type='HC3')
var_rif.fit(X, y)
print("\nVariance RIF Regression Results (HC3 robust standard errors):")
print(var_rif.summary())

#############
############# GINI COEFFICIENT RIF REGRESSION
#############

# Create and fit Gini coefficient RIF regression with bootstrap (more replications)
gini_rif = RIFRegression(statistic='gini', cov_type='bootstrap', bootstrap_reps=100)
gini_rif.fit(X, y)
print("\nGini Coefficient RIF Regression Results (Bootstrap standard errors):")
print(gini_rif.summary())

#############
############# IQR RIF REGRESSION
#############

# Create and fit IQR RIF regression with homoskedastic standard errors
iqr_rif = RIFRegression(statistic='iqr', cov_type='nonrobust')
iqr_rif.fit(X, y)
print("\nIQR RIF Regression Results (Homoskedastic standard errors):")
print(iqr_rif.summary())

#############
############# ENTROPY RIF REGRESSION
#############

# Create and fit entropy RIF regression with HC2 robust standard errors
entropy_rif = RIFRegression(statistic='entropy', cov_type='HC2')
entropy_rif.fit(X, y)
print("\nEntropy RIF Regression Results (HC2 robust standard errors):")
print(entropy_rif.summary())

#############
############# QUANTILE RIF REGRESSION
#############

# Define deciles
deciles = np.arange(0.1, 1.0, 0.1)

# Store results
coefficients = []
std_errors = []

print("Running quantile RIF regression at each decile...")

# Run quantile RIF regression at each decile
for q in deciles:
    print(f"  Processing {q*100:.0f}th percentile...")
    
    try:
        # Fit quantile RIF regression
        rif_model = RIFRegression(statistic='quantile', q=q, cov_type='HC1')
        rif_model.fit(X, y)
        
        # Extract X1 coefficient and standard error (skip intercept)
        coef_x1 = rif_model.results.params[1]
        se_x1 = rif_model.results.bse[1]
        
        coefficients.append(coef_x1)
        std_errors.append(se_x1)
        
    except Exception as e:
        print(f"    Error at q={q}: {str(e)}")
        coefficients.append(np.nan)
        std_errors.append(np.nan)

# Create the plot
plt.figure(figsize=(10, 6))

# Remove NaN values
mask = ~np.isnan(coefficients)
x_clean = deciles[mask]
coefs_clean = np.array(coefficients)[mask]
ses_clean = np.array(std_errors)[mask]

if len(x_clean) > 0:
    # Plot coefficient line
    plt.plot(x_clean, coefs_clean, 'o-', color='#1f77b4', linewidth=2, markersize=6, label='X1 Coefficient')
    
    # Plot confidence intervals (95% CI)
    ci_95 = 1.96 * ses_clean
    plt.fill_between(x_clean, coefs_clean - ci_95, coefs_clean + ci_95, 
                    alpha=0.3, color='#1f77b4', label='95% CI')

# Add horizontal line at zero for reference
plt.axhline(y=2, color='black', linestyle='--', alpha=0.5, linewidth=1)

# Customize plot
plt.xlabel('Quantile', fontsize=12)
plt.ylabel('X1 Coefficient Value', fontsize=12)
plt.title('X1 Coefficient Across Quantiles', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Set x-axis ticks
plt.xticks(deciles, [f'{d:.1f}' for d in deciles])

plt.tight_layout()
plt.show()
