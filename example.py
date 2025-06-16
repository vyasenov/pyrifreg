import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import statsmodels.api as sm

def compute_rif_quantile(y, tau, bw='scott'):
    # 1) estimate q_tau
    q_tau = np.quantile(y, tau)
    # 2) estimate density at q_tau
    kde = gaussian_kde(y, bw_method=bw)
    f_q = kde.evaluate(q_tau)[0]
    # 3) compute RIF
    return q_tau + (tau - (y <= q_tau).astype(float)) / f_q

# Example usage:
df = pd.DataFrame({...})
y = df['wage'].values
X = sm.add_constant(df[['union', 'education', ...]])

# Get RIF for the median
rif_y = compute_rif_quantile(y, tau=0.5)

# OLS
model = sm.OLS(rif_y, X).fit(cov_type='HC3')  # robust se
print(model.summary())
