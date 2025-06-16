"""
Implementation of Recentered Influence Function (RIF) regression.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


class RIFRegression(BaseEstimator, RegressorMixin):
    """
    Recentered Influence Function (RIF) Regression.
    
    This class implements RIF regression for various distributional statistics
    such as mean, quantiles, and variance.
    
    Parameters
    ----------
    statistic : str, default='mean'
        The distributional statistic to compute. Options are:
        - 'mean': Mean
        - 'quantile': Quantile (requires q parameter)
        - 'variance': Variance
    q : float, optional
        Quantile level (between 0 and 1) when statistic='quantile'
    """
    
    def __init__(self, statistic='mean', q=None):
        self.statistic = statistic
        self.q = q
        self.model = None
        self.results = None
        
    def _compute_rif(self, y):
        """Compute the Recentered Influence Function for the chosen statistic."""
        n = len(y)
        
        if self.statistic == 'mean':
            rif = y
        elif self.statistic == 'quantile':
            if self.q is None:
                raise ValueError("Quantile level 'q' must be specified for quantile RIF")
            q_val = np.quantile(y, self.q)
            f_q = stats.gaussian_kde(y)(q_val)
            rif = q_val + (self.q - (y <= q_val)) / f_q
        elif self.statistic == 'variance':
            mean_y = np.mean(y)
            var_y = np.var(y)
            rif = (y - mean_y)**2
        else:
            raise ValueError(f"Unknown statistic: {self.statistic}")
            
        return rif
    
    def fit(self, X, y):
        """
        Fit the RIF regression model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Returns self.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Compute RIF
        rif = self._compute_rif(y)
        
        # Add constant term
        X = add_constant(X)
        
        # Fit OLS regression
        self.model = OLS(rif, X)
        self.results = self.model.fit()
        
        return self
    
    def predict(self, X):
        """
        Predict using the fitted RIF regression model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values
        """
        if self.results is None:
            raise ValueError("Model has not been fitted yet")
            
        X = np.asarray(X)
        X = add_constant(X)
        return self.results.predict(X)
    
    def summary(self):
        """
        Get a summary of the regression results.
        
        Returns
        -------
        summary : statsmodels.regression.linear_model.RegressionResultsWrapper
            Summary of the regression results
        """
        if self.results is None:
            raise ValueError("Model has not been fitted yet")
            
        return self.results.summary() 