"""
Implementation of Recentered Influence Function (RIF) regression.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from .rif_generators import get_rif_generator


class RIFRegression(BaseEstimator, RegressorMixin):
    """
    Recentered Influence Function (RIF) Regression.
    
    This class implements RIF regression for various distributional statistics
    such as mean, quantiles, variance, Gini coefficient, and IQR.
    
    Parameters
    ----------
    statistic : str, default='mean'
        The distributional statistic to compute. Options are:
        - 'mean': Mean
        - 'quantile': Quantile (requires q parameter)
        - 'variance': Variance
        - 'gini': Gini coefficient
        - 'iqr': Interquartile Range
        - 'entropy': Entropy
    q : float, optional
        Quantile level (between 0 and 1) when statistic='quantile'
    """
    
    def __init__(self, statistic='mean', q=None):
        self.statistic = statistic
        self.q = q
        self.model = None
        self.results = None
        self.rif_generator = None
        
    def _get_rif_generator(self):
        """Get the appropriate RIF generator based on the statistic."""
        kwargs = {}
        if self.statistic == 'quantile':
            if self.q is None:
                raise ValueError("Quantile level 'q' must be specified for quantile RIF")
            kwargs['q'] = self.q
        return get_rif_generator(self.statistic, **kwargs)
    
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
        
        # Get RIF generator and compute RIF
        self.rif_generator = self._get_rif_generator()
        rif = self.rif_generator.compute(y)
        
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