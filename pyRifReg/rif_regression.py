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
    cov_type : str, default='HC1'
        Type of covariance matrix to use for standard errors. Options are:
        - 'nonrobust': Standard OLS covariance (homoskedastic)
        - 'HC0': White's heteroskedasticity-robust covariance
        - 'HC1': White's heteroskedasticity-robust covariance with small sample correction
        - 'HC2': MacKinnon-White heteroskedasticity-robust covariance
        - 'HC3': Davidson-MacKinnon heteroskedasticity-robust covariance
        - 'bootstrap': Bootstrap-based covariance estimation
    bootstrap_reps : int, default=1000
        Number of bootstrap replications when cov_type='bootstrap'
    """
    
    def __init__(self, statistic='mean', q=None, cov_type='HC1', bootstrap_reps=1000):
        # Validate statistic parameter
        if not isinstance(statistic, str):
            raise TypeError("statistic must be a string")
        
        valid_statistics = ['mean', 'quantile', 'variance', 'gini', 'iqr', 'entropy']
        if statistic not in valid_statistics:
            raise ValueError(f"statistic must be one of {valid_statistics}, got {statistic}")
        
        # Validate q parameter for quantile statistic
        if statistic == 'quantile':
            if q is None:
                raise ValueError("Quantile level 'q' must be specified for quantile RIF")
            if not isinstance(q, (int, float)):
                raise TypeError("q must be a numeric value")
            if not 0 < q < 1:
                raise ValueError("q must be between 0 and 1")
        elif q is not None:
            raise ValueError("q parameter should only be specified for quantile statistic")
        
        # Validate cov_type parameter
        if not isinstance(cov_type, str):
            raise TypeError("cov_type must be a string")
        
        valid_cov_types = ['nonrobust', 'HC0', 'HC1', 'HC2', 'HC3', 'bootstrap']
        if cov_type not in valid_cov_types:
            raise ValueError(f"cov_type must be one of {valid_cov_types}, got {cov_type}")
        
        # Validate bootstrap_reps parameter
        if cov_type == 'bootstrap':
            if not isinstance(bootstrap_reps, int):
                raise TypeError("bootstrap_reps must be an integer")
            if bootstrap_reps <= 0:
                raise ValueError("bootstrap_reps must be positive")
            if bootstrap_reps < 100:
                raise ValueError("bootstrap_reps should be at least 100 for reliable inference")
        elif bootstrap_reps != 1000:
            raise ValueError("bootstrap_reps should only be specified when cov_type='bootstrap'")
        
        self.statistic = statistic
        self.q = q
        self.cov_type = cov_type
        self.bootstrap_reps = bootstrap_reps
        self.model = None
        self.results = None
        self.rif_generator = None
        self.bootstrap_results = None
        
    def _get_rif_generator(self):
        """Get the appropriate RIF generator based on the statistic."""
        try:
            kwargs = {}
            if self.statistic == 'quantile':
                kwargs['q'] = self.q
            return get_rif_generator(self.statistic, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to create RIF generator for statistic '{self.statistic}': {str(e)}")
    
    def _bootstrap_covariance(self, X, y, rif):
        """
        Compute bootstrap-based covariance matrix for coefficient estimates.
        
        Parameters
        ----------
        X : np.ndarray
            Design matrix with constant term
        y : np.ndarray
            Original target values
        rif : np.ndarray
            RIF values
            
        Returns
        -------
        cov_matrix : np.ndarray
            Bootstrap covariance matrix
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Store bootstrap coefficient estimates
        bootstrap_coeffs = np.zeros((self.bootstrap_reps, n_features))
        
        for i in range(self.bootstrap_reps):
            try:
                # Generate bootstrap indices
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                
                # Bootstrap samples
                X_boot = X[bootstrap_indices]
                y_boot = y[bootstrap_indices]
                
                # Recompute RIF for bootstrap sample
                rif_boot = self.rif_generator.compute(y_boot)
                
                # Check for numerical issues in bootstrap RIF
                if np.any(np.isnan(rif_boot)) or np.any(np.isinf(rif_boot)):
                    continue  # Skip this bootstrap replication
                
                # Fit OLS on bootstrap sample
                model_boot = OLS(rif_boot, X_boot)
                results_boot = model_boot.fit()
                
                # Store coefficient estimates
                bootstrap_coeffs[i] = results_boot.params
                
            except Exception:
                # Skip this bootstrap replication if it fails
                continue
        
        # Remove failed bootstrap replications
        successful_reps = np.sum(~np.any(np.isnan(bootstrap_coeffs), axis=1))
        
        if successful_reps < self.bootstrap_reps * 0.5:
            raise RuntimeError(f"Too many bootstrap replications failed. "
                             f"Only {successful_reps}/{self.bootstrap_reps} successful.")
        
        # Use only successful replications
        bootstrap_coeffs = bootstrap_coeffs[~np.any(np.isnan(bootstrap_coeffs), axis=1)]
        
        # Compute covariance matrix
        cov_matrix = np.cov(bootstrap_coeffs.T)
        
        return cov_matrix, successful_reps
    
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
        # Input validation
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        
        # Convert to numpy arrays
        try:
            X = np.asarray(X)
            y = np.asarray(y)
        except Exception as e:
            raise TypeError(f"Failed to convert inputs to numpy arrays: {str(e)}")
        
        # Check dimensions
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError(f"X must be 1D or 2D array, got {X.ndim}D")
        
        if y.ndim != 1:
            raise ValueError(f"y must be 1D array, got {y.ndim}D")
        
        # Check shapes
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have the same number of samples. "
                           f"X has {X.shape[0]} samples, y has {y.shape[0]} samples")
        
        if X.shape[0] == 0:
            raise ValueError("X and y cannot be empty")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or infinite values")
        
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("y contains NaN or infinite values")
        
        # Check for sufficient data
        if X.shape[0] < 2:
            raise ValueError("At least 2 samples are required for regression")
        
        if X.shape[1] == 0:
            raise ValueError("X must have at least one feature")
        
        # Get RIF generator and compute RIF
        try:
            self.rif_generator = self._get_rif_generator()
            rif = self.rif_generator.compute(y)
        except Exception as e:
            raise RuntimeError(f"Failed to compute RIF: {str(e)}")
        
        # Validate RIF output
        if np.any(np.isnan(rif)) or np.any(np.isinf(rif)):
            raise RuntimeError("RIF computation produced NaN or infinite values")
        
        # Add constant term
        try:
            X = add_constant(X)
        except Exception as e:
            raise RuntimeError(f"Failed to add constant term: {str(e)}")
        
        # Fit OLS regression
        try:
            self.model = OLS(rif, X)
            
            if self.cov_type == 'bootstrap':
                # Use bootstrap for covariance estimation
                self.results = self.model.fit()
                bootstrap_cov, successful_reps = self._bootstrap_covariance(X, y, rif)
                
                # Update results with bootstrap covariance
                self.results.cov_params = lambda: bootstrap_cov
                self.bootstrap_results = {
                    'cov_matrix': bootstrap_cov,
                    'successful_reps': successful_reps,
                    'total_reps': self.bootstrap_reps
                }
            else:
                # Use standard covariance estimation
                self.results = self.model.fit(cov_type=self.cov_type)
                
        except Exception as e:
            raise RuntimeError(f"Failed to fit regression model: {str(e)}")
        
        return self
    
    def get_bootstrap_info(self):
        """
        Get information about bootstrap results.
        
        Returns
        -------
        dict or None
            Dictionary containing bootstrap information if bootstrap was used,
            None otherwise
        """
        if self.cov_type != 'bootstrap':
            return None
        return self.bootstrap_results
    
    def summary(self):
        """
        Get a summary of the regression results.
        
        Returns
        -------
        summary : statsmodels.regression.linear_model.RegressionResultsWrapper
            Summary of the regression results
        """
        if self.results is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        summary = self.results.summary()
        
        # Add bootstrap information to summary if bootstrap was used
        if self.cov_type == 'bootstrap' and self.bootstrap_results is not None:
            bootstrap_info = (
                f"\nBootstrap Results:\n"
                f"  Total replications: {self.bootstrap_results['total_reps']}\n"
                f"  Successful replications: {self.bootstrap_results['successful_reps']}\n"
                f"  Success rate: {self.bootstrap_results['successful_reps']/self.bootstrap_results['total_reps']:.1%}"
            )
            # Note: We can't directly modify the summary object, but the information
            # is available through get_bootstrap_info()
        
        return summary 