"""
Implementation of Recentered Influence Function (RIF) regression.
"""

import numpy as np
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
        elif bootstrap_reps != 1000:
            raise ValueError("bootstrap_reps should only be specified when cov_type='bootstrap'")
        
        self.statistic = statistic
        self.q = q
        self.cov_type = cov_type
        self.bootstrap_reps = bootstrap_reps
        self.model = None
        self.results = None
        self.rif_generator = None
        
    def _get_rif_generator(self):
        """Get the appropriate RIF generator based on the statistic."""
        kwargs = {}
        if self.statistic == 'quantile':
            kwargs['q'] = self.q
        return get_rif_generator(self.statistic, **kwargs)
    
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
        
        # Store bootstrap coefficient estimates
        bootstrap_coeffs = []
        
        for i in range(self.bootstrap_reps):
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
            bootstrap_coeffs.append(results_boot.params)

        # Convert to numpy array and compute covariance matrix
        bootstrap_coeffs = np.array(bootstrap_coeffs)
        cov_matrix = np.cov(bootstrap_coeffs.T)
        
        return cov_matrix
    
    def fit(self, X, y):
        """
        Fit the RIF regression model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or pandas DataFrame
            Training data
        y : array-like of shape (n_samples,) or pandas Series
            Target values
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation for X and y
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        
        # Convert pandas inputs to numpy arrays if needed
        if hasattr(X, 'to_numpy'):
            X = X.to_numpy()
        if hasattr(y, 'to_numpy'):
            y = y.to_numpy()
        
        # Convert to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Check X dimensions and reshape if needed
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError(f"X must be 1D or 2D array, got {X.ndim}D")
        
        # Check shapes compatibility
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have the same number of observations. "
                           f"X has {X.shape[0]} observations, y has {y.shape[0]} observations")
        
        if X.shape[0] == 0:
            raise ValueError("X cannot be empty")
        
        if X.shape[1] == 0:
            raise ValueError("X must have at least one feature")
        
        # Check for NaN or infinite values in X
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or infinite values")
        
        # Check for sufficient data
        if X.shape[0] < 2:
            raise ValueError("At least 2 observations are required for regression")
        
        # Get RIF generator and compute RIF (y validation happens in RIF generators)
        self.rif_generator = self._get_rif_generator()
        rif = self.rif_generator.compute(y)
        
        # Validate RIF output
        if np.any(np.isnan(rif)) or np.any(np.isinf(rif)):
            raise RuntimeError("RIF computation produced NaN or infinite values")
        
        # Add constant term
        X = add_constant(X)
        
        # Fit OLS regression
        self.model = OLS(rif, X)
        
        if self.cov_type == 'bootstrap':
            # Use bootstrap for covariance estimation
            self.results = self.model.fit()
            bootstrap_cov = self._bootstrap_covariance(X, y, rif)
            
            # Update results with bootstrap covariance
            self.results.cov_params = lambda: bootstrap_cov
        else:
            # Use standard covariance estimation
            self.results = self.model.fit(cov_type=self.cov_type)
        
        return self
    
    def predict(self, X):
        """
        Predict using the fitted RIF regression model.
        
        For RIF regression, this returns the fitted values (X * beta) from the
        regression of RIF values on the covariates. This is not a traditional
        prediction in the machine learning sense, but rather the fitted values
        from the RIF regression.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or pandas DataFrame
            Samples to predict for
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values (fitted values from RIF regression)
        """
        if self.results is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Input validation
        if X is None:
            raise ValueError("X cannot be None")
        
        # Convert pandas inputs to numpy arrays if needed
        if hasattr(X, 'to_numpy'):
            X = X.to_numpy()
        
        # Convert to numpy array
        X = np.asarray(X)
        
        # Check dimensions and reshape if needed
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError(f"X must be 1D or 2D array, got {X.ndim}D")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or infinite values")
        
        # Add constant term to match the fitted model
        X = add_constant(X)
        
        # Check that number of features matches the fitted model
        if X.shape[1] != len(self.results.params):
            raise ValueError(f"X has {X.shape[1]} features (including constant), "
                           f"but model was fitted with {len(self.results.params)} parameters")
        
        # Return fitted values
        return self.results.fittedvalues[:len(X)]
    
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
        
        return self.results.summary() 