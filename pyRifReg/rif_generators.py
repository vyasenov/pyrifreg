"""
Collection of Recentered Influence Function (RIF) generators for various distributional statistics.
"""

import numpy as np
from scipy import stats

class RIFGenerator:
    """Base class for RIF generators."""
    
    def __init__(self):
        self._kde = None
    
    def _validate_data(self, y: np.ndarray, min_points: int = 1) -> np.ndarray:
        """
        Validate input data and convert to numpy array.
        
        Parameters
        ----------
        y : array-like
            Input data to validate
        min_points : int, default=1
            Minimum number of data points required
            
        Returns
        -------
        y : np.ndarray
            Validated and converted input data
        """
        if y is None:
            raise ValueError("Input data cannot be None")
        
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        
        if y.ndim != 1:
            raise ValueError(f"Input data must be 1D array, got {y.ndim}D")
        
        if len(y) == 0:
            raise ValueError("Input data cannot be empty")
        
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("Input data contains NaN or infinite values")
        
        if len(y) < min_points:
            raise ValueError(f"At least {min_points} data points are required")
        
        return y
    
    def _estimate_density(self, y: np.ndarray) -> None:
        """Estimate the density function using kernel density estimation."""
        y = self._validate_data(y, min_points=2)
        self._kde = stats.gaussian_kde(y)
    
    def compute(self, y: np.ndarray) -> np.ndarray:
        """
        Compute the RIF for the given statistic.
        
        Parameters
        ----------
        y : array-like
            Input data
            
        Returns
        -------
        rif : array-like
            Recentered Influence Function values
        """
        raise NotImplementedError("Subclasses must implement compute method")


class MeanRIF(RIFGenerator):
    """RIF generator for the mean."""
    
    def compute(self, y: np.ndarray) -> np.ndarray:
        y = self._validate_data(y, min_points=1)
        return y


class QuantileRIF(RIFGenerator):
    """RIF generator for quantiles."""
    
    def __init__(self, q: float):
        """
        Parameters
        ----------
        q : float
            Quantile level (between 0 and 1)
        """
        super().__init__()
        
        if not isinstance(q, (int, float)):
            raise TypeError("Quantile level must be a numeric value")
        
        if not 0 < q < 1:
            raise ValueError("Quantile level must be between 0 and 1")
        
        self.q = q
    
    def compute(self, y: np.ndarray) -> np.ndarray:
        y = self._validate_data(y, min_points=2)
        
        self._estimate_density(y)
        q_val = np.quantile(y, self.q)
        f_q = self._kde(np.array([q_val]))[0]
        
        if f_q <= 0:
            raise RuntimeError(f"Density estimate at quantile {self.q} is non-positive")
        
        return q_val + (self.q - (y <= q_val).astype(float)) / f_q


class VarianceRIF(RIFGenerator):
    """RIF generator for variance."""
    
    def compute(self, y: np.ndarray) -> np.ndarray:
        y = self._validate_data(y, min_points=2)
        mean_y = np.mean(y)
        return (y - mean_y)**2


class GiniRIF(RIFGenerator):
    """RIF generator for Gini coefficient."""
    
    def compute(self, y: np.ndarray) -> np.ndarray:
        y = self._validate_data(y, min_points=2)
        
        if np.any(y < 0):
            raise ValueError("All values must be non-negative for Gini coefficient")
        
        n = len(y)
        mean_y = np.mean(y)
        
        # Compute Gini coefficient
        gini = 0
        for i in range(n):
            for j in range(n):
                gini += np.abs(y[i] - y[j])
        gini = gini / (2 * n * mean_y)
        
        # Compute RIF for Gini
        rif = np.zeros_like(y)
        for i in range(n):
            rank_i = np.sum(y <= y[i])  # Rank of observation i
            rif[i] = gini + (2 * rank_i - n - 1) * y[i] / (n * mean_y) - gini
        
        return rif


class IQRRIF(RIFGenerator):
    """RIF generator for Interquartile Range (IQR)."""
    
    def compute(self, y: np.ndarray) -> np.ndarray:
        y = self._validate_data(y, min_points=4)
        
        self._estimate_density(y)
        q75 = np.quantile(y, 0.75)
        q25 = np.quantile(y, 0.25)
        f_q75 = self._kde(q75)
        f_q25 = self._kde(q25)
        
        if f_q75 <= 0 or f_q25 <= 0:
            raise RuntimeError("Density estimates at quantiles are non-positive")
        
        # RIF for IQR is the difference of RIFs for 75th and 25th quantiles
        rif_q75 = q75 + (0.75 - (y <= q75)) / f_q75
        rif_q25 = q25 + (0.25 - (y <= q25)) / f_q25
        
        return rif_q75 - rif_q25


class EntropyRIF(RIFGenerator):
    """RIF generator for entropy."""
    
    def compute(self, y: np.ndarray) -> np.ndarray:
        y = self._validate_data(y, min_points=2)
        
        if np.any(y <= 0):
            raise ValueError("All values must be positive for entropy estimation")
        
        self._estimate_density(y)
        n = len(y)
        h = 1.06 * np.std(y) * n**(-1/5)  # Silverman's rule of thumb
        
        if h <= 0:
            raise RuntimeError("Bandwidth for density estimation is non-positive")
        
        kde_vals = self._kde(y)
        if np.any(kde_vals <= 0):
            raise RuntimeError("Density estimates contain non-positive values")
        
        entropy = -np.sum(kde_vals * np.log(kde_vals + 1e-10)) * h
        
        # Compute RIF for entropy
        rif = np.zeros_like(y)
        for i in range(n):
            y_temp = y.copy()
            y_temp[i] = y[i] + 1e-6
            kde_temp = stats.gaussian_kde(y_temp)
            kde_temp_vals = kde_temp(y_temp)
            
            if np.any(kde_temp_vals <= 0):
                continue  # Skip this iteration if density is non-positive
            
            entropy_temp = -np.sum(kde_temp_vals * np.log(kde_temp_vals + 1e-10)) * h
            rif[i] = (entropy_temp - entropy) / 1e-6
        
        return rif


def get_rif_generator(statistic: str, **kwargs) -> RIFGenerator:
    """
    Factory function to get the appropriate RIF generator.
    
    Parameters
    ----------
    statistic : str
        Name of the statistic. Options are:
        - 'mean': Mean
        - 'quantile': Quantile (requires q parameter)
        - 'variance': Variance
        - 'gini': Gini coefficient
        - 'iqr': Interquartile Range
        - 'entropy': Entropy
    **kwargs : dict
        Additional parameters for specific generators (e.g., q for quantile)
        
    Returns
    -------
    generator : RIFGenerator
        The appropriate RIF generator instance
    """
    if not isinstance(statistic, str):
        raise TypeError("statistic must be a string")
    
    generators = {
        'mean': MeanRIF,
        'quantile': QuantileRIF,
        'variance': VarianceRIF,
        'gini': GiniRIF,
        'iqr': IQRRIF,
        'entropy': EntropyRIF
    }
    
    if statistic not in generators:
        raise ValueError(f"Unknown statistic: {statistic}. "
                        f"Available options are: {list(generators.keys())}")
    
    return generators[statistic](**kwargs) 